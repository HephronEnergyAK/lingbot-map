"""Strict, capacity-gated, transactional import of Reconstruction Results.

The validation half of this module deliberately has no dependency on ``bpy``.
Blender datablocks are not touched until the complete immutable bundle has
passed validation and the Windows physical-memory gate.
"""

from __future__ import annotations

import ast
import ctypes
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Callable, Mapping
import uuid

from .ipc import IpcError, read_json, require_exact_object, require_text
from .job_lifecycle import is_reparse_point
from .results import (
    CORE_ARRAYS,
    JOB_ID,
    OPTIONAL_ARRAYS,
    RESULT_ID,
    RESULT_SCHEMA_VERSION,
    ReadyResult,
    read_ready_result,
)


OWNERSHIP_SCHEMA = "1.0.0"
POINT_SHADER_SCHEMA = "point-shader-1.0.0"
POINT_MATERIAL_SCHEMA = "point-material-1.0.0"
POINT_DISPLAY_SCHEMA = "point-display-1.0.0"
CAPACITY_MODEL_VERSION = "blender-5.2-pointcloud-windows-x64-v1"

# Pinned by tests/blender_import_benchmark.py on Blender 5.2 Windows x64.
# These intentionally round the measured upper envelope upward.
IMPORT_FIXED_BYTES = 256 * 1024**2
IMPORT_BYTES_PER_POINT = 160
IMPORT_HEADROOM_NUMERATOR = 5
IMPORT_HEADROOM_DENOMINATOR = 4
IMPORT_SYSTEM_RESERVE_BYTES = 4 * 1024**3
MAX_NPY_HEADER_BYTES = 64 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
SEMANTIC_CHUNK_POINTS = 1_000_000
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
np: Any = None

ARRAY_DTYPES = {
    "positions": "<f4",
    "colors": "|u1",
    "confidence": "<f4",
    "radius": "<f4",
    "source_frame": "<u4",
    "camera_to_world": "<f4",
    "model_intrinsics": "<f4",
    "source_intrinsics": "<f4",
    "model_fov_radians": "<f4",
    "source_pts_seconds": "<f8",
    "source_to_model": "<f8",
    "frame_type": "|u1",
    "sky_fraction": "<f4",
}

IMPORT_PHASES = (
    "validation",
    "capacity",
    "staging_collection",
    "root_object",
    "pointcloud",
    "positions",
    "color",
    "confidence",
    "radius",
    "source_frame",
    "material",
    "geometry_nodes",
    "ownership",
    "commit",
)

CancelCheck = Callable[[str], bool]
AvailableMemoryProbe = Callable[[], int]


class ResultImportError(RuntimeError):
    """A Result cannot be imported without weakening a contract."""


class ResultImportCancelled(ResultImportError):
    """The transaction was cancelled before its commit point."""


class ImportCapacityError(ResultImportError):
    """Current physical memory cannot safely contain this complete import."""


@dataclass(frozen=True)
class ImportCapacity:
    model_version: str
    point_count: int
    estimated_peak_bytes: int
    headroom_bytes: int
    reserve_bytes: int
    required_available_bytes: int
    available_physical_bytes: int

    @property
    def allowed(self) -> bool:
        return self.available_physical_bytes >= self.required_available_bytes


@dataclass(frozen=True)
class ValidatedResult:
    ready: ReadyResult
    manifest: Mapping[str, Any]
    manifest_sha256: str
    arrays: Mapping[str, np.ndarray]
    confidence_p5: float
    confidence_p95: float


@dataclass(frozen=True)
class ImportOutcome:
    result_id: str
    collection: Any
    created: bool
    capacity: ImportCapacity
    message: str


@dataclass
class _Created:
    collections: list[Any]
    objects: list[Any]
    pointclouds: list[Any]
    materials: list[Any]
    node_groups: list[Any]

    @classmethod
    def empty(cls) -> "_Created":
        return cls([], [], [], [], [])


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def _require_numpy() -> Any:
    """Load Blender's bundled NumPy only when a Result is explicitly selected."""

    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    return np


def _schema_type_matches(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    return False


def _validate_json_value(
    value: Any,
    schema: Mapping[str, Any] | bool,
    root_schema: Mapping[str, Any],
    path: str,
) -> None:
    """Validate the bounded JSON-Schema subset used by the bundled contract."""

    if schema is True:
        return
    if schema is False:
        raise ResultImportError(f"{path} is forbidden by the Result schema")
    if "$ref" in schema:
        reference = schema["$ref"]
        if (
            not isinstance(reference, str)
            or not reference.startswith("#/$defs/")
            or "/" in reference[len("#/$defs/") :]
        ):
            raise ResultImportError("Bundled Result schema contains an unsafe reference")
        target = root_schema.get("$defs", {}).get(reference[len("#/$defs/") :])
        if not isinstance(target, dict):
            raise ResultImportError("Bundled Result schema reference is missing")
        _validate_json_value(value, target, root_schema, path)
        return
    expected = schema.get("type")
    if expected is not None:
        choices = [expected] if isinstance(expected, str) else expected
        if not isinstance(choices, list) or not any(
            isinstance(name, str) and _schema_type_matches(value, name)
            for name in choices
        ):
            raise ResultImportError(f"{path} has the wrong JSON type")
    if "const" in schema and value != schema["const"]:
        raise ResultImportError(f"{path} does not match the fixed Result contract")
    if "enum" in schema and value not in schema["enum"]:
        raise ResultImportError(f"{path} is outside the Result allowlist")
    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(name, str) for name in required
        ):
            raise ResultImportError("Bundled Result schema has invalid required fields")
        missing = set(required) - set(value)
        if missing:
            raise ResultImportError(f"{path} is missing fields: {sorted(missing)}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ResultImportError("Bundled Result schema properties are invalid")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ResultImportError(f"{path} has unknown fields: {sorted(unknown)}")
        for name, item in value.items():
            child = properties.get(name)
            if child is not None:
                _validate_json_value(
                    item, child, root_schema, f"{path}.{name}"
                )
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ResultImportError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ResultImportError(f"{path} has too many items")
        if schema.get("uniqueItems"):
            encoded = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(encoded) != len(set(encoded)):
                raise ResultImportError(f"{path} contains duplicate items")
        prefix = schema.get("prefixItems", [])
        if not isinstance(prefix, list):
            raise ResultImportError("Bundled Result schema prefixItems are invalid")
        for index, child in enumerate(prefix[: len(value)]):
            _validate_json_value(
                value[index], child, root_schema, f"{path}[{index}]"
            )
        items = schema.get("items")
        if items is not None:
            for index in range(len(prefix), len(value)):
                _validate_json_value(
                    value[index], items, root_schema, f"{path}[{index}]"
                )
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ResultImportError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ResultImportError(f"{path} is too long")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ResultImportError(f"{path} does not match the Result pattern")
        if schema.get("format") == "uuid":
            try:
                uuid.UUID(value)
            except ValueError as exc:
                raise ResultImportError(f"{path} is not a UUID") from exc
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ResultImportError(f"{path} is not an ISO-8601 date-time") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ResultImportError(f"{path} lacks a time-zone offset")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        if "minimum" in schema and value < schema["minimum"]:
            raise ResultImportError(f"{path} is below the Result minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ResultImportError(f"{path} is above the Result maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ResultImportError(f"{path} is not above the Result minimum")


def _validate_bundled_schema(manifest: Mapping[str, Any]) -> None:
    path = (
        Path(__file__).resolve().parent
        / "runtime_bundle"
        / "schemas"
        / "reconstruction-result.schema.json"
    )
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultImportError("Bundled Reconstruction Result schema is unavailable") from exc
    if (
        not isinstance(schema, dict)
        or schema.get("$id")
        != "https://lingbot-map.invalid/schemas/reconstruction-result/1.0.0"
    ):
        raise ResultImportError("Bundled Reconstruction Result schema identity is invalid")
    _validate_json_value(manifest, schema, schema, "$")


def available_physical_memory() -> int:
    """Return Windows physical availability, never pagefile capacity."""

    if os.name != "nt":
        raise ImportCapacityError("Import Capacity Gate requires supported Windows")
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ImportCapacityError("Windows physical-memory query failed")
    return int(status.ullAvailPhys)


def evaluate_import_capacity(
    point_count: int, *, available_probe: AvailableMemoryProbe = available_physical_memory
) -> ImportCapacity:
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 0:
        raise ImportCapacityError("Result point count is invalid")
    estimate = IMPORT_FIXED_BYTES + point_count * IMPORT_BYTES_PER_POINT
    headroom = math.ceil(
        estimate * IMPORT_HEADROOM_NUMERATOR / IMPORT_HEADROOM_DENOMINATOR
    )
    required = headroom + IMPORT_SYSTEM_RESERVE_BYTES
    available = int(available_probe())
    if available < 0:
        raise ImportCapacityError("Available physical memory is invalid")
    result = ImportCapacity(
        CAPACITY_MODEL_VERSION,
        point_count,
        estimate,
        headroom,
        IMPORT_SYSTEM_RESERVE_BYTES,
        required,
        available,
    )
    if not result.allowed:
        raise ImportCapacityError(
            "Import Capacity Gate blocked this Result: "
            f"{available / 1024**3:.2f} GiB available, "
            f"{required / 1024**3:.2f} GiB required "
            f"({CAPACITY_MODEL_VERSION}; no override or partial fallback)"
        )
    return result


def _cancelled(cancel: CancelCheck | None, phase: str) -> None:
    if cancel is not None and bool(cancel(phase)):
        raise ResultImportCancelled(f"Result import cancelled during {phase}")


def _plain_relative_file(root: Path, raw: Any) -> Path:
    text = require_text(raw, label="Result file path", maximum=32767)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or text.startswith(("/", "//"))
        or re.match(r"^[A-Za-z]:", text)
    ):
        raise ResultImportError("Result file path is unsafe")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ResultImportError("Result file path is traversing")
    root = Path(os.path.abspath(root))
    path = root.joinpath(*parts)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ResultImportError("Result file path escaped its publication") from exc
    if not path.is_file() or path.is_symlink() or is_reparse_point(path):
        raise ResultImportError("Result file is absent or not ordinary")
    current = path.parent
    while current != root:
        if current.is_symlink() or is_reparse_point(current):
            raise ResultImportError("Result file resolves through linked content")
        current = current.parent
    return path


def _stable_sha256(
    path: Path, *, cancel: CancelCheck | None = None, phase: str = "validation"
) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    completed = 0
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            _cancelled(cancel, phase)
            digest.update(chunk)
            completed += len(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or completed != before.st_size
    ):
        raise ResultImportError(f"Result file changed during validation: {path.name}")
    return digest.hexdigest(), completed


def _npy_header(path: Path) -> tuple[str, bool, tuple[int, ...], int]:
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ResultImportError(f"NPY magic is invalid: {path.name}")
        version = stream.read(2)
        if version == b"\x01\x00":
            raw_length = stream.read(2)
            if len(raw_length) != 2:
                raise ResultImportError("NPY header length is truncated")
            header_length = struct.unpack("<H", raw_length)[0]
        elif version in {b"\x02\x00", b"\x03\x00"}:
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                raise ResultImportError("NPY header length is truncated")
            header_length = struct.unpack("<I", raw_length)[0]
        else:
            raise ResultImportError("NPY version is unsupported")
        if not 1 <= header_length <= MAX_NPY_HEADER_BYTES:
            raise ResultImportError("NPY header exceeds 64 KiB")
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ResultImportError("NPY header is truncated")
        try:
            document = ast.literal_eval(header.decode("latin1").strip())
        except (UnicodeError, SyntaxError, ValueError, MemoryError) as exc:
            raise ResultImportError("NPY header dictionary is invalid") from exc
        if not isinstance(document, dict) or set(document) != {
            "descr",
            "fortran_order",
            "shape",
        }:
            raise ResultImportError("NPY header has unknown or missing fields")
        dtype = document["descr"]
        fortran = document["fortran_order"]
        shape = document["shape"]
        if (
            not isinstance(dtype, str)
            or not isinstance(fortran, bool)
            or not isinstance(shape, tuple)
            or len(shape) > 4
            or not all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in shape
            )
        ):
            raise ResultImportError("NPY dtype, shape, or order is invalid")
        offset = stream.tell()
    return dtype, fortran, tuple(shape), offset


def _array_descriptor(
    root: Path,
    name: str,
    raw: Any,
    *,
    cancel: CancelCheck | None,
) -> np.ndarray:
    descriptor = require_exact_object(
        raw,
        {"path", "dtype", "shape", "byte_length", "sha256"},
        label=f"arrays.{name}",
    )
    if descriptor["path"] != f"arrays/{name}.npy":
        raise ResultImportError(f"Array path is not canonical: {name}")
    if descriptor["dtype"] != ARRAY_DTYPES[name]:
        raise ResultImportError(f"Array descriptor dtype is invalid: {name}")
    shape = descriptor["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) > 4
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in shape
        )
    ):
        raise ResultImportError(f"Array descriptor shape is invalid: {name}")
    if (
        isinstance(descriptor["byte_length"], bool)
        or not isinstance(descriptor["byte_length"], int)
        or descriptor["byte_length"] < 1
        or not SHA256.fullmatch(str(descriptor["sha256"]))
    ):
        raise ResultImportError(f"Array descriptor checksum or length is invalid: {name}")
    path = _plain_relative_file(root, descriptor["path"])
    dtype, fortran, header_shape, offset = _npy_header(path)
    expected_shape = tuple(shape)
    if dtype != ARRAY_DTYPES[name] or fortran or header_shape != expected_shape:
        raise ResultImportError(f"NPY contract mismatch: {name}")
    expected_size = offset + math.prod(expected_shape) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected_size:
        raise ResultImportError(f"NPY payload length is inconsistent: {name}")
    digest, byte_length = _stable_sha256(
        path, cancel=cancel, phase=f"validation:{name}"
    )
    if byte_length != descriptor["byte_length"] or digest != descriptor["sha256"]:
        raise ResultImportError(f"Array checksum or length mismatch: {name}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, MemoryError) as exc:
        raise ResultImportError(f"NPY cannot be opened safely: {name}") from exc
    if (
        array.dtype.str != dtype
        or tuple(array.shape) != expected_shape
        or not array.flags.c_contiguous
        or array.dtype.hasobject
        or array.dtype.fields is not None
    ):
        raise ResultImportError(f"Loaded NPY representation is inconsistent: {name}")
    return array


def _finite_chunks(
    name: str, array: np.ndarray, *, positive: bool = False
) -> None:
    count = array.shape[0] if array.ndim else 1
    for start in range(0, count, SEMANTIC_CHUNK_POINTS):
        chunk = array[start : start + SEMANTIC_CHUNK_POINTS]
        if not bool(np.isfinite(chunk).all()):
            raise ResultImportError(f"{name} contains non-finite values")
        if positive and not bool((chunk > 0).all()):
            raise ResultImportError(f"{name} contains non-positive values")


def _validate_array_semantics(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    positions = arrays["positions"]
    point_count = positions.shape[0] if positions.ndim == 2 else -1
    if positions.shape != (point_count, 3) or arrays["colors"].shape != (
        point_count,
        3,
    ):
        raise ResultImportError("Point position or color shape is invalid")
    for name in ("confidence", "radius", "source_frame"):
        if arrays[name].shape != (point_count,):
            raise ResultImportError(f"{name} must have shape (N,)")
    cameras = arrays["camera_to_world"]
    frame_count = cameras.shape[0] if cameras.ndim == 3 else -1
    expected = {
        "camera_to_world": (frame_count, 4, 4),
        "model_intrinsics": (frame_count, 3, 3),
        "source_intrinsics": (frame_count, 3, 3),
        "model_fov_radians": (frame_count, 2),
        "source_pts_seconds": (frame_count,),
        "frame_type": (frame_count,),
        "source_to_model": (3, 3),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ResultImportError(f"{name} must have shape {shape}")
    if frame_count < 1:
        raise ResultImportError("Result must contain at least one frame")
    if "sky_fraction" in arrays and arrays["sky_fraction"].shape != (frame_count,):
        raise ResultImportError("sky_fraction must have shape (F,)")
    for name in (
        "positions",
        "confidence",
        "camera_to_world",
        "model_intrinsics",
        "source_intrinsics",
        "model_fov_radians",
        "source_pts_seconds",
        "source_to_model",
    ):
        _finite_chunks(name, arrays[name])
    _finite_chunks("radius", arrays["radius"], positive=True)
    if "sky_fraction" in arrays:
        _finite_chunks("sky_fraction", arrays["sky_fraction"])
        if not bool(
            ((arrays["sky_fraction"] >= 0) & (arrays["sky_fraction"] <= 1)).all()
        ):
            raise ResultImportError("sky_fraction lies outside [0,1]")
    if point_count and (
        not bool((arrays["source_frame"] < frame_count).all())
        or not bool((arrays["source_frame"] <= np.iinfo(np.int32).max).all())
    ):
        raise ResultImportError("source_frame refers outside supported camera frames")
    if not bool(np.isin(arrays["frame_type"], (0, 1, 2)).all()):
        raise ResultImportError("frame_type contains an unknown code")
    timestamps = arrays["source_pts_seconds"]
    if len(timestamps) > 1 and not bool((np.diff(timestamps) > 0).all()):
        raise ResultImportError("Source timestamps are not strictly increasing")
    fov = arrays["model_fov_radians"]
    if not bool(((fov > 0) & (fov < math.pi)).all()):
        raise ResultImportError("Model FOV lies outside (0,pi)")
    for name in ("model_intrinsics", "source_intrinsics"):
        intrinsics = arrays[name].astype(np.float64, copy=False)
        if not bool(
            (intrinsics[:, 0, 0] > 0).all()
            and (intrinsics[:, 1, 1] > 0).all()
        ):
            raise ResultImportError(f"{name} has a non-positive focal length")
        if (
            not np.allclose(
                intrinsics[:, 2, :], (0.0, 0.0, 1.0), atol=1e-6, rtol=0
            )
            or not np.allclose(intrinsics[:, 0, 1], 0, atol=1e-6, rtol=0)
            or not np.allclose(intrinsics[:, 1, 0], 0, atol=1e-6, rtol=0)
        ):
            raise ResultImportError(f"{name} has a malformed homogeneous row or skew")
    if not np.allclose(
        cameras[:, 3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-5, rtol=0
    ):
        raise ResultImportError("camera_to_world has malformed homogeneous rows")
    rotations = cameras[:, :3, :3].astype(np.float64, copy=False)
    if not np.allclose(
        np.einsum("fji,fjk->fik", rotations, rotations),
        np.eye(3),
        atol=1e-4,
        rtol=0,
    ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4, rtol=0):
        raise ResultImportError("camera_to_world rotation is not rigid right-handed")
    transform = arrays["source_to_model"]
    if abs(float(np.linalg.det(transform))) <= 1e-12 or not np.allclose(
        transform[2], (0.0, 0.0, 1.0), atol=1e-9, rtol=0
    ):
        raise ResultImportError("source_to_model is not a nonsingular affine transform")
    mapped = np.einsum(
        "ij,fjk->fik",
        transform.astype(np.float64, copy=False),
        arrays["source_intrinsics"].astype(np.float64, copy=False),
    )
    if not np.allclose(
        mapped,
        arrays["model_intrinsics"].astype(np.float64, copy=False),
        atol=1e-4,
        rtol=1e-5,
    ):
        raise ResultImportError(
            "Source and model intrinsics disagree with source_to_model"
        )
    first_camera = np.array(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    if not np.allclose(cameras[0], first_camera, atol=1e-4, rtol=0):
        raise ResultImportError(
            "First camera does not define the Reconstruction Frame"
        )
    return point_count, frame_count


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    point_count: int,
    frame_count: int,
) -> tuple[float, float]:
    target = require_exact_object(
        manifest["target_scene"],
        {"blend_path", "scene_uuid", "scene_name"},
        label="target_scene",
    )
    for name, value in target.items():
        require_text(value, label=f"target_scene.{name}", maximum=32767)
    if not Path(target["blend_path"]).is_absolute():
        raise ResultImportError("Target Scene blend path is not absolute")
    if (
        isinstance(manifest["timeline_start"], bool)
        or not isinstance(manifest["timeline_start"], int)
    ):
        raise ResultImportError("timeline_start is invalid")
    source = require_exact_object(
        manifest["source"],
        {
            "absolute_path",
            "scene_relative_path",
            "size_bytes",
            "modification_time_ns",
            "sha256",
        },
        label="source",
    )
    if not Path(str(source["absolute_path"])).is_absolute():
        raise ResultImportError("Source absolute path is invalid")
    if source["scene_relative_path"] is not None and (
        not isinstance(source["scene_relative_path"], str)
        or not source["scene_relative_path"].startswith("//")
    ):
        raise ResultImportError("Source scene-relative path is invalid")
    if any(
        isinstance(source[name], bool)
        or not isinstance(source[name], int)
        or source[name] < 1
        for name in ("size_bytes", "modification_time_ns")
    ) or not SHA256.fullmatch(str(source["sha256"])):
        raise ResultImportError("Source identity is invalid")
    if require_exact_object(
        manifest["contracts"],
        {"job_spec", "events", "result"},
        label="contracts",
    ) != {"job_spec": "1.0.0", "events": "1.0.0", "result": "1.0.0"}:
        raise ResultImportError("Manifest contract versions are unsupported")
    profile = manifest["profile"]
    profile_fields = {
        "name",
        "confidence_cutoff_percent",
        "depth_cutoff_percent",
        "import_point_budget",
    }
    if (
        not isinstance(profile, dict)
        or not profile_fields.issubset(profile)
        or not set(profile).issubset(profile_fields | {"retain_dense_predictions"})
    ):
        raise ResultImportError("Profile has unknown or missing fields")
    if (
        isinstance(profile["import_point_budget"], bool)
        or not isinstance(profile["import_point_budget"], int)
        or not 1 <= profile["import_point_budget"] <= 50_000_000
        or point_count > profile["import_point_budget"]
    ):
        raise ResultImportError("Point count exceeds the validated Import Point Budget")
    require_text(profile["name"], label="profile.name", maximum=128)
    for name in ("confidence_cutoff_percent", "depth_cutoff_percent"):
        value = profile[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= value <= 100
        ):
            raise ResultImportError(f"Profile {name} is invalid")
    coordinate = require_exact_object(
        manifest["coordinate_system"],
        {
            "name",
            "handedness",
            "camera_local_axes",
            "voxel_origin",
            "voxel_edge_length",
        },
        label="coordinate_system",
    )
    edge = coordinate["voxel_edge_length"]
    if (
        coordinate["name"] != "blender-z-up-reconstruction-frame"
        or coordinate["handedness"] != "right"
        or coordinate["camera_local_axes"] != "+X right,+Y up,-Z forward"
        or coordinate["voxel_origin"] != [0.0, 0.0, 0.0]
        or isinstance(edge, bool)
        or not isinstance(edge, (int, float))
        or not math.isfinite(float(edge))
        or edge <= 0
    ):
        raise ResultImportError("Coordinate-system contract is invalid")
    counts = require_exact_object(
        manifest["counts"], {"frames", "points"}, label="counts"
    )
    if counts != {"frames": frame_count, "points": point_count}:
        raise ResultImportError("Manifest counts disagree with core arrays")
    if point_count and not np.allclose(
        arrays["radius"], float(edge) / 2.0, atol=1e-6, rtol=1e-6
    ):
        raise ResultImportError("Point radius disagrees with the voxel edge")
    statistics = require_exact_object(
        manifest["confidence_statistics"],
        {"p0", "p5", "p25", "p50", "p75", "p95", "p100"},
        label="confidence_statistics",
    )
    if point_count:
        names = ("p0", "p5", "p25", "p50", "p75", "p95", "p100")
        try:
            recorded = np.asarray([statistics[name] for name in names], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ResultImportError("Confidence statistics are invalid") from exc
        actual = np.percentile(arrays["confidence"], (0, 5, 25, 50, 75, 95, 100))
        if not np.allclose(actual, recorded, atol=1e-6, rtol=1e-6):
            raise ResultImportError("Confidence statistics disagree with retained values")
        p5, p95 = float(recorded[1]), float(recorded[5])
    else:
        if any(value is not None for value in statistics.values()):
            raise ResultImportError("Empty point cloud requires null confidence statistics")
        p5 = p95 = 0.0
    if "sky_fraction" in arrays:
        sky = require_exact_object(
            manifest.get("sky_statistics"),
            {"minimum", "median", "p95", "maximum", "count_above_95_percent"},
            label="sky_statistics",
        )
        actual = np.percentile(arrays["sky_fraction"], (0, 50, 95, 100))
        recorded = np.asarray(
            [sky["minimum"], sky["median"], sky["p95"], sky["maximum"]],
            dtype=np.float64,
        )
        if (
            not np.allclose(actual, recorded, atol=1e-6, rtol=1e-6)
            or isinstance(sky["count_above_95_percent"], bool)
            or sky["count_above_95_percent"]
            != int(np.count_nonzero(arrays["sky_fraction"] > np.float32(0.95)))
        ):
            raise ResultImportError("Sky statistics disagree with sky_fraction")
    elif manifest.get("sky_statistics") is not None:
        raise ResultImportError("Sky statistics require sky_fraction")
    if not isinstance(manifest["warnings"], list):
        raise ResultImportError("Warnings must be an array")
    for warning in manifest["warnings"]:
        record = require_exact_object(warning, {"code", "message"}, label="warning")
        require_text(record["code"], label="warning.code", maximum=128)
        require_text(record["message"], label="warning.message", maximum=16384)
    alignment = manifest.get("window_alignment")
    triggered_count = 0
    if alignment is not None:
        boundaries = alignment["boundaries"]
        expected_boundaries = math.ceil(max(0, frame_count - 64) / 48)
        if len(boundaries) != expected_boundaries:
            raise ResultImportError("Window alignment boundary count is inconsistent")
        previous_end = -1
        for boundary in boundaries:
            start = boundary["source_frame_start"]
            end = boundary["source_frame_end"]
            if (
                end - start != 15
                or start <= previous_end
                or not 0 <= start <= end < frame_count
            ):
                raise ResultImportError("Window alignment boundary identity is invalid")
            previous_end = end
            triggered_count += bool(boundary["triggered_conditions"])
        quality_warnings = sum(
            warning["code"] == "quality_warning"
            for warning in manifest["warnings"]
        )
        if quality_warnings != triggered_count:
            raise ResultImportError(
                "Window alignment Quality Warnings disagree with boundaries"
            )
    if not isinstance(manifest["provenance"], dict):
        raise ResultImportError("Provenance must be an object")
    provenance = manifest["provenance"]
    for name in ("runtime_id", "job_spec_sha256", "model_sha256", "source_sha256"):
        if not SHA256.fullmatch(str(provenance.get(name))):
            raise ResultImportError(f"Provenance identity is invalid: {name}")
    if provenance["source_sha256"] != source["sha256"]:
        raise ResultImportError("Source and provenance checksums disagree")
    models = provenance["models"]
    if not any(
        model["sha256"] == provenance["model_sha256"] for model in models
    ):
        raise ResultImportError("Primary model checksum is absent from provenance")
    retained_dense = profile.get("retain_dense_predictions", False)
    if retained_dense != ("dense_predictions" in manifest):
        raise ResultImportError(
            "Profile and Dense Predictions descriptor disagree"
        )
    profile_provenance = provenance["profile"]
    for name in (
        "name",
        "confidence_cutoff_percent",
        "depth_cutoff_percent",
        "import_point_budget",
    ):
        if profile_provenance[name] != profile[name]:
            raise ResultImportError(
                f"Profile provenance disagrees with Result field: {name}"
            )
    if profile_provenance.get("retain_dense_predictions", False) != retained_dense:
        raise ResultImportError(
            "Profile provenance Dense Predictions setting disagrees"
        )
    if (provenance["inference"]["mode"] == "windowed") != (alignment is not None):
        raise ResultImportError(
            "Inference mode and window-alignment provenance disagree"
        )
    if provenance["inference"]["prediction_heads"] not in (
        ["camera", "depth"],
        ["fixture"],
    ):
        raise ResultImportError("Inference prediction heads are invalid")
    sky_masking = provenance.get("sky_masking")
    if ("sky_fraction" in arrays) != (sky_masking is not None):
        raise ResultImportError(
            "Sky Mask provenance and sky_fraction disagree"
        )
    if sky_masking is not None:
        expected_model = {
            "id": sky_masking["model_id"],
            "role": "auxiliary",
            "sha256": sky_masking["model_sha256"],
        }
        if expected_model not in models:
            raise ResultImportError(
                "Sky Mask Auxiliary Model is absent from provenance"
            )
        if provenance["system"]["onnx_threads"] != sky_masking["onnx_threads"]:
            raise ResultImportError(
                "Sky Mask and system ONNX thread provenance disagree"
            )
    return p5, p95


def validate_result(
    directory: str | Path, *, cancel: CancelCheck | None = None
) -> ValidatedResult:
    """Validate all core files and semantics before Blender allocation."""

    _require_numpy()
    _cancelled(cancel, "validation")
    root = Path(os.path.abspath(directory))
    try:
        ready = read_ready_result(root)
    except (IpcError, OSError, ValueError) as exc:
        raise ResultImportError(f"Result discovery validation failed: {exc}") from exc
    manifest_path = _plain_relative_file(root, "manifest.json")
    manifest_digest, _ = _stable_sha256(
        manifest_path, cancel=cancel, phase="validation:manifest"
    )
    try:
        manifest = read_json(manifest_path)
    except (IpcError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultImportError("Result manifest cannot be read safely") from exc
    if not isinstance(manifest, dict):
        raise ResultImportError("Result manifest must be an object")
    _validate_bundled_schema(manifest)
    required = {
        "schema_version",
        "result_id",
        "job_id",
        "created_utc",
        "target_scene",
        "timeline_start",
        "source",
        "contracts",
        "profile",
        "coordinate_system",
        "counts",
        "arrays",
        "confidence_statistics",
        "warnings",
        "provenance",
        "logs",
    }
    if (
        not required.issubset(manifest)
        or not set(manifest).issubset(
            required | {"dense_predictions", "window_alignment", "sky_statistics"}
        )
        or manifest["schema_version"] != RESULT_SCHEMA_VERSION
        or not RESULT_ID.fullmatch(str(manifest["result_id"]))
        or not JOB_ID.fullmatch(str(manifest["job_id"]))
    ):
        raise ResultImportError("Result manifest identity or field set is invalid")
    try:
        created = datetime.fromisoformat(str(manifest["created_utc"]))
    except ValueError as exc:
        raise ResultImportError("created_utc is not ISO-8601") from exc
    if (
        created.tzinfo is None
        or created.utcoffset() is None
        or created.utcoffset().total_seconds() != 0
    ):
        raise ResultImportError("created_utc must include a UTC offset")
    descriptors = manifest["arrays"]
    if (
        not isinstance(descriptors, dict)
        or not CORE_ARRAYS.issubset(descriptors)
        or not set(descriptors).issubset(CORE_ARRAYS | OPTIONAL_ARRAYS)
    ):
        raise ResultImportError("Core Result array set is incomplete or unknown")
    arrays = {
        name: _array_descriptor(root, name, raw, cancel=cancel)
        for name, raw in descriptors.items()
    }
    point_count, frame_count = _validate_array_semantics(arrays)
    p5, p95 = _validate_manifest_contract(
        manifest, arrays, point_count, frame_count
    )
    logs = manifest["logs"]
    if not isinstance(logs, list):
        raise ResultImportError("Result logs must be an array")
    declared = {"manifest.json", *(f"arrays/{name}.npy" for name in arrays)}
    for raw in logs:
        descriptor = require_exact_object(
            raw, {"path", "byte_length", "sha256"}, label="log"
        )
        path_text = require_text(descriptor["path"], label="log.path", maximum=32767)
        if not path_text.startswith("logs/"):
            raise ResultImportError("Log path is not canonical")
        path = _plain_relative_file(root, path_text)
        digest, length = _stable_sha256(
            path, cancel=cancel, phase="validation:logs"
        )
        if (
            isinstance(descriptor["byte_length"], bool)
            or descriptor["byte_length"] != length
            or not SHA256.fullmatch(str(descriptor["sha256"]))
            or descriptor["sha256"] != digest
        ):
            raise ResultImportError("Log descriptor checksum or length is invalid")
        declared.add(path_text)
    # Optional Dense Predictions have their own contract and may be independently
    # unavailable. Core validation never follows that untrusted subtree.
    found: set[str] = set()
    for base, directories, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        if base_path == root and "dense" in directories:
            directories.remove("dense")
        if base_path.is_symlink() or is_reparse_point(base_path):
            raise ResultImportError("Result contains a linked directory")
        for filename in files:
            path = base_path / filename
            if path.is_symlink() or is_reparse_point(path):
                raise ResultImportError("Result contains a linked file")
            found.add(path.relative_to(root).as_posix())
    if found != declared:
        raise ResultImportError("Core Result contains undeclared or missing files")
    if (
        ready.result_id != manifest["result_id"]
        or ready.job_id != manifest["job_id"]
        or ready.point_count != point_count
        or ready.frame_count != frame_count
    ):
        raise ResultImportError("Discovery and complete validation disagree")
    final_manifest_digest, _ = _stable_sha256(
        manifest_path, cancel=cancel, phase="validation:manifest-final"
    )
    if final_manifest_digest != manifest_digest:
        raise ResultImportError("Result manifest changed during complete validation")
    for name, descriptor in descriptors.items():
        final_digest, final_length = _stable_sha256(
            _plain_relative_file(root, descriptor["path"]),
            cancel=cancel,
            phase=f"validation:{name}:final",
        )
        if (
            final_digest != descriptor["sha256"]
            or final_length != descriptor["byte_length"]
        ):
            raise ResultImportError(
                f"Array changed during complete validation: {name}"
            )
    for descriptor in logs:
        final_digest, final_length = _stable_sha256(
            _plain_relative_file(root, descriptor["path"]),
            cancel=cancel,
            phase="validation:logs-final",
        )
        if (
            final_digest != descriptor["sha256"]
            or final_length != descriptor["byte_length"]
        ):
            raise ResultImportError("Log changed during complete validation")
    return ValidatedResult(ready, manifest, manifest_digest, arrays, p5, p95)


def _metadata(document: ValidatedResult, scene: Any, kind: str) -> dict[str, Any]:
    target = document.manifest["target_scene"]
    return {
        "lingbot_map_owner_schema": OWNERSHIP_SCHEMA,
        "lingbot_map_kind": kind,
        "lingbot_map_result_id": document.ready.result_id,
        "lingbot_map_job_id": document.ready.job_id,
        "lingbot_map_manifest_sha256": document.manifest_sha256,
        "lingbot_map_original_blend_path": target["blend_path"],
        "lingbot_map_original_scene_uuid": target["scene_uuid"],
        "lingbot_map_actual_scene_uuid": str(
            scene.get("lingbot_map_scene_uuid", "")
        ),
    }


def _mark(datablock: Any, values: Mapping[str, Any]) -> None:
    for name, value in values.items():
        datablock[name] = value


def _valid_claim(collection: Any, document: ValidatedResult, scene: Any) -> bool:
    expected = _metadata(document, scene, "reconstruction_collection")
    if any(collection.get(name) != value for name, value in expected.items()):
        return False
    objects = tuple(collection.objects)
    if len(objects) != 2:
        return False
    kinds = {item.get("lingbot_map_kind") for item in objects}
    if kinds != {"reconstruction_root", "point_cloud"}:
        return False
    for item in objects:
        kind = item.get("lingbot_map_kind")
        values = _metadata(document, scene, kind)
        if any(item.get(name) != value for name, value in values.items()):
            return False
    return True


def _scene_collections(scene: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    pending = list(scene.collection.children)
    while pending:
        collection = pending.pop()
        result.append(collection)
        pending.extend(collection.children)
    return tuple(result)


def _find_existing(document: ValidatedResult, scene: Any) -> Any | None:
    claims = [
        collection
        for collection in _scene_collections(scene)
        if collection.get("lingbot_map_result_id") == document.ready.result_id
        and collection.get("lingbot_map_owner_schema") is not None
    ]
    valid = [
        collection for collection in claims if _valid_claim(collection, document, scene)
    ]
    if len(claims) > 1:
        raise ResultImportError(
            "Duplicate Imported Identity: multiple Collections claim this Result ID"
        )
    if claims and not valid:
        raise ResultImportError(
            "Ownership Unknown: matching Collection metadata is incomplete or contradictory"
        )
    return valid[0] if valid else None


def _select_existing(collection: Any, context: Any | None) -> None:
    collection.hide_viewport = False
    if context is None:
        return
    for item in getattr(context, "selected_objects", ()):
        item.select_set(False)
    for item in collection.objects:
        item.hide_set(False)
        item.select_set(True)
    point_object = next(
        (
            item
            for item in collection.objects
            if item.get("lingbot_map_kind") == "point_cloud"
        ),
        None,
    )
    if point_object is not None and getattr(context, "view_layer", None) is not None:
        context.view_layer.objects.active = point_object


def _socket(node: Any, name: str, fallback: int = 0) -> Any:
    return node.inputs.get(name) or node.inputs[fallback]


def _new_shader_group(bpy: Any, created: _Created) -> Any:
    for group in bpy.data.node_groups:
        if (
            group.bl_idname == "ShaderNodeTree"
            and group.get("lingbot_map_schema") == POINT_SHADER_SCHEMA
        ):
            return group
    group = bpy.data.node_groups.new(
        "LingBot Map Point Shader v1", "ShaderNodeTree"
    )
    created.node_groups.append(group)
    group["lingbot_map_schema"] = POINT_SHADER_SCHEMA
    group["lingbot_map_managed"] = True
    interface = group.interface
    for name, socket_type, default in (
        ("Brightness", "NodeSocketFloat", 1.0),
        ("Display Confidence", "NodeSocketFloat", 0.0),
        ("Confidence Low", "NodeSocketFloat", 0.0),
        ("Confidence High", "NodeSocketFloat", 1.0),
    ):
        socket = interface.new_socket(
            name=name, in_out="INPUT", socket_type=socket_type
        )
        socket.default_value = default
    interface.new_socket(
        name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader"
    )
    nodes, links = group.nodes, group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    color = nodes.new("ShaderNodeAttribute")
    color.attribute_name = "color"
    confidence = nodes.new("ShaderNodeAttribute")
    confidence.attribute_name = "confidence"
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.clamp = True
    links.new(confidence.outputs["Fac"], _socket(map_range, "Value"))
    links.new(
        group_input.outputs["Confidence Low"], _socket(map_range, "From Min")
    )
    links.new(
        group_input.outputs["Confidence High"], _socket(map_range, "From Max")
    )
    ramp = nodes.new("ShaderNodeValToRGB")
    viridis = (
        (0.0, (0.2667, 0.0039, 0.3294, 1.0)),
        (0.25, (0.2314, 0.3216, 0.5451, 1.0)),
        (0.50, (0.1294, 0.5686, 0.5490, 1.0)),
        (0.75, (0.3686, 0.7882, 0.3843, 1.0)),
        (1.0, (0.9922, 0.9059, 0.1451, 1.0)),
    )
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[1])
    first = ramp.color_ramp.elements[0]
    first.position, first.color = viridis[0]
    for position, rgba in viridis[1:]:
        element = ramp.color_ramp.elements.new(position)
        element.color = rgba
    links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    mix_color = nodes.new("ShaderNodeMixRGB")
    links.new(group_input.outputs["Display Confidence"], mix_color.inputs[0])
    links.new(color.outputs["Color"], mix_color.inputs[1])
    links.new(ramp.outputs["Color"], mix_color.inputs[2])
    emission = nodes.new("ShaderNodeEmission")
    links.new(mix_color.outputs["Color"], emission.inputs["Color"])
    links.new(group_input.outputs["Brightness"], emission.inputs["Strength"])
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    light_path = nodes.new("ShaderNodeLightPath")
    glossy_or_camera = nodes.new("ShaderNodeMath")
    glossy_or_camera.operation = "MAXIMUM"
    transmission_or_visible = nodes.new("ShaderNodeMath")
    transmission_or_visible.operation = "MAXIMUM"
    links.new(light_path.outputs["Is Camera Ray"], glossy_or_camera.inputs[0])
    links.new(light_path.outputs["Is Glossy Ray"], glossy_or_camera.inputs[1])
    links.new(glossy_or_camera.outputs[0], transmission_or_visible.inputs[0])
    links.new(
        light_path.outputs["Is Transmission Ray"], transmission_or_visible.inputs[1]
    )
    point_info = nodes.new("ShaderNodePointInfo")
    positive_radius = nodes.new("ShaderNodeMath")
    positive_radius.operation = "GREATER_THAN"
    positive_radius.inputs[1].default_value = 0.0
    links.new(point_info.outputs["Radius"], positive_radius.inputs[0])
    visibility = nodes.new("ShaderNodeMath")
    visibility.operation = "MULTIPLY"
    links.new(transmission_or_visible.outputs[0], visibility.inputs[0])
    links.new(positive_radius.outputs[0], visibility.inputs[1])
    mix_shader = nodes.new("ShaderNodeMixShader")
    links.new(visibility.outputs[0], mix_shader.inputs[0])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
    links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], group_output.inputs["Shader"])
    return group


def _new_material(
    bpy: Any, document: ValidatedResult, scene: Any, created: _Created
) -> Any:
    shader_group = _new_shader_group(bpy, created)
    material = bpy.data.materials.new(
        f"LingBot Map {document.ready.result_id[-8:]} Points"
    )
    created.materials.append(material)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeGroup")
    shader.node_tree = shader_group
    shader.inputs["Brightness"].default_value = 1.0
    shader.inputs["Display Confidence"].default_value = 0.0
    shader.inputs["Confidence Low"].default_value = document.confidence_p5
    high = document.confidence_p95
    if high <= document.confidence_p5:
        high = document.confidence_p5 + np.finfo(np.float32).eps
    shader.inputs["Confidence High"].default_value = high
    links.new(shader.outputs["Shader"], output.inputs["Surface"])
    values = _metadata(document, scene, "point_material")
    values.update(
        {
            "lingbot_map_schema": POINT_MATERIAL_SCHEMA,
            "lingbot_map_brightness": 1.0,
            "lingbot_map_display_mode": "RGB",
            "lingbot_map_confidence_p5": document.confidence_p5,
            "lingbot_map_confidence_p95": document.confidence_p95,
        }
    )
    _mark(material, values)
    return material


def _new_geometry_nodes(
    bpy: Any,
    point_object: Any,
    document: ValidatedResult,
    scene: Any,
    created: _Created,
) -> None:
    group = bpy.data.node_groups.new(
        f"LingBot Map {document.ready.result_id[-8:]} Point Display",
        "GeometryNodeTree",
    )
    created.node_groups.append(group)
    input_geometry = group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    scale_socket = group.interface.new_socket(
        name="Radius Scale", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    scale_socket.default_value = 1.0
    scale_socket.min_value = 0.0
    group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    nodes, links = group.nodes, group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    named = nodes.new("GeometryNodeInputNamedAttribute")
    named.data_type = "FLOAT"
    named.inputs["Name"].default_value = "radius"
    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    links.new(named.outputs["Attribute"], multiply.inputs[0])
    links.new(group_input.outputs["Radius Scale"], multiply.inputs[1])
    set_radius = nodes.new("GeometryNodeSetPointRadius")
    links.new(group_input.outputs["Geometry"], set_radius.inputs["Points"])
    links.new(multiply.outputs[0], set_radius.inputs["Radius"])
    links.new(set_radius.outputs["Points"], group_output.inputs["Geometry"])
    values = _metadata(document, scene, "point_display_nodes")
    values["lingbot_map_schema"] = POINT_DISPLAY_SCHEMA
    _mark(group, values)
    modifier = point_object.modifiers.new("LingBot Map Point Radius", "NODES")
    modifier.node_group = group
    getattr(modifier.properties.inputs, scale_socket.identifier).value = 1.0


def _rollback(bpy: Any, created: _Created) -> None:
    # Objects first release their data and collection links. Shared compatible
    # shader groups are tracked only when this attempt uniquely created them.
    for item in reversed(created.objects):
        if item.name in bpy.data.objects:
            bpy.data.objects.remove(item, do_unlink=True)
    for item in reversed(created.pointclouds):
        if item.name in bpy.data.pointclouds and item.users == 0:
            bpy.data.pointclouds.remove(item)
    for item in reversed(created.materials):
        if item.name in bpy.data.materials and item.users == 0:
            bpy.data.materials.remove(item)
    for item in reversed(created.node_groups):
        if item.name in bpy.data.node_groups and item.users == 0:
            bpy.data.node_groups.remove(item)
    for item in reversed(created.collections):
        if item.name in bpy.data.collections:
            bpy.data.collections.remove(item)


_import_running = False
_last_status = "Ready Results remain on disk until imported"


def get_import_status() -> str:
    return _last_status


def set_import_status(message: str) -> None:
    global _last_status
    _last_status = str(message)[:1024]


def import_result(
    directory: str | Path,
    scene: Any,
    *,
    context: Any | None = None,
    cancel: CancelCheck | None = None,
    available_probe: AvailableMemoryProbe = available_physical_memory,
    bpy_module: Any | None = None,
) -> ImportOutcome:
    """Validate, capacity-gate, stage, and atomically link one Collection."""

    global _import_running
    if _import_running:
        raise ResultImportError("Another LingBot Map import is already running")
    _import_running = True
    try:
        document = validate_result(directory, cancel=cancel)
        _cancelled(cancel, "capacity")
        capacity = evaluate_import_capacity(
            document.ready.point_count, available_probe=available_probe
        )
        existing = _find_existing(document, scene)
        if existing is not None:
            _select_existing(existing, context)
            outcome = ImportOutcome(
                document.ready.result_id,
                existing,
                False,
                capacity,
                "Result is already imported in this Scene",
            )
            set_import_status(outcome.message)
            return outcome
        if bpy_module is None:
            import bpy as bpy_module  # type: ignore[import-not-found]

        bpy = bpy_module
        created = _Created.empty()
        committed = False
        try:
            _cancelled(cancel, "staging_collection")
            collection = bpy.data.collections.new(
                f"Reconstruction {document.ready.created_utc[:10]} "
                f"{document.ready.job_id[-8:]}"
            )
            created.collections.append(collection)
            _cancelled(cancel, "root_object")
            root_object = bpy.data.objects.new("Reconstruction Root", None)
            created.objects.append(root_object)
            collection.objects.link(root_object)
            root_object.empty_display_type = "PLAIN_AXES"
            root_object.scale = (1.0, 1.0, 1.0)
            _cancelled(cancel, "pointcloud")
            pointcloud = bpy.data.pointclouds.new("Reconstructed Point Cloud")
            created.pointclouds.append(pointcloud)
            pointcloud.resize(document.ready.point_count)
            point_object = bpy.data.objects.new(
                "Reconstructed Point Cloud", pointcloud
            )
            created.objects.append(point_object)
            collection.objects.link(point_object)
            point_object.parent = root_object
            _cancelled(cancel, "positions")
            pointcloud.points.foreach_set(
                "co", np.asarray(document.arrays["positions"]).reshape(-1)
            )
            _cancelled(cancel, "color")
            color = pointcloud.attributes.new("color", "BYTE_COLOR", "POINT")
            rgba = np.empty((document.ready.point_count, 4), dtype=np.float32)
            rgba[:, :3] = np.asarray(document.arrays["colors"], dtype=np.float32) / 255.0
            rgba[:, 3] = 1.0
            color.data.foreach_set("color_srgb", rgba.reshape(-1))
            del rgba
            _cancelled(cancel, "confidence")
            confidence = pointcloud.attributes.new(
                "confidence", "FLOAT", "POINT"
            )
            confidence.data.foreach_set(
                "value", np.asarray(document.arrays["confidence"])
            )
            _cancelled(cancel, "radius")
            radius = pointcloud.attributes.get("radius")
            if radius is None:
                radius = pointcloud.attributes.new("radius", "FLOAT", "POINT")
            radius.data.foreach_set(
                "value", np.asarray(document.arrays["radius"])
            )
            _cancelled(cancel, "source_frame")
            source_frame = pointcloud.attributes.new(
                "source_frame", "INT", "POINT"
            )
            source_frame.data.foreach_set(
                "value", np.asarray(document.arrays["source_frame"], dtype=np.int32)
            )
            _cancelled(cancel, "material")
            material = _new_material(bpy, document, scene, created)
            pointcloud.materials.append(material)
            _cancelled(cancel, "geometry_nodes")
            _new_geometry_nodes(
                bpy, point_object, document, scene, created
            )
            _cancelled(cancel, "ownership")
            _mark(
                collection,
                _metadata(document, scene, "reconstruction_collection"),
            )
            _mark(root_object, _metadata(document, scene, "reconstruction_root"))
            _mark(point_object, _metadata(document, scene, "point_cloud"))
            _mark(pointcloud, _metadata(document, scene, "point_cloud_data"))
            _cancelled(cancel, "commit")
            scene.collection.children.link(collection)
            committed = True
            _select_existing(collection, context)
            outcome = ImportOutcome(
                document.ready.result_id,
                collection,
                True,
                capacity,
                "Imported; save the Blender file to retain this Collection",
            )
            set_import_status(outcome.message)
            return outcome
        except Exception:
            if not committed:
                _rollback(bpy, created)
            raise
    finally:
        _import_running = False


_auto_attempted: set[tuple[str, str]] = set()


def reset_auto_import_attempts_for_tests() -> None:
    _auto_attempted.clear()


def _normal_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def _idle_for_auto_import(bpy: Any) -> bool:
    if _import_running:
        return False
    screen = getattr(bpy.context, "screen", None)
    if screen is not None and bool(getattr(screen, "is_animation_playing", False)):
        return False
    is_job_running = getattr(bpy.app, "is_job_running", None)
    if callable(is_job_running):
        for name in ("RENDER", "OBJECT_BAKE", "COMPOSITE"):
            try:
                if is_job_running(name):
                    return False
            except (RuntimeError, TypeError):
                continue
    return True


def attempt_auto_import_once(
    ready: ReadyResult,
    *,
    expected_job_id: str,
    bpy_module: Any | None = None,
    available_probe: AvailableMemoryProbe = available_physical_memory,
) -> ImportOutcome | None:
    """Make one immediate completion-time attempt; never queue a later retry."""

    if bpy_module is None:
        import bpy as bpy_module  # type: ignore[import-not-found]
    bpy = bpy_module
    blend_path = str(getattr(bpy.data, "filepath", ""))
    key = (_normal_path(blend_path), ready.result_id)
    if key in _auto_attempted:
        return None
    _auto_attempted.add(key)
    if not blend_path or ready.job_id != expected_job_id or not _idle_for_auto_import(bpy):
        return None
    document = validate_result(ready.directory)
    evaluate_import_capacity(
        document.ready.point_count, available_probe=available_probe
    )
    target = document.manifest["target_scene"]
    if _normal_path(blend_path) != _normal_path(target["blend_path"]):
        return None
    scene = getattr(bpy.context, "scene", None)
    if scene is None or scene.get("lingbot_map_scene_uuid") != target["scene_uuid"]:
        return None
    if (
        sum(
            item.get("lingbot_map_scene_uuid") == target["scene_uuid"]
            for item in bpy.data.scenes
        )
        != 1
    ):
        return None
    return import_result(
        ready.directory,
        scene,
        context=bpy.context,
        available_probe=available_probe,
        bpy_module=bpy,
    )


__all__ = [
    "CAPACITY_MODEL_VERSION",
    "IMPORT_BYTES_PER_POINT",
    "IMPORT_FIXED_BYTES",
    "IMPORT_PHASES",
    "ImportCapacity",
    "ImportCapacityError",
    "ImportOutcome",
    "ResultImportCancelled",
    "ResultImportError",
    "ValidatedResult",
    "attempt_auto_import_once",
    "available_physical_memory",
    "evaluate_import_capacity",
    "get_import_status",
    "import_result",
    "set_import_status",
    "validate_result",
]
