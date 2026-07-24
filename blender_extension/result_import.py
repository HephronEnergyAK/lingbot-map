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
from .job_lifecycle import (
    ensure_unique_scene_uuid,
    is_reparse_point,
    normalized_blend_path,
    project_result_root,
)
from .results import (
    CORE_ARRAYS,
    JOB_ID,
    OPTIONAL_ARRAYS,
    RESULT_ID,
    RESULT_SCHEMA_VERSION,
    ReadyResult,
    read_ready_result,
)
from .source_view import (
    BACKGROUND_MAPPINGS,
    SourceViewContract,
    SourceViewError,
    background_scale,
    candidate_source_paths,
    coverage_to_camera_border,
    scene_aspect_matches,
    validate_source_media,
    validate_source_view_contract,
)


OWNERSHIP_SCHEMA = "1.0.0"
OWNERSHIP_IDENTITY_FIELDS = (
    "lingbot_map_owner_schema",
    "lingbot_map_result_id",
    "lingbot_map_job_id",
    "lingbot_map_manifest_sha256",
    "lingbot_map_original_blend_path",
    "lingbot_map_original_scene_uuid",
    "lingbot_map_actual_scene_uuid",
)
REFERENCE_FIELDS = (
    "lingbot_map_import_blend_path",
    "lingbot_map_result_reference_mode",
    "lingbot_map_result_reference_relative",
    "lingbot_map_result_reference_absolute",
)
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
    "camera",
    "camera_animation",
    "trajectory",
    "source_background",
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
    source_view: SourceViewContract | None


@dataclass(frozen=True)
class ImportOutcome:
    result_id: str
    collection: Any
    created: bool
    capacity: ImportCapacity
    message: str


@dataclass(frozen=True)
class OwnershipInspection:
    status: str
    message: str
    datablocks: tuple[Any, ...]


@dataclass(frozen=True)
class RemovalInventory:
    collection_name: str
    object_count: int
    point_count: int
    shared_datablock_count: int


@dataclass
class _Created:
    collections: list[Any]
    objects: list[Any]
    pointclouds: list[Any]
    cameras: list[Any]
    curves: list[Any]
    movieclips: list[Any]
    actions: list[Any]
    materials: list[Any]
    node_groups: list[Any]

    @classmethod
    def empty(cls) -> "_Created":
        return cls([], [], [], [], [], [], [], [], [])


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
            required
            | {
                "dense_predictions",
                "window_alignment",
                "sky_statistics",
                "source_display",
                "model_coverage",
            }
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
    try:
        source_view = validate_source_view_contract(
            manifest, np.asarray(arrays["source_to_model"])
        )
    except SourceViewError as exc:
        raise ResultImportError(str(exc)) from exc
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
    return ValidatedResult(
        ready, manifest, manifest_digest, arrays, p5, p95, source_view
    )


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


def _relative_result_reference(
    directory: str | Path, blend_path: str | Path
) -> str:
    result = Path(os.path.abspath(directory))
    blend = normalized_blend_path(blend_path)
    try:
        relative = os.path.relpath(result, blend.parent)
    except ValueError:
        return ""
    return "//" + relative.replace(os.sep, "/")


def _reference_metadata(
    document: ValidatedResult, import_blend_path: str | Path
) -> dict[str, str]:
    current = normalized_blend_path(import_blend_path)
    original = normalized_blend_path(
        document.manifest["target_scene"]["blend_path"]
    )
    directory = Path(os.path.abspath(document.ready.directory))
    expected_parent = project_result_root(original) / "results"
    local = (
        os.path.normcase(str(current)) == os.path.normcase(str(original))
        and os.path.normcase(str(directory.parent))
        == os.path.normcase(str(expected_parent))
    )
    return {
        "lingbot_map_import_blend_path": str(current),
        "lingbot_map_result_reference_mode": (
            "project-owned" if local else "external-read-only"
        ),
        "lingbot_map_result_reference_relative": _relative_result_reference(
            directory, current
        ),
        "lingbot_map_result_reference_absolute": str(directory),
    }


def effective_disk_authority(
    collection: Any, current_blend_path: str | Path
) -> str:
    """Return current authority; Save As always reclassifies old data external."""

    try:
        current = normalized_blend_path(current_blend_path)
        imported = normalized_blend_path(
            str(collection.get("lingbot_map_import_blend_path", ""))
        )
    except (OSError, ValueError, RuntimeError):
        return "external-read-only"
    if os.path.normcase(str(current)) != os.path.normcase(str(imported)):
        return "external-read-only"
    mode = collection.get("lingbot_map_result_reference_mode")
    return mode if mode in {"project-owned", "external-read-only"} else "unknown"


def _owner_identity(collection: Any) -> dict[str, str] | None:
    identity = {
        name: collection.get(name) for name in OWNERSHIP_IDENTITY_FIELDS
    }
    if identity["lingbot_map_owner_schema"] != OWNERSHIP_SCHEMA:
        return None
    if (
        not isinstance(identity["lingbot_map_result_id"], str)
        or RESULT_ID.fullmatch(identity["lingbot_map_result_id"]) is None
        or not isinstance(identity["lingbot_map_job_id"], str)
        or JOB_ID.fullmatch(identity["lingbot_map_job_id"]) is None
        or not isinstance(identity["lingbot_map_manifest_sha256"], str)
        or SHA256.fullmatch(identity["lingbot_map_manifest_sha256"]) is None
    ):
        return None
    try:
        uuid.UUID(str(identity["lingbot_map_original_scene_uuid"]))
        uuid.UUID(str(identity["lingbot_map_actual_scene_uuid"]))
        normalized_blend_path(str(identity["lingbot_map_original_blend_path"]))
    except (ValueError, OSError, RuntimeError):
        return None
    if any(
        not isinstance(collection.get(name), str)
        for name in REFERENCE_FIELDS
    ):
        return None
    try:
        normalized_blend_path(
            collection.get("lingbot_map_import_blend_path")
        )
        absolute = Path(
            os.path.abspath(
                collection.get("lingbot_map_result_reference_absolute")
            )
        )
    except (ValueError, OSError, RuntimeError):
        return None
    relative = collection.get("lingbot_map_result_reference_relative")
    mode = collection.get("lingbot_map_result_reference_mode")
    if (
        not absolute.is_absolute()
        or (relative and not relative.startswith("//"))
        or mode not in {"project-owned", "external-read-only"}
    ):
        return None
    return {name: str(value) for name, value in identity.items()}


def _owner_matches(
    datablock: Any, identity: Mapping[str, str], kind: str
) -> bool:
    return (
        datablock is not None
        and datablock.get("lingbot_map_kind") == kind
        and all(datablock.get(name) == value for name, value in identity.items())
    )


def _owned_action(datablock: Any) -> Any | None:
    animation = getattr(datablock, "animation_data", None)
    return getattr(animation, "action", None)


def inspect_collection_ownership(
    collection: Any, scene: Any
) -> OwnershipInspection:
    """Validate the complete managed graph without relying on display names."""

    identity = _owner_identity(collection)
    if (
        identity is None
        or collection.get("lingbot_map_kind")
        != "reconstruction_collection"
        or str(scene.get("lingbot_map_scene_uuid", ""))
        != identity.get("lingbot_map_actual_scene_uuid")
    ):
        return OwnershipInspection(
            "unknown",
            "Ownership Unknown: Collection identity is incomplete or contradictory",
            (),
        )
    expected_object_kinds = {"reconstruction_root", "point_cloud"}
    has_source_view = bool(collection.get("lingbot_map_source_display_json"))
    if has_source_view:
        expected_object_kinds.update(
            {"reconstruction_camera", "camera_trajectory"}
        )
    owned_objects: dict[str, Any] = {}
    unowned_objects: list[Any] = []
    for item in collection.objects:
        kind = item.get("lingbot_map_kind")
        if kind in expected_object_kinds:
            if kind in owned_objects or not _owner_matches(
                item, identity, kind
            ):
                return OwnershipInspection(
                    "unknown",
                    "Ownership Unknown: managed Object metadata is contradictory",
                    (),
                )
            owned_objects[kind] = item
        elif any(item.get(name) is not None for name in OWNERSHIP_IDENTITY_FIELDS):
            return OwnershipInspection(
                "unknown",
                "Ownership Unknown: unexpected Object claims managed identity",
                (),
            )
        else:
            unowned_objects.append(item)
    if set(owned_objects) != expected_object_kinds:
        return OwnershipInspection(
            "unknown",
            "Ownership Unknown: required managed Objects are missing",
            (),
        )
    for child in collection.children:
        if any(
            child.get(name) is not None
            for name in OWNERSHIP_IDENTITY_FIELDS
        ):
            return OwnershipInspection(
                "unknown",
                "Ownership Unknown: nested Collection claims managed identity",
                (),
            )

    graph: list[Any] = [collection, *owned_objects.values()]
    point_object = owned_objects["point_cloud"]
    point_data = getattr(point_object, "data", None)
    if not _owner_matches(point_data, identity, "point_cloud_data"):
        return OwnershipInspection(
            "unknown",
            "Ownership Unknown: PointCloud data ownership is invalid",
            (),
        )
    graph.append(point_data)
    materials = tuple(getattr(point_data, "materials", ()))
    if (
        len(materials) != 1
        or not _owner_matches(materials[0], identity, "point_material")
        or materials[0].get("lingbot_map_schema") != POINT_MATERIAL_SCHEMA
    ):
        return OwnershipInspection(
            "unknown",
            "Ownership Unknown: point Material ownership is invalid",
            (),
        )
    graph.append(materials[0])
    geometry_groups = tuple(
        getattr(modifier, "node_group", None)
        for modifier in point_object.modifiers
        if getattr(modifier, "type", None) == "NODES"
    )
    geometry_groups = tuple(item for item in geometry_groups if item is not None)
    if (
        len(geometry_groups) != 1
        or not _owner_matches(
            geometry_groups[0], identity, "point_display_nodes"
        )
        or geometry_groups[0].get("lingbot_map_schema")
        != POINT_DISPLAY_SCHEMA
    ):
        return OwnershipInspection(
            "unknown",
            "Ownership Unknown: Geometry Nodes ownership is invalid",
            (),
        )
    graph.append(geometry_groups[0])
    shader_groups = tuple(
        node.node_tree
        for node in materials[0].node_tree.nodes
        if getattr(node, "type", None) == "GROUP"
        and getattr(node, "node_tree", None) is not None
    )
    if (
        len(shader_groups) != 1
        or shader_groups[0].get("lingbot_map_schema")
        != POINT_SHADER_SCHEMA
        or not bool(shader_groups[0].get("lingbot_map_managed", False))
    ):
        return OwnershipInspection(
            "unknown",
            "Ownership Unknown: shared point shader schema is invalid",
            (),
        )

    if has_source_view:
        for object_kind, data_kind in (
            ("reconstruction_camera", "reconstruction_camera_data"),
            ("camera_trajectory", "camera_trajectory_data"),
        ):
            data = getattr(owned_objects[object_kind], "data", None)
            if not _owner_matches(data, identity, data_kind):
                return OwnershipInspection(
                    "unknown",
                    f"Ownership Unknown: {data_kind} ownership is invalid",
                    (),
                )
            graph.append(data)
        for owner in (
            owned_objects["reconstruction_camera"],
            owned_objects["reconstruction_camera"].data,
        ):
            action = _owned_action(owner)
            if action is not None:
                if not _owner_matches(
                    action, identity, "camera_animation_action"
                ):
                    return OwnershipInspection(
                        "unknown",
                        "Ownership Unknown: camera Action ownership is invalid",
                        (),
                    )
                graph.append(action)
        backgrounds = tuple(
            owned_objects[
                "reconstruction_camera"
            ].data.background_images
        )
        background_status = str(
            collection.get(
                "lingbot_map_source_background_status", ""
            )
        )
        if (
            background_status == "attached-hidden"
            and len(backgrounds) != 1
        ) or (
            background_status != "attached-hidden"
            and backgrounds
        ):
            return OwnershipInspection(
                "unknown",
                "Ownership Unknown: Source Background attachment is contradictory",
                (),
            )
        for background in backgrounds:
            clip = getattr(background, "clip", None)
            if (
                getattr(background, "source", None) != "MOVIE_CLIP"
                or not _owner_matches(
                    clip, identity, "source_movie_clip"
                )
            ):
                return OwnershipInspection(
                    "unknown",
                    "Ownership Unknown: Source Background clip ownership is invalid",
                    (),
                )
            graph.append(clip)
    unique: list[Any] = []
    seen: set[int] = set()
    for datablock in graph:
        key = int(datablock.as_pointer()) if hasattr(datablock, "as_pointer") else id(datablock)
        if key not in seen:
            seen.add(key)
            unique.append(datablock)
    return OwnershipInspection(
        "managed",
        "Managed ownership metadata is complete and consistent",
        tuple(unique),
    )


def _valid_claim(collection: Any, document: ValidatedResult, scene: Any) -> bool:
    expected = _metadata(document, scene, "reconstruction_collection")
    return (
        all(collection.get(name) == value for name, value in expected.items())
        and inspect_collection_ownership(collection, scene).status == "managed"
    )


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


def _track_action(created: _Created, datablock: Any) -> None:
    animation_data = getattr(datablock, "animation_data", None)
    action = getattr(animation_data, "action", None)
    if action is not None and action not in created.actions:
        created.actions.append(action)


def _set_actions_linear(created: _Created) -> None:
    """Handle Blender 5.2 layered Actions, including Object and Camera slots."""

    for action in created.actions:
        for layer in action.layers:
            for strip in layer.strips:
                for channelbag in strip.channelbags:
                    for fcurve in channelbag.fcurves:
                        for keyframe in fcurve.keyframe_points:
                            keyframe.interpolation = "LINEAR"


def _combined_bounds_diagonal(document: ValidatedResult) -> float:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    positions = document.arrays["positions"]
    for start in range(0, len(positions), SEMANTIC_CHUNK_POINTS):
        values = np.asarray(
            positions[start : start + SEMANTIC_CHUNK_POINTS],
            dtype=np.float64,
        )
        if len(values):
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
    cameras = np.asarray(
        document.arrays["camera_to_world"][:, :3, 3], dtype=np.float64
    )
    minimum = np.minimum(minimum, cameras.min(axis=0))
    maximum = np.maximum(maximum, cameras.max(axis=0))
    return float(np.linalg.norm(maximum - minimum))


def _new_camera_animation(
    bpy: Any,
    collection: Any,
    root_object: Any,
    document: ValidatedResult,
    scene: Any,
    created: _Created,
) -> tuple[Any, Any]:
    from mathutils import Matrix

    contract = document.source_view
    if contract is None:
        raise ResultImportError("Source-view contract is unavailable")
    camera_data = bpy.data.cameras.new(
        f"LingBot Map {document.ready.result_id[-8:]} Camera"
    )
    created.cameras.append(camera_data)
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_object = bpy.data.objects.new(
        "Reconstruction Camera", camera_data
    )
    created.objects.append(camera_object)
    collection.objects.link(camera_object)
    camera_object.parent = root_object
    camera_object.rotation_mode = "QUATERNION"

    camera_to_world = document.arrays["camera_to_world"]
    intrinsics = document.arrays["source_intrinsics"]
    timeline_start = int(document.manifest["timeline_start"])
    previous_rotation = None
    for index in range(document.ready.frame_count):
        matrix = Matrix(
            np.asarray(camera_to_world[index], dtype=np.float64).tolist()
        )
        location, rotation, scale = matrix.decompose()
        if not all(
            math.isclose(float(value), 1.0, abs_tol=1e-4, rel_tol=0)
            for value in scale
        ):
            raise ResultImportError(
                "camera_to_world contains scale outside the rigid contract"
            )
        if (
            previous_rotation is not None
            and rotation.dot(previous_rotation) < 0
        ):
            rotation.negate()
        previous_rotation = rotation.copy()
        camera_object.location = location
        camera_object.rotation_quaternion = rotation
        source = np.asarray(intrinsics[index], dtype=np.float64)
        fx = float(source[0, 0])
        cx = float(source[0, 2])
        cy = float(source[1, 2])
        camera_data.lens = 36.0 * fx / contract.width
        camera_data.shift_x = (contract.width * 0.5 - cx) / contract.width
        camera_data.shift_y = (cy - contract.height * 0.5) / contract.width
        frame = timeline_start + index
        camera_object.keyframe_insert(data_path="location", frame=frame)
        camera_object.keyframe_insert(
            data_path="rotation_quaternion", frame=frame
        )
        camera_data.keyframe_insert(data_path="lens", frame=frame)
        camera_data.keyframe_insert(data_path="shift_x", frame=frame)
        camera_data.keyframe_insert(data_path="shift_y", frame=frame)
    _track_action(created, camera_object)
    _track_action(created, camera_data)
    _set_actions_linear(created)

    diagonal = _combined_bounds_diagonal(document)
    camera_data.clip_start = max(diagonal * 1e-5, 1e-6)
    camera_data.clip_end = max(
        diagonal * 1.1, camera_data.clip_start * 1000.0
    )
    camera_data.display_size = max(diagonal * 0.025, 0.05)
    return camera_object, camera_data


def _new_camera_trajectory(
    bpy: Any,
    collection: Any,
    root_object: Any,
    document: ValidatedResult,
    created: _Created,
) -> tuple[Any, Any]:
    curve = bpy.data.curves.new(
        f"LingBot Map {document.ready.result_id[-8:]} Camera Trajectory",
        "CURVE",
    )
    created.curves.append(curve)
    curve.dimensions = "3D"
    curve.resolution_u = 1
    spline = curve.splines.new("POLY")
    spline.points.add(document.ready.frame_count - 1)
    camera_positions = np.asarray(
        document.arrays["camera_to_world"][:, :3, 3], dtype=np.float64
    )
    homogeneous = np.ones(
        (document.ready.frame_count, 4), dtype=np.float64
    )
    homogeneous[:, :3] = camera_positions
    spline.points.foreach_set("co", homogeneous.reshape(-1))
    trajectory = bpy.data.objects.new("Camera Trajectory", curve)
    created.objects.append(trajectory)
    collection.objects.link(trajectory)
    trajectory.parent = root_object
    trajectory.hide_render = True
    trajectory.show_in_front = True
    return trajectory, curve


def _remove_movieclip_if_unused(bpy: Any, clip: Any) -> None:
    if clip is not None and clip.name in bpy.data.movieclips and clip.users == 0:
        bpy.data.movieclips.remove(clip)


def _load_source_clip(
    bpy: Any,
    document: ValidatedResult,
    *,
    current_blend_path: str | Path | None,
    relink_path: str | Path | None = None,
    cancel: CancelCheck | None = None,
) -> tuple[Any, Path]:
    contract = document.source_view
    if contract is None:
        raise SourceViewError("Source-view contract is unavailable")
    source = document.manifest["source"]
    errors: list[str] = []
    media = None
    for candidate in candidate_source_paths(
        source,
        current_blend_path=current_blend_path,
        relink_path=relink_path,
    ):
        try:
            media = validate_source_media(
                candidate,
                source,
                cancel=(
                    (lambda: (_cancelled(cancel, "source_background") or False))
                    if cancel is not None
                    else None
                ),
            )
            break
        except SourceViewError as exc:
            errors.append(str(exc))
    if media is None:
        raise SourceViewError(
            errors[-1] if errors else "Capture Source media is unavailable"
        )
    clip = None
    try:
        clip = bpy.data.movieclips.load(str(media), check_existing=False)
        if tuple(int(value) for value in clip.size) != contract.coded_size:
            raise SourceViewError(
                "Capture Source coded dimensions disagree with Display Transform"
            )
        if int(clip.frame_duration) != document.ready.frame_count:
            raise SourceViewError(
                "Capture Source frame duration disagrees with the Result"
            )
        clip.frame_start = int(document.manifest["timeline_start"])
        return clip, media
    except ResultImportCancelled:
        _remove_movieclip_if_unused(bpy, clip)
        raise
    except SourceViewError:
        _remove_movieclip_if_unused(bpy, clip)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        _remove_movieclip_if_unused(bpy, clip)
        raise SourceViewError(
            f"Capture Source could not be loaded as a supported MovieClip: {exc}"
        ) from exc


def _configure_background(
    camera_data: Any,
    clip: Any,
    contract: SourceViewContract,
) -> Any:
    background = camera_data.background_images.new()
    background.source = "MOVIE_CLIP"
    background.clip = clip
    background.use_camera_clip = False
    background.frame_method = "FIT"
    background.offset = (0.0, 0.0)
    # FIT is evaluated before Blender rotates the coded movie. Compensate its
    # aspect-derived shrink for quarter-turn and diagonal display transforms.
    background.scale = background_scale(contract)
    mapping = BACKGROUND_MAPPINGS[contract.display_transform]
    background.rotation = mapping.rotation_radians
    background.use_flip_x = mapping.flip_x
    background.use_flip_y = mapping.flip_y
    # Blender 5.2's BACK mode is occluded by the solid viewport background.
    # FRONT is still a camera-only 2D overlay (not arbitrary 3D depth) and is
    # required for the explicitly shown Source Background to be visible.
    background.display_depth = "FRONT"
    background.alpha = 1.0
    background.show_background_image = False
    camera_data.show_background_images = True
    return background


def _try_attach_source_background(
    bpy: Any,
    document: ValidatedResult,
    scene: Any,
    camera_data: Any,
    created: _Created,
    *,
    cancel: CancelCheck | None,
) -> tuple[str, str | None]:
    contract = document.source_view
    if contract is None:
        return "unavailable-legacy-result", None
    if not scene_aspect_matches(scene, contract):
        return "unattached-scene-aspect-mismatch", None
    try:
        clip, media = _load_source_clip(
            bpy,
            document,
            current_blend_path=getattr(bpy.data, "filepath", ""),
            cancel=cancel,
        )
    except SourceViewError as exc:
        return f"unattached-{exc}", None
    created.movieclips.append(clip)
    _configure_background(camera_data, clip, contract)
    return "attached-hidden", str(media)


def _source_view_properties(
    document: ValidatedResult,
    *,
    background_status: str,
    source_path: str | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "lingbot_map_timeline_start": int(document.manifest["timeline_start"]),
        "lingbot_map_frame_count": int(document.ready.frame_count),
        "lingbot_map_source_json": json.dumps(
            document.manifest["source"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "lingbot_map_source_background_status": background_status[:1024],
        "lingbot_map_warnings_json": json.dumps(
            document.manifest["warnings"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    if document.source_view is not None:
        values["lingbot_map_source_display_json"] = json.dumps(
            document.manifest["source_display"],
            sort_keys=True,
            separators=(",", ":"),
        )
        values["lingbot_map_model_coverage_json"] = json.dumps(
            document.manifest["model_coverage"],
            sort_keys=True,
            separators=(",", ":"),
        )
        values["lingbot_map_source_to_model_json"] = json.dumps(
            np.asarray(document.arrays["source_to_model"]).tolist(),
            separators=(",", ":"),
        )
    if source_path is not None:
        values["lingbot_map_source_background_path"] = source_path
    return values


def _rollback(bpy: Any, created: _Created) -> None:
    # Objects first release their data and collection links. Shared compatible
    # shader groups are tracked only when this attempt uniquely created them.
    for item in reversed(created.objects):
        if item.name in bpy.data.objects:
            bpy.data.objects.remove(item, do_unlink=True)
    for item in reversed(created.pointclouds):
        if item.name in bpy.data.pointclouds and item.users == 0:
            bpy.data.pointclouds.remove(item)
    for item in reversed(created.cameras):
        if item.name in bpy.data.cameras and item.users == 0:
            bpy.data.cameras.remove(item)
    for item in reversed(created.curves):
        if item.name in bpy.data.curves and item.users == 0:
            bpy.data.curves.remove(item)
    for item in reversed(created.movieclips):
        _remove_movieclip_if_unused(bpy, item)
    for item in reversed(created.materials):
        if item.name in bpy.data.materials and item.users == 0:
            bpy.data.materials.remove(item)
    for item in reversed(created.node_groups):
        if item.name in bpy.data.node_groups and item.users == 0:
            bpy.data.node_groups.remove(item)
    for item in reversed(created.actions):
        if item.name in bpy.data.actions and item.users == 0:
            bpy.data.actions.remove(item)
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
        if bpy_module is None:
            import bpy as bpy_module  # type: ignore[import-not-found]

        bpy = bpy_module
        try:
            ensure_unique_scene_uuid(
                scene,
                tuple(bpy.data.scenes),
            )
            current_blend_path = str(getattr(bpy.data, "filepath", ""))
            normalized_blend_path(current_blend_path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ResultImportError(str(exc)) from exc
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
        created = _Created.empty()
        committed = False
        original_frame_end = int(scene.frame_end)
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
            camera_object = None
            camera_data = None
            trajectory = None
            trajectory_data = None
            background_status = "unavailable-legacy-result"
            background_path = None
            _cancelled(cancel, "camera")
            if document.source_view is not None:
                camera_object, camera_data = _new_camera_animation(
                    bpy,
                    collection,
                    root_object,
                    document,
                    scene,
                    created,
                )
            _cancelled(cancel, "camera_animation")
            _cancelled(cancel, "trajectory")
            if document.source_view is not None:
                trajectory, trajectory_data = _new_camera_trajectory(
                    bpy,
                    collection,
                    root_object,
                    document,
                    created,
                )
            _cancelled(cancel, "source_background")
            if camera_data is not None:
                background_status, background_path = (
                    _try_attach_source_background(
                        bpy,
                        document,
                        scene,
                        camera_data,
                        created,
                        cancel=cancel,
                    )
                )
            _cancelled(cancel, "ownership")
            _mark(
                collection,
                _metadata(document, scene, "reconstruction_collection"),
            )
            _mark(
                collection,
                _reference_metadata(document, current_blend_path),
            )
            _mark(
                collection,
                _source_view_properties(
                    document,
                    background_status=background_status,
                    source_path=background_path,
                ),
            )
            _mark(root_object, _metadata(document, scene, "reconstruction_root"))
            _mark(point_object, _metadata(document, scene, "point_cloud"))
            _mark(pointcloud, _metadata(document, scene, "point_cloud_data"))
            if camera_object is not None and camera_data is not None:
                _mark(
                    camera_object,
                    _metadata(document, scene, "reconstruction_camera"),
                )
                _mark(
                    camera_data,
                    _metadata(document, scene, "reconstruction_camera_data"),
                )
                camera_data["lingbot_map_source_background_status"] = (
                    background_status
                )
                camera_data["lingbot_map_source_background_playback"] = (
                    "SEQUENTIAL_NON_CYCLING"
                )
                if background_path is not None:
                    camera_data["lingbot_map_source_background_path"] = (
                        background_path
                    )
            if trajectory is not None and trajectory_data is not None:
                _mark(
                    trajectory,
                    _metadata(document, scene, "camera_trajectory"),
                )
                _mark(
                    trajectory_data,
                    _metadata(document, scene, "camera_trajectory_data"),
                )
            for action in created.actions:
                _mark(
                    action,
                    _metadata(
                        document, scene, "camera_animation_action"
                    ),
                )
            for movieclip in created.movieclips:
                _mark(
                    movieclip,
                    _metadata(document, scene, "source_movie_clip"),
                )
            _cancelled(cancel, "commit")
            scene.collection.children.link(collection)
            if document.source_view is not None:
                last_frame = (
                    int(document.manifest["timeline_start"])
                    + document.ready.frame_count
                    - 1
                )
                if scene.frame_end < last_frame:
                    scene.frame_end = last_frame
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
                if scene.frame_end != original_frame_end:
                    scene.frame_end = original_frame_end
                _rollback(bpy, created)
            raise
    finally:
        _import_running = False


def _json_property(
    collection: Any, name: str, expected_type: type
) -> Any:
    raw = collection.get(name)
    if not isinstance(raw, str) or not raw or len(raw) > 1024 * 1024:
        raise ResultImportError(f"Imported source-view metadata is missing: {name}")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ResultImportError(
            f"Imported source-view metadata is invalid: {name}"
        ) from exc
    if not isinstance(value, expected_type):
        raise ResultImportError(
            f"Imported source-view metadata has the wrong type: {name}"
        )
    return value


def imported_source_view_contract(collection: Any) -> SourceViewContract:
    manifest = {
        "source_display": _json_property(
            collection, "lingbot_map_source_display_json", dict
        ),
        "model_coverage": _json_property(
            collection, "lingbot_map_model_coverage_json", dict
        ),
    }
    source_to_model = _json_property(
        collection, "lingbot_map_source_to_model_json", list
    )
    try:
        contract = validate_source_view_contract(manifest, source_to_model)
    except SourceViewError as exc:
        raise ResultImportError(str(exc)) from exc
    if contract is None:
        raise ResultImportError("Imported Result has no source-view contract")
    return contract


def find_reconstruction_camera(collection: Any) -> Any:
    matches = [
        item
        for item in collection.objects
        if item.get("lingbot_map_kind") == "reconstruction_camera"
        and getattr(item, "type", None) == "CAMERA"
    ]
    if len(matches) != 1:
        raise ResultImportError(
            "Imported Result does not own exactly one Reconstruction Camera"
        )
    return matches[0]


def _load_imported_source_clip(
    bpy: Any,
    collection: Any,
    contract: SourceViewContract,
    *,
    relink_path: str | Path | None = None,
) -> tuple[Any, Path]:
    source = _json_property(collection, "lingbot_map_source_json", dict)
    preferred = relink_path or collection.get(
        "lingbot_map_relinked_source_path"
    )
    errors: list[str] = []
    media = None
    candidates = (
        (Path(os.path.abspath(relink_path)),)
        if relink_path is not None
        else candidate_source_paths(
            source,
            current_blend_path=getattr(bpy.data, "filepath", ""),
            relink_path=preferred,
        )
    )
    for candidate in candidates:
        try:
            media = validate_source_media(candidate, source)
            break
        except SourceViewError as exc:
            errors.append(str(exc))
    if media is None:
        raise ResultImportError(
            errors[-1] if errors else "Capture Source media is unavailable"
        )
    frame_count = collection.get("lingbot_map_frame_count")
    timeline_start = collection.get("lingbot_map_timeline_start")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
        or isinstance(timeline_start, bool)
        or not isinstance(timeline_start, int)
    ):
        raise ResultImportError(
            "Imported source-view timeline metadata is invalid"
        )
    clip = None
    try:
        clip = bpy.data.movieclips.load(str(media), check_existing=False)
        if tuple(int(value) for value in clip.size) != contract.coded_size:
            raise ResultImportError(
                "Capture Source coded dimensions disagree with Display Transform"
            )
        if int(clip.frame_duration) != frame_count:
            raise ResultImportError(
                "Capture Source frame duration disagrees with the Result"
            )
        clip.frame_start = timeline_start
        identity = _owner_identity(collection)
        if identity is None:
            raise ResultImportError(
                "Ownership Unknown: cannot attach media to this Collection"
            )
        _mark(
            clip,
            {
                **identity,
                "lingbot_map_kind": "source_movie_clip",
            },
        )
        return clip, media
    except ResultImportError:
        _remove_movieclip_if_unused(bpy, clip)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        _remove_movieclip_if_unused(bpy, clip)
        raise ResultImportError(
            f"Capture Source could not be loaded as a supported MovieClip: {exc}"
        ) from exc


def _replace_camera_background(
    bpy: Any,
    camera_data: Any,
    clip: Any,
    contract: SourceViewContract,
) -> None:
    old_backgrounds = tuple(camera_data.background_images)
    old_clips = tuple(
        item.clip
        for item in old_backgrounds
        if item.source == "MOVIE_CLIP" and item.clip is not None
    )
    new_background = None
    try:
        new_background = _configure_background(camera_data, clip, contract)
        for background in old_backgrounds:
            camera_data.background_images.remove(background)
        for old_clip in old_clips:
            _remove_movieclip_if_unused(bpy, old_clip)
    except Exception:
        if (
            new_background is not None
            and new_background in camera_data.background_images[:]
        ):
            camera_data.background_images.remove(new_background)
        _remove_movieclip_if_unused(bpy, clip)
        raise


def set_scene_resolution_to_source(
    collection: Any,
    scene: Any,
    *,
    bpy_module: Any | None = None,
) -> str:
    """Explicitly revalidate media, set exact source resolution, and attach."""

    if bpy_module is None:
        import bpy as bpy_module  # type: ignore[import-not-found]
    bpy = bpy_module
    contract = imported_source_view_contract(collection)
    camera = find_reconstruction_camera(collection)
    clip, media = _load_imported_source_clip(bpy, collection, contract)
    render = scene.render
    previous = (
        int(render.resolution_x),
        int(render.resolution_y),
        int(render.resolution_percentage),
        float(render.pixel_aspect_x),
        float(render.pixel_aspect_y),
    )
    try:
        render.resolution_x = contract.width
        render.resolution_y = contract.height
        render.resolution_percentage = 100
        render.pixel_aspect_x = 1.0
        render.pixel_aspect_y = 1.0
        if not scene_aspect_matches(scene, contract):
            raise ResultImportError(
                "Scene rejected the exact source-display aspect"
            )
        _replace_camera_background(bpy, camera.data, clip, contract)
    except Exception:
        (
            render.resolution_x,
            render.resolution_y,
            render.resolution_percentage,
            render.pixel_aspect_x,
            render.pixel_aspect_y,
        ) = previous
        _remove_movieclip_if_unused(bpy, clip)
        raise
    collection["lingbot_map_source_background_status"] = "attached-hidden"
    collection["lingbot_map_source_background_path"] = str(media)
    camera.data["lingbot_map_source_background_status"] = "attached-hidden"
    camera.data["lingbot_map_source_background_path"] = str(media)
    return (
        f"Scene Resolution set to {contract.width} × {contract.height}; "
        "Source Background attached and hidden"
    )


def relink_source_background(
    collection: Any,
    scene: Any,
    path: str | Path,
    *,
    bpy_module: Any | None = None,
) -> str:
    """Accept a replacement path only after checksum, media, and mapping gates."""

    if bpy_module is None:
        import bpy as bpy_module  # type: ignore[import-not-found]
    bpy = bpy_module
    contract = imported_source_view_contract(collection)
    camera = find_reconstruction_camera(collection)
    clip, media = _load_imported_source_clip(
        bpy, collection, contract, relink_path=path
    )
    if scene_aspect_matches(scene, contract):
        _replace_camera_background(bpy, camera.data, clip, contract)
        status = "attached-hidden"
        message = "Capture Source relinked; background attached and hidden"
    else:
        _remove_movieclip_if_unused(bpy, clip)
        status = "verified-unattached-scene-aspect-mismatch"
        message = (
            "Capture Source checksum verified; use Set Scene Resolution to "
            "Source before attachment"
        )
    collection["lingbot_map_relinked_source_path"] = str(media)
    collection["lingbot_map_source_background_status"] = status
    collection["lingbot_map_source_background_path"] = str(media)
    camera.data["lingbot_map_source_background_status"] = status
    camera.data["lingbot_map_source_background_path"] = str(media)
    return message


def set_source_background_visibility(
    collection: Any, visible: bool
) -> str:
    camera = find_reconstruction_camera(collection)
    backgrounds = tuple(camera.data.background_images)
    if len(backgrounds) != 1 or backgrounds[0].source != "MOVIE_CLIP":
        raise ResultImportError(
            "Source Background is not attached; revalidate resolution or relink"
        )
    camera.data.show_background_images = True
    backgrounds[0].show_background_image = bool(visible)
    return "Source Background shown" if visible else "Source Background hidden"


_coverage_guides: dict[int, tuple[Any, Any]] = {}


def _draw_coverage_guide(
    area: Any,
    region: Any,
    scene: Any,
    camera: Any,
    contract: SourceViewContract,
) -> None:
    try:
        space = area.spaces.active
        region_3d = space.region_3d
        if (
            area.type != "VIEW_3D"
            or region_3d.view_perspective != "CAMERA"
            or scene.camera != camera
        ):
            return
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        from gpu_extras.batch import batch_for_shader
        import gpu

        projected = []
        for corner in camera.data.view_frame(scene=scene):
            point = location_3d_to_region_2d(
                region, region_3d, camera.matrix_world @ corner
            )
            if point is None:
                return
            projected.append((float(point.x), float(point.y)))
        border = (
            min(point[0] for point in projected),
            min(point[1] for point in projected),
            max(point[0] for point in projected),
            max(point[1] for point in projected),
        )
        polygon = coverage_to_camera_border(contract, border)
        vertices = tuple(
            polygon[index]
            for edge in ((0, 1), (1, 2), (2, 3), (3, 0))
            for index in edge
        )
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        batch = batch_for_shader(shader, "LINES", {"pos": vertices})
        gpu.state.blend_set("ALPHA")
        gpu.state.line_width_set(2.0)
        shader.bind()
        shader.uniform_float("color", (1.0, 0.35, 0.05, 0.95))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
    except (ReferenceError, RuntimeError, SourceViewError):
        return


def toggle_model_coverage_guide(
    context: Any, collection: Any
) -> bool:
    """Toggle one temporary POST_PIXEL guide; no Result/datablock is mutated."""

    area = getattr(context, "area", None)
    region = getattr(context, "region", None)
    if (
        area is None
        or region is None
        or area.type != "VIEW_3D"
        or region.type != "WINDOW"
    ):
        raise ResultImportError(
            "Model Coverage guide requires a 3D View window region"
        )
    key = int(area.as_pointer())
    existing = _coverage_guides.pop(key, None)
    if existing is not None:
        import bpy

        bpy.types.SpaceView3D.draw_handler_remove(existing[0], "WINDOW")
        area.tag_redraw()
        return False
    contract = imported_source_view_contract(collection)
    camera = find_reconstruction_camera(collection)
    import bpy

    handler = bpy.types.SpaceView3D.draw_handler_add(
        _draw_coverage_guide,
        (area, region, context.scene, camera, contract),
        "WINDOW",
        "POST_PIXEL",
    )
    _coverage_guides[key] = (handler, area)
    area.tag_redraw()
    return True


def clear_model_coverage_guides() -> None:
    if not _coverage_guides:
        return
    import bpy

    while _coverage_guides:
        _key, (handler, area) = _coverage_guides.popitem()
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")
            area.tag_redraw()
        except (ReferenceError, RuntimeError):
            continue


def _result_reference_candidates(
    collection: Any, current_blend_path: str | Path
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    relative = collection.get("lingbot_map_result_reference_relative")
    if isinstance(relative, str) and relative.startswith("//"):
        try:
            blend = normalized_blend_path(current_blend_path)
            candidates.append(
                Path(
                    os.path.abspath(
                        blend.parent
                        / Path(relative[2:].replace("/", os.sep))
                    )
                )
            )
        except (OSError, ValueError, RuntimeError):
            pass
    absolute = collection.get("lingbot_map_result_reference_absolute")
    if isinstance(absolute, str) and absolute:
        candidates.append(Path(os.path.abspath(absolute)))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def result_reference_status(
    collection: Any, current_blend_path: str | Path
) -> tuple[str, str | None]:
    """Resolve relative then absolute identity without mutating imported data."""

    result_id = collection.get("lingbot_map_result_id")
    manifest_sha256 = collection.get("lingbot_map_manifest_sha256")
    mismatch = False
    for candidate in _result_reference_candidates(
        collection, current_blend_path
    ):
        try:
            ready = read_ready_result(candidate)
            digest, _length = _stable_sha256(
                candidate / "manifest.json",
                cancel=None,
                phase="reference:manifest",
            )
        except (IpcError, OSError, ResultImportError):
            continue
        if ready.result_id == result_id and digest == manifest_sha256:
            return "available", str(candidate)
        mismatch = True
    return (
        ("identity-mismatch", None)
        if mismatch
        else ("unavailable", None)
    )


def relink_result_reference(
    collection: Any,
    scene: Any,
    directory: str | Path,
    current_blend_path: str | Path,
) -> str:
    """Relink only after complete validation of immutable Result identity."""

    if _owner_identity(collection) is None:
        raise ResultImportError(
            "Ownership Unknown: recorded Result identity cannot be trusted"
        )
    document = validate_result(directory)
    if (
        document.ready.result_id != collection.get("lingbot_map_result_id")
        or document.manifest_sha256
        != collection.get("lingbot_map_manifest_sha256")
    ):
        raise ResultImportError(
            "Relink Result ID or manifest checksum does not match"
        )
    if (
        str(scene.get("lingbot_map_scene_uuid", ""))
        != collection.get("lingbot_map_actual_scene_uuid")
    ):
        raise ResultImportError(
            "Ownership Unknown: actual Scene binding is contradictory"
        )
    values = _reference_metadata(document, current_blend_path)
    _mark(collection, values)
    return (
        "Result reference relinked by exact Result ID and manifest checksum"
    )


def _managed_claims(scene: Any, result_id: str) -> tuple[Any, ...]:
    return tuple(
        collection
        for collection in _scene_collections(scene)
        if collection.get("lingbot_map_result_id") == result_id
        and collection.get("lingbot_map_owner_schema") is not None
    )


def _clear_lingbot_identity(datablock: Any) -> None:
    for name in tuple(datablock.keys()):
        if str(name).startswith("lingbot_map_"):
            del datablock[name]


def _identity_matches(datablock: Any, identity: Mapping[str, str]) -> bool:
    return datablock is not None and all(
        datablock.get(name) == value for name, value in identity.items()
    )


def _exclusive_detach_graph(
    collection: Any, identity: Mapping[str, str]
) -> tuple[Any, ...]:
    """Find only identity-bearing data not shared outside this Collection."""

    found: list[Any] = []

    def append_if_owned(datablock: Any) -> bool:
        if _identity_matches(datablock, identity):
            found.append(datablock)
            return True
        return False

    def append_action(owner: Any) -> None:
        action = _owned_action(owner)
        if (
            action is not None
            and int(getattr(action, "users", 0)) <= 1
        ):
            append_if_owned(action)

    for item in tuple(collection.objects):
        if len(tuple(getattr(item, "users_collection", ()))) != 1:
            continue
        if not append_if_owned(item):
            continue
        append_action(item)
        data = getattr(item, "data", None)
        if data is None or int(getattr(data, "users", 0)) > 1:
            continue
        if not append_if_owned(data):
            continue
        append_action(data)
        for material in tuple(getattr(data, "materials", ())):
            if int(getattr(material, "users", 0)) <= 1:
                append_if_owned(material)
        for modifier in tuple(getattr(item, "modifiers", ())):
            node_group = getattr(modifier, "node_group", None)
            if (
                node_group is not None
                and int(getattr(node_group, "users", 0)) <= 1
            ):
                append_if_owned(node_group)
        for background in tuple(
            getattr(data, "background_images", ())
        ):
            clip = getattr(background, "clip", None)
            if (
                clip is not None
                and int(getattr(clip, "users", 0)) <= 1
            ):
                append_if_owned(clip)
    return tuple(found)


def _detach_collection_copy(
    collection: Any, identity: Mapping[str, str] | None
) -> None:
    exclusive = (
        _exclusive_detach_graph(collection, identity)
        if identity is not None
        else ()
    )
    _clear_lingbot_identity(collection)
    for datablock in exclusive:
        _clear_lingbot_identity(datablock)


def detach_collection_copy(collection: Any) -> None:
    """Detach one Collection non-destructively, preserving shared datablocks."""

    _detach_collection_copy(collection, _owner_identity(collection))


def resolve_duplicate_imports(
    scene: Any, result_id: str, keeper: Any
) -> int:
    claims = _managed_claims(scene, result_id)
    if len(claims) < 2 or sum(item is keeper for item in claims) != 1:
        raise ResultImportError(
            "Choose exactly one duplicate imported Collection as keeper"
        )
    if inspect_collection_ownership(keeper, scene).status != "managed":
        raise ResultImportError(
            "The selected keeper does not have complete managed ownership"
        )
    keeper_identity = _owner_identity(keeper)
    if keeper_identity is None:  # Guard the invariant across future changes.
        raise ResultImportError(
            "The selected keeper does not have complete managed ownership"
        )
    detached = 0
    for collection in claims:
        if collection is not keeper:
            _detach_collection_copy(collection, keeper_identity)
            detached += 1
    return detached


def removal_inventory(
    collection: Any, scene: Any
) -> RemovalInventory:
    inspection = inspect_collection_ownership(collection, scene)
    if inspection.status != "managed":
        raise ResultImportError(inspection.message)
    result_id = str(collection.get("lingbot_map_result_id", ""))
    if len(_managed_claims(scene, result_id)) != 1:
        raise ResultImportError(
            "Duplicate Imported Identity: resolve copies before Remove Version"
        )
    point_object = next(
        item
        for item in collection.objects
        if item.get("lingbot_map_kind") == "point_cloud"
    )
    points = getattr(getattr(point_object, "data", None), "points", ())
    shared = sum(
        int(getattr(datablock, "users", 0)) > 1
        for datablock in inspection.datablocks
        if datablock is not collection
    )
    return RemovalInventory(
        str(getattr(collection, "name", "")),
        len(tuple(collection.objects)),
        len(points),
        shared,
    )


def _unlink_collection_from_scene(scene: Any, target: Any) -> bool:
    parents = [scene.collection]
    changed = False
    while parents:
        parent = parents.pop()
        for child in tuple(parent.children):
            if child is target:
                parent.children.unlink(target)
                changed = True
            else:
                parents.append(child)
    return changed


def remove_managed_version(
    collection: Any,
    scene: Any,
    *,
    bpy_module: Any | None = None,
) -> RemovalInventory:
    """Remove only uniquely owned Blender data; disk Result is untouched."""

    inventory = removal_inventory(collection, scene)
    inspection = inspect_collection_ownership(collection, scene)
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore[import-not-found]
    bpy = bpy_module
    owned_objects = tuple(
        item
        for item in collection.objects
        if item in inspection.datablocks
    )
    unowned_objects = tuple(
        item for item in collection.objects if item not in owned_objects
    )
    for item in unowned_objects:
        if len(tuple(getattr(item, "users_collection", ()))) == 1:
            scene.collection.objects.link(item)
    for child in tuple(collection.children):
        if child not in scene.collection.children[:]:
            scene.collection.children.link(child)
    _unlink_collection_from_scene(scene, collection)
    if int(getattr(collection, "users", 0)) > 0:
        return inventory

    candidates = tuple(
        datablock
        for datablock in inspection.datablocks
        if datablock is not collection and datablock not in owned_objects
    )
    for item in owned_objects:
        if not tuple(getattr(item, "users_collection", ())):
            bpy.data.objects.remove(item, do_unlink=True)
    bpy.data.collections.remove(collection)
    removable = tuple(
        datablock
        for datablock in candidates
        if int(getattr(datablock, "users", 0)) == 0
    )
    if removable:
        bpy.data.batch_remove(ids=set(removable))
    return inventory


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
    "OwnershipInspection",
    "RemovalInventory",
    "ResultImportCancelled",
    "ResultImportError",
    "ValidatedResult",
    "attempt_auto_import_once",
    "available_physical_memory",
    "evaluate_import_capacity",
    "clear_model_coverage_guides",
    "detach_collection_copy",
    "effective_disk_authority",
    "find_reconstruction_camera",
    "get_import_status",
    "import_result",
    "imported_source_view_contract",
    "inspect_collection_ownership",
    "relink_result_reference",
    "relink_source_background",
    "removal_inventory",
    "remove_managed_version",
    "resolve_duplicate_imports",
    "result_reference_status",
    "set_import_status",
    "set_scene_resolution_to_source",
    "set_source_background_visibility",
    "toggle_model_coverage_guide",
    "validate_result",
]
