"""Safe NPY contracts and atomic Reconstruction Result publication."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import struct
from typing import Any, Callable, Mapping
import uuid

import numpy as np

from .ipc import SCHEMA_VERSION, atomic_write_json, read_json, require_exact_object, require_text
from .model_store_compat import is_reparse_point


RESULT_SCHEMA_VERSION = "1.0.0"
MAX_NPY_HEADER_BYTES = 64 * 1024
RESULT_ID = re.compile(r"result-[0-9a-f]{32}\Z")
JOB_ID = re.compile(r"job-[0-9a-f]{32}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CancelCheck = Callable[[], bool]
DiskCheck = Callable[[int], None]


class ResultBundleError(RuntimeError):
    pass


class ResultCancelled(ResultBundleError):
    pass


@dataclass(frozen=True)
class ArrayContract:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class PublishedResult:
    result_id: str
    directory: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ResultPublication:
    job_id: str
    project_root: Path
    target_scene: Mapping[str, Any]
    timeline_start: int
    source: Mapping[str, Any]
    profile: Mapping[str, Any]
    provenance: Mapping[str, Any]
    warnings: tuple[Mapping[str, str], ...]
    arrays: Mapping[str, np.ndarray]
    voxel_edge_length: float
    voxel_origin: tuple[float, float, float]
    created_utc: str | None = None
    result_id: str | None = None


CORE_ARRAY_DTYPES = {
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
}


def is_plain_path(path: Path) -> bool:
    return path.exists() and not path.is_symlink() and not is_reparse_point(path)


def safe_relative_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or value in {".", ".."} or "\\" in value or "\x00" in value:
        raise ResultBundleError("Result path must be a non-empty forward-slash relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or value.startswith("//")
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ResultBundleError("Result path is absolute, drive-qualified, UNC, or traversing")
    root = Path(os.path.abspath(root))
    candidate = root.joinpath(*posix.parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # Defensive on platforms with unusual path rules.
        raise ResultBundleError("Result path escaped its bundle root") from exc
    current = candidate
    while current != root:
        if current.exists() and (current.is_symlink() or is_reparse_point(current)):
            raise ResultBundleError("Result path resolves through linked or reparse content")
        current = current.parent
    return candidate


def _npy_header(path: Path) -> tuple[str, bool, tuple[int, ...], int]:
    if not path.is_file() or path.is_symlink() or is_reparse_point(path):
        raise ResultBundleError(f"NPY path is not an ordinary file: {path.name}")
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ResultBundleError(f"NPY magic is invalid: {path.name}")
        version = stream.read(2)
        if version == b"\x01\x00":
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ResultBundleError("NPY header length is truncated")
            header_length = struct.unpack("<H", length_bytes)[0]
        elif version in {b"\x02\x00", b"\x03\x00"}:
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise ResultBundleError("NPY header length is truncated")
            header_length = struct.unpack("<I", length_bytes)[0]
        else:
            raise ResultBundleError("NPY version is unsupported")
        if not 1 <= header_length <= MAX_NPY_HEADER_BYTES:
            raise ResultBundleError("NPY header exceeds 64 KiB")
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ResultBundleError("NPY header is truncated")
        try:
            document = ast.literal_eval(header.decode("latin1").strip())
        except (UnicodeError, SyntaxError, ValueError, MemoryError) as exc:
            raise ResultBundleError("NPY header dictionary is invalid") from exc
        if not isinstance(document, dict) or set(document) != {"descr", "fortran_order", "shape"}:
            raise ResultBundleError("NPY header has unknown or missing fields")
        dtype = document["descr"]
        fortran = document["fortran_order"]
        shape = document["shape"]
        if not isinstance(dtype, str) or not isinstance(fortran, bool):
            raise ResultBundleError("NPY dtype or order is invalid")
        if (
            not isinstance(shape, tuple)
            or len(shape) > 4
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape)
        ):
            raise ResultBundleError("NPY shape is invalid")
        payload_offset = stream.tell()
    return dtype, fortran, tuple(shape), payload_offset


def validate_npy_file(path: Path, contract: ArrayContract) -> np.memmap:
    dtype, fortran, shape, payload_offset = _npy_header(path)
    if dtype != contract.dtype or shape != contract.shape or fortran:
        raise ResultBundleError(
            f"NPY contract mismatch for {path.name}: {dtype} {shape} fortran={fortran}"
        )
    expected_payload = math.prod(shape) * np.dtype(contract.dtype).itemsize
    if path.stat().st_size != payload_offset + expected_payload:
        raise ResultBundleError(f"NPY byte length is inconsistent: {path.name}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, MemoryError) as exc:
        raise ResultBundleError(f"NPY data cannot be opened safely: {path.name}") from exc
    if array.dtype.str != contract.dtype or tuple(array.shape) != shape or not array.flags.c_contiguous:
        raise ResultBundleError(f"NPY loaded representation is inconsistent: {path.name}")
    return array


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    completed = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            completed += len(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or completed != before.st_size
    ):
        raise ResultBundleError(f"Result file changed while hashing: {path.name}")
    return digest.hexdigest()


def _descriptor(path: Path, relative: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": relative,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_length": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ResultBundleError(f"Result array already exists: {path.name}")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_array_input(name: str, array: np.ndarray) -> None:
    if name not in CORE_ARRAY_DTYPES:
        raise ResultBundleError(f"unknown core Result array: {name}")
    if not isinstance(array, np.ndarray):
        raise ResultBundleError(f"{name} must be an ndarray without implicit conversion")
    if array.dtype.str != CORE_ARRAY_DTYPES[name] or not array.flags.c_contiguous:
        raise ResultBundleError(f"{name} has an invalid dtype or memory order")
    if array.dtype.fields is not None or array.dtype.hasobject:
        raise ResultBundleError(f"{name} uses an object or structured dtype")


def _validate_semantics(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    if set(arrays) != set(CORE_ARRAY_DTYPES):
        raise ResultBundleError("core Result array set is incomplete or contains unknown names")
    for name, array in arrays.items():
        _validate_array_input(name, array)
    positions = arrays["positions"]
    count = positions.shape[0] if positions.ndim == 2 else -1
    if positions.shape != (count, 3):
        raise ResultBundleError("positions must have shape (N,3)")
    for name in ("colors",):
        if arrays[name].shape != (count, 3):
            raise ResultBundleError(f"{name} must have shape (N,3)")
    for name in ("confidence", "radius", "source_frame"):
        if arrays[name].shape != (count,):
            raise ResultBundleError(f"{name} must have shape (N,)")
    camera_to_world = arrays["camera_to_world"]
    frame_count = camera_to_world.shape[0] if camera_to_world.ndim == 3 else -1
    expected_frames = {
        "camera_to_world": (frame_count, 4, 4),
        "model_intrinsics": (frame_count, 3, 3),
        "source_intrinsics": (frame_count, 3, 3),
        "model_fov_radians": (frame_count, 2),
        "source_pts_seconds": (frame_count,),
        "frame_type": (frame_count,),
        "source_to_model": (3, 3),
    }
    for name, shape in expected_frames.items():
        if arrays[name].shape != shape:
            raise ResultBundleError(f"{name} must have shape {shape}")
    for name in (
        "positions", "confidence", "radius", "camera_to_world", "model_intrinsics",
        "source_intrinsics", "model_fov_radians", "source_pts_seconds", "source_to_model",
    ):
        if not bool(np.isfinite(arrays[name]).all()):
            raise ResultBundleError(f"{name} contains non-finite values")
    if not bool((arrays["radius"] > 0).all()):
        raise ResultBundleError("radius contains non-positive values")
    if count and not bool((arrays["source_frame"] < frame_count).all()):
        raise ResultBundleError("source_frame refers outside the camera arrays")
    if frame_count < 1:
        raise ResultBundleError("Result must contain at least one camera frame")
    if not bool(np.isin(arrays["frame_type"], (0, 1, 2)).all()):
        raise ResultBundleError("frame_type contains an unknown code")
    timestamps = arrays["source_pts_seconds"]
    if len(timestamps) > 1 and not bool((np.diff(timestamps) > 0).all()):
        raise ResultBundleError("source timestamps are not strictly increasing")
    fov = arrays["model_fov_radians"]
    if not bool(((fov > 0) & (fov < math.pi)).all()):
        raise ResultBundleError("model FOV is outside (0,pi)")
    for name in ("model_intrinsics", "source_intrinsics"):
        intrinsics = arrays[name].astype(np.float64, copy=False)
        if not bool((intrinsics[:, 0, 0] > 0).all() and (intrinsics[:, 1, 1] > 0).all()):
            raise ResultBundleError(f"{name} contains non-positive focal length")
        expected_last = np.broadcast_to((0.0, 0.0, 1.0), (frame_count, 3))
        if not np.allclose(intrinsics[:, 2, :], expected_last, atol=1e-6, rtol=0):
            raise ResultBundleError(f"{name} has a malformed homogeneous row")
        if not np.allclose(intrinsics[:, 0, 1], 0, atol=1e-6, rtol=0) or not np.allclose(intrinsics[:, 1, 0], 0, atol=1e-6, rtol=0):
            raise ResultBundleError(f"{name} contains unsupported skew")
    homogeneous = np.broadcast_to((0.0, 0.0, 0.0, 1.0), (frame_count, 4))
    if not np.allclose(camera_to_world[:, 3, :], homogeneous, atol=1e-5, rtol=0):
        raise ResultBundleError("camera_to_world has malformed homogeneous rows")
    rotations = camera_to_world[:, :3, :3].astype(np.float64)
    identities = np.einsum("fji,fjk->fik", rotations, rotations)
    if not np.allclose(identities, np.eye(3), atol=1e-4, rtol=0):
        raise ResultBundleError("camera_to_world rotation is not orthonormal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=1e-4, rtol=0):
        raise ResultBundleError("camera_to_world rotation is reflected or non-rigid")
    transform = arrays["source_to_model"]
    if abs(float(np.linalg.det(transform))) <= 1e-12:
        raise ResultBundleError("source_to_model is singular")
    if not np.allclose(transform[2], (0.0, 0.0, 1.0), atol=1e-9, rtol=0):
        raise ResultBundleError("source_to_model is not an affine pixel transform")
    mapped_source = np.einsum(
        "ij,fjk->fik",
        transform.astype(np.float64, copy=False),
        arrays["source_intrinsics"].astype(np.float64, copy=False),
    )
    if not np.allclose(
        mapped_source,
        arrays["model_intrinsics"].astype(np.float64, copy=False),
        atol=1e-4,
        rtol=1e-5,
    ):
        raise ResultBundleError("source/model intrinsics disagree with source_to_model")
    expected_first_camera = np.array(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    if not np.allclose(camera_to_world[0], expected_first_camera, atol=1e-4, rtol=0):
        raise ResultBundleError("first camera does not define the Reconstruction Frame")
    return count, frame_count


def _walk_plain_files(root: Path) -> set[str]:
    found: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink() or is_reparse_point(path):
                    raise ResultBundleError("Result contains linked or reparse content")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    found.add(path.relative_to(root).as_posix())
                else:
                    raise ResultBundleError("Result contains a non-file non-directory entry")
    return found


def validate_result_bundle(root: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    if not root.is_dir() or root.is_symlink() or is_reparse_point(root):
        raise ResultBundleError("Result root is not an ordinary directory")
    manifest = read_json(root / "manifest.json")
    fields = {
        "schema_version", "result_id", "job_id", "created_utc", "target_scene",
        "timeline_start", "source", "contracts", "profile", "coordinate_system",
        "counts", "arrays", "confidence_statistics", "warnings", "provenance", "logs",
    }
    require_exact_object(manifest, fields, label="Result manifest")
    if manifest["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultBundleError("unsupported Reconstruction Result schema version")
    if not RESULT_ID.fullmatch(str(manifest["result_id"])) or not JOB_ID.fullmatch(str(manifest["job_id"])):
        raise ResultBundleError("Result or Job identity is invalid")
    created_utc = require_text(manifest["created_utc"], label="created_utc", maximum=128)
    try:
        parsed_created = datetime.fromisoformat(created_utc)
    except ValueError as exc:
        raise ResultBundleError("created_utc is not an ISO-8601 timestamp") from exc
    if parsed_created.tzinfo is None or parsed_created.utcoffset() is None:
        raise ResultBundleError("created_utc must include an offset")
    if parsed_created.utcoffset().total_seconds() != 0:
        raise ResultBundleError("created_utc must be expressed in UTC")
    target = require_exact_object(
        manifest["target_scene"], {"blend_path", "scene_uuid", "scene_name"}, label="target_scene"
    )
    for name in target:
        require_text(target[name], label=f"target_scene.{name}", maximum=32767)
    if isinstance(manifest["timeline_start"], bool) or not isinstance(manifest["timeline_start"], int):
        raise ResultBundleError("timeline_start is invalid")
    source = require_exact_object(
        manifest["source"],
        {"absolute_path", "scene_relative_path", "size_bytes", "modification_time_ns", "sha256"},
        label="source",
    )
    if not Path(require_text(source["absolute_path"], label="source.absolute_path", maximum=32767)).is_absolute():
        raise ResultBundleError("source absolute path is invalid")
    relative_source = source["scene_relative_path"]
    if relative_source is not None:
        if (
            not isinstance(relative_source, str)
            or not relative_source.startswith("//")
            or len(relative_source.encode("utf-8")) > 32767
        ):
            raise ResultBundleError("source scene-relative path is invalid")
    for name in ("size_bytes", "modification_time_ns"):
        if isinstance(source[name], bool) or not isinstance(source[name], int) or source[name] < 1:
            raise ResultBundleError(f"source {name} is invalid")
    if not SHA256.fullmatch(str(source["sha256"])):
        raise ResultBundleError("source checksum is invalid")
    contracts = require_exact_object(
        manifest["contracts"], {"job_spec", "events", "result"}, label="contracts"
    )
    if contracts != {
        "job_spec": SCHEMA_VERSION,
        "events": SCHEMA_VERSION,
        "result": RESULT_SCHEMA_VERSION,
    }:
        raise ResultBundleError("manifest names unsupported contract versions")
    profile = require_exact_object(
        manifest["profile"],
        {"name", "confidence_cutoff_percent", "depth_cutoff_percent", "import_point_budget"},
        label="profile",
    )
    if (
        isinstance(profile["import_point_budget"], bool)
        or not isinstance(profile["import_point_budget"], int)
        or not 1 <= profile["import_point_budget"] <= 50_000_000
    ):
        raise ResultBundleError("profile point budget is invalid")
    require_text(profile["name"], label="profile.name", maximum=128)
    for name in ("confidence_cutoff_percent", "depth_cutoff_percent"):
        value = profile[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ResultBundleError(f"profile {name} is invalid")
    coordinate = require_exact_object(
        manifest["coordinate_system"],
        {"name", "handedness", "camera_local_axes", "voxel_origin", "voxel_edge_length"},
        label="coordinate_system",
    )
    if coordinate["name"] != "blender-z-up-reconstruction-frame" or coordinate["handedness"] != "right":
        raise ResultBundleError("coordinate system is unsupported")
    if coordinate["camera_local_axes"] != "+X right,+Y up,-Z forward":
        raise ResultBundleError("camera local-axis convention is invalid")
    edge = coordinate["voxel_edge_length"]
    if (
        coordinate["voxel_origin"] != [0.0, 0.0, 0.0]
        or isinstance(edge, bool)
        or not isinstance(edge, (int, float))
        or not math.isfinite(edge)
        or edge <= 0
    ):
        raise ResultBundleError("voxel grid provenance is invalid")
    counts = require_exact_object(manifest["counts"], {"frames", "points"}, label="counts")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts.values()):
        raise ResultBundleError("Result counts are invalid")
    descriptors = manifest["arrays"]
    if not isinstance(descriptors, dict) or set(descriptors) != set(CORE_ARRAY_DTYPES):
        raise ResultBundleError("Result array descriptors are incomplete or unknown")
    arrays: dict[str, np.ndarray] = {}
    declared = {"manifest.json"}
    for name, raw in descriptors.items():
        descriptor = require_exact_object(
            raw, {"path", "dtype", "shape", "byte_length", "sha256"}, label=f"arrays.{name}"
        )
        relative = str(descriptor["path"])
        if relative != f"arrays/{name}.npy":
            raise ResultBundleError(f"array path is not canonical: {name}")
        path = safe_relative_file(root, relative)
        raw_shape = descriptor["shape"]
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) > 4
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in raw_shape)
        ):
            raise ResultBundleError(f"array descriptor shape is invalid: {name}")
        if (
            isinstance(descriptor["byte_length"], bool)
            or not isinstance(descriptor["byte_length"], int)
            or descriptor["byte_length"] < 1
            or not SHA256.fullmatch(str(descriptor["sha256"]))
        ):
            raise ResultBundleError(f"array descriptor size or checksum is invalid: {name}")
        shape = tuple(raw_shape)
        contract = ArrayContract(str(descriptor["dtype"]), shape)
        if contract.dtype != CORE_ARRAY_DTYPES[name]:
            raise ResultBundleError(f"array descriptor dtype is invalid: {name}")
        array = validate_npy_file(path, contract)
        if path.stat().st_size != descriptor["byte_length"] or _sha256_file(path) != descriptor["sha256"]:
            raise ResultBundleError(f"array descriptor length or checksum is invalid: {name}")
        arrays[name] = array
        declared.add(relative)
    point_count, frame_count = _validate_semantics(arrays)
    if counts != {"frames": frame_count, "points": point_count}:
        raise ResultBundleError("manifest counts disagree with core arrays")
    if point_count and not np.allclose(
        arrays["radius"], float(edge) / 2.0, atol=1e-6, rtol=1e-6
    ):
        raise ResultBundleError("point radius disagrees with the retained voxel grid")
    statistics = require_exact_object(
        manifest["confidence_statistics"], {"p0", "p5", "p25", "p50", "p75", "p95", "p100"}, label="confidence_statistics"
    )
    if point_count:
        actual = np.percentile(arrays["confidence"], (0, 5, 25, 50, 75, 95, 100))
        recorded = np.array([statistics[name] for name in ("p0", "p5", "p25", "p50", "p75", "p95", "p100")], dtype=np.float64)
        if not np.allclose(actual, recorded, atol=1e-6, rtol=1e-6):
            raise ResultBundleError("confidence statistics disagree with retained values")
    elif any(value is not None for value in statistics.values()):
        raise ResultBundleError("empty point cloud must use null confidence statistics")
    warnings = manifest["warnings"]
    if not isinstance(warnings, list):
        raise ResultBundleError("warnings must be an array")
    for warning in warnings:
        record = require_exact_object(warning, {"code", "message"}, label="warning")
        require_text(record["code"], label="warning.code", maximum=128)
        require_text(record["message"], label="warning.message", maximum=16384)
    provenance = require_exact_object(
        manifest["provenance"],
        {"runtime_id", "worker_version", "job_spec_sha256", "model_sha256", "source_sha256", "preprocessing_rule_version", "filtering_rule_version", "point_reducer_rule_version", "resource_estimate_version"},
        label="provenance",
    )
    for name in ("runtime_id", "job_spec_sha256", "model_sha256", "source_sha256"):
        if not SHA256.fullmatch(str(provenance[name])):
            raise ResultBundleError(f"provenance checksum or identity is invalid: {name}")
    if provenance["source_sha256"] != source["sha256"]:
        raise ResultBundleError("source identity and provenance checksum disagree")
    require_text(provenance["worker_version"], label="worker_version", maximum=128)
    for name in (
        "preprocessing_rule_version", "filtering_rule_version",
        "point_reducer_rule_version", "resource_estimate_version",
    ):
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", require_text(provenance[name], label=name, maximum=64)):
            raise ResultBundleError(f"provenance rule version is invalid: {name}")
    logs = manifest["logs"]
    if not isinstance(logs, list):
        raise ResultBundleError("logs must be an array")
    for raw in logs:
        descriptor = require_exact_object(raw, {"path", "byte_length", "sha256"}, label="log")
        relative_log = require_text(descriptor["path"], label="log.path", maximum=32767)
        if not relative_log.startswith("logs/"):
            raise ResultBundleError("log path is not canonical")
        if (
            isinstance(descriptor["byte_length"], bool)
            or not isinstance(descriptor["byte_length"], int)
            or descriptor["byte_length"] < 0
            or not SHA256.fullmatch(str(descriptor["sha256"]))
        ):
            raise ResultBundleError("log descriptor size or checksum is invalid")
        path = safe_relative_file(root, relative_log)
        if not path.is_file() or path.is_symlink() or is_reparse_point(path):
            raise ResultBundleError("declared log is not an ordinary file")
        if path.stat().st_size != descriptor["byte_length"] or _sha256_file(path) != descriptor["sha256"]:
            raise ResultBundleError("log descriptor length or checksum is invalid")
        declared.add(relative_log)
    if _walk_plain_files(root) != declared:
        raise ResultBundleError("Result contains undeclared or missing files")
    return manifest


def publish_result_bundle(
    publication: ResultPublication,
    *,
    cancel: CancelCheck,
    disk_check: DiskCheck,
    estimated_total_bytes: int,
) -> PublishedResult:
    project_root = Path(os.path.abspath(publication.project_root))
    if publication.voxel_origin != (0.0, 0.0, 0.0):
        raise ResultBundleError("Point Reducer origin must be the fixed Reconstruction origin")
    results_root = project_root / "results"
    diagnostics_root = project_root / "diagnostics"
    for directory in (project_root, results_root, diagnostics_root):
        if not directory.is_dir() or directory.is_symlink() or is_reparse_point(directory):
            raise ResultBundleError(f"Project Result path is not an ordinary directory: {directory.name}")
    result_id = publication.result_id or f"result-{uuid.uuid4().hex}"
    if not RESULT_ID.fullmatch(result_id) or not JOB_ID.fullmatch(publication.job_id):
        raise ResultBundleError("Result or Job identity is invalid")
    created_utc = publication.created_utc or datetime.now(timezone.utc).isoformat()
    timestamp = re.sub(r"[^0-9]", "", created_utc)[:14]
    if len(timestamp) != 14:
        raise ResultBundleError("created_utc cannot form a safe Result directory name")
    final = results_root / f"{timestamp}Z-{publication.job_id[4:12]}"
    staging = results_root / f".staging-{result_id}-{uuid.uuid4().hex[:8]}"
    if final.exists() or staging.exists():
        raise ResultBundleError("Result publication destination already exists")
    staging.mkdir()
    committed = False
    remaining = int(estimated_total_bytes)
    try:
        descriptors: dict[str, dict[str, Any]] = {}
        for name in sorted(publication.arrays):
            if cancel():
                raise ResultCancelled("Result publication was cancelled before commit")
            array = publication.arrays[name]
            _validate_array_input(name, array)
            disk_check(max(0, remaining))
            relative = f"arrays/{name}.npy"
            path = safe_relative_file(staging, relative)
            _write_npy(path, array)
            validate_npy_file(path, ArrayContract(array.dtype.str, tuple(array.shape)))
            descriptors[name] = _descriptor(path, relative, array)
            remaining = max(0, remaining - path.stat().st_size)
        _validate_semantics(publication.arrays)
        log_path = safe_relative_file(staging, "logs/worker.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"")
        log_descriptor = {
            "path": "logs/worker.log", "byte_length": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        confidence = publication.arrays["confidence"]
        if len(confidence):
            percentiles = np.percentile(confidence, (0, 5, 25, 50, 75, 95, 100))
            statistics = {
                name: float(value)
                for name, value in zip(("p0", "p5", "p25", "p50", "p75", "p95", "p100"), percentiles)
            }
        else:
            statistics = {name: None for name in ("p0", "p5", "p25", "p50", "p75", "p95", "p100")}
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "result_id": result_id,
            "job_id": publication.job_id,
            "created_utc": created_utc,
            "target_scene": dict(publication.target_scene),
            "timeline_start": int(publication.timeline_start),
            "source": dict(publication.source),
            "contracts": {"job_spec": SCHEMA_VERSION, "events": SCHEMA_VERSION, "result": RESULT_SCHEMA_VERSION},
            "profile": dict(publication.profile),
            "coordinate_system": {
                "name": "blender-z-up-reconstruction-frame",
                "handedness": "right",
                "camera_local_axes": "+X right,+Y up,-Z forward",
                "voxel_origin": list(publication.voxel_origin),
                "voxel_edge_length": float(publication.voxel_edge_length),
            },
            "counts": {
                "frames": int(publication.arrays["camera_to_world"].shape[0]),
                "points": int(publication.arrays["positions"].shape[0]),
            },
            "arrays": descriptors,
            "confidence_statistics": statistics,
            "warnings": [dict(item) for item in publication.warnings],
            "provenance": dict(publication.provenance),
            "logs": [log_descriptor],
        }
        if cancel():
            raise ResultCancelled("Result publication was cancelled before commit")
        disk_check(max(0, remaining))
        atomic_write_json(staging / "manifest.json", manifest)
        validate_result_bundle(staging)
        if cancel():
            raise ResultCancelled("Result publication was cancelled before commit")
        os.replace(staging, final)
        committed = True
        return PublishedResult(result_id, final, manifest)
    except Exception as exc:
        if not committed and staging.exists():
            reason = "cancelled" if isinstance(exc, ResultCancelled) else "failed"
            destination = diagnostics_root / f"{publication.job_id}--result-{reason}"
            if destination.exists():
                destination = diagnostics_root / f"{publication.job_id}--result-{reason}-{uuid.uuid4().hex[:8]}"
            os.replace(staging, destination)
            atomic_write_json(
                destination / "failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": publication.job_id,
                    "reason": reason,
                    "error": f"{type(exc).__name__}: {exc}"[:16384],
                },
            )
        raise
