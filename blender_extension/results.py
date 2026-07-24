"""Read-only discovery of atomically published Reconstruction Results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

from .ipc import IpcError, read_json, require_exact_object, require_text
from .job_lifecycle import is_reparse_point, project_result_root


RESULT_SCHEMA_VERSION = "1.0.0"
RESULT_DIRECTORY = re.compile(r"[0-9]{14}Z-[0-9a-f]{8}\Z")
RESULT_ID = re.compile(r"result-[0-9a-f]{32}\Z")
JOB_ID = re.compile(r"job-[0-9a-f]{32}\Z")
CORE_ARRAYS = {
    "positions", "colors", "confidence", "radius", "source_frame",
    "camera_to_world", "model_intrinsics", "source_intrinsics",
    "model_fov_radians", "source_pts_seconds", "source_to_model", "frame_type",
}
OPTIONAL_ARRAYS = {"sky_fraction"}


@dataclass(frozen=True)
class ReadyResult:
    result_id: str
    job_id: str
    created_utc: str
    directory: Path
    point_count: int
    frame_count: int
    profile_name: str
    dense_status: str = "not-retained"
    alignment_boundary_count: int = 0
    quality_warning_count: int = 0
    worst_boundary: str | None = None
    sky_masked: bool = False
    sky_count_above_95_percent: int = 0
    sky_cache_status: str | None = None


def _ordinary_relative_file(root: Path, value: Any) -> Path:
    text = require_text(value, label="Result file path", maximum=32767)
    posix, windows = PurePosixPath(text), PureWindowsPath(text)
    if (
        "\\" in text
        or text in {".", ".."}
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or text.startswith("//")
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise IpcError("Result file path is unsafe")
    path = root.joinpath(*posix.parts)
    if not path.is_file() or path.is_symlink() or is_reparse_point(path):
        raise IpcError("Result file is absent or not ordinary")
    current = path.parent
    while current != root:
        if current.is_symlink() or is_reparse_point(current):
            raise IpcError("Result file resolves through linked content")
        current = current.parent
    return path


def _dense_status(directory: Path, raw: Any) -> str:
    if raw is None:
        return "not-retained"
    try:
        descriptor = require_exact_object(
            raw,
            {
                "schema_version", "completion_state", "manifest_path",
                "manifest_byte_length", "manifest_sha256",
            },
            label="Dense Predictions descriptor",
        )
        version = require_text(
            descriptor["schema_version"], label="Dense Predictions schema", maximum=64
        )
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
            return "unavailable"
        if version != "1.0.0":
            return "incompatible"
        if (
            descriptor["completion_state"] != "complete"
            or descriptor["manifest_path"] != "dense/manifest.json"
            or isinstance(descriptor["manifest_byte_length"], bool)
            or not isinstance(descriptor["manifest_byte_length"], int)
            or not 1 <= descriptor["manifest_byte_length"] <= 16 * 1024 * 1024
            or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor["manifest_sha256"]))
        ):
            return "unavailable"
        path = _ordinary_relative_file(directory, descriptor["manifest_path"])
        before = path.stat()
        if before.st_size != descriptor["manifest_byte_length"]:
            return "unavailable"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or digest.hexdigest() != descriptor["manifest_sha256"]
        ):
            return "unavailable"
        manifest = read_json(path)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != version
            or manifest.get("component") != "dense_predictions"
            or manifest.get("completion_state") != "complete"
        ):
            return "unavailable"
        return "available"
    except (IpcError, OSError, ValueError):
        return "unavailable"


def _alignment_status(raw: Any) -> tuple[int, int, str | None]:
    if raw is None:
        return 0, 0, None
    alignment = require_exact_object(
        raw,
        {
            "schema_version", "rule_version", "strategy", "window_frames", "overlap_keyframes",
            "scale_frames", "keyframe_interval", "loop_closure", "pose_graph",
            "bundle_adjustment", "global_optimization", "boundaries",
        },
        label="window_alignment",
    )
    if (
        alignment["schema_version"] != "1.0.0"
        or alignment["rule_version"] != "1.0.0"
        or alignment["strategy"] != "rolling-similarity"
        or tuple(alignment[name] for name in (
            "window_frames", "overlap_keyframes", "scale_frames", "keyframe_interval"
        )) != (64, 16, 8, 1)
        or any(alignment[name] is not False for name in (
            "loop_closure", "pose_graph", "bundle_adjustment", "global_optimization"
        ))
        or not isinstance(alignment["boundaries"], list)
    ):
        raise IpcError("window alignment contract is invalid")
    warning_boundaries = []
    for boundary in alignment["boundaries"]:
        if not isinstance(boundary, dict) or not isinstance(
            boundary.get("triggered_conditions"), list
        ):
            raise IpcError("window alignment boundary is invalid")
        if boundary["triggered_conditions"]:
            warning_boundaries.append(boundary)
    worst = None
    if warning_boundaries:
        boundary = max(
            warning_boundaries,
            key=lambda item: (
                len(item["triggered_conditions"]),
                float(item.get("source_frame_end", -1)),
            ),
        )
        worst = (
            f"frames {boundary['source_frame_start']}-{boundary['source_frame_end']}: "
            + ", ".join(boundary["triggered_conditions"])
        )
    return len(alignment["boundaries"]), len(warning_boundaries), worst


def read_ready_result(directory: Path, *, scene_uuid: str | None = None) -> ReadyResult:
    directory = Path(os.path.abspath(directory))
    if (
        not RESULT_DIRECTORY.fullmatch(directory.name)
        or not directory.is_dir()
        or directory.is_symlink()
        or is_reparse_point(directory)
    ):
        raise IpcError("Result directory is not an ordinary canonical publication")
    manifest = read_json(directory / "manifest.json")
    required_fields = {
            "schema_version", "result_id", "job_id", "created_utc", "target_scene",
            "timeline_start", "source", "contracts", "profile", "coordinate_system",
            "counts", "arrays", "confidence_statistics", "warnings", "provenance", "logs",
        }
    if (
        not isinstance(manifest, dict)
        or not required_fields.issubset(manifest)
        or not set(manifest).issubset(
            required_fields
            | {
                "dense_predictions",
                "window_alignment",
                "sky_statistics",
                "source_display",
                "model_coverage",
            }
        )
    ):
        raise IpcError("Result manifest has unknown or missing fields")
    if manifest["schema_version"] != RESULT_SCHEMA_VERSION:
        raise IpcError("Result schema version is unsupported")
    result_id = require_text(manifest["result_id"], label="result_id", maximum=64)
    job_id = require_text(manifest["job_id"], label="job_id", maximum=64)
    if not RESULT_ID.fullmatch(result_id) or not JOB_ID.fullmatch(job_id):
        raise IpcError("Result identity is invalid")
    created = require_text(manifest["created_utc"], label="created_utc", maximum=128)
    target = require_exact_object(
        manifest["target_scene"], {"blend_path", "scene_uuid", "scene_name"}, label="target_scene"
    )
    result_scene_uuid = require_text(target["scene_uuid"], label="scene_uuid", maximum=64)
    if scene_uuid is not None and result_scene_uuid != scene_uuid:
        raise IpcError("Result belongs to a different Target Scene")
    counts = require_exact_object(manifest["counts"], {"frames", "points"}, label="counts")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts.values()):
        raise IpcError("Result counts are invalid")
    arrays = manifest["arrays"]
    if (
        not isinstance(arrays, dict)
        or not CORE_ARRAYS.issubset(arrays)
        or not set(arrays).issubset(CORE_ARRAYS | OPTIONAL_ARRAYS)
    ):
        raise IpcError("Result array set is incomplete or unknown")
    for name, value in arrays.items():
        descriptor = require_exact_object(
            value, {"path", "dtype", "shape", "byte_length", "sha256"}, label=f"arrays.{name}"
        )
        if descriptor["path"] != f"arrays/{name}.npy":
            raise IpcError("Result array path is not canonical")
        _ordinary_relative_file(directory, descriptor["path"])
    sky_masked = "sky_fraction" in arrays
    sky_count = 0
    sky_cache_status = None
    if sky_masked:
        sky_descriptor = arrays["sky_fraction"]
        if (
            sky_descriptor["dtype"] != "<f4"
            or sky_descriptor["shape"] != [counts["frames"]]
        ):
            raise IpcError("Sky fraction descriptor is invalid")
        sky_statistics = require_exact_object(
            manifest.get("sky_statistics"),
            {"minimum", "median", "p95", "maximum", "count_above_95_percent"},
            label="sky_statistics",
        )
        for name in ("minimum", "median", "p95", "maximum"):
            value = sky_statistics[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
            ):
                raise IpcError("Sky fraction statistics are invalid")
        sky_count = sky_statistics["count_above_95_percent"]
        if (
            isinstance(sky_count, bool)
            or not isinstance(sky_count, int)
            or not 0 <= sky_count <= counts["frames"]
        ):
            raise IpcError("Sky fraction warning count is invalid")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise IpcError("Result provenance is invalid")
        sky_provenance = require_exact_object(
            provenance.get("sky_masking"),
            {
                "enabled", "model_id", "model_sha256", "rule_version",
                "preprocessing_version", "provider", "onnxruntime_version",
                "batch_size", "onnx_threads", "cache_key", "cache_status",
            },
            label="sky_masking provenance",
        )
        if (
            sky_provenance["enabled"] is not True
            or sky_provenance["provider"] != "CPUExecutionProvider"
            or sky_provenance["batch_size"] != 1
        ):
            raise IpcError("Sky Mask provenance is invalid")
        sky_cache_status = require_text(
            sky_provenance["cache_status"],
            label="sky_masking.cache_status",
            maximum=64,
        )
    elif manifest.get("sky_statistics") is not None:
        raise IpcError("Sky statistics require a sky_fraction array")
    profile = manifest["profile"]
    profile_fields = {
        "name", "confidence_cutoff_percent", "depth_cutoff_percent", "import_point_budget"
    }
    if (
        not isinstance(profile, dict)
        or not profile_fields.issubset(profile)
        or not set(profile).issubset(profile_fields | {"retain_dense_predictions"})
    ):
        raise IpcError("Result profile has unknown or missing fields")
    if not isinstance(manifest["logs"], list):
        raise IpcError("Result logs must be an array")
    for log in manifest["logs"]:
        descriptor = require_exact_object(log, {"path", "byte_length", "sha256"}, label="log")
        _ordinary_relative_file(directory, descriptor["path"])
    boundary_count, quality_warning_count, worst_boundary = _alignment_status(
        manifest.get("window_alignment")
    )
    return ReadyResult(
        result_id,
        job_id,
        created,
        directory,
        counts["points"],
        counts["frames"],
        require_text(profile["name"], label="profile.name", maximum=128),
        _dense_status(directory, manifest.get("dense_predictions")),
        boundary_count,
        quality_warning_count,
        worst_boundary,
        sky_masked,
        sky_count,
        sky_cache_status,
    )


def discover_ready_results(
    blend_path: str | Path, *, scene_uuid: str | None = None
) -> tuple[ReadyResult, ...]:
    root = project_result_root(blend_path) / "results"
    if not root.is_dir() or root.is_symlink() or is_reparse_point(root):
        return ()
    ready: list[ReadyResult] = []
    for item in root.iterdir():
        if not RESULT_DIRECTORY.fullmatch(item.name):
            continue
        try:
            ready.append(read_ready_result(item, scene_uuid=scene_uuid))
        except (IpcError, OSError):
            continue
    return tuple(sorted(ready, key=lambda item: (item.created_utc, item.result_id), reverse=True))
