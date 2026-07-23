"""Read-only discovery of atomically published Reconstruction Results."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ReadyResult:
    result_id: str
    job_id: str
    created_utc: str
    directory: Path
    point_count: int
    frame_count: int
    profile_name: str


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
    require_exact_object(
        manifest,
        {
            "schema_version", "result_id", "job_id", "created_utc", "target_scene",
            "timeline_start", "source", "contracts", "profile", "coordinate_system",
            "counts", "arrays", "confidence_statistics", "warnings", "provenance", "logs",
        },
        label="Result manifest",
    )
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
    if not isinstance(arrays, dict) or set(arrays) != CORE_ARRAYS:
        raise IpcError("Result array set is incomplete or unknown")
    for name, value in arrays.items():
        descriptor = require_exact_object(
            value, {"path", "dtype", "shape", "byte_length", "sha256"}, label=f"arrays.{name}"
        )
        if descriptor["path"] != f"arrays/{name}.npy":
            raise IpcError("Result array path is not canonical")
        _ordinary_relative_file(directory, descriptor["path"])
    profile = require_exact_object(
        manifest["profile"],
        {"name", "confidence_cutoff_percent", "depth_cutoff_percent", "import_point_budget"},
        label="profile",
    )
    if not isinstance(manifest["logs"], list):
        raise IpcError("Result logs must be an array")
    for log in manifest["logs"]:
        descriptor = require_exact_object(log, {"path", "byte_length", "sha256"}, label="log")
        _ordinary_relative_file(directory, descriptor["path"])
    return ReadyResult(
        result_id,
        job_id,
        created,
        directory,
        counts["points"],
        counts["frames"],
        require_text(profile["name"], label="profile.name", maximum=128),
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
