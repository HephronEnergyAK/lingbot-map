"""Bounded, independently versioned Dense Predictions component."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import struct
from typing import Any, Callable, Mapping
import uuid

import numpy as np

from .ipc import atomic_write_json, read_json, require_exact_object, require_text
from .model_store_compat import is_reparse_point


DENSE_SCHEMA_VERSION = "1.0.0"
DENSE_CHUNK_FRAMES = 64
MAX_NPY_HEADER_BYTES = 64 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DiskCheck = Callable[[int], None]


class DensePredictionsError(RuntimeError):
    pass


class DensePredictionsIncompatible(DensePredictionsError):
    pass


@dataclass(frozen=True)
class DenseComponentArtifact:
    staging_directory: Path
    descriptor: Mapping[str, Any]


def estimate_dense_prediction_bytes(frame_count: int, grid_pixels: int) -> int:
    if frame_count < 1 or grid_pixels < 1:
        raise DensePredictionsError("dense estimate dimensions must be positive")
    chunks = math.ceil(frame_count / DENSE_CHUNK_FRAMES)
    payload = frame_count * grid_pixels * np.dtype("<f4").itemsize * 2
    return payload + chunks * 2 * MAX_NPY_HEADER_BYTES + 1024 * 1024


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
        raise DensePredictionsError("Dense Predictions file changed while hashing")
    return digest.hexdigest()


def _safe_relative_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DensePredictionsError("Dense Predictions path is invalid")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or value.startswith("//")
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise DensePredictionsError("Dense Predictions path escaped its component")
    root = Path(os.path.abspath(root))
    path = root.joinpath(*posix.parts)
    current = path
    while current != root:
        if current.exists() and (current.is_symlink() or is_reparse_point(current)):
            raise DensePredictionsError("Dense Predictions path uses linked content")
        current = current.parent
    return path


def _npy_contract(path: Path) -> tuple[str, bool, tuple[int, ...], int]:
    if not path.is_file() or path.is_symlink() or is_reparse_point(path):
        raise DensePredictionsError("Dense Predictions chunk is not an ordinary file")
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise DensePredictionsError("Dense Predictions chunk has invalid NPY magic")
        version = stream.read(2)
        if version == b"\x01\x00":
            raw_length = stream.read(2)
            if len(raw_length) != 2:
                raise DensePredictionsError("Dense Predictions NPY header is truncated")
            header_length = struct.unpack("<H", raw_length)[0]
        elif version in {b"\x02\x00", b"\x03\x00"}:
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                raise DensePredictionsError("Dense Predictions NPY header is truncated")
            header_length = struct.unpack("<I", raw_length)[0]
        else:
            raise DensePredictionsError("Dense Predictions NPY version is unsupported")
        if not 1 <= header_length <= MAX_NPY_HEADER_BYTES:
            raise DensePredictionsError("Dense Predictions NPY header exceeds 64 KiB")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise DensePredictionsError("Dense Predictions NPY header is truncated")
        try:
            header = ast.literal_eval(raw_header.decode("latin1").strip())
        except (UnicodeError, SyntaxError, ValueError, MemoryError) as exc:
            raise DensePredictionsError("Dense Predictions NPY header is invalid") from exc
        if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
            raise DensePredictionsError("Dense Predictions NPY header fields are invalid")
        shape = header["shape"]
        if (
            not isinstance(header["descr"], str)
            or not isinstance(header["fortran_order"], bool)
            or not isinstance(shape, tuple)
            or len(shape) != 3
            or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape)
        ):
            raise DensePredictionsError("Dense Predictions NPY contract is invalid")
        return header["descr"], header["fortran_order"], tuple(shape), stream.tell()


def _validate_npy(path: Path, shape: tuple[int, int, int]) -> None:
    dtype, fortran, observed_shape, offset = _npy_contract(path)
    if dtype != "<f4" or fortran or observed_shape != shape:
        raise DensePredictionsError("Dense Predictions chunk contract mismatch")
    expected = offset + math.prod(shape) * np.dtype("<f4").itemsize
    if path.stat().st_size != expected:
        raise DensePredictionsError("Dense Predictions chunk byte length is invalid")


def _walk_plain_files(root: Path) -> set[str]:
    found: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink() or is_reparse_point(path):
                    raise DensePredictionsError("Dense Predictions contains linked content")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    found.add(path.relative_to(root).as_posix())
                else:
                    raise DensePredictionsError("Dense Predictions contains an unsupported entry")
    return found


def _validate_descriptor(raw: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = require_exact_object(
        raw,
        {
            "schema_version", "completion_state", "manifest_path",
            "manifest_byte_length", "manifest_sha256",
        },
        label="Dense Predictions descriptor",
    )
    version = require_text(descriptor["schema_version"], label="dense schema version", maximum=64)
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise DensePredictionsError("Dense Predictions schema version is invalid")
    if descriptor["completion_state"] != "complete":
        raise DensePredictionsError("Dense Predictions descriptor is not complete")
    if descriptor["manifest_path"] != "dense/manifest.json":
        raise DensePredictionsError("Dense Predictions manifest path is not canonical")
    length = descriptor["manifest_byte_length"]
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 16 * 1024 * 1024:
        raise DensePredictionsError("Dense Predictions manifest length is invalid")
    if not SHA256.fullmatch(str(descriptor["manifest_sha256"])):
        raise DensePredictionsError("Dense Predictions manifest checksum is invalid")
    return dict(descriptor)


def validate_dense_descriptor(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the bounded top-level descriptor, never component chunks."""

    return _validate_descriptor(raw)


def validate_dense_component(
    result_root: Path,
    descriptor: Mapping[str, Any],
    *,
    supported_versions: tuple[str, ...] = (DENSE_SCHEMA_VERSION,),
) -> dict[str, Any]:
    """Fully validate the optional component before any dense consumer reads it."""

    checked = _validate_descriptor(descriptor)
    if checked["schema_version"] not in supported_versions:
        raise DensePredictionsIncompatible(
            f"unsupported Dense Predictions schema {checked['schema_version']}"
        )
    result_root = Path(os.path.abspath(result_root))
    dense_root = result_root / "dense"
    if not dense_root.is_dir() or dense_root.is_symlink() or is_reparse_point(dense_root):
        raise DensePredictionsError("Dense Predictions directory is unavailable")
    manifest_path = _safe_relative_file(result_root, checked["manifest_path"])
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != checked["manifest_byte_length"]
        or _sha256_file(manifest_path) != checked["manifest_sha256"]
    ):
        raise DensePredictionsError("Dense Predictions manifest is missing or corrupt")
    manifest = read_json(manifest_path)
    require_exact_object(
        manifest,
        {
            "schema_version", "component", "completion_state", "frame_count",
            "model_grid", "chunk_frame_limit", "signals", "provenance",
        },
        label="Dense Predictions manifest",
    )
    if manifest["schema_version"] != checked["schema_version"]:
        raise DensePredictionsError("Dense Predictions versions disagree")
    if manifest["component"] != "dense_predictions" or manifest["completion_state"] != "complete":
        raise DensePredictionsError("Dense Predictions component is not complete")
    frame_count = manifest["frame_count"]
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise DensePredictionsError("Dense Predictions frame count is invalid")
    grid = require_exact_object(manifest["model_grid"], {"height", "width"}, label="dense grid")
    if not all(isinstance(grid[name], int) and not isinstance(grid[name], bool) and grid[name] > 0 for name in grid):
        raise DensePredictionsError("Dense Predictions model grid is invalid")
    if manifest["chunk_frame_limit"] != DENSE_CHUNK_FRAMES:
        raise DensePredictionsError("Dense Predictions chunk limit is invalid")
    provenance = require_exact_object(
        manifest["provenance"],
        {
            "job_id", "source_sha256", "model_sha256", "profile_settings_sha256",
            "preprocessing_rule_version", "alignment_rule_version",
        },
        label="dense provenance",
    )
    if not re.fullmatch(r"job-[0-9a-f]{32}", str(provenance["job_id"])):
        raise DensePredictionsError("Dense Predictions Job identity is invalid")
    for name in ("source_sha256", "model_sha256", "profile_settings_sha256"):
        if not SHA256.fullmatch(str(provenance[name])):
            raise DensePredictionsError(f"Dense Predictions provenance is invalid: {name}")
    for name in ("preprocessing_rule_version", "alignment_rule_version"):
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", str(provenance[name])):
            raise DensePredictionsError(f"Dense Predictions rule version is invalid: {name}")
    signals = require_exact_object(
        manifest["signals"], {"depth", "depth_confidence"}, label="dense signals"
    )
    declared = {"manifest.json"}
    expected_starts: list[int] | None = None
    for signal_name in ("depth", "depth_confidence"):
        signal = require_exact_object(
            signals[signal_name], {"dtype", "shape", "chunks"}, label=f"dense {signal_name}"
        )
        if signal["dtype"] != "<f4" or signal["shape"] != [frame_count, grid["height"], grid["width"]]:
            raise DensePredictionsError("Dense Predictions signal shape or dtype is invalid")
        chunks = signal["chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise DensePredictionsError("Dense Predictions chunks are absent")
        starts: list[int] = []
        completed = 0
        for raw in chunks:
            chunk = require_exact_object(
                raw,
                {"path", "dtype", "shape", "frame_start", "frame_count", "byte_length", "sha256"},
                label="Dense Predictions chunk",
            )
            count = chunk["frame_count"]
            if (
                chunk["dtype"] != "<f4"
                or chunk["frame_start"] != completed
                or isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= DENSE_CHUNK_FRAMES
                or chunk["shape"] != [count, grid["height"], grid["width"]]
            ):
                raise DensePredictionsError("Dense Predictions chunk coverage is invalid")
            relative = f"{signal_name}/{completed:08d}-{completed + count - 1:08d}.npy"
            if chunk["path"] != relative:
                raise DensePredictionsError("Dense Predictions chunk path is not canonical")
            path = _safe_relative_file(dense_root, relative)
            length = chunk["byte_length"]
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or length < 1
                or not SHA256.fullmatch(str(chunk["sha256"]))
            ):
                raise DensePredictionsError("Dense Predictions chunk descriptor is invalid")
            _validate_npy(path, (count, grid["height"], grid["width"]))
            if path.stat().st_size != length or _sha256_file(path) != chunk["sha256"]:
                raise DensePredictionsError("Dense Predictions chunk is missing or corrupt")
            starts.append(completed)
            completed += count
            declared.add(relative)
        if completed != frame_count:
            raise DensePredictionsError("Dense Predictions chunks do not cover every frame")
        if expected_starts is None:
            expected_starts = starts
        elif expected_starts != starts:
            raise DensePredictionsError("Dense Predictions signals use different chunk boundaries")
    if _walk_plain_files(dense_root) != declared:
        raise DensePredictionsError("Dense Predictions contains undeclared or missing files")
    return manifest


class DensePredictionWriter:
    """Write at most one 64-frame pair of dense chunks in memory."""

    def __init__(
        self,
        *,
        project_root: Path,
        job_id: str,
        frame_count: int,
        grid_shape: tuple[int, int],
        disk_check: DiskCheck,
    ) -> None:
        if not re.fullmatch(r"job-[0-9a-f]{32}", job_id):
            raise DensePredictionsError("Dense Predictions Job identity is invalid")
        if frame_count < 1 or len(grid_shape) != 2 or min(grid_shape) < 1:
            raise DensePredictionsError("Dense Predictions dimensions are invalid")
        self.project_root = Path(os.path.abspath(project_root))
        self.job_id = job_id
        self.frame_count = int(frame_count)
        self.grid_shape = tuple(int(item) for item in grid_shape)
        self.disk_check = disk_check
        self.estimated_remaining_bytes = estimate_dense_prediction_bytes(
            self.frame_count, math.prod(self.grid_shape)
        )
        results_root = self.project_root / "results"
        if not results_root.is_dir() or results_root.is_symlink() or is_reparse_point(results_root):
            raise DensePredictionsError("Dense Predictions results root is unavailable")
        self.staging = results_root / f".dense-staging-{job_id}-{uuid.uuid4().hex[:8]}"
        self.staging.mkdir()
        self._depth: list[np.ndarray] = []
        self._confidence: list[np.ndarray] = []
        self._chunks: dict[str, list[dict[str, Any]]] = {"depth": [], "depth_confidence": []}
        self._accepted = 0
        self._finished = False

    @property
    def buffered_frames(self) -> int:
        return len(self._depth)

    def accept(self, frame_index: int, depth: np.ndarray, confidence: np.ndarray) -> None:
        if self._finished or frame_index != self._accepted or frame_index >= self.frame_count:
            raise DensePredictionsError("Dense Predictions frame identity is invalid")
        for array, label in ((depth, "depth"), (confidence, "depth_confidence")):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype.str != "<f4"
                or array.shape != self.grid_shape
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
            ):
                raise DensePredictionsError(f"Dense Predictions {label} frame is invalid")
        self._depth.append(depth.copy(order="C"))
        self._confidence.append(confidence.copy(order="C"))
        self._accepted += 1
        if len(self._depth) == DENSE_CHUNK_FRAMES:
            self._flush()

    def _flush(self) -> None:
        if not self._depth:
            return
        count = len(self._depth)
        start = self._accepted - count
        pair = {
            "depth": np.ascontiguousarray(np.stack(self._depth), dtype="<f4"),
            "depth_confidence": np.ascontiguousarray(np.stack(self._confidence), dtype="<f4"),
        }
        self._depth.clear()
        self._confidence.clear()
        for name, array in pair.items():
            self.disk_check(self.estimated_remaining_bytes)
            relative = f"{name}/{start:08d}-{start + count - 1:08d}.npy"
            path = _safe_relative_file(self.staging, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            _validate_npy(path, tuple(array.shape))
            descriptor = {
                "path": relative,
                "dtype": "<f4",
                "shape": list(array.shape),
                "frame_start": start,
                "frame_count": count,
                "byte_length": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            self._chunks[name].append(descriptor)
            self.estimated_remaining_bytes = max(
                0, self.estimated_remaining_bytes - path.stat().st_size
            )

    def finish(self, provenance: Mapping[str, Any]) -> DenseComponentArtifact:
        if self._finished:
            raise DensePredictionsError("Dense Predictions finish is not repeatable")
        if self._accepted != self.frame_count:
            raise DensePredictionsError(
                f"Dense Predictions received {self._accepted} of {self.frame_count} frames"
            )
        self._flush()
        dense_provenance = {
            "job_id": self.job_id,
            "source_sha256": provenance["source_sha256"],
            "model_sha256": provenance["model_sha256"],
            "profile_settings_sha256": provenance["profile"]["settings_sha256"],
            "preprocessing_rule_version": provenance["preprocessing_rule_version"],
            "alignment_rule_version": "1.0.0",
        }
        manifest = {
            "schema_version": DENSE_SCHEMA_VERSION,
            "component": "dense_predictions",
            "completion_state": "complete",
            "frame_count": self.frame_count,
            "model_grid": {"height": self.grid_shape[0], "width": self.grid_shape[1]},
            "chunk_frame_limit": DENSE_CHUNK_FRAMES,
            "signals": {
                name: {
                    "dtype": "<f4",
                    "shape": [self.frame_count, *self.grid_shape],
                    "chunks": chunks,
                }
                for name, chunks in self._chunks.items()
            },
            "provenance": dense_provenance,
        }
        self.disk_check(self.estimated_remaining_bytes)
        manifest_path = self.staging / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        descriptor = {
            "schema_version": DENSE_SCHEMA_VERSION,
            "completion_state": "complete",
            "manifest_path": "dense/manifest.json",
            "manifest_byte_length": manifest_path.stat().st_size,
            "manifest_sha256": _sha256_file(manifest_path),
        }
        self._finished = True
        return DenseComponentArtifact(self.staging, descriptor)

    def abort(self, reason: str) -> Path | None:
        self._depth.clear()
        self._confidence.clear()
        if not self.staging.exists():
            return None
        diagnostics = self.project_root / "diagnostics"
        destination = diagnostics / f"{self.job_id}--dense-{reason}"
        if destination.exists():
            destination = diagnostics / f"{self.job_id}--dense-{reason}-{uuid.uuid4().hex[:8]}"
        os.replace(self.staging, destination)
        atomic_write_json(
            destination / "incomplete.json",
            {
                "schema_version": DENSE_SCHEMA_VERSION,
                "job_id": self.job_id,
                "completion_state": "incomplete",
                "accepted_frames": self._accepted,
                "expected_frames": self.frame_count,
                "reason": str(reason)[:128],
            },
        )
        return destination
