"""Bounded Project inventory and explicit recoverable disk lifecycle actions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import struct
from typing import Any, Callable, Mapping

from .ipc import IpcError, read_json, require_exact_object, require_text
from .job_lifecycle import (
    is_reparse_point,
    normalized_blend_path,
    project_result_root,
)
from .result_import import ResultImportError, validate_result
from .results import JOB_ID, RESULT_DIRECTORY, RESULT_ID


DISCOVERY_BATCH_LIMIT = 50
INVENTORY_PAGE_SIZE = 200
MAX_JOB_ENTRIES = 32
DENSE_SCHEMA_VERSION = "1.0.0"
DENSE_CHUNK_FRAMES = 64
MAX_NPY_HEADER_BYTES = 64 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TRASH_TIMESTAMP = re.compile(r"[0-9]{8}T[0-9]{12}Z\Z")
DIAGNOSTIC_NAME = re.compile(
    # Reserve room in one Windows path component for the Trash marker,
    # timestamp, and longest partial-delete collision suffix.
    r"job-[0-9a-f]{32}--[A-Za-z0-9][A-Za-z0-9._-]{0,157}\Z"
)
PARTIAL_SUFFIX = re.compile(r"--partial_delete(?:-[0-9a-f]{8})?\Z")


class ProjectLifecycleError(RuntimeError):
    """A Project disk action cannot be proven safe."""


class ProjectConflictError(ProjectLifecycleError):
    """The requested destination already exists or identity is ambiguous."""


class ProjectRaceError(ProjectLifecycleError):
    """Content changed after confirmation and before use."""


class PartialDeletionError(ProjectLifecycleError):
    """Permanent deletion stopped with explicitly marked remnants."""

    def __init__(self, message: str, partial_items: tuple[str, ...]):
        super().__init__(message)
        self.partial_items = partial_items


@dataclass(frozen=True)
class InventoryItem:
    category: str
    name: str
    status: str
    ordinary: bool
    result_id: str | None = None
    job_id: str | None = None
    created_utc: str | None = None
    point_count: int | None = None
    frame_count: int | None = None
    profile_name: str | None = None
    dense_status: str | None = None
    scene_uuid: str | None = None
    scene_name: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class InventorySnapshot:
    scanned_entries: int
    complete: bool
    jobs_abnormal: bool
    items: tuple[InventoryItem, ...]

    def page(
        self,
        category: str,
        page: int = 0,
        page_size: int = INVENTORY_PAGE_SIZE,
    ) -> tuple[InventoryItem, ...]:
        if page < 0 or not 1 <= page_size <= INVENTORY_PAGE_SIZE:
            raise ProjectLifecycleError("Inventory page request is invalid")
        matches = tuple(
            item for item in self.items if item.category == category
        )
        start = page * page_size
        return matches[start : start + page_size]


@dataclass(frozen=True)
class TreeRecord:
    relative_path: str
    kind: str
    size: int
    device: int
    inode: int
    modified_ns: int


@dataclass(frozen=True)
class LifecyclePlan:
    action: str
    names: tuple[str, ...]
    sources: tuple[Path, ...]
    destinations: tuple[Path, ...]
    records: tuple[tuple[TreeRecord, ...], ...]
    guards: tuple[Path, ...]
    guard_records: tuple[tuple[TreeRecord, ...], ...]
    item_count: int
    file_count: int
    byte_count: int
    chunk_count: int
    result_id: str | None
    fingerprint: str


@dataclass(frozen=True)
class LifecycleOutcome:
    action: str
    item_count: int
    destinations: tuple[Path, ...]


class PermanentDeletionSession:
    """Delete one validated Trash file per UI step and remain cancellable."""

    def __init__(self, lifecycle: "ProjectLifecycle", plan: LifecyclePlan):
        self._lifecycle = lifecycle
        self._plan = plan
        self._item_index = 0
        self._file_index = 0
        self._item_modified = False
        self._deleted_items = 0
        self._finished = False

    @property
    def completed_files(self) -> int:
        return sum(
            sum(record.kind == "file" for record in records)
            for records in self._plan.records[: self._item_index]
        ) + self._file_index

    @property
    def total_files(self) -> int:
        return self._plan.file_count

    def _partial_failure(self, exc: BaseException) -> None:
        partial = ()
        marker_error = ""
        if self._item_index < len(self._plan.sources):
            source = self._plan.sources[self._item_index]
            if self._item_modified or self._deleted_items:
                try:
                    if _path_occupied(source):
                        partial = (
                            self._lifecycle._mark_partial(source).name,
                        )
                except (OSError, ProjectLifecycleError) as marker_exc:
                    marker_error = (
                        "; partial-delete marker failed safely: "
                        f"{marker_exc}"
                    )
        self._finished = True
        if self._deleted_items or self._item_modified or partial:
            raise PartialDeletionError(
                "Permanent deletion is partial after "
                f"{self._deleted_items} complete item(s): {exc}"
                f"{marker_error}",
                partial,
            ) from exc
        if isinstance(exc, ProjectLifecycleError):
            raise exc
        raise ProjectLifecycleError(
            f"Permanent deletion stopped before changing Trash: {exc}"
        ) from exc

    def cancel(self) -> None:
        if self._finished:
            return
        self._partial_failure(
            InterruptedError("Permanent deletion cancelled between files")
        )

    def step(
        self, maximum_files: int = 1
    ) -> LifecycleOutcome | None:
        if self._finished:
            return LifecycleOutcome(
                "delete", self._deleted_items, ()
            )
        if maximum_files < 1:
            raise ProjectLifecycleError(
                "Deletion step must permit at least one file"
            )
        remaining = maximum_files
        try:
            while self._item_index < len(self._plan.sources):
                source = self._plan.sources[self._item_index]
                records = self._plan.records[self._item_index]
                files = tuple(
                    record for record in records if record.kind == "file"
                )
                expected = {
                    record.relative_path: record for record in records
                }
                while self._file_index < len(files) and remaining:
                    record = files[self._file_index]
                    path = source / Path(
                        record.relative_path.replace("/", os.sep)
                    )
                    _require_plain_ancestors(source, path)
                    stat = _entry_stat(path)
                    current = TreeRecord(
                        record.relative_path,
                        "file",
                        int(stat.st_size),
                        int(stat.st_dev),
                        int(stat.st_ino),
                        int(stat.st_mtime_ns),
                    )
                    if current != expected[record.relative_path]:
                        raise ProjectRaceError(
                            "Trash file changed during permanent deletion"
                        )
                    path.unlink()
                    self._item_modified = True
                    self._file_index += 1
                    remaining -= 1
                if self._file_index < len(files):
                    return None
                directories = sorted(
                    (
                        record
                        for record in records
                        if record.kind == "directory"
                    ),
                    key=lambda record: (
                        record.relative_path.count("/"),
                        record.relative_path,
                    ),
                    reverse=True,
                )
                for record in directories:
                    directory = (
                        source
                        / Path(record.relative_path.replace("/", os.sep))
                    )
                    _require_plain_ancestors(source, directory)
                    stat = _entry_stat(
                        _plain_directory(
                            directory, "Trash directory"
                        )
                    )
                    if (
                        int(stat.st_dev) != record.device
                        or int(stat.st_ino) != record.inode
                    ):
                        raise ProjectRaceError(
                            "Trash directory changed during permanent deletion"
                        )
                    directory.rmdir()
                    self._item_modified = True
                root_record = next(
                    record
                    for record in records
                    if record.kind == "root"
                )
                root_stat = _entry_stat(
                    _plain_directory(source, "Trash item")
                )
                if (
                    int(root_stat.st_dev) != root_record.device
                    or int(root_stat.st_ino) != root_record.inode
                ):
                    raise ProjectRaceError(
                        "Trash item changed during permanent deletion"
                    )
                source.rmdir()
                self._deleted_items += 1
                self._item_index += 1
                self._file_index = 0
                self._item_modified = False
                if self._item_index == len(self._plan.sources):
                    self._finished = True
                    return LifecycleOutcome(
                        "delete", self._deleted_items, ()
                    )
                if not remaining:
                    return None
        except (
            InterruptedError,
            OSError,
            ProjectLifecycleError,
        ) as exc:
            self._partial_failure(exc)
        self._finished = True
        return LifecycleOutcome("delete", self._deleted_items, ())


def _plain_directory(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(path))
    if (
        not path.is_dir()
        or path.is_symlink()
        or is_reparse_point(path)
    ):
        raise ProjectLifecycleError(f"{label} is not an ordinary directory")
    return path


def _path_occupied(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ProjectLifecycleError(
            f"Cannot inspect destination {path.name}: {exc}"
        ) from exc
    return True


def _direct_child(root: Path, name: str, label: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or Path(name).name != name
    ):
        raise ProjectLifecycleError(f"{label} name is unsafe")
    root = _plain_directory(root, f"{label} root")
    path = root / name
    if path.parent != root:
        raise ProjectLifecycleError(f"{label} escaped its root")
    return _plain_directory(path, label)


def _entry_stat(path: Path) -> os.stat_result:
    try:
        stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProjectLifecycleError(
            f"Cannot inspect {path.name}: {exc}"
        ) from exc
    if path.is_symlink() or is_reparse_point(path):
        raise ProjectLifecycleError(
            f"{path.name} contains linked or reparse content"
        )
    return stat


def _require_plain_ancestors(root: Path, path: Path) -> None:
    current = path.parent
    while current != root:
        if root not in current.parents:
            raise ProjectLifecycleError(
                "Lifecycle file escaped its validated item"
            )
        _plain_directory(current, "Lifecycle file ancestor")
        current = current.parent
    _plain_directory(root, "Lifecycle item")


def _tree_records(root: Path) -> tuple[TreeRecord, ...]:
    root = _plain_directory(root, "Lifecycle item")
    root_stat = _entry_stat(root)
    found: list[TreeRecord] = [
        TreeRecord(
            ".",
            "root",
            0,
            int(root_stat.st_dev),
            int(root_stat.st_ino),
            int(root_stat.st_mtime_ns),
        )
    ]
    pending = [root]
    while pending:
        directory = pending.pop()
        before = _entry_stat(
            _plain_directory(directory, "Lifecycle tree directory")
        )
        with os.scandir(directory) as entries:
            current = sorted(tuple(entries), key=lambda item: item.name)
        after = _entry_stat(
            _plain_directory(directory, "Lifecycle tree directory")
        )
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ProjectRaceError(
                f"{root.name} directory changed during inspection"
            )
        for entry in current:
            path = Path(entry.path)
            if entry.is_symlink() or is_reparse_point(path):
                raise ProjectLifecycleError(
                    f"{root.name} contains linked or reparse content"
                )
            stat = _entry_stat(path)
            relative = path.relative_to(root).as_posix()
            if entry.is_dir(follow_symlinks=False):
                found.append(
                    TreeRecord(
                        relative,
                        "directory",
                        0,
                        int(stat.st_dev),
                        int(stat.st_ino),
                        int(stat.st_mtime_ns),
                    )
                )
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                found.append(
                    TreeRecord(
                        relative,
                        "file",
                        int(stat.st_size),
                        int(stat.st_dev),
                        int(stat.st_ino),
                        int(stat.st_mtime_ns),
                    )
                )
            else:
                raise ProjectLifecycleError(
                    f"{root.name} contains an unsupported filesystem entry"
                )
    return tuple(sorted(found, key=lambda item: item.relative_path))


def _records_fingerprint(
    action: str,
    sources: tuple[Path, ...],
    destinations: tuple[Path, ...],
    records: tuple[tuple[TreeRecord, ...], ...],
    guards: tuple[Path, ...] = (),
    guard_records: tuple[tuple[TreeRecord, ...], ...] = (),
) -> str:
    document = {
        "action": action,
        "sources": [str(item) for item in sources],
        "destinations": [str(item) for item in destinations],
        "records": [
            [
                {
                    "path": record.relative_path,
                    "kind": record.kind,
                    "size": record.size,
                    "device": record.device,
                    "inode": record.inode,
                    "modified_ns": record.modified_ns,
                }
                for record in group
            ]
            for group in records
        ],
        "guards": [str(item) for item in guards],
        "guard_records": [
            [
                {
                    "path": record.relative_path,
                    "kind": record.kind,
                    "size": record.size,
                    "device": record.device,
                    "inode": record.inode,
                    "modified_ns": record.modified_ns,
                }
                for record in group
            ]
            for group in guard_records
        ],
    }
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _stable_sha256(path: Path) -> tuple[str, int]:
    if (
        not path.is_file()
        or path.is_symlink()
        or is_reparse_point(path)
    ):
        raise ProjectLifecycleError(
            f"{path.name} is not an ordinary file"
        )
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
        raise ProjectRaceError(f"{path.name} changed while hashing")
    return digest.hexdigest(), completed


def _safe_relative_file(root: Path, value: Any) -> Path:
    text = require_text(
        value, label="Dense Predictions path", maximum=32767
    )
    posix, windows = PurePosixPath(text), PureWindowsPath(text)
    if (
        "\\" in text
        or "\x00" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or text.startswith("//")
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ProjectLifecycleError(
            "Dense Predictions path escaped its component"
        )
    path = root.joinpath(*posix.parts)
    current = path
    while current != root:
        if current.exists() and (
            current.is_symlink() or is_reparse_point(current)
        ):
            raise ProjectLifecycleError(
                "Dense Predictions path uses linked content"
            )
        current = current.parent
    return path


def _target_blend_path(value: Any) -> Path:
    text = require_text(
        value, label="Target Scene blend_path", maximum=32767
    )
    candidate = Path(text)
    if not candidate.is_absolute():
        raise ProjectLifecycleError(
            "Target Scene blend_path is not absolute"
        )
    return normalized_blend_path(candidate)


def _npy_contract(path: Path) -> tuple[str, bool, tuple[int, ...], int]:
    if (
        not path.is_file()
        or path.is_symlink()
        or is_reparse_point(path)
    ):
        raise ProjectLifecycleError(
            "Dense Predictions chunk is not an ordinary file"
        )
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ProjectLifecycleError("Dense NPY magic is invalid")
        version = stream.read(2)
        if version == b"\x01\x00":
            raw_length = stream.read(2)
            if len(raw_length) != 2:
                raise ProjectLifecycleError("Dense NPY header is truncated")
            header_length = struct.unpack("<H", raw_length)[0]
        elif version in {b"\x02\x00", b"\x03\x00"}:
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                raise ProjectLifecycleError("Dense NPY header is truncated")
            header_length = struct.unpack("<I", raw_length)[0]
        else:
            raise ProjectLifecycleError("Dense NPY version is unsupported")
        if not 1 <= header_length <= MAX_NPY_HEADER_BYTES:
            raise ProjectLifecycleError("Dense NPY header exceeds 64 KiB")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ProjectLifecycleError("Dense NPY header is truncated")
        try:
            header = ast.literal_eval(raw_header.decode("latin1").strip())
        except (UnicodeError, SyntaxError, ValueError, MemoryError) as exc:
            raise ProjectLifecycleError("Dense NPY header is invalid") from exc
        if not isinstance(header, dict) or set(header) != {
            "descr",
            "fortran_order",
            "shape",
        }:
            raise ProjectLifecycleError("Dense NPY header fields are invalid")
        shape = header["shape"]
        if (
            not isinstance(header["descr"], str)
            or not isinstance(header["fortran_order"], bool)
            or not isinstance(shape, tuple)
            or len(shape) != 3
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in shape
            )
        ):
            raise ProjectLifecycleError("Dense NPY contract is invalid")
        return (
            header["descr"],
            header["fortran_order"],
            tuple(shape),
            stream.tell(),
        )


def _validate_dense(
    dense_root: Path, descriptor: Mapping[str, Any]
) -> tuple[int, int, tuple[TreeRecord, ...]]:
    checked = require_exact_object(
        descriptor,
        {
            "schema_version",
            "completion_state",
            "manifest_path",
            "manifest_byte_length",
            "manifest_sha256",
        },
        label="Dense Predictions descriptor",
    )
    if (
        checked["schema_version"] != DENSE_SCHEMA_VERSION
        or checked["completion_state"] != "complete"
        or checked["manifest_path"] != "dense/manifest.json"
        or isinstance(checked["manifest_byte_length"], bool)
        or not isinstance(checked["manifest_byte_length"], int)
        or not 1
        <= checked["manifest_byte_length"]
        <= 16 * 1024 * 1024
        or not SHA256.fullmatch(str(checked["manifest_sha256"]))
    ):
        raise ProjectLifecycleError(
            "Dense Predictions descriptor is invalid or incompatible"
        )
    dense_root = _plain_directory(
        dense_root, "Dense Predictions directory"
    )
    before_records = _tree_records(dense_root)
    manifest_path = _safe_relative_file(dense_root, "manifest.json")
    digest, length = _stable_sha256(manifest_path)
    if (
        length != checked["manifest_byte_length"]
        or digest != checked["manifest_sha256"]
    ):
        raise ProjectLifecycleError(
            "Dense Predictions manifest is missing or corrupt"
        )
    manifest = require_exact_object(
        read_json(manifest_path, maximum=16 * 1024 * 1024),
        {
            "schema_version",
            "component",
            "completion_state",
            "frame_count",
            "model_grid",
            "chunk_frame_limit",
            "signals",
            "provenance",
        },
        label="Dense Predictions manifest",
    )
    if (
        manifest["schema_version"] != DENSE_SCHEMA_VERSION
        or manifest["component"] != "dense_predictions"
        or manifest["completion_state"] != "complete"
        or manifest["chunk_frame_limit"] != DENSE_CHUNK_FRAMES
    ):
        raise ProjectLifecycleError(
            "Dense Predictions manifest contract is invalid"
        )
    frame_count = manifest["frame_count"]
    grid = require_exact_object(
        manifest["model_grid"], {"height", "width"}, label="dense grid"
    )
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
        or not all(
            isinstance(grid[name], int)
            and not isinstance(grid[name], bool)
            and grid[name] > 0
            for name in grid
        )
    ):
        raise ProjectLifecycleError("Dense dimensions are invalid")
    provenance = require_exact_object(
        manifest["provenance"],
        {
            "job_id",
            "source_sha256",
            "model_sha256",
            "profile_settings_sha256",
            "preprocessing_rule_version",
            "alignment_rule_version",
        },
        label="dense provenance",
    )
    if not JOB_ID.fullmatch(str(provenance["job_id"])):
        raise ProjectLifecycleError("Dense Job identity is invalid")
    for name in (
        "source_sha256",
        "model_sha256",
        "profile_settings_sha256",
    ):
        if not SHA256.fullmatch(str(provenance[name])):
            raise ProjectLifecycleError("Dense provenance is invalid")
    signals = require_exact_object(
        manifest["signals"],
        {"depth", "depth_confidence"},
        label="dense signals",
    )
    declared = {"manifest.json"}
    expected_starts: list[int] | None = None
    chunk_count = 0
    for signal_name in ("depth", "depth_confidence"):
        signal = require_exact_object(
            signals[signal_name],
            {"dtype", "shape", "chunks"},
            label=f"dense {signal_name}",
        )
        if signal["dtype"] != "<f4" or signal["shape"] != [
            frame_count,
            grid["height"],
            grid["width"],
        ]:
            raise ProjectLifecycleError("Dense signal shape is invalid")
        chunks = signal["chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise ProjectLifecycleError("Dense chunks are absent")
        starts: list[int] = []
        completed = 0
        for raw in chunks:
            chunk = require_exact_object(
                raw,
                {
                    "path",
                    "dtype",
                    "shape",
                    "frame_start",
                    "frame_count",
                    "byte_length",
                    "sha256",
                },
                label="Dense Predictions chunk",
            )
            count = chunk["frame_count"]
            relative = (
                f"{signal_name}/{completed:08d}-"
                f"{completed + int(count) - 1:08d}.npy"
                if isinstance(count, int) and not isinstance(count, bool)
                else ""
            )
            if (
                chunk["dtype"] != "<f4"
                or chunk["frame_start"] != completed
                or isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= DENSE_CHUNK_FRAMES
                or chunk["shape"]
                != [count, grid["height"], grid["width"]]
                or chunk["path"] != relative
                or isinstance(chunk["byte_length"], bool)
                or not isinstance(chunk["byte_length"], int)
                or chunk["byte_length"] < 1
                or not SHA256.fullmatch(str(chunk["sha256"]))
            ):
                raise ProjectLifecycleError(
                    "Dense chunk descriptor is invalid"
                )
            path = _safe_relative_file(dense_root, relative)
            dtype, fortran, shape, offset = _npy_contract(path)
            expected_size = offset + math.prod(shape) * 4
            observed_digest, observed_size = _stable_sha256(path)
            if (
                dtype != "<f4"
                or fortran
                or shape != (count, grid["height"], grid["width"])
                or observed_size != expected_size
                or observed_size != chunk["byte_length"]
                or observed_digest != chunk["sha256"]
            ):
                raise ProjectLifecycleError(
                    "Dense chunk is missing or corrupt"
                )
            starts.append(completed)
            completed += count
            declared.add(relative)
            chunk_count += 1
        if completed != frame_count:
            raise ProjectLifecycleError(
                "Dense chunks do not cover every frame"
            )
        if expected_starts is None:
            expected_starts = starts
        elif expected_starts != starts:
            raise ProjectLifecycleError(
                "Dense signals use different chunk boundaries"
            )
    after_records = _tree_records(dense_root)
    if before_records != after_records:
        raise ProjectRaceError(
            "Dense Predictions changed during complete validation"
        )
    observed = {
        record.relative_path
        for record in after_records
        if record.kind == "file"
    }
    if observed != declared:
        raise ProjectLifecycleError(
            "Dense Predictions contains undeclared files"
        )
    return chunk_count, sum(
        record.size
        for record in after_records
        if record.kind == "file"
    ), after_records


def _result_directory_name(manifest: Mapping[str, Any]) -> str:
    created = require_text(
        manifest["created_utc"], label="created_utc", maximum=128
    )
    timestamp = re.sub(r"[^0-9]", "", created)[:14]
    job_id = require_text(
        manifest["job_id"], label="job_id", maximum=64
    )
    if len(timestamp) != 14 or not JOB_ID.fullmatch(job_id):
        raise ProjectLifecycleError(
            "Result cannot reconstruct its canonical destination"
        )
    return f"{timestamp}Z-{job_id[4:12]}"


def _trash_base(name: str) -> tuple[str, bool]:
    match = PARTIAL_SUFFIX.search(name)
    return (name[: match.start()], True) if match else (name, False)


def _trash_kind(name: str) -> tuple[str, str] | None:
    base, partial = _trash_base(name)
    for marker, kind in (
        ("_result_", "result"),
        ("_dense_", "dense"),
        ("_diagnostic_", "diagnostic"),
    ):
        original, separator, timestamp = base.rpartition(marker)
        if (
            separator
            and original
            and TRASH_TIMESTAMP.fullmatch(timestamp)
        ):
            return kind + ("-partial_delete" if partial else ""), original
    return None


def _result_listing(path: Path, blend_path: Path) -> InventoryItem:
    try:
        manifest_path = path / "manifest.json"
        if (
            manifest_path.is_symlink()
            or is_reparse_point(manifest_path)
        ):
            raise IpcError("manifest uses linked content")
        manifest = read_json(manifest_path)
        result_id = require_text(
            manifest["result_id"], label="result_id", maximum=64
        )
        job_id = require_text(
            manifest["job_id"], label="job_id", maximum=64
        )
        target = require_exact_object(
            manifest["target_scene"],
            {"blend_path", "scene_uuid", "scene_name"},
            label="target_scene",
        )
        counts = require_exact_object(
            manifest["counts"], {"frames", "points"}, label="counts"
        )
        profile = manifest["profile"]
        if (
            manifest.get("schema_version") != "1.0.0"
            or not RESULT_ID.fullmatch(result_id)
            or not JOB_ID.fullmatch(job_id)
            or not isinstance(profile, dict)
            or not isinstance(counts["frames"], int)
            or isinstance(counts["frames"], bool)
            or not isinstance(counts["points"], int)
            or isinstance(counts["points"], bool)
            or counts["frames"] < 0
            or counts["points"] < 0
            or os.path.normcase(
                str(_target_blend_path(target["blend_path"]))
            )
            != os.path.normcase(str(blend_path))
        ):
            raise IpcError("Result listing identity is invalid")
        dense = manifest.get("dense_predictions")
        dense_status = (
            "retained-unvalidated" if isinstance(dense, dict)
            else "not-retained"
        )
        return InventoryItem(
            "results",
            path.name,
            "recognized",
            True,
            result_id=result_id,
            job_id=job_id,
            created_utc=str(manifest.get("created_utc", "")),
            point_count=counts["points"],
            frame_count=counts["frames"],
            profile_name=str(profile.get("name", "")),
            dense_status=dense_status,
            scene_uuid=str(target["scene_uuid"]),
            scene_name=str(target["scene_name"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, IpcError):
        return InventoryItem(
            "results",
            path.name,
            "unrecognized",
            True,
            detail="Result manifest is unrecognized or belongs elsewhere",
        )


class ProjectInventory:
    """Incrementally list only direct Project children across UI ticks."""

    def __init__(self, blend_path: str | Path):
        self.blend_path = normalized_blend_path(blend_path)
        self.root = project_result_root(self.blend_path)
        self._categories = ("jobs", "results", "diagnostics", "trash")
        self._category_index = 0
        self._iterator: Any | None = None
        self._items: list[InventoryItem] = []
        self._scanned = 0
        self._complete = False

    def _open_next(self) -> bool:
        while self._category_index < len(self._categories):
            category = self._categories[self._category_index]
            directory_name = ".jobs" if category == "jobs" else (
                ".trash" if category == "trash" else category
            )
            directory = self.root / directory_name
            if (
                directory.is_dir()
                and not directory.is_symlink()
                and not is_reparse_point(directory)
            ):
                self._iterator = os.scandir(directory)
                return True
            self._category_index += 1
        self._complete = True
        return False

    def _classify(self, category: str, entry: os.DirEntry[str]) -> InventoryItem:
        path = Path(entry.path)
        ordinary = (
            not entry.is_symlink()
            and entry.is_dir(follow_symlinks=False)
            and not is_reparse_point(path)
        )
        if not ordinary:
            return InventoryItem(
                category,
                entry.name,
                "unrecognized",
                False,
                detail="Entry is not an ordinary direct child directory",
            )
        if category == "results":
            if not RESULT_DIRECTORY.fullmatch(entry.name):
                return InventoryItem(
                    category,
                    entry.name,
                    "unrecognized",
                    True,
                    detail="Result directory name is unrecognized",
                )
            return _result_listing(path, self.blend_path)
        if category == "jobs":
            status = "recognized" if JOB_ID.fullmatch(entry.name) else "unrecognized"
            return InventoryItem(category, entry.name, status, True)
        if category == "diagnostics":
            job_match = re.match(r"(job-[0-9a-f]{32})--", entry.name)
            job_id = job_match.group(1) if job_match else None
            staging_ids = {
                item.name
                for item in self._items
                if item.category == "jobs"
                and item.status == "recognized"
            }
            status = (
                "recognized"
                if (
                    DIAGNOSTIC_NAME.fullmatch(entry.name)
                    and job_id not in staging_ids
                )
                else "unrecognized"
            )
            return InventoryItem(
                category,
                entry.name,
                status,
                True,
                job_id=job_id,
                detail=(
                    "Matching active or staging Job still exists"
                    if job_id in staging_ids
                    else ""
                ),
            )
        parsed = _trash_kind(entry.name)
        return InventoryItem(
            category,
            entry.name,
            parsed[0] if parsed else "unrecognized",
            True,
            result_id=(
                parsed[1]
                if parsed and parsed[0].startswith(("result", "dense"))
                else None
            ),
        )

    def advance(self, maximum: int = DISCOVERY_BATCH_LIMIT) -> InventorySnapshot:
        if not 1 <= maximum <= DISCOVERY_BATCH_LIMIT:
            raise ProjectLifecycleError(
                f"Discovery advances by 1 to {DISCOVERY_BATCH_LIMIT} entries"
            )
        remaining = maximum
        while remaining and not self._complete:
            if self._iterator is None and not self._open_next():
                break
            category = self._categories[self._category_index]
            try:
                entry = next(self._iterator)
            except StopIteration:
                self._iterator.close()
                self._iterator = None
                self._category_index += 1
                continue
            try:
                item = self._classify(category, entry)
            except OSError as exc:
                item = InventoryItem(
                    category,
                    entry.name,
                    "unrecognized",
                    False,
                    detail=f"Entry changed during discovery: {exc}",
                )
            self._items.append(item)
            self._scanned += 1
            remaining -= 1
        return self.snapshot()

    def snapshot(self) -> InventorySnapshot:
        items = tuple(
            sorted(
                self._items,
                key=lambda item: (item.category, item.name),
                reverse=True,
            )
        )
        jobs = sum(item.category == "jobs" for item in items)
        return InventorySnapshot(
            self._scanned,
            self._complete,
            jobs > MAX_JOB_ENTRIES,
            items,
        )

    def close(self) -> None:
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        self._complete = True


class ProjectLifecycle:
    """Plan and execute every authorized Project disk lifecycle mutation."""

    def __init__(
        self,
        blend_path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ):
        self.blend_path = normalized_blend_path(blend_path)
        self.root = project_result_root(self.blend_path)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _root(self) -> Path:
        return _plain_directory(self.root, "Project Result Root")

    def _timestamp(self) -> str:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ProjectLifecycleError("Lifecycle clock must be timezone-aware")
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def _require_normal_job_inventory(self) -> None:
        jobs = _plain_directory(self._root() / ".jobs", ".jobs")
        count = 0
        with os.scandir(jobs) as entries:
            for _entry in entries:
                count += 1
                if count > MAX_JOB_ENTRIES:
                    raise ProjectLifecycleError(
                        "Project has more than 32 .jobs entries; inspect "
                        "the abnormal Job inventory before lifecycle actions"
                    )

    @staticmethod
    def _validate_result_stably(
        source: Path, *, require_canonical_directory: bool = True
    ):
        before = _tree_records(source)
        try:
            document = validate_result(
                source,
                require_canonical_directory=require_canonical_directory,
            )
        except (ResultImportError, OSError) as exc:
            raise ProjectLifecycleError(
                f"Result failed complete validation: {exc}"
            ) from exc
        after = _tree_records(source)
        if before != after:
            raise ProjectRaceError(
                "Result changed during complete validation"
            )
        return document, after

    def _owned_result(self, name: str):
        results = _plain_directory(self._root() / "results", "results")
        if not RESULT_DIRECTORY.fullmatch(name):
            raise ProjectLifecycleError(
                "Result is not a canonical direct publication"
            )
        source = _direct_child(results, name, "Result")
        document, records = self._validate_result_stably(source)
        target = _target_blend_path(
            document.manifest["target_scene"]["blend_path"]
        )
        if os.path.normcase(str(target)) != os.path.normcase(
            str(self.blend_path)
        ):
            raise ProjectLifecycleError(
                "External Result has no lifecycle authority in this Project"
            )
        if _path_occupied(
            self.root / ".jobs" / document.ready.job_id
        ):
            raise ProjectLifecycleError(
                "Result still has active or staging Job content"
            )
        return source, document, records

    def _plan(
        self,
        action: str,
        names: tuple[str, ...],
        sources: tuple[Path, ...],
        destinations: tuple[Path, ...],
        *,
        chunk_count: int = 0,
        result_id: str | None = None,
        records: tuple[tuple[TreeRecord, ...], ...] | None = None,
        guards: tuple[Path, ...] = (),
        guard_records: tuple[tuple[TreeRecord, ...], ...] | None = None,
    ) -> LifecyclePlan:
        if len(set(sources)) != len(sources) or (
            action != "delete"
            and len(set(destinations)) != len(destinations)
        ):
            raise ProjectLifecycleError("Lifecycle selection is duplicated")
        if any(_path_occupied(destination) for destination in destinations):
            raise ProjectConflictError(
                "Lifecycle destination already exists; nothing was overwritten"
            )
        if records is None:
            records = tuple(
                _tree_records(source) for source in sources
            )
        if len(records) != len(sources):
            raise ProjectLifecycleError(
                "Lifecycle source evidence is incomplete"
            )
        if guard_records is None:
            guard_records = tuple(
                _tree_records(guard) for guard in guards
            )
        if len(guard_records) != len(guards):
            raise ProjectLifecycleError(
                "Lifecycle guard evidence is incomplete"
            )
        files = sum(
            record.kind == "file"
            for group in records
            for record in group
        )
        byte_count = sum(
            record.size
            for group in records
            for record in group
            if record.kind == "file"
        )
        fingerprint = _records_fingerprint(
            action,
            sources,
            destinations,
            records,
            guards,
            guard_records,
        )
        return LifecyclePlan(
            action=action,
            names=names,
            sources=sources,
            destinations=destinations,
            records=records,
            guards=guards,
            guard_records=guard_records,
            item_count=len(sources),
            file_count=files,
            byte_count=byte_count,
            chunk_count=chunk_count,
            result_id=result_id,
            fingerprint=fingerprint,
        )

    def plan(
        self, action: str, names: tuple[str, ...] | list[str]
    ) -> LifecyclePlan:
        selected = tuple(names)
        if not selected:
            raise ProjectLifecycleError("Choose at least one lifecycle item")
        if len(selected) > INVENTORY_PAGE_SIZE:
            raise ProjectLifecycleError(
                "One explicit lifecycle action is limited to 200 items"
            )
        self._require_normal_job_inventory()
        timestamp = self._timestamp()
        trash = self.root / ".trash"
        if action == "trash_result":
            if len(selected) != 1:
                raise ProjectLifecycleError(
                    "Move Result to Trash accepts exactly one Result"
                )
            source, document, source_records = self._owned_result(
                selected[0]
            )
            result_id = document.ready.result_id
            destination = trash / f"{result_id}_result_{timestamp}"
            return self._plan(
                action,
                selected,
                (source,),
                (destination,),
                result_id=result_id,
                records=(source_records,),
            )
        if action == "trash_dense":
            if len(selected) != 1:
                raise ProjectLifecycleError(
                    "Remove Dense Predictions accepts exactly one Result"
                )
            result, document, result_records = self._owned_result(
                selected[0]
            )
            descriptor = document.manifest.get("dense_predictions")
            if not isinstance(descriptor, Mapping):
                raise ProjectLifecycleError(
                    "Result has no retained Dense Predictions"
                )
            dense = _plain_directory(
                result / "dense", "Dense Predictions directory"
            )
            chunks, _bytes, dense_records = _validate_dense(
                dense, descriptor
            )
            if _tree_records(result) != result_records:
                raise ProjectRaceError(
                    "Owning Result changed during Dense validation"
                )
            result_id = document.ready.result_id
            destination = trash / f"{result_id}_dense_{timestamp}"
            return self._plan(
                action,
                selected,
                (dense,),
                (destination,),
                chunk_count=chunks,
                result_id=result_id,
                records=(dense_records,),
                guards=(result,),
                guard_records=(result_records,),
            )
        if action == "trash_diagnostics":
            diagnostics = _plain_directory(
                self._root() / "diagnostics", "diagnostics"
            )
            sources = []
            destinations = []
            for name in selected:
                if not DIAGNOSTIC_NAME.fullmatch(name):
                    raise ProjectLifecycleError(
                        "Diagnostic entry is unrecognized, active, or staging"
                    )
                job_id = name.split("--", 1)[0]
                staging = self.root / ".jobs" / job_id
                if _path_occupied(staging):
                    raise ProjectLifecycleError(
                        "Diagnostic entry still has active or staging Job "
                        "content"
                    )
                source = _direct_child(
                    diagnostics, name, "Diagnostic entry"
                )
                sources.append(source)
                destinations.append(
                    trash / f"{name}_diagnostic_{timestamp}"
                )
            return self._plan(
                action,
                selected,
                tuple(sources),
                tuple(destinations),
            )
        if action == "restore":
            if len(selected) != 1:
                raise ProjectLifecycleError(
                    "Restore accepts exactly one Trash item"
                )
            trash_root = _plain_directory(trash, "Project Trash")
            source = _direct_child(
                trash_root, selected[0], "Trash item"
            )
            parsed = _trash_kind(selected[0])
            if parsed is None or parsed[0].endswith("partial_delete"):
                raise ProjectLifecycleError(
                    "Partial or unrecognized Trash content cannot be restored"
                )
            kind, identity = parsed
            if kind == "result":
                document, source_records = self._validate_result_stably(
                    source, require_canonical_directory=False
                )
                if document.ready.result_id != identity:
                    raise ProjectLifecycleError(
                        "Trash Result identity is contradictory"
                    )
                if _path_occupied(
                    self.root / ".jobs" / document.ready.job_id
                ):
                    raise ProjectConflictError(
                        "Result restore conflicts with active or staging Job"
                    )
                target = _target_blend_path(
                    document.manifest["target_scene"]["blend_path"]
                )
                if os.path.normcase(str(target)) != os.path.normcase(
                    str(self.blend_path)
                ):
                    raise ProjectLifecycleError(
                        "Trash Result belongs to another Project"
                    )
                destination = (
                    _plain_directory(self._root() / "results", "results")
                    / _result_directory_name(document.manifest)
                )
                return self._plan(
                    action,
                    selected,
                    (source,),
                    (destination,),
                    result_id=identity,
                    records=(source_records,),
                )
            if kind == "dense":
                matches = []
                results = _plain_directory(
                    self._root() / "results", "results"
                )
                inspected = 0
                with os.scandir(results) as entries:
                    for entry in entries:
                        inspected += 1
                        if inspected > 10_000:
                            raise ProjectLifecycleError(
                                "Dense restore Result inventory exceeds "
                                "the explicit-action safety bound"
                            )
                        if (
                            entry.is_symlink()
                            or not entry.is_dir(follow_symlinks=False)
                            or not RESULT_DIRECTORY.fullmatch(entry.name)
                            or is_reparse_point(Path(entry.path))
                        ):
                            continue
                        try:
                            manifest = read_json(
                                Path(entry.path) / "manifest.json"
                            )
                        except (IpcError, OSError):
                            continue
                        if manifest.get("result_id") == identity:
                            matches.append((Path(entry.path), manifest))
                if len(matches) != 1:
                    raise ProjectConflictError(
                        "Dense restore requires one exact owning Result"
                    )
                result, manifest = matches[0]
                owner, owner_records = self._validate_result_stably(
                    result
                )
                if owner.ready.result_id != identity:
                    raise ProjectConflictError(
                        "Dense owning Result identity is contradictory"
                    )
                if _path_occupied(
                    self.root / ".jobs" / owner.ready.job_id
                ):
                    raise ProjectConflictError(
                        "Dense restore conflicts with active or staging Job"
                    )
                try:
                    target = _target_blend_path(
                        owner.manifest["target_scene"]["blend_path"]
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    raise ProjectLifecycleError(
                        "Dense owning Result identity is invalid"
                    ) from exc
                if os.path.normcase(str(target)) != os.path.normcase(
                    str(self.blend_path)
                ):
                    raise ProjectLifecycleError(
                        "Dense owning Result belongs to another Project"
                    )
                destination = result / "dense"
                if _path_occupied(destination):
                    raise ProjectConflictError(
                        "Dense Predictions destination already exists"
                    )
                descriptor = owner.manifest.get("dense_predictions")
                if not isinstance(descriptor, Mapping):
                    raise ProjectLifecycleError(
                        "Owning Result has no Dense descriptor"
                    )
                chunks, _bytes, dense_records = _validate_dense(
                    source, descriptor
                )
                if _tree_records(result) != owner_records:
                    raise ProjectRaceError(
                        "Dense owning Result changed during validation"
                    )
                return self._plan(
                    action,
                    selected,
                    (source,),
                    (destination,),
                    chunk_count=chunks,
                    result_id=identity,
                    records=(dense_records,),
                    guards=(result,),
                    guard_records=(owner_records,),
                )
            original = identity
            if not DIAGNOSTIC_NAME.fullmatch(original):
                raise ProjectLifecycleError(
                    "Diagnostic restore identity is invalid"
                )
            diagnostic_job_id = original.split("--", 1)[0]
            if _path_occupied(
                self.root / ".jobs" / diagnostic_job_id
            ):
                raise ProjectConflictError(
                    "Diagnostic restore conflicts with active or staging Job"
                )
            destination = (
                _plain_directory(
                    self._root() / "diagnostics", "diagnostics"
                )
                / original
            )
            return self._plan(
                action,
                selected,
                (source,),
                (destination,),
            )
        if action == "delete":
            trash_root = _plain_directory(trash, "Project Trash")
            for name in selected:
                if _trash_kind(name) is None:
                    raise ProjectLifecycleError(
                        "Unrecognized Trash entries have no lifecycle action"
                    )
            sources = tuple(
                _direct_child(trash_root, name, "Trash item")
                for name in selected
            )
            return self._plan(
                action,
                selected,
                sources,
                (),
            )
        raise ProjectLifecycleError(f"Unknown lifecycle action: {action}")

    def _revalidate(self, plan: LifecyclePlan) -> None:
        if plan.action != "delete" and any(
            _path_occupied(destination)
            for destination in plan.destinations
        ):
            raise ProjectConflictError(
                "Lifecycle destination appeared after confirmation"
            )
        current_records = tuple(
            _tree_records(source) for source in plan.sources
        )
        current_guard_records = tuple(
            _tree_records(guard) for guard in plan.guards
        )
        fingerprint = _records_fingerprint(
            plan.action,
            plan.sources,
            plan.destinations,
            current_records,
            plan.guards,
            current_guard_records,
        )
        if fingerprint != plan.fingerprint:
            raise ProjectRaceError(
                "Lifecycle content changed after confirmation"
            )

    def _ensure_trash(self) -> Path:
        root = self._root()
        trash = root / ".trash"
        trash.mkdir(exist_ok=True)
        return _plain_directory(trash, "Project Trash")

    @staticmethod
    def _mark_partial(path: Path) -> Path:
        if PARTIAL_SUFFIX.search(path.name):
            return path
        destination = path.with_name(path.name + "--partial_delete")
        if _path_occupied(destination):
            destination = path.with_name(
                path.name + "--partial_delete-" + os.urandom(4).hex()
            )
        os.rename(path, destination)
        return destination

    def _delete(
        self,
        plan: LifecyclePlan,
        *,
        confirmation: str,
        cancel: Callable[[], bool] | None,
    ) -> LifecycleOutcome:
        if confirmation != "DELETE":
            raise ProjectLifecycleError(
                "Permanent deletion requires typing DELETE"
            )
        session = self.begin_delete(plan, confirmation=confirmation)
        outcome = None
        while outcome is None:
            if cancel is not None and cancel():
                session.cancel()
            outcome = session.step()
        return outcome

    def begin_delete(
        self,
        plan: LifecyclePlan,
        *,
        confirmation: str,
    ) -> PermanentDeletionSession:
        if plan.action != "delete":
            raise ProjectLifecycleError(
                "Only a permanent-delete plan can begin deletion"
            )
        if confirmation != "DELETE":
            raise ProjectLifecycleError(
                "Permanent deletion requires typing DELETE"
            )
        self._revalidate(plan)
        return PermanentDeletionSession(self, plan)

    def execute(
        self,
        plan: LifecyclePlan,
        *,
        confirmation: str = "",
        cancel: Callable[[], bool] | None = None,
    ) -> LifecycleOutcome:
        if plan.action == "delete":
            return self._delete(
                plan, confirmation=confirmation, cancel=cancel
            )
        if plan.action.startswith("trash_"):
            self._ensure_trash()
        self._revalidate(plan)
        moved: list[tuple[Path, Path]] = []
        try:
            for source, destination in zip(
                plan.sources, plan.destinations, strict=True
            ):
                _plain_directory(
                    destination.parent, "Lifecycle destination parent"
                )
                if _path_occupied(destination):
                    raise ProjectConflictError(
                        "Lifecycle destination appeared before atomic move"
                    )
                os.rename(source, destination)
                moved.append((source, destination))
                observed = _tree_records(destination)
                expected = plan.records[len(moved) - 1]
                if observed != expected:
                    if (
                        not _path_occupied(source)
                        and _path_occupied(destination)
                    ):
                        os.rename(destination, source)
                    raise ProjectRaceError(
                        "Lifecycle source changed during atomic move"
                    )
        except Exception as exc:
            rollback_failures = []
            for source, destination in reversed(moved):
                if (
                    _path_occupied(destination)
                    and not _path_occupied(source)
                ):
                    try:
                        os.rename(destination, source)
                    except OSError as rollback_exc:
                        rollback_failures.append(
                            f"{destination.name}: {rollback_exc}"
                        )
            if rollback_failures:
                raise ProjectRaceError(
                    "Lifecycle rollback is incomplete; inspect both roots: "
                    + "; ".join(rollback_failures)
                ) from exc
            raise
        return LifecycleOutcome(
            plan.action, plan.item_count, plan.destinations
        )


__all__ = [
    "DISCOVERY_BATCH_LIMIT",
    "INVENTORY_PAGE_SIZE",
    "InventoryItem",
    "InventorySnapshot",
    "LifecycleOutcome",
    "LifecyclePlan",
    "MAX_JOB_ENTRIES",
    "PartialDeletionError",
    "PermanentDeletionSession",
    "ProjectConflictError",
    "ProjectInventory",
    "ProjectLifecycle",
    "ProjectLifecycleError",
    "ProjectRaceError",
]
