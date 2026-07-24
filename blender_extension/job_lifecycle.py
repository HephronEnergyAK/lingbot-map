"""Blender-side launcher, monitor, cancellation, and recovery for finite Jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Mapping
import uuid

from .gpu_capability import _runtime_command, _worker_environment
from .diagnostics import diagnostic_record
from .ipc import (
    IpcError,
    SCHEMA_VERSION,
    atomic_write_json,
    parse_json_line,
    read_json,
    require_exact_object,
    require_schema,
    require_text,
)
from .model_store import is_reparse_point
from .runtime_setup import RuntimeSetupError, process_identity, sha256_file


SCENE_UUID_PROPERTY = "lingbot_map_scene_uuid"
SCENE_UUID_SAVE_REQUIRED_PROPERTY = "lingbot_map_scene_uuid_save_required"
CAPTURE_SOURCE_PROPERTY = "lingbot_map_capture_source"
PROFILE_PROPERTY = "lingbot_map_profile"
CAMERA_ITERATIONS_PROPERTY = "lingbot_map_camera_iterations"
CONFIDENCE_CUTOFF_PROPERTY = "lingbot_map_confidence_cutoff_percent"
DEPTH_CUTOFF_PROPERTY = "lingbot_map_depth_cutoff_percent"
POINT_BUDGET_PROPERTY = "lingbot_map_import_point_budget"
POINT_BUDGET_CONFIRMED_PROPERTY = "lingbot_map_point_budget_confirmed"
RETAIN_DENSE_PROPERTY = "lingbot_map_retain_dense_predictions"
SKY_MASK_PROPERTY = "lingbot_map_sky_mask"
JOB_ID = re.compile(r"job-[0-9a-f]{32}\Z")
TERMINAL_STATES = {"succeeded", "cancelled", "failed"}
MAX_ACTIVE_ENTRIES = 32
MAX_DIAGNOSTIC_ENTRIES = 4096


class JobLifecycleError(RuntimeSetupError):
    pass


class StaleWorkerIdentity(JobLifecycleError):
    pass


@dataclass(frozen=True)
class WorkerRecord:
    pid: int
    creation_time: int
    executable: str
    executable_sha256: str
    nonce: str


@dataclass(frozen=True)
class JobSnapshot:
    state: str = "idle"
    message: str = "No active Reconstruction Job"
    job_id: str | None = None
    target_blend: str | None = None
    target_scene_uuid: str | None = None
    phase: str | None = None
    completed: int = 0
    total: int = 0
    eta_seconds: float | None = None
    heartbeat_sequence: int = 0
    location: str | None = None


@dataclass
class _ActiveJob:
    job_dir: Path
    record: WorkerRecord
    process: subprocess.Popen[bytes] | None
    target_blend: str
    target_scene_uuid: str
    stdout: Any = None
    reconnect_baseline: int | None = None
    reconnect_deadline: float | None = None
    last_heartbeat: int = 0
    last_heartbeat_observed: float = 0.0
    cancel_started: float | None = None
    forced: bool = False
    events_offset: int = 0


def normalized_blend_path(value: str | Path) -> Path:
    text = str(value)
    if not text:
        raise JobLifecycleError("Save the Blender file before launching a Job")
    path = Path(os.path.abspath(text))
    if not path.is_absolute() or path.suffix.lower() != ".blend":
        raise JobLifecycleError("Target Scene requires an absolute saved .blend path")
    return path


def normalized_capture_source(value: str | Path, blend_path: str | Path) -> Path:
    text = str(value).strip()
    if not text:
        raise JobLifecycleError("Choose one Capture Source before launching preflight")
    blend = normalized_blend_path(blend_path)
    if text.startswith("//"):
        text = str(blend.parent / Path(text[2:].replace("/", os.sep)))
    source = Path(os.path.abspath(text))
    if not source.is_absolute() or source.suffix.lower() not in {".mp4", ".mov"}:
        raise JobLifecycleError("Capture Source must be one local MP4 or MOV file")
    if not source.is_file() or source.is_symlink() or is_reparse_point(source):
        raise JobLifecycleError("Capture Source is absent or not an ordinary local file")
    return source


def scene_relative_capture_path(source: str | Path, blend_path: str | Path) -> str | None:
    absolute = Path(os.path.abspath(source))
    blend = normalized_blend_path(blend_path)
    try:
        relative = os.path.relpath(absolute, blend.parent)
    except ValueError:  # Different Windows drive: no scene-relative identity exists.
        return None
    return "//" + relative.replace(os.sep, "/")


def capture_source_draft_path(source: str | Path, blend_path: str | Path) -> str:
    absolute = normalized_capture_source(source, blend_path)
    return scene_relative_capture_path(absolute, blend_path) or str(absolute)


def project_result_root(blend_path: str | Path) -> Path:
    path = normalized_blend_path(blend_path)
    return path.with_name(f"{path.stem}.lingbot-map")


def ensure_unique_scene_uuid(
    scene: Any,
    scenes: list[Any] | tuple[Any, ...],
) -> str:
    value = str(scene.get(SCENE_UUID_PROPERTY, "")).strip()
    if not value:
        value = str(uuid.uuid4())
        scene[SCENE_UUID_PROPERTY] = value
        scene[SCENE_UUID_SAVE_REQUIRED_PROPERTY] = True
        raise JobLifecycleError(
            "A Scene UUID was assigned. Save the .blend, then launch the Job again."
        )
    if bool(scene.get(SCENE_UUID_SAVE_REQUIRED_PROPERTY, False)):
        raise JobLifecycleError(
            "Scene identity changed. Save the .blend before launch or automatic import."
        )
    try:
        canonical = str(uuid.UUID(value))
    except ValueError as exc:
        raise JobLifecycleError("Target Scene has an invalid LingBot Map UUID") from exc
    matches = 0
    for item in scenes:
        try:
            candidate = str(
                uuid.UUID(
                    str(item.get(SCENE_UUID_PROPERTY, "")).strip()
                )
            )
        except ValueError:
            continue
        if candidate == canonical:
            matches += 1
    if matches != 1:
        raise JobLifecycleError("Target Scene UUID is duplicated; repair it before launch")
    return canonical


def duplicate_scene_uuid_groups(
    scenes: list[Any] | tuple[Any, ...],
) -> dict[str, tuple[Any, ...]]:
    groups: dict[str, list[Any]] = {}
    for scene in scenes:
        raw = str(scene.get(SCENE_UUID_PROPERTY, "")).strip()
        if not raw:
            continue
        try:
            canonical = str(uuid.UUID(raw))
        except ValueError:
            continue
        groups.setdefault(canonical, []).append(scene)
    return {
        identifier: tuple(items)
        for identifier, items in groups.items()
        if len(items) > 1
    }


def repair_duplicate_scene_uuids(
    scenes: list[Any] | tuple[Any, ...],
    duplicate_uuid: str,
    keeper: Any,
) -> dict[str, str]:
    """Keep one explicitly selected identity and re-key every other duplicate."""

    try:
        canonical = str(uuid.UUID(str(duplicate_uuid)))
    except ValueError as exc:
        raise JobLifecycleError("Duplicate Scene UUID is invalid") from exc
    matches = []
    for scene in scenes:
        try:
            candidate = str(
                uuid.UUID(
                    str(scene.get(SCENE_UUID_PROPERTY, "")).strip()
                )
            )
        except ValueError:
            continue
        if candidate == canonical:
            matches.append(scene)
    matches = tuple(matches)
    if len(matches) < 2 or sum(scene is keeper for scene in matches) != 1:
        raise JobLifecycleError(
            "Choose exactly one Scene from the duplicate identity group as keeper"
        )
    repaired: dict[str, str] = {}
    for scene in matches:
        if scene is not keeper:
            replacement = str(uuid.uuid4())
            scene[SCENE_UUID_PROPERTY] = replacement
            repaired[str(getattr(scene, "name", ""))] = replacement
        scene[SCENE_UUID_SAVE_REQUIRED_PROPERTY] = True
    return repaired


def _require_plain(path: Path, *, label: str) -> None:
    if path.exists() and (path.is_symlink() or is_reparse_point(path)):
        raise JobLifecycleError(f"{label} must not be a symlink, junction, or reparse point")


def ensure_project_layout(blend_path: str | Path) -> Path:
    root = project_result_root(blend_path)
    _require_plain(root, label="Project Result Root")
    root.mkdir(exist_ok=True)
    for name in (".jobs", "results", "diagnostics"):
        child = root / name
        _require_plain(child, label=name)
        child.mkdir(exist_ok=True)
        if not child.is_dir():
            raise JobLifecycleError(f"Project Result Root child is not a directory: {name}")
    return root


def latest_successful_preflight(
    blend_path: str | Path, capture_source: str | Path
) -> Mapping[str, Any]:
    """Return the newest bounded successful preflight for the current source stat."""

    source = normalized_capture_source(capture_source, blend_path)
    root = ensure_project_layout(blend_path)
    entries = list((root / "diagnostics").iterdir())
    if len(entries) > MAX_DIAGNOSTIC_ENTRIES:
        raise JobLifecycleError("Too many diagnostic entries to discover preflight safely")
    candidates = []
    current = source.stat()
    for directory in entries:
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or is_reparse_point(directory)
            or not re.fullmatch(r"job-[0-9a-f]{32}--preflight-succeeded", directory.name)
        ):
            continue
        path = directory / "preflight-result.json"
        if not path.is_file() or path.is_symlink() or is_reparse_point(path):
            continue
        try:
            document = read_json(path)
            frozen = document["source"]
            timing = document["timing"]
            video = document["video"]
            color = video["color"]
            if (
                frozen["absolute_path"] == str(source)
                and frozen["size_bytes"] == current.st_size
                and frozen["modification_time_ns"] == current.st_mtime_ns
                and 8 <= int(timing["frame_count"])
            ):
                candidates.append((path.stat().st_mtime_ns, document))
        except (KeyError, TypeError, ValueError, OSError, IpcError):
            continue
    if not candidates:
        raise JobLifecycleError(
            "Run a successful preflight for this unchanged Capture Source before Reconstruction"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _active_children(root: Path) -> list[Path]:
    jobs = root / ".jobs"
    entries = list(jobs.iterdir())
    if len(entries) > MAX_ACTIVE_ENTRIES:
        raise JobLifecycleError(
            f"Project has {len(entries)} .jobs entries; automatic lifecycle actions are disabled"
        )
    return [item for item in entries if item.is_dir() and not item.is_symlink() and JOB_ID.fullmatch(item.name)]


def _integer(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IpcError(f"{label} is invalid")
    return value


def _validate_target(value: Any) -> dict[str, Any]:
    target = require_exact_object(value, {"blend_path", "scene_uuid", "scene_name"}, label="Target Scene")
    require_text(target["blend_path"], label="blend_path", maximum=32767)
    require_text(target["scene_uuid"], label="scene_uuid", maximum=64)
    require_text(target["scene_name"], label="scene_name", maximum=1024)
    return target


def validate_control(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "job_id", "job_spec", "runtime_id", "worker",
        "event_schema_version", "status_schema_version", "cancel_request_path", "target_scene",
    }
    control = require_schema(value, fields, label="Job Control")
    if not isinstance(control["job_id"], str) or not JOB_ID.fullmatch(control["job_id"]):
        raise IpcError("Job Control Job ID is invalid")
    spec = require_exact_object(control["job_spec"], {"path", "sha256"}, label="job_spec")
    if spec["path"] != "job-spec.json" or not re.fullmatch(r"[0-9a-f]{64}", str(spec["sha256"])):
        raise IpcError("Job Control JobSpec descriptor is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(control["runtime_id"])):
        raise IpcError("Job Control Runtime ID is invalid")
    if control["event_schema_version"] != SCHEMA_VERSION or control["status_schema_version"] != SCHEMA_VERSION:
        raise IpcError("Job Control stream version is unsupported")
    if control["cancel_request_path"] != "cancel.request":
        raise IpcError("Job Control cancel path is invalid")
    _validate_target(control["target_scene"])
    worker = require_exact_object(
        control["worker"], {"pid", "creation_time", "executable", "executable_sha256", "nonce"}, label="worker"
    )
    _integer(worker["pid"], "worker.pid", 1)
    _integer(worker["creation_time"], "worker.creation_time", 1)
    require_text(worker["executable"], label="worker.executable", maximum=32767)
    if not re.fullmatch(r"[0-9a-f]{64}", str(worker["executable_sha256"])):
        raise IpcError("Worker executable checksum is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", str(worker["nonce"])):
        raise IpcError("Worker nonce is invalid")
    return control


def validate_status(value: Any, job_id: str) -> dict[str, Any]:
    fields = {
        "schema_version", "job_id", "state", "phase", "heartbeat_sequence",
        "heartbeat_utc", "worker_monotonic", "progress_event_sequence", "progress", "error",
    }
    status = require_schema(value, fields, label="status")
    if status["job_id"] != job_id or status["state"] not in {"starting", "running", "cancelling", *TERMINAL_STATES}:
        raise IpcError("status identity or state is invalid")
    require_text(status["phase"], label="status.phase", maximum=256)
    _integer(status["heartbeat_sequence"], "heartbeat_sequence", 0)
    _integer(status["progress_event_sequence"], "progress_event_sequence", 0)
    if isinstance(status["worker_monotonic"], bool) or not isinstance(status["worker_monotonic"], (int, float)):
        raise IpcError("worker_monotonic is invalid")
    require_text(status["heartbeat_utc"], label="heartbeat_utc", maximum=128)
    progress = require_exact_object(
        status["progress"], {"completed", "total", "eta_seconds"}, label="progress"
    )
    _integer(progress["completed"], "progress.completed", 0)
    _integer(progress["total"], "progress.total", 0)
    if progress["completed"] > progress["total"]:
        raise IpcError("progress exceeds total")
    eta = progress["eta_seconds"]
    if eta is not None and (
        isinstance(eta, bool) or not isinstance(eta, (int, float))
        or not math.isfinite(float(eta)) or eta < 0
    ):
        raise IpcError("progress ETA is invalid")
    if status["error"] is not None and (not isinstance(status["error"], str) or len(status["error"].encode("utf-8")) > 16384):
        raise IpcError("status error text is invalid")
    return status


def validate_event(value: Any, job_id: str) -> dict[str, Any]:
    event = require_schema(
        value,
        {
            "schema_version", "job_id", "sequence", "kind", "phase", "completed",
            "total", "eta_seconds", "immediate", "message",
        },
        label="event",
    )
    if event["job_id"] != job_id or event["kind"] not in {"phase", "progress", "warning", "error", "cancelled", "succeeded"}:
        raise IpcError("event identity or kind is invalid")
    _integer(event["sequence"], "event.sequence", 1)
    _integer(event["completed"], "event.completed", 0)
    _integer(event["total"], "event.total", 0)
    if event["completed"] > event["total"]:
        raise IpcError("event progress exceeds total")
    eta = event["eta_seconds"]
    if eta is not None and (
        isinstance(eta, bool) or not isinstance(eta, (int, float))
        or not math.isfinite(float(eta)) or eta < 0
    ):
        raise IpcError("event ETA is invalid")
    if not isinstance(event["immediate"], bool):
        raise IpcError("event immediacy is invalid")
    require_text(event["phase"], label="event.phase", maximum=256)
    if not isinstance(event["message"], str) or len(event["message"].encode("utf-8")) > 16384:
        raise IpcError("event message is invalid")
    return event


def _record_from_control(control: Mapping[str, Any]) -> WorkerRecord:
    return WorkerRecord(**control["worker"])


def _validate_worker_record(value: Any, nonce: str) -> WorkerRecord:
    worker = require_exact_object(
        value, {"pid", "creation_time", "executable", "executable_sha256", "nonce"},
        label="worker.pid",
    )
    record = WorkerRecord(
        _integer(worker["pid"], "worker.pid", 1),
        _integer(worker["creation_time"], "worker.creation_time", 1),
        require_text(worker["executable"], label="worker.executable", maximum=32767),
        str(worker["executable_sha256"]),
        str(worker["nonce"]),
    )
    if not re.fullmatch(r"[0-9a-f]{64}", record.executable_sha256) or record.nonce != nonce:
        raise IpcError("Worker handshake checksum or nonce is invalid")
    return record


def _wait_for_worker_record(job_dir: Path, nonce: str, process: subprocess.Popen[bytes], timeout: float = 10.0) -> WorkerRecord:
    path = job_dir / "worker.pid.json"
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if process.poll() is not None:
            raise JobLifecycleError("Worker exited before publishing its process identity")
        if time.monotonic() >= deadline:
            raise JobLifecycleError("Worker did not publish its process identity within 10 seconds")
        time.sleep(0.025)
    record = _validate_worker_record(read_json(path), nonce)
    if not _same_worker(record, _observed_record(record)):
        raise StaleWorkerIdentity("Worker handshake does not match the live operating-system process")
    return record


def _observed_record(record: WorkerRecord) -> WorkerRecord | None:
    try:
        observed = process_identity(record.pid, record.nonce)
        checksum = sha256_file(Path(observed.executable))
    except (OSError, RuntimeSetupError):
        return None
    return WorkerRecord(
        observed.pid, observed.creation_time, observed.executable, checksum, record.nonce
    )


def _same_worker(expected: WorkerRecord, observed: WorkerRecord | None) -> bool:
    return bool(
        observed
        and observed.pid == expected.pid
        and observed.creation_time == expected.creation_time
        and os.path.normcase(observed.executable) == os.path.normcase(expected.executable)
        and observed.executable_sha256 == expected.executable_sha256
        and observed.nonce == expected.nonce
    )


def _atomic_sentinel(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _terminate_exact(expected: WorkerRecord) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        PROCESS_TERMINATE = 0x0001
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(
            PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, False, expected.pid
        )
        if not handle:
            raise JobLifecycleError(f"Cannot open exact Worker {expected.pid} for termination")
        try:
            created = wintypes.FILETIME()
            exited = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(
                handle, ctypes.byref(created), ctypes.byref(exited),
                ctypes.byref(kernel), ctypes.byref(user),
            ):
                raise StaleWorkerIdentity("Cannot revalidate Worker creation time on the termination handle")
            creation_time = (created.dwHighDateTime << 32) | created.dwLowDateTime
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                raise StaleWorkerIdentity("Cannot revalidate Worker executable on the termination handle")
            executable = str(Path(buffer.value).resolve())
            observed = WorkerRecord(
                expected.pid, creation_time, executable, sha256_file(Path(executable)), expected.nonce
            )
            if not _same_worker(expected, observed):
                raise StaleWorkerIdentity("Refusing to terminate a different process identity")
            if not kernel32.TerminateProcess(handle, 3):
                raise JobLifecycleError(f"Cannot terminate exact Worker {expected.pid}")
        finally:
            kernel32.CloseHandle(handle)
    else:
        import signal
        if not _same_worker(expected, _observed_record(expected)):
            raise StaleWorkerIdentity("Refusing to terminate a different process identity")
        os.kill(expected.pid, signal.SIGTERM)


class JobController:
    """Own at most one Worker lifecycle for this Blender process."""

    def __init__(self, *, heartbeat_window: float = 30.0, reconnect_window: float = 10.0, cancel_grace: float = 60.0):
        self.heartbeat_window = heartbeat_window
        self.reconnect_window = reconnect_window
        self.cancel_grace = cancel_grace
        self._lock = threading.Lock()
        self._snapshot = JobSnapshot()
        self._active: _ActiveJob | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return self._snapshot

    def _publish(self, snapshot: JobSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def has_active_job(self) -> bool:
        with self._lock:
            return self._active is not None

    def launch_fixture(
        self,
        *,
        managed_root: Path,
        blend_path: str | Path,
        scene_uuid: str,
        scene_name: str,
        timeline_start: int,
        steps: int = 40,
        step_delay_seconds: float = 0.05,
        ignore_cancel: bool = False,
        heartbeat_interval_seconds: float = 1.0,
        freeze_heartbeat_after_sequence: int | None = None,
    ) -> str:
        if self.has_active_job():
            raise JobLifecycleError("This Blender process already launched an active Worker")
        target = normalized_blend_path(blend_path)
        if not target.is_file():
            raise JobLifecycleError("Target Scene .blend file does not exist on disk")
        root = ensure_project_layout(target)
        active_entries = _active_children(root)
        if active_entries:
            raise JobLifecycleError("This Project Result Root already contains an active Job")
        runtime, python, runtime_id, _lock_sha = _runtime_command(managed_root)
        trusted_cwd = runtime / "empty-cwd"
        if not trusted_cwd.is_dir() or trusted_cwd.is_symlink() or any(trusted_cwd.iterdir()):
            raise JobLifecycleError("Worker Runtime trusted working directory is absent or not empty")
        job_id = f"job-{uuid.uuid4().hex}"
        job_dir = root / ".jobs" / job_id
        job_dir.mkdir()
        target_scene = {
            "blend_path": str(target), "scene_uuid": str(uuid.UUID(scene_uuid)), "scene_name": scene_name,
        }
        job_spec = {
            "schema_version": SCHEMA_VERSION, "job_id": job_id,
            "target_scene": target_scene, "timeline_start": int(timeline_start),
            "project_root": str(root),
            "fixture": {
                "steps": int(steps), "step_delay_seconds": float(step_delay_seconds),
                "ignore_cancel": bool(ignore_cancel),
                "heartbeat_interval_seconds": float(heartbeat_interval_seconds),
                "freeze_heartbeat_after_sequence": freeze_heartbeat_after_sequence,
            },
        }
        return self._launch_spec(
            runtime=runtime, python=python, runtime_id=runtime_id,
            trusted_cwd=trusted_cwd, job_dir=job_dir, job_spec=job_spec,
            worker_argument="--fixture-job", target=target,
            scene_uuid=scene_uuid, starting_message="Fixture Job is starting",
        )

    def launch_preflight(
        self,
        *,
        managed_root: Path,
        blend_path: str | Path,
        scene_uuid: str,
        scene_name: str,
        timeline_start: int,
        capture_draft_path: str,
    ) -> str:
        if self.has_active_job():
            raise JobLifecycleError("This Blender process already launched an active Worker")
        target = normalized_blend_path(blend_path)
        if not target.is_file():
            raise JobLifecycleError("Target Scene .blend file does not exist on disk")
        source = normalized_capture_source(capture_draft_path, target)
        frozen_stat = source.stat()
        if frozen_stat.st_size < 1:
            raise JobLifecycleError("Capture Source is empty")
        root = ensure_project_layout(target)
        if _active_children(root):
            raise JobLifecycleError("This Project Result Root already contains an active Job")
        runtime, python, runtime_id, _lock_sha = _runtime_command(managed_root)
        trusted_cwd = runtime / "empty-cwd"
        if not trusted_cwd.is_dir() or trusted_cwd.is_symlink() or any(trusted_cwd.iterdir()):
            raise JobLifecycleError("Worker Runtime trusted working directory is absent or not empty")
        job_id = f"job-{uuid.uuid4().hex}"
        job_dir = root / ".jobs" / job_id
        job_dir.mkdir()
        target_scene = {
            "blend_path": str(target), "scene_uuid": str(uuid.UUID(scene_uuid)), "scene_name": scene_name,
        }
        relative = scene_relative_capture_path(source, target)
        job_spec = {
            "schema_version": SCHEMA_VERSION, "job_id": job_id,
            "target_scene": target_scene, "timeline_start": int(timeline_start),
            "project_root": str(root),
            "capture_source": {
                "draft_path": capture_draft_path,
                "absolute_path": str(source),
                "scene_relative_path": relative,
                "size_bytes": int(frozen_stat.st_size),
                "modification_time_ns": int(frozen_stat.st_mtime_ns),
            },
        }
        return self._launch_spec(
            runtime=runtime, python=python, runtime_id=runtime_id,
            trusted_cwd=trusted_cwd, job_dir=job_dir, job_spec=job_spec,
            worker_argument="--preflight-job", target=target,
            scene_uuid=scene_uuid, starting_message="Capture Source preflight is starting",
        )

    def launch_result_fixture(
        self,
        *,
        managed_root: Path,
        blend_path: str | Path,
        scene_uuid: str,
        scene_name: str,
        timeline_start: int,
        capture_draft_path: str,
        confidence_cutoff_percent: float = 50.0,
        depth_cutoff_percent: float = 99.5,
        import_point_budget: int = 8,
        initial_voxel_edge_length: float = 0.01,
    ) -> str:
        """Launch the deterministic CPU Result fixture used by acceptance tests."""
        if self.has_active_job():
            raise JobLifecycleError("This Blender process already launched an active Worker")
        target = normalized_blend_path(blend_path)
        if not target.is_file():
            raise JobLifecycleError("Target Scene .blend file does not exist on disk")
        source = normalized_capture_source(capture_draft_path, target)
        frozen_stat = source.stat()
        if frozen_stat.st_size < 1:
            raise JobLifecycleError("Result Fixture source is empty")
        root = ensure_project_layout(target)
        if _active_children(root):
            raise JobLifecycleError("This Project Result Root already contains an active Job")
        runtime, python, runtime_id, _lock_sha = _runtime_command(managed_root)
        trusted_cwd = runtime / "empty-cwd"
        if not trusted_cwd.is_dir() or trusted_cwd.is_symlink() or any(trusted_cwd.iterdir()):
            raise JobLifecycleError("Worker Runtime trusted working directory is absent or not empty")
        job_id = f"job-{uuid.uuid4().hex}"
        job_dir = root / ".jobs" / job_id
        job_dir.mkdir()
        target_scene = {
            "blend_path": str(target),
            "scene_uuid": str(uuid.UUID(scene_uuid)),
            "scene_name": scene_name,
        }
        job_spec = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "target_scene": target_scene,
            "timeline_start": int(timeline_start),
            "project_root": str(root),
            "result_fixture": {
                "absolute_path": str(source),
                "scene_relative_path": scene_relative_capture_path(source, target),
                "size_bytes": int(frozen_stat.st_size),
                "modification_time_ns": int(frozen_stat.st_mtime_ns),
                "confidence_cutoff_percent": float(confidence_cutoff_percent),
                "depth_cutoff_percent": float(depth_cutoff_percent),
                "import_point_budget": int(import_point_budget),
                "initial_voxel_edge_length": float(initial_voxel_edge_length),
                "heartbeat_interval_seconds": 0.1,
            },
        }
        return self._launch_spec(
            runtime=runtime,
            python=python,
            runtime_id=runtime_id,
            trusted_cwd=trusted_cwd,
            job_dir=job_dir,
            job_spec=job_spec,
            worker_argument="--result-fixture-job",
            target=target,
            scene_uuid=scene_uuid,
            starting_message="Result Fixture Job is starting",
        )

    def launch_reconstruction(
        self,
        *,
        managed_root: Path,
        blend_path: str | Path,
        scene_uuid: str,
        scene_name: str,
        timeline_start: int,
        capture_draft_path: str,
        profile_name: str,
        camera_iterations: int,
        confidence_cutoff_percent: float,
        depth_cutoff_percent: float,
        import_point_budget: int,
        point_budget_confirmed: bool,
        retain_dense_predictions: bool,
        gpu: Mapping[str, Any],
        capability_profile_name: str,
        capability_profile_settings_sha256: str,
        model: Mapping[str, Any],
        preflight_result: Mapping[str, Any],
        sky_mask_enabled: bool = False,
        auxiliary_model: Mapping[str, Any] | None = None,
        initial_voxel_edge_length: float = 0.01,
    ) -> str:
        if self.has_active_job():
            raise JobLifecycleError("This Blender process already launched an active Worker")
        target = normalized_blend_path(blend_path)
        source = normalized_capture_source(capture_draft_path, target)
        root = ensure_project_layout(target)
        if _active_children(root):
            raise JobLifecycleError("This Project Result Root already contains an active Job")
        runtime, python, runtime_id, lock_sha = _runtime_command(managed_root)
        trusted_cwd = runtime / "empty-cwd"
        if not trusted_cwd.is_dir() or trusted_cwd.is_symlink() or any(trusted_cwd.iterdir()):
            raise JobLifecycleError("Worker Runtime trusted working directory is absent or not empty")
        frozen = preflight_result["source"]
        timing = preflight_result["timing"]
        video = preflight_result["video"]
        color = video["color"]
        stat = source.stat()
        if (
            frozen["absolute_path"] != str(source)
            or frozen["size_bytes"] != stat.st_size
            or frozen["modification_time_ns"] != stat.st_mtime_ns
        ):
            raise JobLifecycleError("Successful preflight does not identify the current Capture Source")
        if not isinstance(sky_mask_enabled, bool):
            raise JobLifecycleError("Sky Mask choice must be boolean")
        if sky_mask_enabled:
            if not isinstance(auxiliary_model, Mapping):
                raise JobLifecycleError(
                    "Enabled Sky Masking requires a verified Auxiliary Model"
                )
            auxiliary_document = dict(auxiliary_model)
            if (
                set(auxiliary_document)
                != {"catalog_version", "id", "path", "sha256"}
                or auxiliary_document["id"] != "skyseg"
            ):
                raise JobLifecycleError("Sky Mask Auxiliary Model identity is invalid")
            sky_mask = {"enabled": True, "model": auxiliary_document}
        else:
            if auxiliary_model is not None:
                raise JobLifecycleError(
                    "Disabled Sky Masking cannot name an Auxiliary Model"
                )
            sky_mask = {"enabled": False}
        job_id = f"job-{uuid.uuid4().hex}"
        job_dir = root / ".jobs" / job_id
        job_dir.mkdir()
        target_scene = {
            "blend_path": str(target),
            "scene_uuid": str(uuid.UUID(scene_uuid)),
            "scene_name": scene_name,
        }
        job_spec = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "target_scene": target_scene,
            "timeline_start": int(timeline_start),
            "project_root": str(root),
            "reconstruction": {
                "managed_root": str(Path(os.path.abspath(managed_root))),
                "worker_lock_sha256": lock_sha,
                "source": {
                    "absolute_path": str(source),
                    "scene_relative_path": scene_relative_capture_path(source, target),
                    "size_bytes": int(stat.st_size),
                    "modification_time_ns": int(stat.st_mtime_ns),
                    "sha256": frozen["sha256"],
                },
                "preflight": {
                    "frame_count": int(timing["frame_count"]),
                    "video_stream_index": int(video["stream_index"]),
                    "displayed_width": int(video["displayed_width"]),
                    "displayed_height": int(video["displayed_height"]),
                    "display_transform": video["display_transform"],
                    "color_standard": color["standard"],
                    "color_range": color["range"],
                    "variable_frame_rate": bool(timing["variable_frame_rate"]),
                },
                "profile": {
                    "name": profile_name,
                    "camera_iterations": int(camera_iterations),
                    "confidence_cutoff_percent": float(confidence_cutoff_percent),
                    "depth_cutoff_percent": float(depth_cutoff_percent),
                    "import_point_budget": int(import_point_budget),
                    "point_budget_confirmed": bool(point_budget_confirmed),
                    "retain_dense_predictions": bool(retain_dense_predictions),
                },
                "gpu": {
                    "uuid": gpu["uuid"],
                    "name": gpu["name"],
                    "total_memory": int(gpu["total_memory"]),
                    "driver_version": gpu["driver_version"],
                    "compute_capability": list(gpu["compute_capability"]),
                    "capability_profile_name": capability_profile_name,
                    "capability_profile_settings_sha256": capability_profile_settings_sha256,
                },
                "model": dict(model),
                "sky_mask": sky_mask,
                "heartbeat_interval_seconds": 1.0,
                "initial_voxel_edge_length": float(initial_voxel_edge_length),
            },
        }
        return self._launch_spec(
            runtime=runtime,
            python=python,
            runtime_id=runtime_id,
            trusted_cwd=trusted_cwd,
            job_dir=job_dir,
            job_spec=job_spec,
            worker_argument="--reconstruction-job",
            target=target,
            scene_uuid=scene_uuid,
            starting_message="Reconstruction Job is starting",
        )

    def _launch_spec(
        self,
        *,
        runtime: Path,
        python: Path,
        runtime_id: str,
        trusted_cwd: Path,
        job_dir: Path,
        job_spec: dict[str, Any],
        worker_argument: str,
        target: Path,
        scene_uuid: str,
        starting_message: str,
    ) -> str:
        job_id = str(job_spec["job_id"])
        target_scene = job_spec["target_scene"]
        spec_path = job_dir / "job-spec.json"
        atomic_write_json(spec_path, job_spec)
        nonce = uuid.uuid4().hex
        process = None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        environment = _worker_environment()
        if "reconstruction" in job_spec:
            environment["CUDA_VISIBLE_DEVICES"] = job_spec["reconstruction"]["gpu"]["uuid"]
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        process = subprocess.Popen(
            [str(python), "-I", "-m", "lingbot_map_worker", worker_argument, str(spec_path), "--job-nonce", nonce],
            cwd=trusted_cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            # Fixture failures are structured in status/events. An inherited
            # staging-local stderr handle would forbid atomic directory rename
            # on Windows and couple terminal publication to Blender lifetime.
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        record = None
        try:
            record = _wait_for_worker_record(job_dir, nonce, process)
            control = {
                "schema_version": SCHEMA_VERSION, "job_id": job_id,
                "job_spec": {"path": "job-spec.json", "sha256": sha256_file(spec_path)},
                "runtime_id": runtime_id, "worker": asdict(record),
                "event_schema_version": SCHEMA_VERSION, "status_schema_version": SCHEMA_VERSION,
                "cancel_request_path": "cancel.request", "target_scene": target_scene,
            }
            atomic_write_json(job_dir / "job-control.json", control)
        except Exception:
            if record is not None and _same_worker(record, _observed_record(record)):
                _terminate_exact(record)
            elif process.poll() is None:
                process.kill(); process.wait(timeout=5)
            self._recover_directory(job_dir, "launch-failed")
            raise
        active = _ActiveJob(
            job_dir, record, process, str(target), scene_uuid, process.stdout,
            last_heartbeat_observed=time.monotonic(),
        )
        with self._lock:
            self._active = active
            self._snapshot = JobSnapshot("starting", starting_message, job_id, str(target), scene_uuid, "starting", location=str(job_dir))
        self._start_threads(active)
        return job_id

    def _start_threads(self, active: _ActiveJob) -> None:
        self._stop.clear()
        if active.stdout is not None:
            self._stdout_thread = threading.Thread(target=self._consume_stdout, args=(active,), name="LingBotMap-Job-Stdout", daemon=True)
            self._stdout_thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor, args=(active,), name="LingBotMap-Job-Monitor", daemon=True)
        self._monitor_thread.start()

    def _consume_stdout(self, active: _ActiveJob) -> None:
        try:
            while not self._stop.is_set():
                line = active.stdout.readline()
                if not line:
                    return
                validate_event(parse_json_line(line), active.job_dir.name)
        except (OSError, IpcError) as exc:
            self._publish(
                JobSnapshot(
                    "protocol_error", str(exc), active.job_dir.name,
                    active.target_blend, active.target_scene_uuid,
                    location=str(active.job_dir),
                )
            )

    def _monitor(self, active: _ActiveJob) -> None:
        transient_read_started: float | None = None
        while not self._stop.wait(0.1):
            now = time.monotonic()
            terminal = self._terminal_directory(active.job_dir)
            if terminal is not None:
                self._publish_terminal(terminal, active.job_dir.name)
                self._finish_active(active)
                return
            if active.job_dir.exists():
                try:
                    self._tail_events(active, terminal=False)
                except IpcError as exc:
                    self._publish(JobSnapshot("protocol_error", str(exc), active.job_dir.name, active.target_blend, active.target_scene_uuid, location=str(active.job_dir)))
                    return
                status_path = active.job_dir / "status.json"
                if status_path.is_file():
                    try:
                        status = validate_status(read_json(status_path), active.job_dir.name)
                    except IpcError as exc:
                        if self._terminal_directory(active.job_dir) is not None or not active.job_dir.exists():
                            continue
                        if str(exc).startswith("cannot read IPC file"):
                            transient_read_started = transient_read_started or now
                            if now - transient_read_started < 1.0:
                                continue
                        self._publish(JobSnapshot("protocol_error", str(exc), active.job_dir.name, active.target_blend, active.target_scene_uuid, location=str(active.job_dir)))
                        return
                    transient_read_started = None
                    heartbeat = status["heartbeat_sequence"]
                    if heartbeat > active.last_heartbeat:
                        active.last_heartbeat = heartbeat
                        active.last_heartbeat_observed = now
                    if active.reconnect_baseline is not None:
                        if heartbeat > active.reconnect_baseline:
                            active.reconnect_baseline = None
                            active.reconnect_deadline = None
                        elif now >= (active.reconnect_deadline or now):
                            self._publish(self._snapshot_from_status(status, "unresponsive", "Worker did not publish a higher heartbeat while reconnecting", active))
                            active.reconnect_baseline = None
                            continue
                        else:
                            self._publish(self._snapshot_from_status(status, "reconnecting", "Waiting for a newly observed heartbeat", active))
                            continue
                    if status["state"] in TERMINAL_STATES:
                        self._publish(self._snapshot_from_status(status, status["state"], f"Job {status['state']}", active))
                    elif now - active.last_heartbeat_observed >= self.heartbeat_window:
                        self._publish(self._snapshot_from_status(status, "unresponsive", "Worker heartbeat was not observed within the liveness window", active))
                    else:
                        self._publish(self._snapshot_from_status(status, status["state"], f"Job {status['state']}", active))
            if active.cancel_started is not None and now - active.cancel_started >= self.cancel_grace:
                observed = _observed_record(active.record)
                if not _same_worker(active.record, observed):
                    self._publish(JobSnapshot("stale_identity", "Cancellation grace expired, but exact Worker identity no longer matches", active.job_dir.name, active.target_blend, active.target_scene_uuid, location=str(active.job_dir)))
                    active.cancel_started = None
                else:
                    _terminate_exact(active.record)
                    active.forced = True
                    active.cancel_started = None
            if not _same_worker(active.record, _observed_record(active.record)):
                if active.job_dir.exists():
                    destination = self._recover_directory(active.job_dir, "forced-termination" if active.forced else "interrupted")
                    self._publish(JobSnapshot("forced_termination" if active.forced else "interrupted", "Dead Worker staging was retained as diagnostics", active.job_dir.name, active.target_blend, active.target_scene_uuid, location=str(destination)))
                self._finish_active(active)
                return

    @staticmethod
    def _snapshot_from_status(status: Mapping[str, Any], state: str, message: str, active: _ActiveJob) -> JobSnapshot:
        progress = status["progress"]
        return JobSnapshot(
            state, message, status["job_id"], active.target_blend,
            active.target_scene_uuid, status["phase"], progress["completed"],
            progress["total"], progress["eta_seconds"], status["heartbeat_sequence"],
            str(active.job_dir),
        )

    def _publish_terminal(self, directory: Path, job_id: str) -> None:
        try:
            self._validate_complete_events(directory / "events.jsonl", job_id)
            status = validate_status(read_json(directory / "status.json"), job_id)
            control = validate_control(read_json(directory / "job-control.json"))
        except IpcError as exc:
            self._publish(JobSnapshot("protocol_error", str(exc), job_id, location=str(directory)))
        else:
            target = control["target_scene"]
            active = _ActiveJob(
                directory, _record_from_control(control), None,
                target["blend_path"], target["scene_uuid"],
            )
            snapshot = self._snapshot_from_status(status, status["state"], f"Job {status['state']}", active)
            self._publish(JobSnapshot(**{**asdict(snapshot), "location": str(directory)}))

    @staticmethod
    def _tail_events(active: _ActiveJob, *, terminal: bool, maximum_records: int = 50) -> None:
        path = active.job_dir / "events.jsonl"
        if not path.exists():
            return
        if not path.is_file() or path.is_symlink():
            raise IpcError("events.jsonl is not an ordinary file")
        records = 0
        try:
            with path.open("rb") as stream:
                stream.seek(active.events_offset)
                while records < maximum_records:
                    line = stream.readline(64 * 1024 + 1)
                    if not line:
                        return
                    if len(line) > 64 * 1024:
                        raise IpcError("JSON Lines record exceeds 65536 bytes")
                    if not line.endswith(b"\n"):
                        if terminal:
                            raise IpcError("terminal events.jsonl has an incomplete final line")
                        return
                    validate_event(parse_json_line(line), active.job_dir.name)
                    active.events_offset += len(line)
                    records += 1
        except OSError as exc:
            if not active.job_dir.exists():
                return
            raise IpcError(f"cannot tail events.jsonl: {exc}") from exc

    @staticmethod
    def _validate_complete_events(path: Path, job_id: str) -> None:
        if not path.is_file() or path.is_symlink():
            raise IpcError("terminal events.jsonl is absent or not ordinary")
        with path.open("rb") as stream:
            while True:
                line = stream.readline(64 * 1024 + 1)
                if not line:
                    return
                if len(line) > 64 * 1024:
                    raise IpcError("JSON Lines record exceeds 65536 bytes")
                validate_event(parse_json_line(line), job_id)

    @staticmethod
    def _terminal_directory(job_dir: Path) -> Path | None:
        diagnostics = job_dir.parent.parent / "diagnostics"
        if not diagnostics.is_dir():
            return None
        matches = [
            item for item in diagnostics.iterdir()
            if item.is_dir() and not item.is_symlink() and not is_reparse_point(item)
            and item.name.startswith(f"{job_dir.name}--")
        ]
        return matches[0] if len(matches) == 1 else None

    def _finish_active(self, active: _ActiveJob) -> None:
        with self._lock:
            if self._active is active:
                self._active = None

    @staticmethod
    def _recover_directory(job_dir: Path, reason: str) -> Path:
        diagnostics = job_dir.parent.parent / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        destination = diagnostics / f"{job_dir.name}--{reason}"
        if destination.exists():
            destination = diagnostics / f"{job_dir.name}--{reason}-{uuid.uuid4().hex[:8]}"
        os.replace(job_dir, destination)
        atomic_write_json(destination / "recovery.json", {"schema_version": SCHEMA_VERSION, "job_id": job_dir.name, "reason": reason})
        category = "launch" if reason == "launch-failed" else "lifecycle"
        try:
            spec = read_json(destination / "job-spec.json")
            target_scene = (
                spec.get("target_scene", {})
                if isinstance(spec, dict)
                else {}
            )
        except (IpcError, OSError):
            target_scene = {}
        if not (destination / "diagnostic.json").exists():
            atomic_write_json(
                destination / "diagnostic.json",
                diagnostic_record(
                    error_code=(
                        "launch.worker.failed"
                        if reason == "launch-failed"
                        else f"lifecycle.worker.{reason}"
                    ),
                    category=category,
                    state=(
                        "forced_termination"
                        if reason == "forced-termination"
                        else (
                            "interrupted"
                            if reason == "interrupted"
                            else "failed"
                        )
                    ),
                    phase=reason,
                    job_id=job_dir.name,
                    target_scene=target_scene,
                    detail=reason,
                ),
            )
        return destination

    def request_cancel(self) -> bool:
        with self._lock:
            active = self._active
        if active is None:
            return False
        control = validate_control(read_json(active.job_dir / "job-control.json"))
        if _record_from_control(control) != active.record or not _same_worker(active.record, _observed_record(active.record)):
            raise StaleWorkerIdentity("Refusing to cancel because the complete Worker identity does not match")
        _atomic_sentinel(active.job_dir / "cancel.request")
        if active.cancel_started is None:
            active.cancel_started = time.monotonic()
        self._publish(JobSnapshot("cancelling", "Cancellation requested; waiting for the graceful Worker exit", active.job_dir.name, active.target_blend, active.target_scene_uuid, location=str(active.job_dir)))
        return True

    def observe_reload(self) -> None:
        with self._lock:
            active = self._active
        if active is None:
            return
        status_path = active.job_dir / "status.json"
        baseline = 0
        if status_path.is_file():
            baseline = validate_status(read_json(status_path), active.job_dir.name)["heartbeat_sequence"]
        active.reconnect_baseline = baseline
        active.reconnect_deadline = time.monotonic() + self.reconnect_window
        active.last_heartbeat = baseline
        active.last_heartbeat_observed = time.monotonic()
        self._publish(
            JobSnapshot(
                "reconnecting", "Waiting for a newly observed heartbeat",
                active.job_dir.name, active.target_blend, active.target_scene_uuid,
                heartbeat_sequence=baseline, location=str(active.job_dir),
            )
        )

    def recover_project(self, blend_path: str | Path) -> None:
        if self.has_active_job():
            if self._monitor_thread is None or not self._monitor_thread.is_alive():
                with self._lock:
                    active = self._active
                if active is not None:
                    self._start_threads(active)
            self.observe_reload()
            return
        root = ensure_project_layout(blend_path)
        entries = _active_children(root)
        if len(entries) > 1:
            raise JobLifecycleError("Multiple active Job records require manual inspection")
        if not entries:
            return
        job_dir = entries[0]
        raw_control = read_json(job_dir / "job-control.json")
        schema_version = (
            raw_control.get("schema_version")
            if isinstance(raw_control, dict)
            else None
        )
        if schema_version != SCHEMA_VERSION:
            match = (
                re.fullmatch(r"([0-9]+)\.[0-9]+\.[0-9]+", schema_version)
                if isinstance(schema_version, str)
                else None
            )
            if match is not None and int(match.group(1)) != 1:
                self._publish(
                    JobSnapshot(
                        "legacy_running",
                        "Legacy Job Running; use a compatible Extension to monitor or cancel it",
                        job_dir.name,
                        location=str(job_dir),
                    )
                )
                return
        control = validate_control(raw_control)
        if control["job_id"] != job_dir.name:
            raise IpcError("active directory and Job Control identity differ")
        record = _record_from_control(control)
        if not _same_worker(record, _observed_record(record)):
            destination = self._recover_directory(job_dir, "interrupted")
            self._publish(JobSnapshot("interrupted", "Dead Worker staging was retained as diagnostics", job_dir.name, location=str(destination)))
            return
        target = control["target_scene"]
        active = _ActiveJob(
            job_dir, record, None, target["blend_path"], target["scene_uuid"],
            last_heartbeat_observed=time.monotonic(),
        )
        with self._lock:
            self._active = active
        self._start_threads(active)
        self.observe_reload()

    def detach(self) -> None:
        """Stop Blender monitoring without signalling or cancelling the finite Worker."""
        self._stop.set()
        with self._lock:
            active = self._active
        if active is not None and active.stdout is not None:
            try:
                active.stdout.close()
            except OSError:
                pass
            active.stdout = None
        for thread in (self._stdout_thread, self._monitor_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2)
        self._stdout_thread = None
        self._monitor_thread = None


_controller = JobController()


def get_job_snapshot() -> JobSnapshot:
    return _controller.snapshot()


def start_fixture_job(**kwargs) -> str:
    return _controller.launch_fixture(**kwargs)


def start_preflight_job(**kwargs) -> str:
    return _controller.launch_preflight(**kwargs)


def start_result_fixture_job(**kwargs) -> str:
    return _controller.launch_result_fixture(**kwargs)


def start_reconstruction_job(**kwargs) -> str:
    return _controller.launch_reconstruction(**kwargs)


def cancel_active_job() -> bool:
    return _controller.request_cancel()


def observe_job_reload() -> None:
    _controller.observe_reload()


def recover_jobs_for_blend(blend_path: str | Path) -> None:
    _controller.recover_project(blend_path)


def detach_job_monitor() -> None:
    _controller.detach()


def report_job_recovery_error(error: Exception) -> None:
    _controller._publish(JobSnapshot("protocol_error", str(error)))
