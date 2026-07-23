"""Deterministic, inference-free Worker Job used to qualify lifecycle control."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

from .gpu_lease import WindowsProcessProbe, sha256_file
from .ipc import (
    IpcError,
    MAX_EVENT_LINE_BYTES,
    SCHEMA_VERSION,
    atomic_write_json,
    encode_json,
    read_json,
    require_exact_object,
    require_schema,
    require_text,
)


COMMON_JOB_FIELDS = {
    "schema_version", "job_id", "target_scene", "timeline_start", "project_root"
}
CONTROL_FIELDS = {
    "schema_version", "job_id", "job_spec", "runtime_id", "worker",
    "event_schema_version", "status_schema_version", "cancel_request_path", "target_scene",
}
STATUS_FIELDS = {
    "schema_version", "job_id", "state", "phase", "heartbeat_sequence",
    "heartbeat_utc", "worker_monotonic", "progress_event_sequence", "progress", "error",
}
EVENT_FIELDS = {
    "schema_version", "job_id", "sequence", "kind", "phase", "completed", "total", "message",
}
TERMINAL_STATES = {"succeeded", "cancelled", "failed"}


class FixtureJobError(RuntimeError):
    pass


def _integer(value: Any, label: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IpcError(f"{label} is invalid")
    if maximum is not None and value > maximum:
        raise IpcError(f"{label} is invalid")
    return value


def _finite(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IpcError(f"{label} is invalid")
    result = float(value)
    if not minimum <= result <= maximum:
        raise IpcError(f"{label} is invalid")
    return result


def _target(value: Any) -> dict[str, Any]:
    target = require_exact_object(
        value, {"blend_path", "scene_uuid", "scene_name"}, label="Target Scene"
    )
    require_text(target["blend_path"], label="blend_path", maximum=32767)
    require_text(target["scene_uuid"], label="scene_uuid", maximum=64)
    require_text(target["scene_name"], label="scene_name", maximum=1024)
    if not Path(target["blend_path"]).is_absolute():
        raise IpcError("Target blend path must be absolute")
    try:
        import uuid
        uuid.UUID(target["scene_uuid"])
    except ValueError as exc:
        raise IpcError("Target Scene UUID is invalid") from exc
    return target


def validate_job_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IpcError("JobSpec has unknown or missing fields")
    job_kinds = {name for name in ("fixture", "capture_source") if name in value}
    if len(job_kinds) != 1 or set(value) != COMMON_JOB_FIELDS | job_kinds:
        raise IpcError("JobSpec must contain exactly one supported Job kind")
    job = require_schema(value, COMMON_JOB_FIELDS | job_kinds, label="JobSpec")
    if not re.fullmatch(r"job-[0-9a-f]{32}", require_text(job["job_id"], label="job_id", maximum=64)):
        raise IpcError("JobSpec Job ID is invalid")
    _target(job["target_scene"])
    _integer(job["timeline_start"], "timeline_start", -1048574, 1048574)
    root = Path(require_text(job["project_root"], label="project_root", maximum=32767))
    if not root.is_absolute():
        raise IpcError("Project Result Root must be absolute")
    if "fixture" in job:
        fixture = require_exact_object(
            job["fixture"],
            {"steps", "step_delay_seconds", "ignore_cancel", "heartbeat_interval_seconds", "freeze_heartbeat_after_sequence"},
            label="fixture",
        )
        _integer(fixture["steps"], "fixture.steps", 1, 100000)
        _finite(fixture["step_delay_seconds"], "fixture.step_delay_seconds", 0, 60)
        if not isinstance(fixture["ignore_cancel"], bool):
            raise IpcError("fixture.ignore_cancel is invalid")
        _finite(fixture["heartbeat_interval_seconds"], "fixture.heartbeat_interval_seconds", 0.001, 5)
        freeze = fixture["freeze_heartbeat_after_sequence"]
        if freeze is not None:
            _integer(freeze, "fixture.freeze_heartbeat_after_sequence", 1)
    else:
        capture = require_exact_object(
            job["capture_source"],
            {"draft_path", "absolute_path", "scene_relative_path", "size_bytes", "modification_time_ns"},
            label="capture_source",
        )
        require_text(capture["draft_path"], label="capture_source.draft_path", maximum=32767)
        absolute = Path(require_text(capture["absolute_path"], label="capture_source.absolute_path", maximum=32767))
        if not absolute.is_absolute() or absolute.suffix.lower() not in {".mp4", ".mov"}:
            raise IpcError("Capture Source absolute path is invalid")
        relative = capture["scene_relative_path"]
        if relative is not None and (
            not isinstance(relative, str) or not relative.startswith("//")
            or len(relative.encode("utf-8")) > 32767
        ):
            raise IpcError("Capture Source scene-relative path is invalid")
        _integer(capture["size_bytes"], "capture_source.size_bytes", 1)
        _integer(capture["modification_time_ns"], "capture_source.modification_time_ns", 1)
    return job


def validate_control(value: Any, job: dict[str, Any]) -> dict[str, Any]:
    control = require_schema(value, CONTROL_FIELDS, label="Job Control")
    if control["job_id"] != job["job_id"] or control["target_scene"] != job["target_scene"]:
        raise IpcError("Job Control identity does not match JobSpec")
    if control["event_schema_version"] != SCHEMA_VERSION or control["status_schema_version"] != SCHEMA_VERSION:
        raise IpcError("Job Control names an unsupported stream schema")
    if control["cancel_request_path"] != "cancel.request":
        raise IpcError("Job Control cancel path is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", require_text(control["runtime_id"], label="runtime_id", maximum=64)):
        raise IpcError("Runtime ID is invalid")
    spec = require_exact_object(control["job_spec"], {"path", "sha256"}, label="job_spec")
    if spec["path"] != "job-spec.json" or not re.fullmatch(r"[0-9a-f]{64}", require_text(spec["sha256"], label="job_spec.sha256", maximum=64)):
        raise IpcError("Job Control JobSpec descriptor is invalid")
    worker = require_exact_object(
        control["worker"], {"pid", "creation_time", "executable", "executable_sha256", "nonce"}, label="worker"
    )
    _integer(worker["pid"], "worker.pid", 1)
    _integer(worker["creation_time"], "worker.creation_time", 1)
    executable = Path(require_text(worker["executable"], label="worker.executable", maximum=32767))
    if not executable.is_absolute():
        raise IpcError("Worker executable path is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", require_text(worker["executable_sha256"], label="worker.executable_sha256", maximum=64)):
        raise IpcError("Worker executable checksum is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", require_text(worker["nonce"], label="worker.nonce", maximum=64)):
        raise IpcError("Worker nonce is invalid")
    return control


class StatusStore:
    """The only status/event writer; producers serialize through one lock."""

    def __init__(self, job_dir: Path, job_id: str, total: int):
        self.job_dir = job_dir
        self.job_id = job_id
        self.total = total
        self.status_path = job_dir / "status.json"
        self.events_path = job_dir / "events.jsonl"
        self._lock = threading.Lock()
        self._heartbeat_sequence = 0
        self._event_sequence = 0
        self._state = "starting"
        self._phase = "starting"
        self._completed = 0
        self._error: str | None = None
        self._terminal = False
        self._write_status_locked()

    def _document_locked(self) -> dict[str, object]:
        document = {
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "state": self._state,
            "phase": self._phase,
            "heartbeat_sequence": self._heartbeat_sequence,
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "worker_monotonic": time.monotonic(),
            "progress_event_sequence": self._event_sequence,
            "progress": {"completed": self._completed, "total": self.total},
            "error": self._error,
        }
        require_schema(document, STATUS_FIELDS, label="status")
        return document

    def _write_status_locked(self) -> None:
        atomic_write_json(self.status_path, self._document_locked())

    def heartbeat(self, freeze_after: int | None) -> bool:
        with self._lock:
            if self._terminal:
                return False
            if freeze_after is not None and self._heartbeat_sequence >= freeze_after:
                return True
            self._heartbeat_sequence += 1
            self._write_status_locked()
            return True

    def emit(self, kind: str, phase: str, completed: int, message: str, *, state: str = "running", error: str | None = None) -> None:
        with self._lock:
            self._event_sequence += 1
            self._state = state
            self._phase = require_text(phase, label="event.phase", maximum=256)
            self._completed = _integer(completed, "event.completed", 0, self.total)
            self._error = error
            event = {
                "schema_version": SCHEMA_VERSION, "job_id": self.job_id,
                "sequence": self._event_sequence, "kind": kind, "phase": phase,
                "completed": completed, "total": self.total, "message": message,
            }
            require_schema(event, EVENT_FIELDS, label="event")
            line = encode_json(event) + b"\n"
            if len(line) > MAX_EVENT_LINE_BYTES:
                raise IpcError("event line exceeds its limit")
            with self.events_path.open("ab") as stream:
                stream.write(line); stream.flush(); os.fsync(stream.fileno())
            self._write_status_locked()
            try:
                sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
            except (BrokenPipeError, OSError):
                pass

    def terminal(self, state: str, message: str, *, error: str | None = None) -> None:
        if state not in TERMINAL_STATES:
            raise FixtureJobError(f"invalid terminal state: {state}")
        kind = "error" if state == "failed" else state
        self.emit(kind, state, self._completed, message, state=state, error=error)
        with self._lock:
            self._terminal = True


def _heartbeat_loop(store: StatusStore, stop: threading.Event, interval: float, freeze_after: int | None) -> None:
    while not stop.is_set():
        if not store.heartbeat(freeze_after):
            return
        stop.wait(interval)


def _install_audit_policy() -> None:
    denied_prefixes = ("socket.connect", "socket.bind", "subprocess.", "os.system", "os.spawn")
    def audit(event, _arguments):
        if event.startswith(denied_prefixes):
            raise PermissionError(f"offline Worker policy denied audit event {event}")
    sys.addaudithook(audit)


@contextmanager
def _windows_execution_state():
    if os.name != "nt":
        yield
        return
    import ctypes
    from ctypes import wintypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel32.SetThreadExecutionState.restype = wintypes.DWORD
    if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        raise FixtureJobError(f"SetThreadExecutionState failed: {ctypes.get_last_error()}")
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def _set_below_normal_priority() -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
        raise FixtureJobError(f"SetPriorityClass failed: {ctypes.get_last_error()}")


def _wait_for_control(job_dir: Path, job: dict[str, Any], nonce: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    control_path = job_dir / "job-control.json"
    while not control_path.is_file():
        if time.monotonic() >= deadline:
            raise FixtureJobError("Job Control Envelope was not published within 30 seconds")
        time.sleep(0.025)
    control = validate_control(read_json(control_path), job)
    worker = control["worker"]
    facts = WindowsProcessProbe().current()
    mismatches = []
    if worker["pid"] != facts.pid:
        mismatches.append("pid")
    if worker["creation_time"] != facts.creation_time:
        mismatches.append("creation_time")
    if os.path.normcase(worker["executable"]) != os.path.normcase(facts.executable):
        mismatches.append("executable")
    if worker["executable_sha256"] != facts.executable_sha256:
        mismatches.append("executable_sha256")
    if worker["nonce"] != nonce:
        mismatches.append("nonce")
    if control["job_spec"]["sha256"] != sha256_file(job_dir / "job-spec.json"):
        mismatches.append("job_spec_sha256")
    if mismatches:
        raise FixtureJobError(
            "Job Control does not identify this exact Worker and JobSpec: "
            + ", ".join(mismatches)
        )
    return control


def _publish_worker_identity(job_dir: Path, nonce: str) -> None:
    facts = WindowsProcessProbe().current()
    atomic_write_json(
        job_dir / "worker.pid.json",
        {**asdict(facts), "nonce": nonce},
    )


def _terminal_destination(job: dict[str, Any], state: str) -> Path:
    root = Path(job["project_root"])
    return root / "diagnostics" / f"{job['job_id']}--fixture-{state}"


def run_fixture_job(spec_path: Path, nonce: str) -> int:
    spec_path = Path(os.path.abspath(spec_path))
    if not spec_path.is_absolute() or spec_path.name != "job-spec.json":
        raise FixtureJobError("Fixture JobSpec must be an absolute job-spec.json path")
    job_dir = spec_path.parent
    job = validate_job_spec(read_json(spec_path))
    if "fixture" not in job:
        raise FixtureJobError("Fixture runner requires a fixture JobSpec")
    if job_dir.name != job["job_id"] or job_dir.parent.name != ".jobs":
        raise FixtureJobError("Fixture JobSpec is outside its bounded active directory")
    if Path(os.path.abspath(job["project_root"])) != job_dir.parent.parent:
        raise FixtureJobError("Fixture JobSpec Project Result Root does not contain this Job")
    _publish_worker_identity(job_dir, nonce)
    _wait_for_control(job_dir, job, nonce)
    fixture = job["fixture"]
    store = StatusStore(job_dir, job["job_id"], fixture["steps"])
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(store, stop, float(fixture["heartbeat_interval_seconds"]), fixture["freeze_heartbeat_after_sequence"]),
        name="LingBotMap-Heartbeat", daemon=True,
    )
    state = "failed"
    message = "Fixture Job failed"
    error = None
    return_code = 1
    try:
        _set_below_normal_priority()
        _install_audit_policy()
        heartbeat.start()
        with _windows_execution_state():
            store.emit("phase", "fixture", 0, "Fixture Job started")
            for step in range(1, fixture["steps"] + 1):
                if (job_dir / "cancel.request").exists() and not fixture["ignore_cancel"]:
                    state, message, return_code = "cancelled", "Fixture Job cancelled", 2
                    break
                time.sleep(float(fixture["step_delay_seconds"]))
                store.emit("progress", "fixture", step, f"Fixture step {step} completed")
            else:
                state, message, return_code = "succeeded", "Fixture Job completed", 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:16384]
        message = "Fixture Job failed"
    finally:
        stop.set()
        if heartbeat.is_alive():
            heartbeat.join(timeout=6)
        store.terminal(state, message, error=error)
    destination = _terminal_destination(job, state)
    if destination.exists():
        raise FixtureJobError(f"terminal diagnostics already exist: {destination.name}")
    os.replace(job_dir, destination)
    return return_code
