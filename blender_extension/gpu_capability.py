"""Blender-safe control plane for physical-GPU discovery and qualification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Mapping
import uuid

from .model_store import ModelStore, bundled_model_catalog, is_reparse_point
from .runtime_setup import (
    RuntimeInstaller,
    RuntimeSetupError,
    bundled_runtime,
    process_identity,
    sha256_file,
)


GPU_UUID = re.compile(r"^(?:GPU|MIG)-[A-Za-z0-9-]{8,90}$")
MAX_JSON_BYTES = 2 * 1024 * 1024
PROFILE_NAMES = ("Draft", "Balanced", "High")


class GpuCapabilityError(RuntimeSetupError):
    pass


class TransientStatusRead(GpuCapabilityError):
    pass


@dataclass(frozen=True)
class GpuDevice:
    uuid: str
    name: str
    total_memory: int
    driver_version: str
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class CapabilitySnapshot:
    state: str = "idle"
    message: str = "GPU Profiles have not been tested"
    test_id: str | None = None
    gpu_uuid: str | None = None
    phase: str | None = None
    completed: int = 0
    total: int = 0
    results: tuple[Mapping[str, object], ...] = ()


def _worker_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP", "LOCALAPPDATA"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            # getpass.getuser() imports POSIX-only pwd when every user-name
            # variable is absent. Use a non-authoritative fixed value rather
            # than inheriting user-controlled identity into the Worker.
            "USERNAME": "LingBotMapWorker",
        }
    )
    return environment


def _runtime_command(managed_root: Path) -> tuple[Path, Path, str, str]:
    managed_root = Path(os.path.abspath(managed_root))
    _require_plain_ancestors(managed_root)
    bundle = bundled_runtime()
    runtime = RuntimeInstaller(managed_root, bundle).validate_existing()
    python = runtime / ".venv" / "Scripts" / "python.exe"
    lock = runtime / "uv.lock"
    if not python.is_file() or not lock.is_file():
        raise GpuCapabilityError("Published Worker Runtime executable or lock is invalid")
    _require_plain_chain(python, managed_root)
    _require_plain_chain(lock, managed_root)
    return runtime, python, bundle.identity.runtime_id, sha256_file(lock)


def _require_plain_chain(path: Path, boundary: Path) -> None:
    current = path
    boundary = Path(os.path.abspath(boundary))
    while True:
        if current.exists() and is_reparse_point(current):
            raise GpuCapabilityError(f"Linked managed path is forbidden: {current}")
        if current == boundary:
            return
        if current.parent == current or not current.is_relative_to(boundary):
            raise GpuCapabilityError("Managed path escaped its configured root")
        current = current.parent


def _require_plain_ancestors(path: Path) -> None:
    current = Path(os.path.abspath(path))
    while True:
        if current.exists() and is_reparse_point(current):
            raise GpuCapabilityError(f"Linked managed path is forbidden: {current}")
        if current.parent == current:
            return
        current = current.parent


def _read_json(path: Path, *, maximum: int = MAX_JSON_BYTES) -> object:
    if not path.is_file() or is_reparse_point(path) or path.stat().st_size > maximum:
        raise GpuCapabilityError(f"Invalid GPU capability JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GpuCapabilityError(f"Cannot parse GPU capability JSON: {exc}") from exc


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_devices(document: object) -> tuple[GpuDevice, ...]:
    if not isinstance(document, dict) or set(document) != {"schema_version", "devices"}:
        raise GpuCapabilityError("GPU discovery response has unknown or missing fields")
    if document["schema_version"] != 1 or not isinstance(document["devices"], list):
        raise GpuCapabilityError("GPU discovery response version is unsupported")
    devices = []
    for raw in document["devices"]:
        expected = {
            "uuid", "name", "total_memory", "driver_version", "compute_capability"
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise GpuCapabilityError("GPU discovery returned an invalid device record")
        capability = raw["compute_capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or not all(isinstance(item, int) and item >= 0 for item in capability)
        ):
            raise GpuCapabilityError("GPU discovery returned invalid CUDA capability")
        device = GpuDevice(
            str(raw["uuid"]),
            str(raw["name"]),
            int(raw["total_memory"]),
            str(raw["driver_version"]),
            (capability[0], capability[1]),
        )
        if (
            not GPU_UUID.fullmatch(device.uuid)
            or not device.name
            or not device.driver_version
            or device.total_memory <= 0
        ):
            raise GpuCapabilityError("GPU discovery returned an incomplete physical identity")
        devices.append(device)
    if len({item.uuid for item in devices}) != len(devices):
        raise GpuCapabilityError("GPU discovery returned duplicate physical UUIDs")
    return tuple(sorted(devices, key=lambda item: item.uuid))


def discover_physical_gpus(managed_root: Path) -> tuple[GpuDevice, ...]:
    runtime, python, _runtime_id, _lock_sha = _runtime_command(managed_root)
    command = [str(python), "-I", "-m", "lingbot_map_worker", "--discover-gpus"]
    completed = subprocess.run(
        command,
        cwd=runtime,
        env=_worker_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise GpuCapabilityError(
            f"Physical GPU discovery failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        document = json.loads(completed.stdout, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, ValueError) as exc:
        raise GpuCapabilityError(f"Physical GPU discovery returned invalid JSON: {exc}") from exc
    return _parse_devices(document)


def select_gpu_uuid(devices: tuple[GpuDevice, ...], persisted_uuid: str) -> str:
    if persisted_uuid:
        if not GPU_UUID.fullmatch(persisted_uuid):
            raise GpuCapabilityError("The persisted GPU selection is not a physical UUID")
        if sum(item.uuid == persisted_uuid for item in devices) != 1:
            raise GpuCapabilityError(
                f"Selected GPU {persisted_uuid} is unavailable; choose an available UUID"
            )
        return persisted_uuid
    if len(devices) == 1:
        return devices[0].uuid
    if not devices:
        raise GpuCapabilityError("No compatible NVIDIA GPU was discovered")
    choices = ", ".join(f"{item.name} ({item.uuid})" for item in devices)
    raise GpuCapabilityError(
        "Multiple NVIDIA GPUs were discovered; persist one physical UUID: " + choices
    )


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _status_snapshot(path: Path, test_id: str, gpu_uuid: str) -> CapabilitySnapshot:
    try:
        document = _read_json(path)
    except GpuCapabilityError as exc:
        if isinstance(exc.__cause__, OSError):
            raise TransientStatusRead(str(exc)) from exc
        raise
    expected = {
        "schema_version", "test_id", "state", "gpu_uuid", "phase",
        "completed", "total", "results",
    }
    optional = {"error", "classification", "owner"}
    if not isinstance(document, dict) or not expected.issubset(document) or (
        set(document) - expected - optional
    ):
        raise GpuCapabilityError("Capability status has unknown or missing fields")
    state = str(document["state"])
    if (
        document["schema_version"] != 1
        or document["test_id"] != test_id
        or document["gpu_uuid"] != gpu_uuid
        or state not in {"running", "succeeded", "cancelled", "blocked", "failed"}
        or not isinstance(document["results"], list)
    ):
        raise GpuCapabilityError("Capability status identity or state is invalid")
    message = str(document.get("error") or (
        "GPU Profile qualification succeeded" if state == "succeeded" else
        "GPU Profile qualification is running"
    ))
    return CapabilitySnapshot(
        state,
        message,
        test_id,
        gpu_uuid,
        str(document["phase"]),
        int(document["completed"]),
        int(document["total"]),
        tuple(document["results"]),
    )


class CapabilityController:
    """Own exactly one non-modal capability-test child and its cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = CapabilitySnapshot()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._child_identity = None
        self._cancel_requested = threading.Event()

    def snapshot(self) -> CapabilitySnapshot:
        with self._lock:
            return self._snapshot

    def start(self, managed_root: Path, gpu_uuid: str) -> None:
        if not GPU_UUID.fullmatch(gpu_uuid):
            raise GpuCapabilityError("Capability test requires one persisted physical GPU UUID")
        managed_root = Path(os.path.abspath(managed_root))
        _require_plain_ancestors(managed_root)
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise GpuCapabilityError("A GPU Profile qualification is already running")
            self._cancel_requested.clear()
            self._snapshot = CapabilitySnapshot(
                "preparing", "Validating Runtime and Reconstruction Model", gpu_uuid=gpu_uuid
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(managed_root, gpu_uuid),
                name="LingBotMap-GPU-Capability",
                daemon=True,
            )
            self._thread.start()

    def _run(self, managed_root: Path, gpu_uuid: str) -> None:
        test_id = f"cap-{uuid.uuid4().hex}"
        test_root = managed_root / "capability-tests" / test_id
        status_path = test_root / "status.json"
        process = None
        child = None
        try:
            if is_reparse_point(managed_root):
                raise GpuCapabilityError("Managed Runtime root must not be a linked path")
            runtime, python, runtime_id, lock_sha = _runtime_command(managed_root)
            catalog = bundled_model_catalog()
            reconstruction = tuple(item for item in catalog.entries if item.role == "reconstruction")
            if len(reconstruction) != 1:
                raise GpuCapabilityError("Model Catalog must contain one Reconstruction Model")
            entry = reconstruction[0]
            model_path = ModelStore(managed_root, catalog).validate(entry)
            _require_plain_chain(model_path, managed_root)
            if self._cancel_requested.is_set():
                raise InterruptedError("GPU Profile qualification was cancelled before launch")
            test_root.mkdir(parents=True)
            if is_reparse_point(test_root):
                raise GpuCapabilityError("Capability test root must not be a linked path")
            request_path = test_root / "request.json"
            nonce = uuid.uuid4().hex
            _atomic_json(
                request_path,
                {
                    "schema_version": 1,
                    "test_id": test_id,
                    "nonce": nonce,
                    "runtime_id": runtime_id,
                    "worker_lock_sha256": lock_sha,
                    "managed_root": str(managed_root),
                    "gpu_uuid": gpu_uuid,
                    "model": {
                        "catalog_version": catalog.version,
                        "id": entry.id,
                        "sha256": entry.artifact.sha256,
                        "path": str(model_path),
                    },
                    "profiles": list(PROFILE_NAMES),
                },
            )
            command = [
                str(python), "-I", "-m", "lingbot_map_worker",
                "--capability-test", str(request_path),
            ]
            stdout_path = test_root / "stdout.log"
            stderr_path = test_root / "stderr.log"
            stdout_file = stdout_path.open("wb")
            stderr_file = stderr_path.open("wb")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=runtime,
                    env=_worker_environment(),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            finally:
                # Popen duplicates the inheritable OS handles for the child.
                # Closing the parent copies prevents leaks while file-backed
                # progress avoids bounded anonymous-pipe deadlocks.
                stdout_file.close()
                stderr_file.close()
            child = process_identity(process.pid, nonce)
            with self._lock:
                self._process = process
                self._child_identity = child
                self._snapshot = CapabilitySnapshot(
                    "running", "GPU Profile qualification is starting", test_id, gpu_uuid,
                    "starting", 0, len(PROFILE_NAMES), (),
                )
            cancel_started = None
            while process.poll() is None:
                if status_path.is_file():
                    try:
                        current_status = _status_snapshot(status_path, test_id, gpu_uuid)
                    except TransientStatusRead:
                        pass
                    else:
                        self._publish(current_status)
                if self._cancel_requested.is_set():
                    (test_root / "cancel.requested").touch(exist_ok=True)
                    if cancel_started is None:
                        cancel_started = time.monotonic()
                    self._publish(
                        CapabilitySnapshot(
                            "cancelling", "Cancelling GPU Profile qualification", test_id,
                            gpu_uuid, "cancelling", 0, len(PROFILE_NAMES), (),
                        )
                    )
                    if time.monotonic() - cancel_started > 15:
                        self._terminate_exact(process, child)
                time.sleep(0.1)
            process.wait()
            stdout = _read_log(stdout_path)
            stderr = _read_log(stderr_path)
            if status_path.is_file():
                deadline = time.monotonic() + 2
                while True:
                    try:
                        terminal = _status_snapshot(status_path, test_id, gpu_uuid)
                        break
                    except TransientStatusRead:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
            elif self._cancel_requested.is_set():
                self._quarantine_cancelled_cache(managed_root, test_id)
                terminal = CapabilitySnapshot(
                    "cancelled", "GPU Profile qualification was cancelled", test_id,
                    gpu_uuid, "cancelled", 0, len(PROFILE_NAMES), (),
                )
                _atomic_json(
                    status_path,
                    {
                        "schema_version": 1, "test_id": test_id, "state": "cancelled",
                        "gpu_uuid": gpu_uuid, "phase": "cancelled", "completed": 0,
                        "total": len(PROFILE_NAMES), "results": [],
                        "error": terminal.message,
                    },
                )
            else:
                raise GpuCapabilityError(
                    f"Capability child exited {process.returncode} without terminal status: "
                    f"{stderr.strip() or stdout.strip()}"
                )
            if terminal.state == "running" and not self._cancel_requested.is_set():
                raise GpuCapabilityError(
                    f"Capability child exited {process.returncode} without terminal status: "
                    f"{stderr.strip() or stdout.strip()}"
                )
            if self._cancel_requested.is_set() and terminal.state != "cancelled":
                self._quarantine_cancelled_cache(managed_root, test_id)
                terminal = CapabilitySnapshot(
                    "cancelled", "GPU Profile qualification was cancelled", test_id,
                    gpu_uuid, "cancelled", 0, len(PROFILE_NAMES), (),
                )
                _atomic_json(
                    status_path,
                    {
                        "schema_version": 1, "test_id": test_id, "state": "cancelled",
                        "gpu_uuid": gpu_uuid, "phase": "cancelled", "completed": 0,
                        "total": len(PROFILE_NAMES), "results": [],
                        "error": terminal.message,
                    },
                )
            self._publish(terminal)
        except InterruptedError as exc:
            self._publish(CapabilitySnapshot("cancelled", str(exc), test_id, gpu_uuid))
        except Exception as exc:
            if process is not None and process.poll() is None:
                try:
                    if child is not None:
                        self._terminate_exact(process, child)
                    else:
                        # Popen's Windows process handle is the exact child even
                        # when the secondary metadata probe failed.
                        process.kill()
                        process.wait(timeout=5)
                except Exception as termination_error:
                    exc = GpuCapabilityError(
                        f"{exc}; exact child cleanup also failed: {termination_error}"
                    )
            self._publish(CapabilitySnapshot("failed", str(exc), test_id, gpu_uuid))
        finally:
            with self._lock:
                self._process = None
                self._child_identity = None

    def _publish(self, snapshot: CapabilitySnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    @staticmethod
    def _terminate_exact(process: subprocess.Popen[str], child) -> None:
        if process.poll() is not None:
            return
        try:
            observed = process_identity(process.pid, child.nonce)
        except RuntimeSetupError:
            return
        if (
            observed.pid != child.pid
            or observed.creation_time != child.creation_time
            or os.path.normcase(observed.executable) != os.path.normcase(child.executable)
        ):
            raise GpuCapabilityError("Refusing to terminate a different process identity")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    @staticmethod
    def _quarantine_cancelled_cache(managed_root: Path, test_id: str) -> None:
        cache = managed_root / "capability-cache"
        if not cache.is_dir() or is_reparse_point(cache):
            return
        destination = cache / "invalidated" / (
            "cancelled-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        for path in sorted(cache.glob("*.json")):
            try:
                document = _read_json(path)
            except GpuCapabilityError:
                continue
            if isinstance(document, dict) and document.get("test_id") == test_id:
                destination.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination / path.name)

    def cancel(self) -> bool:
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return False
            self._cancel_requested.set()
            current = self._snapshot
            self._snapshot = CapabilitySnapshot(
                "cancelling", "Cancelling GPU Profile qualification", current.test_id,
                current.gpu_uuid, current.phase, current.completed, current.total, (),
            )
            return True

    def shutdown(self, timeout: float = 20.0) -> None:
        """Cancel and prevent an exact capability child from surviving unload."""

        self.cancel()
        with self._lock:
            thread = self._thread
        if not thread or not thread.is_alive():
            return
        thread.join(timeout)
        if thread.is_alive():
            with self._lock:
                process = self._process
                child = self._child_identity
            if process is not None and child is not None:
                self._terminate_exact(process, child)
                thread.join(5)


_controller = CapabilityController()


def get_capability_snapshot() -> CapabilitySnapshot:
    return _controller.snapshot()


def start_gpu_capability(managed_root: Path, gpu_uuid: str) -> None:
    _controller.start(managed_root, gpu_uuid)


def cancel_gpu_capability() -> bool:
    return _controller.cancel()


def shutdown_gpu_capability() -> None:
    _controller.shutdown()


def _read_log(path: Path, maximum: int = MAX_JSON_BYTES) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > maximum:
                stream.seek(size - maximum)
            content = stream.read(maximum)
    except OSError:
        return ""
    return content.decode("utf-8", errors="replace").strip()
