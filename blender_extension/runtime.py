"""Process-local host and non-modal Setup state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

from .host import HostDecision
from .model_store import ModelEntry, ModelStore, bundled_model_catalog
from .runtime_setup import (
    CancellationToken,
    RuntimeInstaller,
    RuntimeSetupError,
    SetupCancelled,
    bundled_runtime,
    default_managed_root,
)


_host_decision: HostDecision | None = None


@dataclass(frozen=True)
class SetupSnapshot:
    state: str = "idle"
    message: str = "Worker Runtime is not configured"
    runtime_id: str | None = None
    path: str | None = None


class SetupController:
    """Own one background Setup without exposing Blender objects to its thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = SetupSnapshot()
        self._cancellation: CancellationToken | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self) -> SetupSnapshot:
        with self._lock:
            return self._snapshot

    def start(self, managed_root: Path | None, *, offline: bool, online_access: bool) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeSetupError("Worker Runtime Setup is already running")
            root = managed_root or default_managed_root()
            cancellation = CancellationToken()
            self._cancellation = cancellation
            mode = "Offline" if offline else "Online"
            self._snapshot = SetupSnapshot("running", f"{mode} Worker Runtime Setup is running")
            thread = threading.Thread(
                target=self._run,
                args=(root, offline, online_access, cancellation),
                name="LingBotMap-Runtime-Setup",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def _run(
        self,
        root: Path,
        offline: bool,
        online_access: bool,
        cancellation: CancellationToken,
    ) -> None:
        try:
            bundle = bundled_runtime()
            path = RuntimeInstaller(root, bundle).setup(
                offline=offline, online_access=online_access, cancellation=cancellation
            )
        except Exception as exc:
            state = "cancelled" if isinstance(exc, SetupCancelled) else "failed"
            snapshot = SetupSnapshot(state, str(exc))
        else:
            snapshot = SetupSnapshot(
                "ready", "Worker Runtime is ready", bundle.identity.runtime_id, str(path)
            )
        with self._lock:
            self._snapshot = snapshot
            self._cancellation = None

    def cancel(self) -> bool:
        with self._lock:
            if not self._cancellation:
                return False
            self._cancellation.cancel()
            self._snapshot = SetupSnapshot("cancelling", "Cancelling Worker Runtime Setup")
            return True


_setup_controller = SetupController()


@dataclass(frozen=True)
class ModelSetupSnapshot:
    state: str = "idle"
    action: str | None = None
    model_id: str | None = None
    message: str = "No model acquisition is active"
    completed: int = 0
    total: int = 0
    path: str | None = None


class ModelSetupController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = ModelSetupSnapshot()
        self._cancellation: CancellationToken | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self) -> ModelSetupSnapshot:
        with self._lock:
            return self._snapshot

    def start_download(
        self,
        managed_root: Path | None,
        model_id: str,
        *,
        offline: bool,
        online_access: bool,
    ) -> None:
        self._start(
            "download",
            managed_root,
            model_id,
            lambda store, token: store.acquire(
                model_id,
                offline=offline,
                online_access=online_access,
                cancellation=token,
                progress=self._progress,
            ),
        )

    def start_import(
        self, managed_root: Path | None, model_id: str, source: Path
    ) -> None:
        self._start(
            "import",
            managed_root,
            model_id,
            lambda store, token: store.import_local(
                source,
                expected_model_id=model_id,
                cancellation=token,
                progress=self._progress,
            )[1],
        )

    def _start(self, action, managed_root, model_id, operation) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeSetupError("Another model acquisition or import is already running")
            catalog = bundled_model_catalog()
            entry = catalog.by_id(model_id)
            root = managed_root or default_managed_root()
            token = CancellationToken()
            self._cancellation = token
            self._snapshot = ModelSetupSnapshot(
                "running", action, model_id, f"{action.title()} {entry.display_name}", 0,
                entry.artifact.length,
            )
            thread = threading.Thread(
                target=self._run,
                args=(root, entry, action, operation, token),
                name=f"LingBotMap-Model-{action}",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def _run(self, root, entry: ModelEntry, action, operation, token) -> None:
        try:
            store = ModelStore(root, bundled_model_catalog())
            path = operation(store, token)
        except Exception as exc:
            state = "cancelled" if isinstance(exc, SetupCancelled) else "failed"
            snapshot = ModelSetupSnapshot(state, action, entry.id, str(exc))
        else:
            snapshot = ModelSetupSnapshot(
                "ready", action, entry.id, f"{entry.display_name} is registered", entry.artifact.length,
                entry.artifact.length, str(path),
            )
        with self._lock:
            self._snapshot = snapshot
            self._cancellation = None

    def _progress(self, completed: int, total: int) -> None:
        with self._lock:
            current = self._snapshot
            self._snapshot = ModelSetupSnapshot(
                current.state,
                current.action,
                current.model_id,
                current.message,
                completed,
                total,
                current.path,
            )

    def cancel(self) -> bool:
        with self._lock:
            if not self._cancellation:
                return False
            self._cancellation.cancel()
            current = self._snapshot
            self._snapshot = ModelSetupSnapshot(
                "cancelling", current.action, current.model_id,
                f"Cancelling {current.model_id} {current.action}",
                current.completed, current.total,
            )
            return True


_model_setup_controller = ModelSetupController()


def set_host_decision(decision: HostDecision) -> None:
    global _host_decision
    _host_decision = decision


def get_host_decision() -> HostDecision | None:
    return _host_decision


def clear_host_decision() -> None:
    global _host_decision
    _host_decision = None


def get_setup_snapshot() -> SetupSnapshot:
    return _setup_controller.snapshot()


def start_runtime_setup(managed_root: Path | None, *, offline: bool, online_access: bool) -> None:
    _setup_controller.start(managed_root, offline=offline, online_access=online_access)


def cancel_runtime_setup() -> bool:
    return _setup_controller.cancel()


def get_model_setup_snapshot() -> ModelSetupSnapshot:
    return _model_setup_controller.snapshot()


def start_model_download(
    managed_root: Path | None,
    model_id: str,
    *,
    offline: bool,
    online_access: bool,
) -> None:
    _model_setup_controller.start_download(
        managed_root, model_id, offline=offline, online_access=online_access
    )


def start_model_import(managed_root: Path | None, model_id: str, source: Path) -> None:
    _model_setup_controller.start_import(managed_root, model_id, source)


def cancel_model_setup() -> bool:
    return _model_setup_controller.cancel()
