"""Immutable Worker Runtime provisioning without importing Worker code.

The installer is deliberately usable outside Blender so locking, recovery, and
artifact policy can be exercised by ordinary unit and Windows integration tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
from typing import Callable, Iterable, Mapping, Sequence
import urllib.request
import uuid
import zipfile

if __package__:
    from .diagnostics import diagnostic_record
else:
    # This module is deliberately executable as a standalone installer/test
    # boundary, without importing Blender's package initializer.
    def diagnostic_record(
        *,
        error_code,
        category,
        state,
        phase,
        detail,
        **_unused,
    ):
        return {
            "schema_version": "1.0.0",
            "error_code": str(error_code),
            "category": str(category),
            "state": str(state),
            "phase": str(phase),
            "job_id": None,
            "target_scene": {},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "machine_name": os.environ.get("COMPUTERNAME", ""),
            "username": (
                os.environ.get("USERNAME")
                or os.environ.get("USER")
                or ""
            ),
            "detail": str(detail)[:16384],
        }


PYTHON_VERSION = "3.10.20"
WORKER_DISTRIBUTION = "lingbot-map-worker"
WORKER_VERSION = "0.1.0"
RUNTIME_IDENTITY_VERSION = 1


class RuntimeSetupError(RuntimeError):
    """A setup failure safe to surface in Setup diagnostics."""


class RuntimeBusyError(RuntimeSetupError):
    """The exact Runtime identity is being constructed by another owner."""


class MissingArtifactsError(RuntimeSetupError):
    def __init__(self, missing: Sequence["Artifact"]):
        self.missing = tuple(missing)
        detail = ", ".join(f"{item.name} ({item.filename})" for item in missing)
        super().__init__(
            "Offline Setup is missing verified catalog artifacts: " + detail
            + ". Run explicit Online Setup once, or place the exact files in the artifact cache."
        )


class SetupCancelled(RuntimeSetupError):
    """The user cancelled Setup and its staging was retained as diagnostics."""


@dataclass(frozen=True)
class Artifact:
    name: str
    filename: str
    url: str
    length: int
    sha256: str
    metadata: Mapping[str, object]

    @classmethod
    def from_entry(cls, name: str, entry: Mapping[str, object]) -> "Artifact":
        required = {"filename", "url", "length", "sha256"}
        absent = required.difference(entry)
        if absent:
            raise RuntimeSetupError(f"Artifact {name!r} lacks: {', '.join(sorted(absent))}")
        url = str(entry["url"])
        digest = str(entry["sha256"]).lower()
        if not url.startswith("https://"):
            raise RuntimeSetupError(f"Artifact {name!r} must use HTTPS")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeSetupError(f"Artifact {name!r} has an invalid SHA-256")
        filename = str(entry["filename"])
        if Path(filename).name != filename or filename in {".", ".."}:
            raise RuntimeSetupError(f"Artifact {name!r} has an unsafe filename")
        metadata = {key: value for key, value in entry.items() if key not in required}
        return cls(name, filename, url, int(entry["length"]), digest, metadata)


@dataclass(frozen=True)
class RuntimeIdentity:
    runtime_id: str
    inputs: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    creation_time: int
    executable: str
    nonce: str


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, artifact: Artifact) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == artifact.length
        and sha256_file(path) == artifact.sha256
    )


class RuntimeBundle:
    """The complete checked-in input set from which a Runtime ID is derived."""

    REQUIRED_GLOBS = (
        "artifact-catalog.json",
        "pyproject.toml",
        "uv.lock",
        "runtime-inventory.json",
        "wheels/*.whl",
        "schemas/**/*.json",
        "model-catalog.json",
        "model-licenses/*",
        "LICENSES/*",
        "NOTICES/*",
    )

    def __init__(self, root: Path):
        self.root = root.resolve()
        catalog_path = self.root / "artifact-catalog.json"
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeSetupError(f"Cannot read artifact catalog: {exc}") from exc
        self.artifacts = tuple(
            Artifact.from_entry(name, entry)
            for name, entry in sorted(catalog.items())
            if name != "catalog_version" and isinstance(entry, dict)
        )
        if {artifact.name for artifact in self.artifacts} != {"python", "uv"}:
            raise RuntimeSetupError("Artifact catalog must contain exactly pinned python and uv")
        self.payload_files = self._discover_payload()
        self.expected_inventory = self._locked_inventory()
        self.identity = self._derive_identity()

    def _locked_inventory(self) -> tuple[tuple[str, str], ...]:
        try:
            document = tomllib.loads((self.root / "uv.lock").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise RuntimeSetupError(f"Cannot read frozen Runtime lock inventory: {exc}") from exc
        packages = document.get("package", [])
        if not isinstance(packages, list):
            raise RuntimeSetupError("Runtime lock package inventory must be an array")
        locked = {}
        for package in packages:
            if not isinstance(package, dict):
                raise RuntimeSetupError("Runtime lock package entry must be an object")
            source = package.get("source", {})
            if isinstance(source, dict) and "virtual" in source:
                continue
            name = str(package.get("name", "")).lower().replace("_", "-")
            version = str(package.get("version", ""))
            if not name or not version:
                raise RuntimeSetupError("Runtime lock package lacks an exact name or version")
            if name in locked:
                raise RuntimeSetupError("Runtime lock contains duplicate package names")
            locked[name] = version
        try:
            manifest = json.loads(
                (self.root / "runtime-inventory.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeSetupError(f"Cannot read Windows Runtime inventory: {exc}") from exc
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "platform", "python", "packages"}
            or manifest["schema_version"] != 1
            or manifest["platform"] != "windows-x64"
            or manifest["python"] != PYTHON_VERSION
            or not isinstance(manifest["packages"], list)
        ):
            raise RuntimeSetupError("Windows Runtime inventory identity is invalid")
        inventory = []
        for item in manifest["packages"]:
            if not isinstance(item, dict) or set(item) != {"name", "version"}:
                raise RuntimeSetupError("Windows Runtime inventory package is invalid")
            name = str(item["name"]).lower().replace("_", "-")
            version = str(item["version"])
            if locked.get(name) != version:
                raise RuntimeSetupError(
                    f"Windows Runtime inventory is not pinned by uv.lock: {name}=={version}"
                )
            inventory.append((name, version))
        if len({name for name, _version in inventory}) != len(inventory):
            raise RuntimeSetupError("Runtime lock contains duplicate package names")
        return tuple(sorted(inventory))

    def _discover_payload(self) -> tuple[Path, ...]:
        found: set[Path] = set()
        for pattern in self.REQUIRED_GLOBS:
            matches = {path for path in self.root.glob(pattern) if path.is_file()}
            if not matches:
                raise RuntimeSetupError(f"Runtime bundle is missing required input {pattern!r}")
            found.update(matches)
        return tuple(sorted(found, key=lambda path: path.relative_to(self.root).as_posix()))

    def _derive_identity(self) -> RuntimeIdentity:
        inputs: list[Mapping[str, object]] = []
        for path in self.payload_files:
            inputs.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "length": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        document = {
            "identity_version": RUNTIME_IDENTITY_VERSION,
            "inputs": inputs,
        }
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return RuntimeIdentity(hashlib.sha256(encoded).hexdigest(), tuple(inputs))

    def copy_payload(self, destination: Path) -> None:
        for source in self.payload_files:
            relative = source.relative_to(self.root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        manifest = {
            "identity_version": RUNTIME_IDENTITY_VERSION,
            "runtime_id": self.identity.runtime_id,
            "inputs": self.identity.inputs,
        }
        (destination / "runtime-identity.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def validate_payload(self, directory: Path) -> None:
        identity_path = directory / "runtime-identity.json"
        try:
            manifest = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeSetupError(f"Runtime identity manifest is unreadable: {exc}") from exc
        if manifest.get("runtime_id") != self.identity.runtime_id:
            raise RuntimeSetupError("Runtime identity does not match the requested bundle")
        for item in self.identity.inputs:
            path = directory / str(item["path"])
            if not path.is_file() or path.stat().st_size != item["length"] or sha256_file(path) != item["sha256"]:
                raise RuntimeSetupError(f"Runtime input validation failed: {item['path']}")


Downloader = Callable[[str, Path], None]


def download_https(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "LingBotMap-Reconstruction/0.1.0"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=60) as response, destination.open("wb") as output:
        if response.geturl().split(":", 1)[0].lower() != "https":
            raise RuntimeSetupError("Artifact download redirected away from HTTPS")
        shutil.copyfileobj(response, output)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, artifact: Artifact) -> Path:
        return self.root / artifact.sha256 / artifact.filename

    def acquire(
        self,
        artifacts: Iterable[Artifact],
        *,
        offline: bool,
        online_access: bool,
        downloader: Downloader = download_https,
    ) -> Mapping[str, Path]:
        artifacts = tuple(artifacts)
        missing = [artifact for artifact in artifacts if not verify_file(self.path_for(artifact), artifact)]
        if offline and missing:
            raise MissingArtifactsError(missing)
        if missing and not online_access:
            raise RuntimeSetupError(
                "Blender Online Access is disabled. Enable it or use Offline Setup with a populated cache."
            )
        for artifact in missing:
            target = self.path_for(artifact)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.download")
            try:
                downloader(artifact.url, temporary)
                if not verify_file(temporary, artifact):
                    raise RuntimeSetupError(f"Downloaded artifact failed size/SHA-256 validation: {artifact.name}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return {artifact.name: self.path_for(artifact) for artifact in artifacts}


def current_process_identity(nonce: str | None = None) -> ProcessIdentity:
    return process_identity(os.getpid(), nonce or uuid.uuid4().hex)


def process_identity(pid: int, nonce: str) -> ProcessIdentity:
    executable = sys.executable
    creation_time = 0
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            raise RuntimeSetupError(f"Cannot identify process {pid}")
        try:
            created = wintypes.FILETIME()
            exited = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
                raise RuntimeSetupError(f"Cannot read process creation time for {pid}")
            creation_time = (created.dwHighDateTime << 32) | created.dwLowDateTime
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                raise RuntimeSetupError(f"Cannot read executable path for {pid}")
            executable = buffer.value
        finally:
            kernel32.CloseHandle(handle)
    else:
        creation_time = int(time.time_ns())
    return ProcessIdentity(pid, creation_time, str(Path(executable).resolve()), nonce)


class RuntimeLock:
    """An OS-held per-ID lock whose JSON records the exact process owner."""

    def __init__(self, path: Path, owner: ProcessIdentity):
        self.path = path
        self.owner_path = path.with_suffix(path.suffix + ".owner.json")
        self.owner = owner
        self._stream = None
        self.previous_owner: Mapping[str, object] | None = None

    def __enter__(self) -> "RuntimeLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        else:
            try:
                os.write(descriptor, b"\0")
            finally:
                os.close(descriptor)
        stream = self.path.open("r+b")
        try:
            self._lock(stream)
            self.previous_owner = self._read_owner()
            temporary = self.owner_path.with_name(f".{self.owner_path.name}.{self.owner.nonce}.tmp")
            temporary.write_text(json.dumps(asdict(self.owner), sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.owner_path)
            self._stream = stream
            return self
        except BlockingIOError as exc:
            stream.close()
            owner = self._read_owner() or {"unknown": True}
            raise RuntimeBusyError(f"Runtime Setup is owned by {json.dumps(owner, sort_keys=True)}") from exc
        except Exception:
            stream.close()
            raise

    def _read_owner(self) -> Mapping[str, object] | None:
        try:
            return json.loads(self.owner_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return {"unreadable": True}

    @staticmethod
    def _lock(stream) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError(str(exc)) from exc
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(stream) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stream is None:
            return
        stored = self._read_owner()
        if stored != asdict(self.owner):
            raise RuntimeSetupError("Refusing to release a Runtime lock owned by another process identity")
        self._unlock(self._stream)
        self._stream.close()
        self._stream = None


class ProcessRunner:
    """Runs an exact child and terminates only if its process identity still matches."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        cancellation: CancellationToken,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            list(command), cwd=cwd, env=dict(env), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
        )
        try:
            child = process_identity(process.pid, uuid.uuid4().hex)
        except RuntimeSetupError:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
                raise
            stdout, stderr = process.communicate()
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if result.returncode:
                raise RuntimeSetupError(
                    f"Command failed ({result.returncode}): {Path(command[0]).name}\n{stderr.strip()}"
                )
            return result
        while process.poll() is None:
            if cancellation.cancelled:
                try:
                    observed = process_identity(process.pid, child.nonce)
                except RuntimeSetupError:
                    observed = None
                if observed and observed.pid == child.pid and observed.creation_time == child.creation_time and observed.executable == child.executable:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                raise SetupCancelled("Runtime Setup was cancelled; incomplete staging was retained")
            time.sleep(0.05)
        stdout, stderr = process.communicate()
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if result.returncode:
            raise RuntimeSetupError(
                f"Command failed ({result.returncode}): {Path(command[0]).name}\n{stderr.strip()}"
            )
        return result


def isolated_environment(
    staging: Path, *, offline: bool, cache_root: Path | None = None
) -> dict[str, str]:
    """Build an allowlisted environment; no user Python/package/network config survives."""

    environment: dict[str, str] = {}
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "UV_CACHE_DIR": str((cache_root or staging / ".uv-cache").resolve()),
            "UV_PROJECT_ENVIRONMENT": str(staging / ".venv"),
            "UV_NO_CONFIG": "1",
            "UV_NO_PYTHON_DOWNLOADS": "1",
            "UV_MANAGED_PYTHON": "1",
        }
    )
    if offline:
        environment["UV_OFFLINE"] = "1"
    return environment


class RuntimeInstaller:
    def __init__(
        self,
        managed_root: Path,
        bundle: RuntimeBundle,
        *,
        downloader: Downloader = download_https,
        runner: ProcessRunner | None = None,
    ):
        self.managed_root = managed_root.resolve()
        self.bundle = bundle
        self.downloader = downloader
        self.runner = runner or ProcessRunner()
        self.runtimes_root = self.managed_root / "runtimes"
        self.diagnostics_root = self.managed_root / "setup-diagnostics"
        self.artifact_store = ArtifactStore(self.managed_root / "artifact-cache")

    @property
    def runtime_path(self) -> Path:
        return self.runtimes_root / self.bundle.identity.runtime_id

    def validate_existing(self, cancellation: CancellationToken | None = None) -> Path:
        """Validate and return the published Runtime for this exact bundle identity."""

        cancellation = cancellation or CancellationToken()
        directory = self.runtime_path
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeSetupError(
                f"Worker Runtime {self.bundle.identity.runtime_id} is not published"
            )
        try:
            ready = json.loads((directory / "READY.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeSetupError("Published Worker Runtime has no valid READY record") from exc
        if (
            not isinstance(ready, dict)
            or set(ready) != {"runtime_id", "python", "worker", "published_at"}
            or ready.get("runtime_id") != self.bundle.identity.runtime_id
            or ready.get("python") != PYTHON_VERSION
            or ready.get("worker") != f"{WORKER_DISTRIBUTION}=={WORKER_VERSION}"
        ):
            raise RuntimeSetupError("Published Worker Runtime READY identity does not match")
        self._validate_runtime(directory, cancellation)
        return directory

    def setup(
        self,
        *,
        offline: bool,
        online_access: bool,
        cancellation: CancellationToken | None = None,
    ) -> Path:
        cancellation = cancellation or CancellationToken()
        if cancellation.cancelled:
            raise SetupCancelled("Runtime Setup was cancelled before it started")
        if not offline and not online_access:
            raise RuntimeSetupError(
                "Blender Online Access is disabled. Enable it or use explicit Offline Setup."
            )
        artifacts = self.artifact_store.acquire(
            self.bundle.artifacts,
            offline=offline,
            online_access=online_access,
            downloader=self.downloader,
        )
        if cancellation.cancelled:
            raise SetupCancelled("Runtime Setup was cancelled during artifact acquisition")
        self.runtimes_root.mkdir(parents=True, exist_ok=True)
        owner = current_process_identity()
        lock_path = self.managed_root / "locks" / f"{self.bundle.identity.runtime_id}.lock"
        with RuntimeLock(lock_path, owner) as runtime_lock:
            if self.runtime_path.is_symlink():
                raise RuntimeSetupError("Refusing to use a linked Runtime identity path")
            if self.runtime_path.exists():
                try:
                    self._validate_runtime(self.runtime_path, cancellation)
                except RuntimeSetupError as exc:
                    self._retain_diagnostic(
                        self.runtime_path, "invalid-published-runtime", {"error": str(exc)}
                    )
                else:
                    return self.runtime_path
            self._retain_stale_staging(runtime_lock.previous_owner)
            # Keep the same-parent path short enough for Python's Windows file APIs;
            # the complete identity is recorded inside and the lock remains full-ID.
            staging = self.runtimes_root / f".staging-{self.bundle.identity.runtime_id[:16]}-{owner.nonce}"
            staging.mkdir()
            try:
                (staging / ".runtime-id").write_text(self.bundle.identity.runtime_id, encoding="ascii")
                self.bundle.copy_payload(staging)
                uv = self._extract_uv(artifacts["uv"], staging)
                python = self._extract_python(artifacts["python"], staging)
                environment = isolated_environment(
                    staging, offline=offline, cache_root=self.managed_root / "uv-cache"
                )
                uv_version = str(
                    next(item.metadata["version"] for item in self.bundle.artifacts if item.name == "uv")
                )
                result = self.runner.run(
                    [str(uv), "--version"], cwd=staging, env=environment, cancellation=cancellation
                )
                if not result.stdout.startswith(f"uv {uv_version} "):
                    raise RuntimeSetupError(f"Private uv validation failed: {result.stdout.strip()}")
                command = [
                    str(uv), "sync", "--frozen", "--no-dev", "--no-config",
                    "--managed-python", "--no-python-downloads", "--python", str(python),
                ]
                if offline:
                    command.append("--offline")
                self.runner.run(command, cwd=staging, env=environment, cancellation=cancellation)
                (staging / "empty-cwd").mkdir()
                self._validate_runtime(staging, cancellation)
                self._relocate_virtual_environment(staging)
                ready = {
                    "runtime_id": self.bundle.identity.runtime_id,
                    "python": PYTHON_VERSION,
                    "worker": f"{WORKER_DISTRIBUTION}=={WORKER_VERSION}",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                }
                (staging / "READY.json").write_text(
                    json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                os.replace(staging, self.runtime_path)
                return self.runtime_path
            except Exception as exc:
                if staging.exists():
                    self._retain_diagnostic(staging, "cancelled" if isinstance(exc, SetupCancelled) else "failed", {"error": str(exc)})
                raise

    def _relocate_virtual_environment(self, staging: Path) -> None:
        """Point the Windows venv launcher at the immutable post-publish path."""

        configuration = staging / ".venv" / "pyvenv.cfg"
        if os.name != "nt":
            return
        try:
            content = configuration.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeSetupError(f"Cannot relocate Runtime virtual environment: {exc}") from exc
        staging_text = str(staging)
        if staging_text.lower() not in content.lower():
            raise RuntimeSetupError("Runtime virtual environment does not identify its staging owner")
        # Preserve spelling from uv while replacing case-insensitively once.
        offset = content.lower().index(staging_text.lower())
        relocated = content[:offset] + str(self.runtime_path) + content[offset + len(staging_text):]
        configuration.write_text(relocated, encoding="utf-8")
        if staging_text.lower() in configuration.read_text(encoding="utf-8").lower():
            raise RuntimeSetupError("Runtime virtual environment still references staging")

    def _extract_uv(self, archive: Path, staging: Path) -> Path:
        destination = staging / "tools" / "uv"
        destination.mkdir(parents=True)
        with zipfile.ZipFile(archive) as source:
            _safe_extract_zip(source, destination)
        candidates = list(destination.rglob("uv.exe" if os.name == "nt" else "uv"))
        if len(candidates) != 1:
            raise RuntimeSetupError("Pinned uv archive must contain exactly one uv executable")
        return candidates[0]

    def _extract_python(self, archive: Path, staging: Path) -> Path:
        # The exact uv artifact key is identity metadata; a short on-disk path
        # keeps the immutable Runtime usable on Windows without long-path policy.
        destination = staging / "py"
        destination.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as source:
            _safe_extract_tar(source, destination)
        interpreter = destination / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        if not interpreter.is_file():
            raise RuntimeSetupError("Pinned CPython archive must contain exactly one root interpreter")
        return interpreter

    def _validate_runtime(self, directory: Path, cancellation: CancellationToken) -> None:
        self.bundle.validate_payload(directory)
        empty_cwd = directory / "empty-cwd"
        if (
            not empty_cwd.is_dir()
            or empty_cwd.is_symlink()
            or any(empty_cwd.iterdir())
        ):
            raise RuntimeSetupError("Runtime trusted Worker working directory is absent or not empty")
        python = directory / "py" / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        if not python.is_file():
            raise RuntimeSetupError("Runtime does not contain exactly one managed CPython interpreter")
        environment = isolated_environment(
            directory, offline=True, cache_root=self.managed_root / "uv-cache"
        )
        probe = (
            "import json,platform,struct,sys;"
            "print(json.dumps({'version':platform.python_version(),'bits':struct.calcsize('P')*8,'exe':sys.executable}))"
        )
        result = self.runner.run([str(python), "-I", "-c", probe], cwd=directory, env=environment, cancellation=cancellation)
        facts = json.loads(result.stdout.strip())
        if facts.get("version") != PYTHON_VERSION or facts.get("bits") != 64:
            raise RuntimeSetupError(f"Managed CPython validation failed: {facts}")
        if Path(str(facts.get("exe"))).resolve() != python.resolve():
            raise RuntimeSetupError("Managed CPython executable escaped the Runtime")
        virtual_python = directory / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not virtual_python.is_file() or not virtual_python.resolve().is_relative_to(directory.resolve()):
            raise RuntimeSetupError("Runtime virtual environment is missing or escaped the Runtime")
        result = self.runner.run(
            [str(virtual_python), "-I", "-m", "lingbot_map_worker", "--identity"],
            cwd=directory, env=environment, cancellation=cancellation,
        )
        identity = json.loads(result.stdout.strip())
        if identity != {"distribution": WORKER_DISTRIBUTION, "version": WORKER_VERSION}:
            raise RuntimeSetupError(f"Worker package validation failed: {identity}")
        inventory_probe = (
            "import importlib.metadata,json,re;"
            "print(json.dumps(sorted((re.sub(r'[-_.]+','-',d.metadata['Name'].lower()),d.version) for d in importlib.metadata.distributions())))"
        )
        result = self.runner.run([str(virtual_python), "-I", "-c", inventory_probe], cwd=directory, env=environment, cancellation=cancellation)
        inventory = json.loads(result.stdout.strip())
        expected_inventory = [list(item) for item in self.bundle.expected_inventory]
        if inventory != expected_inventory:
            raise RuntimeSetupError(f"Unexpected Runtime package inventory: {inventory}")

    def _retain_stale_staging(self, previous_owner: Mapping[str, object] | None) -> None:
        for staging in sorted(self.runtimes_root.glob(".staging-*")):
            if staging.is_dir() and self._staging_runtime_id(staging) == self.bundle.identity.runtime_id:
                self._retain_diagnostic(staging, "stale-owner-recovery", {"previous_owner": previous_owner})

    @staticmethod
    def _staging_runtime_id(staging: Path) -> str | None:
        marker = staging / ".runtime-id"
        if marker.is_file():
            return marker.read_text(encoding="ascii").strip()
        manifest = staging / "runtime-identity.json"
        if manifest.is_file():
            try:
                return str(json.loads(manifest.read_text(encoding="utf-8"))["runtime_id"])
            except (OSError, KeyError, json.JSONDecodeError):
                return None
        return None

    def _retain_diagnostic(self, staging: Path, reason: str, detail: Mapping[str, object]) -> Path:
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.diagnostics_root / (
            f"{stamp}-{self.bundle.identity.runtime_id[:16]}-{uuid.uuid4().hex[:12]}"
        )
        os.replace(staging, destination)
        record = {
            **diagnostic_record(
                error_code=f"setup.runtime.{reason}",
                category="setup",
                state="cancelled" if reason == "cancelled" else "failed",
                phase="runtime-setup",
                detail=detail.get("error", reason),
            ),
            "reason": reason,
            "runtime_id": self.bundle.identity.runtime_id,
            **detail,
        }
        (destination / "setup-diagnostic.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(root):
            raise RuntimeSetupError(f"Unsafe uv archive member: {member.filename}")
    archive.extractall(destination)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if (
            not target.is_relative_to(root)
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            raise RuntimeSetupError(f"Unsafe CPython archive member: {member.name}")
    archive.extractall(destination)


def cleanup_runtimes(
    runtimes_root: Path,
    selected_runtime_ids: Iterable[str],
    *,
    installed_extension_refs: Iterable[str],
    active_job_refs: Iterable[str],
    confirmed: bool,
) -> tuple[Path, ...]:
    """Delete explicit unreferenced IDs; never traverse links or infer candidates."""

    if not confirmed:
        raise RuntimeSetupError("Runtime cleanup requires explicit confirmation")
    protected = set(installed_extension_refs) | set(active_job_refs)
    removed: list[Path] = []
    root = runtimes_root.resolve()
    for runtime_id in selected_runtime_ids:
        if runtime_id in protected:
            continue
        if len(runtime_id) != 64 or any(character not in "0123456789abcdef" for character in runtime_id):
            raise RuntimeSetupError(f"Invalid Runtime ID selected for cleanup: {runtime_id!r}")
        candidate = root / runtime_id
        if candidate.is_symlink():
            raise RuntimeSetupError(f"Refusing to clean a linked Runtime path: {candidate}")
        if candidate.is_dir():
            shutil.rmtree(candidate)
            removed.append(candidate)
    return tuple(removed)


def default_managed_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeSetupError("LOCALAPPDATA is unavailable; choose a Managed Runtime Root")
    return Path(local_app_data) / "LingBotMap"


def bundled_runtime() -> RuntimeBundle:
    return RuntimeBundle(Path(__file__).resolve().parent / "runtime_bundle")
