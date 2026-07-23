"""Exact-owner, process-held Windows lease for one physical GPU UUID."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Protocol

from .gpu_devices import require_gpu_uuid


MAX_LEASE_METADATA_BYTES = 64 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class GpuLeaseError(RuntimeError):
    pass


class GpuBusyError(GpuLeaseError):
    def __init__(self, message: str, *, owner: Mapping[str, object] | None = None):
        self.owner = owner
        super().__init__(message)


class AmbiguousGpuLeaseError(GpuBusyError):
    pass


@dataclass(frozen=True)
class ProcessFacts:
    pid: int
    creation_time: int
    executable: str
    executable_sha256: str


@dataclass(frozen=True)
class LeaseOwner(ProcessFacts):
    metadata_version: int
    gpu_uuid: str
    nonce: str
    runtime_id: str
    action_kind: str
    action_id: str


class ProcessProbe(Protocol):
    def current(self) -> ProcessFacts: ...

    def observe(self, pid: int) -> ProcessFacts | None: ...


class LeaseHandle(Protocol):
    def read(self, maximum_bytes: int) -> bytes: ...

    def replace(self, content: bytes) -> None: ...

    def close(self) -> None: ...


class LeaseHandleProvider(Protocol):
    def try_acquire(self, path: Path) -> LeaseHandle | None: ...

    def read_shared(self, path: Path, maximum_bytes: int) -> bytes: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WindowsProcessProbe:
    def current(self) -> ProcessFacts:
        facts = self.observe(os.getpid())
        if facts is None:
            raise GpuLeaseError("Cannot observe the current capability-test process")
        return facts

    def observe(self, pid: int) -> ProcessFacts | None:
        if os.name != "nt":
            raise GpuLeaseError("Native v1 GPU leases require Windows")
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
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            error = ctypes.get_last_error()
            if error in {87, 1168}:
                return None
            if error == 5:
                raise GpuLeaseError(f"Process identity for PID {pid} is access denied")
            return None
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise GpuLeaseError(f"Cannot read exit state for PID {pid}")
            if exit_code.value != 259:  # STILL_ACTIVE
                return None
            created = wintypes.FILETIME()
            exited = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                raise GpuLeaseError(f"Cannot read process creation time for PID {pid}")
            creation_time = (created.dwHighDateTime << 32) | created.dwLowDateTime
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            ):
                if ctypes.get_last_error() in {87, 1168}:
                    return None
                raise GpuLeaseError(f"Cannot read executable path for PID {pid}")
            executable = str(Path(buffer.value).resolve())
        finally:
            kernel32.CloseHandle(handle)
        return ProcessFacts(pid, creation_time, executable, sha256_file(Path(executable)))


class _WindowsLeaseHandle:
    def __init__(self, handle: int):
        self._handle = handle

    def read(self, maximum_bytes: int) -> bytes:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetFilePointerEx.argtypes = [
            wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD
        ]
        kernel32.SetFilePointerEx.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
        ]
        kernel32.ReadFile.restype = wintypes.BOOL
        position = ctypes.c_longlong(0)
        if not kernel32.SetFilePointerEx(self._handle, position, None, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(maximum_bytes + 1)
        read = wintypes.DWORD()
        if not kernel32.ReadFile(
            self._handle, buffer, maximum_bytes + 1, ctypes.byref(read), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if read.value > maximum_bytes:
            raise AmbiguousGpuLeaseError("GPU Lease metadata exceeds its size limit")
        return buffer.raw[: read.value]

    def replace(self, content: bytes) -> None:
        import ctypes
        from ctypes import wintypes

        if len(content) > MAX_LEASE_METADATA_BYTES:
            raise GpuLeaseError("GPU Lease owner metadata exceeds its size limit")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetFilePointerEx.argtypes = [
            wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD
        ]
        kernel32.SetFilePointerEx.restype = wintypes.BOOL
        kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
        kernel32.SetEndOfFile.restype = wintypes.BOOL
        kernel32.WriteFile.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
        ]
        kernel32.WriteFile.restype = wintypes.BOOL
        kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        position = ctypes.c_longlong(0)
        if not kernel32.SetFilePointerEx(self._handle, position, None, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.SetEndOfFile(self._handle):
            raise ctypes.WinError(ctypes.get_last_error())
        if content:
            written = wintypes.DWORD()
            buffer = ctypes.create_string_buffer(content)
            if not kernel32.WriteFile(
                self._handle, buffer, len(content), ctypes.byref(written), None
            ) or written.value != len(content):
                raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.FlushFileBuffers(self._handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle is None:
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(self._handle)
        self._handle = None


class WindowsLeaseHandleProvider:
    def try_acquire(self, path: Path) -> LeaseHandle | None:
        if os.name != "nt":
            raise GpuLeaseError("Native v1 GPU leases require Windows")
        import ctypes
        from ctypes import wintypes

        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        FILE_SHARE_READ = 0x00000001
        OPEN_ALWAYS = 4
        FILE_ATTRIBUTE_NORMAL = 0x00000080
        FILE_FLAG_WRITE_THROUGH = 0x80000000
        ERROR_SHARING_VIOLATION = 32
        ERROR_LOCK_VIOLATION = 33
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        handle = kernel32.CreateFileW(
            str(path),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ,
            None,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
            None,
        )
        if handle == INVALID_HANDLE_VALUE:
            error = ctypes.get_last_error()
            if error in {ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION}:
                return None
            raise ctypes.WinError(error)
        return _WindowsLeaseHandle(handle)

    def read_shared(self, path: Path, maximum_bytes: int) -> bytes:
        try:
            with path.open("rb") as stream:
                content = stream.read(maximum_bytes + 1)
        except FileNotFoundError:
            return b""
        if len(content) > maximum_bytes:
            raise AmbiguousGpuLeaseError("GPU Lease metadata exceeds its size limit")
        return content


def fixed_gpu_lease_path(gpu_uuid: str, *, local_app_data: Path | None = None) -> Path:
    require_gpu_uuid(gpu_uuid)
    if local_app_data is None:
        value = os.environ.get("LOCALAPPDATA")
        if not value:
            raise GpuLeaseError("LOCALAPPDATA is unavailable for the fixed GPU Lease domain")
        local_app_data = Path(value)
    return local_app_data.resolve() / "LingBotMap" / "coordination" / "gpu" / f"{gpu_uuid}.lease"


def _duplicates_rejected(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate key: {key}")
        document[key] = value
    return document


def _parse_owner(content: bytes, gpu_uuid: str) -> LeaseOwner | None:
    if not content:
        return None
    try:
        document = json.loads(
            content.decode("utf-8"), object_pairs_hook=_duplicates_rejected
        )
        expected = {
            "pid", "creation_time", "executable", "executable_sha256",
            "metadata_version", "gpu_uuid", "nonce", "runtime_id",
            "action_kind", "action_id",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise ValueError("unknown or missing fields")
        owner = LeaseOwner(**document)
        if (
            owner.metadata_version != 1
            or owner.gpu_uuid != gpu_uuid
            or owner.pid <= 0
            or owner.creation_time <= 0
            or not HEX64.fullmatch(owner.executable_sha256)
            or not HEX64.fullmatch(owner.runtime_id)
            or not owner.nonce
            or owner.action_kind not in {"capability-test", "reconstruction-job"}
            or not owner.action_id
        ):
            raise ValueError("invalid field value")
        return owner
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AmbiguousGpuLeaseError(f"GPU Lease metadata is ambiguous: {exc}") from exc


def _same_process(owner: LeaseOwner, facts: ProcessFacts) -> bool:
    return (
        owner.pid == facts.pid
        and owner.creation_time == facts.creation_time
        and os.path.normcase(str(Path(owner.executable).resolve()))
        == os.path.normcase(str(Path(facts.executable).resolve()))
        and owner.executable_sha256 == facts.executable_sha256
    )


class GpuLease:
    """The child owns this OS handle for its complete CUDA lifetime."""

    def __init__(
        self,
        gpu_uuid: str,
        *,
        nonce: str,
        runtime_id: str,
        action_kind: str,
        action_id: str,
        local_app_data: Path | None = None,
        probe: ProcessProbe | None = None,
        handles: LeaseHandleProvider | None = None,
    ) -> None:
        self.gpu_uuid = require_gpu_uuid(gpu_uuid)
        self.path = fixed_gpu_lease_path(gpu_uuid, local_app_data=local_app_data)
        self._probe = probe or WindowsProcessProbe()
        self._handles = handles or WindowsLeaseHandleProvider()
        facts = self._probe.current()
        self.owner = LeaseOwner(
            facts.pid,
            facts.creation_time,
            facts.executable,
            facts.executable_sha256,
            1,
            gpu_uuid,
            nonce,
            runtime_id,
            action_kind,
            action_id,
        )
        self._handle: LeaseHandle | None = None

    def __enter__(self) -> "GpuLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._handles.try_acquire(self.path)
        if handle is None:
            content = self._handles.read_shared(self.path, MAX_LEASE_METADATA_BYTES)
            try:
                owner = _parse_owner(content, self.gpu_uuid)
            except AmbiguousGpuLeaseError as exc:
                raise AmbiguousGpuLeaseError(
                    "GPU Busy: the live lease has ambiguous owner metadata"
                ) from exc
            detail = asdict(owner) if owner else {"unknown": True}
            if owner is not None:
                try:
                    observed = self._probe.observe(owner.pid)
                except GpuLeaseError as exc:
                    raise AmbiguousGpuLeaseError(
                        "GPU Busy: the live lease owner cannot be validated", owner=detail
                    ) from exc
                if observed is None or not _same_process(owner, observed):
                    raise AmbiguousGpuLeaseError(
                        "GPU Busy: OS ownership and metadata disagree", owner=detail
                    )
            raise GpuBusyError("GPU Busy: an exact live LingBot Map owner holds the lease", owner=detail)

        try:
            previous = _parse_owner(handle.read(MAX_LEASE_METADATA_BYTES), self.gpu_uuid)
            if previous is not None:
                observed = self._probe.observe(previous.pid)
                if observed is not None and _same_process(previous, observed):
                    raise AmbiguousGpuLeaseError(
                        "GPU Lease metadata names a live exact owner after its OS handle disappeared",
                        owner=asdict(previous),
                    )
            encoded = (
                json.dumps(asdict(self.owner), sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            handle.replace(encoded)
            if _parse_owner(handle.read(MAX_LEASE_METADATA_BYTES), self.gpu_uuid) != self.owner:
                raise GpuLeaseError("GPU Lease owner publication could not be verified")
            self._handle = handle
            return self
        except Exception:
            handle.close()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        try:
            stored = _parse_owner(
                self._handle.read(MAX_LEASE_METADATA_BYTES), self.gpu_uuid
            )
            if stored != self.owner:
                raise GpuLeaseError("Refusing to release a GPU Lease owned by another identity")
            self._handle.replace(b"")
        finally:
            self._handle.close()
            self._handle = None
