"""Bounded Worker human logging and structured terminal diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
import threading
import traceback
from typing import Any, Callable, Mapping
import warnings

from .ipc import SCHEMA_VERSION, atomic_write_json


HUMAN_LOG_BYTES = 16 * 1024 * 1024
HUMAN_LOG_FILES = 4
MAX_EXCEPTION_TEXT_BYTES = 16 * 1024
MAX_TRACEBACK_FRAMES = 64


def _bounded_text(value: object, maximum: int = MAX_EXCEPTION_TEXT_BYTES) -> str:
    encoded = str(value).encode("utf-8", errors="replace")
    if len(encoded) <= maximum:
        return encoded.decode("utf-8", errors="replace")
    marker = "…".encode("utf-8")
    if maximum <= len(marker):
        return encoded[:maximum].decode("utf-8", errors="ignore")
    return (
        encoded[: maximum - len(marker)].decode("utf-8", errors="ignore")
        + "…"
    )


class RotatingHumanLog:
    """A thread-safe newest-first four-file UTF-8 log."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(
        self,
        directory: Path,
        *,
        file_bytes: int = HUMAN_LOG_BYTES,
        file_count: int = HUMAN_LOG_FILES,
        on_discard: Callable[[int], None] | None = None,
    ):
        if file_bytes < 1 or file_count < 1:
            raise ValueError("Human log bounds must be positive")
        self.directory = directory
        self.file_bytes = file_bytes
        self.file_count = file_count
        self.on_discard = on_discard
        self._lock = threading.RLock()
        self._stream = None
        self._size = 0
        self.discarded_bytes = 0
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        directory_stat = os.lstat(self.directory)
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_ISLNK(directory_stat.st_mode)
            or (
                getattr(directory_stat, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
        ):
            raise RuntimeError(
                "Human log directory must be an ordinary directory"
            )
        self._open_active()

    @property
    def buffer(self):
        return self

    @property
    def closed(self) -> bool:
        return self._closed

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def _path(self, generation: int) -> Path:
        return self.directory / (
            "human.log" if generation == 0 else f"human.log.{generation}"
        )

    def _open_active(self) -> None:
        path = self._path(0)
        stream = path.open("ab", buffering=0)
        handle_stat = os.fstat(stream.fileno())
        path_stat = os.lstat(path)
        identity = (
            handle_stat.st_dev,
            handle_stat.st_ino,
            handle_stat.st_size,
        )
        path_identity = (
            path_stat.st_dev,
            path_stat.st_ino,
            path_stat.st_size,
        )
        if (
            identity != path_identity
            or not stat.S_ISREG(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or path_stat.st_nlink != 1
            or (
                getattr(path_stat, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
        ):
            stream.close()
            raise RuntimeError(
                "Linked or non-file active human log is forbidden"
            )
        self._stream = stream
        self._size = path_stat.st_size
        if self._size > self.file_bytes:
            raise RuntimeError("Existing human.log exceeds its configured bound")

    def _rotate(self) -> None:
        assert self._stream is not None
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._stream = None
        discarded = 0
        oldest = self._path(self.file_count - 1)
        if oldest.exists():
            value = os.lstat(oldest)
            if not oldest.is_file() or oldest.is_symlink():
                raise RuntimeError("Linked or non-file human log generation is forbidden")
            discarded = value.st_size
            oldest.unlink()
        for generation in range(self.file_count - 2, -1, -1):
            source = self._path(generation)
            if source.exists():
                os.replace(source, self._path(generation + 1))
        self._open_active()
        if discarded:
            self.discarded_bytes += discarded
            if self.on_discard is not None:
                self.on_discard(self.discarded_bytes)

    def write(self, value) -> int:
        if isinstance(value, str):
            data = value.encode("utf-8", errors="replace")
            result = len(value)
        else:
            data = bytes(value)
            result = len(data)
        if not data:
            return result
        with self._lock:
            if self._closed:
                raise ValueError("write to closed human log")
            offset = 0
            while offset < len(data):
                if self._size == self.file_bytes:
                    self._rotate()
                available = self.file_bytes - self._size
                chunk = data[offset : offset + available]
                assert self._stream is not None
                self._stream.write(chunk)
                self._size += len(chunk)
                offset += len(chunk)
        return result

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._stream is None or self._stream.closed:
                return
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._stream is not None and not self._stream.closed:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()
            self._stream = None
            self._closed = True


class HumanLogSession:
    """Redirect arbitrary Worker stdout/stderr while preserving protocol stdout."""

    def __init__(
        self,
        directory: Path,
        *,
        on_discard: Callable[[int], None],
        on_warning: Callable[[str], None] | None = None,
        file_bytes: int = HUMAN_LOG_BYTES,
        file_count: int = HUMAN_LOG_FILES,
    ):
        self.stream = RotatingHumanLog(
            directory,
            file_bytes=file_bytes,
            file_count=file_count,
            on_discard=on_discard,
        )
        self.on_warning = on_warning
        self._stdout = None
        self._stderr = None
        self._showwarning = None

    @property
    def discarded_bytes(self) -> int:
        return self.stream.discarded_bytes

    def start(self) -> "HumanLogSession":
        if self._stdout is not None:
            raise RuntimeError("Human log capture is already active")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._showwarning = warnings.showwarning

        def showwarning(
            message,
            category,
            filename,
            lineno,
            file=None,
            line=None,
        ):
            formatted = warnings.formatwarning(
                message,
                category,
                filename,
                lineno,
                line,
            )
            self.stream.write(formatted)
            if self.on_warning is not None:
                try:
                    self.on_warning(
                        f"{category.__name__}: {_bounded_text(message, 4096)}"
                    )
                except Exception:
                    pass

        warnings.showwarning = showwarning
        sys.stdout = self.stream
        sys.stderr = self.stream
        return self

    def close(self) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            if self._showwarning is not None:
                warnings.showwarning = self._showwarning
            self._stdout = None
            self._stderr = None
            self._showwarning = None
        self.stream.close()

    def __enter__(self) -> "HumanLogSession":
        return self.start()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _exception_document(
    error_code: str,
    exception: BaseException,
) -> dict[str, Any]:
    extracted = traceback.TracebackException.from_exception(exception)
    frames = []
    for frame in list(extracted.stack)[-MAX_TRACEBACK_FRAMES:]:
        frames.append(
            {
                "file": _bounded_text(frame.filename, 32767),
                "line": int(frame.lineno),
                "function": _bounded_text(frame.name, 1024),
                "source": _bounded_text(frame.line or "", 4096),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "error_code": error_code,
        "type": _bounded_text(type(exception).__name__, 1024),
        "message": _bounded_text(exception),
        "traceback": frames,
    }


def write_terminal_diagnostic(
    job_dir: Path,
    job: Mapping[str, Any],
    *,
    error_code: str,
    state: str,
    phase: str,
    detail: object,
    discarded_log_bytes: int,
    exception: BaseException | None = None,
) -> None:
    """Write bounded common identity and exception records before terminal rename."""

    spec_path = job_dir / "job-spec.json"
    record = {
        "schema_version": SCHEMA_VERSION,
        "error_code": error_code,
        "category": "pipeline",
        "state": state,
        "phase": _bounded_text(phase, 512),
        "job_id": job["job_id"],
        "target_scene": dict(job["target_scene"]),
        "job_spec_sha256": _sha256(spec_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "machine_name": os.environ.get("COMPUTERNAME", ""),
        "username": (
            os.environ.get("USERNAME")
            or os.environ.get("USER")
            or ""
        ),
        "detail": _bounded_text(detail),
        "human_log": {
            "file_count": HUMAN_LOG_FILES,
            "file_bytes": HUMAN_LOG_BYTES,
            "truncated": discarded_log_bytes > 0,
            "discarded_bytes": discarded_log_bytes,
        },
    }
    atomic_write_json(job_dir / "diagnostic.json", record)
    if exception is None:
        return
    atomic_write_json(
        job_dir / "exception.json",
        _exception_document(error_code, exception),
    )


def write_bootstrap_diagnostic(
    spec_path: Path,
    exception: BaseException,
) -> None:
    """Retain structured launch evidence before a StatusStore can exist."""

    path = Path(os.path.abspath(spec_path))
    job_dir = path.parent
    if (
        path.name != "job-spec.json"
        or job_dir.parent.name != ".jobs"
        or re.fullmatch(r"job-[0-9a-f]{32}", job_dir.name) is None
    ):
        return
    directory_stat = os.lstat(job_dir)
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or stat.S_ISLNK(directory_stat.st_mode)
        or (
            getattr(directory_stat, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    ):
        return
    error_code = "launch.worker.bootstrap-failed"
    spec_sha256 = None
    try:
        spec_stat = os.lstat(path)
        if (
            stat.S_ISREG(spec_stat.st_mode)
            and not stat.S_ISLNK(spec_stat.st_mode)
            and spec_stat.st_size <= 1024 * 1024
        ):
            spec_sha256 = _sha256(path)
    except OSError:
        pass
    atomic_write_json(
        job_dir / "diagnostic.json",
        {
            "schema_version": SCHEMA_VERSION,
            "error_code": error_code,
            "category": "launch",
            "state": "failed",
            "phase": "bootstrap",
            "job_id": job_dir.name,
            "target_scene": {},
            "job_spec_sha256": spec_sha256,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "machine_name": os.environ.get("COMPUTERNAME", ""),
            "username": (
                os.environ.get("USERNAME")
                or os.environ.get("USER")
                or ""
            ),
            "detail": _bounded_text(exception),
            "human_log": {
                "file_count": HUMAN_LOG_FILES,
                "file_bytes": HUMAN_LOG_BYTES,
                "truncated": False,
                "discarded_bytes": 0,
            },
        },
    )
    atomic_write_json(
        job_dir / "exception.json",
        _exception_document(error_code, exception),
    )
