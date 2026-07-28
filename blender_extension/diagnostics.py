"""Bounded, allowlisted, privacy-preserving portable diagnostic reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
from typing import Any, Mapping
import uuid
import zipfile


REPORT_SCHEMA_VERSION = "1.0.0"
REDACTION_POLICY_VERSION = "1.0.0"
DIAGNOSTIC_RECORD_SCHEMA_VERSION = "1.0.0"

MAX_DIRECTORY_ENTRIES = 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_EVENTS_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024
MAX_PORTABLE_BYTES = 80 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_TEXT_VALUE_BYTES = 16 * 1024
MAX_CLIPBOARD_BYTES = 256 * 1024

_DIAGNOSTIC_NAME = re.compile(
    r"(?:job-[0-9a-f]{32}--[A-Za-z0-9][A-Za-z0-9._-]{0,180}"
    r"|[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{16}-[0-9a-f]{12,32})"
)
_JOB_ID = re.compile(r"job-[0-9a-f]{32}")
_GPU_UUID = re.compile(
    r"\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"
)
_WINDOWS_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\r\n\t\"'<>|]*"
)
_POSIX_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9._-])/(?:Users|home|var|tmp|opt|mnt|media|srv)/"
    r"[^\r\n\t\"'<>|]*"
)
_WINDOWS_USER = re.compile(
    r"(?i)(\\Users\\|/Users/|\\Documents and Settings\\)"
    r"([^\\/\r\n\t\"'<>|]+)"
)

_JSON_ALLOWLIST: Mapping[str, str] = {
    "diagnostic.json": "records/diagnostic.json",
    "exception.json": "records/exception.json",
    "failure.json": "records/failure.json",
    "incomplete.json": "records/incomplete.json",
    "job-spec.json": "records/job-spec.json",
    "job-control.json": "records/job-control.json",
    "status.json": "records/status.json",
    "setup-diagnostic.json": "records/setup-diagnostic.json",
    "model-download-diagnostic.json": "records/model-download-diagnostic.json",
    "worker.pid.json": "records/worker-identity.json",
}
_TEXT_ALLOWLIST: Mapping[str, str] = {
    "human.log": "logs/human.log",
    "human.log.1": "logs/human.log.1",
    "human.log.2": "logs/human.log.2",
    "human.log.3": "logs/human.log.3",
}
_EVENT_ALLOWLIST: Mapping[str, str] = {
    "events.jsonl": "records/events.jsonl",
}
_SOURCE_KEYS = {
    "source",
    "source_id",
    "source_identifier",
    "source_path",
    "source_sha256",
}
_PATH_KEYS = {
    "path",
    "absolute_path",
    "blend_path",
    "project_root",
    "managed_root",
    "executable",
    "capture_source",
    "source_path",
    "user_profile",
    "home",
    "cwd",
    "temp",
    "tmp",
}
_ENV_KEYS = {
    "env",
    "environment",
    "environment_variables",
    "process_environment",
}
_MACHINE_KEYS = {
    "hostname",
    "host_name",
    "machine",
    "machine_name",
    "computername",
    "computer_name",
}
_USER_KEYS = {
    "user",
    "username",
    "user_name",
    "account",
    "account_name",
}
_GPU_KEYS = {"gpu_uuid", "device_uuid"}


class DiagnosticReportError(RuntimeError):
    """A stable, bounded failure safe to show in Blender."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class PortableDiagnosticReport:
    """The immutable, already-redacted logical report before ZIP encoding."""

    entries: tuple[tuple[str, bytes], ...]
    manifest: Mapping[str, Any]
    clipboard_text: str


def _bounded_text(value: object, maximum: int = MAX_TEXT_VALUE_BYTES) -> str:
    text = str(value)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= maximum:
        return text
    marker = "…".encode("utf-8")
    if maximum <= len(marker):
        return encoded[:maximum].decode("utf-8", errors="ignore")
    return (
        encoded[: maximum - len(marker)].decode("utf-8", errors="ignore")
        + "…"
    )


def _is_reparse_point(path: Path) -> bool:
    try:
        value = os.lstat(path)
    except OSError:
        return False
    attribute = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attribute & marker)


def _plain_directory(path: Path, label: str) -> Path:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise DiagnosticReportError(
            "diagnostic-export-source-unavailable",
            f"{label} is unavailable: {_bounded_text(exc, 1024)}",
        ) from exc
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse_point(path)
    ):
        raise DiagnosticReportError(
            "diagnostic-export-linked-source",
            f"{label} must be an ordinary directory",
        )
    return path


def _file_identity(path: Path, maximum: int) -> tuple[int, int, int, int]:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise DiagnosticReportError(
            "diagnostic-export-file-unavailable",
            f"{path.name} is unavailable: {_bounded_text(exc, 1024)}",
        ) from exc
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse_point(path)
        or value.st_nlink != 1
    ):
        raise DiagnosticReportError(
            "diagnostic-export-linked-file",
            f"{path.name} is not an ordinary file",
        )
    if value.st_size > maximum:
        raise DiagnosticReportError(
            "diagnostic-export-file-too-large",
            f"{path.name} exceeds {maximum} bytes",
        )
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_stable(path: Path, maximum: int) -> bytes:
    before = _file_identity(path, maximum)
    try:
        with path.open("rb") as stream:
            handle_before = os.fstat(stream.fileno())
            handle_identity = (
                handle_before.st_dev,
                handle_before.st_ino,
                handle_before.st_size,
                handle_before.st_mtime_ns,
            )
            handle_attributes = getattr(
                handle_before,
                "st_file_attributes",
                0,
            )
            reparse_marker = getattr(
                stat,
                "FILE_ATTRIBUTE_REPARSE_POINT",
                0x400,
            )
            if (
                not stat.S_ISREG(handle_before.st_mode)
                or handle_before.st_nlink != 1
                or handle_attributes & reparse_marker
            ):
                raise DiagnosticReportError(
                    "diagnostic-export-linked-file",
                    f"{path.name} opened as a linked or non-file object",
                )
            if handle_identity != before:
                raise DiagnosticReportError(
                    "diagnostic-export-race",
                    f"{path.name} changed before it could be read",
                )
            data = stream.read(maximum + 1)
            handle_after = os.fstat(stream.fileno())
            if (
                (
                    handle_after.st_dev,
                    handle_after.st_ino,
                    handle_after.st_size,
                    handle_after.st_mtime_ns,
                )
                != handle_identity
                or handle_after.st_nlink != 1
            ):
                raise DiagnosticReportError(
                    "diagnostic-export-race",
                    f"{path.name} changed while the report was built",
                )
    except OSError as exc:
        raise DiagnosticReportError(
            "diagnostic-export-read-failed",
            f"{path.name} could not be read: {_bounded_text(exc, 1024)}",
        ) from exc
    if len(data) > maximum:
        raise DiagnosticReportError(
            "diagnostic-export-file-too-large",
            f"{path.name} changed beyond {maximum} bytes",
        )
    after = _file_identity(path, maximum)
    if before != after or len(data) != before[2]:
        raise DiagnosticReportError(
            "diagnostic-export-race",
            f"{path.name} changed while the report was built",
        )
    return data


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def _validate_tree(value: Any, depth: int = 1) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _validate_tree(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _validate_tree(child, depth + 1)


def _parse_json(data: bytes) -> Any:
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 BOM is forbidden")
    value = json.loads(
        data.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    _validate_tree(value)
    return value


def _canonical_json(value: Any) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encoded + b"\n"


def _publish_new_file(temporary: Path, target: Path) -> None:
    """Publish atomically without replacing a destination created by a race."""

    if os.name == "nt":
        # Windows os.rename is atomic and fails when the target already exists.
        os.rename(temporary, target)
        return
    # POSIX rename replaces an existing target, so use an atomic same-directory
    # hard-link publication and then remove the temporary name.
    os.link(temporary, target)
    temporary.unlink()


class _Redactor:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._source_aliases: dict[str, str] = {}
        candidates = {
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
            Path.home().name,
        }
        self._usernames = list(
            sorted((value for value in candidates if len(value) >= 2), key=len, reverse=True)
        )
        machines = {
            os.environ.get("COMPUTERNAME", ""),
            platform.node(),
            socket.gethostname(),
        }
        self._machines = list(
            sorted((value for value in machines if len(value) >= 2), key=len, reverse=True)
        )
        self._environment_values: list[str] = []

    def _source_alias(self, value: object) -> str:
        token = _bounded_text(value)
        alias = self._source_aliases.get(token)
        if alias is None:
            alias = f"[SOURCE-{len(self._source_aliases) + 1}]"
            self._source_aliases[token] = alias
        return alias

    def text(
        self,
        value: str,
        *,
        maximum: int = MAX_TEXT_VALUE_BYTES,
    ) -> str:
        if not self.enabled:
            return _bounded_text(value, maximum)
        result = _GPU_UUID.sub("GPU-[REDACTED]", value)
        result = _WINDOWS_USER.sub(
            lambda match: match.group(1) + "[USER]",
            result,
        )
        result = _WINDOWS_ABSOLUTE.sub("[ABSOLUTE_PATH]", result)
        result = _POSIX_ABSOLUTE.sub("[ABSOLUTE_PATH]", result)
        for username in self._usernames:
            result = re.sub(re.escape(username), "[USER]", result, flags=re.IGNORECASE)
        for machine in self._machines:
            result = re.sub(re.escape(machine), "[MACHINE]", result, flags=re.IGNORECASE)
        for environment_value in self._environment_values:
            result = result.replace(environment_value, "[ENVIRONMENT_VALUE]")
        for source_value, alias in self._source_aliases.items():
            result = result.replace(source_value, alias)
        return _bounded_text(result, maximum)

    @staticmethod
    def _strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from _Redactor._strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from _Redactor._strings(child)

    def value(self, value: Any, *, key: str = "", parents: tuple[str, ...] = ()) -> Any:
        if not self.enabled:
            if isinstance(value, dict):
                return {
                    str(child_key): self.value(child, key=str(child_key), parents=parents + (key,))
                    for child_key, child in value.items()
                }
            if isinstance(value, list):
                return [self.value(child, parents=parents + (key,)) for child in value]
            return self.text(value) if isinstance(value, str) else value

        folded = key.casefold()
        ancestry = {item.casefold() for item in parents}
        if folded in _ENV_KEYS:
            self._environment_values.extend(
                text
                for text in self._strings(value)
                if len(text) >= 2
            )
            return "[REDACTED_ENVIRONMENT]"
        if folded in _USER_KEYS:
            if isinstance(value, str) and len(value) >= 2:
                self._usernames.append(value)
            return "[REDACTED_USER]"
        if folded in _MACHINE_KEYS:
            if isinstance(value, str) and len(value) >= 2:
                self._machines.append(value)
            return "[REDACTED_MACHINE]"
        if folded in _GPU_KEYS or (
            folded == "uuid" and ("gpu" in ancestry or "device" in ancestry)
        ):
            return "GPU-[REDACTED]"
        source_context = bool(ancestry & _SOURCE_KEYS) or folded in _SOURCE_KEYS
        if (
            source_context
            and not isinstance(value, (dict, list))
            and folded in {
            "source",
            "sha256",
            "id",
            "identifier",
            "filename",
            "name",
            "source_sha256",
            "absolute_path",
            "scene_relative_path",
            }
        ):
            return self._source_alias(value)
        if folded in _PATH_KEYS or folded.endswith("_path") or folded.endswith("_root"):
            if isinstance(value, str) and not Path(value).is_absolute() and not value.startswith("\\\\"):
                return self.text(value)
            return "[REDACTED_ABSOLUTE_PATH]"
        if isinstance(value, dict):
            return {
                str(child_key): self.value(
                    child,
                    key=str(child_key),
                    parents=parents + (key,),
                )
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [
                self.value(child, parents=parents + (key,))
                for child in value
            ]
        if isinstance(value, str):
            return self.text(value)
        return value


def _error_record(filename: str, exc: BaseException) -> dict[str, str]:
    code = (
        exc.code
        if isinstance(exc, DiagnosticReportError)
        else "diagnostic-export-malformed-record"
    )
    return {
        "code": code,
        "file": filename,
        "message": _bounded_text(exc, 2048),
    }


def _summary(
    diagnostic_name: str,
    record: Mapping[str, Any] | None,
    *,
    redacted: bool,
    errors: tuple[Mapping[str, str], ...],
    excluded_count: int,
    scan_truncated: bool,
) -> bytes:
    code = "unknown"
    category = "unknown"
    state = "unknown"
    phase = "unknown"
    job_id = "unknown"
    if record is not None:
        code = _bounded_text(record.get("error_code", record.get("code", code)), 512)
        category = _bounded_text(record.get("category", category), 512)
        state = _bounded_text(record.get("state", state), 512)
        phase = _bounded_text(record.get("phase", phase), 512)
        job_id = _bounded_text(record.get("job_id", job_id), 512)
    lines = (
        "LingBot Map Portable Diagnostic Report",
        f"Report schema: {REPORT_SCHEMA_VERSION}",
        f"Redaction policy: {REDACTION_POLICY_VERSION}",
        f"Identity mode: {'redacted' if redacted else 'unredacted'}",
        f"Diagnostic: {diagnostic_name}",
        f"Stable error reference: {code}",
        f"Category: {category}",
        f"State: {state}",
        f"Phase: {phase}",
        f"Job ID: {job_id}",
        f"Excluded or unlisted entries: {excluded_count}",
        f"Directory scan truncated: {str(scan_truncated).lower()}",
        f"Record errors: {len(errors)}",
        "",
    )
    return "\n".join(lines).encode("utf-8")


def build_portable_report(
    diagnostic_directory: str | Path,
    *,
    redact: bool = True,
    versions: Mapping[str, object] | None = None,
) -> PortableDiagnosticReport:
    """Build one bounded logical report without following any filesystem link."""

    source = _plain_directory(Path(diagnostic_directory), "Diagnostic entry")
    _plain_directory(source.parent, "Diagnostics root")
    _plain_directory(source.parent.parent, "Diagnostic owner root")
    if not _DIAGNOSTIC_NAME.fullmatch(source.name):
        raise DiagnosticReportError(
            "diagnostic-export-unrecognized-source",
            "Diagnostic entry name is not recognized",
        )
    redactor = _Redactor(redact)
    present: dict[str, Path] = {}
    excluded_count = 0
    scan_truncated = False
    with os.scandir(source) as iterator:
        for index, entry in enumerate(iterator):
            if index >= MAX_DIRECTORY_ENTRIES:
                scan_truncated = True
                excluded_count += 1
                break
            if (
                entry.name in _JSON_ALLOWLIST
                or entry.name in _TEXT_ALLOWLIST
                or entry.name in _EVENT_ALLOWLIST
            ):
                present[entry.name] = Path(entry.path)
            else:
                excluded_count += 1

    entries: list[tuple[str, bytes]] = []
    errors: list[Mapping[str, str]] = []
    primary_record: Mapping[str, Any] | None = None
    parsed_json: list[tuple[str, str, Any]] = []

    for source_name, report_name in sorted(_JSON_ALLOWLIST.items()):
        path = present.get(source_name)
        if path is None:
            continue
        try:
            document = _parse_json(_read_stable(path, MAX_JSON_BYTES))
            parsed_json.append((source_name, report_name, document))
        except (DiagnosticReportError, UnicodeError, ValueError, TypeError) as exc:
            errors.append(_error_record(source_name, exc))

    # Discover identity values across every structured record first so a
    # traceback or message that appears before its JobSpec cannot leak them.
    for _source_name, _report_name, document in parsed_json:
        redactor.value(document)
    for source_name, report_name, document in parsed_json:
        try:
            redacted = redactor.value(document)
            if source_name == "diagnostic.json" and isinstance(redacted, dict):
                primary_record = redacted
            entries.append((report_name, _canonical_json(redacted)))
        except (UnicodeError, ValueError, TypeError) as exc:
            errors.append(_error_record(source_name, exc))

    for source_name, report_name in sorted(_EVENT_ALLOWLIST.items()):
        path = present.get(source_name)
        if path is None:
            continue
        try:
            raw = _read_stable(path, MAX_EVENTS_BYTES)
            output = io.BytesIO()
            for line_number, line in enumerate(raw.splitlines(), 1):
                if not line:
                    continue
                if len(line) > 64 * 1024:
                    raise ValueError(
                        f"event line {line_number} exceeds 65536 bytes"
                    )
                document = _parse_json(line)
                encoded = _canonical_json(redactor.value(document))
                if output.tell() + len(encoded) > MAX_EVENTS_BYTES:
                    raise DiagnosticReportError(
                        "diagnostic-export-events-too-large",
                        "Redacted events exceed their portable limit",
                    )
                output.write(encoded)
            entries.append((report_name, output.getvalue()))
        except (DiagnosticReportError, UnicodeError, ValueError, TypeError) as exc:
            errors.append(_error_record(source_name, exc))

    for source_name, report_name in sorted(_TEXT_ALLOWLIST.items()):
        path = present.get(source_name)
        if path is None:
            continue
        try:
            raw = _read_stable(path, MAX_LOG_BYTES)
            text = raw.decode("utf-8", errors="strict")
            if any(
                ord(character) < 32
                and character not in "\t\r\n\x1b"
                for character in text
            ):
                raise ValueError(
                    "human log contains binary control data"
                )
            encoded = redactor.text(
                text,
                maximum=MAX_LOG_BYTES,
            ).encode("utf-8")
            if len(encoded) > MAX_LOG_BYTES:
                encoded = encoded[:MAX_LOG_BYTES]
            entries.append((report_name, encoded))
        except (
            DiagnosticReportError,
            UnicodeError,
            ValueError,
        ) as exc:
            errors.append(_error_record(source_name, exc))

    version_record = redactor.value(
        {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "redaction_policy_version": REDACTION_POLICY_VERSION,
            "identity_mode": "redacted" if redact else "unredacted",
            "software": dict(versions or {}),
        }
    )
    entries.append(("versions.json", _canonical_json(version_record)))
    errors_tuple = tuple(errors)
    entries.append(
        (
            "errors.json",
            _canonical_json(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "errors": errors_tuple,
                }
            ),
        )
    )
    summary = _summary(
        (
            redactor.text(source.name)
            if source.name.startswith("job-")
            else "[DIAGNOSTIC-ENTRY]"
        ),
        primary_record,
        redacted=redact,
        errors=errors_tuple,
        excluded_count=excluded_count,
        scan_truncated=scan_truncated,
    )
    entries.append(("summary.txt", summary))
    entries.sort(key=lambda item: item[0])

    total = sum(len(data) for _name, data in entries)
    if total > MAX_PORTABLE_BYTES:
        raise DiagnosticReportError(
            "diagnostic-export-report-too-large",
            f"Portable content exceeds {MAX_PORTABLE_BYTES} bytes",
        )
    manifest_entries = [
        {
            "path": name,
            "length": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in entries
    ]
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "redaction_policy_version": REDACTION_POLICY_VERSION,
        "identity_mode": "redacted" if redact else "unredacted",
        "included_sections": [entry["path"] for entry in manifest_entries],
        "entries": manifest_entries,
        "excluded_entry_count": excluded_count,
        "directory_scan_truncated": scan_truncated,
    }
    manifest_bytes = _canonical_json(manifest)
    final_entries = tuple(
        sorted(entries + [("report-manifest.json", manifest_bytes)])
    )

    clipboard = (
        summary.decode("utf-8")
        + "\nPortable manifest:\n"
        + manifest_bytes.decode("utf-8")
        + "\nStructured record:\n"
        + (
            json.dumps(
                primary_record,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            if primary_record is not None
            else "(not available)"
        )
    )
    if len(clipboard.encode("utf-8")) > MAX_CLIPBOARD_BYTES:
        clipboard = (
            clipboard.encode("utf-8")[:MAX_CLIPBOARD_BYTES]
            .decode("utf-8", errors="ignore")
            + "\n[CLIPBOARD REPRESENTATION TRUNCATED]\n"
        )
    return PortableDiagnosticReport(final_entries, manifest, clipboard)


def export_portable_report(
    diagnostic_directory: str | Path,
    destination: str | Path,
    *,
    redact: bool = True,
    versions: Mapping[str, object] | None = None,
) -> Mapping[str, Any]:
    """Atomically write a deterministic ZIP containing only logical report files."""

    target = Path(destination)
    if target.suffix.casefold() != ".zip":
        target = target.with_suffix(".zip")
    parent = _plain_directory(target.parent, "Export destination")
    if target.exists() or target.is_symlink() or _is_reparse_point(target):
        raise DiagnosticReportError(
            "diagnostic-export-destination-occupied",
            "Export destination already exists",
        )
    report = build_portable_report(
        diagnostic_directory,
        redact=redact,
        versions=versions,
    )
    temporary = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, data in report.entries:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100600 << 16
                archive.writestr(info, data)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        _publish_new_file(temporary, target)
    except OSError as exc:
        raise DiagnosticReportError(
            "diagnostic-export-write-failed",
            f"Portable report could not be written: {_bounded_text(exc, 1024)}",
        ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    archive_bytes = _read_stable(
        target,
        MAX_PORTABLE_BYTES + 1024 * 1024,
    )
    return {
        "path": str(target),
        "length": len(archive_bytes),
        "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "manifest": report.manifest,
    }


def diagnostic_record(
    *,
    error_code: str,
    category: str,
    state: str,
    phase: str,
    job_id: str | None = None,
    target_scene: Mapping[str, object] | None = None,
    detail: object = "",
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Create the stable common identity shared by all diagnostic producers."""

    if not re.fullmatch(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+", error_code):
        raise DiagnosticReportError(
            "diagnostic-record-invalid-code",
            "Stable error code is invalid",
        )
    if category not in {"setup", "launch", "pipeline", "import", "lifecycle"}:
        raise DiagnosticReportError(
            "diagnostic-record-invalid-category",
            "Diagnostic category is invalid",
        )
    if state not in {"failed", "cancelled", "interrupted", "forced_termination"}:
        raise DiagnosticReportError(
            "diagnostic-record-invalid-state",
            "Diagnostic state is invalid",
        )
    if job_id is not None and not _JOB_ID.fullmatch(job_id):
        raise DiagnosticReportError(
            "diagnostic-record-invalid-job",
            "Diagnostic Job ID is invalid",
        )
    created = timestamp or datetime.now(timezone.utc)
    if created.tzinfo is None or created.utcoffset() is None:
        raise DiagnosticReportError(
            "diagnostic-record-invalid-time",
            "Diagnostic timestamp must have a timezone",
        )
    return {
        "schema_version": DIAGNOSTIC_RECORD_SCHEMA_VERSION,
        "error_code": error_code,
        "category": category,
        "state": state,
        "phase": _bounded_text(phase, 512),
        "job_id": job_id,
        "target_scene": dict(target_scene or {}),
        "created_utc": created.astimezone(timezone.utc).isoformat(),
        "machine_name": platform.node(),
        "username": (
            os.environ.get("USERNAME")
            or os.environ.get("USER")
            or Path.home().name
        ),
        "detail": _bounded_text(detail),
    }


def retain_extension_diagnostic(
    project_root: str | Path,
    *,
    error_code: str,
    category: str,
    state: str,
    phase: str,
    detail: object,
    target_scene: Mapping[str, object] | None = None,
    job_id: str | None = None,
) -> Path:
    """Persist one bounded Extension-side failure without copying staging data."""

    root = _plain_directory(Path(project_root), "Project Result Root")
    diagnostics = _plain_directory(root / "diagnostics", "diagnostics")
    identity = job_id if job_id and _JOB_ID.fullmatch(job_id) else f"job-{uuid.uuid4().hex}"
    suffix = f"extension-{category}-{state}"
    destination = diagnostics / f"{identity}--{suffix}"
    if destination.exists():
        destination = diagnostics / f"{identity}--{suffix}-{uuid.uuid4().hex[:8]}"
    try:
        destination.mkdir()
        record = diagnostic_record(
            error_code=error_code,
            category=category,
            state=state,
            phase=phase,
            job_id=job_id,
            target_scene=target_scene,
            detail=detail,
        )
        record["operation_id"] = identity
        temporary = destination / f".diagnostic.{uuid.uuid4().hex}.tmp"
        with temporary.open("xb") as stream:
            data = _canonical_json(record)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination / "diagnostic.json")
        return destination
    except OSError as exc:
        raise DiagnosticReportError(
            "diagnostic-retention-failed",
            f"Extension diagnostic could not be retained: {_bounded_text(exc, 1024)}",
        ) from exc
