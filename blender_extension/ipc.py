"""Strict, bounded readers and writers for versioned Job IPC."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
import uuid


SCHEMA_VERSION = "1.0.0"
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_EVENT_LINE_BYTES = 64 * 1024
MAX_JSON_DEPTH = 32
MAX_DIAGNOSTIC_TEXT = 16 * 1024


class IpcError(ValueError):
    pass


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IpcError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise IpcError(f"non-finite JSON number: {value}")


def _validate_tree(value: Any, depth: int = 1) -> None:
    if depth > MAX_JSON_DEPTH:
        raise IpcError(f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise IpcError("JSON object keys must be strings")
            _validate_tree(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _validate_tree(child, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise IpcError("JSON numbers must be finite")


def parse_json_bytes(data: bytes, *, maximum: int = MAX_DOCUMENT_BYTES) -> Any:
    if len(data) > maximum:
        raise IpcError(f"JSON input exceeds {maximum} bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise IpcError("UTF-8 BOM is forbidden")
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IpcError(f"invalid UTF-8 JSON: {exc}") from exc
    _validate_tree(value)
    return value


def read_json(path: Path, *, maximum: int = MAX_DOCUMENT_BYTES) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            raise IpcError(f"IPC path is not an ordinary file: {path.name}")
        size = path.stat().st_size
        if size > maximum:
            raise IpcError(f"JSON input exceeds {maximum} bytes")
        return parse_json_bytes(path.read_bytes(), maximum=maximum)
    except OSError as exc:
        raise IpcError(f"cannot read IPC file {path.name}: {exc}") from exc


def parse_json_line(data: bytes) -> Any:
    if not data.endswith(b"\n"):
        raise IpcError("JSON Lines record is not newline-terminated")
    if len(data) > MAX_EVENT_LINE_BYTES:
        raise IpcError(f"JSON Lines record exceeds {MAX_EVENT_LINE_BYTES} bytes")
    body = data[:-1]
    if body.endswith(b"\r"):
        body = body[:-1]
    return parse_json_bytes(body, maximum=MAX_EVENT_LINE_BYTES)


def encode_json(document: Mapping[str, object]) -> bytes:
    _validate_tree(document)
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IpcError(f"cannot encode IPC JSON: {exc}") from exc
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise IpcError(f"JSON output exceeds {MAX_DOCUMENT_BYTES} bytes")
    return encoded


def atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    encoded = encode_json(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + 1.0
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def require_exact_object(
    value: Any, fields: set[str], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise IpcError(f"{label} has unknown or missing fields")
    return value


def require_text(value: Any, *, label: str, maximum: int = MAX_DIAGNOSTIC_TEXT) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise IpcError(f"{label} is empty or exceeds its UTF-8 limit")
    return value


def require_schema(document: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    result = require_exact_object(document, fields, label=label)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise IpcError(f"unsupported {label} schema version")
    return result
