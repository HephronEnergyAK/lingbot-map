"""Catalog-only acquisition for immutable Reconstruction and Auxiliary Models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Callable, Iterable, Mapping
import urllib.error
import urllib.request
import uuid

from .runtime_setup import (
    CancellationToken,
    RuntimeLock,
    RuntimeSetupError,
    SetupCancelled,
    current_process_identity,
    sha256_file,
)


DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_CHECKPOINT_SIZE = 16 * 1024 * 1024
MODEL_CATALOG_VERSION = "1.0.0"


class ModelStoreError(RuntimeSetupError):
    """A bounded model Setup error suitable for persistent UI state."""


class UnsupportedModelError(ModelStoreError):
    """A local file does not exactly match a catalogued checksum."""


class MissingModelArtifactsError(ModelStoreError):
    def __init__(self, entries: Iterable["ModelEntry"]):
        self.entries = tuple(entries)
        detail = ", ".join(
            f"{entry.id} ({entry.artifact.filename}, sha256:{entry.artifact.sha256})"
            for entry in self.entries
        )
        super().__init__(
            "Offline Setup is missing exact catalogued model artifacts: "
            + detail
            + ". Run the explicit online Download action or import the byte-for-byte catalog match."
        )


@dataclass(frozen=True)
class ModelArtifact:
    filename: str
    length: int
    sha256: str
    url: str
    source_repository: str
    source_revision: str


@dataclass(frozen=True)
class ModelLicenseRecord:
    status: str
    spdx_expression: str
    covers_weights: bool
    captured_license_text_path: str
    captured_license_text_sha256: str
    record_path: str
    record_sha256: str
    copyright_or_attribution_source: str
    evidence_basis: str
    release_gate: str


@dataclass(frozen=True)
class ModelEntry:
    id: str
    display_name: str
    role: str
    required: bool
    architecture: str
    input_contract: Mapping[str, object]
    serialization: Mapping[str, object]
    artifact: ModelArtifact
    license_record: ModelLicenseRecord


@dataclass(frozen=True)
class ResumeState:
    url: str
    expected_length: int
    completed: int
    validator_header: str
    validator_value: str
    range_supported: bool


Progress = Callable[[int, int], None]


def _duplicates_rejected(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelStoreError(f"Duplicate JSON key in model metadata: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates_rejected
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelStoreError(f"Cannot read strict model metadata {path.name}: {exc}") from exc
    if not isinstance(document, dict):
        raise ModelStoreError(f"Model metadata must be an object: {path.name}")
    return document


def _safe_relative(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ":" in relative or "\\" in relative:
        raise ModelStoreError(f"Unsafe catalog path: {relative!r}")
    target = (root / value).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ModelStoreError(f"Catalog path escapes its bundle: {relative!r}")
    return target


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse)


class ModelCatalog:
    """A strict, immutable catalog copied into both Extension and Runtime."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        document = _read_json(self.path)
        if set(document) != {"catalog_version", "models"}:
            raise ModelStoreError("Model Catalog has unknown or missing top-level fields")
        if document["catalog_version"] != MODEL_CATALOG_VERSION:
            raise ModelStoreError(f"Unsupported Model Catalog version: {document['catalog_version']}")
        raw_models = document["models"]
        if not isinstance(raw_models, list) or not raw_models:
            raise ModelStoreError("Model Catalog must contain at least one model")
        self.version = str(document["catalog_version"])
        self.sha256 = sha256_file(self.path)
        entries = tuple(self._parse_entry(raw) for raw in raw_models)
        if len({entry.id for entry in entries}) != len(entries):
            raise ModelStoreError("Model Catalog contains duplicate model IDs")
        if len({entry.artifact.sha256 for entry in entries}) != len(entries):
            raise ModelStoreError("Model Catalog assigns one checksum to multiple entries")
        if not any(entry.role == "reconstruction" and entry.required for entry in entries):
            raise ModelStoreError("Model Catalog lacks a required Reconstruction Model")
        self.entries = entries
        self._by_id = {entry.id: entry for entry in entries}
        self._by_sha256 = {entry.artifact.sha256: entry for entry in entries}

    def _parse_entry(self, raw: object) -> ModelEntry:
        if not isinstance(raw, dict):
            raise ModelStoreError("Every Model Catalog entry must be an object")
        required = {
            "id", "display_name", "role", "required", "architecture",
            "input_contract", "serialization", "artifact", "license_record",
        }
        if set(raw) != required:
            raise ModelStoreError(f"Model entry has unknown or missing fields: {raw.get('id', '?')}")
        model_id = str(raw["id"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", model_id):
            raise ModelStoreError(f"Unsafe model ID: {model_id!r}")
        role = str(raw["role"])
        if role not in {"reconstruction", "auxiliary"}:
            raise ModelStoreError(f"Unknown model role: {role!r}")
        artifact_raw = raw["artifact"]
        if not isinstance(artifact_raw, dict) or set(artifact_raw) != {
            "filename", "length", "sha256", "url", "source_repository", "source_revision"
        }:
            raise ModelStoreError(f"Invalid artifact record for {model_id}")
        filename = str(artifact_raw["filename"])
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ModelStoreError(f"Unsafe model filename: {filename!r}")
        digest = str(artifact_raw["sha256"]).lower()
        revision = str(artifact_raw["source_revision"])
        url = str(artifact_raw["url"])
        repository = str(artifact_raw["source_repository"])
        if not _valid_digest(digest) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ModelStoreError(f"Model {model_id} lacks an exact checksum or revision")
        if not url.startswith("https://") or not repository.startswith("https://") or revision not in url:
            raise ModelStoreError(f"Model {model_id} must use an immutable HTTPS source URL")
        artifact = ModelArtifact(
            filename, int(artifact_raw["length"]), digest, url, repository, revision
        )
        if artifact.length <= 0:
            raise ModelStoreError(f"Model {model_id} has an invalid expected length")
        serialization = raw["serialization"]
        if not isinstance(serialization, dict) or serialization.get("arbitrary_pickle_allowed") is not False:
            raise ModelStoreError(f"Model {model_id} does not forbid arbitrary pickle")
        license_raw = raw["license_record"]
        license_fields = {
            "status", "spdx_expression", "covers_weights", "captured_license_text_path",
            "captured_license_text_sha256", "record_path", "record_sha256",
            "copyright_or_attribution_source", "evidence_basis", "release_gate",
        }
        if not isinstance(license_raw, dict) or set(license_raw) != license_fields:
            raise ModelStoreError(f"Invalid Model License Record for {model_id}")
        license_record = ModelLicenseRecord(**license_raw)
        for relative, expected in (
            (license_record.captured_license_text_path, license_record.captured_license_text_sha256),
            (license_record.record_path, license_record.record_sha256),
        ):
            path = _safe_relative(self.path.parent, relative)
            if not path.is_file() or not _valid_digest(expected) or sha256_file(path) != expected:
                raise ModelStoreError(f"Model License Record hash mismatch for {model_id}: {relative}")
        return ModelEntry(
            model_id,
            str(raw["display_name"]),
            role,
            bool(raw["required"]),
            str(raw["architecture"]),
            dict(raw["input_contract"]),
            dict(serialization),
            artifact,
            license_record,
        )

    def by_id(self, model_id: str) -> ModelEntry:
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise ModelStoreError(f"Unknown Model Catalog ID: {model_id}") from exc

    def by_sha256(self, digest: str) -> ModelEntry | None:
        return self._by_sha256.get(digest)


def _default_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ModelStore:
    def __init__(self, managed_root: Path, catalog: ModelCatalog, *, opener=None):
        self.managed_root = managed_root.resolve()
        self.catalog = catalog
        self.models_root = self.managed_root / "models"
        self.downloads_root = self.managed_root / "model-downloads"
        self.diagnostics_root = self.managed_root / "model-download-diagnostics"
        self.locks_root = self.managed_root / "model-locks"
        self.opener = opener or _default_opener()

    def model_path(self, entry: ModelEntry) -> Path:
        return self.models_root / entry.artifact.sha256

    def artifact_path(self, entry: ModelEntry) -> Path:
        return self.model_path(entry) / "artifact"

    def partial_path(self, entry: ModelEntry) -> Path:
        return self.downloads_root / f"{entry.artifact.sha256}.partial"

    def partial_metadata_path(self, entry: ModelEntry) -> Path:
        return self.downloads_root / f"{entry.artifact.sha256}.partial.json"

    def quick_status(self, entry: ModelEntry) -> str:
        directory = self.model_path(entry)
        artifact = self.artifact_path(entry)
        registration = directory / "model-registration.json"
        if (
            (directory.exists() and is_reparse_point(directory))
            or (artifact.exists() and is_reparse_point(artifact))
            or not registration.is_file()
        ):
            return "missing"
        try:
            document = _read_json(registration)
        except ModelStoreError:
            return "invalid"
        if (
            not self._registration_matches(document, entry)
            or not artifact.is_file()
            or artifact.stat().st_size != entry.artifact.length
        ):
            return "invalid"
        return "registered"

    def validate(self, entry: ModelEntry) -> Path:
        directory = self.model_path(entry)
        artifact = self.artifact_path(entry)
        if not directory.is_dir() or is_reparse_point(directory):
            raise ModelStoreError(f"Managed model is missing or linked: {entry.id}")
        if not artifact.is_file() or is_reparse_point(artifact):
            raise ModelStoreError(f"Managed model artifact is missing or linked: {entry.id}")
        if artifact.stat().st_size != entry.artifact.length or sha256_file(artifact) != entry.artifact.sha256:
            raise ModelStoreError(f"Managed model checksum validation failed: {entry.id}")
        registration = _read_json(directory / "model-registration.json")
        if not self._registration_matches(registration, entry):
            raise ModelStoreError(f"Managed model registration does not match the Catalog: {entry.id}")
        return artifact

    def ensure_offline(self, model_ids: Iterable[str]) -> tuple[Path, ...]:
        entries = tuple(self.catalog.by_id(model_id) for model_id in model_ids)
        missing = [entry for entry in entries if self.quick_status(entry) != "registered"]
        if missing:
            raise MissingModelArtifactsError(missing)
        return tuple(self.validate(entry) for entry in entries)

    def acquire(
        self,
        model_id: str,
        *,
        offline: bool,
        online_access: bool,
        cancellation: CancellationToken | None = None,
        progress: Progress | None = None,
    ) -> Path:
        entry = self.catalog.by_id(model_id)
        cancellation = cancellation or CancellationToken()
        progress = progress or (lambda _completed, _total: None)
        if self.quick_status(entry) == "registered":
            return self.validate(entry)
        if offline:
            raise MissingModelArtifactsError((entry,))
        if not online_access:
            raise ModelStoreError(
                "Blender Online Access is disabled. Enable it or use Local Import/Offline Setup."
            )
        self.models_root.mkdir(parents=True, exist_ok=True)
        owner = current_process_identity()
        lock_path = self.locks_root / f"{entry.artifact.sha256}.lock"
        with RuntimeLock(lock_path, owner):
            if self.quick_status(entry) == "registered":
                return self.validate(entry)
            if self.model_path(entry).exists():
                self._quarantine_model(entry, "invalid-existing-model")
            return self._download_and_publish(entry, cancellation, progress, owner.nonce)

    def import_local(
        self,
        source: Path,
        *,
        expected_model_id: str | None = None,
        cancellation: CancellationToken | None = None,
        progress: Progress | None = None,
    ) -> tuple[ModelEntry, Path]:
        if source.is_symlink() or (source.exists() and is_reparse_point(source)):
            raise UnsupportedModelError("Local model import refuses reparse paths")
        source = source.resolve(strict=True)
        cancellation = cancellation or CancellationToken()
        progress = progress or (lambda _completed, _total: None)
        if not source.is_file() or is_reparse_point(source):
            raise UnsupportedModelError("Local model import requires one ordinary non-reparse file")
        with source.open("rb") as stream:
            initial = os.fstat(stream.fileno())
            digest = self._hash_stream(stream, initial.st_size, cancellation, progress)
            entry = self.catalog.by_sha256(digest)
            if entry is None:
                raise UnsupportedModelError(
                    f"Unsupported Model: sha256:{digest} is not present in Model Catalog {self.catalog.version}"
                )
            if expected_model_id is not None and entry.id != expected_model_id:
                raise UnsupportedModelError(
                    f"Selected file matches {entry.id}, not the requested {expected_model_id}"
                )
            if initial.st_size != entry.artifact.length:
                raise UnsupportedModelError(f"Catalogued model length mismatch: {entry.id}")
            self.models_root.mkdir(parents=True, exist_ok=True)
            owner = current_process_identity()
            lock_path = self.locks_root / f"{entry.artifact.sha256}.lock"
            with RuntimeLock(lock_path, owner):
                if self.quick_status(entry) == "registered":
                    return entry, self.validate(entry)
                if self.model_path(entry).exists():
                    self._quarantine_model(entry, "invalid-existing-model")
                if shutil.disk_usage(self.models_root).free < entry.artifact.length:
                    raise ModelStoreError(
                        f"Local import requires {entry.artifact.length} additional bytes of free space"
                    )
                staging = self.models_root / f".staging-{entry.artifact.sha256[:16]}-{owner.nonce}"
                staging.mkdir()
                target = staging / "artifact"
                try:
                    stream.seek(0)
                    with target.open("wb") as output:
                        self._copy_stream(stream, output, entry.artifact.length, cancellation, progress)
                    final_stat = os.fstat(stream.fileno())
                    if (initial.st_dev, initial.st_ino, initial.st_size) != (
                        final_stat.st_dev, final_stat.st_ino, final_stat.st_size
                    ):
                        raise ModelStoreError("Local model identity changed while it was imported")
                    self._finish_staging(entry, staging, target)
                    os.replace(staging, self.model_path(entry))
                    return entry, self.validate(entry)
                except Exception as exc:
                    if staging.exists():
                        self._quarantine_paths(
                            entry, "cancelled-local-import" if isinstance(exc, SetupCancelled) else "failed-local-import",
                            [staging], {"error": str(exc)},
                        )
                    raise

    def discard_partial(self, model_id: str, *, confirmed: bool) -> None:
        if not confirmed:
            raise ModelStoreError("Discarding a model partial requires explicit confirmation")
        entry = self.catalog.by_id(model_id)
        owner = current_process_identity()
        with RuntimeLock(self.locks_root / f"{entry.artifact.sha256}.lock", owner):
            self.partial_path(entry).unlink(missing_ok=True)
            self.partial_metadata_path(entry).unlink(missing_ok=True)

    def _download_and_publish(
        self,
        entry: ModelEntry,
        cancellation: CancellationToken,
        progress: Progress,
        nonce: str,
    ) -> Path:
        self.downloads_root.mkdir(parents=True, exist_ok=True)
        partial = self.partial_path(entry)
        state = self._load_resume_state(entry)
        response = None
        append = False
        completed_without_response = bool(
            state is not None and state.completed == entry.artifact.length
        )
        if state is not None and not completed_without_response:
            request = urllib.request.Request(
                entry.artifact.url,
                headers={
                    "User-Agent": "LingBotMap-Reconstruction/0.1.0",
                    "Range": f"bytes={state.completed}-",
                    "If-Range": state.validator_value,
                },
            )
            try:
                response = self.opener.open(request, timeout=60)
            except urllib.error.HTTPError:
                response = None
            if response is not None and self._valid_resume_response(response, entry, state):
                append = True
            else:
                if response is not None:
                    response.close()
                self._quarantine_paths(entry, "invalid-resume-response", [partial, self.partial_metadata_path(entry)], {})
                state = None
                response = None
        if response is None and not completed_without_response:
            request = urllib.request.Request(
                entry.artifact.url,
                headers={"User-Agent": "LingBotMap-Reconstruction/0.1.0"},
            )
            response = self.opener.open(request, timeout=60)
            state = self._state_from_full_response(response, entry)
            append = False
        assert state is not None
        mode = "ab" if append else "wb"
        completed = state.completed if append or completed_without_response else 0
        checkpoint_at = completed + DOWNLOAD_CHECKPOINT_SIZE
        try:
            if response is not None:
                with response, partial.open(mode) as output:
                    while completed < entry.artifact.length:
                        if cancellation.cancelled:
                            raise SetupCancelled(f"Model download cancelled: {entry.id}")
                        chunk = response.read(min(DOWNLOAD_CHUNK_SIZE, entry.artifact.length - completed))
                        if not chunk:
                            raise ModelStoreError(f"Model download ended at {completed} of {entry.artifact.length} bytes")
                        output.write(chunk)
                        completed += len(chunk)
                        progress(completed, entry.artifact.length)
                        if completed >= checkpoint_at:
                            output.flush()
                            os.fsync(output.fileno())
                            self._write_resume_state(
                                entry,
                                ResumeState(
                                    state.url,
                                    state.expected_length,
                                    completed,
                                    state.validator_header,
                                    state.validator_value,
                                    state.range_supported,
                                ),
                            )
                            checkpoint_at = completed + DOWNLOAD_CHECKPOINT_SIZE
                    if response.read(1):
                        raise ModelStoreError("Model response exceeded its catalogued length")
                    output.flush()
                    os.fsync(output.fileno())
        except Exception as exc:
            retained = ResumeState(
                state.url,
                state.expected_length,
                completed,
                state.validator_header,
                state.validator_value,
                state.range_supported,
            )
            if completed > 0 and state.range_supported and state.validator_value:
                self._write_resume_state(entry, retained)
            else:
                self._quarantine_paths(entry, "non-resumable-interruption", [partial], {"error": str(exc)})
                self.partial_metadata_path(entry).unlink(missing_ok=True)
            raise
        self._write_resume_state(
            entry,
            ResumeState(
                state.url, state.expected_length, completed,
                state.validator_header, state.validator_value, state.range_supported,
            ),
        )
        with partial.open("rb") as stream:
            actual = self._hash_stream(stream, entry.artifact.length, cancellation, progress)
        if actual != entry.artifact.sha256:
            self._quarantine_paths(
                entry, "checksum-mismatch", [partial, self.partial_metadata_path(entry)],
                {"expected_sha256": entry.artifact.sha256, "actual_sha256": actual},
            )
            raise ModelStoreError(f"Downloaded model checksum mismatch: {entry.id}")
        staging = self.models_root / f".staging-{entry.artifact.sha256[:16]}-{nonce}"
        staging.mkdir()
        target = staging / "artifact"
        os.replace(partial, target)
        try:
            self._finish_staging(entry, staging, target)
            os.replace(staging, self.model_path(entry))
        except Exception as exc:
            if staging.exists():
                self._quarantine_paths(entry, "failed-model-publication", [staging], {"error": str(exc)})
            raise
        finally:
            self.partial_metadata_path(entry).unlink(missing_ok=True)
        return self.validate(entry)

    def _state_from_full_response(self, response, entry: ModelEntry) -> ResumeState:
        if response.status != 200 or not str(response.geturl()).startswith("https://"):
            response.close()
            raise ModelStoreError(f"Model server did not return a complete HTTPS response: {entry.id}")
        if int(response.headers.get("Content-Length", -1)) != entry.artifact.length:
            response.close()
            raise ModelStoreError(f"Model server length differs from Catalog: {entry.id}")
        validator_header, validator_value = self._response_validator(response)
        range_supported = response.headers.get("Accept-Ranges", "").lower().strip() == "bytes"
        return ResumeState(
            entry.artifact.url,
            entry.artifact.length,
            0,
            validator_header,
            validator_value,
            range_supported,
        )

    @staticmethod
    def _response_validator(response) -> tuple[str, str]:
        etag = response.headers.get("ETag")
        if etag:
            return "ETag", etag
        modified = response.headers.get("Last-Modified")
        return ("Last-Modified", modified) if modified else ("", "")

    def _valid_resume_response(self, response, entry: ModelEntry, state: ResumeState) -> bool:
        if response.status != 206 or not str(response.geturl()).startswith("https://"):
            return False
        expected_range = f"bytes {state.completed}-{entry.artifact.length - 1}/{entry.artifact.length}"
        if response.headers.get("Content-Range") != expected_range:
            return False
        if int(response.headers.get("Content-Length", -1)) != entry.artifact.length - state.completed:
            return False
        if response.headers.get(state.validator_header) != state.validator_value:
            return False
        return response.headers.get("Accept-Ranges", "").lower().strip() == "bytes"

    def _load_resume_state(self, entry: ModelEntry) -> ResumeState | None:
        partial = self.partial_path(entry)
        metadata = self.partial_metadata_path(entry)
        if not partial.exists() and not metadata.exists():
            return None
        try:
            document = _read_json(metadata)
            state = ResumeState(**document)
            valid = (
                partial.is_file()
                and not is_reparse_point(partial)
                and state.url == entry.artifact.url
                and state.expected_length == entry.artifact.length
                and state.completed <= partial.stat().st_size
                and 0 < state.completed <= entry.artifact.length
                and state.validator_header in {"ETag", "Last-Modified"}
                and bool(state.validator_value)
                and state.range_supported is True
            )
        except (ModelStoreError, OSError, TypeError):
            valid = False
        if not valid:
            self._quarantine_paths(entry, "invalid-partial-metadata", [partial, metadata], {})
            return None
        if partial.stat().st_size > state.completed:
            # Bytes after the last fsync+metadata checkpoint were never committed
            # as resumable state; discard only that uncertain tail.
            with partial.open("r+b") as stream:
                stream.truncate(state.completed)
        return state

    def _write_resume_state(self, entry: ModelEntry, state: ResumeState) -> None:
        path = self.partial_metadata_path(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(state.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _finish_staging(self, entry: ModelEntry, staging: Path, target: Path) -> None:
        if target.stat().st_size != entry.artifact.length or sha256_file(target) != entry.artifact.sha256:
            raise ModelStoreError(f"Staged model checksum validation failed: {entry.id}")
        (staging / "model-registration.json").write_text(
            json.dumps(self._registration(entry), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _registration(self, entry: ModelEntry) -> Mapping[str, object]:
        return {
            "registration_version": 1,
            "catalog_sha256": self.catalog.sha256,
            "catalog_version": self.catalog.version,
            "model_id": entry.id,
            "role": entry.role,
            "filename": entry.artifact.filename,
            "length": entry.artifact.length,
            "sha256": entry.artifact.sha256,
            "source_repository": entry.artifact.source_repository,
            "source_revision": entry.artifact.source_revision,
            "source_url": entry.artifact.url,
            "license_status": entry.license_record.status,
            "license_spdx_expression": entry.license_record.spdx_expression,
        }

    @staticmethod
    def _registration_matches(document: Mapping[str, object], entry: ModelEntry) -> bool:
        # Catalog versions may coexist and reuse already verified bytes by
        # checksum; the registration records provenance but does not pin a
        # compatible consumer to the originating Catalog hash or model ID.
        return (
            document.get("registration_version") == 1
            and document.get("length") == entry.artifact.length
            and document.get("sha256") == entry.artifact.sha256
        )

    @staticmethod
    def _hash_stream(
        stream, total: int, cancellation: CancellationToken, progress: Progress
    ) -> str:
        digest = hashlib.sha256()
        completed = 0
        while True:
            if cancellation.cancelled:
                raise SetupCancelled("Local model hashing was cancelled")
            chunk = stream.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            completed += len(chunk)
            progress(completed, total)
        return digest.hexdigest()

    @staticmethod
    def _copy_stream(
        source, destination, total: int, cancellation: CancellationToken, progress: Progress
    ) -> None:
        completed = 0
        while True:
            if cancellation.cancelled:
                raise SetupCancelled("Local model copy was cancelled")
            chunk = source.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            destination.write(chunk)
            completed += len(chunk)
            progress(completed, total)
        destination.flush()
        os.fsync(destination.fileno())

    def _quarantine_model(self, entry: ModelEntry, reason: str) -> None:
        self._quarantine_paths(entry, reason, [self.model_path(entry)], {})

    def _quarantine_paths(
        self,
        entry: ModelEntry,
        reason: str,
        paths: Iterable[Path],
        detail: Mapping[str, object],
    ) -> Path:
        existing = [path for path in paths if path.exists() or path.is_symlink()]
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.diagnostics_root / (
            f"{stamp}-{entry.artifact.sha256[:16]}-{uuid.uuid4().hex[:12]}"
        )
        destination.mkdir()
        for path in existing:
            os.replace(path, destination / path.name)
        record = {
            "reason": reason,
            "model_id": entry.id,
            "sha256": entry.artifact.sha256,
            **detail,
        }
        (destination / "model-download-diagnostic.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination


@lru_cache(maxsize=1)
def bundled_model_catalog() -> ModelCatalog:
    return ModelCatalog(Path(__file__).resolve().parent / "runtime_bundle" / "model-catalog.json")
