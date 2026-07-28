"""Build and inspect the clean-tag LingBot Map Release Package."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


RELEASE_TOOL_VERSION = "1.0.0"
SEMVER_TAG = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
COMMIT_ID = re.compile(r"^[0-9a-f]{40,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|FIXME|CHANGEME|REPLACE[_ -]?ME|COMING SOON)\b",
    re.IGNORECASE,
)
GENERATED_NOTICE = "runtime_bundle/NOTICES/runtime-dependencies.txt"
PACKAGE_POLICY = "release/package-policy.json"
DEPENDENCY_LICENSES = "release/dependency-licenses.json"
SBOM_PATH = "release/SBOM.spdx.json"
PROVENANCE_PATH = "release/provenance.json"
MANIFEST_PATH = "release/package-manifest.json"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MAX_WHEEL_EXPANDED_BYTES = 64 * 1024 * 1024
APPLICATION_LICENSE_EXPRESSION = "Apache-2.0"
STABLE_RELEASE_GATE_NAMES = (
    "ada_16gb_native_qualification",
    "full_native_release_suite",
    "official_platform_readiness_review",
    "weight_specific_model_licenses",
)
ENGINEERING_QUALIFICATION_SCOPE = {
    "native_host": "windows-11-x64",
    "blender": "5.2-lts",
    "reference_gpu": "nvidia-geforce-rtx-5090-blackwell-32-gb",
    "ada_qualified": False,
    "general_nvidia_qualified": False,
}


class ReleasePackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildMetadata:
    repository: str
    commit: str
    tag: str
    workflow_ref: str
    run_id: str
    run_attempt: str
    created_utc: str


@dataclass(frozen=True)
class ReleasePolicy:
    schema_version: str
    repository: str
    extension_id: str
    maximum_file_bytes: int
    maximum_archive_bytes: int
    source_files: tuple[str, ...]
    generated_files: tuple[str, ...]
    required_manual_locales: tuple[str, ...]
    stable_release_gates: tuple[tuple[str, bool], ...]
    forbidden_suffixes: tuple[str, ...]
    forbidden_path_components: tuple[str, ...]

    @property
    def archive_files(self) -> tuple[str, ...]:
        return tuple(sorted(self.source_files + self.generated_files))

    @property
    def pending_stable_release_gates(self) -> tuple[str, ...]:
        return tuple(
            name for name, complete in self.stable_release_gates if not complete
        )


@dataclass(frozen=True)
class ReleaseArtifacts:
    archive: Path
    sha256: Path
    sbom: Path
    provenance: Path


def _strict_json_bytes(data: bytes, description: str):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReleasePackageError(
                    f"duplicate JSON key in {description}: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReleasePackageError(
                    f"non-finite JSON value in {description}: {value}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleasePackageError(
            f"invalid strict JSON in {description}"
        ) from exc


def _strict_json_file(path: Path):
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReleasePackageError(f"could not read {path}") from exc
    return _strict_json_bytes(data, str(path))


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_archive_name(name: str) -> str:
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise ReleasePackageError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != name
    ):
        raise ReleasePackageError(f"unsafe archive path: {name!r}")
    return name


def _validate_path_inventory(paths: tuple[str, ...], description: str) -> None:
    folded = [path.casefold() for path in paths]
    if not paths or len(folded) != len(set(folded)):
        raise ReleasePackageError(
            f"{description} must be non-empty and contain no "
            "case-insensitive duplicates"
        )
    for path in paths:
        _safe_archive_name(path)
    if tuple(sorted(paths, key=str.casefold)) != paths:
        raise ReleasePackageError(f"{description} must be sorted")


def load_policy(path: str | Path) -> ReleasePolicy:
    document = _strict_json_file(Path(path))
    required = {
        "schema_version",
        "repository",
        "extension_id",
        "maximum_file_bytes",
        "maximum_archive_bytes",
        "source_files",
        "generated_files",
        "required_manual_locales",
        "stable_release_gates",
        "forbidden_suffixes",
        "forbidden_path_components",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ReleasePackageError("release package policy fields are invalid")
    for field in (
        "source_files",
        "generated_files",
        "required_manual_locales",
        "forbidden_suffixes",
        "forbidden_path_components",
    ):
        if (
            not isinstance(document[field], list)
            or not all(isinstance(item, str) for item in document[field])
        ):
            raise ReleasePackageError(
                f"release package policy list is invalid: {field}"
            )
    stable_release_gates = document["stable_release_gates"]
    if (
        not isinstance(stable_release_gates, dict)
        or set(stable_release_gates) != set(STABLE_RELEASE_GATE_NAMES)
        or not all(
            isinstance(stable_release_gates[name], bool)
            for name in STABLE_RELEASE_GATE_NAMES
        )
    ):
        raise ReleasePackageError("stable release gate inventory is invalid")
    policy = ReleasePolicy(
        schema_version=str(document["schema_version"]),
        repository=str(document["repository"]),
        extension_id=str(document["extension_id"]),
        maximum_file_bytes=int(document["maximum_file_bytes"]),
        maximum_archive_bytes=int(document["maximum_archive_bytes"]),
        source_files=tuple(document["source_files"]),
        generated_files=tuple(document["generated_files"]),
        required_manual_locales=tuple(document["required_manual_locales"]),
        stable_release_gates=tuple(
            (name, stable_release_gates[name])
            for name in STABLE_RELEASE_GATE_NAMES
        ),
        forbidden_suffixes=tuple(document["forbidden_suffixes"]),
        forbidden_path_components=tuple(
            document["forbidden_path_components"]
        ),
    )
    if policy.schema_version != "1.0.0":
        raise ReleasePackageError("release package policy must be 1.0.0")
    if not policy.repository or not policy.extension_id:
        raise ReleasePackageError("release package identity is incomplete")
    if (
        policy.maximum_file_bytes < 1
        or policy.maximum_archive_bytes < policy.maximum_file_bytes
    ):
        raise ReleasePackageError("release package size bounds are invalid")
    _validate_path_inventory(
        policy.source_files,
        "source allowlist",
    )
    _validate_path_inventory(
        policy.generated_files,
        "generated allowlist",
    )
    if set(policy.source_files) & set(policy.generated_files):
        raise ReleasePackageError("source and generated allowlists overlap")
    required_generated = {
        PACKAGE_POLICY,
        DEPENDENCY_LICENSES,
        GENERATED_NOTICE,
        SBOM_PATH,
        PROVENANCE_PATH,
        MANIFEST_PATH,
    }
    if set(policy.generated_files) != required_generated:
        raise ReleasePackageError(
            "generated allowlist does not match release metadata contract"
        )
    if tuple(policy.required_manual_locales) != ("en_US", "zh_HANT"):
        raise ReleasePackageError(
            "release package must require en_US and zh_HANT manuals"
        )
    return policy


def _load_license_catalog(path: Path) -> dict:
    document = _strict_json_file(path)
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "evidence", "packages"}
        or document["schema_version"] != "1.0.0"
        or not isinstance(document["evidence"], str)
        or not document["evidence"].strip()
        or not isinstance(document["packages"], list)
    ):
        raise ReleasePackageError("dependency license catalog is invalid")
    seen = set()
    for package in document["packages"]:
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "spdx_license_expression",
            "metadata_url",
        }:
            raise ReleasePackageError(
                "dependency license record fields are invalid"
            )
        identity = (package["name"], package["version"])
        if (
            identity in seen
            or not all(
                isinstance(package[field], str) and package[field].strip()
                for field in package
            )
            or package["spdx_license_expression"] == "NOASSERTION"
            or not package["metadata_url"].startswith("https://")
        ):
            raise ReleasePackageError(
                f"dependency license record is invalid: {identity!r}"
            )
        seen.add(identity)
    if document["packages"] != sorted(
        document["packages"],
        key=lambda item: (item["name"].casefold(), item["version"]),
    ):
        raise ReleasePackageError(
            "dependency license records must be sorted"
        )
    return document


def _validate_metadata(metadata: BuildMetadata, policy: ReleasePolicy) -> str:
    if metadata.repository != policy.repository:
        raise ReleasePackageError(
            "repository does not match the release package policy"
        )
    if not COMMIT_ID.fullmatch(metadata.commit):
        raise ReleasePackageError("release commit is not a full Git object ID")
    match = SEMVER_TAG.fullmatch(metadata.tag)
    if match is None:
        raise ReleasePackageError("release tag is not strict v-prefixed SemVer")
    version = metadata.tag[1:]
    if match.group(1) != "0":
        raise ReleasePackageError(
            "the engineering Release Package producer accepts only 0.x tags"
        )
    if not metadata.workflow_ref.startswith(
        f"{metadata.repository}/.github/workflows/release.yml@refs/tags/"
    ):
        raise ReleasePackageError(
            "workflow identity is not the tagged release workflow"
        )
    if not metadata.workflow_ref.endswith("/" + metadata.tag):
        raise ReleasePackageError(
            "workflow identity does not agree with the release tag"
        )
    if not metadata.run_id.isdecimal() or not metadata.run_attempt.isdecimal():
        raise ReleasePackageError("workflow run identity is invalid")
    try:
        created = datetime.fromisoformat(
            metadata.created_utc.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ReleasePackageError(
            "release creation timestamp is invalid"
        ) from exc
    if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(
        created
    ):
        raise ReleasePackageError(
            "release creation timestamp must be UTC"
        )
    return version


def _toml(data: bytes, description: str) -> dict:
    try:
        return tomllib.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ReleasePackageError(f"invalid TOML in {description}") from exc


def _wheel_entries(data: bytes, description: str) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as wheel:
            infos = wheel.infolist()
            names = [info.filename for info in infos]
            folded = [name.casefold() for name in names]
            if len(folded) != len(set(folded)):
                raise ReleasePackageError(
                    f"case-insensitive duplicate entry in bundled wheel "
                    f"{description}"
                )
            total = 0
            for info in infos:
                _safe_archive_name(info.filename)
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or stat.S_ISLNK(mode)
                    or info.file_size > MAX_WHEEL_EXPANDED_BYTES
                ):
                    raise ReleasePackageError(
                        f"unsafe bundled wheel entry: {description}: "
                        f"{info.filename}"
                    )
                total += info.file_size
            if total > MAX_WHEEL_EXPANDED_BYTES:
                raise ReleasePackageError(
                    f"bundled wheel expands beyond its size bound: {description}"
                )
            return {info.filename: wheel.read(info) for info in infos}
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        if isinstance(exc, ReleasePackageError):
            raise
        raise ReleasePackageError(
            f"bundled wheel is not a valid ZIP: {description}"
        ) from exc


def _wheel_metadata(
    data: bytes,
    description: str,
) -> tuple[str, str, str]:
    entries = _wheel_entries(data, description)
    metadata_names = [
        name for name in entries if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_names) != 1:
        raise ReleasePackageError(
            f"bundled wheel has invalid METADATA: {description}"
        )
    metadata_bytes = entries[metadata_names[0]]
    message = BytesParser(policy=email_policy).parsebytes(metadata_bytes)
    name = message.get("Name", "").strip()
    version = message.get("Version", "").strip()
    license_expression = message.get("License-Expression", "").strip()
    if not name or not version or not license_expression:
        raise ReleasePackageError(
            f"bundled wheel identity is incomplete: {description}"
        )
    return name, version, license_expression


def _validate_wheel_record(
    entries: dict[str, bytes],
    dist_info: str,
    description: str,
) -> None:
    record_name = f"{dist_info}/RECORD"
    try:
        text = entries[record_name].decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (KeyError, UnicodeError, csv.Error) as exc:
        raise ReleasePackageError(
            f"bundled wheel RECORD is invalid: {description}"
        ) from exc
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise ReleasePackageError(
                f"bundled wheel RECORD is invalid: {description}"
            )
        path, digest, size = row
        _safe_archive_name(path)
        if path in records:
            raise ReleasePackageError(
                f"bundled wheel RECORD is duplicated: {description}"
            )
        records[path] = (digest, size)
    if set(records) != set(entries):
        raise ReleasePackageError(
            f"bundled wheel RECORD inventory is incomplete: {description}"
        )
    for path, data in entries.items():
        digest, size = records[path]
        if path == record_name:
            valid = digest == "" and size == ""
        else:
            encoded = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()
            ).rstrip(b"=").decode("ascii")
            valid = digest == f"sha256={encoded}" and size == str(len(data))
        if not valid:
            raise ReleasePackageError(
                f"bundled wheel RECORD mismatch: {description}: {path}"
            )


def _runtime_inventory(entries: dict[str, bytes]) -> dict:
    inventory = _strict_json_bytes(
        entries["runtime_bundle/runtime-inventory.json"],
        "runtime inventory",
    )
    if (
        not isinstance(inventory, dict)
        or set(inventory)
        != {"schema_version", "platform", "python", "packages"}
        or inventory["schema_version"] != 1
        or inventory["platform"] != "windows-x64"
        or inventory["python"] != "3.10.20"
        or not isinstance(inventory["packages"], list)
    ):
        raise ReleasePackageError("runtime inventory is invalid")
    identities = []
    for package in inventory["packages"]:
        if (
            not isinstance(package, dict)
            or set(package) != {"name", "version"}
            or not isinstance(package["name"], str)
            or not isinstance(package["version"], str)
            or not package["name"]
            or not package["version"]
        ):
            raise ReleasePackageError("runtime package identity is invalid")
        identities.append((package["name"], package["version"]))
    if len(identities) != len(set(identities)):
        raise ReleasePackageError("runtime package identities are duplicated")
    _validate_runtime_lock_inventory(entries, set(identities))
    return inventory


def _windows_lock_marker_applies(marker: str | None) -> bool:
    if marker is None:
        return True
    results = {
        "sys_platform == 'linux'": False,
        "sys_platform == 'win32'": True,
        (
            "platform_machine == 'AMD64' or platform_machine == 'aarch64' "
            "or platform_machine == 'amd64' or platform_machine == 'arm64' "
            "or platform_machine == 'x86_64'"
        ): True,
    }
    if marker not in results:
        raise ReleasePackageError(
            f"frozen Runtime lock contains an unsupported marker: {marker!r}"
        )
    return results[marker]


def _validate_runtime_lock_inventory(
    entries: dict[str, bytes],
    inventory: set[tuple[str, str]],
) -> None:
    lock = _toml(entries["runtime_bundle/uv.lock"], "frozen Runtime lock")
    if (
        lock.get("version") != 1
        or lock.get("requires-python") != "==3.10.20"
        or not isinstance(lock.get("package"), list)
    ):
        raise ReleasePackageError("frozen Runtime lock format is invalid")
    packages = {}
    for package in lock["package"]:
        if (
            not isinstance(package, dict)
            or not isinstance(package.get("name"), str)
            or not isinstance(package.get("version"), str)
            or not isinstance(package.get("dependencies", []), list)
            or package["name"] in packages
        ):
            raise ReleasePackageError(
                "frozen Runtime lock package inventory is invalid"
            )
        packages[package["name"]] = package
    root = packages.get("lingbot-map-runtime")
    if root is None or root.get("source") != {"virtual": "."}:
        raise ReleasePackageError("frozen Runtime lock root is invalid")

    selected = set()
    pending = list(root.get("dependencies", []))
    while pending:
        dependency = pending.pop()
        if (
            not isinstance(dependency, dict)
            or not isinstance(dependency.get("name"), str)
        ):
            raise ReleasePackageError(
                "frozen Runtime dependency record is invalid"
            )
        if not _windows_lock_marker_applies(dependency.get("marker")):
            continue
        name = dependency["name"]
        package = packages.get(name)
        if package is None:
            raise ReleasePackageError(
                f"frozen Runtime dependency is missing: {name}"
            )
        identity = (name, package["version"])
        if identity in selected:
            continue
        selected.add(identity)
        pending.extend(package.get("dependencies", []))
    if selected != inventory:
        missing = sorted(selected - inventory)
        unexpected = sorted(inventory - selected)
        raise ReleasePackageError(
            "frozen Runtime inventory does not match the Windows x64 lock "
            f"closure (missing={missing!r}, unexpected={unexpected!r})"
        )


def _validate_dependency_licenses(
    inventory: dict,
    license_catalog: dict,
) -> dict[tuple[str, str], dict]:
    expected = {
        (package["name"], package["version"])
        for package in inventory["packages"]
    }
    records = {
        (package["name"], package["version"]): package
        for package in license_catalog["packages"]
    }
    if expected != set(records):
        missing = sorted(expected - set(records))
        unexpected = sorted(set(records) - expected)
        raise ReleasePackageError(
            "dependency license catalog does not match frozen Runtime "
            f"(missing={missing!r}, unexpected={unexpected!r})"
        )
    return records


def _validate_model_licenses(
    entries: dict[str, bytes],
    *,
    stable_release: bool,
) -> bool:
    catalog = _strict_json_bytes(
        entries["runtime_bundle/model-catalog.json"],
        "Model Catalog",
    )
    if (
        not isinstance(catalog, dict)
        or set(catalog) != {"catalog_version", "models"}
        or not isinstance(catalog["catalog_version"], str)
        or not isinstance(catalog["models"], list)
    ):
        raise ReleasePackageError("Model Catalog is invalid")
    all_weights_resolved = True
    for model in catalog["models"]:
        license_record = model.get("license_record")
        if not isinstance(license_record, dict):
            raise ReleasePackageError("Model Catalog license record is missing")
        covers_weights = license_record.get("covers_weights")
        gate = license_record.get("release_gate")
        for field in (
            "captured_license_text_path",
            "captured_license_text_sha256",
            "record_path",
            "record_sha256",
            "spdx_expression",
            "evidence_basis",
        ):
            if not isinstance(license_record.get(field), str) or not (
                license_record[field]
            ):
                raise ReleasePackageError(
                    f"Model Catalog license record field is missing: {field}"
                )
        for path_field, digest_field in (
            ("captured_license_text_path", "captured_license_text_sha256"),
            ("record_path", "record_sha256"),
        ):
            path = "runtime_bundle/" + license_record[path_field]
            digest = license_record[digest_field]
            if path not in entries or not SHA256.fullmatch(digest):
                raise ReleasePackageError(
                    f"Model Catalog license evidence is missing: {path}"
                )
            if _sha256(entries[path]) != digest:
                raise ReleasePackageError(
                    f"Model Catalog license evidence checksum mismatch: {path}"
                )
        if covers_weights is not True:
            all_weights_resolved = False
            if gate != "blocked-for-1.0":
                raise ReleasePackageError(
                    "unresolved weight license is not blocked for 1.0"
                )
    if stable_release and not all_weights_resolved:
        raise ReleasePackageError(
            "weight-specific license evidence blocks version 1.0.0"
        )
    return all_weights_resolved


def _validate_schema_catalog(entries: dict[str, bytes]) -> str:
    catalog = _strict_json_bytes(
        entries["runtime_bundle/schemas/catalog.json"],
        "schema catalog",
    )
    if (
        not isinstance(catalog, dict)
        or set(catalog) != {"catalog_version", "contracts"}
        or not isinstance(catalog["catalog_version"], str)
        or not isinstance(catalog["contracts"], list)
    ):
        raise ReleasePackageError("schema catalog is invalid")
    for contract in catalog["contracts"]:
        if not isinstance(contract, dict) or not {
            "path",
            "sha256",
        } <= set(contract):
            raise ReleasePackageError("schema catalog entry is invalid")
        path = "runtime_bundle/schemas/" + contract["path"]
        digest = contract["sha256"]
        if path not in entries or not SHA256.fullmatch(digest):
            raise ReleasePackageError(
                f"schema catalog path or checksum is invalid: {path}"
            )
        if _sha256(entries[path]) != digest:
            raise ReleasePackageError(
                f"schema catalog checksum mismatch: {path}"
            )
    return catalog["catalog_version"]


def _validate_lock_and_wheels(
    entries: dict[str, bytes],
    version: str,
) -> dict[str, str]:
    wheel_paths = {
        "lingbot-map": (
            f"runtime_bundle/wheels/"
            f"lingbot_map-{version}-py3-none-any.whl"
        ),
        "lingbot-map-worker": (
            f"runtime_bundle/wheels/"
            f"lingbot_map_worker-{version}-py3-none-any.whl"
        ),
    }
    identities = {}
    for expected_name, path in wheel_paths.items():
        if path not in entries:
            raise ReleasePackageError(f"required bundled wheel is missing: {path}")
        actual_name, actual_version, license_expression = _wheel_metadata(
            entries[path],
            path,
        )
        if (
            actual_name != expected_name
            or actual_version != version
            or license_expression != APPLICATION_LICENSE_EXPRESSION
        ):
            raise ReleasePackageError(
                f"bundled wheel identity, version, or license mismatch: {path}"
            )
        identities[expected_name] = _sha256(entries[path])

    lock = _toml(entries["runtime_bundle/uv.lock"], "frozen Runtime lock")
    if lock.get("version") != 1 or not isinstance(lock.get("package"), list):
        raise ReleasePackageError("frozen Runtime lock format is invalid")
    locked = {
        package.get("name"): package
        for package in lock["package"]
        if package.get("name") in wheel_paths
    }
    if set(locked) != set(wheel_paths):
        raise ReleasePackageError(
            "frozen Runtime lock is missing application wheels"
        )
    for name, path in wheel_paths.items():
        package = locked[name]
        if package.get("version") != version:
            raise ReleasePackageError(
                f"frozen Runtime lock version mismatch for {name}"
            )
        source_path = package.get("source", {}).get("path", "")
        expected_source = path.removeprefix("runtime_bundle/")
        if source_path != expected_source:
            raise ReleasePackageError(
                f"frozen Runtime lock path mismatch for {name}"
            )
        hashes = {
            wheel.get("hash", "").removeprefix("sha256:")
            for wheel in package.get("wheels", [])
            if wheel.get("filename") == PurePosixPath(path).name
        }
        if hashes != {identities[name]}:
            raise ReleasePackageError(
                f"frozen Runtime lock checksum mismatch for {name}"
            )
    return identities


def _validate_application_wheel_sources(
    source_root: Path,
    entries: dict[str, bytes],
    version: str,
) -> None:
    specifications = (
        (
            f"runtime_bundle/wheels/lingbot_map-{version}-py3-none-any.whl",
            source_root / "lingbot_map",
            "lingbot_map",
            source_root / "LICENSE.txt",
            "licenses/LICENSE.txt",
        ),
        (
            (
                "runtime_bundle/wheels/"
                f"lingbot_map_worker-{version}-py3-none-any.whl"
            ),
            source_root / "worker/src/lingbot_map_worker",
            "lingbot_map_worker",
            source_root / "worker/NOTICE.txt",
            "licenses/NOTICE.txt",
        ),
    )
    for (
        wheel_path,
        package_root,
        package_name,
        notice_source,
        notice_name,
    ) in specifications:
        wheel_entries = _wheel_entries(entries[wheel_path], wheel_path)
        source_entries = {}
        if not package_root.is_dir():
            raise ReleasePackageError(
                f"tagged application source is missing: {package_root}"
            )
        for path in package_root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ReleasePackageError(
                    f"tagged application source is a symlink: {path}"
                )
            relative = path.relative_to(package_root).as_posix()
            data = path.read_bytes()
            source_entries[f"{package_name}/{relative}"] = data
        dist_info = f"{package_name}-{version}.dist-info"
        dist_info_entries = {
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/top_level.txt",
            f"{dist_info}/RECORD",
            f"{dist_info}/{notice_name}",
        }
        expected_entries = set(source_entries) | dist_info_entries
        if set(wheel_entries) != expected_entries:
            missing = sorted(expected_entries - set(wheel_entries))
            unexpected = sorted(set(wheel_entries) - expected_entries)
            raise ReleasePackageError(
                f"bundled wheel install payload does not match tagged source: "
                f"{wheel_path} (missing={missing!r}, "
                f"unexpected={unexpected!r})"
            )
        mismatched = sorted(
            name
            for name in source_entries
            if (
                wheel_entries[name].replace(b"\r\n", b"\n")
                if PurePosixPath(name).suffix.casefold() in {".py", ".txt"}
                else wheel_entries[name]
            )
            != (
                source_entries[name].replace(b"\r\n", b"\n")
                if PurePosixPath(name).suffix.casefold() in {".py", ".txt"}
                else source_entries[name]
            )
        )
        if mismatched:
            raise ReleasePackageError(
                f"bundled wheel source does not match tagged source: "
                f"{wheel_path}: {mismatched!r}"
            )
        try:
            notice_bytes = notice_source.read_bytes()
        except OSError as exc:
            raise ReleasePackageError(
                f"tagged application notice is missing: {notice_source}"
            ) from exc
        if wheel_entries[f"{dist_info}/{notice_name}"].replace(
            b"\r\n", b"\n"
        ) != notice_bytes.replace(b"\r\n", b"\n"):
            raise ReleasePackageError(
                f"bundled wheel notice does not match tagged source: "
                f"{wheel_path}"
            )
        if wheel_entries[f"{dist_info}/top_level.txt"] != (
            package_name + "\n"
        ).encode("ascii"):
            raise ReleasePackageError(
                f"bundled wheel top-level package is invalid: {wheel_path}"
            )
        wheel_metadata = BytesParser(policy=email_policy).parsebytes(
            wheel_entries[f"{dist_info}/WHEEL"]
        )
        if (
            wheel_metadata.get("Root-Is-Purelib", "").casefold() != "true"
            or wheel_metadata.get_all("Tag", []) != ["py3-none-any"]
        ):
            raise ReleasePackageError(
                f"bundled wheel platform contract is invalid: {wheel_path}"
            )
        _validate_wheel_record(wheel_entries, dist_info, wheel_path)


def _validate_stable_release_gates(
    policy: ReleasePolicy,
    *,
    stable_release: bool,
) -> None:
    if stable_release and policy.pending_stable_release_gates:
        raise ReleasePackageError(
            "stable 1.0 release gates remain incomplete: "
            f"{policy.pending_stable_release_gates!r}"
        )


def _version_from_entries(
    source_root: Path,
    entries: dict[str, bytes],
    metadata: BuildMetadata,
    policy: ReleasePolicy,
) -> tuple[str, dict[str, str], bool]:
    version = _validate_metadata(metadata, policy)
    manifest = _toml(
        entries["blender_manifest.toml"],
        "Blender Extension manifest",
    )
    if manifest.get("id") != policy.extension_id:
        raise ReleasePackageError("Extension manifest ID mismatch")
    if manifest.get("version") != version:
        raise ReleasePackageError(
            "tag and Extension manifest versions do not agree"
        )
    stable = int(version.split(".", 1)[0]) >= 1
    weights_resolved = _validate_model_licenses(
        entries,
        stable_release=stable,
    )
    _validate_stable_release_gates(policy, stable_release=stable)

    project_versions = {
        "repository": _toml(
            (source_root / "pyproject.toml").read_bytes(),
            "repository pyproject",
        ).get("project", {}).get("version"),
        "worker": _toml(
            (source_root / "worker/pyproject.toml").read_bytes(),
            "Worker pyproject",
        ).get("project", {}).get("version"),
        "runtime": _toml(
            entries["runtime_bundle/pyproject.toml"],
            "Runtime pyproject",
        ).get("project", {}).get("version"),
    }
    if set(project_versions.values()) != {version}:
        raise ReleasePackageError(
            "repository, Worker, Runtime, tag, and manifest versions "
            f"do not agree: {project_versions!r}"
        )
    wheel_hashes = _validate_lock_and_wheels(entries, version)
    _validate_application_wheel_sources(source_root, entries, version)
    return version, wheel_hashes, weights_resolved


def _dependency_notice(
    inventory: dict,
    records: dict[tuple[str, str], dict],
) -> bytes:
    lines = [
        "LingBot Map frozen Runtime dependency notices",
        "",
        "The Release Package bundles application wheels and a frozen lock.",
        "Third-party Runtime dependencies are acquired only during an explicit",
        "Setup action. Exact-version license records used by the SPDX SBOM:",
        "",
    ]
    for package in inventory["packages"]:
        record = records[(package["name"], package["version"])]
        lines.extend(
            [
                f"{package['name']} {package['version']}",
                f"  SPDX: {record['spdx_license_expression']}",
                f"  Metadata: {record['metadata_url']}",
            ]
        )
    lines.extend(
        [
            "",
            "The installed distributions carry their complete license texts",
            "and notices. This inventory does not replace those upstream terms.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _spdx_id(name: str) -> str:
    return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name)


def _build_sbom(
    *,
    metadata: BuildMetadata,
    version: str,
    inventory: dict,
    records: dict[tuple[str, str], dict],
    wheel_hashes: dict[str, str],
    policy: ReleasePolicy,
) -> dict:
    packages = [
        {
            "SPDXID": "SPDXRef-Package-Extension",
            "name": policy.extension_id,
            "versionInfo": version,
            "downloadLocation": (
                f"https://github.com/{metadata.repository}/tree/{metadata.commit}"
            ),
            "filesAnalyzed": False,
            "licenseConcluded": "GPL-3.0-or-later",
            "licenseDeclared": "GPL-3.0-or-later",
            "copyrightText": (
                "Copyright 2026 LingBot Map Contributors"
            ),
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": (
                        f"pkg:blender/{policy.extension_id}@{version}"
                    ),
                }
            ],
        }
    ]
    for package in inventory["packages"]:
        identity = (package["name"], package["version"])
        record = records[identity]
        item = {
            "SPDXID": _spdx_id(package["name"]),
            "name": package["name"],
            "versionInfo": package["version"],
            "downloadLocation": record["metadata_url"],
            "filesAnalyzed": False,
            "licenseConcluded": record["spdx_license_expression"],
            "licenseDeclared": record["spdx_license_expression"],
            "copyrightText": "See exact-version upstream metadata",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": (
                        f"pkg:pypi/{package['name']}@{package['version']}"
                    ),
                }
            ],
        }
        if package["name"] in wheel_hashes:
            item["checksums"] = [
                {
                    "algorithm": "SHA256",
                    "checksumValue": wheel_hashes[package["name"]],
                }
            ]
        packages.append(item)

    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-Package-Extension",
        },
        {
            "spdxElementId": "SPDXRef-Package-Extension",
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": _spdx_id("lingbot-map"),
        },
        {
            "spdxElementId": "SPDXRef-Package-Extension",
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": _spdx_id("lingbot-map-worker"),
        },
    ]
    for package in inventory["packages"]:
        if package["name"] in {"lingbot-map", "lingbot-map-worker"}:
            continue
        relationships.append(
            {
                "spdxElementId": _spdx_id("lingbot-map-worker"),
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": _spdx_id(package["name"]),
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{policy.extension_id}-{version}",
        "documentNamespace": (
            f"https://github.com/{metadata.repository}/releases/"
            f"{metadata.tag}/sbom/{metadata.commit}"
        ),
        "creationInfo": {
            "created": metadata.created_utc,
            "creators": [
                "Organization: LingBot Map Contributors",
                f"Tool: release_package.py-{RELEASE_TOOL_VERSION}",
            ],
        },
        "packages": packages,
        "relationships": relationships,
    }


def _build_provenance(
    metadata: BuildMetadata,
    version: str,
    entries: dict[str, bytes],
) -> dict:
    return {
        "schema_version": "1.0.0",
        "builder": {
            "kind": "github-actions",
            "workflow_ref": metadata.workflow_ref,
            "run_id": metadata.run_id,
            "run_attempt": metadata.run_attempt,
            "tool": f"release_package.py-{RELEASE_TOOL_VERSION}",
        },
        "source": {
            "kind": "git-archive-of-tagged-commit",
            "repository": metadata.repository,
            "commit": metadata.commit,
            "tag": metadata.tag,
            "version": version,
            "created_utc": metadata.created_utc,
        },
        "materials": [
            {
                "path": path,
                "length": len(data),
                "sha256": _sha256(data),
            }
            for path, data in sorted(entries.items())
        ],
    }


def _package_manifest(
    *,
    metadata: BuildMetadata,
    version: str,
    entries: dict[str, bytes],
    schema_catalog_version: str,
    model_catalog_version: str,
    weight_licenses_resolved: bool,
    policy: ReleasePolicy,
) -> dict:
    stable = int(version.split(".", 1)[0]) >= 1
    return {
        "schema_version": "1.0.0",
        "package": {
            "id": "lingbot_map_reconstruction",
            "version": version,
            "archive": f"lingbot_map_reconstruction-{version}.zip",
        },
        "source": {
            "repository": metadata.repository,
            "commit": metadata.commit,
            "tag": metadata.tag,
            "workflow_ref": metadata.workflow_ref,
            "run_id": metadata.run_id,
            "run_attempt": metadata.run_attempt,
        },
        "versions": {
            "extension": version,
            "worker": version,
            "model_core": version,
            "schema_catalog": schema_catalog_version,
            "model_catalog": model_catalog_version,
        },
        "claims": {
            "engineering_0_x": not stable,
            "stable_1_0": stable,
            "official_extensions_platform_ready": False,
            "weight_specific_licenses_resolved": weight_licenses_resolved,
            "custom_updater": False,
            "engineering_qualification_scope": (
                ENGINEERING_QUALIFICATION_SCOPE
            ),
            "stable_release_gates": dict(policy.stable_release_gates),
        },
        "files": [
            {
                "path": path,
                "length": len(data),
                "sha256": _sha256(data),
            }
            for path, data in sorted(entries.items())
        ],
    }


def _read_source_files(
    source_root: Path,
    policy: ReleasePolicy,
) -> dict[str, bytes]:
    extension_root = (source_root / "blender_extension").resolve()
    entries = {}
    for relative in policy.source_files:
        path = extension_root / PurePosixPath(relative)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(extension_root)
        except (OSError, ValueError) as exc:
            raise ReleasePackageError(
                f"allowlisted source file is missing or escaped: {relative}"
            ) from exc
        if path.is_symlink() or not resolved.is_file():
            raise ReleasePackageError(
                f"allowlisted source is not a regular file: {relative}"
            )
        data = resolved.read_bytes()
        if len(data) > policy.maximum_file_bytes:
            raise ReleasePackageError(
                f"allowlisted source exceeds size bound: {relative}"
            )
        entries[relative] = data
    return entries


def _validate_forbidden_paths(
    paths: set[str],
    policy: ReleasePolicy,
) -> None:
    forbidden_components = {
        item.casefold() for item in policy.forbidden_path_components
    }
    forbidden_suffixes = {
        item.casefold() for item in policy.forbidden_suffixes
    }
    for path in paths:
        pure = PurePosixPath(path)
        if any(
            part.casefold() in forbidden_components for part in pure.parts
        ):
            raise ReleasePackageError(
                f"forbidden machine or user path in release package: {path}"
            )
        if pure.suffix.casefold() in forbidden_suffixes:
            raise ReleasePackageError(
                f"forbidden file type in release package: {path}"
            )


def _write_deterministic_zip(
    destination: Path,
    entries: dict[str, bytes],
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits |= 0x800
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    os.replace(temporary, destination)


def assemble_release(
    source_root: str | Path,
    output_directory: str | Path,
    metadata: BuildMetadata,
    *,
    policy_path: str | Path,
    license_catalog_path: str | Path,
) -> ReleaseArtifacts:
    source = Path(source_root).resolve()
    output = Path(output_directory).resolve()
    policy_file = Path(policy_path).resolve()
    license_file = Path(license_catalog_path).resolve()
    policy = load_policy(policy_file)
    license_catalog = _load_license_catalog(license_file)
    entries = _read_source_files(source, policy)
    version, wheel_hashes, weight_licenses_resolved = (
        _version_from_entries(source, entries, metadata, policy)
    )
    schema_catalog_version = _validate_schema_catalog(entries)
    model_catalog = _strict_json_bytes(
        entries["runtime_bundle/model-catalog.json"],
        "Model Catalog",
    )
    inventory = _runtime_inventory(entries)
    license_records = _validate_dependency_licenses(
        inventory,
        license_catalog,
    )

    try:
        from scripts import validate_localization

        validate_localization.validate(source)
    except Exception as exc:
        if isinstance(exc, ReleasePackageError):
            raise
        raise ReleasePackageError(
            "source localization or offline manual release gate failed"
        ) from exc

    entries[PACKAGE_POLICY] = policy_file.read_bytes()
    entries[DEPENDENCY_LICENSES] = license_file.read_bytes()
    entries[GENERATED_NOTICE] = _dependency_notice(
        inventory,
        license_records,
    )
    entries[SBOM_PATH] = _json_bytes(
        _build_sbom(
            metadata=metadata,
            version=version,
            inventory=inventory,
            records=license_records,
            wheel_hashes=wheel_hashes,
            policy=policy,
        )
    )
    entries[PROVENANCE_PATH] = _json_bytes(
        _build_provenance(metadata, version, entries)
    )
    entries[MANIFEST_PATH] = _json_bytes(
        _package_manifest(
            metadata=metadata,
            version=version,
            entries=entries,
            schema_catalog_version=schema_catalog_version,
            model_catalog_version=model_catalog["catalog_version"],
            weight_licenses_resolved=weight_licenses_resolved,
            policy=policy,
        )
    )

    if set(entries) != set(policy.archive_files):
        raise ReleasePackageError(
            "assembled archive entries do not match the explicit allowlist"
        )
    _validate_forbidden_paths(set(entries), policy)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{policy.extension_id}-{version}.zip"
    _write_deterministic_zip(archive, entries)
    if archive.stat().st_size > policy.maximum_archive_bytes:
        raise ReleasePackageError("release archive exceeds size bound")

    archive_digest = _sha256(archive.read_bytes())
    sha_path = archive.with_suffix(".zip.sha256")
    sbom_path = archive.with_suffix(".sbom.spdx.json")
    provenance_path = archive.with_suffix(".provenance.json")
    sha_path.write_text(
        f"{archive_digest}  {archive.name}\n",
        encoding="ascii",
        newline="\n",
    )
    sbom_path.write_bytes(entries[SBOM_PATH])
    external_provenance = _strict_json_bytes(
        entries[PROVENANCE_PATH],
        "internal provenance",
    )
    external_provenance["subject"] = {
        "name": archive.name,
        "length": archive.stat().st_size,
        "sha256": archive_digest,
    }
    external_provenance["package_manifest_sha256"] = _sha256(
        entries[MANIFEST_PATH]
    )
    provenance_path.write_bytes(_json_bytes(external_provenance))
    artifacts = ReleaseArtifacts(
        archive=archive,
        sha256=sha_path,
        sbom=sbom_path,
        provenance=provenance_path,
    )
    validate_release(
        archive,
        policy_path=policy_file,
        license_catalog_path=license_file,
        expected_metadata=metadata,
    )
    validate_companions(artifacts)
    return artifacts


def _read_archive(
    archive_path: Path,
    policy: ReleasePolicy,
) -> dict[str, bytes]:
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size > policy.maximum_archive_bytes
    ):
        raise ReleasePackageError("release archive is missing or oversized")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            folded = [name.casefold() for name in names]
            if len(folded) != len(set(folded)):
                raise ReleasePackageError(
                    "release archive contains case-insensitive duplicate entries"
                )
            for info in infos:
                _safe_archive_name(info.filename)
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or stat.S_ISLNK(mode)
                    or info.file_size > policy.maximum_file_bytes
                ):
                    raise ReleasePackageError(
                        f"unsafe release archive entry: {info.filename}"
                    )
            if sum(info.file_size for info in infos) > (
                policy.maximum_archive_bytes
            ):
                raise ReleasePackageError(
                    "release archive expands beyond its size bound"
                )
            entries = {
                info.filename: archive.read(info)
                for info in infos
            }
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        if isinstance(exc, ReleasePackageError):
            raise
        raise ReleasePackageError("release archive is invalid") from exc
    expected = set(policy.archive_files)
    actual = set(entries)
    if actual != expected:
        raise ReleasePackageError(
            "release archive entries do not match the explicit allowlist "
            f"(missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r})"
        )
    _validate_forbidden_paths(actual, policy)
    return entries


def _validate_no_placeholders(entries: dict[str, bytes]) -> None:
    text_suffixes = {
        ".css",
        ".html",
        ".json",
        ".lock",
        ".py",
        ".toml",
        ".txt",
    }
    for path, data in entries.items():
        if PurePosixPath(path).suffix.casefold() not in text_suffixes:
            continue
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ReleasePackageError(
                f"text release entry is not strict UTF-8: {path}"
            ) from exc
        match = PLACEHOLDER.search(text)
        if match:
            raise ReleasePackageError(
                f"placeholder content in final release archive: "
                f"{path}: {match.group(0)}"
            )


def _manifest_metadata(document: dict) -> BuildMetadata:
    try:
        source = document["source"]
        return BuildMetadata(
            repository=source["repository"],
            commit=source["commit"],
            tag=source["tag"],
            workflow_ref=source["workflow_ref"],
            run_id=str(source["run_id"]),
            run_attempt=str(source["run_attempt"]),
            created_utc="1970-01-01T00:00:00Z",
        )
    except (KeyError, TypeError) as exc:
        raise ReleasePackageError(
            "package manifest source identity is invalid"
        ) from exc


def _validate_package_manifest(
    entries: dict[str, bytes],
    archive_name: str,
    expected_metadata: BuildMetadata | None,
    policy: ReleasePolicy,
) -> tuple[dict, BuildMetadata, str]:
    document = _strict_json_bytes(
        entries[MANIFEST_PATH],
        "package manifest",
    )
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "package",
        "source",
        "versions",
        "claims",
        "files",
    }:
        raise ReleasePackageError("package manifest fields are invalid")
    if document["schema_version"] != "1.0.0":
        raise ReleasePackageError("package manifest version is invalid")
    package = document["package"]
    if (
        not isinstance(package, dict)
        or package.get("id") != policy.extension_id
        or not isinstance(package.get("version"), str)
    ):
        raise ReleasePackageError("package manifest identity is invalid")
    metadata = _manifest_metadata(document)
    if expected_metadata is not None:
        for field in (
            "repository",
            "commit",
            "tag",
            "workflow_ref",
            "run_id",
            "run_attempt",
        ):
            if getattr(metadata, field) != getattr(expected_metadata, field):
                raise ReleasePackageError(
                    f"package manifest {field} does not match CI"
                )
        metadata = expected_metadata
    else:
        provenance = _strict_json_bytes(
            entries[PROVENANCE_PATH],
            "package provenance",
        )
        metadata = BuildMetadata(
            **{
                **metadata.__dict__,
                "created_utc": provenance.get("source", {}).get(
                    "created_utc",
                    "",
                ),
            }
        )
    version = _validate_metadata(metadata, policy)
    if package["version"] != version:
        raise ReleasePackageError(
            "package manifest tag and version do not agree"
        )
    files = document["files"]
    if not isinstance(files, list):
        raise ReleasePackageError("package manifest file list is invalid")
    records = {}
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "length",
            "sha256",
        }:
            raise ReleasePackageError(
                "package manifest file record is invalid"
            )
        path = record["path"]
        if (
            path in records
            or path == MANIFEST_PATH
            or path not in entries
            or record["length"] != len(entries[path])
            or not SHA256.fullmatch(record["sha256"])
        ):
            raise ReleasePackageError(
                f"package manifest checksum record is invalid: {path!r}"
            )
        if record["sha256"] != _sha256(entries[path]):
            raise ReleasePackageError(
                f"package manifest checksum mismatch: {path}"
            )
        records[path] = record
    expected_files = set(entries) - {MANIFEST_PATH}
    if set(records) != expected_files:
        raise ReleasePackageError(
            "package manifest checksum inventory is incomplete"
        )
    if package.get("archive") != archive_name:
        raise ReleasePackageError(
            "package manifest archive filename is invalid"
        )
    return document, metadata, version


def _validate_final_localization(entries: dict[str, bytes]) -> None:
    try:
        from scripts import validate_localization

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = root / "blender_extension"
            for path, data in entries.items():
                destination = extension / PurePosixPath(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            validate_localization.validate(root)
    except Exception as exc:
        raise ReleasePackageError(
            "final ZIP localization or offline manual gate failed"
        ) from exc


def _validate_sbom(
    entries: dict[str, bytes],
    inventory: dict,
    records: dict[tuple[str, str], dict],
    version: str,
    policy: ReleasePolicy,
    metadata: BuildMetadata,
    wheel_hashes: dict[str, str],
) -> None:
    sbom = _strict_json_bytes(entries[SBOM_PATH], "SPDX SBOM")
    expected = _build_sbom(
        metadata=metadata,
        version=version,
        inventory=inventory,
        records=records,
        wheel_hashes=wheel_hashes,
        policy=policy,
    )
    if sbom != expected:
        raise ReleasePackageError(
            "SPDX SBOM does not exactly describe the frozen Runtime inventory"
        )


def validate_release(
    archive: str | Path,
    *,
    policy_path: str | Path,
    license_catalog_path: str | Path,
    expected_metadata: BuildMetadata | None = None,
) -> dict[str, object]:
    archive_path = Path(archive).resolve()
    policy_file = Path(policy_path).resolve()
    license_file = Path(license_catalog_path).resolve()
    policy = load_policy(policy_file)
    license_catalog = _load_license_catalog(license_file)
    entries = _read_archive(archive_path, policy)
    _validate_no_placeholders(entries)
    if entries[PACKAGE_POLICY] != policy_file.read_bytes():
        raise ReleasePackageError(
            "archived package policy does not match tagged source"
        )
    if entries[DEPENDENCY_LICENSES] != license_file.read_bytes():
        raise ReleasePackageError(
            "archived dependency licenses do not match tagged source"
        )
    manifest, metadata, version = _validate_package_manifest(
        entries,
        archive_path.name,
        expected_metadata,
        policy,
    )
    if archive_path.name != f"{policy.extension_id}-{version}.zip":
        raise ReleasePackageError(
            "archive filename and package version do not agree"
        )

    manifest_toml = _toml(
        entries["blender_manifest.toml"],
        "Blender Extension manifest",
    )
    if (
        manifest_toml.get("id") != policy.extension_id
        or manifest_toml.get("version") != version
    ):
        raise ReleasePackageError(
            "final ZIP manifest identity or version mismatch"
        )
    wheel_hashes = _validate_lock_and_wheels(entries, version)
    schema_version = _validate_schema_catalog(entries)
    weights_resolved = _validate_model_licenses(
        entries,
        stable_release=int(version.split(".", 1)[0]) >= 1,
    )
    model_catalog = _strict_json_bytes(
        entries["runtime_bundle/model-catalog.json"],
        "Model Catalog",
    )
    inventory = _runtime_inventory(entries)
    records = _validate_dependency_licenses(inventory, license_catalog)
    expected_notice = _dependency_notice(inventory, records)
    if entries[GENERATED_NOTICE] != expected_notice:
        raise ReleasePackageError(
            "generated Runtime dependency notice is mismatched"
        )
    _validate_sbom(
        entries,
        inventory,
        records,
        version,
        policy,
        metadata,
        wheel_hashes,
    )
    provenance = _strict_json_bytes(
        entries[PROVENANCE_PATH],
        "package provenance",
    )
    builder = provenance.get("builder", {})
    if (
        provenance.get("schema_version") != "1.0.0"
        or builder.get("kind") != "github-actions"
        or builder.get("workflow_ref") != metadata.workflow_ref
        or str(builder.get("run_id")) != metadata.run_id
        or str(builder.get("run_attempt")) != metadata.run_attempt
        or provenance.get("source", {}).get("repository")
        != metadata.repository
        or provenance.get("source", {}).get("commit") != metadata.commit
        or provenance.get("source", {}).get("tag") != metadata.tag
        or provenance.get("source", {}).get("version") != version
        or provenance.get("source", {}).get("kind")
        != "git-archive-of-tagged-commit"
    ):
        raise ReleasePackageError("package provenance identity mismatch")
    materials = provenance.get("materials")
    if not isinstance(materials, list):
        raise ReleasePackageError("package provenance materials are invalid")
    material_records = {}
    for material in materials:
        if not isinstance(material, dict) or set(material) != {
            "path",
            "length",
            "sha256",
        }:
            raise ReleasePackageError(
                "package provenance material record is invalid"
            )
        path = material["path"]
        if (
            path in material_records
            or path not in entries
            or material["length"] != len(entries[path])
            or not SHA256.fullmatch(material["sha256"])
            or material["sha256"] != _sha256(entries[path])
        ):
            raise ReleasePackageError(
                f"package provenance material mismatch: {path!r}"
            )
        material_records[path] = material
    expected_materials = set(entries) - {PROVENANCE_PATH, MANIFEST_PATH}
    if set(material_records) != expected_materials:
        raise ReleasePackageError(
            "package provenance material inventory is incomplete"
        )
    if manifest["versions"] != {
        "extension": version,
        "worker": version,
        "model_core": version,
        "schema_catalog": schema_version,
        "model_catalog": model_catalog["catalog_version"],
    }:
        raise ReleasePackageError("package component versions mismatch")
    claims = manifest["claims"]
    stable = int(version.split(".", 1)[0]) >= 1
    if claims != {
        "engineering_0_x": not stable,
        "stable_1_0": stable,
        "official_extensions_platform_ready": False,
        "weight_specific_licenses_resolved": weights_resolved,
        "custom_updater": False,
        "engineering_qualification_scope": ENGINEERING_QUALIFICATION_SCOPE,
        "stable_release_gates": dict(policy.stable_release_gates),
    }:
        raise ReleasePackageError("package release claims are untruthful")
    _validate_final_localization(entries)
    return {
        "archive": archive_path.name,
        "version": version,
        "repository": metadata.repository,
        "commit": metadata.commit,
        "files": len(entries),
        "sha256": _sha256(archive_path.read_bytes()),
        "worker_wheel_sha256": wheel_hashes["lingbot-map-worker"],
    }


def validate_companions(artifacts: ReleaseArtifacts) -> None:
    archive_bytes = artifacts.archive.read_bytes()
    digest = _sha256(archive_bytes)
    expected_sha = f"{digest}  {artifacts.archive.name}\n"
    if artifacts.sha256.read_text("ascii") != expected_sha:
        raise ReleasePackageError("published SHA-256 companion is invalid")
    with zipfile.ZipFile(artifacts.archive) as archive:
        internal_sbom = archive.read(SBOM_PATH)
        internal_manifest = archive.read(MANIFEST_PATH)
        internal_provenance = _strict_json_bytes(
            archive.read(PROVENANCE_PATH),
            "internal provenance",
        )
    if artifacts.sbom.read_bytes() != internal_sbom:
        raise ReleasePackageError("published SPDX SBOM companion is invalid")
    provenance = _strict_json_file(artifacts.provenance)
    expected_provenance = dict(internal_provenance)
    expected_provenance["subject"] = {
        "name": artifacts.archive.name,
        "length": len(archive_bytes),
        "sha256": digest,
    }
    expected_provenance["package_manifest_sha256"] = _sha256(
        internal_manifest
    )
    if provenance != expected_provenance:
        raise ReleasePackageError("published provenance companion is invalid")


def require_ci_environment(environment: dict[str, str]) -> None:
    required = {
        "GITHUB_ACTIONS": "true",
        "CI": "true",
        "GITHUB_REF_TYPE": "tag",
    }
    for name, expected in required.items():
        if environment.get(name, "").casefold() != expected:
            raise ReleasePackageError(
                "Release Packages may be produced only by GitHub Actions "
                f"from a tag ({name} mismatch)"
            )
    for name in (
        "GITHUB_REPOSITORY",
        "GITHUB_SHA",
        "GITHUB_REF_NAME",
        "GITHUB_WORKFLOW_REF",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
    ):
        if not environment.get(name):
            raise ReleasePackageError(
                f"GitHub Actions release identity is missing: {name}"
            )


def _run_git(
    repository: Path,
    arguments: list[str],
    *,
    text: bool = True,
) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = str(exc.stderr or "").strip()
        raise ReleasePackageError(
            f"Git command failed: {' '.join(arguments)}: {detail}"
        ) from exc
    return result.stdout


def _extract_git_tar(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    try:
        with tarfile.open(tar_path, mode="r:") as archive:
            for member in archive:
                name = member.name.rstrip("/")
                if not name:
                    continue
                _safe_archive_name(name)
                target = destination / PurePosixPath(name)
                resolved_parent = target.parent.resolve()
                try:
                    resolved_parent.relative_to(destination.resolve())
                except ValueError as exc:
                    raise ReleasePackageError(
                        f"Git archive path escaped: {name}"
                    ) from exc
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.size > 128 * 1024 * 1024:
                    raise ReleasePackageError(
                        f"Git archive entry is unsafe: {name}"
                    )
                total += member.size
                if total > 1024 * 1024 * 1024:
                    raise ReleasePackageError(
                        "Git archive expands beyond its size bound"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise ReleasePackageError(
                        f"Git archive file could not be read: {name}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except (OSError, tarfile.TarError) as exc:
        if isinstance(exc, ReleasePackageError):
            raise
        raise ReleasePackageError("could not extract Git archive") from exc


def export_tagged_tree(
    repository: str | Path,
    commit: str,
    tag: str,
    destination: str | Path,
) -> str:
    root = Path(repository).resolve()
    target = Path(destination).resolve()
    if not COMMIT_ID.fullmatch(commit):
        raise ReleasePackageError("release commit is not a full Git object ID")
    if SEMVER_TAG.fullmatch(tag) is None:
        raise ReleasePackageError("release tag is not strict SemVer")
    tagged = str(
        _run_git(
            root,
            ["rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        )
    ).strip().lower()
    if tagged != commit.lower():
        raise ReleasePackageError(
            "release tag does not resolve to the requested commit"
        )
    if target.exists():
        raise ReleasePackageError("Git archive destination already exists")
    with tempfile.TemporaryDirectory() as temporary:
        tar_path = Path(temporary) / "source.tar"
        _run_git(
            root,
            [
                "archive",
                "--format=tar",
                f"--output={tar_path}",
                tagged,
            ],
        )
        _extract_git_tar(tar_path, target)
    return tagged


def _commit_timestamp(repository: Path, commit: str) -> str:
    raw = str(
        _run_git(
            repository,
            ["show", "-s", "--format=%cI", commit],
        )
    ).strip()
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ReleasePackageError(
            "Git commit timestamp is invalid"
        ) from exc
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _metadata_from_ci(repository_root: Path) -> BuildMetadata:
    environment = dict(os.environ)
    require_ci_environment(environment)
    commit = environment["GITHUB_SHA"].lower()
    return BuildMetadata(
        repository=environment["GITHUB_REPOSITORY"],
        commit=commit,
        tag=environment["GITHUB_REF_NAME"],
        workflow_ref=environment["GITHUB_WORKFLOW_REF"],
        run_id=environment["GITHUB_RUN_ID"],
        run_attempt=environment["GITHUB_RUN_ATTEMPT"],
        created_utc=_commit_timestamp(repository_root, commit),
    )


def _artifact_paths(archive: Path) -> ReleaseArtifacts:
    return ReleaseArtifacts(
        archive=archive,
        sha256=archive.with_suffix(".zip.sha256"),
        sbom=archive.with_suffix(".sbom.spdx.json"),
        provenance=archive.with_suffix(".provenance.json"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    build_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--archive", type=Path, required=True)
    validate_parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args(argv)
    repository_root = arguments.repository_root.resolve()

    if arguments.command == "build":
        metadata = _metadata_from_ci(repository_root)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tagged-source"
            export_tagged_tree(
                repository_root,
                metadata.commit,
                metadata.tag,
                source,
            )
            artifacts = assemble_release(
                source,
                arguments.output_dir,
                metadata,
                policy_path=source / "release/package-policy.json",
                license_catalog_path=(
                    source / "release/dependency-licenses.json"
                ),
            )
        result = {
            "archive": str(artifacts.archive),
            "sha256": str(artifacts.sha256),
            "sbom": str(artifacts.sbom),
            "provenance": str(artifacts.provenance),
        }
        print(
            "LINGBOT_MAP_RELEASE_BUILD="
            + json.dumps(result, sort_keys=True, separators=(",", ":"))
        )
        return 0

    expected = None
    if os.environ.get("GITHUB_ACTIONS", "").casefold() == "true":
        expected = _metadata_from_ci(repository_root)
    summary = validate_release(
        arguments.archive,
        policy_path=repository_root / "release/package-policy.json",
        license_catalog_path=(
            repository_root / "release/dependency-licenses.json"
        ),
        expected_metadata=expected,
    )
    validate_companions(_artifact_paths(arguments.archive.resolve()))
    print(
        "LINGBOT_MAP_RELEASE_VALIDATION="
        + json.dumps(summary, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
