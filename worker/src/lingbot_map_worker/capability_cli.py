"""Strict file/JSON interface for GPU discovery and capability-test children."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping

from .capability import (
    CapabilityCache,
    CapabilityCancelled,
    CapabilityGpuFailure,
    CapabilitySuite,
    ModelFingerprint,
    _atomic_json,
)
from .gpu_devices import NvmlDeviceProvider, select_physical_gpu
from .gpu_lease import AmbiguousGpuLeaseError, GpuBusyError
from .gpu_profiles import profiles_by_name
from .torch_capability import TorchCapabilityWorkload


MAX_REQUEST_BYTES = 64 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9-]{8,128}$")


class CapabilityRequestError(RuntimeError):
    pass


def _is_reparse_point(path: Path) -> bool:
    if os.name != "nt" or not path.exists():
        return path.is_symlink()
    attributes = path.stat(follow_symlinks=False).st_file_attributes
    return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _require_plain_chain(path: Path, boundary: Path) -> None:
    current = path
    boundary = Path(os.path.abspath(boundary))
    while True:
        if _is_reparse_point(current):
            raise CapabilityRequestError(f"Linked path is forbidden: {current}")
        if current == boundary:
            return
        if current.parent == current or not current.is_relative_to(boundary):
            raise CapabilityRequestError("Capability path escaped its managed boundary")
        current = current.parent


@dataclass(frozen=True)
class CapabilityRequest:
    test_id: str
    nonce: str
    runtime_id: str
    worker_lock_sha256: str
    managed_root: Path
    gpu_uuid: str
    catalog_version: str
    model_id: str
    model_sha256: str
    model_path: Path
    profile_names: tuple[str, ...]

    @property
    def test_root(self) -> Path:
        return self.managed_root / "capability-tests" / self.test_id

    @property
    def status_path(self) -> Path:
        return self.test_root / "status.json"

    @property
    def cancel_path(self) -> Path:
        return self.test_root / "cancel.requested"


def _duplicates_rejected(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_request(path: Path) -> CapabilityRequest:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise CapabilityRequestError("Capability request must be one absolute ordinary file")
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise CapabilityRequestError("Capability request exceeds 64 KiB")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates_rejected
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapabilityRequestError(f"Cannot parse strict capability request: {exc}") from exc
    expected = {
        "schema_version", "test_id", "nonce", "runtime_id", "worker_lock_sha256",
        "managed_root", "gpu_uuid", "model", "profiles",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise CapabilityRequestError("Capability request has unknown or missing fields")
    model = document["model"]
    if not isinstance(model, dict) or set(model) != {
        "catalog_version", "id", "sha256", "path"
    }:
        raise CapabilityRequestError("Capability request has an invalid model identity")
    profiles = document["profiles"]
    if not isinstance(profiles, list) or not all(isinstance(item, str) for item in profiles):
        raise CapabilityRequestError("Capability request profiles must be a string array")
    managed_input = Path(str(document["managed_root"]))
    model_input = Path(str(model["path"]))
    if not managed_input.is_absolute() or not model_input.is_absolute():
        raise CapabilityRequestError("Managed Runtime root and Model path must be absolute")
    managed_plain = Path(os.path.abspath(managed_input))
    model_plain = Path(os.path.abspath(model_input))
    _require_plain_chain(model_plain, managed_plain)
    request = CapabilityRequest(
        str(document["test_id"]),
        str(document["nonce"]),
        str(document["runtime_id"]),
        str(document["worker_lock_sha256"]),
        managed_plain.resolve(),
        str(document["gpu_uuid"]),
        str(model["catalog_version"]),
        str(model["id"]),
        str(model["sha256"]),
        model_plain.resolve(),
        tuple(profiles),
    )
    if document["schema_version"] != 1:
        raise CapabilityRequestError("Unsupported capability request schema version")
    if not TOKEN.fullmatch(request.test_id) or not TOKEN.fullmatch(request.nonce):
        raise CapabilityRequestError("Capability test ID or nonce is invalid")
    if not HEX64.fullmatch(request.runtime_id) or not HEX64.fullmatch(request.worker_lock_sha256):
        raise CapabilityRequestError("Runtime or Worker lock identity is invalid")
    if not HEX64.fullmatch(request.model_sha256):
        raise CapabilityRequestError("Reconstruction Model checksum is invalid")
    if (
        os.path.normcase(str(request.managed_root)) != os.path.normcase(str(managed_plain))
        or os.path.normcase(str(request.model_path)) != os.path.normcase(str(model_plain))
    ):
        raise CapabilityRequestError("Managed Runtime or Model path traversed a link")
    expected_model = request.managed_root / "models" / request.model_sha256 / "artifact"
    if request.model_path != expected_model or not request.model_path.is_file():
        raise CapabilityRequestError("Model path is not its checksum-addressed managed artifact")
    _require_plain_chain(request.model_path, request.managed_root)
    _require_plain_chain(request.test_root, request.managed_root)
    _require_plain_chain(request.managed_root / "capability-cache", request.managed_root)
    expected_request = request.test_root / "request.json"
    if Path(os.path.abspath(path)) != expected_request:
        raise CapabilityRequestError("Capability request is outside its exact test identity root")
    _require_plain_chain(expected_request, request.managed_root)
    profiles_by_name(request.profile_names)
    return request


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime(request: CapabilityRequest) -> None:
    executable = Path(sys.executable).resolve()
    expected = request.managed_root / "runtimes" / request.runtime_id
    if not executable.is_relative_to(expected):
        raise CapabilityRequestError("Capability child escaped its immutable Runtime")
    marker = expected / ".runtime-id"
    lock = expected / "uv.lock"
    if (
        not marker.is_file()
        or marker.read_text(encoding="ascii").strip() != request.runtime_id
        or not lock.is_file()
        or _sha256(lock) != request.worker_lock_sha256
    ):
        raise CapabilityRequestError("Immutable Runtime or Worker lock identity changed")


def discover_gpus() -> int:
    devices = NvmlDeviceProvider().discover()
    print(
        json.dumps(
            {"schema_version": 1, "devices": [device.to_json() for device in devices]},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def run_capability_test(request_path: Path) -> int:
    request = _read_request(request_path)
    _validate_runtime(request)
    request.test_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        request.status_path,
        {
            "schema_version": 1,
            "test_id": request.test_id,
            "state": "running",
            "gpu_uuid": request.gpu_uuid,
            "phase": "device-validation",
            "completed": 0,
            "total": len(request.profile_names),
            "results": [],
        },
    )
    provider = NvmlDeviceProvider()
    device = select_physical_gpu(provider.discover(), request.gpu_uuid)
    # This immutable physical UUID, never its prior ordinal, controls CUDA visibility.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device.uuid
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    profiles = profiles_by_name(request.profile_names)
    cache = CapabilityCache(request.managed_root / "capability-cache")
    workload = TorchCapabilityWorkload(
        model_path=request.model_path,
        model_sha256=request.model_sha256,
        runtime_id=request.runtime_id,
        worker_lock_sha256=request.worker_lock_sha256,
    )

    def cancelled() -> bool:
        return request.cancel_path.is_file()

    latest = {"phase": "starting", "completed": 0, "total": 0}

    def progress(phase: str, completed: int, total: int) -> None:
        latest.update(phase=phase, completed=completed, total=total)
        _atomic_json(
            request.status_path,
            {
                "schema_version": 1,
                "test_id": request.test_id,
                "state": "running",
                "gpu_uuid": request.gpu_uuid,
                "phase": phase,
                "completed": completed,
                "total": total,
                "results": [],
            },
        )
        print(
            json.dumps(
                {"event": "capability_progress", "phase": phase, "completed": completed, "total": total},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    suite = CapabilitySuite(
        device=device,
        model=ModelFingerprint(
            request.catalog_version, request.model_id, request.model_sha256
        ),
        workload=workload,
        cache=cache,
        runtime_id=request.runtime_id,
        action_id=request.test_id,
        nonce=request.nonce,
    )
    terminal: dict[str, object]
    exit_code = 0
    try:
        results = suite.run(profiles, cancelled=cancelled, progress=progress)
        terminal = {
            "state": "succeeded",
            "phase": "complete",
            "completed": len(results),
            "total": len(results),
            "results": [result.to_json() for result in results],
        }
    except CapabilityCancelled as exc:
        terminal = {"state": "cancelled", "error": str(exc), "results": []}
        exit_code = 11
    except (GpuBusyError, AmbiguousGpuLeaseError) as exc:
        terminal = {
            "state": "blocked",
            "classification": "gpu-busy",
            "error": str(exc),
            "owner": exc.owner,
            "results": [],
        }
        exit_code = 10
    except CapabilityGpuFailure as exc:
        terminal = {
            "state": "failed",
            "classification": exc.classification,
            "error": str(exc),
            "results": [],
        }
        exit_code = 12
    except Exception as exc:
        terminal = {
            "state": "failed",
            "classification": "capability-test-error",
            "error": str(exc),
            "results": [],
        }
        exit_code = 12
    document = {
        "schema_version": 1,
        "test_id": request.test_id,
        "gpu_uuid": request.gpu_uuid,
        "phase": terminal.pop("phase", latest["phase"]),
        "completed": terminal.pop("completed", latest["completed"]),
        "total": terminal.pop("total", latest["total"]),
        **terminal,
    }
    _atomic_json(request.status_path, document)
    print(
        json.dumps(
            {"event": "capability_terminal", **document},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return exit_code
