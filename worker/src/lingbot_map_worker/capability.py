"""Profile qualification, exact cache identity, launch gate, and GPU fail-closed rules."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol
import uuid

from .gpu_devices import DeviceProvider, PhysicalGpu
from .gpu_lease import GpuLease
from .gpu_profiles import ReconstructionProfile


CAPABILITY_CACHE_VERSION = 1


class CapabilityError(RuntimeError):
    pass


class CapabilityCancelled(CapabilityError):
    pass


class CapabilityGpuFailure(CapabilityError):
    def __init__(self, classification: str, message: str):
        self.classification = classification
        super().__init__(message)


class CapabilityUnavailable(CapabilityError):
    pass


class InsufficientFreeVram(CapabilityError):
    pass


@dataclass(frozen=True)
class CapabilityStack:
    runtime_id: str
    worker_lock_sha256: str
    worker_version: str
    torch_version: str
    cuda_version: str
    attention_backend: str = "sdpa"
    execution_mode: str = "eager"


@dataclass(frozen=True)
class ModelFingerprint:
    catalog_version: str
    model_id: str
    model_sha256: str


@dataclass(frozen=True)
class CapabilityIdentity:
    gpu_uuid: str
    gpu_name: str
    gpu_total_memory: int
    driver_version: str
    compute_capability: tuple[int, int]
    stack: CapabilityStack
    model: ModelFingerprint
    profile_name: str
    profile_settings_sha256: str
    cache_version: int = CAPABILITY_CACHE_VERSION

    def to_json(self) -> dict[str, object]:
        document = asdict(self)
        document["compute_capability"] = list(self.compute_capability)
        return document

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_json())

    @property
    def base_sha256(self) -> str:
        document = self.to_json()
        document.pop("profile_name")
        document.pop("profile_settings_sha256")
        return _canonical_sha256(document)


@dataclass(frozen=True)
class CapabilityResult:
    identity: CapabilityIdentity
    state: str
    measured_peak_bytes: int
    required_free_bytes: int
    tested_total_bytes: int
    test_id: str
    tested_at: str

    def to_json(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_json(),
            "identity_sha256": self.identity.sha256,
            "state": self.state,
            "measured_peak_bytes": self.measured_peak_bytes,
            "required_free_bytes": self.required_free_bytes,
            "tested_total_bytes": self.tested_total_bytes,
            "test_id": self.test_id,
            "tested_at": self.tested_at,
        }


@dataclass(frozen=True)
class Measurement:
    peak_reserved_bytes: int


class CapabilityWorkload(Protocol):
    def stack(self) -> CapabilityStack: ...

    def measure(
        self,
        profile: ReconstructionProfile,
        *,
        cancelled: Callable[[], bool],
        progress: Callable[[str, int, int], None],
    ) -> Measurement: ...

    def classify_gpu_failure(self, exception: BaseException) -> str | None: ...


def _canonical_sha256(document: Mapping[str, object]) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _duplicates_rejected(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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


def _identity_from_json(raw: object) -> CapabilityIdentity:
    if not isinstance(raw, dict):
        raise ValueError("capability identity must be an object")
    expected = {
        "gpu_uuid", "gpu_name", "gpu_total_memory", "driver_version",
        "compute_capability", "stack", "model", "profile_name",
        "profile_settings_sha256", "cache_version",
    }
    if set(raw) != expected:
        raise ValueError("capability identity has unknown or missing fields")
    stack = raw["stack"]
    model = raw["model"]
    capability = raw["compute_capability"]
    if not isinstance(stack, dict) or not isinstance(model, dict):
        raise ValueError("capability stack and model fingerprints must be objects")
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("compute capability must contain two integers")
    return CapabilityIdentity(
        str(raw["gpu_uuid"]),
        str(raw["gpu_name"]),
        int(raw["gpu_total_memory"]),
        str(raw["driver_version"]),
        (int(capability[0]), int(capability[1])),
        CapabilityStack(**stack),
        ModelFingerprint(**model),
        str(raw["profile_name"]),
        str(raw["profile_settings_sha256"]),
        int(raw["cache_version"]),
    )


class CapabilityCache:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.invalidated_root = self.root / "invalidated"

    def path(self, identity: CapabilityIdentity) -> Path:
        return self.root / f"{identity.sha256}.json"

    def write(self, result: CapabilityResult) -> Path:
        if result.state not in {"qualified", "unqualified"}:
            raise CapabilityError("Cancelled or failed capability state cannot enter the cache")
        if result.measured_peak_bytes <= 0:
            raise CapabilityError("A capability result requires a positive measured peak")
        path = self.path(result.identity)
        _atomic_json(path, result.to_json())
        if self.read(result.identity) != result:
            raise CapabilityError("Capability cache publication could not be verified")
        return path

    def read(self, identity: CapabilityIdentity) -> CapabilityResult | None:
        path = self.path(identity)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            document = json.loads(raw, object_pairs_hook=_duplicates_rejected)
            expected = {
                "identity", "identity_sha256", "state", "measured_peak_bytes",
                "required_free_bytes", "tested_total_bytes", "test_id", "tested_at",
            }
            if not isinstance(document, dict) or set(document) != expected:
                raise ValueError("unknown or missing fields")
            stored_identity = _identity_from_json(document["identity"])
            result = CapabilityResult(
                stored_identity,
                str(document["state"]),
                int(document["measured_peak_bytes"]),
                int(document["required_free_bytes"]),
                int(document["tested_total_bytes"]),
                str(document["test_id"]),
                str(document["tested_at"]),
            )
            if (
                document["identity_sha256"] != stored_identity.sha256
                or stored_identity != identity
                or result.state not in {"qualified", "unqualified"}
                or result.measured_peak_bytes <= 0
                or result.required_free_bytes != math.ceil(result.measured_peak_bytes * 1.2)
                or result.tested_total_bytes != identity.gpu_total_memory
                or (result.state == "qualified")
                != (result.tested_total_bytes >= result.required_free_bytes)
            ):
                raise ValueError("capability result identity or measurement mismatch")
            return result
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CapabilityError(f"Invalid capability cache entry {path.name}: {exc}") from exc

    def invalidate_matching(self, identity: CapabilityIdentity, reason: str) -> tuple[Path, ...]:
        if not self.root.is_dir():
            return ()
        moved = []
        for path in sorted(self.root.glob("*.json")):
            try:
                document = json.loads(
                    path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates_rejected
                )
                stored = _identity_from_json(document["identity"])
            except Exception:
                continue
            if stored.base_sha256 != identity.base_sha256:
                continue
            destination = (
                self.invalidated_root
                / f"{path.stem}-{uuid.uuid4().hex[:12]}"
            )
            destination.mkdir(parents=True)
            os.replace(path, destination / path.name)
            _atomic_json(
                destination / "invalidation.json",
                {"reason": reason, "invalidated_at": datetime.now(timezone.utc).isoformat()},
            )
            moved.append(destination)
        return tuple(moved)


def capability_identity(
    device: PhysicalGpu,
    stack: CapabilityStack,
    model: ModelFingerprint,
    profile: ReconstructionProfile,
) -> CapabilityIdentity:
    return CapabilityIdentity(
        device.uuid,
        device.name,
        device.total_memory,
        device.driver_version,
        device.compute_capability,
        stack,
        model,
        profile.name,
        profile.settings_sha256,
    )


class CapabilitySuite:
    """Run fixed workloads once under the child's exact physical GPU lease."""

    def __init__(
        self,
        *,
        device: PhysicalGpu,
        model: ModelFingerprint,
        workload: CapabilityWorkload,
        cache: CapabilityCache,
        runtime_id: str,
        action_id: str,
        nonce: str,
        lease_factory=GpuLease,
    ) -> None:
        self.device = device
        self.model = model
        self.workload = workload
        self.cache = cache
        self.runtime_id = runtime_id
        self.action_id = action_id
        self.nonce = nonce
        self.lease_factory = lease_factory

    def run(
        self,
        profiles: Iterable[ReconstructionProfile],
        *,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[str, int, int], None] = lambda _phase, _done, _total: None,
    ) -> tuple[CapabilityResult, ...]:
        selected = tuple(profiles)
        if not selected:
            raise CapabilityError("No Reconstruction Profiles were selected")
        if cancelled():
            raise CapabilityCancelled("GPU capability test was cancelled before lease acquisition")
        with self.lease_factory(
            self.device.uuid,
            nonce=self.nonce,
            runtime_id=self.runtime_id,
            action_kind="capability-test",
            action_id=self.action_id,
        ):
            stack = self.workload.stack()
            if stack.runtime_id != self.runtime_id:
                raise CapabilityError("Capability Worker stack does not match its Runtime identity")
            measurements: dict[tuple[object, ...], Measurement] = {}
            results = []
            for index, profile in enumerate(selected):
                if cancelled():
                    raise CapabilityCancelled("GPU capability test was cancelled")
                identity = capability_identity(self.device, stack, self.model, profile)
                try:
                    measurement = measurements.get(profile.workload_key)
                    if measurement is None:
                        measurement = self.workload.measure(
                            profile, cancelled=cancelled, progress=progress
                        )
                        measurements[profile.workload_key] = measurement
                except CapabilityCancelled:
                    raise
                except Exception as exc:
                    classification = self.workload.classify_gpu_failure(exc)
                    if classification:
                        self.cache.invalidate_matching(identity, classification)
                        raise CapabilityGpuFailure(classification, str(exc)) from exc
                    raise
                peak = int(measurement.peak_reserved_bytes)
                if peak <= 0:
                    raise CapabilityError(f"Profile {profile.name} produced no VRAM measurement")
                required = math.ceil(peak * 1.2)
                state = "qualified" if self.device.total_memory >= required else "unqualified"
                result = CapabilityResult(
                    identity,
                    state,
                    peak,
                    required,
                    self.device.total_memory,
                    self.action_id,
                    datetime.now(timezone.utc).isoformat(),
                )
                results.append(result)
                progress("profiles", index + 1, len(selected))
            if cancelled():
                raise CapabilityCancelled(
                    "GPU capability test was cancelled before cache publication"
                )
            # Publish only after the complete selected suite succeeds. A cancel or
            # GPU failure therefore cannot leave a partial capability result.
            for result in results:
                self.cache.write(result)
            return tuple(results)


class CapabilityLaunchGate:
    """A Job must pass this under its own child-held lease; it never queues."""

    def __init__(
        self,
        cache: CapabilityCache,
        devices: DeviceProvider,
        *,
        lease_factory=GpuLease,
    ) -> None:
        self.cache = cache
        self.devices = devices
        self.lease_factory = lease_factory

    @contextmanager
    def acquire(
        self,
        identity: CapabilityIdentity,
        *,
        nonce: str,
        job_id: str,
    ):
        result = self.cache.read(identity)
        if result is None or result.state != "qualified":
            raise CapabilityUnavailable(
                f"Profile {identity.profile_name} has no matching qualified capability result"
            )
        with self.lease_factory(
            identity.gpu_uuid,
            nonce=nonce,
            runtime_id=identity.stack.runtime_id,
            action_kind="reconstruction-job",
            action_id=job_id,
        ):
            free, total = self.devices.memory_info(identity.gpu_uuid)
            if total != identity.gpu_total_memory:
                raise CapabilityUnavailable("Physical GPU total VRAM changed; requalification is required")
            if free < result.required_free_bytes:
                raise InsufficientFreeVram(
                    f"GPU Busy: {free} free bytes is below the required "
                    f"{result.required_free_bytes} bytes for {identity.profile_name}; "
                    "close competing GPU applications and explicitly retry"
                )
            yield result
