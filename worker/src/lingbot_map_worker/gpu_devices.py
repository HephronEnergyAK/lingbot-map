"""Physical NVIDIA device discovery without mutable CUDA ordinals as identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Protocol


GPU_UUID_PATTERN = re.compile(r"^(?:GPU|MIG)-[A-Za-z0-9-]{8,90}$")


class DeviceDiscoveryError(RuntimeError):
    pass


class DeviceSelectionError(DeviceDiscoveryError):
    pass


@dataclass(frozen=True)
class PhysicalGpu:
    uuid: str
    name: str
    total_memory: int
    driver_version: str
    compute_capability: tuple[int, int]

    def to_json(self) -> dict[str, object]:
        document = asdict(self)
        document["compute_capability"] = list(self.compute_capability)
        return document


class DeviceProvider(Protocol):
    def discover(self) -> tuple[PhysicalGpu, ...]: ...

    def memory_info(self, uuid: str) -> tuple[int, int]: ...


class NvmlDeviceProvider:
    """Official NVML binding Adapter; importing it does not initialize CUDA."""

    def __init__(self, pynvml_module=None):
        if pynvml_module is None:
            import pynvml as pynvml_module

        self._nvml = pynvml_module

    def discover(self) -> tuple[PhysicalGpu, ...]:
        nvml = self._nvml
        try:
            nvml.nvmlInit()
            driver = _nvml_text(nvml.nvmlSystemGetDriverVersion())
            devices = []
            for index in range(int(nvml.nvmlDeviceGetCount())):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                uuid = _nvml_text(nvml.nvmlDeviceGetUUID(handle))
                name = _nvml_text(nvml.nvmlDeviceGetName(handle))
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                capability = tuple(
                    int(value) for value in nvml.nvmlDeviceGetCudaComputeCapability(handle)
                )
                devices.append(
                    PhysicalGpu(uuid, name, int(memory.total), driver, capability)
                )
            return normalize_devices(devices)
        except Exception as exc:
            raise DeviceDiscoveryError(f"NVIDIA NVML device discovery failed: {exc}") from exc
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass

    def memory_info(self, uuid: str) -> tuple[int, int]:
        require_gpu_uuid(uuid)
        nvml = self._nvml
        try:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            return int(memory.free), int(memory.total)
        except Exception as exc:
            raise DeviceDiscoveryError(f"Cannot read current VRAM for {uuid}: {exc}") from exc
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def _nvml_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def require_gpu_uuid(value: str) -> str:
    if not GPU_UUID_PATTERN.fullmatch(value):
        raise DeviceSelectionError(f"invalid physical GPU UUID: {value!r}")
    return value


def normalize_devices(devices: Iterable[PhysicalGpu]) -> tuple[PhysicalGpu, ...]:
    result = tuple(devices)
    if len({device.uuid for device in result}) != len(result):
        raise DeviceDiscoveryError("NVML returned duplicate physical GPU UUIDs")
    for device in result:
        require_gpu_uuid(device.uuid)
        if device.total_memory <= 0 or not device.name or not device.driver_version:
            raise DeviceDiscoveryError(f"incomplete physical GPU identity: {device.uuid}")
        if len(device.compute_capability) != 2 or min(device.compute_capability) < 0:
            raise DeviceDiscoveryError(f"invalid CUDA compute capability: {device.uuid}")
    return tuple(sorted(result, key=lambda item: item.uuid))


def select_physical_gpu(
    devices: Iterable[PhysicalGpu], persisted_uuid: str | None
) -> PhysicalGpu:
    available = normalize_devices(devices)
    if persisted_uuid:
        require_gpu_uuid(persisted_uuid)
        matches = [device for device in available if device.uuid == persisted_uuid]
        if len(matches) != 1:
            raise DeviceSelectionError(
                f"Selected GPU {persisted_uuid} is unavailable; select an available physical UUID"
            )
        return matches[0]
    if len(available) == 1:
        return available[0]
    if not available:
        raise DeviceSelectionError("No compatible NVIDIA GPU was discovered")
    raise DeviceSelectionError(
        "Multiple compatible NVIDIA GPUs were discovered; select one persistent physical UUID"
    )
