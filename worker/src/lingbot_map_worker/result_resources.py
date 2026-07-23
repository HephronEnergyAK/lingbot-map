"""Project Disk and Worker Memory Gates for Result construction."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import math
import os
from pathlib import Path
import shutil
from typing import Protocol


GIB = 1024 ** 3
HEADROOM_NUMERATOR = 5
HEADROOM_DENOMINATOR = 4
DISK_RESERVE_BYTES = 2 * GIB
MEMORY_RESERVE_BYTES = 4 * GIB
LOG_ALLOWANCE_BYTES = 64 * 1024 ** 2
RESULT_FIXED_ALLOWANCE_BYTES = 1024 ** 2


class ResourceGateError(RuntimeError):
    pass


class ResourceProbe(Protocol):
    def available_disk_bytes(self, path: Path) -> int: ...
    def available_physical_memory_bytes(self) -> int: ...


class SystemResourceProbe:
    def available_disk_bytes(self, path: Path) -> int:
        return int(shutil.disk_usage(path).free)

    def available_physical_memory_bytes(self) -> int:
        if os.name != "nt":
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ResourceGateError(
                f"GlobalMemoryStatusEx failed: {ctypes.get_last_error()}"
            )
        return int(status.ullAvailPhys)


@dataclass
class FixedResourceProbe:
    """Deterministic Adapter used by contract tests and fixture Jobs."""

    disk_bytes: int
    physical_memory_bytes: int

    def available_disk_bytes(self, _path: Path) -> int:
        return int(self.disk_bytes)

    def available_physical_memory_bytes(self) -> int:
        return int(self.physical_memory_bytes)


def with_headroom(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceGateError("resource estimate must be a non-negative integer")
    return math.ceil(value * HEADROOM_NUMERATOR / HEADROOM_DENOMINATOR)


def require_project_disk(
    probe: ResourceProbe,
    path: Path,
    estimated_remaining_bytes: int,
) -> int:
    available = int(probe.available_disk_bytes(path))
    required = with_headroom(estimated_remaining_bytes) + DISK_RESERVE_BYTES
    if available < required:
        raise ResourceGateError(
            f"Project Disk Gate requires {required} available bytes; observed {available}"
        )
    return available


def require_worker_memory(probe: ResourceProbe, estimated_peak_bytes: int) -> int:
    available = int(probe.available_physical_memory_bytes())
    required = with_headroom(estimated_peak_bytes) + MEMORY_RESERVE_BYTES
    if available < required:
        raise ResourceGateError(
            f"Worker Memory Gate requires {required} physical bytes; observed {available}"
        )
    return available


def estimate_core_result_bytes(frame_count: int, point_budget: int, *, sky: bool = False) -> int:
    if frame_count < 1 or point_budget < 1:
        raise ResourceGateError("frame count and point budget must be positive")
    point_payload = point_budget * (12 + 3 + 4 + 4 + 4)
    camera_payload = frame_count * (64 + 36 + 36 + 8 + 8 + 1)
    sky_payload = frame_count * 4 if sky else 0
    npy_header_allowance = 13 * 65536
    return (
        point_payload + camera_payload + sky_payload + npy_header_allowance
        + LOG_ALLOWANCE_BYTES + RESULT_FIXED_ALLOWANCE_BYTES
    )


def estimate_fixture_memory_bytes(frame_count: int, grid_pixels: int, point_budget: int) -> int:
    """Pinned conservative coefficient set for the deterministic CPU fixture."""

    if frame_count < 1 or grid_pixels < 1 or point_budget < 1:
        raise ResourceGateError("fixture memory dimensions must be positive")
    fixed = 64 * 1024 ** 2
    per_frame_metadata = 4096 * frame_count
    per_grid_candidate = 64 * grid_pixels
    per_reducer_entry = 256 * (point_budget + 1)
    output_arrays = 27 * point_budget
    return fixed + per_frame_metadata + per_grid_candidate + per_reducer_entry + output_arrays


def estimate_dense_buffer_bytes(frame_count: int, grid_pixels: int) -> int:
    """Maximum paired depth/confidence chunk held before a 64-frame flush."""

    if frame_count < 1 or grid_pixels < 1:
        raise ResourceGateError("dense buffer dimensions must be positive")
    return min(frame_count, 64) * grid_pixels * 4 * 2


def estimate_window_alignment_memory_bytes(grid_pixels: int) -> int:
    """CPU buffers for one 64-frame window plus the prior 16-frame overlap."""

    if grid_pixels < 1:
        raise ResourceGateError("window alignment grid must be positive")
    canonical_inputs = 64 * grid_pixels * (3 * 4 + 3)
    decoded_predictions = (64 + 16) * grid_pixels * (4 + 4 + 3)
    alignment_scratch = 16 * grid_pixels * 8 * 5
    camera_and_metadata = (64 + 16) * 2048
    return (
        canonical_inputs
        + decoded_predictions
        + alignment_scratch
        + camera_and_metadata
    )
