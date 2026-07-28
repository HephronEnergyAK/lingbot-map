"""Process-scoped Windows sleep inhibition and resume health boundary."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import time
from typing import Any, Callable, Protocol, Sequence


class SleepGuardError(RuntimeError):
    pass


class ExecutionStateAdapter(Protocol):
    def acquire(self) -> None: ...
    def release(self) -> None: ...
    def awake_seconds(self) -> float: ...


class WindowsExecutionState:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        if os.name != "nt":
            raise SleepGuardError("native sleep inhibition requires Windows")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        self.kernel32.SetThreadExecutionState.restype = wintypes.DWORD
        self.kernel32.QueryUnbiasedInterruptTime.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong)
        ]
        self.kernel32.QueryUnbiasedInterruptTime.restype = wintypes.BOOL

    def acquire(self) -> None:
        if not self.kernel32.SetThreadExecutionState(
            self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
        ):
            raise SleepGuardError(
                f"SetThreadExecutionState acquire failed: {ctypes.get_last_error()}"
            )

    def release(self) -> None:
        if not self.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS):
            raise SleepGuardError(
                f"SetThreadExecutionState release failed: {ctypes.get_last_error()}"
            )

    def awake_seconds(self) -> float:
        value = ctypes.c_ulonglong()
        if not self.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
            raise SleepGuardError(
                f"QueryUnbiasedInterruptTime failed: {ctypes.get_last_error()}"
            )
        return value.value / 10_000_000.0


class WindowsSleepGuard:
    def __init__(
        self,
        adapter: ExecutionStateAdapter,
        *,
        resume_checks: Sequence[Callable[[], None]],
        on_resume: Callable[[], None] = lambda: None,
        wall_time: Callable[[], float] = time.time,
        suspension_threshold_seconds: float = 2.0,
        maximum_suspensions: int = 64,
    ) -> None:
        self.adapter = adapter
        self.resume_checks = tuple(resume_checks)
        self.on_resume = on_resume
        self.wall_time = wall_time
        self.threshold = suspension_threshold_seconds
        self.maximum_suspensions = maximum_suspensions
        self.suspension_count = 0
        self.suspension_seconds = 0.0
        self._last_wall = 0.0
        self._last_awake = 0.0
        self._entered = False

    def __enter__(self) -> "WindowsSleepGuard":
        self.adapter.acquire()
        self._last_wall = self.wall_time()
        self._last_awake = self.adapter.awake_seconds()
        self._entered = True
        return self

    def boundary(self) -> None:
        if not self._entered:
            raise SleepGuardError("sleep guard is not active")
        self.adapter.acquire()
        wall = self.wall_time()
        awake = self.adapter.awake_seconds()
        suspended = (wall - self._last_wall) - (awake - self._last_awake)
        self._last_wall, self._last_awake = wall, awake
        if suspended >= self.threshold:
            if self.suspension_count >= self.maximum_suspensions:
                raise SleepGuardError("suspension history exceeded its bounded persistence limit")
            for check in self.resume_checks:
                check()
            self.suspension_count += 1
            self.suspension_seconds += suspended
            self.on_resume()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._entered = False
        self.adapter.release()
