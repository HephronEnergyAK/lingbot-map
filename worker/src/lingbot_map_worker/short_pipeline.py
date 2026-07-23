"""Bounded Pipeline for complete Capture Sources of at most 3000 frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from lingbot_map.model_adapter import (
    CanonicalFrame,
    FramePrediction,
    ReconstructionModelAdapter,
)

from .canonical_preprocessing import CanonicalImage, canonicalize_srgb
from .gpu_profiles import InferencePlan, ProfileSelection, inference_plan, resolve_profile


class ShortPipelineError(RuntimeError):
    pass


class PipelineCancelled(ShortPipelineError):
    pass


@dataclass(frozen=True)
class SourceFrame:
    frame_index: int
    pts_seconds: float
    srgb: np.ndarray


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    phase: str
    completed: int
    total: int | None
    eta_seconds: float | None
    immediate: bool


class ResultSink(Protocol):
    def accept(
        self,
        prediction: FramePrediction,
        canonical: CanonicalImage,
        pts_seconds: float,
    ) -> None: ...
    def finish(self) -> Any: ...


class ExecutionGuard(Protocol):
    suspension_count: int
    suspension_seconds: float
    def __enter__(self) -> "ExecutionGuard": ...
    def boundary(self) -> None: ...
    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


AdapterFactory = Callable[
    [InferencePlan, ProfileSelection | Any, tuple[int, int, int]],
    ReconstructionModelAdapter,
]
ProgressCallback = Callable[[ProgressEvent], None]
HealthCheck = Callable[[str, int], None]
CancelCheck = Callable[[], bool]


class ProgressReporter:
    """At-most-4Hz ordinary events with stable rolling phase-local ETA."""

    def __init__(
        self,
        emit: ProgressCallback,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        minimum_eta_samples: int = 8,
    ) -> None:
        self.emit = emit
        self.monotonic = monotonic
        self.minimum_eta_samples = minimum_eta_samples
        self._phase = ""
        self._total: int | None = None
        self._completed = 0
        self._last_emit = -math.inf
        self._last_sample_time = 0.0
        self._last_sample_completed = 0
        self._rates: deque[float] = deque(maxlen=32)

    def boundary(self, phase: str, completed: int, total: int | None) -> None:
        now = self.monotonic()
        self._phase, self._total, self._completed = phase, total, completed
        self._last_sample_time = now
        self._last_sample_completed = completed
        self._last_emit = now
        self._rates.clear()
        self.emit(ProgressEvent("phase", phase, completed, total, None, True))

    def advance(self, completed: int, *, force: bool = False) -> None:
        now = self.monotonic()
        elapsed = now - self._last_sample_time
        units = completed - self._last_sample_completed
        if elapsed > 0 and units > 0:
            self._rates.append(units / elapsed)
            self._last_sample_time = now
            self._last_sample_completed = completed
        self._completed = completed
        if not force and now - self._last_emit < 0.25:
            return
        eta = self._eta()
        self.emit(
            ProgressEvent("progress", self._phase, completed, self._total, eta, force)
        )
        self._last_emit = now

    def resume(self) -> None:
        self._rates.clear()
        self._last_sample_time = self.monotonic()
        self._last_sample_completed = self._completed
        self.emit(
            ProgressEvent(
                "resume", self._phase, self._completed, self._total, None, True
            )
        )

    def _eta(self) -> float | None:
        if self._total is None or len(self._rates) < self.minimum_eta_samples:
            return None
        mean = statistics.fmean(self._rates)
        if mean <= 0:
            return None
        variation = statistics.pstdev(self._rates) / mean
        if variation > 0.2:
            return None
        return max(0.0, (self._total - self._completed) / mean)


class NullExecutionGuard:
    suspension_count = 0
    suspension_seconds = 0.0

    def __enter__(self) -> "NullExecutionGuard":
        return self

    def boundary(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


class ShortReconstructionPipeline:
    """Coordinate one bounded Adapter run without retaining prediction history."""

    def __init__(
        self,
        adapter_factory: AdapterFactory,
        *,
        preprocessor: Callable[[np.ndarray], CanonicalImage] = canonicalize_srgb,
    ) -> None:
        self.adapter_factory = adapter_factory
        self.preprocessor = preprocessor
        self.maximum_pending_colors = 0

    def run(
        self,
        *,
        frame_count: int,
        frames: Iterable[SourceFrame],
        profile: ProfileSelection,
        result_sink: ResultSink,
        progress: ProgressReporter,
        execution_guard: ExecutionGuard,
        health_check: HealthCheck,
        cancel: CancelCheck,
    ) -> Any:
        plan = inference_plan(frame_count)
        if plan.mode != "streaming":
            raise ShortPipelineError("short-source Pipeline accepts at most 3000 frames")
        resolved_profile = resolve_profile(profile)
        adapter: ReconstructionModelAdapter | None = None
        pending: dict[int, tuple[CanonicalImage, float]] = {}
        completed = 0
        last_pts = -math.inf
        with execution_guard:
            execution_guard.boundary()
            health_check("before-model", 0)
            if cancel():
                raise PipelineCancelled("cancelled before model loading")
            progress.boundary("source-decode", 0, frame_count)
            try:
                for expected_index, source in enumerate(frames):
                    if expected_index >= frame_count:
                        raise ShortPipelineError("FrameSource produced more frames than preflight")
                    if source.frame_index != expected_index:
                        raise ShortPipelineError("FrameSource indices are not contiguous")
                    if not math.isfinite(source.pts_seconds):
                        raise ShortPipelineError("FrameSource PTS is non-finite")
                    if source.pts_seconds <= last_pts:
                        raise ShortPipelineError("FrameSource PTS is not strictly increasing")
                    last_pts = source.pts_seconds
                    if cancel():
                        raise PipelineCancelled(f"cancelled before frame {expected_index}")
                    execution_guard.boundary()
                    health_check("inference", expected_index)
                    canonical = self.preprocessor(source.srgb)
                    if adapter is None:
                        progress.boundary("model-load", 0, 1)
                        adapter = self.adapter_factory(
                            plan,
                            resolved_profile,
                            tuple(canonical.model_input.shape),
                        )
                        progress.advance(1, force=True)
                        progress.boundary("inference", 0, frame_count)
                    pending[expected_index] = (canonical, source.pts_seconds)
                    self.maximum_pending_colors = max(
                        self.maximum_pending_colors, len(pending)
                    )
                    predictions = adapter.submit(
                        CanonicalFrame(expected_index, canonical.model_input)
                    )
                    for prediction in predictions:
                        if cancel():
                            raise PipelineCancelled(
                                f"cancelled before finalizing frame {prediction.frame_index}"
                            )
                        try:
                            retained, pts = pending.pop(prediction.frame_index)
                        except KeyError as exc:
                            raise ShortPipelineError(
                                "Adapter emitted a frame outside the bounded pending set"
                            ) from exc
                        result_sink.accept(prediction, retained, pts)
                        completed += 1
                    progress.advance(completed)
                if adapter is None or completed != frame_count or pending:
                    raise ShortPipelineError(
                        f"FrameSource/Adapter finalized {completed} of {frame_count} frames"
                    )
                adapter.finish()
                adapter = None
                progress.advance(frame_count, force=True)
                if cancel():
                    raise PipelineCancelled("cancelled before Result finalization")
                execution_guard.boundary()
                health_check("finalization", frame_count)
                progress.boundary("finalization", 0, None)
                result = result_sink.finish()
                progress.advance(1, force=True)
                return result
            finally:
                pending.clear()
                if adapter is not None:
                    adapter.close()
