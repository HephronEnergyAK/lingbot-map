"""Bounded rolling-window Pipeline for Capture Sources above 3000 frames."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Iterable, Protocol, Sequence

from .canonical_preprocessing import CanonicalImage, canonicalize_srgb
from .gpu_profiles import InferencePlan, ProfileSelection, inference_plan, resolve_profile
from .result_pipeline import AlignedPrediction
from .short_pipeline import (
    CancelCheck,
    ExecutionGuard,
    HealthCheck,
    PipelineCancelled,
    ProgressReporter,
    ResultSink,
    ShortPipelineError,
    SourceFrame,
)
from .window_alignment import RollingWindowAligner, apply_similarity


class LongPipelineError(ShortPipelineError):
    pass


@dataclass(frozen=True)
class WindowFrame:
    frame_index: int
    pts_seconds: float
    canonical: CanonicalImage


class WindowPredictor(Protocol):
    def predict(
        self,
        frames: Sequence[WindowFrame],
        *,
        cancel: CancelCheck,
    ) -> Sequence[AlignedPrediction]: ...
    def close(self) -> None: ...


WindowPredictorFactory = Callable[
    [InferencePlan, ProfileSelection | Any, tuple[int, int, int]], WindowPredictor
]


class WindowedReconstructionPipeline:
    """Finalize each frame once while retaining only 64 + 16 frame state."""

    def __init__(
        self,
        predictor_factory: WindowPredictorFactory,
        *,
        aligner: RollingWindowAligner | None = None,
        preprocessor: Callable[[Any], CanonicalImage] = canonicalize_srgb,
    ) -> None:
        self.predictor_factory = predictor_factory
        self.aligner = aligner or RollingWindowAligner()
        self.preprocessor = preprocessor
        self.maximum_window_frames = 0
        self.maximum_overlap_frames = 0
        self.window_count = 0

    @staticmethod
    def _global_frame_type(frame_index: int) -> int:
        return 0 if frame_index < 8 else 1

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
        if plan.mode != "windowed":
            raise LongPipelineError("long-source Pipeline requires more than 3000 frames")
        if (
            plan.window_frames != 64
            or plan.overlap_keyframes != 16
            or plan.scale_frames != 8
            or plan.keyframe_interval != 1
        ):
            raise LongPipelineError("windowed inference plan is not the fixed v1 contract")
        resolved_profile = resolve_profile(profile)
        predictor: WindowPredictor | None = None
        current: list[WindowFrame] = []
        previous_overlap: list[tuple[AlignedPrediction, WindowFrame]] = []
        completed = 0
        decoded = 0
        last_pts = float("-inf")

        def accept(item: tuple[AlignedPrediction, WindowFrame]) -> None:
            nonlocal completed
            prediction, source = item
            if prediction.frame_index != completed or source.frame_index != completed:
                raise LongPipelineError("window finalization reordered or duplicated a frame")
            if cancel():
                raise PipelineCancelled(
                    f"cancelled before finalizing frame {prediction.frame_index}"
                )
            result_sink.accept(prediction, source.canonical, source.pts_seconds)
            completed += 1

        def process_window(*, final: bool) -> None:
            nonlocal predictor, current, previous_overlap
            if not current:
                raise LongPipelineError("windowed Pipeline received an empty window")
            if len(current) > plan.window_frames:
                raise LongPipelineError("windowed Pipeline exceeded 64 input frames")
            if previous_overlap and len(current) <= plan.overlap_keyframes:
                raise LongPipelineError("final window contains no new presentation frame")
            if cancel():
                raise PipelineCancelled("cancelled before window inference")
            execution_guard.boundary()
            health_check("window-inference", self.window_count)
            if predictor is None:
                progress.boundary("model-load", 0, 1)
                predictor = self.predictor_factory(
                    plan,
                    resolved_profile,
                    tuple(current[0].canonical.model_input.shape),
                )
                progress.advance(1, force=True)
                progress.boundary("windowed-inference", completed, frame_count)
            predicted = tuple(predictor.predict(tuple(current), cancel=cancel))
            if len(predicted) != len(current):
                raise LongPipelineError("window predictor did not return every frame")
            normalized: list[AlignedPrediction] = []
            for source, prediction in zip(current, predicted):
                if prediction.frame_index != source.frame_index:
                    raise LongPipelineError("window predictor changed frame identity")
                normalized.append(
                    replace(
                        prediction,
                        frame_type=self._global_frame_type(source.frame_index),
                        source_pts_seconds=source.pts_seconds,
                    )
                )
            self.window_count += 1
            if not previous_overlap:
                mapped = normalized
            else:
                overlap_count = plan.overlap_keyframes
                previous_predictions = [item[0] for item in previous_overlap]
                current_overlap = normalized[:overlap_count]
                transform = self.aligner.estimate(previous_predictions, current_overlap)
                mapped = [apply_similarity(item, transform) for item in normalized]
                metrics = self.aligner.metrics(
                    previous_predictions, mapped[:overlap_count], transform
                )
                recorder = getattr(result_sink, "record_window_boundary", None)
                if not callable(recorder):
                    raise LongPipelineError(
                        "windowed Result sink cannot record alignment provenance"
                    )
                recorder(metrics)
                for item in previous_overlap:
                    accept(item)
            overlap_count = 0 if not previous_overlap else plan.overlap_keyframes
            unique = list(zip(mapped[overlap_count:], current[overlap_count:]))
            if final:
                for item in unique:
                    accept(item)
                previous_overlap = []
                current = []
            else:
                if len(current) != plan.window_frames:
                    raise LongPipelineError("non-final window must use all 64 slots")
                finalize_count = len(unique) - plan.overlap_keyframes
                if finalize_count < 1:
                    raise LongPipelineError("window has no bounded finalization frontier")
                for item in unique[:finalize_count]:
                    accept(item)
                previous_overlap = unique[finalize_count:]
                if len(previous_overlap) != plan.overlap_keyframes:
                    raise LongPipelineError("window overlap did not retain exactly 16 frames")
                current = [item[1] for item in previous_overlap]
                self.maximum_overlap_frames = max(
                    self.maximum_overlap_frames, len(previous_overlap)
                )
            self.maximum_window_frames = max(self.maximum_window_frames, len(predicted))
            progress.advance(completed)

        try:
            with execution_guard:
                execution_guard.boundary()
                health_check("before-model", 0)
                if cancel():
                    raise PipelineCancelled("cancelled before long-source model loading")
                progress.boundary("source-decode", 0, frame_count)
                for expected_index, source in enumerate(frames):
                    if expected_index >= frame_count:
                        raise LongPipelineError("FrameSource produced more frames than preflight")
                    if source.frame_index != expected_index:
                        raise LongPipelineError("FrameSource indices are not contiguous")
                    if not math.isfinite(source.pts_seconds) or source.pts_seconds <= last_pts:
                        raise LongPipelineError("FrameSource PTS is not strictly increasing")
                    last_pts = source.pts_seconds
                    if cancel():
                        raise PipelineCancelled(f"cancelled before frame {expected_index}")
                    canonical = self.preprocessor(source.srgb)
                    current.append(WindowFrame(expected_index, source.pts_seconds, canonical))
                    decoded += 1
                    self.maximum_window_frames = max(
                        self.maximum_window_frames, len(current)
                    )
                    if predictor is None:
                        progress.advance(decoded)
                    if len(current) == plan.window_frames:
                        process_window(final=decoded == frame_count)
                if decoded != frame_count:
                    raise LongPipelineError(
                        f"FrameSource produced {decoded} of {frame_count} frames"
                    )
                if current:
                    process_window(final=True)
                if completed != frame_count or previous_overlap:
                    raise LongPipelineError(
                        f"windowed Pipeline finalized {completed} of {frame_count} frames"
                    )
                if predictor is None:
                    raise LongPipelineError("window predictor was never constructed")
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
            current.clear()
            previous_overlap.clear()
            if predictor is not None:
                predictor.close()
