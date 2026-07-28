"""Installed-Runtime integration for the bounded 3001-frame window contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lingbot_map_worker.canonical_preprocessing import CanonicalImage
from lingbot_map_worker.gpu_profiles import ProfileSelection
from lingbot_map_worker.long_pipeline import WindowedReconstructionPipeline
from lingbot_map_worker.result_pipeline import AlignedPrediction
from lingbot_map_worker.short_pipeline import (
    NullExecutionGuard,
    ProgressReporter,
    SourceFrame,
)
import lingbot_map_worker.long_pipeline as installed_module


def canonical(_rgb):
    return CanonicalImage(
        np.zeros((3, 2, 2), dtype="<f4"),
        np.zeros((2, 2, 3), dtype="|u1"),
        np.eye(3, dtype="<f8"),
        ((0, 0), (2, 0), (2, 2), (0, 2)),
    )


class Predictor:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def predict(self, frames, *, cancel):
        self.calls += 1
        raw_scale = 1.0 if self.calls == 1 else 0.5
        output = []
        for frame in frames:
            assert not cancel()
            camera_to_world = np.eye(4, dtype="<f8")
            camera_to_world[0, 3] = frame.frame_index * 0.01 * raw_scale
            output.append(
                AlignedPrediction(
                    frame.frame_index,
                    1,
                    frame.pts_seconds,
                    np.ascontiguousarray(np.linalg.inv(camera_to_world), dtype="<f8"),
                    np.ascontiguousarray(
                        ((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)),
                        dtype="<f8",
                    ),
                    np.full((2, 2), raw_scale, dtype="<f4"),
                    np.ones((2, 2), dtype="<f4"),
                    frame.canonical.color_rgb,
                )
            )
        return tuple(output)

    def close(self):
        self.closed = True


class Sink:
    def __init__(self):
        self.frames = []
        self.boundaries = []

    def accept(self, prediction, _canonical, _pts):
        self.frames.append(prediction.frame_index)

    def record_window_boundary(self, metrics):
        self.boundaries.append(metrics)

    def finish(self):
        return "ready"


def main() -> None:
    count = 3001
    rgb = np.zeros((1, 1, 3), dtype="|u1")
    source = (SourceFrame(index, index / 25.0, rgb) for index in range(count))
    predictor = Predictor()
    sink = Sink()
    pipeline = WindowedReconstructionPipeline(
        lambda _plan, _profile, _shape: predictor,
        preprocessor=canonical,
    )
    result = pipeline.run(
        frame_count=count,
        frames=source,
        profile=ProfileSelection("Draft", 1, 70, 99.5, 1_000_000),
        result_sink=sink,
        progress=ProgressReporter(lambda _event: None),
        execution_guard=NullExecutionGuard(),
        health_check=lambda _phase, _index: None,
        cancel=lambda: False,
    )
    assert result == "ready"
    assert sink.frames == list(range(count))
    assert predictor.closed
    assert pipeline.maximum_window_frames == 64
    assert pipeline.maximum_overlap_frames == 16
    assert predictor.calls == 63
    assert len(sink.boundaries) == 62
    assert all(item.triggered_conditions == ("relative_scale",) for item in sink.boundaries)
    marker = {
        "frames": count,
        "windows": predictor.calls,
        "boundaries": len(sink.boundaries),
        "maximum_window_frames": pipeline.maximum_window_frames,
        "maximum_overlap_frames": pipeline.maximum_overlap_frames,
        "quality_warnings": sum(item.warning() is not None for item in sink.boundaries),
        "installed_module": str(Path(installed_module.__file__).resolve()),
    }
    print("LINGBOT_MAP_LONG_WINDOW=" + json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
