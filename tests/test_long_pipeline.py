from __future__ import annotations

from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

if np is not None:
    from lingbot_map_worker.canonical_preprocessing import CanonicalImage
    from lingbot_map_worker.gpu_profiles import ProfileSelection, inference_plan
    from lingbot_map_worker.long_pipeline import (
        LongPipelineError,
        WindowedReconstructionPipeline,
    )
    from lingbot_map_worker.provenance import result_provenance
    from lingbot_map_worker.result_pipeline import (
        AlignedPrediction,
        IncrementalBundleResultSink,
        IncrementalResultRequest,
        ResultProfile,
    )
    from lingbot_map_worker.result_resources import FixedResourceProbe
    from lingbot_map_worker.short_pipeline import (
        NullExecutionGuard,
        PipelineCancelled,
        ProgressReporter,
        SourceFrame,
    )
    from lingbot_map_worker.window_alignment import (
        RollingWindowAligner,
        SimilarityTransform,
        WindowAlignmentError,
        apply_similarity,
    )


def canonical(_rgb):
    return CanonicalImage(
        np.zeros((3, 2, 2), dtype="<f4"),
        np.zeros((2, 2, 3), dtype="|u1"),
        np.eye(3, dtype="<f8"),
        ((0, 0), (2, 0), (2, 2), (0, 2)),
    )


def frames(count):
    rgb = np.zeros((1, 1, 3), dtype="|u1")
    for index in range(count):
        yield SourceFrame(index, index / 25.0, rgb)


class Predictor:
    def __init__(self, *, scaled=False, reflected=False):
        self.scaled = scaled
        self.reflected = reflected
        self.calls = 0
        self.closed = False
        self.maximum_input = 0

    def predict(self, window, *, cancel):
        self.calls += 1
        self.maximum_input = max(self.maximum_input, len(window))
        scale = 0.5 if self.scaled and self.calls > 1 else 1.0
        output = []
        for source in window:
            if cancel():
                raise PipelineCancelled("cancelled in predictor")
            camera_to_world = np.eye(4, dtype="<f8")
            camera_to_world[0, 3] = source.frame_index * 0.01 * scale
            if self.reflected and self.calls > 1:
                camera_to_world[0, 0] = -1.0
            world_to_camera = np.ascontiguousarray(
                np.linalg.inv(camera_to_world), dtype="<f8"
            )
            output.append(
                AlignedPrediction(
                    source.frame_index,
                    1,
                    source.pts_seconds,
                    world_to_camera,
                    np.ascontiguousarray(
                        ((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)),
                        dtype="<f8",
                    ),
                    np.full((2, 2), scale, dtype="<f4"),
                    np.ones((2, 2), dtype="<f4"),
                    np.zeros((2, 2, 3), dtype="|u1"),
                )
            )
        return tuple(output)

    def close(self):
        self.closed = True


class Sink:
    def __init__(self):
        self.identities = []
        self.prepared = {}
        self.prepared_count = 0
        self.maximum_prepared = 0
        self.boundaries = []
        self.finished = 0

    def prepare_frame(self, frame_index, canonical):
        self.prepared[frame_index] = canonical
        self.prepared_count += 1
        self.maximum_prepared = max(self.maximum_prepared, len(self.prepared))

    def accept(self, prediction, canonical, pts):
        if self.prepared.pop(prediction.frame_index, None) is not canonical:
            raise AssertionError("long Pipeline did not retain the same canonical frame")
        self.identities.append((prediction.frame_index, prediction.frame_type, pts))

    def record_window_boundary(self, metrics):
        self.boundaries.append(metrics)

    def finish(self):
        self.finished += 1
        return "ready"


@unittest.skipIf(np is None, "Worker NumPy stack is not installed")
class LongPipelineTests(unittest.TestCase):
    def _run(self, count=3001, *, predictor=None, cancel=lambda: False):
        predictor = predictor or Predictor()
        constructed = []

        def factory(plan, _profile, shape):
            constructed.append((plan, shape))
            return predictor

        pipeline = WindowedReconstructionPipeline(factory, preprocessor=canonical)
        sink = Sink()
        result = pipeline.run(
            frame_count=count,
            frames=frames(count),
            profile=ProfileSelection("Draft", 1, 70, 99.5, 1_000_000),
            result_sink=sink,
            progress=ProgressReporter(lambda _event: None),
            execution_guard=NullExecutionGuard(),
            health_check=lambda _phase, _index: None,
            cancel=cancel,
        )
        return result, pipeline, sink, predictor, constructed

    def test_3001_and_multiple_windows_finalize_every_presentation_frame_once(self):
        for count in (3001, 3100):
            with self.subTest(count=count):
                result, pipeline, sink, predictor, constructed = self._run(count)
                self.assertEqual(result, "ready")
                self.assertEqual(
                    [item[0] for item in sink.identities], list(range(count))
                )
                self.assertEqual(sink.prepared_count, count)
                self.assertFalse(sink.prepared)
                self.assertLessEqual(sink.maximum_prepared, 64)
                self.assertEqual([item[1] for item in sink.identities[:8]], [0] * 8)
                self.assertTrue(all(item[1] == 1 for item in sink.identities[8:]))
                self.assertLessEqual(pipeline.maximum_window_frames, 64)
                self.assertLessEqual(pipeline.maximum_overlap_frames, 16)
                self.assertEqual(len(sink.boundaries), predictor.calls - 1)
                self.assertEqual(len(constructed), 1)
                self.assertEqual(constructed[0][0].window_frames, 64)
                self.assertEqual(constructed[0][0].overlap_keyframes, 16)
                self.assertEqual(predictor.maximum_input, 64)
                self.assertTrue(predictor.closed)
                self.assertEqual(sink.finished, 1)

    def test_high_computable_scale_residual_is_a_stable_warning(self):
        _, _, sink, _, _ = self._run(predictor=Predictor(scaled=True))
        first = sink.boundaries[0]
        self.assertEqual(first.triggered_conditions, ("relative_scale",))
        self.assertEqual(first.warning()["code"], "quality_warning")
        self.assertIn("relative_scale", first.warning()["message"])

    def test_reflection_and_invalid_similarity_fail_without_retry_or_finish(self):
        predictor = Predictor(reflected=True)
        with self.assertRaisesRegex(WindowAlignmentError, "reflected"):
            self._run(predictor=predictor)
        self.assertTrue(predictor.closed)

        frame = Predictor().predict(
            tuple(
                type("F", (), {"frame_index": index, "pts_seconds": index / 25, "canonical": canonical(None)})
                for index in range(8)
            ),
            cancel=lambda: False,
        )[0]
        with self.assertRaises(WindowAlignmentError):
            apply_similarity(
                frame,
                SimilarityTransform(1.0, np.diag((-1.0, 1.0, 1.0)), np.zeros(3)),
            )

    def test_missing_overlap_depth_and_identity_mismatch_fail_structurally(self):
        predictor = Predictor()
        window = tuple(
            type("F", (), {"frame_index": index, "pts_seconds": index / 25, "canonical": canonical(None)})
            for index in range(16)
        )
        previous = list(predictor.predict(window, cancel=lambda: False))
        current = list(predictor.predict(window, cancel=lambda: False))
        current[0] = AlignedPrediction(
            999,
            current[0].frame_type,
            current[0].source_pts_seconds,
            current[0].world_to_camera_opencv,
            current[0].model_intrinsics,
            current[0].depth,
            current[0].confidence,
            current[0].rgb,
        )
        with self.assertRaisesRegex(WindowAlignmentError, "identities"):
            RollingWindowAligner().estimate(previous, current)
        current[0] = previous[0]
        current = [
            AlignedPrediction(
                item.frame_index,
                item.frame_type,
                item.source_pts_seconds,
                item.world_to_camera_opencv,
                item.model_intrinsics,
                np.zeros_like(item.depth),
                item.confidence,
                item.rgb,
            )
            for item in current
        ]
        with self.assertRaisesRegex(WindowAlignmentError, "no valid"):
            RollingWindowAligner().estimate(previous, current)

    def test_float32_pose_noise_uses_documented_rigid_tolerance(self):
        predictor = Predictor()
        window = tuple(
            type(
                "F",
                (),
                {
                    "frame_index": index,
                    "pts_seconds": index / 25,
                    "canonical": canonical(None),
                },
            )
            for index in range(2)
        )
        baseline = list(predictor.predict(window, cancel=lambda: False))

        def with_scale_error(value, error):
            matrix = value.world_to_camera_opencv.copy()
            matrix[0, 0] += error
            return AlignedPrediction(
                value.frame_index,
                value.frame_type,
                value.source_pts_seconds,
                np.ascontiguousarray(matrix, dtype="<f8"),
                value.model_intrinsics,
                value.depth,
                value.confidence,
                value.rgb,
            )

        float32_noise = [
            with_scale_error(item, 2.0e-7) for item in baseline
        ]
        transform = RollingWindowAligner().estimate(
            float32_noise,
            float32_noise,
        )
        self.assertAlmostEqual(transform.scale, 1.0)

        malformed = [
            with_scale_error(item, 1.0e-5) for item in baseline
        ]
        with self.assertRaisesRegex(WindowAlignmentError, "orthonormal"):
            RollingWindowAligner().estimate(malformed, malformed)

    def test_cancellation_closes_predictor_and_never_finishes(self):
        checks = 0

        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 180

        predictor = Predictor()
        with self.assertRaises(PipelineCancelled):
            self._run(predictor=predictor, cancel=cancel)
        self.assertTrue(predictor.closed)

    def test_3000_is_not_a_long_pipeline(self):
        with self.assertRaises(LongPipelineError):
            self._run(3000)

    def test_3001_frame_result_records_alignment_and_explicitly_denies_global_optimization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "target.lingbot-map"
            (project / "results").mkdir(parents=True)
            (project / "diagnostics").mkdir()
            blend = root / "target.blend"
            blend.touch()
            source = root / "capture.mp4"
            source.write_bytes(b"deterministic-long-window-contract-fixture")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            request = IncrementalResultRequest(
                job_id="job-" + "1" * 32,
                project_root=project,
                target_scene={
                    "blend_path": str(blend),
                    "scene_uuid": "12345678-1234-1234-1234-123456789abc",
                    "scene_name": "Scene",
                },
                timeline_start=1,
                source={
                    "absolute_path": str(source),
                    "scene_relative_path": "//capture.mp4",
                    "size_bytes": source.stat().st_size,
                    "modification_time_ns": source.stat().st_mtime_ns,
                    "sha256": source_sha,
                },
                source_to_model=np.eye(3, dtype="<f8"),
                frame_count=3001,
                profile=ResultProfile("Custom", 0, 100, 50_000, 0.01),
                provenance=result_provenance(
                    runtime_id="2" * 64,
                    worker_version="0.1.0",
                    job_spec_sha256="3" * 64,
                    model_id="reconstruction-model",
                    model_sha256="4" * 64,
                    source_sha256=source_sha,
                    profile_name="Custom",
                    camera_iterations=1,
                    confidence_cutoff_percent=0,
                    depth_cutoff_percent=100,
                    import_point_budget=50_000,
                    plan=inference_plan(3001),
                    gpu=None,
                    suspension_count=0,
                    suspension_seconds=0,
                ),
                created_utc="2026-07-23T08:00:00+00:00",
                result_id="result-" + "5" * 32,
            )
            sink = IncrementalBundleResultSink(
                request,
                prediction_decoder=lambda prediction, _canonical, _pts: prediction,
                resource_probe=FixedResourceProbe(20 * 1024**3, 20 * 1024**3),
                cancel=lambda: False,
            )
            predictor = Predictor(scaled=True)
            pipeline = WindowedReconstructionPipeline(
                lambda _plan, _profile, _shape: predictor,
                preprocessor=canonical,
            )
            outcome = pipeline.run(
                frame_count=3001,
                frames=frames(3001),
                profile=ProfileSelection("Custom", 1, 0, 100, 50_000),
                result_sink=sink,
                progress=ProgressReporter(lambda _event: None),
                execution_guard=NullExecutionGuard(),
                health_check=lambda _phase, _index: None,
                cancel=lambda: False,
            )
            manifest = outcome.published.manifest
            self.assertEqual(manifest["counts"]["frames"], 3001)
            alignment = manifest["window_alignment"]
            self.assertEqual(alignment["strategy"], "rolling-similarity")
            self.assertEqual(len(alignment["boundaries"]), predictor.calls - 1)
            self.assertEqual(
                sum(warning["code"] == "quality_warning" for warning in manifest["warnings"]),
                len(alignment["boundaries"]),
            )
            for field in (
                "loop_closure",
                "pose_graph",
                "bundle_adjustment",
                "global_optimization",
            ):
                self.assertFalse(alignment[field])


if __name__ == "__main__":
    unittest.main()
