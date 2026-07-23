from __future__ import annotations

from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    from PIL import Image  # noqa: F401
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

if np is not None:
    from lingbot_map.model_adapter import ReconstructionModelAdapter
    from lingbot_map_worker.canonical_preprocessing import canonicalize_srgb
    from lingbot_map_worker.gpu_profiles import (
        PROFILES,
        ProfileSelection,
        inference_plan,
        resolve_profile,
    )
    from lingbot_map_worker.short_pipeline import (
        NullExecutionGuard,
        PipelineCancelled,
        ProgressReporter,
        ShortReconstructionPipeline,
        SourceFrame,
    )
    from lingbot_map_worker.sleep_guard import SleepGuardError, WindowsSleepGuard
    from lingbot_map_worker.result_pipeline import (
        AlignedPrediction,
        IncrementalBundleResultSink,
        IncrementalResultRequest,
        ResultProfile,
    )
    from lingbot_map_worker.result_resources import FixedResourceProbe, ResourceGateError
    from lingbot_map_worker.provenance import result_provenance
    from lingbot_map_worker.production_model import TorchPredictionDecoder
    import lingbot_map.utils.pose_enc as pose_enc


class FakeArray:
    def __init__(self, shape):
        self.shape = tuple(shape)


class Batch:
    def __init__(self, count, tracker):
        self.camera_shape = (count, 9)
        self.depth_shape = (count, 2, 2)
        self.confidence_shape = (count, 2, 2)
        self.tracker = tracker
        tracker["live"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["live"])

    def all_finite(self):
        return True

    def frame(self, _index):
        return FakeArray((9,)), FakeArray((2, 2)), FakeArray((2, 2))

    def release(self):
        self.tracker["live"] -= 1


class Backend:
    def __init__(self, fail_at=None):
        self.tracker = {"live": 0, "peak": 0}
        self.persist = []
        self.calls = 0
        self.fail_at = fail_at
        self.closed = False

    def begin(self, frames):
        return Batch(len(frames), self.tracker)

    def predict(self, _frame, *, persist_keyframe):
        self.calls += 1
        self.persist.append(persist_keyframe)
        if self.calls == self.fail_at:
            raise RuntimeError("CUDA device lost")
        return Batch(1, self.tracker)

    def close(self):
        self.closed = True


class Sink:
    def __init__(self):
        self.predictions = []
        self.finished = 0

    def accept(self, prediction, _canonical, pts):
        self.predictions.append((prediction.frame_index, int(prediction.frame_type), pts))

    def finish(self):
        self.finished += 1
        return "ready"


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def tick(self, seconds):
        self.value += seconds


def canonical(_rgb):
    from lingbot_map_worker.canonical_preprocessing import CanonicalImage

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


@unittest.skipIf(np is None, "Worker NumPy/Pillow stack is not installed")
class ProfileAndPreprocessingTests(unittest.TestCase):
    def test_named_profiles_custom_confirmation_and_absolute_max(self):
        self.assertEqual(
            [(p.name, p.confidence_cutoff_percent, p.camera_iterations, p.import_point_budget) for p in PROFILES],
            [("Draft", 70, 1, 1_000_000), ("Balanced", 50, 4, 5_000_000), ("High", 30, 4, 10_000_000)],
        )
        for profile in PROFILES:
            selected = ProfileSelection(
                profile.name,
                profile.camera_iterations,
                profile.confidence_cutoff_percent,
                profile.depth_cutoff_percent,
                profile.import_point_budget,
            )
            self.assertEqual(resolve_profile(selected).name, profile.name)
        changed = ProfileSelection("High", 4, 29, 99.5, 10_000_000)
        self.assertEqual(resolve_profile(changed).name, "Custom")
        retained = ProfileSelection(
            "High", 4, 30, 99.5, 10_000_000, False, True
        )
        self.assertEqual(resolve_profile(retained).name, "Custom")
        with self.assertRaisesRegex(ValueError, "confirmation"):
            resolve_profile(ProfileSelection("Custom", 4, 50, 99.5, 10_000_001))
        self.assertEqual(
            resolve_profile(ProfileSelection("Custom", 4, 50, 99.5, 10_000_001, True)).name,
            "Custom",
        )
        with self.assertRaisesRegex(ValueError, "50000000"):
            resolve_profile(ProfileSelection("Custom", 4, 50, 99.5, 50_000_001, True))

    def test_mode_and_keyframe_boundary_goldens(self):
        expected = {
            8: ("streaming", 1),
            320: ("streaming", 1),
            321: ("streaming", 2),
            3000: ("streaming", 10),
            3001: ("windowed", 1),
        }
        for count, value in expected.items():
            with self.subTest(count=count):
                plan = inference_plan(count)
                self.assertEqual((plan.mode, plan.keyframe_interval), value)

    def test_canonical_resize_crop_color_and_normalization_are_fixed(self):
        landscape = np.full((4, 8, 3), (64, 128, 255), dtype="|u1")
        result = canonicalize_srgb(landscape)
        self.assertEqual(result.color_rgb.shape, (252, 518, 3))
        self.assertEqual(result.model_input.shape, (3, 252, 518))
        self.assertEqual(result.color_rgb[10, 10].tolist(), [64, 128, 255])
        np.testing.assert_allclose(result.model_input[:, 10, 10], np.array((64, 128, 255)) / 255)
        portrait = canonicalize_srgb(np.zeros((8, 4, 3), dtype="|u1"))
        self.assertEqual(portrait.color_rgb.shape, (518, 518, 3))
        self.assertEqual(portrait.source_to_model[1, 2], -259.0)


@unittest.skipIf(np is None, "Worker NumPy/Pillow stack is not installed")
class ShortPipelineTests(unittest.TestCase):
    def _run(self, count, *, backend=None, health=lambda _phase, _index: None, cancel=lambda: False):
        backend = backend or Backend()
        factories = []
        def factory(plan, profile, shape):
            factories.append((plan, profile, shape))
            return ReconstructionModelAdapter(
                backend,
                frame_shape=shape,
                num_scale_frames=plan.scale_frames,
                keyframe_interval=plan.keyframe_interval,
            )
        pipeline = ShortReconstructionPipeline(factory, preprocessor=canonical)
        sink = Sink()
        events = []
        progress = ProgressReporter(events.append)
        result = pipeline.run(
            frame_count=count,
            frames=frames(count),
            profile=ProfileSelection("Draft", 1, 70, 99.5, 1_000_000),
            result_sink=sink,
            progress=progress,
            execution_guard=NullExecutionGuard(),
            health_check=health,
            cancel=cancel,
        )
        return result, pipeline, sink, backend, factories, events

    def test_8_320_321_and_3000_finalize_every_frame_with_bounded_history(self):
        for count, interval in ((8, 1), (320, 1), (321, 2), (3000, 10)):
            with self.subTest(count=count):
                result, pipeline, sink, backend, factories, _events = self._run(count)
                self.assertEqual(result, "ready")
                self.assertEqual(len(sink.predictions), count)
                self.assertEqual([item[0] for item in sink.predictions], list(range(count)))
                self.assertLessEqual(pipeline.maximum_pending_colors, 8)
                self.assertEqual(backend.tracker["peak"], 1)
                self.assertEqual(factories[0][0].keyframe_interval, interval)
                self.assertEqual(sink.finished, 1)

    def test_cuda_resource_and_cancellation_fail_without_retry_or_finish(self):
        backend = Backend(fail_at=2)
        with self.assertRaisesRegex(RuntimeError, "CUDA device lost"):
            self._run(12, backend=backend)
        self.assertTrue(backend.closed)

        def no_memory(phase, index):
            if phase == "inference" and index == 4:
                raise MemoryError("Worker Memory Gate")
        with self.assertRaisesRegex(MemoryError, "Worker Memory Gate"):
            self._run(12, health=no_memory)

        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 7
        with self.assertRaises(PipelineCancelled):
            self._run(20, cancel=cancel)
        self.assertGreaterEqual(checks, 7)

    def test_progress_is_throttled_and_eta_requires_stable_samples(self):
        clock = Clock()
        events = []
        reporter = ProgressReporter(events.append, monotonic=clock)
        reporter.boundary("inference", 0, 20)
        for completed in range(1, 11):
            clock.tick(0.1)
            reporter.advance(completed)
        ordinary = [event for event in events if event.kind == "progress"]
        self.assertLessEqual(len(ordinary), 4)
        self.assertTrue(any(event.eta_seconds is not None for event in ordinary))
        reporter.resume()
        self.assertIsNone(events[-1].eta_seconds)
        self.assertTrue(events[-1].immediate)

    def test_complete_pipeline_seam_publishes_ready_result_incrementally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "target.lingbot-map"
            (project / "results").mkdir(parents=True)
            (project / "diagnostics").mkdir()
            blend = root / "target.blend"
            blend.touch()
            source = root / "capture.mp4"
            source.write_bytes(b"short-pipeline-source")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            request = IncrementalResultRequest(
                job_id="job-" + "1" * 32,
                project_root=project,
                target_scene={
                    "blend_path": str(blend),
                    "scene_uuid": "12345678-1234-1234-1234-123456789abc",
                    "scene_name": "Scene",
                },
                timeline_start=11,
                source={
                    "absolute_path": str(source),
                    "scene_relative_path": "//capture.mp4",
                    "size_bytes": source.stat().st_size,
                    "modification_time_ns": source.stat().st_mtime_ns,
                    "sha256": source_sha,
                },
                source_to_model=np.eye(3, dtype="<f8"),
                frame_count=8,
                profile=ResultProfile("Draft", 70, 99.5, 1_000_000, 0.01),
                provenance=result_provenance(
                    runtime_id="2" * 64,
                    worker_version="0.1.0",
                    job_spec_sha256="3" * 64,
                    model_id="fixture-model",
                    model_sha256="4" * 64,
                    source_sha256=source_sha,
                    profile_name="Draft",
                    camera_iterations=1,
                    confidence_cutoff_percent=70,
                    depth_cutoff_percent=99.5,
                    import_point_budget=1_000_000,
                    plan=inference_plan(8),
                    gpu=None,
                    suspension_count=0,
                    suspension_seconds=0,
                    fixture=True,
                ),
                created_utc="2026-07-23T05:00:00+00:00",
                result_id="result-" + "5" * 32,
            )
            def decode(prediction, canonical_image, pts):
                return AlignedPrediction(
                    prediction.frame_index,
                    int(prediction.frame_type),
                    pts,
                    np.eye(4, dtype="<f8"),
                    np.array(((2, 0, 0.5), (0, 2, 0.5), (0, 0, 1)), dtype="<f8"),
                    np.ones((2, 2), dtype="<f4"),
                    np.array(((1, 2), (3, 4)), dtype="<f4"),
                    canonical_image.color_rgb,
                )
            with self.assertRaisesRegex(ResourceGateError, "Disk Gate"):
                IncrementalBundleResultSink(
                    request,
                    prediction_decoder=decode,
                    resource_probe=FixedResourceProbe(0, 20 * 1024**3),
                    cancel=lambda: False,
                )
            memory_sink = IncrementalBundleResultSink(
                request,
                prediction_decoder=decode,
                resource_probe=FixedResourceProbe(20 * 1024**3, 0),
                cancel=lambda: False,
            )
            memory_backend = Backend()
            memory_pipeline = ShortReconstructionPipeline(
                lambda plan, _profile, shape: ReconstructionModelAdapter(
                    memory_backend,
                    frame_shape=shape,
                    num_scale_frames=plan.scale_frames,
                    keyframe_interval=plan.keyframe_interval,
                ),
                preprocessor=canonical,
            )
            with self.assertRaisesRegex(ResourceGateError, "Memory Gate"):
                memory_pipeline.run(
                    frame_count=8,
                    frames=frames(8),
                    profile=ProfileSelection("Draft", 1, 70, 99.5, 1_000_000),
                    result_sink=memory_sink,
                    progress=ProgressReporter(lambda _event: None),
                    execution_guard=NullExecutionGuard(),
                    health_check=lambda _phase, _index: None,
                    cancel=lambda: False,
                )
            self.assertTrue(memory_backend.closed)
            sink = IncrementalBundleResultSink(
                request,
                prediction_decoder=decode,
                resource_probe=FixedResourceProbe(20 * 1024**3, 20 * 1024**3),
                cancel=lambda: False,
            )
            backend = Backend()
            pipeline = ShortReconstructionPipeline(
                lambda plan, _profile, shape: ReconstructionModelAdapter(
                    backend,
                    frame_shape=shape,
                    num_scale_frames=plan.scale_frames,
                    keyframe_interval=plan.keyframe_interval,
                ),
                preprocessor=canonical,
            )
            outcome = pipeline.run(
                frame_count=8,
                frames=frames(8),
                profile=ProfileSelection("Draft", 1, 70, 99.5, 1_000_000),
                result_sink=sink,
                progress=ProgressReporter(lambda _event: None),
                execution_guard=NullExecutionGuard(),
                health_check=lambda _phase, _index: None,
                cancel=lambda: False,
            )
            self.assertTrue(outcome.published.directory.is_dir())
            self.assertEqual(outcome.published.manifest["counts"]["frames"], 8)
            self.assertEqual(sink.finalized_frame_count, 8)

    def test_production_prediction_decoder_emits_cpu_world_to_camera_depth_and_confidence(self):
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def reshape(self, *shape): return Tensor(self.value.reshape(*shape))
            def detach(self): return self
            def cpu(self): return self
            def float(self): return Tensor(self.value.astype(np.float32))
            def numpy(self): return self.value
            def __getitem__(self, key): return Tensor(self.value[key])

        extrinsics = Tensor(np.array([[[
            [1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]
        ]]], dtype=np.float32))
        intrinsics = Tensor(np.array([[[
            [2, 0, 1], [0, 2, 1], [0, 0, 1]
        ]]], dtype=np.float32))
        prediction = type("Prediction", (), {
            "frame_index": 0, "frame_type": 1, "depth_shape": (2, 2),
            "camera_pose_encoding": Tensor(np.arange(9, dtype=np.float32)),
            "depth": Tensor(np.ones((2, 2), dtype=np.float32)),
            "confidence": Tensor(np.full((2, 2), 2, dtype=np.float32)),
        })()
        canonical_image = canonicalize_srgb(np.zeros((2, 2, 3), dtype="|u1"))
        with mock.patch.object(
            pose_enc,
            "pose_encoding_to_extri_intri",
            return_value=(extrinsics, intrinsics),
        ):
            decoded = TorchPredictionDecoder()(prediction, canonical_image, 0.25)
        np.testing.assert_array_equal(decoded.world_to_camera_opencv[:3, 3], (2, 3, 4))
        self.assertEqual(decoded.world_to_camera_opencv.dtype.str, "<f8")
        self.assertEqual(decoded.depth.dtype.str, "<f4")
        self.assertEqual(decoded.confidence.dtype.str, "<f4")


class FakeExecutionState:
    def __init__(self):
        self.awake = 0.0
        self.acquired = 0
        self.released = 0
        self.fail = False

    def acquire(self):
        if self.fail:
            raise SleepGuardError("sleep request lost")
        self.acquired += 1

    def release(self):
        self.released += 1

    def awake_seconds(self):
        return self.awake


@unittest.skipIf(np is None, "Worker stack is not installed")
class SleepGuardTests(unittest.TestCase):
    def test_resume_reacquires_and_runs_every_health_gate_before_continuing(self):
        adapter = FakeExecutionState()
        wall = Clock()
        checks = []
        resumed = []
        guard = WindowsSleepGuard(
            adapter,
            resume_checks=(lambda: checks.append("cuda"), lambda: checks.append("resources")),
            on_resume=lambda: resumed.append(True),
            wall_time=wall,
        )
        with guard:
            adapter.awake += 1
            wall.tick(1)
            guard.boundary()
            adapter.awake += 1
            wall.tick(6)
            guard.boundary()
        self.assertEqual(checks, ["cuda", "resources"])
        self.assertEqual(guard.suspension_count, 1)
        self.assertEqual(guard.suspension_seconds, 5.0)
        self.assertEqual(resumed, [True])
        self.assertEqual(adapter.released, 1)

    def test_maintenance_failure_is_fatal(self):
        adapter = FakeExecutionState()
        guard = WindowsSleepGuard(adapter, resume_checks=())
        with self.assertRaisesRegex(SleepGuardError, "lost"):
            with guard:
                adapter.fail = True
                adapter.awake += 1
                guard.boundary()


if __name__ == "__main__":
    unittest.main()
