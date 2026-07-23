from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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
    from lingbot_map_worker.sky_masking import (
        SKY_MASK_BATCH_SIZE,
        SKY_MASK_MAX_IN_FLIGHT,
        SKY_MASK_PROVIDER,
        SkyMaskCancelled,
        SkyMaskError,
        SkyMaskRequest,
        SkyMaskSession,
    )


class FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.execution_mode = None


class FakeInferenceSession:
    def __init__(self, runtime, path, options, providers):
        self.runtime = runtime
        self.path = path
        self.options = options
        self.providers_argument = providers
        self.fallback_disabled = False
        self.run_count = 0

    def disable_fallback(self):
        self.fallback_disabled = True

    def get_providers(self):
        return [SKY_MASK_PROVIDER]

    def get_inputs(self):
        return [SimpleNamespace(name="input", type="tensor(float)")]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="primary", type="tensor(float)", shape=[1, 1, 320, 320]
            ),
            SimpleNamespace(
                name="auxiliary", type="tensor(float)", shape=[1, 1, 320, 320]
            ),
        ]

    def run(self, outputs, feeds):
        self.run_count += 1
        if outputs != ["primary"]:
            raise AssertionError(f"unexpected SkySeg output selection: {outputs!r}")
        self.runtime.inputs.append(feeds["input"].copy())
        if self.runtime.fail_after is not None and self.run_count > self.runtime.fail_after:
            raise RuntimeError("injected SkySeg failure")
        gradient = np.linspace(
            -2.0, 2.0, 320 * 320, dtype="<f4"
        ).reshape(1, 1, 320, 320)
        return [gradient]


class FakeRuntime:
    __version__ = "1.23.2"
    SessionOptions = FakeSessionOptions
    ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")

    def __init__(self, *, fail_after=None):
        self.fail_after = fail_after
        self.inputs = []
        self.sessions = []

    def InferenceSession(self, path, *, sess_options, providers):
        session = FakeInferenceSession(self, path, sess_options, providers)
        self.sessions.append(session)
        return session


@unittest.skipIf(np is None, "NumPy is not installed")
class SkyMaskSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "models" / "skyseg.onnx"
        self.model.parent.mkdir()
        self.model.write_bytes(b"catalogued fake SkySeg model")
        self.model_sha = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.source_sha = hashlib.sha256(b"source capture").hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def request(self, *, frames=3, cancel=lambda: False, **overrides):
        values = {
            "managed_root": self.root,
            "source_sha256": self.source_sha,
            "video_stream_index": 0,
            "display_transform": "identity",
            "color_standard": "bt709",
            "color_range": "limited",
            "frame_count": frames,
            "model_grid_shape": (2, 2),
            "model_id": "skyseg",
            "model_path": self.model,
            "model_sha256": self.model_sha,
            "worker_version": "0.1.0",
            "onnx_threads": 2,
            "cancel": cancel,
        }
        values.update(overrides)
        return SkyMaskRequest(**values)

    @staticmethod
    def canonical(value=64):
        color = np.full((2, 2, 3), value, dtype="|u1")
        model = np.ascontiguousarray(color.transpose(2, 0, 1), dtype="<f4")
        model /= np.float32(255.0)
        return CanonicalImage(
            model,
            color,
            np.eye(3, dtype="<f8"),
            ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        )

    def run_complete(self, runtime, request=None):
        request = request or self.request()
        session = SkyMaskSession(request, runtime_module=runtime)
        masks = []
        for index in range(request.frame_count):
            canonical = self.canonical(index + 32)
            model_before = canonical.model_input.copy()
            color_before = canonical.color_rgb.copy()
            session.prepare(index, canonical)
            mask, fraction = session.mask_for(index, request.model_grid_shape)
            masks.append(mask)
            self.assertGreaterEqual(float(fraction), 0.0)
            self.assertLessEqual(float(fraction), 1.0)
            np.testing.assert_array_equal(canonical.model_input, model_before)
            np.testing.assert_array_equal(canonical.color_rgb, color_before)
        return session, session.finish(), masks

    def test_fixed_cpu_provider_threads_batch_and_unchanged_model_tensor(self):
        runtime = FakeRuntime()
        session, outcome, masks = self.run_complete(runtime)
        ort = runtime.sessions[0]
        self.assertEqual(ort.providers_argument, [SKY_MASK_PROVIDER])
        self.assertTrue(ort.fallback_disabled)
        self.assertEqual(ort.options.intra_op_num_threads, 2)
        self.assertEqual(ort.options.inter_op_num_threads, 1)
        self.assertEqual(ort.options.execution_mode, "sequential")
        self.assertEqual(outcome.provenance["provider"], SKY_MASK_PROVIDER)
        self.assertEqual(outcome.provenance["batch_size"], SKY_MASK_BATCH_SIZE)
        self.assertEqual(outcome.provenance["onnx_threads"], 2)
        self.assertEqual(outcome.sky_fraction.shape, (3,))
        self.assertEqual(len(masks), 3)
        self.assertEqual(ort.run_count, 4)  # one session probe plus every source frame

    def test_production_module_has_no_network_or_child_process_surface(self):
        path = WORKER_SOURCE / "lingbot_map_worker" / "sky_masking.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden = {"httpx", "multiprocessing", "requests", "socket", "subprocess", "urllib"}
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        self.assertEqual(imports & forbidden, set())

    def test_complete_cache_is_reused_without_source_frame_inference(self):
        first, first_outcome, first_masks = self.run_complete(FakeRuntime())
        self.assertEqual(first_outcome.provenance["cache_status"], "generated")
        self.assertTrue((first.cache_root / first.cache_key / "manifest.json").is_file())

        runtime = FakeRuntime()
        second, second_outcome, second_masks = self.run_complete(runtime)
        self.assertEqual(second.cache_key, first.cache_key)
        self.assertEqual(second_outcome.provenance["cache_status"], "hit")
        self.assertEqual(runtime.sessions[0].run_count, 1)  # prelaunch session probe only
        for expected, actual in zip(first_masks, second_masks):
            np.testing.assert_array_equal(actual, expected)

    def test_corrupt_cache_is_quarantined_and_regenerated_once(self):
        first, _outcome, _masks = self.run_complete(FakeRuntime())
        chunk = next((first.cache_root / first.cache_key / "chunks").glob("*.npy"))
        chunk.write_bytes(b"corrupt")

        runtime = FakeRuntime()
        regenerated, outcome, _masks = self.run_complete(runtime)
        self.assertEqual(outcome.provenance["cache_status"], "regenerated")
        self.assertIn("sky_mask_cache_corrupt", {item["code"] for item in outcome.warnings})
        self.assertEqual(runtime.sessions[0].run_count, 4)
        quarantined = tuple((regenerated.cache_root / "corrupt").iterdir())
        self.assertEqual(len(quarantined), 1)

        second_chunk = next(
            (regenerated.cache_root / regenerated.cache_key / "chunks").glob("*.npy")
        )
        second_chunk.write_bytes(b"corrupt again")
        failing_runtime = FakeRuntime(fail_after=1)
        failing = SkyMaskSession(
            self.request(), runtime_module=failing_runtime
        )
        failing.prepare(0, self.canonical())
        with self.assertRaisesRegex(SkyMaskError, "generation failed"):
            failing.mask_for(0, (2, 2))
        failing.abort()
        self.assertEqual(failing_runtime.sessions[0].run_count, 2)
        self.assertEqual(len(tuple((failing.cache_root / "corrupt").iterdir())), 2)

    def test_cancellation_never_finishes_a_mixed_mask_set(self):
        cancelled = {"value": False}
        runtime = FakeRuntime()
        session = SkyMaskSession(
            self.request(cancel=lambda: cancelled["value"]),
            runtime_module=runtime,
        )
        session.prepare(0, self.canonical())
        cancelled["value"] = True
        with self.assertRaises(SkyMaskCancelled):
            session.mask_for(0, (2, 2))
        session.abort()
        self.assertFalse((session.cache_root / session.cache_key).exists())

    def test_queue_is_bounded_to_the_fixed_long_window(self):
        runtime = FakeRuntime()
        session = SkyMaskSession(
            self.request(frames=SKY_MASK_MAX_IN_FLIGHT + 1),
            runtime_module=runtime,
        )
        for index in range(SKY_MASK_MAX_IN_FLIGHT):
            session.prepare(index, self.canonical())
        self.assertEqual(session.maximum_in_flight, SKY_MASK_MAX_IN_FLIGHT)
        with self.assertRaisesRegex(SkyMaskError, "bounded 64-frame queue"):
            session.prepare(SKY_MASK_MAX_IN_FLIGHT, self.canonical())
        session.abort()

    def test_cache_identity_changes_with_display_preprocessing_identity(self):
        first = SkyMaskSession(self.request(), runtime_module=FakeRuntime())
        first_key = first.cache_key
        first.abort()
        second = SkyMaskSession(
            self.request(display_transform="rotate-90"),
            runtime_module=FakeRuntime(),
        )
        self.assertNotEqual(second.cache_key, first_key)
        second.abort()


if __name__ == "__main__":
    unittest.main()
