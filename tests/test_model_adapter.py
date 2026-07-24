from __future__ import annotations

import unittest
from types import SimpleNamespace

from lingbot_map.model_adapter import (
    AdapterStateError,
    CanonicalFrame,
    FrameOrderError,
    FrameType,
    GCTStreamBackend,
    InvalidPredictionError,
    ReconstructionModelAdapter,
)


class FakeArray:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeBatch:
    def __init__(
        self,
        count,
        *,
        depth_shape=(7, 11),
        finite=True,
        camera_shape=None,
        tracker=None,
    ):
        self.camera_shape = camera_shape or (count, 9)
        self.depth_shape = (count, *depth_shape)
        self.confidence_shape = self.depth_shape
        self._finite = finite
        self._depth_frame_shape = depth_shape
        self.released = False
        self._tracker = tracker
        if tracker is not None:
            tracker["live"] += 1
            tracker["peak"] = max(tracker["peak"], tracker["live"])

    def all_finite(self):
        return self._finite

    def frame(self, index):
        return (
            FakeArray((9,)),
            FakeArray(self._depth_frame_shape),
            FakeArray(self._depth_frame_shape),
        )

    def release(self):
        if self.released:
            return
        self.released = True
        if self._tracker is not None:
            self._tracker["live"] -= 1


class RecordingBackend:
    def __init__(self, *, invalid_batch=None):
        self.begin_calls = []
        self.persist_flags = []
        self.batches = []
        self.closed = False
        self.tracker = {"live": 0, "peak": 0}
        self.invalid_batch = invalid_batch

    def begin(self, scale_frames):
        self.begin_calls.append(tuple(scale_frames))
        batch = self.invalid_batch or FakeBatch(
            len(scale_frames), tracker=self.tracker
        )
        self.batches.append(batch)
        return batch

    def predict(self, frame, *, persist_keyframe):
        self.persist_flags.append(persist_keyframe)
        batch = FakeBatch(1, tracker=self.tracker)
        self.batches.append(batch)
        return batch

    def close(self):
        self.closed = True


class NativeModelDouble:
    use_sdpa = True
    enable_camera = True
    enable_depth = True
    enable_point = False
    enable_local_point = False
    point_head = None
    local_point_head = None

    def __init__(self):
        self.aggregator = type("Aggregator", (), {"use_flashinfer": False})()
        self.cleaned = 0

    def modules(self):
        return (self, self.aggregator)

    def eval(self):
        return self

    def clean_kv_cache(self):
        self.cleaned += 1


FRAME_SHAPE = (3, 518, 392)


def canonical_frame(index):
    return CanonicalFrame(index, FakeArray(FRAME_SHAPE))


class ReconstructionModelAdapterTests(unittest.TestCase):
    def test_native_backend_accepts_only_guarded_sdpa_camera_depth_model(self):
        model = NativeModelDouble()
        backend = GCTStreamBackend(model, torch_module=object())

        backend.close()

        self.assertEqual(model.cleaned, 1)

    def test_native_backend_rejects_flashinfer_compile_and_point_heads(self):
        cases = []

        flashinfer = NativeModelDouble()
        flashinfer.aggregator.use_flashinfer = True
        cases.append((flashinfer, "FlashInfer"))

        compiled = NativeModelDouble()
        compiled.aggregator._orig_mod = object()
        cases.append((compiled, "compiled"))

        point_head = NativeModelDouble()
        point_head.point_head = object()
        cases.append((point_head, "point head"))

        for model, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    GCTStreamBackend(model, torch_module=object())

    def test_native_backend_converts_canonical_numpy_like_input_to_tensor(self):
        converted = []
        tensor = object()
        fake_torch = SimpleNamespace(
            is_tensor=lambda value: value is tensor,
            as_tensor=lambda value: converted.append(value) or tensor,
        )
        backend = GCTStreamBackend(NativeModelDouble(), torch_module=fake_torch)
        canonical = object()

        self.assertIs(backend._input_tensor(canonical), tensor)
        self.assertIs(backend._input_tensor(tensor), tensor)
        self.assertEqual(converted, [canonical])
        backend.close()

    def test_incremental_scale_keyframe_and_non_keyframe_order(self):
        backend = RecordingBackend()
        adapter = ReconstructionModelAdapter(
            backend,
            frame_shape=FRAME_SHAPE,
            num_scale_frames=2,
            keyframe_interval=2,
        )

        self.assertEqual(adapter.submit(canonical_frame(0)), ())
        scale = adapter.submit(canonical_frame(1))
        first = adapter.submit(canonical_frame(2))
        second = adapter.submit(canonical_frame(3))
        third = adapter.submit(canonical_frame(4))

        self.assertEqual([item.frame_index for item in scale], [0, 1])
        self.assertEqual(
            [item.frame_type for item in scale],
            [FrameType.SCALE, FrameType.SCALE],
        )
        self.assertEqual(first[0].frame_type, FrameType.KEYFRAME)
        self.assertEqual(second[0].frame_type, FrameType.NON_KEYFRAME)
        self.assertEqual(third[0].frame_type, FrameType.KEYFRAME)
        self.assertEqual(backend.persist_flags, [True, False, True])
        self.assertEqual(first[0].camera_shape, (9,))
        self.assertEqual(first[0].depth_shape, (7, 11))
        self.assertEqual(first[0].confidence_shape, (7, 11))
        self.assertTrue(all(batch.released for batch in backend.batches))

        adapter.finish()
        self.assertTrue(backend.closed)

    def test_rejects_out_of_order_frame_without_advancing(self):
        backend = RecordingBackend()
        adapter = ReconstructionModelAdapter(
            backend, frame_shape=FRAME_SHAPE, num_scale_frames=2
        )

        with self.assertRaisesRegex(FrameOrderError, "expected source frame 0"):
            adapter.submit(canonical_frame(1))

        self.assertEqual(adapter.next_frame_index, 0)
        self.assertEqual(adapter.buffered_frame_count, 0)
        self.assertEqual(backend.begin_calls, [])
        adapter.close()

    def test_invalid_non_finite_output_fails_closed_and_releases_batch(self):
        invalid = FakeBatch(2, finite=False)
        backend = RecordingBackend(invalid_batch=invalid)
        adapter = ReconstructionModelAdapter(
            backend, frame_shape=FRAME_SHAPE, num_scale_frames=2
        )

        adapter.submit(canonical_frame(0))
        with self.assertRaisesRegex(InvalidPredictionError, "must all be finite"):
            adapter.submit(canonical_frame(1))

        self.assertTrue(invalid.released)
        self.assertTrue(backend.closed)
        with self.assertRaises(AdapterStateError):
            adapter.submit(canonical_frame(2))

    def test_invalid_camera_shape_fails_closed(self):
        invalid = FakeBatch(2, camera_shape=(2, 8))
        backend = RecordingBackend(invalid_batch=invalid)
        adapter = ReconstructionModelAdapter(
            backend, frame_shape=FRAME_SHAPE, num_scale_frames=2
        )

        adapter.submit(canonical_frame(0))
        with self.assertRaisesRegex(InvalidPredictionError, "camera output shape"):
            adapter.submit(canonical_frame(1))

        self.assertTrue(invalid.released)
        self.assertTrue(backend.closed)

    def test_prediction_batches_and_scale_buffer_remain_bounded(self):
        backend = RecordingBackend()
        adapter = ReconstructionModelAdapter(
            backend,
            frame_shape=FRAME_SHAPE,
            num_scale_frames=8,
            keyframe_interval=3,
        )
        peak_buffered = 0

        for index in range(200):
            adapter.submit(canonical_frame(index))
            peak_buffered = max(peak_buffered, adapter.buffered_frame_count)
            self.assertEqual(backend.tracker["live"], 0)

        self.assertLessEqual(peak_buffered, 7)
        self.assertEqual(adapter.buffered_frame_count, 0)
        self.assertEqual(backend.tracker["peak"], 1)
        adapter.finish()

    def test_short_sequence_is_not_padded(self):
        backend = RecordingBackend()
        adapter = ReconstructionModelAdapter(
            backend, frame_shape=FRAME_SHAPE, num_scale_frames=8
        )
        adapter.submit(canonical_frame(0))

        with self.assertRaisesRegex(AdapterStateError, "8 scale frames required"):
            adapter.finish()

        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
