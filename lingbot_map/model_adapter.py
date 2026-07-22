"""Headless, incremental seam for LingBot-Map camera and depth inference.

The public :class:`ReconstructionModelAdapter` deliberately knows nothing
about downloads, worker runtimes, Blender, files, or networking.  It accepts
already canonicalized frames and delegates model execution to an injected
backend.  This keeps the production PyTorch implementation replaceable by
small controlled doubles in unit tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ContextManager, Mapping, Protocol, Sequence, runtime_checkable


CAMERA_POSE_ENCODING_SHAPE = (9,)


class ModelAdapterError(RuntimeError):
    """Base error for the reconstruction model seam."""


class AdapterStateError(ModelAdapterError):
    """Raised when an operation is invalid for the Adapter state."""


class FrameOrderError(ModelAdapterError):
    """Raised when canonical frames are not submitted in source order."""


class InvalidPredictionError(ModelAdapterError):
    """Raised when model output violates the camera/depth contract."""


class FrameType(IntEnum):
    """Stable frame classification used by the Reconstruction Result schema."""

    SCALE = 0
    KEYFRAME = 1
    NON_KEYFRAME = 2


@dataclass(frozen=True)
class CanonicalFrame:
    """One canonical preprocessed frame with its immutable source identity."""

    frame_index: int
    pixels: Any


@dataclass(frozen=True)
class FramePrediction:
    """Finite camera, depth, and confidence output for one source frame."""

    frame_index: int
    frame_type: FrameType
    camera_pose_encoding: Any
    depth: Any
    confidence: Any
    camera_shape: tuple[int, ...]
    depth_shape: tuple[int, ...]
    confidence_shape: tuple[int, ...]


@runtime_checkable
class PredictionBatch(Protocol):
    """Backend-owned prediction storage consumed and released by the Adapter."""

    @property
    def camera_shape(self) -> tuple[int, ...]: ...

    @property
    def depth_shape(self) -> tuple[int, ...]: ...

    @property
    def confidence_shape(self) -> tuple[int, ...]: ...

    def all_finite(self) -> bool: ...

    def frame(self, index: int) -> tuple[Any, Any, Any]: ...

    def release(self) -> None: ...


@runtime_checkable
class IncrementalModelBackend(Protocol):
    """Deployment-free execution interface used by the model Adapter."""

    def begin(self, scale_frames: Sequence[Any]) -> PredictionBatch: ...

    def predict(self, frame: Any, *, persist_keyframe: bool) -> PredictionBatch: ...

    def close(self) -> None: ...


def _shape_of(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise InvalidPredictionError(f"{label} has no valid explicit shape") from exc


class ReconstructionModelAdapter:
    """Validate and emit bounded incremental camera/depth predictions.

    Initial scale frames are the only input frames buffered by the Adapter.
    Once the scale batch is emitted, every subsequent call produces exactly
    one finalized prediction and the backend batch is released immediately.
    """

    def __init__(
        self,
        backend: IncrementalModelBackend,
        *,
        frame_shape: tuple[int, int, int],
        num_scale_frames: int = 8,
        keyframe_interval: int = 1,
    ) -> None:
        if len(frame_shape) != 3 or frame_shape[0] != 3:
            raise ValueError("frame_shape must be (3, height, width)")
        if frame_shape[1] <= 0 or frame_shape[2] <= 0:
            raise ValueError("canonical frame dimensions must be positive")
        if num_scale_frames <= 0:
            raise ValueError("num_scale_frames must be positive")
        if keyframe_interval <= 0:
            raise ValueError("keyframe_interval must be positive")

        self._backend = backend
        self._frame_shape = tuple(int(part) for part in frame_shape)
        self._num_scale_frames = int(num_scale_frames)
        self._keyframe_interval = int(keyframe_interval)
        self._scale_buffer: list[Any] = []
        self._next_frame_index = 0
        self._started = False
        self._closed = False
        self._failed = False

    @property
    def buffered_frame_count(self) -> int:
        """Number of canonical input frames retained by the Adapter."""

        return len(self._scale_buffer)

    @property
    def next_frame_index(self) -> int:
        return self._next_frame_index

    def submit(self, frame: CanonicalFrame) -> tuple[FramePrediction, ...]:
        """Submit one canonical frame in source order.

        The first ``num_scale_frames - 1`` calls return an empty tuple.  The
        final scale-frame call emits that bounded scale batch; later calls
        emit one keyframe or non-keyframe prediction each.
        """

        self._require_open()
        if not isinstance(frame.frame_index, int) or frame.frame_index < 0:
            raise FrameOrderError("frame_index must be a non-negative integer")
        if frame.frame_index != self._next_frame_index:
            raise FrameOrderError(
                f"expected source frame {self._next_frame_index}, got {frame.frame_index}"
            )
        actual_shape = _shape_of(frame.pixels, label="canonical frame")
        if actual_shape != self._frame_shape:
            raise FrameOrderError(
                f"frame {frame.frame_index} shape {actual_shape} does not match "
                f"canonical shape {self._frame_shape}"
            )

        self._next_frame_index += 1
        if not self._started:
            self._scale_buffer.append(frame.pixels)
            if len(self._scale_buffer) < self._num_scale_frames:
                return ()

            try:
                batch = self._backend.begin(tuple(self._scale_buffer))
                predictions = self._consume_batch(
                    batch,
                    first_frame_index=0,
                    frame_types=(FrameType.SCALE,) * self._num_scale_frames,
                )
            except Exception:
                self._fail()
                raise
            finally:
                self._scale_buffer.clear()
            self._started = True
            return predictions

        relative_index = frame.frame_index - self._num_scale_frames
        is_keyframe = relative_index % self._keyframe_interval == 0
        frame_type = FrameType.KEYFRAME if is_keyframe else FrameType.NON_KEYFRAME
        try:
            batch = self._backend.predict(frame.pixels, persist_keyframe=is_keyframe)
            return self._consume_batch(
                batch,
                first_frame_index=frame.frame_index,
                frame_types=(frame_type,),
            )
        except Exception:
            self._fail()
            raise

    def finish(self) -> None:
        """Finish the sequence and release model state.

        A sequence shorter than its required scale batch is invalid rather
        than being padded or duplicated.
        """

        self._require_open()
        if not self._started:
            count = len(self._scale_buffer)
            self.close()
            raise AdapterStateError(
                f"sequence ended with {count} frames; {self._num_scale_frames} scale frames required"
            )
        self.close()

    def close(self) -> None:
        """Idempotently release all Adapter and backend state."""

        if self._closed:
            return
        self._scale_buffer.clear()
        self._backend.close()
        self._closed = True

    def __enter__(self) -> ReconstructionModelAdapter:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _consume_batch(
        self,
        batch: PredictionBatch,
        *,
        first_frame_index: int,
        frame_types: tuple[FrameType, ...],
    ) -> tuple[FramePrediction, ...]:
        expected_count = len(frame_types)
        try:
            camera_shape = tuple(int(part) for part in batch.camera_shape)
            depth_shape = tuple(int(part) for part in batch.depth_shape)
            confidence_shape = tuple(int(part) for part in batch.confidence_shape)

            if camera_shape != (expected_count, *CAMERA_POSE_ENCODING_SHAPE):
                raise InvalidPredictionError(
                    f"camera output shape {camera_shape} must be "
                    f"({expected_count}, {CAMERA_POSE_ENCODING_SHAPE[0]})"
                )
            if len(depth_shape) != 3 or depth_shape[0] != expected_count:
                raise InvalidPredictionError(
                    f"depth output shape {depth_shape} must be (frames, height, width)"
                )
            if confidence_shape != depth_shape:
                raise InvalidPredictionError(
                    f"confidence shape {confidence_shape} must match depth shape {depth_shape}"
                )
            if not batch.all_finite():
                raise InvalidPredictionError("camera, depth, and confidence must all be finite")

            predictions = []
            expected_depth_shape = depth_shape[1:]
            for offset, frame_type in enumerate(frame_types):
                camera, depth, confidence = batch.frame(offset)
                actual_camera_shape = _shape_of(camera, label="camera prediction")
                actual_depth_shape = _shape_of(depth, label="depth prediction")
                actual_confidence_shape = _shape_of(
                    confidence, label="confidence prediction"
                )
                if actual_camera_shape != CAMERA_POSE_ENCODING_SHAPE:
                    raise InvalidPredictionError(
                        f"camera frame shape {actual_camera_shape} must be "
                        f"{CAMERA_POSE_ENCODING_SHAPE}"
                    )
                if actual_depth_shape != expected_depth_shape:
                    raise InvalidPredictionError(
                        f"depth frame shape {actual_depth_shape} must be {expected_depth_shape}"
                    )
                if actual_confidence_shape != expected_depth_shape:
                    raise InvalidPredictionError(
                        f"confidence frame shape {actual_confidence_shape} must be "
                        f"{expected_depth_shape}"
                    )
                predictions.append(
                    FramePrediction(
                        frame_index=first_frame_index + offset,
                        frame_type=frame_type,
                        camera_pose_encoding=camera,
                        depth=depth,
                        confidence=confidence,
                        camera_shape=actual_camera_shape,
                        depth_shape=actual_depth_shape,
                        confidence_shape=actual_confidence_shape,
                    )
                )
            return tuple(predictions)
        finally:
            batch.release()

    def _fail(self) -> None:
        self._failed = True
        self.close()

    def _require_open(self) -> None:
        if self._failed:
            raise AdapterStateError("Adapter failed and cannot accept more frames")
        if self._closed:
            raise AdapterStateError("Adapter is closed")


class _TorchPredictionBatch:
    """Normalize one GCT forward result to the PredictionBatch contract."""

    def __init__(self, output: Mapping[str, Any], torch_module: Any) -> None:
        missing = {"pose_enc", "depth", "depth_conf"}.difference(output)
        if missing:
            raise InvalidPredictionError(
                f"model output is missing required keys: {', '.join(sorted(missing))}"
            )
        self._output: Mapping[str, Any] | None = output
        self._camera = output["pose_enc"]
        self._depth = output["depth"]
        self._confidence = output["depth_conf"]
        self._torch = torch_module

    @property
    def camera_shape(self) -> tuple[int, ...]:
        shape = _shape_of(self._camera, label="camera batch")
        return shape[1:] if len(shape) == 3 and shape[0] == 1 else shape

    @property
    def depth_shape(self) -> tuple[int, ...]:
        shape = _shape_of(self._depth, label="depth batch")
        if len(shape) == 5 and shape[0] == 1 and shape[-1] == 1:
            return shape[1:-1]
        return shape

    @property
    def confidence_shape(self) -> tuple[int, ...]:
        shape = _shape_of(self._confidence, label="confidence batch")
        return shape[1:] if len(shape) == 4 and shape[0] == 1 else shape

    def all_finite(self) -> bool:
        return all(
            bool(self._torch.isfinite(value).all().item())
            for value in (self._camera, self._depth, self._confidence)
        )

    def frame(self, index: int) -> tuple[Any, Any, Any]:
        return (
            self._camera[0, index].detach(),
            self._depth[0, index, ..., 0].detach(),
            self._confidence[0, index].detach(),
        )

    def release(self) -> None:
        self._output = None
        self._camera = None
        self._depth = None
        self._confidence = None


class GCTStreamBackend:
    """Eager PyTorch SDPA backend for native Windows production inference.

    The caller supplies an already constructed, weight-loaded model.  This
    class performs no checkpoint IO or acquisition and intentionally exposes
    neither FlashInfer nor ``torch.compile``.
    """

    def __init__(
        self,
        model: Any,
        *,
        num_scale_frames: int = 8,
        device: Any | None = None,
        autocast_dtype: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if torch_module is None:
            import torch as torch_module  # Local import keeps model doubles lightweight.

        modules = model.modules() if callable(getattr(model, "modules", None)) else (model,)
        if any(getattr(module, "_orig_mod", None) is not None for module in modules):
            raise ValueError("compiled models are not permitted by the native v1 backend")
        if getattr(model, "use_sdpa", None) is not True:
            raise ValueError("native v1 requires the PyTorch SDPA attention backend")
        aggregator = getattr(model, "aggregator", None)
        if getattr(aggregator, "use_flashinfer", False):
            raise ValueError("FlashInfer is not permitted by the native v1 backend")
        if getattr(model, "enable_camera", None) is not True:
            raise ValueError("native v1 requires the camera head")
        if getattr(model, "enable_depth", None) is not True:
            raise ValueError("native v1 requires the depth head")
        if (
            getattr(model, "enable_point", None) is not False
            or getattr(model, "point_head", None) is not None
        ):
            raise ValueError("the optional dense point head must be disabled")
        if (
            getattr(model, "enable_local_point", None) is not False
            or getattr(model, "local_point_head", None) is not None
        ):
            raise ValueError("the optional local point head must be disabled")
        if num_scale_frames <= 0:
            raise ValueError("num_scale_frames must be positive")

        self._model = model.eval()
        self._torch = torch_module
        self._num_scale_frames = int(num_scale_frames)
        self._device = device
        self._autocast_dtype = autocast_dtype
        self._closed = False

    def begin(self, scale_frames: Sequence[Any]) -> PredictionBatch:
        self._require_open()
        if len(scale_frames) != self._num_scale_frames:
            raise ValueError(
                f"expected {self._num_scale_frames} scale frames, got {len(scale_frames)}"
            )
        self._model.clean_kv_cache()
        images = self._torch.stack(tuple(scale_frames), dim=0).unsqueeze(0)
        images = self._move_input(images)
        output = self._forward(
            images,
            num_frame_for_scale=self._num_scale_frames,
            num_frame_per_block=self._num_scale_frames,
        )
        return _TorchPredictionBatch(output, self._torch)

    def predict(self, frame: Any, *, persist_keyframe: bool) -> PredictionBatch:
        self._require_open()
        images = frame.unsqueeze(0).unsqueeze(0)
        images = self._move_input(images)
        if not persist_keyframe:
            self._model._set_skip_append(True)
        try:
            output = self._forward(
                images,
                num_frame_for_scale=self._num_scale_frames,
                num_frame_per_block=1,
            )
        finally:
            if not persist_keyframe:
                self._model._set_skip_append(False)
        return _TorchPredictionBatch(output, self._torch)

    def close(self) -> None:
        if self._closed:
            return
        self._model.clean_kv_cache()
        self._closed = True

    def _forward(self, images: Any, **kwargs: Any) -> Mapping[str, Any]:
        with self._torch.inference_mode(), self._autocast_context():
            return self._model(
                images,
                causal_inference=True,
                **kwargs,
            )

    def _autocast_context(self) -> ContextManager[Any]:
        if self._autocast_dtype is None:
            return nullcontext()
        return self._torch.amp.autocast("cuda", dtype=self._autocast_dtype)

    def _move_input(self, images: Any) -> Any:
        if self._device is None:
            return images
        return images.to(self._device, non_blocking=True)

    def _require_open(self) -> None:
        if self._closed:
            raise AdapterStateError("model backend is closed")


def build_native_windows_gct_model(**architecture: Any) -> Any:
    """Construct the guarded native-v1 GCT model without loading weights.

    Model acquisition and safe weight loading belong to Setup and the Worker,
    not this model seam.  Guarded arguments cannot be overridden by callers.
    """

    forbidden = {
        "use_sdpa",
        "enable_camera",
        "enable_depth",
        "enable_point",
        "enable_local_point",
        "enable_track",
    }.intersection(architecture)
    if forbidden:
        raise ValueError(f"guarded model options cannot be overridden: {sorted(forbidden)}")

    from lingbot_map.models.gct_stream import GCTStream

    return GCTStream(
        use_sdpa=True,
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_local_point=False,
        enable_track=False,
        **architecture,
    )
