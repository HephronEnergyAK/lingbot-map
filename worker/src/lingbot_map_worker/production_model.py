"""Production-only model loading and prediction decoding for short jobs."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from lingbot_map.model_adapter import (
    CanonicalFrame,
    GCTStreamBackend,
    ReconstructionModelAdapter,
    build_native_windows_gct_model,
)
from .canonical_preprocessing import CanonicalImage
from .gpu_profiles import InferencePlan, ReconstructionProfile
from .result_pipeline import AlignedPrediction
from .short_pipeline import PipelineCancelled
from .torch_capability import (
    ALLOWED_UNEXPECTED_PREFIXES,
    EXPECTED_CUDA_VERSION,
    EXPECTED_TORCH_VERSION,
)


class ProductionModelError(RuntimeError):
    pass


def sha256_model(path: Path, cancel: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            if cancel():
                raise ProductionModelError("cancelled while revalidating Reconstruction Model")
            digest.update(chunk)
    return digest.hexdigest()


def _load_production_model(
    *,
    model_path: Path,
    expected_sha256: str,
    plan: InferencePlan,
    profile: ReconstructionProfile,
    cancel: Callable[[], bool],
    torch_module: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Load the catalog-pinned camera/depth model with no executable pickle."""

    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if str(torch.__version__) != EXPECTED_TORCH_VERSION or str(torch.version.cuda) != EXPECTED_CUDA_VERSION:
        raise ProductionModelError(
            f"Pinned PyTorch stack mismatch: torch={torch.__version__}, CUDA={torch.version.cuda}"
        )
    path = Path(model_path).resolve()
    if not path.is_file() or sha256_model(path, cancel) != expected_sha256:
        raise ProductionModelError("Reconstruction Model checksum is absent or changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProductionModelError("Job Worker must see exactly its selected physical GPU")
    if cancel():
        raise ProductionModelError("cancelled before model loading")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items()
    ):
        raise ProductionModelError("weights-only checkpoint is not a string-to-tensor state_dict")
    model = build_native_windows_gct_model(
        img_size=profile.image_size,
        patch_size=profile.patch_size,
        max_frame_num=profile.window_frames,
        kv_cache_sliding_window=profile.window_frames,
        kv_cache_scale_frames=profile.scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        camera_num_iterations=profile.camera_iterations,
        use_gradient_checkpoint=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise ProductionModelError(f"Reconstruction Model is missing {len(missing)} required keys")
    disallowed = [
        key for key in unexpected if not key.startswith(ALLOWED_UNEXPECTED_PREFIXES)
    ]
    if disallowed:
        raise ProductionModelError(
            f"Reconstruction Model has {len(disallowed)} unexpected state_dict keys"
        )
    del checkpoint, state_dict
    gc.collect()
    model = model.to(device).eval()
    capability = torch.cuda.get_device_capability(device)
    autocast_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    return model, device, autocast_dtype, torch


def load_production_adapter(
    *,
    model_path: Path,
    expected_sha256: str,
    plan: InferencePlan,
    profile: ReconstructionProfile,
    frame_shape: tuple[int, int, int],
    cancel: Callable[[], bool],
    torch_module: Any | None = None,
) -> ReconstructionModelAdapter:
    """Load one streaming Adapter for a source of at most 3000 frames."""

    model, device, autocast_dtype, torch = _load_production_model(
        model_path=model_path,
        expected_sha256=expected_sha256,
        plan=plan,
        profile=profile,
        cancel=cancel,
        torch_module=torch_module,
    )
    backend = GCTStreamBackend(
        model,
        num_scale_frames=plan.scale_frames,
        device=device,
        autocast_dtype=autocast_dtype,
        torch_module=torch,
    )
    return ReconstructionModelAdapter(
        backend,
        frame_shape=frame_shape,
        num_scale_frames=plan.scale_frames,
        keyframe_interval=plan.keyframe_interval,
    )


class ProductionWindowPredictor:
    """Reuse one loaded model while discarding KV state after every window."""

    def __init__(
        self,
        model: Any,
        *,
        plan: InferencePlan,
        device: Any,
        autocast_dtype: Any,
        torch_module: Any,
    ) -> None:
        self.model = model
        self.plan = plan
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.torch = torch_module
        self.decoder = TorchPredictionDecoder()
        self._closed = False

    def predict(
        self,
        frames: Sequence[Any],
        *,
        cancel: Callable[[], bool],
    ) -> Sequence[AlignedPrediction]:
        if self._closed:
            raise ProductionModelError("window predictor is closed")
        if not self.plan.scale_frames <= len(frames) <= self.plan.window_frames:
            raise ProductionModelError("window frame count is outside the fixed model contract")
        backend = GCTStreamBackend(
            self.model,
            num_scale_frames=self.plan.scale_frames,
            device=self.device,
            autocast_dtype=self.autocast_dtype,
            torch_module=self.torch,
        )
        adapter = ReconstructionModelAdapter(
            backend,
            frame_shape=tuple(frames[0].canonical.model_input.shape),
            num_scale_frames=self.plan.scale_frames,
            keyframe_interval=1,
        )
        decoded: list[AlignedPrediction] = []
        try:
            for local_index, frame in enumerate(frames):
                if cancel():
                    raise PipelineCancelled("cancelled during window inference")
                predictions = adapter.submit(
                    CanonicalFrame(local_index, frame.canonical.model_input)
                )
                for prediction in predictions:
                    source = frames[prediction.frame_index]
                    aligned = self.decoder(
                        prediction, source.canonical, source.pts_seconds
                    )
                    decoded.append(
                        replace(
                            aligned,
                            frame_index=source.frame_index,
                            source_pts_seconds=source.pts_seconds,
                        )
                    )
            adapter.finish()
        except Exception:
            adapter.close()
            raise
        if len(decoded) != len(frames):
            raise ProductionModelError("window model omitted a presentation frame")
        return tuple(decoded)

    def close(self) -> None:
        if self._closed:
            return
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self._closed = True


def load_production_window_predictor(
    *,
    model_path: Path,
    expected_sha256: str,
    plan: InferencePlan,
    profile: ReconstructionProfile,
    cancel: Callable[[], bool],
    torch_module: Any | None = None,
) -> ProductionWindowPredictor:
    if plan.mode != "windowed" or (
        plan.window_frames,
        plan.overlap_keyframes,
        plan.scale_frames,
        plan.keyframe_interval,
    ) != (64, 16, 8, 1):
        raise ProductionModelError("window predictor requires the fixed v1 plan")
    model, device, autocast_dtype, torch = _load_production_model(
        model_path=model_path,
        expected_sha256=expected_sha256,
        plan=plan,
        profile=profile,
        cancel=cancel,
        torch_module=torch_module,
    )
    return ProductionWindowPredictor(
        model,
        plan=plan,
        device=device,
        autocast_dtype=autocast_dtype,
        torch_module=torch,
    )


class TorchPredictionDecoder:
    """Move one finalized camera/depth output to bounded CPU result arrays."""

    def __call__(
        self,
        prediction: Any,
        canonical: CanonicalImage,
        pts_seconds: float,
    ) -> AlignedPrediction:
        pose = prediction.camera_pose_encoding.reshape(1, 1, 9)
        height, width = prediction.depth_shape
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose, image_size_hw=(height, width)
        )
        world_to_camera = extrinsics[0, 0].detach().cpu().numpy()
        homogeneous = np.eye(4, dtype="<f8")
        homogeneous[:3, :] = np.asarray(world_to_camera, dtype="<f8")
        model_intrinsics = np.asarray(
            intrinsics[0, 0].detach().cpu().numpy(), dtype="<f8"
        )
        depth = np.ascontiguousarray(prediction.depth.detach().float().cpu().numpy(), dtype="<f4")
        confidence = np.ascontiguousarray(
            prediction.confidence.detach().float().cpu().numpy(), dtype="<f4"
        )
        return AlignedPrediction(
            prediction.frame_index,
            int(prediction.frame_type),
            pts_seconds,
            homogeneous,
            model_intrinsics,
            depth,
            confidence,
            canonical.color_rgb,
        )
