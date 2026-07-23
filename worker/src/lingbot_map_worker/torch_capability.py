"""Real eager-SDPA PyTorch workload for native Windows GPU qualification."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import re
from typing import Callable, Mapping

from . import __version__
from .capability import (
    CapabilityCancelled,
    CapabilityError,
    CapabilityStack,
    Measurement,
)
from .gpu_profiles import ReconstructionProfile


EXPECTED_TORCH_VERSION = "2.11.0+cu130"
EXPECTED_CUDA_VERSION = "13.0"
ALLOWED_UNEXPECTED_PREFIXES = ("point_head.", "local_point_head.", "track_head.")


def _sha256_file(path: Path, cancelled: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            if cancelled():
                raise CapabilityCancelled("GPU capability test was cancelled while hashing the model")
            digest.update(chunk)
    return digest.hexdigest()


class TorchCapabilityWorkload:
    def __init__(
        self,
        *,
        model_path: Path,
        model_sha256: str,
        runtime_id: str,
        worker_lock_sha256: str,
        torch_module=None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.model_sha256 = model_sha256
        self.runtime_id = runtime_id
        self.worker_lock_sha256 = worker_lock_sha256
        self._torch = torch_module

    def _torch_module(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def stack(self) -> CapabilityStack:
        torch = self._torch_module()
        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)
        if torch_version != EXPECTED_TORCH_VERSION or cuda_version != EXPECTED_CUDA_VERSION:
            raise CapabilityError(
                f"Pinned PyTorch stack mismatch: torch={torch_version}, CUDA={cuda_version}"
            )
        return CapabilityStack(
            self.runtime_id,
            self.worker_lock_sha256,
            __version__,
            torch_version,
            cuda_version,
        )

    def measure(
        self,
        profile: ReconstructionProfile,
        *,
        cancelled: Callable[[], bool],
        progress: Callable[[str, int, int], None],
    ) -> Measurement:
        if _sha256_file(self.model_path, cancelled) != self.model_sha256:
            raise CapabilityError("Reconstruction Model checksum changed before capability testing")
        torch = self._torch_module()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise CapabilityError(
                "Capability child must see exactly its selected physical GPU UUID"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = None
        backend = None
        checkpoint = None
        state_dict = None
        try:
            if cancelled():
                raise CapabilityCancelled("GPU capability test was cancelled before model load")
            progress(f"{profile.name}:model-load", 0, 1)
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
            if not isinstance(state_dict, Mapping) or not all(
                isinstance(key, str) and torch.is_tensor(value)
                for key, value in state_dict.items()
            ):
                raise CapabilityError("weights-only checkpoint is not a string-to-tensor state_dict")

            from lingbot_map.model_adapter import (
                GCTStreamBackend,
                build_native_windows_gct_model,
            )

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
                raise CapabilityError(
                    f"Reconstruction Model is missing {len(missing)} required state_dict keys"
                )
            disallowed = [
                key for key in unexpected
                if not key.startswith(ALLOWED_UNEXPECTED_PREFIXES)
            ]
            if disallowed:
                raise CapabilityError(
                    f"Reconstruction Model has {len(disallowed)} unexpected state_dict keys"
                )
            del checkpoint, state_dict
            checkpoint = state_dict = None
            gc.collect()
            model = model.to(device).eval()
            capability = torch.cuda.get_device_capability(device)
            autocast_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            backend = GCTStreamBackend(
                model,
                num_scale_frames=profile.scale_frames,
                device=device,
                autocast_dtype=autocast_dtype,
                torch_module=torch,
            )
            progress(f"{profile.name}:model-load", 1, 1)

            frame = torch.zeros(
                (3, profile.image_size, profile.image_size), dtype=torch.float32
            )
            scale = tuple(frame for _ in range(profile.scale_frames))
            batch = backend.begin(scale)
            batch.release()
            torch.cuda.synchronize(device)
            progress(profile.name, profile.scale_frames, profile.window_frames)
            for index in range(profile.scale_frames, profile.window_frames):
                if cancelled():
                    raise CapabilityCancelled(
                        f"GPU capability test was cancelled during {profile.name}"
                    )
                batch = backend.predict(frame, persist_keyframe=True)
                batch.release()
                torch.cuda.synchronize(device)
                progress(profile.name, index + 1, profile.window_frames)
            peak = int(torch.cuda.max_memory_reserved(device))
            if peak <= 0:
                raise CapabilityError("PyTorch reported no peak reserved CUDA memory")
            return Measurement(peak)
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            del backend, model, checkpoint, state_dict
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def classify_gpu_failure(self, exception: BaseException) -> str | None:
        torch = self._torch_module()
        if isinstance(exception, getattr(torch.cuda, "OutOfMemoryError", ())):
            return "cuda-out-of-memory"
        message = str(exception).lower()
        patterns = (
            ("cuda-out-of-memory", r"cuda.*out of memory|cudaerror_memoryallocation"),
            ("cuda-device-loss", r"device lost|cudaerror_devicelost|device-side assert"),
            ("cuda-driver-reset", r"driver.*(reset|shutting down)|cudaerror_deviceuninitialized"),
            ("windows-tdr", r"launch timeout|cudaerror_launchtimeout|timed out and was terminated"),
        )
        for classification, pattern in patterns:
            if re.search(pattern, message):
                return classification
        return None
