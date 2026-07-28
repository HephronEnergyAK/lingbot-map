"""Exact, schema-shaped Reconstruction Result provenance builders."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from .gpu_profiles import InferencePlan


def _settings_sha256(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(profile), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def result_provenance(
    *,
    runtime_id: str,
    worker_version: str,
    job_spec_sha256: str,
    model_id: str,
    model_sha256: str,
    source_sha256: str,
    profile_name: str,
    camera_iterations: int,
    confidence_cutoff_percent: float,
    depth_cutoff_percent: float,
    import_point_budget: int,
    plan: InferencePlan | None,
    gpu: Mapping[str, Any] | None,
    suspension_count: int,
    suspension_seconds: float,
    decoder_threads: int = 1,
    preprocessing_threads: int = 1,
    output_threads: int = 1,
    onnx_threads: int = 1,
    torch_intraop_threads: int = 1,
    torch_interop_threads: int = 1,
    fixture: bool = False,
    retain_dense_predictions: bool = False,
) -> dict[str, Any]:
    profile = {
        "name": profile_name,
        "camera_iterations": int(camera_iterations),
        "confidence_cutoff_percent": float(confidence_cutoff_percent),
        "depth_cutoff_percent": float(depth_cutoff_percent),
        "import_point_budget": int(import_point_budget),
        "retain_dense_predictions": bool(retain_dense_predictions),
    }
    logical_processors = max(1, int(os.cpu_count() or 1))
    global_budget = max(1, min(8, logical_processors - 2))
    return {
        "runtime_id": runtime_id,
        "worker_version": worker_version,
        "job_spec_sha256": job_spec_sha256,
        "model_sha256": model_sha256,
        "source_sha256": source_sha256,
        "preprocessing_rule_version": "1.0.0",
        "filtering_rule_version": "1.0.0",
        "point_reducer_rule_version": "1.0.0",
        "resource_estimate_version": "1.0.0",
        "models": [
            {
                "id": model_id,
                "role": "fixture" if fixture else "reconstruction",
                "sha256": model_sha256,
            }
        ],
        "gpu": dict(gpu) if gpu is not None else None,
        "profile": {**profile, "settings_sha256": _settings_sha256(profile)},
        "preprocessing": {
            "image_size": 518,
            "patch_size": 14,
            "mode": "canonical-crop-v1",
            "resize": "pillow-bicubic",
            "color": "uint8-srgb",
            "normalization": "float32-rgb-divide-255",
        },
        "inference": {
            "mode": "fixture" if plan is None else plan.mode,
            "keyframe_interval": 0 if plan is None else plan.keyframe_interval,
            "scale_frames": 0 if plan is None else plan.scale_frames,
            "window_frames": 0 if plan is None else plan.window_frames,
            "overlap_keyframes": 0 if plan is None else plan.overlap_keyframes,
            "prediction_heads": ["fixture"] if fixture else ["camera", "depth"],
        },
        "suspension": {
            "count": int(suspension_count),
            "total_seconds": float(suspension_seconds),
        },
        "system": {
            "logical_processors": logical_processors,
            "global_thread_budget": global_budget,
            "decoder_threads": int(decoder_threads),
            "preprocessing_threads": int(preprocessing_threads),
            "output_threads": int(output_threads),
            "onnx_threads": int(onnx_threads),
            "torch_intraop_threads": int(torch_intraop_threads),
            "torch_interop_threads": int(torch_interop_threads),
            "priority": "below-normal",
        },
    }
