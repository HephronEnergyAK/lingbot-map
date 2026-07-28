"""Validate raw neural outputs using invariants and fixture-specific ranges."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_release_corpus import (  # noqa: E402
    CorpusValidationError,
    load_json,
)


ARRAY_CONTRACTS = {
    "world_to_camera_opencv.npy": ("<f8", 3),
    "model_intrinsics.npy": ("<f8", 3),
    "depth.npy": ("<f4", 3),
    "confidence.npy": ("<f4", 3),
    "source_pts_seconds.npy": ("<f8", 1),
    "frame_type.npy": ("|u1", 1),
}
PROVENANCE_FIELDS = {
    "schema_version",
    "fixture_id",
    "source_sha256",
    "runtime_id",
    "model_id",
    "model_sha256",
    "profile",
    "frame_count",
}


def _load_array(root: Path, name: str, dtype: str, ndim: int) -> np.ndarray:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise CorpusValidationError(f"neural output is missing plain file: {name}")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CorpusValidationError(f"neural output array is invalid: {name}") from exc
    if (
        not isinstance(array, np.ndarray)
        or array.dtype.str != dtype
        or array.ndim != ndim
        or not array.flags.c_contiguous
    ):
        raise CorpusValidationError(
            f"neural output dtype/order/rank is invalid: {name}"
        )
    if not bool(np.isfinite(array).all()):
        raise CorpusValidationError(f"neural output contains non-finite values: {name}")
    return array


def _percentile(array: np.ndarray, percentile: float) -> float:
    return float(np.percentile(array.astype(np.float64, copy=False), percentile))


def _rotation_step_p95_degrees(world_to_camera: np.ndarray) -> float:
    rotations = world_to_camera[:, :3, :3].astype(np.float64, copy=False)
    if len(rotations) < 2:
        return 0.0
    relative = np.einsum(
        "fji,fjk->fik",
        rotations[:-1],
        rotations[1:],
    )
    cosines = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.percentile(np.degrees(np.arccos(cosines)), 95))


def _metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    depth = arrays["depth.npy"]
    confidence = arrays["confidence.npy"]
    cameras = np.linalg.inv(arrays["world_to_camera_opencv.npy"])
    centers = cameras[:, :3, 3]
    extent = (
        float(np.max(np.linalg.norm(centers - centers[0], axis=1)))
        if len(centers)
        else 0.0
    )
    return {
        "depth_p05": _percentile(depth, 5),
        "depth_median": _percentile(depth, 50),
        "depth_p95": _percentile(depth, 95),
        "confidence_p05": _percentile(confidence, 5),
        "confidence_median": _percentile(confidence, 50),
        "confidence_p95": _percentile(confidence, 95),
        "camera_path_extent": extent,
        "rotation_step_p95_degrees": _rotation_step_p95_degrees(
            arrays["world_to_camera_opencv.npy"]
        ),
    }


def validate_neural_output(
    output: Path,
    fixture_id: str,
    *,
    oracle_path: Path | None = None,
) -> dict[str, Any]:
    oracle = load_json(
        oracle_path
        if oracle_path is not None
        else ROOT / "release_corpus" / "neural-oracles.json"
    )
    if oracle.get("cross_gpu_checksum_allowed") is not False:
        raise CorpusValidationError("cross-GPU neural checksum is not forbidden")
    if oracle.get("invariant_tolerances") != {
        "rigid_orthonormal_atol": 1e-6
    }:
        raise CorpusValidationError("neural invariant tolerances drifted")
    fixtures = oracle.get("fixtures")
    if not isinstance(fixtures, dict) or fixture_id not in fixtures:
        raise CorpusValidationError(f"unknown neural fixture: {fixture_id}")
    fixture = fixtures[fixture_id]
    if fixture.get("status") != "ready" or not isinstance(
        fixture.get("ranges"), dict
    ):
        raise CorpusValidationError(
            f"neural fixture is not calibrated: {fixture_id}: "
            f"{fixture.get('status')}"
        )

    output = output.resolve()
    if not output.is_dir() or output.is_symlink():
        raise CorpusValidationError("neural output root must be a plain directory")
    expected_files = set(ARRAY_CONTRACTS) | {"provenance.json"}
    actual_files = {
        path.name
        for path in output.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != expected_files:
        raise CorpusValidationError(
            "neural output file set differs from the exact contract"
        )
    arrays = {
        name: _load_array(output, name, dtype, ndim)
        for name, (dtype, ndim) in ARRAY_CONTRACTS.items()
    }
    cameras = arrays["world_to_camera_opencv.npy"]
    intrinsics = arrays["model_intrinsics.npy"]
    depth = arrays["depth.npy"]
    confidence = arrays["confidence.npy"]
    timestamps = arrays["source_pts_seconds.npy"]
    frame_type = arrays["frame_type.npy"]
    frame_count = len(timestamps)
    if (
        cameras.shape != (frame_count, 4, 4)
        or intrinsics.shape != (frame_count, 3, 3)
        or depth.shape[:1] != (frame_count,)
        or confidence.shape != depth.shape
        or frame_type.shape != (frame_count,)
        or depth.shape[1] < 1
        or depth.shape[2] < 1
    ):
        raise CorpusValidationError("neural arrays are structurally misaligned")
    if fixture.get("expected_frame_count") != frame_count:
        raise CorpusValidationError("neural frame count disagrees with fixture")
    if frame_count < 1 or (
        frame_count > 1 and not bool((np.diff(timestamps) > 0).all())
    ):
        raise CorpusValidationError(
            "neural timestamps are not finite and strictly increasing"
        )
    if not bool(np.isin(frame_type, (0, 1, 2)).all()):
        raise CorpusValidationError("neural frame type is outside 0/1/2")
    homogeneous = np.broadcast_to((0.0, 0.0, 0.0, 1.0), (frame_count, 4))
    rotations = cameras[:, :3, :3]
    identities = np.einsum("fji,fjk->fik", rotations, rotations)
    if (
        not np.allclose(cameras[:, 3, :], homogeneous, atol=1e-9, rtol=0)
        or not np.allclose(identities, np.eye(3), atol=1e-6, rtol=0)
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6, rtol=0)
    ):
        raise CorpusValidationError(
            "neural world-to-camera matrices are not right-handed rigid transforms"
        )
    expected_intrinsic_row = np.broadcast_to(
        (0.0, 0.0, 1.0), (frame_count, 3)
    )
    if (
        not bool((intrinsics[:, 0, 0] > 0).all())
        or not bool((intrinsics[:, 1, 1] > 0).all())
        or not np.allclose(
            intrinsics[:, 2, :], expected_intrinsic_row, atol=1e-9, rtol=0
        )
        or not np.allclose(intrinsics[:, 0, 1], 0.0, atol=1e-9, rtol=0)
        or not np.allclose(intrinsics[:, 1, 0], 0.0, atol=1e-9, rtol=0)
    ):
        raise CorpusValidationError("neural intrinsics are malformed")
    if not bool((depth > 0).all()):
        raise CorpusValidationError("neural depth must be finite and positive")

    provenance = load_json(output / "provenance.json")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_FIELDS:
        raise CorpusValidationError("neural provenance fields differ")
    if (
        provenance["schema_version"] != "1.0.0"
        or provenance["fixture_id"] != fixture_id
        or provenance["source_sha256"] != fixture["source_sha256"]
        or provenance["frame_count"] != frame_count
        or not isinstance(provenance["profile"], str)
        or not provenance["profile"]
    ):
        raise CorpusValidationError("neural provenance identity disagrees")
    for name in ("runtime_id", "model_sha256"):
        value = provenance[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise CorpusValidationError(f"neural provenance checksum is invalid: {name}")
    if not isinstance(provenance["model_id"], str) or not provenance["model_id"]:
        raise CorpusValidationError("neural provenance model id is invalid")

    metrics = _metrics(arrays)
    ranges = fixture["ranges"]
    if set(ranges) != set(oracle["range_metrics"]):
        raise CorpusValidationError("neural fixture range set is incomplete")
    for name, value in metrics.items():
        bounds = ranges[name]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
                for bound in bounds
            )
            or float(bounds[0]) > float(bounds[1])
        ):
            raise CorpusValidationError(f"neural range is malformed: {name}")
        if not float(bounds[0]) <= value <= float(bounds[1]):
            raise CorpusValidationError(
                f"neural quality metric is outside calibrated range: "
                f"{name}={value!r}, range={bounds!r}"
            )
    return {
        "schema_version": "1.0.0",
        "fixture_id": fixture_id,
        "frame_count": frame_count,
        "metrics": metrics,
        "cross_gpu_checksum_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("fixture_id")
    parser.add_argument("--oracle", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = validate_neural_output(
            arguments.output,
            arguments.fixture_id,
            oracle_path=arguments.oracle,
        )
    except CorpusValidationError as exc:
        print(f"LINGBOT_MAP_NEURAL_ORACLE_ERROR={exc}", file=sys.stderr)
        return 2
    print(
        "LINGBOT_MAP_NEURAL_ORACLE="
        + json.dumps(result, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
