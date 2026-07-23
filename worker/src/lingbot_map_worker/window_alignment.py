"""Rolling Sim(3) alignment and versioned Quality Warnings for long Jobs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .result_pipeline import AlignedPrediction


ALIGNMENT_RULE_VERSION = "1.0.0"
MIN_STRUCTURAL_SCALE = 1e-4
MAX_STRUCTURAL_SCALE = 1e4
MAX_STRUCTURAL_TRANSLATION = 1e12
QUALITY_SCALE_MIN = 0.8
QUALITY_SCALE_MAX = 1.25
QUALITY_ROTATION_P95_DEGREES = 5.0
QUALITY_CENTER_P95 = 0.05
QUALITY_LOG_DEPTH_P95 = math.log(1.25)


class WindowAlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def validate(self) -> "SimilarityTransform":
        scale = float(self.scale)
        rotation = np.asarray(self.rotation)
        translation = np.asarray(self.translation)
        if (
            not math.isfinite(scale)
            or not MIN_STRUCTURAL_SCALE <= scale <= MAX_STRUCTURAL_SCALE
        ):
            raise WindowAlignmentError("window similarity scale is non-finite or implausible")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise WindowAlignmentError("window similarity rotation is non-finite or malformed")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
            raise WindowAlignmentError("window similarity rotation is not orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not math.isclose(determinant, 1.0, abs_tol=1e-7):
            raise WindowAlignmentError("window similarity rotation is singular or reflected")
        if (
            translation.shape != (3,)
            or not np.isfinite(translation).all()
            or float(np.max(np.abs(translation))) > MAX_STRUCTURAL_TRANSLATION
        ):
            raise WindowAlignmentError("window similarity translation is non-finite or implausible")
        return SimilarityTransform(
            scale,
            np.ascontiguousarray(rotation, dtype="<f8"),
            np.ascontiguousarray(translation, dtype="<f8"),
        )


@dataclass(frozen=True)
class BoundaryMetrics:
    source_frame_start: int
    source_frame_end: int
    paired_keyframes: int
    relative_scale: float
    rotation_median_degrees: float
    rotation_p95_degrees: float
    normalized_center_distance_median: float
    normalized_center_distance_p95: float
    absolute_log_depth_ratio_median: float
    absolute_log_depth_ratio_p95: float
    camera_pair_count: int
    depth_pixel_count: int
    triggered_conditions: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        return {
            "source_frame_start": self.source_frame_start,
            "source_frame_end": self.source_frame_end,
            "paired_keyframes": self.paired_keyframes,
            "relative_scale": self.relative_scale,
            "rotation_difference_degrees": {
                "median": self.rotation_median_degrees,
                "p95": self.rotation_p95_degrees,
            },
            "normalized_camera_center_distance": {
                "median": self.normalized_center_distance_median,
                "p95": self.normalized_center_distance_p95,
            },
            "absolute_log_depth_ratio": {
                "median": self.absolute_log_depth_ratio_median,
                "p95": self.absolute_log_depth_ratio_p95,
            },
            "camera_pair_count": self.camera_pair_count,
            "depth_pixel_count": self.depth_pixel_count,
            "triggered_conditions": list(self.triggered_conditions),
        }

    def warning(self) -> Mapping[str, str] | None:
        if not self.triggered_conditions:
            return None
        return {
            "code": "quality_warning",
            "message": (
                f"Window boundary frames {self.source_frame_start}-"
                f"{self.source_frame_end} exceeded {', '.join(self.triggered_conditions)}"
            ),
        }


def _camera_to_world(frame: AlignedPrediction) -> np.ndarray:
    matrix = np.asarray(frame.world_to_camera_opencv)
    if matrix.dtype.str != "<f8" or matrix.shape != (4, 4) or not matrix.flags.c_contiguous:
        raise WindowAlignmentError("overlap camera is not canonical little-endian float64")
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix[3], (0, 0, 0, 1), atol=1e-9, rtol=0
    ):
        raise WindowAlignmentError("overlap camera is non-finite or malformed")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
        raise WindowAlignmentError("overlap camera rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
        raise WindowAlignmentError("overlap camera rotation is singular or reflected")
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise WindowAlignmentError("overlap camera is singular") from exc


def _percentiles(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise WindowAlignmentError("window boundary has no finite residual samples")
    median, p95 = np.percentile(values, (50, 95))
    return float(median), float(p95)


def _valid_depth_pair(
    previous: AlignedPrediction, current: AlignedPrediction
) -> tuple[np.ndarray, np.ndarray]:
    if previous.depth.shape != current.depth.shape:
        raise WindowAlignmentError("overlap depth grids are structurally inconsistent")
    valid = (
        np.isfinite(previous.depth)
        & np.isfinite(current.depth)
        & (previous.depth > 0)
        & (current.depth > 0)
    )
    if not np.any(valid):
        raise WindowAlignmentError("window boundary has no valid overlapping depth")
    return previous.depth[valid].astype(np.float64), current.depth[valid].astype(np.float64)


def apply_similarity(
    frame: AlignedPrediction, transform: SimilarityTransform
) -> AlignedPrediction:
    transform = transform.validate()
    camera = _camera_to_world(frame)
    mapped = np.eye(4, dtype="<f8")
    mapped[:3, :3] = transform.rotation @ camera[:3, :3]
    mapped[:3, 3] = (
        transform.scale * (transform.rotation @ camera[:3, 3])
        + transform.translation
    )
    try:
        world_to_camera = np.ascontiguousarray(np.linalg.inv(mapped), dtype="<f8")
    except np.linalg.LinAlgError as exc:
        raise WindowAlignmentError("mapped overlap camera is singular") from exc
    depth = np.ascontiguousarray(frame.depth * transform.scale, dtype="<f4")
    if not np.isfinite(depth).all():
        raise WindowAlignmentError("mapped overlap depth is non-finite")
    return AlignedPrediction(
        frame.frame_index,
        frame.frame_type,
        frame.source_pts_seconds,
        world_to_camera,
        frame.model_intrinsics,
        depth,
        frame.confidence,
        frame.rgb,
    )


class RollingWindowAligner:
    """Estimate one finite rolling similarity from camera and depth overlap."""

    def estimate(
        self,
        previous: Sequence[AlignedPrediction],
        current: Sequence[AlignedPrediction],
    ) -> SimilarityTransform:
        if not previous or len(previous) != len(current):
            raise WindowAlignmentError("window overlap pair count is missing or inconsistent")
        previous_cameras: list[np.ndarray] = []
        current_cameras: list[np.ndarray] = []
        depth_ratios: list[np.ndarray] = []
        for left, right in zip(previous, current):
            if left.frame_index != right.frame_index:
                raise WindowAlignmentError("window overlap frame identities disagree")
            previous_cameras.append(_camera_to_world(left))
            current_cameras.append(_camera_to_world(right))
            previous_depth, current_depth = _valid_depth_pair(left, right)
            depth_ratios.append(previous_depth / current_depth)
        ratios = np.concatenate(depth_ratios)
        if not len(ratios) or not np.isfinite(ratios).all():
            raise WindowAlignmentError("window overlap cannot establish a finite scale")
        scale = float(np.median(ratios))
        relative_rotations = np.stack(
            [
                left[:3, :3] @ right[:3, :3].T
                for left, right in zip(previous_cameras, current_cameras)
            ]
        )
        try:
            u, _singular, vt = np.linalg.svd(np.mean(relative_rotations, axis=0))
        except np.linalg.LinAlgError as exc:
            raise WindowAlignmentError("window overlap rotation fit failed") from exc
        rotation = u @ vt
        if float(np.linalg.det(rotation)) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        translations = np.stack(
            [
                left[:3, 3] - scale * (rotation @ right[:3, 3])
                for left, right in zip(previous_cameras, current_cameras)
            ]
        )
        translation = np.median(translations, axis=0)
        return SimilarityTransform(scale, rotation, translation).validate()

    def metrics(
        self,
        previous: Sequence[AlignedPrediction],
        mapped_current: Sequence[AlignedPrediction],
        transform: SimilarityTransform,
    ) -> BoundaryMetrics:
        if not previous or len(previous) != len(mapped_current):
            raise WindowAlignmentError("window boundary metrics lack paired keyframes")
        rotations: list[float] = []
        centers: list[float] = []
        depth_logs: list[np.ndarray] = []
        depth_values: list[np.ndarray] = []
        for left, right in zip(previous, mapped_current):
            if left.frame_index != right.frame_index:
                raise WindowAlignmentError("window boundary metric identities disagree")
            left_camera = _camera_to_world(left)
            right_camera = _camera_to_world(right)
            relative = left_camera[:3, :3].T @ right_camera[:3, :3]
            cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
            rotations.append(math.degrees(math.acos(cosine)))
            centers.append(float(np.linalg.norm(left_camera[:3, 3] - right_camera[:3, 3])))
            left_depth, right_depth = _valid_depth_pair(left, right)
            depth_values.extend((left_depth, right_depth))
            depth_logs.append(np.abs(np.log(left_depth / right_depth)))
        median_depth = float(np.median(np.concatenate(depth_values)))
        if not math.isfinite(median_depth) or median_depth <= 0:
            raise WindowAlignmentError("window boundary has no positive median depth")
        rotation_median, rotation_p95 = _percentiles(
            np.asarray(rotations, dtype=np.float64)
        )
        center_median, center_p95 = _percentiles(
            np.asarray(centers, dtype=np.float64) / median_depth
        )
        depth_ratios = np.concatenate(depth_logs)
        depth_median, depth_p95 = _percentiles(depth_ratios)
        conditions: list[str] = []
        if len(previous) < 2:
            conditions.append("paired_keyframes")
        if not QUALITY_SCALE_MIN <= transform.scale <= QUALITY_SCALE_MAX:
            conditions.append("relative_scale")
        if rotation_p95 > QUALITY_ROTATION_P95_DEGREES:
            conditions.append("rotation_p95")
        if center_p95 > QUALITY_CENTER_P95:
            conditions.append("center_distance_p95")
        if depth_p95 > QUALITY_LOG_DEPTH_P95:
            conditions.append("log_depth_ratio_p95")
        return BoundaryMetrics(
            previous[0].frame_index,
            previous[-1].frame_index,
            len(previous),
            float(transform.scale),
            rotation_median,
            rotation_p95,
            center_median,
            center_p95,
            depth_median,
            depth_p95,
            len(previous),
            int(len(depth_ratios)),
            tuple(conditions),
        )
