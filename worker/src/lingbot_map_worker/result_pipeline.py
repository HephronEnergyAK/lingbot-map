"""Deep seam that turns aligned model predictions into one bounded Result."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .point_reducer import PointCandidate, PointReducer
from .result_bundle import (
    PublishedResult,
    ResultCancelled,
    ResultPublication,
    publish_result_bundle,
)
from .result_resources import (
    ResourceProbe,
    estimate_core_result_bytes,
    estimate_fixture_memory_bytes,
    require_project_disk,
    require_worker_memory,
)


CancelCheck = Callable[[], bool]


class ResultPipelineError(ValueError):
    pass


@dataclass(frozen=True)
class AlignedPrediction:
    """One output frame after model-window overlap alignment."""

    frame_index: int
    frame_type: int
    source_pts_seconds: float
    world_to_camera_opencv: np.ndarray
    model_intrinsics: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    rgb: np.ndarray


@dataclass(frozen=True)
class ResultProfile:
    name: str
    confidence_cutoff_percent: float
    depth_cutoff_percent: float
    import_point_budget: int
    initial_voxel_edge_length: float


@dataclass(frozen=True)
class ResultBuildRequest:
    job_id: str
    project_root: Path
    target_scene: Mapping[str, Any]
    timeline_start: int
    source: Mapping[str, Any]
    source_to_model: np.ndarray
    predictions: Sequence[AlignedPrediction]
    profile: ResultProfile
    provenance: Mapping[str, Any]
    warnings: tuple[Mapping[str, str], ...] = ()
    created_utc: str | None = None
    result_id: str | None = None


@dataclass(frozen=True)
class FrameFilterStatistics:
    frame_index: int
    valid_depth_count: int
    confidence_retained_count: int
    depth_retained_count: int
    confidence_threshold: float | None
    depth_threshold: float | None


@dataclass(frozen=True)
class ResultBuildOutcome:
    published: PublishedResult
    filters: tuple[FrameFilterStatistics, ...]
    maximum_reducer_entries: int


_BLENDER_FROM_OPENCV_CAMERA = np.diag((1.0, -1.0, -1.0, 1.0))
_FIRST_CAMERA_TARGET = np.array(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def _strict_array(value: Any, dtype: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ResultPipelineError(f"{label} must be an ndarray without implicit conversion")
    if value.dtype.str != dtype or value.shape != shape or not value.flags.c_contiguous:
        raise ResultPipelineError(f"{label} has an invalid dtype, shape, or memory order")
    if value.dtype.hasobject or value.dtype.fields is not None:
        raise ResultPipelineError(f"{label} cannot use object or structured values")
    return value


def _validate_profile(profile: ResultProfile) -> None:
    if not profile.name or len(profile.name) > 128:
        raise ResultPipelineError("profile name is invalid")
    for value, label in (
        (profile.confidence_cutoff_percent, "confidence cutoff"),
        (profile.depth_cutoff_percent, "depth cutoff"),
    ):
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
            raise ResultPipelineError(f"{label} must be in [0,100]")


def _validate_predictions(
    predictions: Sequence[AlignedPrediction], source_to_model: np.ndarray
) -> tuple[int, int]:
    if not predictions:
        raise ResultPipelineError("at least one aligned prediction is required")
    _strict_array(source_to_model, "<f8", (3, 3), "source_to_model")
    if not np.isfinite(source_to_model).all() or abs(float(np.linalg.det(source_to_model))) <= 1e-12:
        raise ResultPipelineError("source_to_model must be finite and invertible")
    model_height = model_width = -1
    last_pts = -math.inf
    for expected_index, frame in enumerate(predictions):
        if frame.frame_index != expected_index:
            raise ResultPipelineError("aligned frame indices must be contiguous and ordered")
        if frame.frame_type not in (0, 1, 2):
            raise ResultPipelineError("frame_type is outside the stable 0/1/2 code set")
        if not math.isfinite(frame.source_pts_seconds) or frame.source_pts_seconds <= last_pts:
            raise ResultPipelineError("source PTS values must be finite and strictly increasing")
        last_pts = frame.source_pts_seconds
        if not isinstance(frame.depth, np.ndarray) or frame.depth.ndim != 2:
            raise ResultPipelineError("depth must be a two-dimensional ndarray")
        height, width = frame.depth.shape
        if height < 1 or width < 1:
            raise ResultPipelineError("prediction grids cannot be empty")
        if expected_index == 0:
            model_height, model_width = height, width
        elif (height, width) != (model_height, model_width):
            raise ResultPipelineError("all aligned predictions must share one model grid")
        _strict_array(frame.depth, "<f4", (height, width), "depth")
        _strict_array(frame.confidence, "<f4", (height, width), "confidence")
        _strict_array(frame.rgb, "|u1", (height, width, 3), "rgb")
        w2c = _strict_array(
            frame.world_to_camera_opencv, "<f8", (4, 4), "world_to_camera_opencv"
        )
        intrinsics = _strict_array(frame.model_intrinsics, "<f8", (3, 3), "model_intrinsics")
        if not np.isfinite(w2c).all() or not np.isfinite(intrinsics).all():
            raise ResultPipelineError("camera matrices must be finite")
        if not np.allclose(w2c[3], (0, 0, 0, 1), atol=1e-9, rtol=0):
            raise ResultPipelineError("world_to_camera_opencv has a malformed homogeneous row")
        rotation = w2c[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
            raise ResultPipelineError("world_to_camera_opencv rotation is not orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
            raise ResultPipelineError("world_to_camera_opencv is reflected or non-rigid")
        if (
            intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], (0, 0, 1), atol=1e-9, rtol=0)
            or not math.isclose(float(intrinsics[0, 1]), 0.0, abs_tol=1e-9)
            or not math.isclose(float(intrinsics[1, 0]), 0.0, abs_tol=1e-9)
        ):
            raise ResultPipelineError("model intrinsics are unsupported or malformed")
    return model_height, model_width


def _normalization(predictions: Sequence[AlignedPrediction]) -> tuple[np.ndarray, list[np.ndarray]]:
    opencv_c2w = [np.linalg.inv(frame.world_to_camera_opencv) for frame in predictions]
    first_blender = opencv_c2w[0] @ _BLENDER_FROM_OPENCV_CAMERA
    normalization = _FIRST_CAMERA_TARGET @ np.linalg.inv(first_blender)
    return normalization, opencv_c2w


def _filter_frame(
    frame: AlignedPrediction,
    profile: ResultProfile,
) -> tuple[np.ndarray, FrameFilterStatistics]:
    valid = np.isfinite(frame.depth) & (frame.depth > 0) & np.isfinite(frame.confidence)
    valid_count = int(np.count_nonzero(valid))
    if not valid_count:
        return valid, FrameFilterStatistics(frame.frame_index, 0, 0, 0, None, None)
    confidence_threshold = float(
        np.percentile(frame.confidence[valid], profile.confidence_cutoff_percent)
    )
    retained = valid & (frame.confidence >= confidence_threshold)
    confidence_count = int(np.count_nonzero(retained))
    depth_threshold: float | None = None
    if profile.depth_cutoff_percent < 100 and confidence_count:
        depth_threshold = float(
            np.percentile(frame.depth[retained], profile.depth_cutoff_percent)
        )
        retained &= frame.depth <= depth_threshold
    depth_count = int(np.count_nonzero(retained))
    return retained, FrameFilterStatistics(
        frame.frame_index,
        valid_count,
        confidence_count,
        depth_count,
        confidence_threshold,
        depth_threshold,
    )


def _camera_arrays(
    predictions: Sequence[AlignedPrediction],
    normalization: np.ndarray,
    opencv_c2w: Sequence[np.ndarray],
    source_to_model: np.ndarray,
    model_height: int,
    model_width: int,
) -> dict[str, np.ndarray]:
    frame_count = len(predictions)
    cameras = np.empty((frame_count, 4, 4), dtype="<f4")
    model_intrinsics = np.empty((frame_count, 3, 3), dtype="<f4")
    source_intrinsics = np.empty((frame_count, 3, 3), dtype="<f4")
    fov = np.empty((frame_count, 2), dtype="<f4")
    source_inverse = np.linalg.inv(source_to_model)
    for index, (frame, camera) in enumerate(zip(predictions, opencv_c2w)):
        cameras[index] = normalization @ camera @ _BLENDER_FROM_OPENCV_CAMERA
        model_intrinsics[index] = frame.model_intrinsics
        source_intrinsics[index] = source_inverse @ frame.model_intrinsics
        fov[index] = (
            2 * math.atan(model_width / (2 * float(frame.model_intrinsics[0, 0]))),
            2 * math.atan(model_height / (2 * float(frame.model_intrinsics[1, 1]))),
        )
    return {
        "camera_to_world": cameras,
        "model_intrinsics": model_intrinsics,
        "source_intrinsics": source_intrinsics,
        "model_fov_radians": fov,
        "source_pts_seconds": np.ascontiguousarray(
            [frame.source_pts_seconds for frame in predictions], dtype="<f8"
        ),
        "source_to_model": np.ascontiguousarray(source_to_model, dtype="<f8"),
        "frame_type": np.ascontiguousarray(
            [frame.frame_type for frame in predictions], dtype="|u1"
        ),
    }


def build_reconstruction_result(
    request: ResultBuildRequest,
    *,
    resource_probe: ResourceProbe,
    cancel: CancelCheck,
) -> ResultBuildOutcome:
    """Build, validate, and atomically publish one complete Reconstruction Result."""

    _validate_profile(request.profile)
    model_height, model_width = _validate_predictions(
        request.predictions, request.source_to_model
    )
    frame_count = len(request.predictions)
    point_budget = request.profile.import_point_budget
    estimated_result = estimate_core_result_bytes(frame_count, point_budget)
    estimated_memory = estimate_fixture_memory_bytes(
        frame_count, model_height * model_width, point_budget
    )
    require_project_disk(resource_probe, request.project_root, estimated_result)
    require_worker_memory(resource_probe, estimated_memory)
    if cancel():
        raise ResultCancelled("Result construction was cancelled before filtering")

    normalization, opencv_c2w = _normalization(request.predictions)
    reducer = PointReducer(
        point_budget,
        initial_edge_length=request.profile.initial_voxel_edge_length,
        origin=(0.0, 0.0, 0.0),
    )
    filter_statistics: list[FrameFilterStatistics] = []
    for frame, camera in zip(request.predictions, opencv_c2w):
        require_worker_memory(resource_probe, estimated_memory)
        require_project_disk(resource_probe, request.project_root, estimated_result)
        if cancel():
            raise ResultCancelled("Result construction was cancelled between frames")
        retained, statistics = _filter_frame(frame, request.profile)
        filter_statistics.append(statistics)
        rows, columns = np.nonzero(retained)
        if not len(rows):
            continue
        depth = frame.depth[rows, columns].astype(np.float64, copy=False)
        intrinsics = frame.model_intrinsics
        camera_points = np.column_stack(
            (
                (columns - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * depth / intrinsics[1, 1],
                depth,
                np.ones(len(depth), dtype=np.float64),
            )
        )
        world = (normalization @ camera @ camera_points.T).T[:, :3]
        for point, row, column in zip(world, rows, columns):
            reducer.add(
                PointCandidate(
                    tuple(float(value) for value in point),
                    tuple(int(value) for value in frame.rgb[row, column]),
                    float(frame.confidence[row, column]),
                    frame.frame_index,
                    int(row) * model_width + int(column),
                )
            )

    reduced = reducer.finish()
    arrays = {
        "positions": reduced.positions,
        "colors": reduced.colors,
        "confidence": reduced.confidence,
        "radius": reduced.radius,
        "source_frame": reduced.source_frame,
        **_camera_arrays(
            request.predictions,
            normalization,
            opencv_c2w,
            request.source_to_model,
            model_height,
            model_width,
        ),
    }
    profile = {
        "name": request.profile.name,
        "confidence_cutoff_percent": float(request.profile.confidence_cutoff_percent),
        "depth_cutoff_percent": float(request.profile.depth_cutoff_percent),
        "import_point_budget": point_budget,
    }
    publication = ResultPublication(
        job_id=request.job_id,
        project_root=request.project_root,
        target_scene=request.target_scene,
        timeline_start=request.timeline_start,
        source=request.source,
        profile=profile,
        provenance=request.provenance,
        warnings=request.warnings,
        arrays=arrays,
        voxel_edge_length=reduced.edge_length,
        voxel_origin=(0.0, 0.0, 0.0),
        created_utc=request.created_utc,
        result_id=request.result_id,
    )

    def disk_check(remaining: int) -> None:
        require_project_disk(resource_probe, request.project_root, remaining)

    published = publish_result_bundle(
        publication,
        cancel=cancel,
        disk_check=disk_check,
        estimated_total_bytes=estimated_result,
    )
    return ResultBuildOutcome(
        published,
        tuple(filter_statistics),
        reducer.maximum_occupied_entries,
    )
