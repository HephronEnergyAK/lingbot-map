"""Deep seam that turns aligned model predictions into one bounded Result."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .dense_predictions import DensePredictionWriter, estimate_dense_prediction_bytes
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
    estimate_dense_buffer_bytes,
    require_project_disk,
    require_worker_memory,
)
from .sky_masking import SkyMaskSession


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
    retain_dense_predictions: bool = False


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


@dataclass(frozen=True)
class IncrementalResultRequest:
    job_id: str
    project_root: Path
    target_scene: Mapping[str, Any]
    timeline_start: int
    source: Mapping[str, Any]
    source_to_model: np.ndarray
    frame_count: int
    profile: ResultProfile
    provenance: Mapping[str, Any]
    warnings: tuple[Mapping[str, str], ...] = ()
    created_utc: str | None = None
    result_id: str | None = None
    model_grid_shape: tuple[int, int] | None = None


PredictionDecoder = Callable[[Any, Any, float], AlignedPrediction]
ProvenanceFactory = Callable[[], Mapping[str, Any]]


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
    if not isinstance(profile.retain_dense_predictions, bool):
        raise ResultPipelineError("Dense Predictions retention must be boolean")


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
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, FrameFilterStatistics]:
    valid = np.isfinite(frame.depth) & (frame.depth > 0) & np.isfinite(frame.confidence)
    if eligible_mask is not None:
        if (
            not isinstance(eligible_mask, np.ndarray)
            or eligible_mask.dtype.str != "|u1"
            or eligible_mask.shape != frame.depth.shape
            or not eligible_mask.flags.c_contiguous
            or not bool(np.isin(eligible_mask, (0, 1)).all())
        ):
            raise ResultPipelineError("Sky Mask eligibility grid is invalid")
        valid &= eligible_mask.astype(bool, copy=False)
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
        "retain_dense_predictions": request.profile.retain_dense_predictions,
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


class IncrementalBundleResultSink:
    """Filter and reduce each finalized frame without retaining dense history."""

    def __init__(
        self,
        request: IncrementalResultRequest,
        *,
        prediction_decoder: PredictionDecoder,
        resource_probe: ResourceProbe,
        cancel: CancelCheck,
        provenance_factory: ProvenanceFactory | None = None,
        sky_mask_session: SkyMaskSession | None = None,
    ) -> None:
        _validate_profile(request.profile)
        if request.frame_count < 1:
            raise ResultPipelineError("incremental Result frame count must be positive")
        _strict_array(request.source_to_model, "<f8", (3, 3), "source_to_model")
        if (
            not np.isfinite(request.source_to_model).all()
            or abs(float(np.linalg.det(request.source_to_model))) <= 1e-12
        ):
            raise ResultPipelineError("source_to_model must be finite and invertible")
        self.request = request
        self.prediction_decoder = prediction_decoder
        self.resource_probe = resource_probe
        self.cancel = cancel
        self.provenance_factory = provenance_factory
        self.sky_mask_session = sky_mask_session
        self.estimated_result = estimate_core_result_bytes(
            request.frame_count,
            request.profile.import_point_budget,
            sky=sky_mask_session is not None,
        )
        self.dense_writer: DensePredictionWriter | None = None
        if request.profile.retain_dense_predictions:
            if (
                request.model_grid_shape is None
                or len(request.model_grid_shape) != 2
                or min(request.model_grid_shape) < 1
            ):
                raise ResultPipelineError(
                    "Dense Predictions require the exact preflight model grid"
                )
            dense_estimate = estimate_dense_prediction_bytes(
                request.frame_count, math.prod(request.model_grid_shape)
            )
            require_project_disk(
                resource_probe,
                request.project_root,
                self.estimated_result + dense_estimate,
            )
            self.dense_writer = DensePredictionWriter(
                project_root=request.project_root,
                job_id=request.job_id,
                frame_count=request.frame_count,
                grid_shape=request.model_grid_shape,
                disk_check=lambda remaining: require_project_disk(
                    resource_probe,
                    request.project_root,
                    self.estimated_result + remaining,
                ),
            )
        self.reducer = PointReducer(
            request.profile.import_point_budget,
            initial_edge_length=request.profile.initial_voxel_edge_length,
            origin=(0.0, 0.0, 0.0),
        )
        self.filters: list[FrameFilterStatistics] = []
        self.cameras: list[np.ndarray] = []
        self.model_intrinsics: list[np.ndarray] = []
        self.source_intrinsics: list[np.ndarray] = []
        self.fov: list[tuple[float, float]] = []
        self.timestamps: list[float] = []
        self.frame_types: list[int] = []
        self.window_boundaries: list[dict[str, Any]] = []
        self.window_warnings: list[Mapping[str, str]] = []
        self.normalization: np.ndarray | None = None
        self.model_shape: tuple[int, int] | None = None
        self._finished = False
        require_project_disk(resource_probe, request.project_root, self.estimated_result)

    def prepare_frame(self, frame_index: int, canonical: Any) -> None:
        if self._finished:
            raise ResultPipelineError("incremental Result sink is already finished")
        if self.sky_mask_session is not None:
            self.sky_mask_session.prepare(frame_index, canonical)

    @property
    def estimated_remaining_bytes(self) -> int:
        dense = (
            self.dense_writer.estimated_remaining_bytes
            if self.dense_writer is not None
            else 0
        )
        return self.estimated_result + dense

    @property
    def finalized_frame_count(self) -> int:
        return len(self.cameras)

    def accept(self, prediction: Any, canonical: Any, pts_seconds: float) -> None:
        if self._finished:
            raise ResultPipelineError("incremental Result sink is already finished")
        expected = self.finalized_frame_count
        aligned = self.prediction_decoder(prediction, canonical, pts_seconds)
        if aligned.frame_index != expected:
            raise ResultPipelineError("prediction decoder changed or reordered frame identity")
        if expected >= self.request.frame_count:
            raise ResultPipelineError("incremental Result received excess predictions")
        if self.timestamps and aligned.source_pts_seconds <= self.timestamps[-1]:
            raise ResultPipelineError("incremental Result timestamps are not strictly increasing")
        validation = replace(aligned, frame_index=0, source_pts_seconds=0.0)
        height, width = _validate_predictions((validation,), self.request.source_to_model)
        if self.model_shape is None:
            self.model_shape = (height, width)
        elif self.model_shape != (height, width):
            raise ResultPipelineError("incremental Result model grid changed between frames")
        estimated_memory = estimate_fixture_memory_bytes(
            self.request.frame_count,
            height * width,
            self.request.profile.import_point_budget,
        )
        if self.dense_writer is not None:
            estimated_memory += estimate_dense_buffer_bytes(
                self.request.frame_count, height * width
            )
        require_worker_memory(self.resource_probe, estimated_memory)
        require_project_disk(
            self.resource_probe, self.request.project_root, self.estimated_result
        )
        if self.cancel():
            raise ResultCancelled("Result construction was cancelled between frames")
        opencv_c2w = np.linalg.inv(aligned.world_to_camera_opencv)
        if self.normalization is None:
            first_blender = opencv_c2w @ _BLENDER_FROM_OPENCV_CAMERA
            self.normalization = _FIRST_CAMERA_TARGET @ np.linalg.inv(first_blender)
        eligible_mask = None
        if self.sky_mask_session is not None:
            eligible_mask, _sky_fraction = self.sky_mask_session.mask_for(
                expected, (height, width)
            )
        retained, statistics = _filter_frame(
            aligned, self.request.profile, eligible_mask
        )
        if self.dense_writer is not None:
            self.dense_writer.accept(expected, aligned.depth, aligned.confidence)
        self.filters.append(statistics)
        rows, columns = np.nonzero(retained)
        if len(rows):
            depth = aligned.depth[rows, columns].astype(np.float64, copy=False)
            intrinsics = aligned.model_intrinsics
            camera_points = np.column_stack(
                (
                    (columns - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                    (rows - intrinsics[1, 2]) * depth / intrinsics[1, 1],
                    depth,
                    np.ones(len(depth), dtype=np.float64),
                )
            )
            world = (self.normalization @ opencv_c2w @ camera_points.T).T[:, :3]
            for point, row, column in zip(world, rows, columns):
                self.reducer.add(
                    PointCandidate(
                        tuple(float(value) for value in point),
                        tuple(int(value) for value in aligned.rgb[row, column]),
                        float(aligned.confidence[row, column]),
                        expected,
                        int(row) * width + int(column),
                    )
                )
        source_inverse = np.linalg.inv(self.request.source_to_model)
        self.cameras.append(
            np.ascontiguousarray(
                self.normalization @ opencv_c2w @ _BLENDER_FROM_OPENCV_CAMERA,
                dtype="<f4",
            )
        )
        self.model_intrinsics.append(
            np.ascontiguousarray(aligned.model_intrinsics, dtype="<f4")
        )
        self.source_intrinsics.append(
            np.ascontiguousarray(source_inverse @ aligned.model_intrinsics, dtype="<f4")
        )
        self.fov.append(
            (
                2 * math.atan(width / (2 * float(aligned.model_intrinsics[0, 0]))),
                2 * math.atan(height / (2 * float(aligned.model_intrinsics[1, 1]))),
            )
        )
        self.timestamps.append(float(aligned.source_pts_seconds))
        self.frame_types.append(int(aligned.frame_type))

    def record_window_boundary(self, metrics: Any) -> None:
        if self._finished:
            raise ResultPipelineError("incremental Result sink is already finished")
        document = metrics.document()
        warning = metrics.warning()
        self.window_boundaries.append(document)
        if warning is not None:
            self.window_warnings.append(warning)

    def finish(self) -> ResultBuildOutcome:
        if self._finished:
            raise ResultPipelineError("incremental Result sink finish is not repeatable")
        self._finished = True
        if self.finalized_frame_count != self.request.frame_count:
            raise ResultPipelineError(
                f"incremental Result finalized {self.finalized_frame_count} of "
                f"{self.request.frame_count} frames"
            )
        reduced = self.reducer.finish()
        sky_outcome = (
            self.sky_mask_session.finish()
            if self.sky_mask_session is not None
            else None
        )
        arrays = {
            "positions": reduced.positions,
            "colors": reduced.colors,
            "confidence": reduced.confidence,
            "radius": reduced.radius,
            "source_frame": reduced.source_frame,
            "camera_to_world": np.ascontiguousarray(np.stack(self.cameras), dtype="<f4"),
            "model_intrinsics": np.ascontiguousarray(np.stack(self.model_intrinsics), dtype="<f4"),
            "source_intrinsics": np.ascontiguousarray(np.stack(self.source_intrinsics), dtype="<f4"),
            "model_fov_radians": np.ascontiguousarray(self.fov, dtype="<f4"),
            "source_pts_seconds": np.ascontiguousarray(self.timestamps, dtype="<f8"),
            "source_to_model": np.ascontiguousarray(self.request.source_to_model, dtype="<f8"),
            "frame_type": np.ascontiguousarray(self.frame_types, dtype="|u1"),
        }
        if sky_outcome is not None:
            arrays["sky_fraction"] = sky_outcome.sky_fraction
        profile = {
            "name": self.request.profile.name,
            "confidence_cutoff_percent": float(
                self.request.profile.confidence_cutoff_percent
            ),
            "depth_cutoff_percent": float(self.request.profile.depth_cutoff_percent),
            "import_point_budget": self.request.profile.import_point_budget,
            "retain_dense_predictions": self.request.profile.retain_dense_predictions,
        }
        provenance = (
            self.provenance_factory()
            if self.provenance_factory is not None
            else self.request.provenance
        )
        if sky_outcome is not None:
            provenance = dict(provenance)
            models = list(provenance["models"])
            models.append(
                {
                    "id": sky_outcome.provenance["model_id"],
                    "role": "auxiliary",
                    "sha256": sky_outcome.provenance["model_sha256"],
                }
            )
            provenance["models"] = models
            provenance["sky_masking"] = dict(sky_outcome.provenance)
        dense_component = (
            self.dense_writer.finish(provenance)
            if self.dense_writer is not None
            else None
        )
        publication = ResultPublication(
            job_id=self.request.job_id,
            project_root=self.request.project_root,
            target_scene=self.request.target_scene,
            timeline_start=self.request.timeline_start,
            source=self.request.source,
            profile=profile,
            provenance=provenance,
            warnings=(
                *self.request.warnings,
                *self.window_warnings,
                *(sky_outcome.warnings if sky_outcome is not None else ()),
            ),
            arrays=arrays,
            voxel_edge_length=reduced.edge_length,
            voxel_origin=(0.0, 0.0, 0.0),
            created_utc=self.request.created_utc,
            result_id=self.request.result_id,
            dense_component=dense_component,
            window_alignment=(
                {
                    "schema_version": "1.0.0",
                    "rule_version": "1.0.0",
                    "strategy": "rolling-similarity",
                    "window_frames": 64,
                    "overlap_keyframes": 16,
                    "scale_frames": 8,
                    "keyframe_interval": 1,
                    "loop_closure": False,
                    "pose_graph": False,
                    "bundle_adjustment": False,
                    "global_optimization": False,
                    "boundaries": self.window_boundaries,
                }
                if self.window_boundaries
                else None
            ),
        )

        def disk_check(remaining: int) -> None:
            require_project_disk(
                self.resource_probe, self.request.project_root, remaining
            )

        try:
            published = publish_result_bundle(
                publication,
                cancel=self.cancel,
                disk_check=disk_check,
                estimated_total_bytes=self.estimated_result,
            )
        except Exception:
            self.abort("failed")
            raise
        return ResultBuildOutcome(
            published,
            tuple(self.filters),
            self.reducer.maximum_occupied_entries,
        )

    def abort(self, reason: str) -> Path | None:
        if self.sky_mask_session is not None:
            self.sky_mask_session.abort()
        if self.dense_writer is None:
            return None
        return self.dense_writer.abort(reason)
