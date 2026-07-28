"""Deterministic, fixed-origin, bounded-memory Point Reducer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


MAX_IMPORT_POINT_BUDGET = 50_000_000


class PointReducerError(ValueError):
    pass


@dataclass(frozen=True)
class PointCandidate:
    position: tuple[float, float, float]
    color: tuple[int, int, int]
    confidence: float
    source_frame: int
    pixel_index: int


@dataclass(frozen=True)
class ReducedPoints:
    positions: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    radius: np.ndarray
    source_frame: np.ndarray
    voxel_coordinates: tuple[tuple[int, int, int], ...]
    edge_length: float


def _winner(left: PointCandidate, right: PointCandidate) -> PointCandidate:
    left_key = (-left.confidence, left.source_frame, left.pixel_index)
    right_key = (-right.confidence, right.source_frame, right.pixel_index)
    return left if left_key <= right_key else right


class PointReducer:
    """Retain one deterministic original candidate per nested voxel grid."""

    def __init__(
        self,
        budget: int,
        *,
        initial_edge_length: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= MAX_IMPORT_POINT_BUDGET:
            raise PointReducerError(
                f"Import Point Budget must be in [1, {MAX_IMPORT_POINT_BUDGET}]"
            )
        if not math.isfinite(initial_edge_length) or initial_edge_length <= 0:
            raise PointReducerError("initial voxel edge length must be finite and positive")
        if len(origin) != 3 or not all(math.isfinite(float(value)) for value in origin):
            raise PointReducerError("voxel origin must contain three finite values")
        self.budget = budget
        self.edge_length = float(initial_edge_length)
        self.origin = tuple(float(value) for value in origin)
        self._voxels: dict[tuple[int, int, int], PointCandidate] = {}
        self.maximum_occupied_entries = 0

    @property
    def occupied_entries(self) -> int:
        return len(self._voxels)

    def add(self, candidate: PointCandidate) -> None:
        self._validate_candidate(candidate)
        coordinate = self._coordinate(candidate.position)
        current = self._voxels.get(coordinate)
        if current is not None:
            self._voxels[coordinate] = _winner(current, candidate)
            return
        self._voxels[coordinate] = candidate
        self.maximum_occupied_entries = max(
            self.maximum_occupied_entries, len(self._voxels)
        )
        if len(self._voxels) > self.budget:
            self._coarsen_to_budget()

    def add_many(self, candidates: Iterable[PointCandidate]) -> None:
        for candidate in candidates:
            self.add(candidate)

    def finish(self) -> ReducedPoints:
        ordered_coordinates = tuple(sorted(self._voxels))
        representatives = [self._voxels[key] for key in ordered_coordinates]
        count = len(representatives)
        positions = np.empty((count, 3), dtype="<f4")
        colors = np.empty((count, 3), dtype="|u1")
        confidence = np.empty((count,), dtype="<f4")
        source_frame = np.empty((count,), dtype="<u4")
        for index, representative in enumerate(representatives):
            positions[index] = representative.position
            colors[index] = representative.color
            confidence[index] = representative.confidence
            source_frame[index] = representative.source_frame
        radius = np.full((count,), self.edge_length / 2.0, dtype="<f4")
        return ReducedPoints(
            positions,
            colors,
            confidence,
            radius,
            source_frame,
            ordered_coordinates,
            self.edge_length,
        )

    def _coarsen_to_budget(self) -> None:
        doublings = 0
        while len(self._voxels) > self.budget:
            retained = tuple(self._voxels.values())
            self._voxels.clear()
            self.edge_length *= 2.0
            if not math.isfinite(self.edge_length):
                raise PointReducerError("voxel edge length overflowed before reaching budget")
            for candidate in retained:
                coordinate = self._coordinate(candidate.position)
                current = self._voxels.get(coordinate)
                self._voxels[coordinate] = (
                    candidate if current is None else _winner(current, candidate)
                )
            doublings += 1
            if doublings > 1074:
                raise PointReducerError(
                    "fixed-origin voxel grid cannot satisfy the requested point budget"
                )
        self.maximum_occupied_entries = max(
            self.maximum_occupied_entries, len(self._voxels)
        )

    def _coordinate(self, position: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(
            math.floor((float(value) - origin) / self.edge_length)
            for value, origin in zip(position, self.origin)
        )  # type: ignore[return-value]

    @staticmethod
    def _validate_candidate(candidate: PointCandidate) -> None:
        if len(candidate.position) != 3 or not all(
            math.isfinite(float(value)) for value in candidate.position
        ):
            raise PointReducerError("point position must contain three finite values")
        if len(candidate.color) != 3 or not all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255
            for value in candidate.color
        ):
            raise PointReducerError("point color must contain three uint8 values")
        if not math.isfinite(float(candidate.confidence)):
            raise PointReducerError("point confidence must be finite")
        if (
            isinstance(candidate.source_frame, bool)
            or not isinstance(candidate.source_frame, int)
            or not 0 <= candidate.source_frame <= 0xFFFFFFFF
        ):
            raise PointReducerError("source frame must be a uint32 value")
        if (
            isinstance(candidate.pixel_index, bool)
            or not isinstance(candidate.pixel_index, int)
            or candidate.pixel_index < 0
        ):
            raise PointReducerError("row-major pixel index must be non-negative")
