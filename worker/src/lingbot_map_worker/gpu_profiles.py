"""Immutable native-v1 Reconstruction Profile capability workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable


@dataclass(frozen=True)
class ReconstructionProfile:
    name: str
    camera_iterations: int
    confidence_cutoff_percent: int
    import_point_budget: int
    depth_cutoff_percent: float = 99.5
    image_size: int = 518
    patch_size: int = 14
    scale_frames: int = 8
    window_frames: int = 64
    attention_backend: str = "sdpa"
    execution_mode: str = "eager"

    @property
    def settings_sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    @property
    def workload_key(self) -> tuple[int, int, int, int, int, str, str]:
        """Profiles with an identical GPU workload may share one measurement."""

        return (
            self.camera_iterations,
            self.image_size,
            self.patch_size,
            self.scale_frames,
            self.window_frames,
            self.attention_backend,
            self.execution_mode,
        )


PROFILES = (
    ReconstructionProfile("Draft", 1, 70, 1_000_000),
    ReconstructionProfile("Balanced", 4, 50, 5_000_000),
    ReconstructionProfile("High", 4, 30, 10_000_000),
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}


def profiles_by_name(names: Iterable[str]) -> tuple[ReconstructionProfile, ...]:
    selected: list[ReconstructionProfile] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate Reconstruction Profile: {name}")
        try:
            profile = PROFILE_BY_NAME[name]
        except KeyError as exc:
            raise ValueError(f"unknown Reconstruction Profile: {name}") from exc
        selected.append(profile)
        seen.add(name)
    if not selected:
        raise ValueError("at least one Reconstruction Profile is required")
    return tuple(selected)


@dataclass(frozen=True)
class ProfileSelection:
    name: str
    camera_iterations: int
    confidence_cutoff_percent: float
    depth_cutoff_percent: float
    import_point_budget: int
    point_budget_confirmed: bool = False


@dataclass(frozen=True)
class InferencePlan:
    mode: str
    frame_count: int
    keyframe_interval: int
    scale_frames: int = 8
    window_frames: int = 64
    overlap_keyframes: int = 16


def resolve_profile(selection: ProfileSelection) -> ReconstructionProfile | ProfileSelection:
    """Validate settings and return a named profile only for an exact match."""

    if (
        isinstance(selection.camera_iterations, bool)
        or not isinstance(selection.camera_iterations, int)
        or not 1 <= selection.camera_iterations <= 16
    ):
        raise ValueError("camera iterations must be in [1,16]")
    for value, label in (
        (selection.confidence_cutoff_percent, "confidence cutoff"),
        (selection.depth_cutoff_percent, "depth cutoff"),
    ):
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
            raise ValueError(f"{label} must be in [0,100]")
    budget = selection.import_point_budget
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 50_000_000:
        raise ValueError("Import Point Budget must be in [1,50000000]")
    if budget > 10_000_000 and not selection.point_budget_confirmed:
        raise ValueError("Import Point Budget above 10000000 requires explicit confirmation")
    for profile in PROFILES:
        if (
            selection.camera_iterations == profile.camera_iterations
            and float(selection.confidence_cutoff_percent) == profile.confidence_cutoff_percent
            and float(selection.depth_cutoff_percent) == profile.depth_cutoff_percent
            and budget == profile.import_point_budget
        ):
            return profile
    return ProfileSelection(
        "Custom",
        selection.camera_iterations,
        float(selection.confidence_cutoff_percent),
        float(selection.depth_cutoff_percent),
        budget,
        selection.point_budget_confirmed,
    )


def inference_plan(frame_count: int) -> InferencePlan:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or not 8 <= frame_count:
        raise ValueError("Capture Source must contain at least 8 frames")
    if frame_count <= 3000:
        interval = 1 if frame_count <= 320 else math.ceil(frame_count / 320)
        return InferencePlan("streaming", frame_count, interval)
    return InferencePlan("windowed", frame_count, 1)
