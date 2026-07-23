"""Immutable native-v1 Reconstruction Profile capability workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class ReconstructionProfile:
    name: str
    camera_iterations: int
    confidence_cutoff_percent: int
    import_point_budget: int
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
