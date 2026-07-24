"""Source-aligned camera, background, and Model Coverage contracts.

This module intentionally imports neither Blender nor NumPy at module load.
Its coordinate and identity functions are shared by transactional import,
explicit UI actions, and Blender visual-oracle tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

from .job_lifecycle import is_reparse_point


DISPLAY_TRANSFORMS = frozenset(
    {
        "identity",
        "rotate_90_ccw",
        "rotate_180",
        "rotate_270_ccw",
        "reflect_x",
        "reflect_y",
        "reflect_main_diagonal",
        "reflect_anti_diagonal",
    }
)
SWAPS_DIMENSIONS = frozenset(
    {
        "rotate_90_ccw",
        "rotate_270_ccw",
        "reflect_main_diagonal",
        "reflect_anti_diagonal",
    }
)
HASH_CHUNK_BYTES = 1024 * 1024


class SourceViewError(ValueError):
    """Source-aligned inspection metadata or media is inconsistent."""


@dataclass(frozen=True)
class SourceViewContract:
    width: int
    height: int
    display_transform: str
    coverage_polygon: tuple[tuple[float, float], ...]
    source_fraction: float
    model_width: int
    model_height: int

    @property
    def coded_size(self) -> tuple[int, int]:
        if self.display_transform in SWAPS_DIMENSIONS:
            return self.height, self.width
        return self.width, self.height


@dataclass(frozen=True)
class BackgroundMapping:
    rotation_radians: float
    flip_x: bool
    flip_y: bool


# Blender applies the flips in coded-image space and then the view rotation.
# The diagonal mappings are therefore R(90 degrees) * Fx/Fy.
BACKGROUND_MAPPINGS: Mapping[str, BackgroundMapping] = {
    "identity": BackgroundMapping(0.0, False, False),
    "rotate_90_ccw": BackgroundMapping(-math.pi / 2.0, False, False),
    "rotate_180": BackgroundMapping(math.pi, False, False),
    "rotate_270_ccw": BackgroundMapping(math.pi / 2.0, False, False),
    "reflect_x": BackgroundMapping(0.0, True, False),
    "reflect_y": BackgroundMapping(0.0, False, True),
    "reflect_main_diagonal": BackgroundMapping(
        math.pi / 2.0, False, True
    ),
    "reflect_anti_diagonal": BackgroundMapping(
        math.pi / 2.0, True, False
    ),
}


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SourceViewError(f"{label} must be a finite number")
    return float(value)


def validate_source_view_contract(
    manifest: Mapping[str, Any],
    source_to_model: Sequence[Sequence[float]],
) -> SourceViewContract | None:
    """Return a strict optional contract; legacy core-only Results return None."""

    source_display = manifest.get("source_display")
    model_coverage = manifest.get("model_coverage")
    if (source_display is None) != (model_coverage is None):
        raise SourceViewError(
            "source_display and model_coverage must be present together"
        )
    if source_display is None:
        return None
    if not isinstance(source_display, dict) or set(source_display) != {
        "width",
        "height",
        "display_transform",
    }:
        raise SourceViewError("source_display has unknown or missing fields")
    width = source_display["width"]
    height = source_display["height"]
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
    ):
        raise SourceViewError("source_display dimensions are invalid")
    transform = source_display["display_transform"]
    if transform not in DISPLAY_TRANSFORMS:
        raise SourceViewError("source_display Display Transform is unsupported")

    if not isinstance(model_coverage, dict) or set(model_coverage) != {
        "coordinate_space",
        "polygon",
        "source_fraction",
        "model_width",
        "model_height",
    }:
        raise SourceViewError("model_coverage has unknown or missing fields")
    if model_coverage["coordinate_space"] != "source-display-pixel-edges":
        raise SourceViewError("Model Coverage coordinate space is unsupported")
    model_width = model_coverage["model_width"]
    model_height = model_coverage["model_height"]
    if (
        isinstance(model_width, bool)
        or isinstance(model_height, bool)
        or not isinstance(model_width, int)
        or not isinstance(model_height, int)
        or model_width != 518
        or not 14 <= model_height <= 518
        or model_height % 14
    ):
        raise SourceViewError("Model Coverage grid is not the frozen 518/14 grid")
    polygon_raw = model_coverage["polygon"]
    if (
        not isinstance(polygon_raw, list)
        or len(polygon_raw) != 4
        or any(not isinstance(point, list) or len(point) != 2 for point in polygon_raw)
    ):
        raise SourceViewError("Model Coverage polygon must contain four points")
    polygon = tuple(
        (
            _finite_number(point[0], "Model Coverage x"),
            _finite_number(point[1], "Model Coverage y"),
        )
        for point in polygon_raw
    )
    if any(
        x < 0 or x > width or y < 0 or y > height for x, y in polygon
    ):
        raise SourceViewError(
            "Model Coverage polygon lies outside source-display pixel edges"
        )
    if (
        len(source_to_model) != 3
        or any(len(row) != 3 for row in source_to_model)
    ):
        raise SourceViewError("source_to_model is not a 3x3 matrix")
    matrix = tuple(
        tuple(_finite_number(value, "source_to_model") for value in row)
        for row in source_to_model
    )
    expected = (
        (0.0, 0.0, 1.0),
        (float(model_width), 0.0, 1.0),
        (float(model_width), float(model_height), 1.0),
        (0.0, float(model_height), 1.0),
    )
    for point, target in zip(polygon, expected):
        vector = (point[0], point[1], 1.0)
        mapped = tuple(
            sum(matrix[row][column] * vector[column] for column in range(3))
            for row in range(3)
        )
        if any(
            not math.isclose(
                mapped[index], target[index], abs_tol=1e-6, rel_tol=1e-9
            )
            for index in range(3)
        ):
            raise SourceViewError(
                "Model Coverage polygon disagrees with source_to_model"
            )
    area = abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % 4][1]
            - polygon[index][1] * polygon[(index + 1) % 4][0]
            for index in range(4)
        )
    ) / 2.0
    fraction = _finite_number(
        model_coverage["source_fraction"], "Model Coverage source_fraction"
    )
    actual_fraction = area / float(width * height)
    if (
        not 0 < fraction <= 1
        or not math.isclose(
            fraction, actual_fraction, abs_tol=1e-9, rel_tol=1e-9
        )
    ):
        raise SourceViewError(
            "Model Coverage source fraction disagrees with its polygon"
        )
    return SourceViewContract(
        width,
        height,
        transform,
        polygon,
        fraction,
        model_width,
        model_height,
    )


def transform_pixel(
    transform: str,
    x: int,
    y: int,
    coded_width: int,
    coded_height: int,
) -> tuple[int, int]:
    """Map one coded pixel centre to the decoder's displayed pixel index."""

    if transform not in DISPLAY_TRANSFORMS:
        raise SourceViewError("Display Transform is unsupported")
    if not 0 <= x < coded_width or not 0 <= y < coded_height:
        raise SourceViewError("coded pixel lies outside the source")
    if transform == "identity":
        return x, y
    if transform == "rotate_90_ccw":
        return y, coded_width - 1 - x
    if transform == "rotate_180":
        return coded_width - 1 - x, coded_height - 1 - y
    if transform == "rotate_270_ccw":
        return coded_height - 1 - y, x
    if transform == "reflect_x":
        return coded_width - 1 - x, y
    if transform == "reflect_y":
        return x, coded_height - 1 - y
    if transform == "reflect_main_diagonal":
        return y, x
    return coded_height - 1 - y, coded_width - 1 - x


def background_scale(contract: SourceViewContract) -> float:
    """Compensate Blender FIT for a background rotated by a quarter turn."""

    if contract.display_transform not in SWAPS_DIMENSIONS:
        return 1.0
    coded_width, coded_height = contract.coded_size
    return max(
        coded_width / coded_height,
        coded_height / coded_width,
    )


def coverage_to_camera_border(
    contract: SourceViewContract,
    border: tuple[float, float, float, float],
) -> tuple[tuple[float, float], ...]:
    """Map source pixel-edge coverage into a Camera View screen border."""

    left, bottom, right, top = border
    if not all(math.isfinite(value) for value in border):
        raise SourceViewError("Camera View border is non-finite")
    if right <= left or top <= bottom:
        raise SourceViewError("Camera View border is empty")
    return tuple(
        (
            left + (x / contract.width) * (right - left),
            top - (y / contract.height) * (top - bottom),
        )
        for x, y in contract.coverage_polygon
    )


def scene_aspect_matches(scene: Any, contract: SourceViewContract) -> bool:
    render = scene.render
    values = (
        float(render.resolution_x),
        float(render.resolution_y),
        float(render.pixel_aspect_x),
        float(render.pixel_aspect_y),
    )
    if not all(math.isfinite(value) and value > 0 for value in values):
        return False
    scene_cross = values[0] * values[2] * contract.height
    source_cross = values[1] * values[3] * contract.width
    return math.isclose(
        scene_cross, source_cross, abs_tol=1e-9, rel_tol=1e-12
    )


def candidate_source_paths(
    source: Mapping[str, Any],
    *,
    current_blend_path: str | Path | None,
    relink_path: str | Path | None = None,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if relink_path:
        candidates.append(Path(os.path.abspath(relink_path)))
    relative = source.get("scene_relative_path")
    if (
        isinstance(relative, str)
        and relative.startswith("//")
        and current_blend_path
    ):
        blend = Path(os.path.abspath(current_blend_path))
        candidates.append(Path(os.path.abspath(blend.parent / relative[2:])))
    absolute = source.get("absolute_path")
    if isinstance(absolute, str) and absolute:
        candidates.append(Path(os.path.abspath(absolute)))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def validate_source_media(
    path: str | Path,
    source: Mapping[str, Any],
    *,
    cancel: Callable[[], bool] | None = None,
) -> Path:
    """Accept an ordinary MP4/MOV only when its exact recorded SHA-256 matches."""

    media = Path(os.path.abspath(path))
    if (
        media.suffix.lower() not in {".mp4", ".mov"}
        or not media.is_file()
        or media.is_symlink()
        or is_reparse_point(media)
    ):
        raise SourceViewError(
            "Capture Source is absent or is not an ordinary local MP4/MOV"
        )
    before = media.stat()
    attributes = getattr(before, "st_file_attributes", 0)
    if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise SourceViewError("Capture Source must not be a reparse point")
    expected_size = source.get("size_bytes")
    expected_sha256 = source.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise SourceViewError("Recorded Capture Source identity is invalid")
    digest = hashlib.sha256()
    completed = 0
    with media.open("rb") as stream:
        while True:
            if cancel is not None and cancel():
                raise SourceViewError("Capture Source validation was cancelled")
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            completed += len(chunk)
    after = media.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or completed != before.st_size
    ):
        raise SourceViewError(
            "Capture Source changed while its checksum was validated"
        )
    if completed != expected_size or digest.hexdigest() != expected_sha256:
        raise SourceViewError(
            "Capture Source checksum does not match the recorded Result identity"
        )
    return media


__all__ = [
    "BACKGROUND_MAPPINGS",
    "DISPLAY_TRANSFORMS",
    "BackgroundMapping",
    "SourceViewContract",
    "SourceViewError",
    "background_scale",
    "candidate_source_paths",
    "coverage_to_camera_border",
    "scene_aspect_matches",
    "transform_pixel",
    "validate_source_media",
    "validate_source_view_contract",
]
