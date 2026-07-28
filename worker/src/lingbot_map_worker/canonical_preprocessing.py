"""Pinned canonical crop/resize and model-input conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


IMAGE_SIZE = 518
PATCH_SIZE = 14
PREPROCESSING_RULE_VERSION = "1.0.0"


class PreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalImage:
    model_input: np.ndarray
    color_rgb: np.ndarray
    source_to_model: np.ndarray
    coverage_polygon: tuple[tuple[float, float], ...]


def canonical_geometry(
    source_height: int, source_width: int
) -> tuple[int, int, np.ndarray, tuple[tuple[float, float], ...]]:
    if (
        isinstance(source_height, bool)
        or isinstance(source_width, bool)
        or not isinstance(source_height, int)
        or not isinstance(source_width, int)
        or source_height < 1
        or source_width < 1
    ):
        raise PreprocessingError("source dimensions must be positive integers")
    resized_width = IMAGE_SIZE
    resized_height = max(
        PATCH_SIZE,
        round(source_height * (resized_width / source_width) / PATCH_SIZE) * PATCH_SIZE,
    )
    crop_top = max(0, (resized_height - IMAGE_SIZE) // 2)
    crop_bottom = min(resized_height, crop_top + IMAGE_SIZE)
    if crop_bottom - crop_top < PATCH_SIZE:
        raise PreprocessingError("canonical crop produced an invalid model grid")
    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    transform = np.array(
        ((scale_x, 0.0, 0.0), (0.0, scale_y, -float(crop_top)), (0.0, 0.0, 1.0)),
        dtype="<f8",
    )
    source_top = crop_top / scale_y
    source_bottom = crop_bottom / scale_y
    coverage = (
        (0.0, source_top),
        (float(source_width), source_top),
        (float(source_width), source_bottom),
        (0.0, source_bottom),
    )
    return crop_bottom - crop_top, resized_width, transform, coverage


def canonicalize_srgb(rgb: np.ndarray) -> CanonicalImage:
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype.str != "|u1"
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or not rgb.flags.c_contiguous
    ):
        raise PreprocessingError("canonical input must be C-order uint8 sRGB (H,W,3)")
    source_height, source_width = rgb.shape[:2]
    if source_height < 1 or source_width < 1:
        raise PreprocessingError("source frame cannot be empty")
    target_height, resized_width, transform, coverage = canonical_geometry(
        source_height, source_width
    )
    resized_height = max(
        PATCH_SIZE,
        round(source_height * (resized_width / source_width) / PATCH_SIZE) * PATCH_SIZE,
    )
    image = Image.fromarray(rgb).resize(
        (resized_width, resized_height), Image.Resampling.BICUBIC
    )
    crop_top = max(0, (resized_height - IMAGE_SIZE) // 2)
    crop_bottom = min(resized_height, crop_top + IMAGE_SIZE)
    if crop_bottom - crop_top != target_height:
        raise PreprocessingError("canonical geometry changed during preprocessing")
    color = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[crop_top:crop_bottom])
    model = np.ascontiguousarray(color.transpose(2, 0, 1), dtype="<f4") / np.float32(255.0)
    model = np.ascontiguousarray(model, dtype="<f4")
    if not np.isfinite(model).all() or float(model.min()) < 0 or float(model.max()) > 1:
        raise PreprocessingError("model normalization produced invalid values")
    return CanonicalImage(model, color, transform, coverage)
