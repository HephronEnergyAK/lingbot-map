"""Pinned PyAV Decoder Adapter for deterministic complete-source preflight."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any, Callable, Iterable

import av
import numpy as np


HASH_CHUNK_BYTES = 1024 * 1024
MINIMUM_FRAMES = 8
TRANSFORM_SCALE = 1 << 16
DISPLAY_W_SCALE = 1 << 30


class DecoderError(RuntimeError):
    pass


class PreflightCancelled(DecoderError):
    pass


@dataclass(frozen=True)
class SourceIdentity:
    size_bytes: int
    modification_time_ns: int
    sha256: str


@dataclass(frozen=True)
class DisplayTransform:
    name: str
    matrix: tuple[int, int, int, int]
    swaps_dimensions: bool


@dataclass(frozen=True)
class ColorDescription:
    standard: str
    range: str
    primaries: int
    transfer: int
    space: int
    range_code: int


@dataclass(frozen=True)
class PreflightReport:
    identity: SourceIdentity
    container_format: str
    video_stream_index: int
    video_codec: str
    coded_width: int
    coded_height: int
    displayed_width: int
    displayed_height: int
    display_transform: DisplayTransform
    color: ColorDescription
    pixel_format: str
    stream_time_base: tuple[int, int]
    nominal_frame_rate: str | None
    timestamps_seconds: tuple[float, ...]
    variable_frame_rate: bool
    rgb_sha256: str
    rgb_bytes: int
    audio_streams: tuple[dict[str, object], ...]
    pyav_version: str
    ffmpeg_libraries: tuple[tuple[str, tuple[int, ...]], ...]


CancelCheck = Callable[[], bool]
Progress = Callable[[str, int], None]


TRANSFORMS: dict[tuple[int, int, int, int], DisplayTransform] = {
    (1, 0, 0, 1): DisplayTransform("identity", (1, 0, 0, 1), False),
    (0, -1, 1, 0): DisplayTransform("rotate_90_ccw", (0, -1, 1, 0), True),
    (-1, 0, 0, -1): DisplayTransform("rotate_180", (-1, 0, 0, -1), False),
    (0, 1, -1, 0): DisplayTransform("rotate_270_ccw", (0, 1, -1, 0), True),
    (-1, 0, 0, 1): DisplayTransform("reflect_x", (-1, 0, 0, 1), False),
    (1, 0, 0, -1): DisplayTransform("reflect_y", (1, 0, 0, -1), False),
    (0, 1, 1, 0): DisplayTransform("reflect_main_diagonal", (0, 1, 1, 0), True),
    (0, -1, -1, 0): DisplayTransform("reflect_anti_diagonal", (0, -1, -1, 0), True),
}


def _cancelled(cancel: CancelCheck) -> None:
    if cancel():
        raise PreflightCancelled("Capture Source preflight was cancelled")


def _plain_local_source(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if not path.is_absolute() or path.suffix.lower() not in {".mp4", ".mov"}:
        raise DecoderError("Capture Source must be one absolute local MP4 or MOV file")
    if not path.is_file() or path.is_symlink():
        raise DecoderError("Capture Source is absent or not an ordinary file")
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise DecoderError("Capture Source must not be a reparse point")
    return path


def source_identity(
    path: Path,
    cancel: CancelCheck,
    progress: Progress | None = None,
) -> SourceIdentity:
    path = _plain_local_source(path)
    before = path.stat()
    digest = hashlib.sha256()
    completed = 0
    with path.open("rb") as stream:
        while True:
            _cancelled(cancel)
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            completed += len(chunk)
            if progress:
                progress("hash", completed)
    after = path.stat()
    if (
        after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or completed != before.st_size
    ):
        raise DecoderError("Capture Source changed while its identity was computed")
    return SourceIdentity(before.st_size, before.st_mtime_ns, digest.hexdigest())


def require_same_source(path: Path, expected: SourceIdentity, cancel: CancelCheck) -> None:
    current = source_identity(path, cancel)
    if current != expected:
        raise DecoderError("Capture Source identity changed after preflight")


def select_video_stream(container: Any) -> Any:
    def has_disposition(stream: Any, name: str) -> bool:
        disposition = getattr(stream, "disposition", None)
        flag = getattr(type(disposition), name, None)
        if flag is not None:
            try:
                return bool(disposition & flag)
            except TypeError:
                pass
        return bool(getattr(disposition, name, False))

    ordinary = []
    for stream in container.streams.video:
        if has_disposition(stream, "attached_pic"):
            continue
        ordinary.append(stream)
    if not ordinary:
        raise DecoderError("Capture Source has no ordinary video track")
    if len(ordinary) == 1:
        return ordinary[0]
    defaults = [
        stream for stream in ordinary
        if has_disposition(stream, "default")
    ]
    if len(defaults) != 1:
        raise DecoderError("Capture Source has multiple video tracks without one unique default")
    return defaults[0]


def _normalized_matrix(values: Iterable[int]) -> tuple[int, int, int, int]:
    raw = tuple(int(value) for value in values)
    if len(raw) != 9:
        raise DecoderError("Display Transform matrix does not contain nine fixed-point values")
    if raw[2] != 0 or raw[5] != 0 or raw[6] != 0 or raw[7] != 0 or raw[8] != DISPLAY_W_SCALE:
        raise DecoderError("Display Transform contains perspective or translation")
    linear = (raw[0], raw[1], raw[3], raw[4])
    if any(value % TRANSFORM_SCALE for value in linear):
        raise DecoderError("Display Transform is not an exact integer orthogonal mapping")
    normalized = tuple(value // TRANSFORM_SCALE for value in linear)
    if normalized not in TRANSFORMS:
        raise DecoderError("Display Transform is not one of the eight supported mappings")
    return normalized  # type: ignore[return-value]


def display_transform(frame: Any, stream: Any, inherited: DisplayTransform | None) -> DisplayTransform:
    side_data = getattr(frame, "side_data", None)
    matrix_data = side_data.get("DISPLAYMATRIX") if side_data is not None else None
    if matrix_data is not None:
        values = struct.unpack("=9i", bytes(matrix_data))
        return TRANSFORMS[_normalized_matrix(values)]
    rotate = str(getattr(stream, "metadata", {}).get("rotate", "")).strip()
    if rotate:
        try:
            angle = int(rotate) % 360
        except ValueError as exc:
            raise DecoderError("Display rotation metadata is not an integer quarter turn") from exc
        rotations = {
            0: (1, 0, 0, 1), 90: (0, -1, 1, 0),
            180: (-1, 0, 0, -1), 270: (0, 1, -1, 0),
        }
        if angle not in rotations:
            raise DecoderError("Display rotation is not an exact quarter turn")
        return TRANSFORMS[rotations[angle]]
    return inherited or TRANSFORMS[(1, 0, 0, 1)]


def apply_display_transform(rgb: np.ndarray, transform: DisplayTransform) -> np.ndarray:
    matrix = transform.matrix
    if matrix == (1, 0, 0, 1):
        result = rgb
    elif matrix == (0, -1, 1, 0):
        result = np.rot90(rgb, 1)
    elif matrix == (-1, 0, 0, -1):
        result = np.rot90(rgb, 2)
    elif matrix == (0, 1, -1, 0):
        result = np.rot90(rgb, 3)
    elif matrix == (-1, 0, 0, 1):
        result = np.flip(rgb, axis=1)
    elif matrix == (1, 0, 0, -1):
        result = np.flip(rgb, axis=0)
    elif matrix == (0, 1, 1, 0):
        result = np.transpose(rgb, (1, 0, 2))
    elif matrix == (0, -1, -1, 0):
        result = np.flip(np.transpose(rgb, (1, 0, 2)), axis=(0, 1))
    else:  # pragma: no cover - DisplayTransform construction is closed.
        raise DecoderError("Unsupported Display Transform")
    return np.ascontiguousarray(result, dtype=np.uint8)


def color_description(frame: Any) -> ColorDescription:
    primaries = int(frame.color_primaries)
    transfer = int(frame.color_trc)
    space = int(frame.colorspace)
    range_code = int(frame.color_range)
    if range_code not in {1, 2}:
        raise DecoderError("Capture Source color range must be explicitly limited or full")
    if (primaries, transfer, space) == (1, 1, 1):
        standard = "bt709"
    elif primaries in {5, 6} and transfer in {5, 6} and space in {5, 6}:
        standard = "bt601"
    else:
        raise DecoderError(
            "Capture Source declares HDR or unsupported color metadata; "
            "SDR BT.601 or BT.709 is required"
        )
    return ColorDescription(
        standard, "limited" if range_code == 1 else "full",
        primaries, transfer, space, range_code,
    )


def _has_alpha(frame: Any) -> bool:
    return any(bool(getattr(component, "is_alpha", False)) for component in frame.format.components)


def _sample_aspect_ratio(frame: Any, stream: Any) -> Fraction:
    candidates = (
        getattr(frame, "sample_aspect_ratio", None),
        getattr(stream, "sample_aspect_ratio", None),
        getattr(getattr(stream, "codec_context", None), "sample_aspect_ratio", None),
    )
    value = next((candidate for candidate in candidates if candidate is not None), None)
    if value is None:
        return Fraction(1, 1)
    return Fraction(value)


def _audio_metadata(container: Any) -> tuple[dict[str, object], ...]:
    result = []
    for stream in container.streams.audio:
        context = stream.codec_context
        layout = getattr(context, "layout", None)
        channels = getattr(layout, "nb_channels", None)
        if channels is None and layout is not None:
            channels = len(getattr(layout, "channels", ()))
        result.append(
            {
                "stream_index": int(stream.index),
                "codec": str(getattr(context, "name", "unknown")),
                "sample_rate": int(getattr(context, "sample_rate", 0) or 0),
                "channels": int(channels or 0),
            }
        )
    return tuple(result)


def _fraction_text(value: Any) -> str | None:
    if value is None:
        return None
    fraction = Fraction(value)
    return f"{fraction.numerator}/{fraction.denominator}"


def preflight_capture_source(
    path: Path,
    *,
    cancel: CancelCheck = lambda: False,
    progress: Progress | None = None,
    thread_budget: int = 1,
) -> PreflightReport:
    path = _plain_local_source(path)
    initial_identity = source_identity(path, cancel, progress)
    timestamps: list[float] = []
    exact_timestamps: list[Fraction] = []
    rgb_digest = hashlib.sha256()
    rgb_bytes = 0
    baseline_dimensions: tuple[int, int] | None = None
    baseline_transform: DisplayTransform | None = None
    baseline_color: ColorDescription | None = None
    baseline_format: str | None = None
    decoded = 0
    last_pts: int | None = None
    try:
        with av.open(str(path), mode="r") as container:
            stream = select_video_stream(container)
            stream_index = int(stream.index)
            stream.codec_context.thread_count = max(1, int(thread_budget))
            stream.codec_context.thread_type = "SLICE"
            if stream.time_base is None:
                raise DecoderError("Selected video track has no time base")
            stream_time_base = Fraction(stream.time_base)
            audio = _audio_metadata(container)
            container_format = str(container.format.name)
            codec_name = str(stream.codec_context.name)
            nominal_rate = _fraction_text(getattr(stream, "average_rate", None))
            for frame in container.decode(stream):
                _cancelled(cancel)
                if bool(getattr(frame, "is_corrupt", False)):
                    raise DecoderError(f"Frame {decoded} is corrupt")
                if bool(frame.interlaced_frame):
                    raise DecoderError(f"Frame {decoded} is interlaced")
                if _has_alpha(frame):
                    raise DecoderError(f"Frame {decoded} has an alpha channel")
                if _sample_aspect_ratio(frame, stream) != Fraction(1, 1):
                    raise DecoderError(f"Frame {decoded} has non-square pixels")
                if frame.pts is None or frame.time_base is None:
                    raise DecoderError(f"Frame {decoded} has no presentation timestamp")
                exact_seconds = Fraction(int(frame.pts)) * Fraction(frame.time_base)
                if exact_timestamps and exact_seconds <= exact_timestamps[-1]:
                    raise DecoderError(f"Frame {decoded} timestamp is not strictly increasing")
                seconds = float(exact_seconds)
                if not math.isfinite(seconds):
                    raise DecoderError(f"Frame {decoded} timestamp is non-finite")
                current_transform = display_transform(frame, stream, baseline_transform)
                current_color = color_description(frame)
                current_dimensions = (int(frame.width), int(frame.height))
                current_format = str(frame.format.name)
                if baseline_dimensions is None:
                    baseline_dimensions = current_dimensions
                    baseline_transform = current_transform
                    baseline_color = current_color
                    baseline_format = current_format
                elif (
                    current_dimensions != baseline_dimensions
                    or current_transform != baseline_transform
                    or current_color != baseline_color
                    or current_format != baseline_format
                ):
                    raise DecoderError(f"Frame {decoded} changes dimensions, transform, color, or pixel format")
                rgb = apply_display_transform(
                    frame.to_ndarray(format="rgb24"), current_transform
                )
                rgb_digest.update(rgb.tobytes(order="C"))
                rgb_bytes += int(rgb.nbytes)
                timestamps.append(seconds)
                exact_timestamps.append(exact_seconds)
                last_pts = int(frame.pts)
                decoded += 1
                if progress:
                    progress("decode", decoded)
    except PreflightCancelled:
        raise
    except DecoderError:
        raise
    except av.error.FFmpegError as exc:
        raise DecoderError(
            f"Decode failed at frame {decoded}, pts {last_pts}: {exc}"
        ) from exc
    if decoded < MINIMUM_FRAMES:
        raise DecoderError(
            f"Capture Source has {decoded} decodable frames; at least {MINIMUM_FRAMES} are required"
        )
    if baseline_dimensions is None or baseline_transform is None or baseline_color is None or baseline_format is None:
        raise DecoderError("Capture Source produced no video frames")
    require_same_source(path, initial_identity, cancel)
    deltas = [right - left for left, right in zip(exact_timestamps, exact_timestamps[1:])]
    variable = bool(deltas and any(delta != deltas[0] for delta in deltas[1:]))
    coded_width, coded_height = baseline_dimensions
    displayed_width, displayed_height = (
        (coded_height, coded_width)
        if baseline_transform.swaps_dimensions
        else (coded_width, coded_height)
    )
    libraries = tuple(
        sorted((str(name), tuple(int(item) for item in version)) for name, version in av.library_versions.items())
    )
    return PreflightReport(
        initial_identity, container_format, stream_index, codec_name,
        coded_width, coded_height, displayed_width, displayed_height,
        baseline_transform, baseline_color, baseline_format,
        (stream_time_base.numerator, stream_time_base.denominator), nominal_rate,
        tuple(timestamps), variable, rgb_digest.hexdigest(), rgb_bytes,
        audio, str(av.__version__), libraries,
    )
