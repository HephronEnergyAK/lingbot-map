"""Build deterministic, non-personal release-corpus media in a scratch root."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

try:
    import av
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "The pinned release-corpus environment requires PyAV and NumPy"
    ) from exc

from lingbot_map_worker.decoder import preflight_capture_source
from lingbot_map_worker.gpu_profiles import inference_plan


GENERATOR_ID = "lingbot-release-corpus-media-v1"
WIDTH = 32
HEIGHT = 24
BOUNDARY_COUNTS = (8, 320, 321, 3000, 3001)
STRESS_COUNT = 25_000
VFR_PTS = (0, 33, 67, 101, 151, 184, 250, 284)
COLOR_CASES = {
    "bt709-limited": (1, 1, 1, 1),
    "bt709-full": (1, 1, 1, 2),
    "bt601-limited": (5, 6, 6, 1),
    "bt601-full": (6, 5, 5, 2),
}


def _is_linked(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _synthetic_rgb(index: int) -> np.ndarray:
    x = np.arange(WIDTH, dtype=np.uint16)[None, :]
    y = np.arange(HEIGHT, dtype=np.uint16)[:, None]
    phase = index % 256
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 0] = ((x * 7 + phase * 3) % 256).astype(np.uint8)
    image[:, :, 1] = ((y * 11 + phase * 5) % 256).astype(np.uint8)
    image[:, :, 2] = (
        (x * 13 + y * 17 + phase * 19) % 256
    ).astype(np.uint8)
    identity = int(index).to_bytes(4, "little", signed=False)
    image[0, :4] = np.frombuffer(identity * 3, dtype=np.uint8).reshape(
        4, 3
    )
    image[-4:, -4:] = (
        (index * 29) % 256,
        (index * 31) % 256,
        (index * 37) % 256,
    )
    return image


def _encode_video(
    path: Path,
    *,
    frames: Iterable[np.ndarray],
    frame_count: int,
    time_base: Fraction,
    pts_values: Iterable[int],
    color: tuple[int, int, int, int],
    rate: int = 30,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(
        str(path),
        "w",
        format="mp4",
        options={"movflags": "+faststart"},
    )
    try:
        container.metadata.clear()
        stream = container.add_stream(
            "libx264",
            rate=rate,
            options={
                "bf": "0",
                "crf": "0",
                "g": "30",
                "preset": "ultrafast",
                "sc_threshold": "0",
            },
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.thread_count = 1
        (
            stream.codec_context.color_primaries,
            stream.codec_context.color_trc,
            stream.codec_context.colorspace,
            stream.codec_context.color_range,
        ) = color
        count = 0
        for array, pts in zip(frames, pts_values):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.uint8
                or array.shape != (height, width, 3)
            ):
                raise ValueError("corpus frame violates the fixed RGB contract")
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = int(pts)
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        if count != frame_count:
            raise ValueError(
                f"expected {frame_count} frames but encoded {count}"
            )
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _write_synthetic(
    path: Path,
    *,
    frame_count: int,
    color: tuple[int, int, int, int] = COLOR_CASES[
        "bt709-limited"
    ],
    vfr: bool = False,
) -> None:
    if vfr:
        if frame_count != len(VFR_PTS):
            raise ValueError("the VFR oracle has exactly eight frames")
        time_base = Fraction(1, 1000)
        pts_values: Iterable[int] = VFR_PTS
    else:
        time_base = Fraction(1, 30)
        pts_values = range(frame_count)
    _encode_video(
        path,
        frames=(_synthetic_rgb(index) for index in range(frame_count)),
        frame_count=frame_count,
        time_base=time_base,
        pts_values=pts_values,
        color=color,
    )


def _ordered_source_manifest_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_courthouse(path: Path) -> dict[str, object]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The pinned release-corpus environment requires Pillow"
        ) from exc
    sources = sorted((ROOT / "example" / "courthouse").glob("*.png"))
    if len(sources) != 286:
        raise RuntimeError(
            "The pinned courthouse source must contain exactly 286 PNGs"
        )

    def arrays():
        for source in sources:
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (294, 518, 3):
                raise RuntimeError(
                    f"Courthouse frame shape drifted: {source.name}"
                )
            yield np.ascontiguousarray(rgb)

    _encode_video(
        path,
        frames=arrays(),
        frame_count=len(sources),
        time_base=Fraction(1, 30),
        pts_values=range(len(sources)),
        color=COLOR_CASES["bt709-limited"],
        width=518,
        height=294,
    )
    return {
        "source_files": len(sources),
        "source_manifest_sha256": _ordered_source_manifest_hash(sources),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamps_sha256(values: tuple[float, ...]) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _record(
    path: Path,
    *,
    fixture_id: str,
    kind: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    report = preflight_capture_source(path)
    frame_count = len(report.timestamps_seconds)
    plan = inference_plan(frame_count)
    result: dict[str, object] = {
        "id": fixture_id,
        "kind": kind,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "frame_count": frame_count,
        "mode": plan.mode,
        "keyframe_interval": plan.keyframe_interval,
        "video_codec": report.video_codec,
        "coded_dimensions": [
            report.coded_width,
            report.coded_height,
        ],
        "displayed_dimensions": [
            report.displayed_width,
            report.displayed_height,
        ],
        "display_transform": report.display_transform.name,
        "color_standard": report.color.standard,
        "color_range": report.color.range,
        "variable_frame_rate": report.variable_frame_rate,
        "timestamps_sha256": _timestamps_sha256(
            report.timestamps_seconds
        ),
        "decoded_rgb_sha256": report.rgb_sha256,
    }
    if extra:
        result.update(extra)
    return result


def _selected(
    fixture_id: str,
    only: frozenset[str],
) -> bool:
    return not only or fixture_id in only


def build(
    output: Path,
    *,
    include_stress: bool,
    include_real: bool,
    only: frozenset[str] = frozenset(),
) -> dict[str, object]:
    destination = Path(os.path.abspath(output))
    if destination.exists():
        if (
            _is_linked(destination)
            or not destination.is_dir()
            or any(destination.iterdir())
        ):
            raise RuntimeError(
                "release-corpus output must be empty and plain"
            )
    else:
        destination.mkdir(parents=True)
    records = []
    for count in BOUNDARY_COUNTS:
        fixture_id = f"synthetic-boundary-{count}"
        if not _selected(fixture_id, only):
            continue
        path = destination / f"{fixture_id}.mp4"
        _write_synthetic(path, frame_count=count)
        records.append(
            _record(
                path,
                fixture_id=fixture_id,
                kind="synthetic-boundary",
            )
        )
    fixture_id = "synthetic-vfr-8"
    if _selected(fixture_id, only):
        path = destination / f"{fixture_id}.mp4"
        _write_synthetic(path, frame_count=8, vfr=True)
        records.append(
            _record(
                path,
                fixture_id=fixture_id,
                kind="synthetic-vfr",
                extra={"pts_milliseconds": list(VFR_PTS)},
            )
        )
    for name, color in COLOR_CASES.items():
        fixture_id = f"synthetic-color-{name}"
        if not _selected(fixture_id, only):
            continue
        path = destination / f"{fixture_id}.mp4"
        _write_synthetic(path, frame_count=8, color=color)
        records.append(
            _record(
                path,
                fixture_id=fixture_id,
                kind="synthetic-color",
            )
        )
    fixture_id = "real-courthouse-streaming-286"
    if include_real and _selected(fixture_id, only):
        path = destination / f"{fixture_id}.mp4"
        provenance = _write_courthouse(path)
        records.append(
            _record(
                path,
                fixture_id=fixture_id,
                kind="real-streaming",
                extra=provenance,
            )
        )
    fixture_id = "synthetic-stress-25000"
    if include_stress and _selected(fixture_id, only):
        path = destination / f"{fixture_id}.mp4"
        _write_synthetic(path, frame_count=STRESS_COUNT)
        records.append(
            _record(
                path,
                fixture_id=fixture_id,
                kind="synthetic-stress",
            )
        )
    unknown = sorted(
        only - {str(record["id"]) for record in records}
    )
    if unknown:
        raise RuntimeError(
            "Unknown or disabled fixture selection: " + repr(unknown)
        )
    document = {
        "schema_version": "1.0.0",
        "generator_id": GENERATOR_ID,
        "toolchain": {
            "python": ".".join(
                str(value) for value in sys.version_info[:3]
            ),
            "pyav": av.__version__,
            "numpy": np.__version__,
            "ffmpeg_libraries": {
                name: list(version)
                for name, version in sorted(av.library_versions.items())
            },
        },
        "stress_included": include_stress,
        "personal_capture_sources": False,
        "fixtures": records,
    }
    (destination / "generated-manifest.json").write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Generate the 25,000-frame milestone-only fixture",
    )
    parser.add_argument(
        "--exclude-real",
        action="store_true",
        help="Skip the repository-owned courthouse capture",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="FIXTURE_ID",
    )
    arguments = parser.parse_args(argv)
    document = build(
        arguments.output,
        include_stress=arguments.include_stress,
        include_real=not arguments.exclude_real,
        only=frozenset(arguments.only),
    )
    print(
        "LINGBOT_MAP_RELEASE_CORPUS_BUILD="
        + json.dumps(
            {
                "fixtures": len(document["fixtures"]),
                "stress_included": document["stress_included"],
                "toolchain": document["toolchain"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
