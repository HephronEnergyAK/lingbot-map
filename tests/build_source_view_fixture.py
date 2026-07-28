"""Build eight deterministic Result/source-view fixtures with the real Worker."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

from lingbot_map_worker.canonical_preprocessing import canonical_geometry
from lingbot_map_worker.provenance import result_provenance
from lingbot_map_worker.result_bundle import (
    ResultPublication,
    publish_result_bundle,
    validate_result_bundle,
)


TRANSFORMS = (
    "identity",
    "rotate_90_ccw",
    "rotate_180",
    "rotate_270_ccw",
    "reflect_x",
    "reflect_y",
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
)
SWAPS = {
    "rotate_90_ccw",
    "rotate_270_ccw",
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
}
CODED_WIDTH = 64
CODED_HEIGHT = 48
FRAME_COUNT = 3
TIMELINE_START = 17
SCENE_UUID = "6a2e2d1f-f8be-4ceb-b19c-9e46595a5bb3"


def _write_capture(path: Path) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream(
        "libx264", rate=25, options={"crf": "0", "preset": "ultrafast"}
    )
    stream.width = CODED_WIDTH
    stream.height = CODED_HEIGHT
    stream.pix_fmt = "yuv420p"
    colors = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 80, 255),
        (255, 255, 0),
    )
    for frame_index in range(FRAME_COUNT):
        image = np.zeros((CODED_HEIGHT, CODED_WIDTH, 3), dtype=np.uint8)
        image[:8, :8] = colors[0]
        image[:8, -8:] = colors[1]
        image[-8:, :8] = colors[2]
        image[-8:, -8:] = colors[3]
        x = 16 + frame_index * 12
        image[12:36, x : x + 4] = (255, 255, 255)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = frame_index
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _arrays(
    displayed_width: int,
    displayed_height: int,
    source_to_model: np.ndarray,
) -> dict[str, np.ndarray]:
    camera = np.repeat(
        np.array(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype="<f4",
        )[None, :, :],
        FRAME_COUNT,
        axis=0,
    )
    camera[1, 0, 3] = np.float32(0.25)
    camera[2, 0, 3] = np.float32(0.5)
    source_intrinsics = np.zeros((FRAME_COUNT, 3, 3), dtype="<f4")
    focal = np.asarray(
        (
            displayed_width * 0.80,
            displayed_width * 0.90,
            displayed_width * 0.75,
        ),
        dtype=np.float32,
    )
    source_intrinsics[:, 0, 0] = focal
    source_intrinsics[:, 1, 1] = focal
    source_intrinsics[0, 1, 1] = focal[0] * np.float32(1.10)
    source_intrinsics[:, 0, 2] = (
        displayed_width * 0.5,
        displayed_width * 0.5 + 1.0,
        displayed_width * 0.5 - 1.0,
    )
    source_intrinsics[:, 1, 2] = displayed_height * 0.5
    source_intrinsics[:, 2, 2] = 1.0
    model_intrinsics = np.einsum(
        "ij,fjk->fik",
        source_to_model,
        source_intrinsics.astype(np.float64),
    ).astype("<f4")
    return {
        "positions": np.ascontiguousarray(
            ((-0.5, 2.0, -0.5), (0.0, 2.5, 0.0), (0.5, 3.0, 0.5)),
            dtype="<f4",
        ),
        "colors": np.ascontiguousarray(
            ((255, 0, 0), (0, 255, 0), (0, 80, 255)), dtype="|u1"
        ),
        "confidence": np.ascontiguousarray((0.25, 0.5, 0.9), dtype="<f4"),
        "radius": np.ascontiguousarray((0.05, 0.05, 0.05), dtype="<f4"),
        "source_frame": np.ascontiguousarray((0, 1, 2), dtype="<u4"),
        "camera_to_world": np.ascontiguousarray(camera, dtype="<f4"),
        "model_intrinsics": np.ascontiguousarray(
            model_intrinsics, dtype="<f4"
        ),
        "source_intrinsics": np.ascontiguousarray(
            source_intrinsics, dtype="<f4"
        ),
        "model_fov_radians": np.ascontiguousarray(
            ((1.0, 0.8), (0.9, 0.7), (1.1, 0.9)), dtype="<f4"
        ),
        "source_pts_seconds": np.ascontiguousarray(
            (0.0, 0.04, 0.08), dtype="<f8"
        ),
        "source_to_model": np.ascontiguousarray(
            source_to_model, dtype="<f8"
        ),
        "frame_type": np.ascontiguousarray((0, 1, 2), dtype="|u1"),
    }


def build(output: Path) -> dict[str, object]:
    output = output.resolve()
    if output.exists():
        raise RuntimeError("fixture output must not already exist")
    output.mkdir(parents=True)
    capture = output / "capture.mp4"
    _write_capture(capture)
    source_sha = hashlib.sha256(capture.read_bytes()).hexdigest()
    blend = output / "source-view.blend"
    projects = output / "projects"
    index: dict[str, str] = {}
    for ordinal, transform in enumerate(TRANSFORMS, 1):
        project = projects / transform / "source-view.lingbot-map"
        (project / "results").mkdir(parents=True)
        (project / "diagnostics").mkdir()
        if transform in SWAPS:
            width, height = CODED_HEIGHT, CODED_WIDTH
        else:
            width, height = CODED_WIDTH, CODED_HEIGHT
        model_height, model_width, source_to_model, coverage = (
            canonical_geometry(height, width)
        )
        source_fraction = (
            (coverage[1][0] - coverage[0][0])
            * (coverage[2][1] - coverage[1][1])
            / (width * height)
        )
        job_id = f"job-{ordinal:032x}"
        provenance = result_provenance(
            runtime_id="2" * 64,
            worker_version="0.1.0",
            job_spec_sha256=f"{ordinal:x}" * 64,
            model_id="source-view-fixture",
            model_sha256="4" * 64,
            source_sha256=source_sha,
            profile_name="Source View Fixture",
            camera_iterations=1,
            confidence_cutoff_percent=0.0,
            depth_cutoff_percent=100.0,
            import_point_budget=3,
            plan=None,
            gpu=None,
            suspension_count=0,
            suspension_seconds=0,
            fixture=True,
        )
        publication = ResultPublication(
            job_id=job_id,
            project_root=project,
            target_scene={
                "blend_path": str(blend),
                "scene_uuid": SCENE_UUID,
                "scene_name": "Scene",
            },
            timeline_start=TIMELINE_START,
            source={
                "absolute_path": str(capture),
                "scene_relative_path": "//capture.mp4",
                "size_bytes": capture.stat().st_size,
                "modification_time_ns": capture.stat().st_mtime_ns,
                "sha256": source_sha,
            },
            profile={
                "name": "Source View Fixture",
                "confidence_cutoff_percent": 0.0,
                "depth_cutoff_percent": 100.0,
                "import_point_budget": 3,
                "retain_dense_predictions": False,
            },
            provenance=provenance,
            warnings=(
                {
                    "code": "source-intrinsics-axis-disagreement",
                    "message": (
                        "1 frame differs by more than 5% between source-display "
                        "fx and fy; exact horizontal mapping retained."
                    ),
                },
                {
                    "code": "source-focal-discontinuity",
                    "message": (
                        "Adjacent source-display focal jumps exceed 5%; "
                        "per-frame values retained without smoothing."
                    ),
                },
            ),
            arrays=_arrays(width, height, source_to_model),
            voxel_edge_length=0.1,
            voxel_origin=(0.0, 0.0, 0.0),
            source_display={
                "width": width,
                "height": height,
                "display_transform": transform,
            },
            model_coverage={
                "coordinate_space": "source-display-pixel-edges",
                "polygon": [[float(x), float(y)] for x, y in coverage],
                "source_fraction": float(source_fraction),
                "model_width": model_width,
                "model_height": model_height,
            },
            created_utc=(
                datetime(2026, 7, 23, ordinal, tzinfo=timezone.utc).isoformat()
            ),
            result_id=f"result-{ordinal:032x}",
        )
        result = publish_result_bundle(
            publication,
            cancel=lambda: False,
            disk_check=lambda _remaining: None,
            estimated_total_bytes=1024 * 1024,
        )
        validate_result_bundle(result.directory)
        index[transform] = str(result.directory)
    document = {
        "capture": str(capture),
        "capture_sha256": source_sha,
        "blend": str(blend),
        "scene_uuid": SCENE_UUID,
        "timeline_start": TIMELINE_START,
        "frame_count": FRAME_COUNT,
        "coded_size": [CODED_WIDTH, CODED_HEIGHT],
        "results": index,
    }
    (output / "index.json").write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )
    return document


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_source_view_fixture.py OUTPUT")
    print(
        "LINGBOT_MAP_SOURCE_VIEW_FIXTURE="
        + json.dumps(build(Path(sys.argv[1])), sort_keys=True)
    )
