"""Real installed-Runtime proof for optional Dense Predictions publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _worker(project: Path) -> int:
    import numpy as np

    from lingbot_map_worker.dense_predictions import validate_dense_component
    from lingbot_map_worker.gpu_profiles import inference_plan
    from lingbot_map_worker.provenance import result_provenance
    from lingbot_map_worker.result_bundle import validate_result_bundle
    from lingbot_map_worker.result_pipeline import (
        AlignedPrediction,
        IncrementalBundleResultSink,
        IncrementalResultRequest,
        ResultProfile,
    )
    from lingbot_map_worker.result_resources import SystemResourceProbe

    project.mkdir()
    (project / "results").mkdir()
    (project / "diagnostics").mkdir()
    root = project.parent
    blend = root / "target.blend"
    blend.touch()
    source = root / "capture.mp4"
    source.write_bytes(b"installed-runtime-dense-fixture")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = result_provenance(
        runtime_id="2" * 64,
        worker_version="0.1.0",
        job_spec_sha256="3" * 64,
        model_id="fixture-model",
        model_sha256="4" * 64,
        source_sha256=source_sha,
        profile_name="Custom",
        camera_iterations=1,
        confidence_cutoff_percent=70,
        depth_cutoff_percent=99.5,
        import_point_budget=8,
        plan=inference_plan(65),
        gpu=None,
        suspension_count=0,
        suspension_seconds=0,
        fixture=True,
        retain_dense_predictions=True,
    )
    request = IncrementalResultRequest(
        job_id="job-" + "d" * 32,
        project_root=project,
        target_scene={
            "blend_path": str(blend),
            "scene_uuid": "12345678-1234-4321-8765-123456789abc",
            "scene_name": "Dense Fixture Scene",
        },
        timeline_start=1,
        source={
            "absolute_path": str(source),
            "scene_relative_path": "//capture.mp4",
            "size_bytes": source.stat().st_size,
            "modification_time_ns": source.stat().st_mtime_ns,
            "sha256": source_sha,
        },
        source_to_model=np.eye(3, dtype="<f8"),
        frame_count=65,
        profile=ResultProfile("Custom", 70, 99.5, 8, 0.01, True),
        provenance=provenance,
        created_utc="2026-07-23T06:30:00+00:00",
        result_id="result-" + "e" * 32,
        model_grid_shape=(2, 2),
    )
    sink = IncrementalBundleResultSink(
        request,
        prediction_decoder=lambda prediction, _canonical, _pts: prediction,
        resource_probe=SystemResourceProbe(),
        cancel=lambda: False,
    )
    intrinsics = np.array(
        ((2.0, 0.0, 0.5), (0.0, 2.0, 0.5), (0.0, 0.0, 1.0)),
        dtype="<f8",
    )
    for index in range(65):
        sink.accept(
            AlignedPrediction(
                index,
                1 if index % 8 == 0 else 2,
                index / 25.0,
                np.eye(4, dtype="<f8"),
                intrinsics,
                np.full((2, 2), index + 1, dtype="<f4"),
                np.full((2, 2), index + 2, dtype="<f4"),
                np.zeros((2, 2, 3), dtype="|u1"),
            ),
            None,
            index / 25.0,
        )
    outcome = sink.finish()
    result = outcome.published.directory
    core = validate_result_bundle(result)
    dense = validate_dense_component(result, core["dense_predictions"])
    print(
        json.dumps(
            {
                "result_id": core["result_id"],
                "frames": core["counts"]["frames"],
                "dense_state": dense["completion_state"],
                "depth_chunks": [
                    chunk["frame_count"]
                    for chunk in dense["signals"]["depth"]["chunks"]
                ],
                "confidence_chunks": [
                    chunk["frame_count"]
                    for chunk in dense["signals"]["depth_confidence"]["chunks"]
                ],
                "dtype": dense["signals"]["depth"]["dtype"],
                "full_validation": True,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return _worker(Path(sys.argv[2]))
    if len(sys.argv) != 2 or os.name != "nt":
        raise SystemExit("usage: dense_predictions_windows_integration.py RUNTIME")
    runtime = Path(sys.argv[1])
    python = runtime / ".venv" / "Scripts" / "python.exe"
    with tempfile.TemporaryDirectory(prefix="lingbot-map-dense11-") as temporary:
        project = Path(temporary) / "target.lingbot-map"
        completed = subprocess.run(
            [str(python), "-I", str(Path(__file__).resolve()), "--worker", str(project)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        evidence = json.loads(completed.stdout)
        assert evidence["frames"] == 65
        assert evidence["depth_chunks"] == [64, 1]
        assert evidence["confidence_chunks"] == [64, 1]
        assert evidence["dtype"] == "<f4"
        assert evidence["full_validation"] is True
        print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
