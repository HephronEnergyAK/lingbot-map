"""Deterministic CPU Worker Job that publishes a contract-complete Result."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np

from . import __version__
from .fixture_job import (
    FixtureJobError,
    StatusStore,
    _heartbeat_loop,
    _install_audit_policy,
    _publish_worker_identity,
    _set_below_normal_priority,
    _terminal_destination,
    _wait_for_control,
    _windows_execution_state,
    validate_job_spec,
)
from .gpu_lease import sha256_file
from .ipc import read_json
from .result_bundle import ResultCancelled
from .result_pipeline import (
    AlignedPrediction,
    ResultBuildRequest,
    ResultProfile,
    build_reconstruction_result,
)
from .result_resources import FixedResourceProbe
from .provenance import result_provenance


_FIXTURE_MODEL_SHA256 = hashlib.sha256(b"lingbot-map-result-fixture-model-1.0.0").hexdigest()


def _prediction(index: int, camera_x: float) -> AlignedPrediction:
    world_to_camera = np.eye(4, dtype="<f8")
    world_to_camera[0, 3] = -camera_x
    depth = np.array(
        ((1.0, 1.5, 2.0), (2.5, np.nan, -1.0), (3.0, 3.5, 20.0)),
        dtype="<f4",
    )
    confidence = np.array(
        ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)),
        dtype="<f4",
    )
    rgb = np.arange(27, dtype="|u1").reshape((3, 3, 3)) + index
    return AlignedPrediction(
        index,
        index,
        index * 0.04,
        world_to_camera,
        np.array(((3.0, 0.0, 1.0), (0.0, 3.0, 1.0), (0.0, 0.0, 1.0)), dtype="<f8"),
        depth,
        confidence,
        np.ascontiguousarray(rgb),
    )


def _source(fixture: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(os.path.abspath(fixture["absolute_path"]))
    before = path.stat()
    if (
        not path.is_file()
        or path.is_symlink()
        or before.st_size != fixture["size_bytes"]
        or before.st_mtime_ns != fixture["modification_time_ns"]
    ):
        raise FixtureJobError("Result Fixture source identity changed before execution")
    digest = sha256_file(path)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise FixtureJobError("Result Fixture source changed while hashing")
    return (
        {
            "absolute_path": str(path),
            "scene_relative_path": fixture["scene_relative_path"],
            "size_bytes": before.st_size,
            "modification_time_ns": before.st_mtime_ns,
            "sha256": digest,
        },
        digest,
    )


def run_result_fixture_job(spec_path: Path, nonce: str) -> int:
    spec_path = Path(os.path.abspath(spec_path))
    if not spec_path.is_absolute() or spec_path.name != "job-spec.json":
        raise FixtureJobError("Result Fixture JobSpec must be an absolute job-spec.json path")
    job_dir = spec_path.parent
    job = validate_job_spec(read_json(spec_path))
    if "result_fixture" not in job:
        raise FixtureJobError("Result Fixture runner requires a result_fixture JobSpec")
    if job_dir.name != job["job_id"] or job_dir.parent.name != ".jobs":
        raise FixtureJobError("Result Fixture JobSpec is outside its bounded active directory")
    if Path(os.path.abspath(job["project_root"])) != job_dir.parent.parent:
        raise FixtureJobError("Result Fixture Project Result Root does not contain this Job")
    _publish_worker_identity(job_dir, nonce)
    control = _wait_for_control(job_dir, job, nonce)
    fixture = job["result_fixture"]
    store = StatusStore(job_dir, job["job_id"], 3)
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(store, stop, float(fixture["heartbeat_interval_seconds"]), None),
        name="LingBotMap-Heartbeat",
        daemon=True,
    )
    state, message, error, return_code = "failed", "Result Fixture failed", None, 1
    try:
        _set_below_normal_priority()
        _install_audit_policy()
        heartbeat.start()
        with _windows_execution_state():
            store.emit("phase", "result-fixture", 0, "Result Fixture filtering started")
            source, source_sha256 = _source(fixture)
            store.emit("progress", "result-fixture", 1, "Result Fixture source identity frozen")
            request = ResultBuildRequest(
                job_id=job["job_id"],
                project_root=Path(job["project_root"]),
                target_scene=job["target_scene"],
                timeline_start=job["timeline_start"],
                source=source,
                source_to_model=np.eye(3, dtype="<f8"),
                predictions=(_prediction(0, 0.0), _prediction(1, 0.25)),
                profile=ResultProfile(
                    "Fixture",
                    float(fixture["confidence_cutoff_percent"]),
                    float(fixture["depth_cutoff_percent"]),
                    int(fixture["import_point_budget"]),
                    float(fixture["initial_voxel_edge_length"]),
                ),
                provenance=result_provenance(
                    runtime_id=control["runtime_id"],
                    worker_version=__version__,
                    job_spec_sha256=control["job_spec"]["sha256"],
                    model_id="result-fixture-model",
                    model_sha256=_FIXTURE_MODEL_SHA256,
                    source_sha256=source_sha256,
                    profile_name="Fixture",
                    camera_iterations=1,
                    confidence_cutoff_percent=float(fixture["confidence_cutoff_percent"]),
                    depth_cutoff_percent=float(fixture["depth_cutoff_percent"]),
                    import_point_budget=int(fixture["import_point_budget"]),
                    plan=None,
                    gpu=None,
                    suspension_count=0,
                    suspension_seconds=0,
                    fixture=True,
                ),
                warnings=({"code": "fixture", "message": "Deterministic CPU acceptance fixture"},),
                created_utc=datetime.now(timezone.utc).isoformat(),
            )
            store.emit("progress", "result-fixture", 2, "Result Fixture predictions aligned")
            outcome = build_reconstruction_result(
                request,
                resource_probe=FixedResourceProbe(20 * 1024**3, 20 * 1024**3),
                cancel=lambda: (job_dir / "cancel.request").exists(),
            )
            store.emit(
                "progress", "result-fixture", 3,
                f"Ready Result {outcome.published.result_id} published",
            )
            state, message, return_code = "succeeded", "Result Fixture produced a Ready Result", 0
    except ResultCancelled:
        state, message, return_code = "cancelled", "Result Fixture cancelled before commit", 2
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:16384]
    finally:
        stop.set()
        if heartbeat.is_alive():
            heartbeat.join(timeout=6)
        store.terminal(state, message, error=error)
    destination = _terminal_destination(job, state).with_name(
        f"{job['job_id']}--result-fixture-{state}"
    )
    if destination.exists():
        raise FixtureJobError(f"terminal diagnostics already exist: {destination.name}")
    os.replace(job_dir, destination)
    return return_code
