"""Finite Capture Source preflight Job using the qualified lifecycle protocol."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import struct
import threading

from .decoder import DecoderError, PreflightCancelled, preflight_capture_source, require_same_source
from .fixture_job import (
    FixtureJobError,
    StatusStore,
    _heartbeat_loop,
    _install_audit_policy,
    _publish_worker_identity,
    _set_below_normal_priority,
    _wait_for_control,
    _windows_execution_state,
    validate_job_spec,
)
from .ipc import SCHEMA_VERSION, atomic_write_json, read_json


class PreflightJobError(FixtureJobError):
    pass


def _cancelled(job_dir: Path) -> bool:
    return (job_dir / "cancel.request").exists()


def _write_timestamps(path: Path, timestamps: tuple[float, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    length = 0
    with path.open("xb") as stream:
        for timestamp in timestamps:
            encoded = struct.pack("<d", timestamp)
            stream.write(encoded)
            digest.update(encoded)
            length += len(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest(), length


def _result_document(job: dict, report, timestamp_sha: str, timestamp_length: int) -> dict:
    capture = job["capture_source"]
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "source": {
            **capture,
            "sha256": report.identity.sha256,
            "container_format": report.container_format,
        },
        "decoder": {
            "adapter": "pyav",
            "pyav_version": report.pyav_version,
            "color_conversion": "pyav-video-frame-to-ndarray-rgb24",
            "output_color": "srgb-8-bit",
            "ffmpeg_libraries": {
                name: list(version) for name, version in report.ffmpeg_libraries
            },
        },
        "video": {
            "stream_index": report.video_stream_index,
            "codec": report.video_codec,
            "coded_width": report.coded_width,
            "coded_height": report.coded_height,
            "displayed_width": report.displayed_width,
            "displayed_height": report.displayed_height,
            "pixel_format": report.pixel_format,
            "display_transform": report.display_transform.name,
            "color": asdict(report.color),
        },
        "timing": {
            "frame_count": len(report.timestamps_seconds),
            "stream_time_base": list(report.stream_time_base),
            "nominal_frame_rate": report.nominal_frame_rate,
            "variable_frame_rate": report.variable_frame_rate,
        },
        "audio_streams": list(report.audio_streams),
        "canonical_rgb": {
            "path": None,
            "sha256": report.rgb_sha256,
            "length": report.rgb_bytes,
        },
        "timestamps": {
            "path": "timestamps.f64le",
            "sha256": timestamp_sha,
            "length": timestamp_length,
        },
    }


def run_preflight_job(spec_path: Path, nonce: str) -> int:
    spec_path = Path(os.path.abspath(spec_path))
    if not spec_path.is_absolute() or spec_path.name != "job-spec.json":
        raise PreflightJobError("Preflight JobSpec must be an absolute job-spec.json path")
    job_dir = spec_path.parent
    job = validate_job_spec(read_json(spec_path))
    if "capture_source" not in job:
        raise PreflightJobError("Preflight runner requires a capture_source JobSpec")
    if job_dir.name != job["job_id"] or job_dir.parent.name != ".jobs":
        raise PreflightJobError("Preflight JobSpec is outside its bounded active directory")
    if Path(os.path.abspath(job["project_root"])) != job_dir.parent.parent:
        raise PreflightJobError("Preflight JobSpec Project Result Root does not contain this Job")

    capture = job["capture_source"]
    source = Path(capture["absolute_path"])
    _publish_worker_identity(job_dir, nonce)
    _wait_for_control(job_dir, job, nonce)
    store = StatusStore(job_dir, job["job_id"], int(capture["size_bytes"]))
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(store, stop, 1.0, None),
        name="LingBotMap-Heartbeat",
        daemon=True,
    )
    state, message, error, return_code = "failed", "Capture Source preflight failed", None, 1
    try:
        _set_below_normal_priority()
        _install_audit_policy()
        heartbeat.start()
        with _windows_execution_state():
            store.emit("phase", "preflight", 0, "Capture Source preflight started")
            try:
                initial_stat = source.stat()
            except OSError as exc:
                raise PreflightJobError(f"Capture Source cannot be examined: {exc}") from exc
            if (
                initial_stat.st_size != capture["size_bytes"]
                or initial_stat.st_mtime_ns != capture["modification_time_ns"]
            ):
                raise PreflightJobError("Capture Source changed after the Job Draft was frozen")

            def progress(phase: str, completed: int) -> None:
                bounded = (
                    min(int(capture["size_bytes"]), completed)
                    if phase == "hash"
                    else int(capture["size_bytes"])
                )
                store.emit("progress", phase, bounded, f"{phase} progress {completed}")

            report = preflight_capture_source(
                source,
                cancel=lambda: _cancelled(job_dir),
                progress=progress,
                thread_budget=1,
            )
            timestamp_sha, timestamp_length = _write_timestamps(
                job_dir / "timestamps.f64le", report.timestamps_seconds
            )
            require_same_source(source, report.identity, lambda: _cancelled(job_dir))
            atomic_write_json(
                job_dir / "preflight-result.json",
                _result_document(job, report, timestamp_sha, timestamp_length),
            )
            state, message, return_code = "succeeded", "Capture Source preflight completed", 0
    except PreflightCancelled:
        state, message, return_code = "cancelled", "Capture Source preflight cancelled", 2
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:16384]
    finally:
        stop.set()
        if heartbeat.is_alive():
            heartbeat.join(timeout=6)
        store.terminal(state, message, error=error)
    destination = Path(job["project_root"]) / "diagnostics" / f"{job['job_id']}--preflight-{state}"
    if destination.exists():
        raise PreflightJobError(f"terminal diagnostics already exist: {destination.name}")
    os.replace(job_dir, destination)
    return return_code
