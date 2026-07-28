"""Run the bundled Worker wheel through a cancelled native Windows Job."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "lingbot_worker_diagnostics_export",
    ROOT / "blender_extension" / "diagnostics.py",
)
diagnostics_module = importlib.util.module_from_spec(DIAGNOSTICS_SPEC)
sys.modules[DIAGNOSTICS_SPEC.name] = diagnostics_module
assert DIAGNOSTICS_SPEC.loader is not None
DIAGNOSTICS_SPEC.loader.exec_module(diagnostics_module)


def _write_json(path: Path, document) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _wait(path: Path, timeout: float = 15.0) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return path
        time.sleep(0.025)
    raise AssertionError(f"Timed out waiting for {path}")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: worker_diagnostics_windows_integration.py WORKER_PYTHON"
        )
    if os.name != "nt":
        raise SystemExit("Worker diagnostics integration requires Windows")
    worker_python = Path(sys.argv[1])
    if not worker_python.is_file():
        raise SystemExit("Worker Python does not exist")

    with tempfile.TemporaryDirectory(
        prefix="lingbot-map-worker-diagnostics18-"
    ) as temporary:
        workspace = Path(temporary)
        project = workspace / "target.lingbot-map"
        jobs = project / ".jobs"
        diagnostics = project / "diagnostics"
        results = project / "results"
        for directory in (jobs, diagnostics, results):
            directory.mkdir(parents=True, exist_ok=True)
        trusted_cwd = workspace / "empty-cwd"
        trusted_cwd.mkdir()
        blend = workspace / "target.blend"
        blend.write_bytes(b"BLENDER")
        job_id = f"job-{uuid.uuid4().hex}"
        nonce = uuid.uuid4().hex
        job_dir = jobs / job_id
        job_dir.mkdir()
        spec = {
            "schema_version": "1.0.0",
            "job_id": job_id,
            "target_scene": {
                "blend_path": str(blend),
                "scene_uuid": str(uuid.uuid4()),
                "scene_name": "Diagnostics Fixture",
            },
            "timeline_start": 1,
            "project_root": str(project),
            "fixture": {
                "steps": 100,
                "step_delay_seconds": 0.02,
                "ignore_cancel": False,
                "heartbeat_interval_seconds": 0.1,
                "freeze_heartbeat_after_sequence": None,
            },
        }
        spec_path = job_dir / "job-spec.json"
        _write_json(spec_path, spec)
        process = subprocess.Popen(
            [
                str(worker_python),
                "-I",
                "-m",
                "lingbot_map_worker",
                "--fixture-job",
                str(spec_path),
                "--job-nonce",
                nonce,
            ],
            cwd=trusted_cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        worker_record_path = _wait(job_dir / "worker.pid.json")
        worker_record = json.loads(worker_record_path.read_text("utf-8"))
        # Windows venv launchers may create the exact base-Python Worker as a
        # child, so the Worker's self-published PID is authoritative.
        assert isinstance(worker_record["pid"], int)
        assert worker_record["pid"] > 0
        assert worker_record["nonce"] == nonce
        _write_json(
            job_dir / "job-control.json",
            {
                "schema_version": "1.0.0",
                "job_id": job_id,
                "job_spec": {
                    "path": "job-spec.json",
                    "sha256": hashlib.sha256(
                        spec_path.read_bytes()
                    ).hexdigest(),
                },
                "runtime_id": "0" * 64,
                "worker": worker_record,
                "event_schema_version": "1.0.0",
                "status_schema_version": "1.0.0",
                "cancel_request_path": "cancel.request",
                "target_scene": spec["target_scene"],
            },
        )
        _wait(job_dir / "status.json")
        _write_json(job_dir / "cancel.request", {"cancel": True})
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 2, (process.returncode, stderr)
        terminal = _wait(
            diagnostics / f"{job_id}--fixture-cancelled"
        )
        record = json.loads(
            (terminal / "diagnostic.json").read_text("utf-8")
        )
        status = json.loads((terminal / "status.json").read_text("utf-8"))
        events = [
            json.loads(line)
            for line in (terminal / "events.jsonl").read_text(
                "utf-8"
            ).splitlines()
        ]
        protocol = [json.loads(line) for line in stdout.splitlines()]
        assert not stderr
        assert record["error_code"] == "pipeline.fixture.cancelled"
        assert record["category"] == "pipeline"
        assert record["state"] == "cancelled"
        assert record["job_id"] == job_id
        assert status["state"] == "cancelled"
        assert status["error"] == "pipeline.fixture.cancelled"
        assert events[-1]["kind"] == "cancelled"
        assert protocol[-1] == events[-1]
        assert (terminal / "human.log").is_file()
        assert not (terminal / "exception.json").exists()

        report_path = workspace / "cancelled-portable.zip"
        report = diagnostics_module.export_portable_report(
            terminal,
            report_path,
        )
        assert report["manifest"]["identity_mode"] == "redacted"

        bootstrap_id = f"job-{uuid.uuid4().hex}"
        bootstrap_dir = jobs / bootstrap_id
        bootstrap_dir.mkdir()
        bootstrap_spec = {
            **spec,
            "job_id": bootstrap_id,
        }
        bootstrap_path = bootstrap_dir / "job-spec.json"
        _write_json(bootstrap_path, bootstrap_spec)
        bootstrap_nonce = uuid.uuid4().hex
        bootstrap = subprocess.Popen(
            [
                str(worker_python),
                "-I",
                "-m",
                "lingbot_map_worker",
                "--fixture-job",
                str(bootstrap_path),
                "--job-nonce",
                bootstrap_nonce,
            ],
            cwd=trusted_cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        bootstrap_worker = json.loads(
            _wait(bootstrap_dir / "worker.pid.json").read_text("utf-8")
        )
        _write_json(
            bootstrap_dir / "job-control.json",
            {
                "schema_version": "1.0.0",
                "job_id": bootstrap_id,
                "job_spec": {
                    "path": "job-spec.json",
                    "sha256": hashlib.sha256(
                        bootstrap_path.read_bytes()
                    ).hexdigest(),
                },
                "runtime_id": "0" * 64,
                "worker": {
                    **bootstrap_worker,
                    "nonce": "0" * 32,
                },
                "event_schema_version": "1.0.0",
                "status_schema_version": "1.0.0",
                "cancel_request_path": "cancel.request",
                "target_scene": bootstrap_spec["target_scene"],
            },
        )
        bootstrap_stdout, bootstrap_stderr = bootstrap.communicate(
            timeout=20
        )
        assert bootstrap.returncode == 1
        assert not bootstrap_stdout
        assert bootstrap_stderr
        bootstrap_record = json.loads(
            (bootstrap_dir / "diagnostic.json").read_text("utf-8")
        )
        bootstrap_exception = json.loads(
            (bootstrap_dir / "exception.json").read_text("utf-8")
        )
        assert bootstrap_record["error_code"] == (
            "launch.worker.bootstrap-failed"
        )
        assert bootstrap_exception["error_code"] == (
            "launch.worker.bootstrap-failed"
        )
        print(
            "LINGBOT_MAP_WORKER_DIAGNOSTICS_WINDOWS="
            + json.dumps(
                {
                    "return_code": process.returncode,
                    "terminal_state": status["state"],
                    "error_code": record["error_code"],
                    "structured_events": len(events),
                    "protocol_lines": len(protocol),
                    "human_log_files": len(
                        tuple(terminal.glob("human.log*"))
                    ),
                    "portable_zip_sha256": report["sha256"],
                    "bootstrap_error_code": bootstrap_record[
                        "error_code"
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
