"""Native Windows filesystem proof for bounded logs and portable diagnostics."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "lingbot_diagnostics_windows",
    ROOT / "blender_extension" / "diagnostics.py",
)
diagnostics_module = importlib.util.module_from_spec(DIAGNOSTICS_SPEC)
sys.modules[DIAGNOSTICS_SPEC.name] = diagnostics_module
assert DIAGNOSTICS_SPEC.loader is not None
DIAGNOSTICS_SPEC.loader.exec_module(diagnostics_module)
DiagnosticReportError = diagnostics_module.DiagnosticReportError
export_portable_report = diagnostics_module.export_portable_report
from lingbot_map_worker.human_log import (  # noqa: E402
    HUMAN_LOG_BYTES,
    HUMAN_LOG_FILES,
    RotatingHumanLog,
)


JOB_ID = "job-fedcba9876543210fedcba9876543210"


def main() -> int:
    if os.name != "nt":
        raise SystemExit("Diagnostics integration requires native Windows")
    with tempfile.TemporaryDirectory(
        prefix="lingbot-map-diagnostics18-"
    ) as temporary:
        workspace = Path(temporary)
        log_root = workspace / "log-source"
        discarded = []
        stream = RotatingHumanLog(
            log_root,
            on_discard=discarded.append,
        )
        block = b"A" * HUMAN_LOG_BYTES
        for _index in range(HUMAN_LOG_FILES):
            stream.write(block)
        stream.write(b"B")
        stream.close()
        logs = tuple(sorted(log_root.glob("human.log*")))
        assert len(logs) == HUMAN_LOG_FILES
        assert all(path.stat().st_size <= HUMAN_LOG_BYTES for path in logs)
        assert discarded == [HUMAN_LOG_BYTES]

        owner = workspace / "project.lingbot-map"
        diagnostics = owner / "diagnostics"
        diagnostic = diagnostics / f"{JOB_ID}--reconstruction-failed"
        diagnostic.mkdir(parents=True)
        record = {
            "schema_version": "1.0.0",
            "error_code": "pipeline.reconstruction.failed",
            "category": "pipeline",
            "state": "failed",
            "phase": "inference",
            "job_id": JOB_ID,
            "machine_name": "PRIVATE-MACHINE",
            "username": "PrivateUser",
            "project_root": r"D:\Secret Project\map.lingbot-map",
            "environment": {"API_TOKEN": "secret-environment-value"},
            "source": {
                "absolute_path": r"C:\Users\PrivateUser\capture.mov",
                "sha256": "c" * 64,
            },
            "gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        }
        (diagnostic / "diagnostic.json").write_text(
            json.dumps(record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for source in logs:
            target = diagnostic / source.name
            os.replace(source, target)
        with (diagnostic / "human.log").open("ab") as output:
            output.write(
                (
                    "\nPrivateUser PRIVATE-MACHINE "
                    r"C:\Users\PrivateUser\capture.mov "
                    "secret-environment-value "
                    + "c" * 64
                ).encode("utf-8")
            )
        (diagnostic / "capture.mov").write_bytes(b"source media")
        (diagnostic / "points.npy").write_bytes(b"point array")
        (diagnostic / "model.safetensors").write_bytes(b"model weights")

        destination = workspace / "portable.zip"
        outcome = export_portable_report(diagnostic, destination)
        assert outcome["manifest"]["identity_mode"] == "redacted"
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            assert "records/diagnostic.json" in names
            assert all(name.startswith("logs/") for name in names if "human.log" in name)
            assert not {
                "capture.mov",
                "points.npy",
                "model.safetensors",
            } & names
            content = b"\n".join(archive.read(name) for name in names)
        for secret in (
            b"PrivateUser",
            b"PRIVATE-MACHINE",
            b"Secret Project",
            b"secret-environment-value",
            b"capture.mov",
            b"c" * 64,
            b"GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            b"source media",
            b"point array",
            b"model weights",
        ):
            assert secret not in content, secret

        hardlink_diagnostic = diagnostics / f"{JOB_ID}--hardlink-failed"
        hardlink_diagnostic.mkdir()
        outside = workspace / "outside.log"
        outside.write_text("do not follow", encoding="utf-8")
        os.link(outside, hardlink_diagnostic / "human.log")
        try:
            report = export_portable_report(
                hardlink_diagnostic,
                workspace / "hardlink.zip",
            )
        except DiagnosticReportError:
            report = None
        if report is not None:
            with zipfile.ZipFile(report["path"]) as archive:
                errors = json.loads(archive.read("errors.json"))
                assert errors["errors"][0]["code"] == (
                    "diagnostic-export-linked-file"
                )
                assert b"do not follow" not in b"".join(
                    archive.read(name) for name in archive.namelist()
                )

        print(
            "LINGBOT_MAP_DIAGNOSTICS_WINDOWS="
            + json.dumps(
                {
                    "worker_log_files": len(logs),
                    "worker_log_file_bytes": HUMAN_LOG_BYTES,
                    "discarded_bytes": discarded[0],
                    "portable_zip_bytes": outcome["length"],
                    "portable_zip_sha256": outcome["sha256"],
                    "identity_mode": "redacted",
                    "hardlink_followed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
