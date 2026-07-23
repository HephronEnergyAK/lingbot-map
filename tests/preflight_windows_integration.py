"""Real Windows detached-process proof for Capture Source preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import uuid
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "bpy" not in sys.modules:
    bpy = ModuleType("bpy")
    bpy.app = SimpleNamespace(version=(5, 2, 1), online_access=False)
    bpy.types = SimpleNamespace(
        AddonPreferences=type("AddonPreferences", (), {}),
        Operator=type("Operator", (), {}),
        Panel=type("Panel", (), {}),
    )
    bpy.utils = SimpleNamespace(register_class=lambda _cls: None, unregister_class=lambda _cls: None)
    props = ModuleType("bpy.props")
    props.BoolProperty = lambda **kwargs: kwargs
    props.StringProperty = lambda **kwargs: kwargs
    bpy.props = props
    sys.modules["bpy"] = bpy
    sys.modules["bpy.props"] = props

from blender_extension.gpu_capability import _runtime_command
from blender_extension.job_lifecycle import JobController
import blender_extension.job_lifecycle as job_lifecycle


def _wait_terminal(controller: JobController, timeout: float = 30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if snapshot.state in {
            "succeeded", "failed", "cancelled", "protocol_error", "interrupted"
        }:
            return snapshot
        time.sleep(0.05)
    raise RuntimeError(f"preflight did not terminate within {timeout} seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    arguments = parser.parse_args()
    if os.name != "nt":
        parser.error("this integration proof is Windows-only")
    workspace = arguments.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        parser.error("integration workspace must be empty")
    runtime, python, runtime_id, _lock_sha = _runtime_command(arguments.managed_root)
    source = workspace / "capture.mp4"
    subprocess.run(
        [str(python), str(ROOT / "tests" / "preflight_media_fixture.py"), str(source)],
        check=True,
        cwd=runtime / "empty-cwd",
    )
    blend = workspace / "capture-project.blend"
    blend.touch()
    controller = JobController(heartbeat_window=10, reconnect_window=3, cancel_grace=5)
    job_id = controller.launch_preflight(
        managed_root=arguments.managed_root,
        blend_path=blend,
        scene_uuid=str(uuid.uuid4()),
        scene_name="Scene",
        timeline_start=17,
        capture_draft_path="//capture.mp4",
    )
    snapshot = _wait_terminal(controller)
    if snapshot.state != "succeeded" or not snapshot.location:
        raise RuntimeError(f"preflight failed: {snapshot}")
    terminal = Path(snapshot.location)
    result = json.loads((terminal / "preflight-result.json").read_text(encoding="utf-8"))
    timestamps = (terminal / "timestamps.f64le").read_bytes()
    expected_files = {
        "events.jsonl", "job-control.json", "job-spec.json", "preflight-result.json",
        "status.json", "timestamps.f64le", "worker.pid.json",
    }
    if {item.name for item in terminal.iterdir()} != expected_files:
        raise RuntimeError("terminal preflight retained unexpected files or pixels")
    assertions = {
        "job_id": job_id,
        "runtime_id": runtime_id,
        "state": snapshot.state,
        "terminal": str(terminal),
        "frame_count": result["timing"]["frame_count"],
        "variable_frame_rate": result["timing"]["variable_frame_rate"],
        "capture_absolute": result["source"]["absolute_path"],
        "capture_relative": result["source"]["scene_relative_path"],
        "source_sha256": result["source"]["sha256"],
        "timestamp_sha256": hashlib.sha256(timestamps).hexdigest(),
        "timestamp_length": len(timestamps),
        "canonical_pixel_path": result["canonical_rgb"]["path"],
        "pyav_version": result["decoder"]["pyav_version"],
    }
    if assertions["frame_count"] != 8 or assertions["timestamp_length"] != 64:
        raise RuntimeError("preflight did not publish exactly eight float64 timestamps")
    if not assertions["variable_frame_rate"] or assertions["canonical_pixel_path"] is not None:
        raise RuntimeError("preflight VFR/pixel-retention contract failed")
    if assertions["capture_relative"] != "//capture.mp4" or assertions["pyav_version"] != "17.1.0":
        raise RuntimeError("preflight identity or decoder pin is wrong")

    replacement = workspace / "replacement.mp4"
    replacement.write_bytes(source.read_bytes())
    replacement_blend = workspace / "replacement-project.blend"
    replacement_blend.touch()
    replacement_controller = JobController(heartbeat_window=10, reconnect_window=3, cancel_grace=5)
    real_atomic_write = job_lifecycle.atomic_write_json
    replacement_mutated = False

    def mutate_before_control(path, document):
        nonlocal replacement_mutated
        if path.name == "job-control.json" and not replacement_mutated:
            original = replacement.stat()
            os.utime(
                replacement,
                ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
            )
            replacement_mutated = True
        return real_atomic_write(path, document)

    with mock.patch.object(job_lifecycle, "atomic_write_json", side_effect=mutate_before_control):
        replacement_job = replacement_controller.launch_preflight(
            managed_root=arguments.managed_root,
            blend_path=replacement_blend,
            scene_uuid=str(uuid.uuid4()),
            scene_name="Scene",
            timeline_start=1,
            capture_draft_path="//replacement.mp4",
        )
    replacement_snapshot = _wait_terminal(replacement_controller)
    replacement_terminal = Path(replacement_snapshot.location or "")
    replacement_status = json.loads(
        (replacement_terminal / "status.json").read_text(encoding="utf-8")
    )
    if replacement_snapshot.state != "failed" or "changed after the Job Draft was frozen" not in str(replacement_status["error"]):
        raise RuntimeError(f"replacement was not rejected: {replacement_snapshot}")
    if (replacement_terminal / "preflight-result.json").exists():
        raise RuntimeError("rejected replacement published a preflight result")
    assertions["replacement_job_id"] = replacement_job
    assertions["replacement_state"] = replacement_snapshot.state

    cancelled = workspace / "cancelled.mp4"
    cancelled.write_bytes(source.read_bytes())
    cancelled_blend = workspace / "cancelled-project.blend"
    cancelled_blend.touch()
    cancelled_controller = JobController(heartbeat_window=10, reconnect_window=3, cancel_grace=5)

    def cancel_before_control(path, document):
        if path.name == "job-control.json":
            (path.parent / "cancel.request").touch()
        return real_atomic_write(path, document)

    with mock.patch.object(job_lifecycle, "atomic_write_json", side_effect=cancel_before_control):
        cancelled_job = cancelled_controller.launch_preflight(
            managed_root=arguments.managed_root,
            blend_path=cancelled_blend,
            scene_uuid=str(uuid.uuid4()),
            scene_name="Scene",
            timeline_start=1,
            capture_draft_path="//cancelled.mp4",
        )
    cancelled_snapshot = _wait_terminal(cancelled_controller)
    if cancelled_snapshot.state != "cancelled":
        raise RuntimeError(f"queued cancellation was not honored: {cancelled_snapshot}")
    assertions["cancelled_job_id"] = cancelled_job
    assertions["cancelled_state"] = cancelled_snapshot.state
    print(json.dumps(assertions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
