"""Real-process Windows lifecycle qualification for issue #7.

Run with the managed Runtime root as argv[1]. The helper mode intentionally
exits its launcher process while the detached Worker is still running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import uuid


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("blender_extension")
package.__path__ = [str(ROOT / "blender_extension")]
sys.modules["blender_extension"] = package

from blender_extension.ipc import atomic_write_json, read_json  # noqa: E402
from blender_extension.job_lifecycle import (  # noqa: E402
    JobController,
    JobLifecycleError,
    StaleWorkerIdentity,
    _record_from_control,
    _same_worker,
    _observed_record,
    _terminate_exact,
)


SCENE_UUID = "12345678-1234-4321-8765-123456789abc"


def wait_for(controller: JobController, states: set[str], timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if snapshot.state in states:
            return snapshot
        if snapshot.state in {"interrupted", "protocol_error", "failed", "forced_termination"}:
            log = Path(snapshot.location or "") / "worker.log"
            detail = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            status_path = Path(snapshot.location or "") / "status.json"
            status = read_json(status_path) if status_path.is_file() else None
            raise AssertionError(f"unexpected terminal state: {snapshot}; status={status}; worker.log={detail}")
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {states}: {controller.snapshot()}")


def wait_for_path(path: Path, timeout: float = 20.0) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = list(path.parent.glob(path.name))
        if len(matches) == 1:
            return matches[0]
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def launch(controller: JobController, managed_root: Path, blend: Path, **fixture) -> str:
    blend.touch(exist_ok=True)
    return controller.launch_fixture(
        managed_root=managed_root,
        blend_path=blend,
        scene_uuid=SCENE_UUID,
        scene_name="Fixture Scene",
        timeline_start=37,
        **fixture,
    )


def helper_launch_and_exit(managed_root: Path, workspace: Path, marker: Path) -> None:
    controller = JobController()
    blend = workspace / "closed.blend"
    job_id = launch(
        controller, managed_root, blend,
        steps=30, step_delay_seconds=0.1, heartbeat_interval_seconds=0.2,
    )
    job_dir = workspace / "closed.lingbot-map" / ".jobs" / job_id
    control = read_json(job_dir / "job-control.json")
    marker.write_text(json.dumps({"job_id": job_id, "pid": control["worker"]["pid"]}), encoding="utf-8")
    os._exit(0)


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--launch-and-exit":
        helper_launch_and_exit(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: job_lifecycle_windows_integration.py MANAGED_ROOT")
    if os.name != "nt":
        raise SystemExit("Windows integration requires Windows")
    managed_root = Path(sys.argv[1])
    evidence: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="lingbot-map-job7-") as temporary:
        workspace = Path(temporary)

        normal = JobController()
        normal_id = launch(normal, managed_root, workspace / "normal.blend", steps=4, step_delay_seconds=0.05, heartbeat_interval_seconds=0.1)
        normal_snapshot = wait_for(normal, {"succeeded"})
        normal_dir = Path(normal_snapshot.location)
        spec = read_json(normal_dir / "job-spec.json")
        assert spec["timeline_start"] == 37 and spec["target_scene"]["scene_uuid"] == SCENE_UUID
        assert normal_dir.parent.name == "diagnostics" and not (workspace / "normal.lingbot-map" / ".jobs" / normal_id).exists()
        event_lines = (normal_dir / "events.jsonl").read_bytes().splitlines(keepends=True)
        assert event_lines and all(line.endswith(b"\n") for line in event_lines)
        evidence["normal_completion"] = normal_snapshot.state

        switched = JobController(reconnect_window=1.0)
        switch_id = launch(switched, managed_root, workspace / "switch-a.blend", steps=40, step_delay_seconds=0.05, heartbeat_interval_seconds=0.1)
        wait_for(switched, {"running"})
        try:
            launch(switched, managed_root, workspace / "switch-b.blend", steps=1, step_delay_seconds=0.01)
        except JobLifecycleError:
            pass
        else:
            raise AssertionError("process-level active Job limit was bypassed")
        switched.recover_project(workspace / "switch-b.blend")
        assert wait_for(switched, {"reconnecting"}).target_blend.endswith("switch-a.blend")
        reconnected = wait_for(switched, {"running"})
        assert reconnected.heartbeat_sequence > 0 and reconnected.target_blend.endswith("switch-a.blend")
        switched.request_cancel()
        assert wait_for(switched, {"cancelled"}).job_id == switch_id
        evidence["file_switch_reload"] = "higher heartbeat required"

        unresponsive = JobController(heartbeat_window=0.4, reconnect_window=0.3)
        launch(
            unresponsive, managed_root, workspace / "unresponsive.blend",
            steps=50, step_delay_seconds=0.05, heartbeat_interval_seconds=0.05,
            freeze_heartbeat_after_sequence=1,
        )
        stalled = wait_for(unresponsive, {"unresponsive"}, timeout=5)
        assert stalled.heartbeat_sequence == 1
        unresponsive.request_cancel()
        wait_for(unresponsive, {"cancelled"})
        evidence["unresponsive"] = "observed without automatic kill"

        graceful = JobController(cancel_grace=2.0)
        launch(graceful, managed_root, workspace / "cancel.blend", steps=100, step_delay_seconds=0.05, heartbeat_interval_seconds=0.1)
        wait_for(graceful, {"running"})
        graceful.request_cancel()
        wait_for(graceful, {"cancelled"})
        evidence["graceful_cancel"] = "one-way sentinel"

        stale = JobController(cancel_grace=1.0)
        stale_id = launch(stale, managed_root, workspace / "stale.blend", steps=100, step_delay_seconds=0.03, heartbeat_interval_seconds=0.1)
        wait_for(stale, {"running"})
        stale_dir = workspace / "stale.lingbot-map" / ".jobs" / stale_id
        control_path = stale_dir / "job-control.json"
        control = read_json(control_path)
        original = json.loads(json.dumps(control))
        control["worker"]["executable_sha256"] = "0" * 64
        atomic_write_json(control_path, control)
        try:
            stale.request_cancel()
        except StaleWorkerIdentity:
            pass
        else:
            raise AssertionError("stale executable checksum authorized cancellation")
        assert not (stale_dir / "cancel.request").exists()
        atomic_write_json(control_path, original)
        stale.request_cancel()
        wait_for(stale, {"cancelled"})
        evidence["stale_identity"] = "no sentinel and no signal"

        forced = JobController(cancel_grace=0.25)
        launch(
            forced, managed_root, workspace / "forced.blend",
            steps=1000, step_delay_seconds=0.05, ignore_cancel=True,
            heartbeat_interval_seconds=0.1,
        )
        wait_for(forced, {"running"})
        forced.request_cancel()
        forced_snapshot = wait_for(forced, {"forced_termination"}, timeout=10)
        recovery = read_json(Path(forced_snapshot.location) / "recovery.json")
        assert recovery["reason"] == "forced-termination"
        evidence["forced_termination"] = "exact identity after grace"

        interrupted = JobController()
        interrupted_id = launch(
            interrupted, managed_root, workspace / "interrupted.blend",
            steps=1000, step_delay_seconds=0.05, heartbeat_interval_seconds=0.1,
        )
        wait_for(interrupted, {"running"})
        interrupted_dir = workspace / "interrupted.lingbot-map" / ".jobs" / interrupted_id
        record = _record_from_control(read_json(interrupted_dir / "job-control.json"))
        assert _same_worker(record, _observed_record(record))
        interrupted.detach()
        _terminate_exact(record)
        deadline = time.monotonic() + 5
        while _same_worker(record, _observed_record(record)) and time.monotonic() < deadline:
            time.sleep(0.05)
        recovered = JobController()
        recovered.recover_project(workspace / "interrupted.blend")
        interrupted_snapshot = recovered.snapshot()
        assert interrupted_snapshot.state == "interrupted"
        assert read_json(Path(interrupted_snapshot.location) / "recovery.json")["reason"] == "interrupted"
        evidence["dead_staging"] = "diagnostics, never resumed"

        marker = workspace / "closed-marker.json"
        helper = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--launch-and-exit", str(managed_root), str(workspace), str(marker)],
            check=True, timeout=10,
        )
        assert helper.returncode == 0
        closed = json.loads(marker.read_text(encoding="utf-8"))
        closed_terminal = wait_for_path(workspace / "closed.lingbot-map" / "diagnostics" / f"{closed['job_id']}--fixture-succeeded", timeout=15)
        assert read_json(closed_terminal / "status.json")["state"] == "succeeded"
        evidence["launcher_exit"] = "detached Worker completed"

    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
