"""Real Windows Worker proof for bounded Result construction and discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("blender_extension")
package.__path__ = [str(ROOT / "blender_extension")]
sys.modules["blender_extension"] = package

from blender_extension.job_lifecycle import JobController  # noqa: E402
from blender_extension.results import discover_ready_results  # noqa: E402


SCENE_UUID = "12345678-1234-4321-8765-123456789abc"


def wait_for_terminal(controller: JobController, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if snapshot.state == "succeeded":
            return snapshot
        if snapshot.state in {
            "failed", "cancelled", "interrupted", "forced_termination",
            "protocol_error", "stale_identity",
        }:
            status = Path(snapshot.location or "") / "status.json"
            detail = status.read_text(encoding="utf-8") if status.is_file() else ""
            raise AssertionError(f"Result Fixture failed: {snapshot}; status={detail}")
        time.sleep(0.05)
    raise AssertionError(f"Result Fixture timed out: {controller.snapshot()}")


def main() -> int:
    if len(sys.argv) != 2 or os.name != "nt":
        raise SystemExit("usage: result_fixture_windows_integration.py MANAGED_ROOT")
    managed_root = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="lingbot-map-result9-") as temporary:
        workspace = Path(temporary)
        blend = workspace / "target.blend"
        blend.touch()
        source = workspace / "capture.mp4"
        source.write_bytes(b"deterministic result fixture capture identity")
        controller = JobController()
        job_id = controller.launch_result_fixture(
            managed_root=managed_root,
            blend_path=blend,
            scene_uuid=SCENE_UUID,
            scene_name="Result Fixture Scene",
            timeline_start=23,
            capture_draft_path="//capture.mp4",
            confidence_cutoff_percent=50,
            depth_cutoff_percent=99.5,
            import_point_budget=8,
            initial_voxel_edge_length=0.01,
        )
        terminal = wait_for_terminal(controller)
        terminal_path = Path(terminal.location)
        assert terminal_path.parent.name == "diagnostics"
        assert terminal_path.name == f"{job_id}--result-fixture-succeeded"
        ready = discover_ready_results(blend, scene_uuid=SCENE_UUID)
        assert len(ready) == 1
        result = ready[0]
        control = json.loads(
            (terminal_path / "job-control.json").read_text(encoding="utf-8")
        )
        runtime_id = control["runtime_id"]
        runtime = managed_root / "runtimes" / runtime_id
        verifier = subprocess.run(
            [
                str(runtime / ".venv" / "Scripts" / "python.exe"), "-I", "-c",
                (
                    "import json,sys,numpy as np;"
                    "from pathlib import Path;"
                    "from lingbot_map_worker.result_bundle import validate_result_bundle;"
                    "p=Path(sys.argv[1]);m=validate_result_bundle(p);"
                    "a=np.load(p/'arrays/camera_to_world.npy',allow_pickle=False);"
                    "print(json.dumps({'frames':m['counts']['frames'],'points':m['counts']['points'],"
                    "'first_camera':a[0].tolist(),'result_id':m['result_id']},sort_keys=True))"
                ),
                str(result.directory),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        verified = json.loads(verifier.stdout)
        assert verified["frames"] == 2
        assert 1 <= verified["points"] <= 8
        assert verified["first_camera"] == [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        print(
            json.dumps(
                {
                    "runtime_id": runtime_id,
                    "job_id": job_id,
                    "result_id": result.result_id,
                    "frames": result.frame_count,
                    "points": result.point_count,
                    "ready_discovery": True,
                    "worker_validation": True,
                    "terminal": terminal.state,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
