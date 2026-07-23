"""Real Blender 5.2 smoke for Result fixture publication and discovery."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import types

import bpy


ROOT = Path(__file__).resolve().parents[1]
MANAGED_ROOT = Path(r"C:\tmp\lingbot-map-runtime-issue4-final")
PROJECT_DIR = Path(r"C:\tmp\lingbot-map-blender-result9")
SCENE_UUID = "12345678-1234-4321-8765-123456789abc"


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    blend = PROJECT_DIR / "result.blend"
    source = PROJECT_DIR / "capture.mp4"
    source.write_bytes(b"blender result fixture capture identity")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    scene.render.fps = 23
    scene.render.fps_base = 1.0
    original_frame = scene.frame_current

    package_name = "lingbot_map_issue9_source"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "blender_extension")]
    sys.modules[package_name] = package
    jobs = __import__(package_name + ".job_lifecycle", fromlist=["*"])
    results = __import__(package_name + ".results", fromlist=["*"])
    controller = jobs.JobController()
    job_id = controller.launch_result_fixture(
        managed_root=MANAGED_ROOT,
        blend_path=blend,
        scene_uuid=SCENE_UUID,
        scene_name=scene.name,
        timeline_start=original_frame,
        capture_draft_path="//capture.mp4",
    )
    deadline = time.monotonic() + 30
    while controller.snapshot().state not in {
        "succeeded", "failed", "cancelled", "protocol_error", "interrupted"
    }:
        assert time.monotonic() < deadline, controller.snapshot()
        time.sleep(0.05)
    snapshot = controller.snapshot()
    assert snapshot.state == "succeeded", snapshot
    ready = results.discover_ready_results(blend, scene_uuid=SCENE_UUID)
    assert len(ready) == 1, ready
    result = ready[0]
    assert result.frame_count == 2
    assert 1 <= result.point_count <= 8
    assert scene.render.fps == 23 and scene.render.fps_base == 1.0
    assert scene.frame_current == original_frame
    marker = {
        "blender_version": bpy.app.version_string,
        "job_id": job_id,
        "state": snapshot.state,
        "result_id": result.result_id,
        "frames": result.frame_count,
        "points": result.point_count,
        "ready_discovery": True,
        "scene_fps": scene.render.fps,
        "scene_frame": scene.frame_current,
    }
    (PROJECT_DIR / "marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    print("LINGBOT_MAP_BLENDER_RESULT=" + json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
