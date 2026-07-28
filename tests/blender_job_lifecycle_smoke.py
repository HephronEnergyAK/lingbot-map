"""Launch in Blender 5.2, then let Blender exit before the Worker completes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import time

import bpy


MODULE_NAME = "bl_ext.user_default.lingbot_map_reconstruction"
MANAGED_ROOT = Path(r"C:\tmp\lingbot-map-runtime-issue4-final")
PROJECT_DIR = Path(r"C:\tmp\lingbot-map-blender-job7")


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    blend = PROJECT_DIR / "fixture.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    extension = importlib.import_module(MODULE_NAME)
    jobs = importlib.import_module(MODULE_NAME + ".job_lifecycle")
    preferences = bpy.context.preferences.addons[MODULE_NAME].preferences
    preferences.runtime_root = str(MANAGED_ROOT)

    scene = bpy.context.scene
    scene.pop(jobs.SCENE_UUID_PROPERTY, None)
    try:
        first = bpy.ops.lingbot_map.run_fixture_job()
    except RuntimeError as exc:
        assert "Scene UUID was assigned" in str(exc), exc
    else:
        assert first == {"CANCELLED"}, first
    scene_uuid = scene[jobs.SCENE_UUID_PROPERTY]
    assert scene_uuid
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    second = bpy.ops.lingbot_map.run_fixture_job()
    assert second == {"FINISHED"}, second
    deadline = time.monotonic() + 10
    while jobs.get_job_snapshot().state not in {"running", "failed", "interrupted"}:
        assert time.monotonic() < deadline, jobs.get_job_snapshot()
        time.sleep(0.05)
    snapshot = jobs.get_job_snapshot()
    assert snapshot.state == "running", snapshot
    job_dir = Path(snapshot.location)
    spec = jobs.read_json(job_dir / "job-spec.json")
    control = jobs.read_json(job_dir / "job-control.json")
    assert spec["target_scene"]["blend_path"] == str(blend)
    assert spec["target_scene"]["scene_uuid"] == scene_uuid
    assert spec["timeline_start"] == scene.frame_current
    assert control["worker"]["pid"] > 0
    assert control["worker"]["executable_sha256"]

    marker = {
        "blender_version": bpy.app.version_string,
        "job_id": snapshot.job_id,
        "project_root": str(PROJECT_DIR / "fixture.lingbot-map"),
        "target_blend": snapshot.target_blend,
        "scene_uuid": snapshot.target_scene_uuid,
        "state_at_blender_exit": snapshot.state,
    }
    (PROJECT_DIR / "marker.json").write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    print("LINGBOT_MAP_BLENDER_JOB_LAUNCH=" + json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
