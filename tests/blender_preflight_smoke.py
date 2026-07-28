"""Real Blender 5.2 Scene Job Draft and complete-source preflight smoke."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import time

import bpy


MODULE_NAME = "bl_ext.user_default.lingbot_map_reconstruction"
MANAGED_ROOT = Path(r"C:\tmp\lingbot-map-runtime-issue4-final")
PROJECT_DIR = Path(r"C:\tmp\lingbot-map-blender-preflight8")


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    source = PROJECT_DIR / "capture.mp4"
    assert source.is_file(), source
    blend = PROJECT_DIR / "preflight.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    extension = importlib.import_module(MODULE_NAME)
    jobs = importlib.import_module(MODULE_NAME + ".job_lifecycle")
    preferences = bpy.context.preferences.addons[MODULE_NAME].preferences
    preferences.runtime_root = str(MANAGED_ROOT)
    scene = bpy.context.scene
    scene.render.fps = 23
    scene.render.fps_base = 1.0
    original_timeline = scene.frame_current
    draft = jobs.capture_source_draft_path(source, blend)
    assert draft == "//capture.mp4", draft
    setattr(scene, jobs.CAPTURE_SOURCE_PROPERTY, draft)
    assert hasattr(bpy.types.Scene, jobs.CAPTURE_SOURCE_PROPERTY)

    scene.pop(jobs.SCENE_UUID_PROPERTY, None)
    try:
        first = bpy.ops.lingbot_map.run_preflight_job()
    except RuntimeError as exc:
        assert "Scene UUID was assigned" in str(exc), exc
    else:
        assert first == {"CANCELLED"}, first
    scene_uuid = scene[jobs.SCENE_UUID_PROPERTY]
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    launched = bpy.ops.lingbot_map.run_preflight_job()
    assert launched == {"FINISHED"}, launched
    launch_snapshot = jobs.get_job_snapshot()
    job_dir = Path(launch_snapshot.location)
    frozen = jobs.read_json(job_dir / "job-spec.json")
    assert frozen["capture_source"]["absolute_path"] == str(source)
    assert frozen["capture_source"]["draft_path"] == draft
    assert frozen["capture_source"]["scene_relative_path"] == draft
    assert frozen["target_scene"]["scene_uuid"] == scene_uuid
    assert frozen["timeline_start"] == original_timeline

    setattr(scene, jobs.CAPTURE_SOURCE_PROPERTY, "//retargeted-after-launch.mp4")
    deadline = time.monotonic() + 20
    while jobs.get_job_snapshot().state not in {"succeeded", "failed", "cancelled", "protocol_error"}:
        assert time.monotonic() < deadline, jobs.get_job_snapshot()
        time.sleep(0.05)
    snapshot = jobs.get_job_snapshot()
    assert snapshot.state == "succeeded", snapshot
    terminal = Path(snapshot.location)
    result = jobs.read_json(terminal / "preflight-result.json")
    assert result["source"]["absolute_path"] == str(source), result["source"]
    assert result["source"]["draft_path"] == draft, result["source"]
    assert result["timing"]["frame_count"] == 8
    assert result["canonical_rgb"]["path"] is None
    assert scene.render.fps == 23 and scene.render.fps_base == 1.0
    assert scene.frame_current == original_timeline

    marker = {
        "blender_version": bpy.app.version_string,
        "job_id": snapshot.job_id,
        "state": snapshot.state,
        "scene_uuid": scene_uuid,
        "draft_before_launch": draft,
        "draft_after_launch": getattr(scene, jobs.CAPTURE_SOURCE_PROPERTY),
        "frozen_absolute": result["source"]["absolute_path"],
        "frame_count": result["timing"]["frame_count"],
        "scene_fps": scene.render.fps,
        "scene_frame": scene.frame_current,
        "pixels_retained": result["canonical_rgb"]["path"] is not None,
    }
    (PROJECT_DIR / "marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    print("LINGBOT_MAP_BLENDER_PREFLIGHT=" + json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
