"""Blender 5.2 smoke for window Quality Warning reporting metadata."""

from __future__ import annotations

import importlib
import json

import bpy


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    results = importlib.import_module(
        "bl_ext.user_default.lingbot_map_reconstruction.results"
    )
    boundary = {
        "source_frame_start": 48,
        "source_frame_end": 63,
        "triggered_conditions": ["relative_scale", "rotation_p95"],
    }
    document = {
        "schema_version": "1.0.0",
        "rule_version": "1.0.0",
        "strategy": "rolling-similarity",
        "window_frames": 64,
        "overlap_keyframes": 16,
        "scale_frames": 8,
        "keyframe_interval": 1,
        "loop_closure": False,
        "pose_graph": False,
        "bundle_adjustment": False,
        "global_optimization": False,
        "boundaries": [boundary],
    }
    count, warnings, worst = results._alignment_status(document)
    assert (count, warnings) == (1, 1)
    assert worst == "frames 48-63: relative_scale, rotation_p95"
    marker = {
        "blender_version": bpy.app.version_string,
        "boundaries": count,
        "quality_warnings": warnings,
        "worst_boundary": worst,
    }
    print("LINGBOT_MAP_BLENDER_WINDOW=" + json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
