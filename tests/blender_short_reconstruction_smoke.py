"""Real Blender 5.2 API smoke for short-Reconstruction Profile controls."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import blender_extension
from blender_extension.job_lifecycle import (
    CAMERA_ITERATIONS_PROPERTY,
    CONFIDENCE_CUTOFF_PROPERTY,
    POINT_BUDGET_PROPERTY,
    PROFILE_PROPERTY,
)


blender_extension.register()
try:
    scene = bpy.context.scene
    setattr(scene, PROFILE_PROPERTY, "Balanced")
    assert getattr(scene, CAMERA_ITERATIONS_PROPERTY) == 4
    assert getattr(scene, CONFIDENCE_CUTOFF_PROPERTY) == 50.0
    assert getattr(scene, POINT_BUDGET_PROPERTY) == 5_000_000
    setattr(scene, CONFIDENCE_CUTOFF_PROPERTY, 49.0)
    assert getattr(scene, PROFILE_PROPERTY) == "Custom"
    operator = getattr(bpy.ops.lingbot_map, "run_reconstruction_job", None)
    assert operator is not None
    print(json.dumps({
        "blender": list(bpy.app.version),
        "balanced_defaults": [4, 50.0, 5_000_000],
        "edited_profile": "Custom",
        "operator_registered": True,
    }, sort_keys=True))
finally:
    blender_extension.unregister()
