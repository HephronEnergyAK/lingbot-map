"""Create an isolated Blender preference file for visible visual oracles."""

from __future__ import annotations

import bpy


if not bpy.app.background:
    raise RuntimeError("visual preference bootstrap must run in background mode")

bpy.context.preferences.view.show_splash = False
result = bpy.ops.wm.save_userpref()
if result != {"FINISHED"}:
    raise RuntimeError("Blender did not save the isolated visual-test preferences")
if bpy.context.preferences.view.show_splash:
    raise RuntimeError("isolated visual-test profile still enables the splash")
print("LINGBOT_MAP_BLENDER_VISUAL_PREFERENCES=ready")
