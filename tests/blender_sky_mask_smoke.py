"""Real Blender 5.2 registration smoke for the Sky Mask Job Draft control."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types

import bpy


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "lingbot_map_issue13_extension"
SKY_MASK_PROPERTY = "lingbot_map_sky_mask"


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT / "blender_extension")]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package
    extension = importlib.import_module(PACKAGE_NAME + ".__init__")

    extension.register()
    try:
        sky_property = bpy.types.Scene.bl_rna.properties[SKY_MASK_PROPERTY]
        assert sky_property.type == "BOOLEAN", sky_property.type
        assert sky_property.default is False, sky_property.default
        assert "not Dynamic Content removal" in sky_property.description
        assert getattr(bpy.context.scene, SKY_MASK_PROPERTY) is False
        assert hasattr(bpy.types, "LINGBOTMAP_PT_reconstruct")
        print(
            "LINGBOT_MAP_BLENDER_SKY_MASK="
            + json.dumps(
                {
                    "blender_version": bpy.app.version_string,
                    "property": SKY_MASK_PROPERTY,
                    "type": sky_property.type,
                    "default": sky_property.default,
                    "dynamic_content_claim": False,
                    "panel_registered": True,
                },
                sort_keys=True,
            )
        )
    finally:
        extension.unregister()
    assert SKY_MASK_PROPERTY not in bpy.types.Scene.bl_rna.properties
    assert not hasattr(bpy.types, "LINGBOTMAP_PT_reconstruct")


if __name__ == "__main__":
    main()
