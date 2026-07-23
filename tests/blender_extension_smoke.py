"""Run by Blender 5.2 to verify real Extension registration lifecycle."""

from __future__ import annotations

import addon_utils
import importlib
import json
import sys

import bpy


MODULE_NAME = "bl_ext.user_default.lingbot_map_reconstruction"
PANEL_TYPES = (
    "LINGBOTMAP_PT_setup",
    "LINGBOTMAP_PT_reconstruct",
    "LINGBOTMAP_PT_active_job",
    "LINGBOTMAP_PT_results",
    "LINGBOTMAP_PT_diagnostics",
)


def panels_registered():
    return all(hasattr(bpy.types, name) for name in PANEL_TYPES)


def main():
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    loaded_default, loaded_state = addon_utils.check(MODULE_NAME)
    assert loaded_state, (loaded_default, loaded_state)

    extension = importlib.import_module(MODULE_NAME)
    decision = extension.get_host_decision()
    assert decision is not None
    assert panels_registered()
    preferences_class = next(
        extension_class
        for extension_class in extension.CLASSES
        if extension_class.__name__ == "LINGBOTMAP_Preferences"
    )
    preferences = preferences_class.bl_rna.properties
    for property_name in ("runtime_root", "gpu_uuid", "offline_setup"):
        assert property_name in preferences, property_name
    sky_property = bpy.types.Scene.bl_rna.properties["lingbot_map_sky_mask"]
    assert sky_property.type == "BOOLEAN", sky_property.type
    assert sky_property.default is False, sky_property.default
    assert "not Dynamic Content removal" in sky_property.description
    assert not any(
        name == "lingbot_map"
        or name.startswith("lingbot_map.")
        or name == "worker"
        or name.startswith("worker.")
        for name in sys.modules
    )

    addon_utils.disable(MODULE_NAME, default_set=False)
    assert not panels_registered()
    assert extension.get_host_decision() is None

    extension = importlib.reload(extension)
    addon_utils.enable(MODULE_NAME, default_set=False)
    assert panels_registered()
    assert extension.get_host_decision() is not None

    addon_utils.disable(MODULE_NAME, default_set=False)
    assert not panels_registered()
    addon_utils.enable(MODULE_NAME, default_set=False)
    assert panels_registered()

    print(
        "LINGBOT_MAP_BLENDER_SMOKE="
        + json.dumps(
            {
                "blender_version": bpy.app.version_string,
                "module": MODULE_NAME,
                "host_supported": extension.get_host_decision().supported,
                "host_code": extension.get_host_decision().code,
                "panels": list(PANEL_TYPES),
                "preferences": ["runtime_root", "gpu_uuid", "offline_setup"],
                "sky_mask_property": {
                    "type": sky_property.type,
                    "default": sky_property.default,
                    "dynamic_content_claim": False,
                },
                "reload": "passed",
                "disable_enable": "passed",
                "worker_or_model_imported": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
