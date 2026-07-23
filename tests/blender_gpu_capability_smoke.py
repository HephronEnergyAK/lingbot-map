"""Run the real Blender operator through the frozen Worker GPU suite."""

from __future__ import annotations

import importlib
import json
import sys
import time

import bpy


MODULE = "bl_ext.user_default.lingbot_map_reconstruction"
MANAGED_ROOT = r"C:\tmp\lingbot-map-runtime-issue4-final"
GPU_UUID = "GPU-3dd69ad9-06be-796b-5f6a-0a9159cd288c"


def main() -> None:
    capability = importlib.import_module(MODULE + ".gpu_capability")
    preferences = bpy.context.preferences.addons[MODULE].preferences
    preferences.runtime_root = MANAGED_ROOT
    preferences.gpu_uuid = ""
    result = bpy.ops.lingbot_map.test_gpu_profiles()
    assert result == {"FINISHED"}, result
    assert preferences.gpu_uuid == GPU_UUID, preferences.gpu_uuid
    deadline = time.monotonic() + 1800
    while capability.get_capability_snapshot().state not in {
        "succeeded", "cancelled", "blocked", "failed"
    }:
        assert time.monotonic() < deadline, capability.get_capability_snapshot()
        time.sleep(0.1)
    snapshot = capability.get_capability_snapshot()
    assert snapshot.state == "succeeded", snapshot
    assert snapshot.gpu_uuid == GPU_UUID, snapshot
    assert [item["identity"]["profile_name"] for item in snapshot.results] == [
        "Draft", "Balanced", "High"
    ]
    assert not any(
        name == "lingbot_map"
        or name.startswith("lingbot_map.")
        or name.startswith("lingbot_map_worker")
        for name in sys.modules
    )
    print(
        "LINGBOT_MAP_BLENDER_GPU_SMOKE="
        + json.dumps(
            {
                "blender_version": bpy.app.version_string,
                "gpu_uuid_persisted": preferences.gpu_uuid,
                "state": snapshot.state,
                "profiles": [
                    {
                        "name": item["identity"]["profile_name"],
                        "state": item["state"],
                        "peak": item["measured_peak_bytes"],
                        "required_free": item["required_free_bytes"],
                    }
                    for item in snapshot.results
                ],
                "worker_or_model_imported_into_blender": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
