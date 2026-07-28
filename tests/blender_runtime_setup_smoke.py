"""Invoke the real Blender operator and prove strict offline failure wiring."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import time

import bpy


MODULE_NAME = "bl_ext.user_default.lingbot_map_reconstruction"
EMPTY_ROOT = Path(r"C:\tmp\lingbot-map-blender-offline-missing")


def main() -> None:
    extension = importlib.import_module(MODULE_NAME)
    runtime = importlib.import_module(MODULE_NAME + ".runtime")
    preferences = bpy.context.preferences.addons[MODULE_NAME].preferences
    preferences.runtime_root = str(EMPTY_ROOT)
    preferences.offline_setup = True

    result = bpy.ops.lingbot_map.setup_runtime()
    assert result == {"FINISHED"}, result
    deadline = time.monotonic() + 30
    while runtime.get_setup_snapshot().state in {"running", "cancelling"}:
        assert time.monotonic() < deadline, runtime.get_setup_snapshot()
        time.sleep(0.05)
    snapshot = runtime.get_setup_snapshot()
    assert snapshot.state == "failed", snapshot
    assert "Offline Setup is missing verified catalog artifacts" in snapshot.message, snapshot
    assert "python" in snapshot.message and "uv" in snapshot.message, snapshot
    assert not (EMPTY_ROOT / "runtimes").exists(), EMPTY_ROOT

    result = bpy.ops.lingbot_map.download_model(model_id="skyseg")
    assert result == {"FINISHED"}, result
    deadline = time.monotonic() + 30
    while runtime.get_model_setup_snapshot().state in {"running", "cancelling"}:
        assert time.monotonic() < deadline, runtime.get_model_setup_snapshot()
        time.sleep(0.05)
    model_snapshot = runtime.get_model_setup_snapshot()
    assert model_snapshot.state == "failed", model_snapshot
    assert "Offline Setup is missing exact catalogued model artifacts" in model_snapshot.message
    assert "skyseg" in model_snapshot.message
    assert "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39" in model_snapshot.message
    assert not (EMPTY_ROOT / "models").exists(), EMPTY_ROOT
    print(
        "LINGBOT_MAP_RUNTIME_OPERATOR_SMOKE="
        + json.dumps(
            {
                "state": snapshot.state,
                "offline": True,
                "actionable_missing_artifacts": True,
                "runtime_published": False,
                "model_registered": False,
                "model_missing_report_exact": True,
                "operator": extension.CLASSES[1].bl_idname,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
