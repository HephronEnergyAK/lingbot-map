"""Blender 5.2 proof that disk lifecycle never mutates shared scene content."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import bpy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_extension.job_lifecycle import (  # noqa: E402
    JobController,
    SCENE_UUID_PROPERTY,
)
from blender_extension.project_lifecycle import ProjectLifecycle  # noqa: E402
from blender_extension.result_import import (  # noqa: E402
    import_result,
    result_reference_status,
)
from blender_extension.results import discover_ready_results  # noqa: E402


SCENE_UUID = "12345678-1234-4321-8765-123456789abc"


def _wait(controller: JobController, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if snapshot.state == "succeeded":
            return snapshot
        if snapshot.state in {
            "failed",
            "cancelled",
            "interrupted",
            "forced_termination",
            "protocol_error",
            "stale_identity",
        }:
            raise AssertionError(f"Fixture failed: {snapshot}")
        time.sleep(0.05)
    raise AssertionError(f"Fixture timed out: {controller.snapshot()}")


def main() -> int:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(arguments) != 1:
        raise SystemExit(
            "usage: blender --background --python "
            "blender_project_lifecycle_smoke.py -- MANAGED_ROOT"
        )
    managed_root = Path(arguments[0])
    workspace = Path(
        tempfile.mkdtemp(prefix="lingbot-map-blender-lifecycle17-", dir="C:\\tmp")
    )
    try:
        blend = workspace / "shared.blend"
        scene = bpy.context.scene
        scene[SCENE_UUID_PROPERTY] = SCENE_UUID
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        source = workspace / "capture.mp4"
        source.write_bytes(b"blender lifecycle fixture")
        controller = JobController()
        job_id = controller.launch_result_fixture(
            managed_root=managed_root,
            blend_path=blend,
            scene_uuid=SCENE_UUID,
            scene_name=scene.name,
            timeline_start=scene.frame_current,
            capture_draft_path="//capture.mp4",
            confidence_cutoff_percent=50,
            depth_cutoff_percent=99.5,
            import_point_budget=8,
            initial_voxel_edge_length=0.01,
        )
        _wait(controller)
        ready = discover_ready_results(blend, scene_uuid=SCENE_UUID)
        assert len(ready) == 1
        result = ready[0]
        imported = import_result(
            result.directory, scene, context=bpy.context
        )
        collection = imported.collection
        objects = tuple(collection.all_objects)
        assert objects
        shared_object = next(
            item for item in objects if item.data is not None
        )
        shared_data = shared_object.data
        shared = bpy.data.collections.new("User Shared Reference")
        scene.collection.children.link(shared)
        shared.objects.link(shared_object)
        before_object_pointer = shared_object.as_pointer()
        before_data_pointer = shared_data.as_pointer()

        lifecycle = ProjectLifecycle(blend)
        trashed = lifecycle.execute(
            lifecycle.plan(
                "trash_result", [result.directory.name]
            )
        ).destinations[0]
        status_while_trashed, _path = result_reference_status(
            collection, str(blend)
        )
        assert status_while_trashed == "unavailable"
        assert collection.name in bpy.data.collections
        assert shared.name in bpy.data.collections
        assert shared_object.name in bpy.data.objects
        assert shared_object.as_pointer() == before_object_pointer
        assert shared_object.data.as_pointer() == before_data_pointer
        assert shared_object in shared.objects[:]

        lifecycle.execute(
            lifecycle.plan("restore", [trashed.name])
        )
        restored_status, restored_path = result_reference_status(
            collection, str(blend)
        )
        assert restored_status == "available"
        assert Path(restored_path) == result.directory
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "result_id": result.result_id,
                    "shared_object_preserved": True,
                    "shared_data_preserved": True,
                    "collection_usable_while_trashed": True,
                    "reference_unavailable_while_trashed": True,
                    "reference_available_after_restore": True,
                },
                sort_keys=True,
            )
        )
    finally:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        shutil.rmtree(workspace, ignore_errors=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
