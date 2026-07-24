"""Real Blender 5.2 lifecycle oracle for Result and Scene identity."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

import bpy


ROOT = Path(__file__).resolve().parents[1]
MANAGED_ROOT = Path(
    os.environ.get(
        "LINGBOT_MAP_TEST_MANAGED_ROOT",
        r"C:\tmp\lingbot-map-runtime-issue15-final",
    )
)
WORKSPACE = Path(r"C:\tmp\lingbot-map-result-identity") / f"run-{os.getpid()}"
SCENE_UUID = "22345678-1234-4321-8765-123456789abc"
OTHER_SCENE_UUID = "32345678-1234-4321-8765-123456789abc"


class _Layout:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.enabled = True

    def label(self, *, text, **_kwargs):
        self.events.append(("label", text))

    def operator(self, operator_id, **_kwargs):
        self.events.append(("operator", operator_id))
        return SimpleNamespace()

    def prop(self, _owner, property_name, **_kwargs):
        self.events.append(("prop", property_name))

    def separator(self):
        self.events.append(("separator",))

    def box(self):
        return _Layout(self.events)

    def row(self, **_kwargs):
        return _Layout(self.events)


def _wait(controller) -> None:
    deadline = time.monotonic() + 30
    while controller.snapshot().state not in {
        "succeeded",
        "failed",
        "cancelled",
        "protocol_error",
        "interrupted",
    }:
        assert time.monotonic() < deadline, controller.snapshot()
        time.sleep(0.05)
    assert controller.snapshot().state == "succeeded", controller.snapshot()


def _launch_fixture(jobs, results, blend: Path, scene, source: Path):
    controller = jobs.JobController()
    job_id = controller.launch_result_fixture(
        managed_root=MANAGED_ROOT,
        blend_path=blend,
        scene_uuid=SCENE_UUID,
        scene_name=scene.name,
        timeline_start=scene.frame_current,
        capture_draft_path=str(source),
    )
    _wait(controller)
    discovered = tuple(
        item
        for item in results.discover_ready_results(
            blend, scene_uuid=SCENE_UUID
        )
        if item.job_id == job_id
    )
    assert len(discovered) == 1, discovered
    return discovered[0]


def _expect_import_error(importer, callback, fragment: str) -> None:
    try:
        callback()
    except importer.ResultImportError as exc:
        assert fragment in str(exc), (fragment, str(exc))
    else:
        raise AssertionError(f"Expected ResultImportError containing {fragment!r}")


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    original_blend = WORKSPACE / "original.blend"
    source = WORKSPACE / "capture.mp4"
    source.write_bytes(b"result identity fixture capture")
    scene = bpy.context.scene
    scene["lingbot_map_scene_uuid"] = SCENE_UUID
    bpy.ops.wm.save_as_mainfile(filepath=str(original_blend))

    sys.path.insert(0, str(ROOT))
    import blender_extension as extension
    from blender_extension import job_lifecycle as jobs
    from blender_extension import result_import as importer
    from blender_extension import results

    extension.register()

    active_controller = jobs.JobController()
    active_job_id = active_controller.launch_fixture(
        managed_root=MANAGED_ROOT,
        blend_path=original_blend,
        scene_uuid=SCENE_UUID,
        scene_name=scene.name,
        timeline_start=scene.frame_current,
        steps=30,
        step_delay_seconds=0.03,
        heartbeat_interval_seconds=0.05,
    )
    active_save_as = WORKSPACE / "active-save-as.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(active_save_as))
    active_snapshot = active_controller.snapshot()
    assert Path(active_snapshot.target_blend) == original_blend
    active_spec = json.loads(
        (
            jobs.project_result_root(original_blend)
            / ".jobs"
            / active_job_id
            / "job-spec.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        Path(active_spec["target_scene"]["blend_path"])
        == original_blend
    )
    assert not jobs.project_result_root(active_save_as).exists()
    _wait(active_controller)
    bpy.ops.wm.save_as_mainfile(filepath=str(original_blend))

    first = _launch_fixture(jobs, results, original_blend, scene, source)
    first_import = importer.import_result(
        first.directory,
        scene,
        bpy_module=bpy,
        available_probe=lambda: 64 * 1024**3,
    )
    first_collection = first_import.collection
    assert first_collection["lingbot_map_result_reference_relative"].startswith(
        "//"
    )
    assert (
        Path(
            first_collection[
                "lingbot_map_result_reference_absolute"
            ]
        )
        == first.directory
    )
    assert (
        first_collection["lingbot_map_manifest_sha256"]
        == importer.validate_result(first.directory).manifest_sha256
    )
    first_inspection = importer.inspect_collection_ownership(
        first_collection, scene
    )
    assert first_inspection.status == "managed", first_inspection
    for index, datablock in enumerate(first_inspection.datablocks):
        if hasattr(datablock, "name"):
            datablock.name = f"User Rename {index}"
    scene.name = "User Renamed Scene"
    point_data = next(
        item
        for item in first_collection.objects
        if item.get("lingbot_map_kind") == "point_cloud"
    ).data
    original_point = tuple(point_data.points[0].co)
    edited_point = (
        original_point[0] + 1.25,
        original_point[1],
        original_point[2],
    )
    point_data.points[0].co = edited_point
    assert (
        importer.inspect_collection_ownership(first_collection, scene).status
        == "managed"
    )
    repeated = importer.import_result(
        first.directory,
        scene,
        bpy_module=bpy,
        available_probe=lambda: 64 * 1024**3,
    )
    assert not repeated.created and repeated.collection == first_collection
    assert tuple(point_data.points[0].co) == edited_point

    duplicate_scene = scene.copy()
    duplicate_scene.name = "Duplicated Scene"
    groups = jobs.duplicate_scene_uuid_groups(tuple(bpy.data.scenes))
    assert set(groups) == {SCENE_UUID}
    reconstruct_panel = SimpleNamespace(layout=_Layout())
    extension.ui.LINGBOTMAP_PT_reconstruct.draw(
        reconstruct_panel, SimpleNamespace(scene=scene)
    )
    assert (
        "operator",
        "lingbot_map.repair_duplicate_scene_uuid",
    ) in reconstruct_panel.layout.events
    _expect_import_error(
        importer,
        lambda: importer.import_result(
            first.directory,
            scene,
            bpy_module=bpy,
            available_probe=lambda: 64 * 1024**3,
        ),
        "duplicated",
    )
    old_actual = first_collection["lingbot_map_actual_scene_uuid"]
    repaired = jobs.repair_duplicate_scene_uuids(
        tuple(bpy.data.scenes), SCENE_UUID, scene
    )
    assert repaired == {
        duplicate_scene.name: duplicate_scene["lingbot_map_scene_uuid"]
    }
    assert uuid.UUID(duplicate_scene["lingbot_map_scene_uuid"])
    assert first_collection["lingbot_map_actual_scene_uuid"] == old_actual
    _expect_import_error(
        importer,
        lambda: importer.import_result(
            first.directory,
            scene,
            bpy_module=bpy,
            available_probe=lambda: 64 * 1024**3,
        ),
        "Save",
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(original_blend))
    assert not scene.get(jobs.SCENE_UUID_SAVE_REQUIRED_PROPERTY, False)
    assert not duplicate_scene.get(
        jobs.SCENE_UUID_SAVE_REQUIRED_PROPERTY, False
    )
    repeated_after_repair = importer.import_result(
        first.directory,
        scene,
        bpy_module=bpy,
        available_probe=lambda: 64 * 1024**3,
    )
    assert not repeated_after_repair.created
    bpy.data.scenes.remove(duplicate_scene)

    target_scene = bpy.data.scenes.new("Explicit Import Target")
    target_scene["lingbot_map_scene_uuid"] = OTHER_SCENE_UUID
    bpy.ops.wm.save_as_mainfile(filepath=str(original_blend))
    imported_into = importer.import_result(
        first.directory,
        target_scene,
        bpy_module=bpy,
        available_probe=lambda: 64 * 1024**3,
    )
    assert imported_into.created
    assert (
        imported_into.collection["lingbot_map_original_scene_uuid"]
        == SCENE_UUID
    )
    assert (
        imported_into.collection["lingbot_map_actual_scene_uuid"]
        == OTHER_SCENE_UUID
    )
    assert (
        imported_into.collection["lingbot_map_original_blend_path"]
        == str(original_blend.resolve())
    )

    second = _launch_fixture(jobs, results, original_blend, scene, source)
    assert second.result_id != first.result_id
    _expect_import_error(
        importer,
        lambda: importer.relink_result_reference(
            first_collection, scene, second.directory, original_blend
        ),
        "does not match",
    )

    save_as_directory = WORKSPACE / "save-as-project"
    save_as_directory.mkdir()
    save_as_blend = save_as_directory / "save-as.blend"
    bpy.context.window.scene = scene
    bpy.ops.wm.save_as_mainfile(filepath=str(save_as_blend))
    assert (
        importer.effective_disk_authority(
            first_collection, save_as_blend
        )
        == "external-read-only"
    )
    assert not jobs.project_result_root(save_as_blend).exists()
    assert jobs.project_result_root(original_blend).is_dir()
    status, resolved = importer.result_reference_status(
        first_collection, save_as_blend
    )
    assert status == "available" and Path(resolved) == first.directory

    external = importer.import_result(
        second.directory,
        scene,
        bpy_module=bpy,
        available_probe=lambda: 64 * 1024**3,
    ).collection
    assert (
        external["lingbot_map_result_reference_mode"]
        == "external-read-only"
    )
    assert (
        importer.effective_disk_authority(external, save_as_blend)
        == "external-read-only"
    )

    moved_parent = WORKSPACE / "relocated-results"
    moved_parent.mkdir()
    moved_directory = moved_parent / second.directory.name
    second.directory.rename(moved_directory)
    points_before = len(
        next(
            item
            for item in external.objects
            if item.get("lingbot_map_kind") == "point_cloud"
        ).data.points
    )
    assert importer.result_reference_status(
        external, save_as_blend
    ) == ("unavailable", None)
    assert points_before > 0
    assert "exact Result ID" in importer.relink_result_reference(
        external, scene, moved_directory, save_as_blend
    )
    assert importer.result_reference_status(
        external, save_as_blend
    ) == ("available", str(moved_directory.resolve()))

    duplicate_collection = external.copy()
    duplicate_collection.name = "Append Style Shared Copy"
    scene.collection.children.link(duplicate_collection)
    assert len(
        [
            item
            for item in scene.collection.children
            if item.get("lingbot_map_result_id") == second.result_id
        ]
    ) == 2
    detached_count = importer.resolve_duplicate_imports(
        scene, second.result_id, external
    )
    assert detached_count == 1
    assert duplicate_collection.get("lingbot_map_result_id") is None
    assert len(duplicate_collection.objects) == len(external.objects)
    assert all(
        item in duplicate_collection.objects[:] for item in external.objects
    )

    exclusive_copy = external.copy()
    exclusive_copy.name = "Append Style Independent Copy"
    scene.collection.children.link(exclusive_copy)
    copied_objects = []
    for shared_object in tuple(exclusive_copy.objects):
        exclusive_copy.objects.unlink(shared_object)
        copied_object = shared_object.copy()
        if shared_object.data is not None:
            copied_object.data = shared_object.data.copy()
        exclusive_copy.objects.link(copied_object)
        copied_objects.append(copied_object)
    exclusive_copy["lingbot_map_result_reference_mode"] = "contradictory"
    assert importer.inspect_collection_ownership(
        exclusive_copy, scene
    ).status == "unknown"
    assert importer.resolve_duplicate_imports(
        scene, second.result_id, external
    ) == 1
    assert exclusive_copy.get("lingbot_map_result_id") is None
    assert copied_objects
    for copied_object in copied_objects:
        assert copied_object.get("lingbot_map_result_id") is None
        if copied_object.data is not None:
            assert copied_object.data.get("lingbot_map_result_id") is None
    assert all(item.name in bpy.data.objects for item in copied_objects)

    point_object = next(
        item
        for item in external.objects
        if item.get("lingbot_map_kind") == "point_cloud"
    )
    material = point_object.data.materials[0]
    expected_result_id = material["lingbot_map_result_id"]
    material["lingbot_map_result_id"] = first.result_id
    unknown = importer.inspect_collection_ownership(external, scene)
    assert unknown.status == "unknown"
    _expect_import_error(
        importer,
        lambda: importer.remove_managed_version(
            external, scene, bpy_module=bpy
        ),
        "Ownership Unknown",
    )
    material["lingbot_map_result_id"] = expected_result_id
    assert importer.inspect_collection_ownership(
        external, scene
    ).status == "managed"

    user_mesh = bpy.data.meshes.new("User Mesh")
    user_object = bpy.data.objects.new("User Object", user_mesh)
    external.objects.link(user_object)
    user_child = bpy.data.collections.new("User Child Collection")
    child_mesh = bpy.data.meshes.new("Child User Mesh")
    child_object = bpy.data.objects.new("Child User Object", child_mesh)
    user_child.objects.link(child_object)
    external.children.link(user_child)
    inventory = importer.removal_inventory(external, scene)
    assert inventory.object_count == len(external.objects)
    assert inventory.point_count == points_before
    assert inventory.shared_datablock_count > 0
    external_name = external.name
    removed = importer.remove_managed_version(
        external, scene, bpy_module=bpy
    )
    assert removed == inventory
    assert external_name not in bpy.data.collections
    assert duplicate_collection.name in bpy.data.collections
    assert user_object.name in bpy.data.objects
    assert user_object in scene.collection.objects[:]
    assert user_child in scene.collection.children[:]
    assert child_object in user_child.objects[:]
    assert moved_directory.is_dir(), "Remove Version must not delete disk data"
    assert len(duplicate_collection.objects) > 0

    results_class = extension.ui.LINGBOTMAP_PT_results
    results_panel = SimpleNamespace(
        layout=_Layout(),
        _draw_import_into_actions=results_class._draw_import_into_actions,
        _draw_managed_collection=results_class._draw_managed_collection,
    )
    results_class.draw(results_panel, SimpleNamespace(scene=scene))
    operator_ids = {
        event[1]
        for event in results_panel.layout.events
        if event[0] == "operator"
    }
    assert "lingbot_map.import_external_result" in operator_ids
    assert "lingbot_map.remove_result_version" in operator_ids

    marker = {
        "blender_version": bpy.app.version_string,
        "active_job_save_as_not_retargeted": True,
        "scene_uuid_keeper_explicit": True,
        "scene_uuid_save_required": True,
        "no_job_rebind": True,
        "import_into_preserved_original_and_actual": True,
        "save_as_externalized_without_migration": True,
        "external_import_read_only": True,
        "missing_disk_data_usable": points_before,
        "exact_relink": True,
        "rename_independent_ownership": True,
        "unknown_ownership_blocks_remove": True,
        "duplicate_detach_preserved_shared_data": True,
        "remove_unique_only": True,
        "disk_result_preserved": moved_directory.is_dir(),
    }
    (WORKSPACE / "marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    print(
        "LINGBOT_MAP_RESULT_IDENTITY="
        + json.dumps(marker, sort_keys=True)
    )


if __name__ == "__main__":
    main()
