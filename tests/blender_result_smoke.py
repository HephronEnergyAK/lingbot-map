"""Real Blender 5.2 transaction tests for Result publication and import."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
import shutil
import sys
import time
import types
from types import SimpleNamespace

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANAGED_ROOT = Path(r"C:\tmp\lingbot-map-runtime-issue4-final")
PROJECT_DIR = Path(r"C:\tmp\lingbot-map-blender-result14") / f"run-{os.getpid()}"
SCENE_UUID = "12345678-1234-4321-8765-123456789abc"


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    blend = PROJECT_DIR / "result.blend"
    source = PROJECT_DIR / "capture.mp4"
    source.write_bytes(b"blender result fixture capture identity")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    scene["lingbot_map_scene_uuid"] = SCENE_UUID
    scene.render.fps = 23
    scene.render.fps_base = 1.0
    original_frame = scene.frame_current
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.375
    unrelated_collection = bpy.data.collections.new("Unrelated User Collection")
    scene.collection.children.link(unrelated_collection)
    bpy.ops.object.camera_add()
    unrelated_camera = bpy.context.object
    unrelated_camera.name = "Unrelated User Camera"
    scene.camera = unrelated_camera
    unrelated_collection.objects.link(unrelated_camera)

    package_name = "lingbot_map_issue14_source"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "blender_extension")]
    sys.modules[package_name] = package
    jobs = __import__(package_name + ".job_lifecycle", fromlist=["*"])
    results = __import__(package_name + ".results", fromlist=["*"])
    importer = __import__(package_name + ".result_import", fromlist=["*"])
    controller = jobs.JobController()
    job_id = controller.launch_result_fixture(
        managed_root=MANAGED_ROOT,
        blend_path=blend,
        scene_uuid=SCENE_UUID,
        scene_name=scene.name,
        timeline_start=original_frame,
        capture_draft_path="//capture.mp4",
    )
    deadline = time.monotonic() + 30
    while controller.snapshot().state not in {
        "succeeded", "failed", "cancelled", "protocol_error", "interrupted"
    }:
        assert time.monotonic() < deadline, controller.snapshot()
        time.sleep(0.05)
    snapshot = controller.snapshot()
    assert snapshot.state == "succeeded", snapshot
    ready = results.discover_ready_results(blend, scene_uuid=SCENE_UUID)
    assert len(ready) == 1, ready
    result = ready[0]
    assert result.frame_count == 2
    assert 1 <= result.point_count <= 8
    windows_capacity = importer.evaluate_import_capacity(result.point_count)
    assert windows_capacity.allowed

    def data_counts():
        return {
            "collections": len(bpy.data.collections),
            "objects": len(bpy.data.objects),
            "pointclouds": len(bpy.data.pointclouds),
            "materials": len(bpy.data.materials),
            "node_groups": len(bpy.data.node_groups),
        }

    baseline = data_counts()

    def rewrite_manifest(directory, mutate):
        path = directory / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document)
        path.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def mutation_case(name, mutate_files):
        directory = PROJECT_DIR / "mutations" / name / result.directory.name
        shutil.copytree(result.directory, directory)
        mutate_files(directory)
        try:
            importer.import_result(
                directory,
                scene,
                bpy_module=bpy,
                available_probe=lambda: 64 * 1024**3,
            )
        except importer.ResultImportError:
            pass
        else:
            raise AssertionError(f"Invalid Result was accepted: {name}")
        assert data_counts() == baseline, name

    mutation_case(
        "schema-unknown-field",
        lambda directory: rewrite_manifest(
            directory, lambda document: document.update({"unknown": True})
        ),
    )
    mutation_case(
        "unsafe-path",
        lambda directory: rewrite_manifest(
            directory,
            lambda document: document["arrays"]["positions"].update(
                {"path": "../positions.npy"}
            ),
        ),
    )

    def corrupt_checksum(directory):
        path = directory / "arrays" / "positions.npy"
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            original = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([original[0] ^ 1]))

    mutation_case("checksum-mismatch", corrupt_checksum)
    mutation_case(
        "dtype-mismatch",
        lambda directory: rewrite_manifest(
            directory,
            lambda document: document["arrays"]["positions"].update(
                {"dtype": "<f8"}
            ),
        ),
    )
    mutation_case(
        "cross-file-count",
        lambda directory: rewrite_manifest(
            directory,
            lambda document: document["counts"].update(
                {"points": document["counts"]["points"] + 1}
            ),
        ),
    )

    def corrupt_semantics(directory):
        array_path = directory / "arrays" / "camera_to_world.npy"
        camera = np.load(array_path, mmap_mode="r+")
        camera[0, 0, 0] = np.float32(2.0)
        camera.flush()
        del camera
        digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
        rewrite_manifest(
            directory,
            lambda document: document["arrays"]["camera_to_world"].update(
                {"sha256": digest}
            ),
        )

    mutation_case("semantic-camera", corrupt_semantics)
    mutation_case(
        "undeclared-file",
        lambda directory: (directory / "unexpected.bin").write_bytes(b"x"),
    )

    for phase in importer.IMPORT_PHASES:
        try:
            importer.import_result(
                result.directory,
                scene,
                bpy_module=bpy,
                available_probe=lambda: 64 * 1024**3,
                cancel=lambda current, expected=phase: current == expected,
            )
        except importer.ResultImportCancelled:
            pass
        else:
            raise AssertionError(f"Cancellation phase did not cancel: {phase}")
        assert data_counts() == baseline, (phase, data_counts(), baseline)
        assert not any(
            child.get("lingbot_map_result_id") == result.result_id
            for child in scene.collection.children
        ), phase

    try:
        importer.import_result(
            result.directory,
            scene,
            bpy_module=bpy,
            available_probe=lambda: 1,
        )
    except importer.ImportCapacityError:
        pass
    else:
        raise AssertionError("Capacity rejection did not block import")
    assert data_counts() == baseline

    importer.reset_auto_import_attempts_for_tests()
    assert (
        importer.attempt_auto_import_once(
            result,
            expected_job_id="job-" + "0" * 32,
            bpy_module=bpy,
            available_probe=lambda: 64 * 1024**3,
        )
        is None
    )
    assert (
        importer.attempt_auto_import_once(
            result,
            expected_job_id=result.job_id,
            bpy_module=bpy,
            available_probe=lambda: 64 * 1024**3,
        )
        is None
    ), "A failed automatic gate must never become a deferred import"
    assert data_counts() == baseline

    class FakeScene(dict):
        pass

    def fake_bpy(
        *,
        filepath=str(blend),
        active_scene=scene,
        scenes=(scene,),
        playing=False,
        running_job=None,
    ):
        return SimpleNamespace(
            data=SimpleNamespace(filepath=filepath, scenes=scenes),
            context=SimpleNamespace(
                scene=active_scene,
                screen=SimpleNamespace(is_animation_playing=playing),
            ),
            app=SimpleNamespace(
                is_job_running=lambda name: name == running_job
            ),
        )

    auto_gate_modules = (
        fake_bpy(playing=True),
        fake_bpy(running_job="RENDER"),
        fake_bpy(running_job="OBJECT_BAKE"),
        fake_bpy(filepath=str(PROJECT_DIR / "other.blend")),
        fake_bpy(
            active_scene=FakeScene(
                lingbot_map_scene_uuid="00000000-0000-4000-8000-000000000000"
            ),
            scenes=(scene,),
        ),
        fake_bpy(
            scenes=(
                scene,
                FakeScene(lingbot_map_scene_uuid=SCENE_UUID),
            )
        ),
    )
    for fake in auto_gate_modules:
        importer.reset_auto_import_attempts_for_tests()
        assert (
            importer.attempt_auto_import_once(
                result,
                expected_job_id=result.job_id,
                bpy_module=fake,
                available_probe=lambda: 64 * 1024**3,
            )
            is None
        )
        assert data_counts() == baseline

    importer.reset_auto_import_attempts_for_tests()
    imported = importer.attempt_auto_import_once(
        result,
        expected_job_id=result.job_id,
        bpy_module=bpy,
        available_probe=lambda: 64 * 1024**3,
    )
    assert imported is not None and imported.created
    collection = imported.collection
    assert collection in scene.collection.children[:]
    assert len(collection.objects) == 2
    root = next(
        item
        for item in collection.objects
        if item.get("lingbot_map_kind") == "reconstruction_root"
    )
    point_object = next(
        item
        for item in collection.objects
        if item.get("lingbot_map_kind") == "point_cloud"
    )
    pointcloud = point_object.data
    assert point_object.parent == root
    assert tuple(root.scale) == (1.0, 1.0, 1.0)
    assert len(pointcloud.points) == result.point_count
    assert pointcloud.attributes["position"].domain == "POINT"
    assert pointcloud.attributes["color"].data_type == "BYTE_COLOR"
    assert pointcloud.attributes["color"].domain == "POINT"
    assert pointcloud.attributes["confidence"].data_type == "FLOAT"
    assert pointcloud.attributes["radius"].data_type == "FLOAT"
    assert pointcloud.attributes["source_frame"].data_type == "INT"
    actual_positions = np.empty((result.point_count, 3), dtype=np.float32)
    pointcloud.points.foreach_get("co", actual_positions.reshape(-1))
    np.testing.assert_allclose(
        actual_positions, np.load(result.directory / "arrays" / "positions.npy")
    )
    actual_colors = np.empty((result.point_count, 4), dtype=np.float32)
    pointcloud.attributes["color"].data.foreach_get(
        "color_srgb", actual_colors.reshape(-1)
    )
    expected_colors = (
        np.load(result.directory / "arrays" / "colors.npy").astype(np.float32)
        / 255.0
    )
    np.testing.assert_allclose(actual_colors[:, :3], expected_colors, atol=1 / 255)
    np.testing.assert_allclose(actual_colors[:, 3], 1.0)
    for name in ("confidence", "radius"):
        actual = np.empty(result.point_count, dtype=np.float32)
        pointcloud.attributes[name].data.foreach_get("value", actual)
        np.testing.assert_allclose(
            actual, np.load(result.directory / "arrays" / f"{name}.npy")
        )
    actual_source_frame = np.empty(result.point_count, dtype=np.int32)
    pointcloud.attributes["source_frame"].data.foreach_get(
        "value", actual_source_frame
    )
    np.testing.assert_array_equal(
        actual_source_frame,
        np.load(result.directory / "arrays" / "source_frame.npy"),
    )
    assert len(pointcloud.materials) == 1
    material = pointcloud.materials[0]
    assert material.get("lingbot_map_schema") == "point-material-1.0.0"
    shader_nodes = [
        node.node_tree
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeGroup"
    ]
    assert len(shader_nodes) == 1
    shader_group = shader_nodes[0]
    assert shader_group.get("lingbot_map_schema") == "point-shader-1.0.0"
    assert any(node.bl_idname == "ShaderNodeEmission" for node in shader_group.nodes)
    assert any(node.bl_idname == "ShaderNodeValToRGB" for node in shader_group.nodes)
    assert any(node.bl_idname == "ShaderNodePointInfo" for node in shader_group.nodes)
    assert any(node.bl_idname == "ShaderNodeLightPath" for node in shader_group.nodes)
    ramp = next(
        node for node in shader_group.nodes if node.bl_idname == "ShaderNodeValToRGB"
    )
    assert len(ramp.color_ramp.elements) == 5
    np.testing.assert_allclose(
        ramp.color_ramp.elements[0].color[:3],
        (0.2667, 0.0039, 0.3294),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        ramp.color_ramp.elements[-1].color[:3],
        (0.9922, 0.9059, 0.1451),
        atol=1e-4,
    )
    assert len(point_object.modifiers) == 1
    display_group = point_object.modifiers[0].node_group
    assert display_group.get("lingbot_map_schema") == "point-display-1.0.0"
    assert any(
        node.bl_idname == "GeometryNodeInputNamedAttribute"
        and node.inputs["Name"].default_value == "radius"
        for node in display_group.nodes
    )
    assert any(
        node.bl_idname == "GeometryNodeSetPointRadius"
        for node in display_group.nodes
    )
    radius_socket = next(
        item
        for item in display_group.interface.items_tree
        if item.item_type == "SOCKET"
        and item.in_out == "INPUT"
        and item.name == "Radius Scale"
    )
    assert (
        getattr(
            point_object.modifiers[0].properties.inputs,
            radius_socket.identifier,
        ).value
        == 1.0
    )
    for datablock in (collection, root, point_object, pointcloud, material, display_group):
        assert datablock.get("lingbot_map_owner_schema") == "1.0.0"
        assert datablock.get("lingbot_map_result_id") == result.result_id
        assert datablock.get("lingbot_map_actual_scene_uuid") == SCENE_UUID
    repeated = importer.import_result(
        result.directory,
        scene,
        bpy_module=bpy,
        available_probe=lambda: 64 * 1024**3,
    )
    assert not repeated.created and repeated.collection == collection

    assert scene.render.fps == 23 and scene.render.fps_base == 1.0
    assert scene.frame_current == original_frame
    assert scene.render.engine == "BLENDER_WORKBENCH"
    assert scene.view_settings.look == "AgX - Medium High Contrast"
    assert scene.view_settings.exposure == 0.375
    assert scene.camera == unrelated_camera
    assert unrelated_camera in unrelated_collection.objects[:]
    marker = {
        "blender_version": bpy.app.version_string,
        "job_id": job_id,
        "state": snapshot.state,
        "result_id": result.result_id,
        "frames": result.frame_count,
        "points": result.point_count,
        "ready_discovery": True,
        "invalid_result_cases": 7,
        "auto_gate_cases": 7,
        "transactional_import": imported.created,
        "cancellation_phases": list(importer.IMPORT_PHASES),
        "capacity_rejection": True,
        "capacity_model": windows_capacity.model_version,
        "available_physical_bytes": windows_capacity.available_physical_bytes,
        "required_available_bytes": windows_capacity.required_available_bytes,
        "idempotent": not repeated.created,
        "point_attributes": sorted(
            attribute.name for attribute in pointcloud.attributes
        ),
        "material_schema": material.get("lingbot_map_schema"),
        "shader_schema": shader_group.get("lingbot_map_schema"),
        "geometry_nodes_schema": display_group.get("lingbot_map_schema"),
        "unrelated_state_preserved": True,
        "scene_fps": scene.render.fps,
        "scene_frame": scene.frame_current,
    }
    (PROJECT_DIR / "marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    print("LINGBOT_MAP_BLENDER_RESULT=" + json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
