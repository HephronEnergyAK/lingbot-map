"""Blender 5.2 camera/source-alignment and transactional visual-coordinate oracle."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import sys
import types

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(r"C:\tmp\lingbot-map-source-view15")
INDEX = json.loads((FIXTURE_ROOT / "index.json").read_text(encoding="utf-8"))
TRANSFORMS = tuple(INDEX["results"])
TIMELINE_START = int(INDEX["timeline_start"])
FRAME_COUNT = int(INDEX["frame_count"])


def _action_fcurves(datablock):
    action = datablock.animation_data.action
    return tuple(
        fcurve
        for layer in action.layers
        for strip in layer.strips
        for channelbag in strip.channelbags
        for fcurve in channelbag.fcurves
    )


def _data_counts():
    return {
        "collections": len(bpy.data.collections),
        "objects": len(bpy.data.objects),
        "pointclouds": len(bpy.data.pointclouds),
        "cameras": len(bpy.data.cameras),
        "curves": len(bpy.data.curves),
        "movieclips": len(bpy.data.movieclips),
        "actions": len(bpy.data.actions),
        "materials": len(bpy.data.materials),
        "node_groups": len(bpy.data.node_groups),
    }


def _reference_transform(image, transform):
    if transform == "identity":
        return image
    if transform == "rotate_90_ccw":
        return np.rot90(image, 1)
    if transform == "rotate_180":
        return np.rot90(image, 2)
    if transform == "rotate_270_ccw":
        return np.rot90(image, 3)
    if transform == "reflect_x":
        return np.flip(image, axis=1)
    if transform == "reflect_y":
        return np.flip(image, axis=0)
    if transform == "reflect_main_diagonal":
        return np.transpose(image, (1, 0))
    return np.flip(np.transpose(image, (1, 0)), axis=(0, 1))


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    package_name = "lingbot_map_issue15_source"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "blender_extension")]
    sys.modules[package_name] = package
    importer = __import__(package_name + ".result_import", fromlist=["*"])
    source_view = __import__(package_name + ".source_view", fromlist=["*"])

    scene = bpy.context.scene
    scene["lingbot_map_scene_uuid"] = INDEX["scene_uuid"]
    scene.render.fps = 23
    scene.render.fps_base = 1.0
    scene.frame_start = 5
    scene.frame_end = 10
    scene.use_preview_range = True
    scene.frame_preview_start = -3
    scene.frame_preview_end = 8
    scene.render.resolution_x = 111
    scene.render.resolution_y = 111
    scene.render.resolution_percentage = 73
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    blend = Path(INDEX["blend"])
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    unrelated_data = bpy.data.cameras.new("Unrelated User Camera Data")
    unrelated_camera = bpy.data.objects.new(
        "Unrelated User Camera", unrelated_data
    )
    scene.collection.objects.link(unrelated_camera)
    scene.camera = unrelated_camera

    identity_result = Path(INDEX["results"]["identity"])
    baseline = _data_counts()
    for phase in importer.IMPORT_PHASES:
        try:
            importer.import_result(
                identity_result,
                scene,
                bpy_module=bpy,
                available_probe=lambda: 64 * 1024**3,
                cancel=lambda current, expected=phase: current == expected,
            )
        except importer.ResultImportCancelled:
            pass
        else:
            raise AssertionError(
                f"source-view cancellation phase did not cancel: {phase}"
            )
        assert _data_counts() == baseline, (
            phase,
            _data_counts(),
            baseline,
        )
        assert scene.frame_end == 10

    capture = Path(INDEX["capture"])
    hidden_capture = capture.with_suffix(".missing")
    os.replace(capture, hidden_capture)
    scene.render.resolution_x = 64
    scene.render.resolution_y = 48
    scene.render.resolution_percentage = 100
    try:
        identity_outcome = importer.import_result(
            identity_result,
            scene,
            bpy_module=bpy,
            available_probe=lambda: 64 * 1024**3,
        )
    finally:
        os.replace(hidden_capture, capture)
    identity_collection = identity_outcome.collection
    assert identity_outcome.created
    assert identity_collection.get(
        "lingbot_map_source_background_status"
    ).startswith("unattached-")
    assert len(
        importer.find_reconstruction_camera(identity_collection).data.background_images
    ) == 0
    wrong = FIXTURE_ROOT / "wrong.mov"
    wrong.write_bytes(b"not the recorded source")
    try:
        importer.relink_source_background(
            identity_collection, scene, wrong, bpy_module=bpy
        )
    except importer.ResultImportError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("wrong-checksum relink was accepted")
    relink_message = importer.relink_source_background(
        identity_collection, scene, capture, bpy_module=bpy
    )
    assert "attached and hidden" in relink_message

    imported = {"identity": identity_collection}
    resolution_actions = []
    for transform in TRANSFORMS:
        if transform == "identity":
            collection = identity_collection
        else:
            scene.render.resolution_x = 111
            scene.render.resolution_y = 111
            scene.render.resolution_percentage = 73
            before_resolution = (
                scene.render.resolution_x,
                scene.render.resolution_y,
                scene.render.resolution_percentage,
            )
            outcome = importer.import_result(
                Path(INDEX["results"][transform]),
                scene,
                bpy_module=bpy,
                available_probe=lambda: 64 * 1024**3,
            )
            collection = outcome.collection
            assert outcome.created
            assert (
                scene.render.resolution_x,
                scene.render.resolution_y,
                scene.render.resolution_percentage,
            ) == before_resolution, transform
            assert (
                collection.get("lingbot_map_source_background_status")
                == "unattached-scene-aspect-mismatch"
            )
            message = importer.set_scene_resolution_to_source(
                collection, scene, bpy_module=bpy
            )
            resolution_actions.append((transform, message))
        imported[transform] = collection
        contract = importer.imported_source_view_contract(collection)
        camera = importer.find_reconstruction_camera(collection)
        trajectory = next(
            item
            for item in collection.objects
            if item.get("lingbot_map_kind") == "camera_trajectory"
        )
        root = next(
            item
            for item in collection.objects
            if item.get("lingbot_map_kind") == "reconstruction_root"
        )
        point_cloud = next(
            item
            for item in collection.objects
            if item.get("lingbot_map_kind") == "point_cloud"
        )
        assert len(collection.objects) == 4
        assert camera.parent == root
        assert trajectory.parent == root
        assert point_cloud.parent == root
        assert trajectory.hide_render
        assert tuple(root.scale) == (1.0, 1.0, 1.0)
        result = Path(INDEX["results"][transform])
        expected_camera = np.load(
            result / "arrays" / "camera_to_world.npy"
        )
        source_intrinsics = np.load(
            result / "arrays" / "source_intrinsics.npy"
        )
        for index in range(FRAME_COUNT):
            scene.frame_set(TIMELINE_START + index)
            bpy.context.view_layer.update()
            np.testing.assert_allclose(
                np.asarray(camera.matrix_local),
                expected_camera[index],
                atol=1e-5,
            )
            expected_lens = (
                36.0 * source_intrinsics[index, 0, 0] / contract.width
            )
            assert math.isclose(
                camera.data.lens,
                float(expected_lens),
                abs_tol=1e-5,
                rel_tol=1e-6,
            )
            expected_shift_x = (
                contract.width * 0.5
                - float(source_intrinsics[index, 0, 2])
            ) / contract.width
            expected_shift_y = (
                float(source_intrinsics[index, 1, 2])
                - contract.height * 0.5
            ) / contract.width
            assert math.isclose(
                camera.data.shift_x, expected_shift_x, abs_tol=1e-6
            )
            assert math.isclose(
                camera.data.shift_y, expected_shift_y, abs_tol=1e-6
            )
            local_axis = Vector((0.0, 0.0, -1.0))
            optical_world = camera.matrix_world @ local_axis
            projected = world_to_camera_view(
                scene, camera, optical_world
            )
            expected_x = float(source_intrinsics[index, 0, 2]) / contract.width
            expected_y = (
                1.0
                - float(source_intrinsics[index, 1, 2]) / contract.height
            )
            assert (
                abs(projected.x - expected_x) * contract.width < 1.0
            )
            assert (
                abs(projected.y - expected_y) * contract.height < 1.0
            )
        for datablock in (camera, camera.data):
            for fcurve in _action_fcurves(datablock):
                assert all(
                    point.interpolation == "LINEAR"
                    for point in fcurve.keyframe_points
                )
        assert camera.data.sensor_fit == "HORIZONTAL"
        assert camera.data.sensor_width == 36.0
        diagonal = np.linalg.norm(
            np.max(
                np.concatenate(
                    (
                        np.load(result / "arrays" / "positions.npy"),
                        expected_camera[:, :3, 3],
                    )
                ),
                axis=0,
            )
            - np.min(
                np.concatenate(
                    (
                        np.load(result / "arrays" / "positions.npy"),
                        expected_camera[:, :3, 3],
                    )
                ),
                axis=0,
            )
        )
        expected_clip_start = max(float(diagonal) * 1e-5, 1e-6)
        expected_clip_end = max(
            float(diagonal) * 1.1, expected_clip_start * 1000.0
        )
        assert math.isclose(
            camera.data.clip_start, expected_clip_start, rel_tol=1e-5
        )
        assert math.isclose(
            camera.data.clip_end, expected_clip_end, rel_tol=1e-5
        )
        trajectory_points = np.empty((FRAME_COUNT, 4), dtype=np.float64)
        trajectory.data.splines[0].points.foreach_get(
            "co", trajectory_points.reshape(-1)
        )
        np.testing.assert_allclose(
            trajectory_points[:, :3], expected_camera[:, :3, 3]
        )
        np.testing.assert_allclose(trajectory_points[:, 3], 1.0)

        backgrounds = tuple(camera.data.background_images)
        assert len(backgrounds) == 1
        background = backgrounds[0]
        mapping = source_view.BACKGROUND_MAPPINGS[transform]
        assert background.source == "MOVIE_CLIP"
        assert background.clip.frame_start == TIMELINE_START
        assert background.clip.frame_duration == FRAME_COUNT
        assert tuple(background.clip.size) == contract.coded_size
        assert background.frame_method == "FIT"
        assert not background.show_background_image
        assert background.display_depth == "FRONT"
        assert tuple(background.offset) == (0.0, 0.0)
        assert math.isclose(
            background.scale,
            source_view.background_scale(contract),
            abs_tol=1e-6,
        )
        assert math.isclose(
            background.rotation, mapping.rotation_radians, abs_tol=1e-6
        )
        assert background.use_flip_x == mapping.flip_x
        assert background.use_flip_y == mapping.flip_y
        source = np.arange(
            contract.coded_size[0] * contract.coded_size[1],
            dtype=np.int32,
        ).reshape((contract.coded_size[1], contract.coded_size[0]))
        expected_display = _reference_transform(source, transform)
        actual_display = np.empty_like(expected_display)
        for coded_y in range(contract.coded_size[1]):
            for coded_x in range(contract.coded_size[0]):
                display_x, display_y = source_view.transform_pixel(
                    transform,
                    coded_x,
                    coded_y,
                    *contract.coded_size,
                )
                actual_display[display_y, display_x] = source[
                    coded_y, coded_x
                ]
        np.testing.assert_array_equal(actual_display, expected_display)
        assert actual_display.shape == (contract.height, contract.width)

        coverage = source_view.coverage_to_camera_border(
            contract, (0.0, 0.0, float(contract.width), float(contract.height))
        )
        for source_point, display_point in zip(
            contract.coverage_polygon, coverage
        ):
            expected_point = (
                source_point[0],
                contract.height - source_point[1],
            )
            assert abs(display_point[0] - expected_point[0]) <= 1.0
            assert abs(display_point[1] - expected_point[1]) <= 1.0

        repeated = importer.import_result(
            result,
            scene,
            bpy_module=bpy,
            available_probe=lambda: 64 * 1024**3,
        )
        assert not repeated.created and repeated.collection == collection

    assert scene.camera == unrelated_camera
    assert scene.render.fps == 23 and scene.render.fps_base == 1.0
    assert scene.frame_start == 5, (
        scene.frame_start,
        scene.frame_end,
        scene.frame_preview_start,
        scene.frame_preview_end,
    )
    assert scene.frame_end == TIMELINE_START + FRAME_COUNT - 1
    assert scene.frame_preview_start == -3
    assert scene.frame_preview_end == 8
    assert scene.use_preview_range
    warning_codes = {
        warning["code"]
        for warning in json.loads(
            imported["identity"]["lingbot_map_warnings_json"]
        )
    }
    assert {
        "source-intrinsics-axis-disagreement",
        "source-focal-discontinuity",
    }.issubset(warning_codes)
    marker = {
        "blender_version": bpy.app.version_string,
        "display_transforms": list(TRANSFORMS),
        "timeline_start": TIMELINE_START,
        "frame_count": FRAME_COUNT,
        "first_middle_last": [
            TIMELINE_START,
            TIMELINE_START + 1,
            TIMELINE_START + 2,
        ],
        "camera_pose_lens_shift_linear": True,
        "trajectory_every_pose_render_hidden": True,
        "source_background_viewport_only_hidden": True,
        "sequential_non_cycling": True,
        "explicit_resolution_actions": len(resolution_actions),
        "missing_media_core_usable": True,
        "wrong_checksum_relink_rejected": True,
        "exact_checksum_relink_accepted": True,
        "background_pixel_oracle_max_error": 0,
        "coverage_pixel_oracle_max_error": 0,
        "nonzero_timeline_start": True,
        "fps_preserved": True,
        "scene_camera_preserved": True,
        "cancellation_phases": list(importer.IMPORT_PHASES),
    }
    (FIXTURE_ROOT / "blender-marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    print("LINGBOT_MAP_BLENDER_SOURCE_VIEW=" + json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
