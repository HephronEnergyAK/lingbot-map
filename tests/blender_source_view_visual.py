"""Visible Blender 5.2 viewport oracle for every source Display Transform."""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import types

import bpy
from bpy_extras.view3d_utils import location_3d_to_region_2d
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(
    os.environ.get(
        "LINGBOT_MAP_SOURCE_VIEW_FIXTURE",
        r"C:\tmp\lingbot-map-source-view15",
    )
).resolve()
OUTPUT = FIXTURE / "visual"
STATE = {
    "cases": [],
    "case_index": 0,
    "collections": {},
    "errors": [],
    "background_max_error": 0.0,
    "coverage_max_error": 0.0,
}


def _datablock_counts():
    return {
        "collections": len(bpy.data.collections),
        "objects": len(bpy.data.objects),
        "pointclouds": len(bpy.data.pointclouds),
        "cameras": len(bpy.data.cameras),
        "curves": len(bpy.data.curves),
        "movieclips": len(bpy.data.movieclips),
        "materials": len(bpy.data.materials),
        "node_groups": len(bpy.data.node_groups),
    }


def _camera_border(camera, scene, region, region_3d):
    points = []
    for corner in camera.data.view_frame(scene=scene):
        point = location_3d_to_region_2d(
            region, region_3d, camera.matrix_world @ corner
        )
        if point is None:
            raise AssertionError("Camera View border could not be projected")
        points.append(
            (float(region.x + point.x), float(region.y + point.y))
        )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _expected_bbox(source_view, transform, coded_size, rectangle):
    x0, y0, x1, y1 = rectangle
    pixels = [
        source_view.transform_pixel(
            transform, x, y, coded_size[0], coded_size[1]
        )
        for y in range(y0, y1)
        for x in range(x0, x1)
    ]
    return (
        float(min(point[0] for point in pixels)),
        float(min(point[1] for point in pixels)),
        float(max(point[0] for point in pixels) + 1),
        float(max(point[1] for point in pixels) + 1),
    )


def _actual_bbox(mask, border, display_size):
    y_indices, x_indices = np.nonzero(mask)
    if not len(x_indices):
        raise AssertionError("Expected visual marker is absent")
    left, bottom, right, top = border
    width, height = display_size
    screen_left = float(x_indices.min())
    screen_right = float(x_indices.max() + 1)
    screen_bottom = float(y_indices.min())
    screen_top = float(y_indices.max() + 1)
    return (
        (screen_left - left) * width / (right - left),
        (top - screen_top) * height / (top - bottom),
        (screen_right - left) * width / (right - left),
        (top - screen_bottom) * height / (top - bottom),
    )


def _bbox_error(actual, expected):
    return max(abs(left - right) for left, right in zip(actual, expected))


def _point_segment_projection(point, start, end):
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return 0.0, math.hypot(px - ax, py - ay)
    amount = max(
        0.0,
        min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared),
    )
    return (
        amount,
        math.hypot(px - (ax + amount * dx), py - (ay + amount * dy)),
    )


def _point_segment_distance(point, start, end):
    return _point_segment_projection(point, start, end)[1]


def _analyze_screenshot(path, transform, frame_index, guide_enabled):
    importer = STATE["importer"]
    source_view = STATE["source_view"]
    collection = STATE["collections"][transform]
    contract = importer.imported_source_view_contract(collection)
    camera = importer.find_reconstruction_camera(collection)
    scene = STATE["scene"]
    region = STATE["region"]
    region_3d = STATE["space"].region_3d
    border = _camera_border(camera, scene, region, region_3d)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image_width, image_height = tuple(int(value) for value in image.size)
        values = np.empty(image_width * image_height * 4, dtype=np.float32)
        image.pixels.foreach_get(values)
        pixels = values.reshape((image_height, image_width, 4))[:, :, :3]
    finally:
        bpy.data.images.remove(image)
    left, bottom, right, top = border
    xs = np.arange(image_width, dtype=np.float64) + 0.5
    ys = np.arange(image_height, dtype=np.float64) + 0.5
    inside = (
        (ys[:, None] >= bottom)
        & (ys[:, None] <= top)
        & (xs[None, :] >= left)
        & (xs[None, :] <= right)
    )
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    masks = {
        "red": inside & (red > 0.65) & (green < 0.35) & (blue < 0.35),
        "green": inside & (green > 0.65) & (red < 0.35) & (blue < 0.35),
        "blue": inside & (blue > 0.55) & (red < 0.35) & (green < 0.55),
        "yellow": inside & (red > 0.65) & (green > 0.65) & (blue < 0.35),
        "white": inside & (red > 0.82) & (green > 0.82) & (blue > 0.82),
    }
    coded_width, coded_height = contract.coded_size
    rectangles = {
        "red": (0, 0, 8, 8),
        "green": (coded_width - 8, 0, coded_width, 8),
        "blue": (0, coded_height - 8, 8, coded_height),
        "yellow": (
            coded_width - 8,
            coded_height - 8,
            coded_width,
            coded_height,
        ),
        "white": (
            16 + frame_index * 12,
            12,
            20 + frame_index * 12,
            36,
        ),
    }
    background_error = 0.0
    if not guide_enabled:
        errors = {}
        display_x = (xs - left) * contract.width / (right - left)
        display_y = (top - ys) * contract.height / (top - bottom)
        for name, rectangle in rectangles.items():
            expected = _expected_bbox(
                source_view, transform, contract.coded_size, rectangle
            )
            marker_mask = masks[name]
            if name == "white":
                marker_mask = marker_mask & (
                    (display_y[:, None] >= expected[1] - 2.0)
                    & (display_y[:, None] <= expected[3] + 2.0)
                    & (display_x[None, :] >= expected[0] - 2.0)
                    & (display_x[None, :] <= expected[2] + 2.0)
                )
            actual = _actual_bbox(
                marker_mask,
                border,
                (contract.width, contract.height),
            )
            errors[name] = {
                "error": _bbox_error(actual, expected),
                "actual": actual,
                "expected": expected,
            }
        background_error = max(item["error"] for item in errors.values())
        if background_error > 1.0:
            raise AssertionError(
                f"{transform} frame {frame_index} background error "
                f"{background_error:.3f} display pixels: {errors}"
            )
        STATE["background_max_error"] = max(
            STATE["background_max_error"], background_error
        )

    coverage_error = 0.0
    if guide_enabled:
        orange = (
            inside
            & (red > 0.70)
            & (green > 0.15)
            & (green < 0.65)
            & (blue < 0.25)
        )
        screen_y, screen_x = np.nonzero(orange)
        if not len(screen_x):
            raise AssertionError(f"{transform} Model Coverage guide is absent")
        points = [
            (
                (float(x) + 0.5 - left)
                * contract.width
                / (right - left),
                (top - (float(y) + 0.5))
                * contract.height
                / (top - bottom),
            )
            for x, y in zip(screen_x, screen_y)
        ]
        polygon = contract.coverage_polygon
        segments = tuple(
            (polygon[index], polygon[(index + 1) % len(polygon)])
            for index in range(len(polygon))
        )
        coverage_error = max(
            min(
                _point_segment_distance(point, start, end)
                for start, end in segments
            )
            for point in points
        )
        # The guide is alpha-blended over the decoder's colored corner
        # markers. Those exact corner pixels need not satisfy the orange mask,
        # so endpoint proximity is not an alignment oracle. Require every
        # segment to be visible through its middle half instead, while the
        # maximum perpendicular error above retains the exact one-display-pixel
        # alignment bound.
        for start, end in segments:
            amounts = [
                amount
                for point in points
                for amount, distance in (
                    _point_segment_projection(point, start, end),
                )
                if distance <= 1.0
            ]
            if not any(0.25 <= amount <= 0.75 for amount in amounts):
                raise AssertionError(
                    f"{transform} Model Coverage segment is absent"
                )
        if coverage_error > 1.0:
            raise AssertionError(
                f"{transform} coverage error {coverage_error:.3f} "
                "display pixels"
            )
        STATE["coverage_max_error"] = max(
            STATE["coverage_max_error"], coverage_error
        )
    return {
        "transform": transform,
        "frame_index": frame_index,
        "background_error": background_error,
        "coverage_error": coverage_error,
        "camera_border": list(border),
        "screenshot": str(path),
    }


def _prepare_case():
    if STATE["case_index"] >= len(STATE["cases"]):
        _finish()
        return None
    transform, frame_index, guide_enabled = STATE["cases"][
        STATE["case_index"]
    ]
    importer = STATE["importer"]
    collection = STATE["collections"][transform]
    camera = importer.find_reconstruction_camera(collection)
    scene = STATE["scene"]
    contract = importer.imported_source_view_contract(collection)
    for current in STATE["collections"].values():
        current_camera = importer.find_reconstruction_camera(current)
        backgrounds = tuple(current_camera.data.background_images)
        if backgrounds:
            backgrounds[0].show_background_image = current is collection
    scene.camera = camera
    scene.render.resolution_x = contract.width
    scene.render.resolution_y = contract.height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.frame_set(int(STATE["index"]["timeline_start"]) + frame_index)
    STATE["space"].region_3d.view_perspective = "CAMERA"
    # Keep portrait and landscape camera borders fully inside the screenshot.
    STATE["space"].region_3d.view_camera_zoom = -10.0
    STATE["space"].region_3d.view_camera_offset = (0.0, 0.0)
    if guide_enabled:
        before = _datablock_counts()
        with bpy.context.temp_override(
            window=STATE["window"],
            screen=STATE["window"].screen,
            area=STATE["area"],
            region=STATE["region"],
            scene=scene,
        ):
            shown = importer.toggle_model_coverage_guide(
                bpy.context, collection
            )
        if not shown or _datablock_counts() != before:
            raise AssertionError(
                "Model Coverage guide created a datablock or did not enable"
            )
    STATE["current"] = (transform, frame_index, guide_enabled, camera)
    STATE["area"].tag_redraw()
    bpy.app.timers.register(_capture_case, first_interval=0.25)
    return None


def _capture_case():
    transform, frame_index, guide_enabled, _camera = STATE["current"]
    path = OUTPUT / f"{transform}-{frame_index}.png"
    try:
        with bpy.context.temp_override(
            window=STATE["window"],
            screen=STATE["window"].screen,
            area=STATE["area"],
            region=STATE["region"],
            scene=STATE["scene"],
        ):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
            result = bpy.ops.screen.screenshot(
                filepath=str(path), hide_props_region=True
            )
        if result != {"FINISHED"} or not path.is_file():
            raise AssertionError("Blender screenshot operator did not finish")
        STATE.setdefault("results", []).append(
            _analyze_screenshot(
                path, transform, frame_index, guide_enabled
            )
        )
        if guide_enabled:
            before = _datablock_counts()
            collection = STATE["collections"][transform]
            with bpy.context.temp_override(
                window=STATE["window"],
                screen=STATE["window"].screen,
                area=STATE["area"],
                region=STATE["region"],
                scene=STATE["scene"],
            ):
                shown = STATE["importer"].toggle_model_coverage_guide(
                    bpy.context, collection
                )
            if shown or _datablock_counts() != before:
                raise AssertionError(
                    "Model Coverage guide cleanup mutated a datablock"
                )
        STATE["case_index"] += 1
        bpy.app.timers.register(_prepare_case, first_interval=0.05)
    except Exception as exc:
        _fail(exc)
    return None


def _setup():
    index = json.loads((FIXTURE / "index.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(exist_ok=True)
    installed = os.environ.get("LINGBOT_MAP_INSTALLED_EXTENSION") == "1"
    if installed:
        package_name = "bl_ext.user_default.lingbot_map_reconstruction"
    else:
        package_name = "lingbot_map_issue15_visual"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "blender_extension")]
        sys.modules[package_name] = package
    importer = importlib.import_module(package_name + ".result_import")
    source_view = importlib.import_module(package_name + ".source_view")
    module_path = Path(importer.__file__).resolve()
    if installed:
        assert not module_path.is_relative_to(ROOT.resolve()), module_path
    scene = bpy.context.scene
    scene["lingbot_map_scene_uuid"] = index["scene_uuid"]
    bpy.ops.wm.save_as_mainfile(filepath=index["blend"])
    scene.render.resolution_x = 111
    scene.render.resolution_y = 111
    scene.render.resolution_percentage = 100
    for transform, directory in index["results"].items():
        outcome = importer.import_result(
            directory,
            scene,
            bpy_module=bpy,
            available_probe=lambda: 64 * 1024**3,
        )
        importer.set_scene_resolution_to_source(
            outcome.collection, scene, bpy_module=bpy
        )
        importer.set_source_background_visibility(
            outcome.collection, False
        )
        STATE["collections"][transform] = outcome.collection
        for item in outcome.collection.objects:
            if item.get("lingbot_map_kind") in {
                "point_cloud",
                "camera_trajectory",
            }:
                item.hide_viewport = True
    imported_objects = {
        item
        for collection in STATE["collections"].values()
        for item in collection.objects
    }
    for item in tuple(scene.collection.objects):
        if item not in imported_objects:
            item.hide_viewport = True
    window = bpy.context.window_manager.windows[0]
    area = next(item for item in window.screen.areas if item.type == "VIEW_3D")
    region = next(item for item in area.regions if item.type == "WINDOW")
    space = area.spaces.active
    space.overlay.show_overlays = True
    space.overlay.show_floor = False
    space.overlay.show_axis_x = False
    space.overlay.show_axis_y = False
    space.overlay.show_axis_z = False
    space.overlay.show_cursor = False
    space.overlay.show_text = False
    space.overlay.show_extras = False
    space.show_gizmo = False
    STATE.update(
        {
            "index": index,
            "importer": importer,
            "source_view": source_view,
            "installed_extension": installed,
            "extension_module_path": str(module_path),
            "scene": scene,
            "window": window,
            "area": area,
            "region": region,
            "space": space,
            "cases": [
                (transform, frame_index, False)
                for transform in index["results"]
                for frame_index in range(int(index["frame_count"]))
            ]
            + [
                (transform, 0, True)
                for transform in index["results"]
            ],
        }
    )
    bpy.app.timers.register(_prepare_case, first_interval=0.25)


def _finish():
    marker = {
        "state": "succeeded",
        "blender_version": bpy.app.version_string,
        "display_transforms": list(STATE["index"]["results"]),
        "frames": [0, 1, 2],
        "screenshots": len(STATE["results"]),
        "background_max_error_display_pixels": STATE["background_max_error"],
        "coverage_max_error_display_pixels": STATE["coverage_max_error"],
        "temporary_guide_datablocks": 0,
        "installed_extension": STATE["installed_extension"],
        "extension_module_path": STATE["extension_module_path"],
        "results": STATE["results"],
    }
    (FIXTURE / "visual-marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    print("LINGBOT_MAP_BLENDER_SOURCE_VIEW_VISUAL=" + json.dumps(marker, sort_keys=True))
    bpy.ops.wm.quit_blender()


def _fail(exc):
    marker = {
        "state": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "case_index": STATE.get("case_index"),
        "current": list(STATE.get("current", ()))[:3],
    }
    (FIXTURE / "visual-marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    print("LINGBOT_MAP_BLENDER_SOURCE_VIEW_VISUAL=" + json.dumps(marker, sort_keys=True))
    try:
        STATE.get("importer").clear_model_coverage_guides()
    except Exception:
        pass
    bpy.ops.wm.quit_blender()


def _setup_guarded():
    try:
        _setup()
    except Exception as exc:
        _fail(exc)
    return None


if __name__ == "__main__":
    # The Windows runner supplies an isolated userpref.blend with the splash
    # disabled. Blender timers do not advance while a fresh-profile splash owns
    # the temporary window, and the splash-close operator is not registered
    # early enough for a --python script to close it reliably.
    bpy.app.timers.register(_setup_guarded, first_interval=0.5)
