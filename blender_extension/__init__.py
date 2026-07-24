"""LingBot Map Reconstruction Blender Extension shell."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from .host import probe_supported_host
from .runtime import (
    cancel_model_setup,
    cancel_runtime_setup,
    clear_host_decision,
    get_host_decision,
    set_host_decision,
)
from .ui import CLASSES
from .gpu_capability import shutdown_gpu_capability
from .job_lifecycle import (
    CAPTURE_SOURCE_PROPERTY,
    CAMERA_ITERATIONS_PROPERTY,
    CONFIDENCE_CUTOFF_PROPERTY,
    DEPTH_CUTOFF_PROPERTY,
    POINT_BUDGET_CONFIRMED_PROPERTY,
    POINT_BUDGET_PROPERTY,
    PROFILE_PROPERTY,
    RETAIN_DENSE_PROPERTY,
    SKY_MASK_PROPERTY,
    detach_job_monitor,
    get_job_snapshot,
    recover_jobs_for_blend,
    report_job_recovery_error,
)
from .result_import import (
    clear_model_coverage_guides,
    ImportCapacityError,
    ResultImportError,
    attempt_auto_import_once,
    set_import_status,
)
from .results import discover_ready_results


_registered_classes: list[type] = []
_last_completed_job_for_auto_import: str | None = None
_PROFILE_GUARD = "_lingbot_map_profile_update"
_PROFILE_DEFAULTS = {
    "Draft": (1, 70.0, 99.5, 1_000_000),
    "Balanced": (4, 50.0, 99.5, 5_000_000),
    "High": (4, 30.0, 99.5, 10_000_000),
}


def _apply_named_profile(scene, _context) -> None:
    name = str(getattr(scene, PROFILE_PROPERTY, "Draft"))
    if name not in _PROFILE_DEFAULTS:
        return
    scene[_PROFILE_GUARD] = True
    try:
        camera, confidence, depth, budget = _PROFILE_DEFAULTS[name]
        setattr(scene, CAMERA_ITERATIONS_PROPERTY, camera)
        setattr(scene, CONFIDENCE_CUTOFF_PROPERTY, confidence)
        setattr(scene, DEPTH_CUTOFF_PROPERTY, depth)
        setattr(scene, POINT_BUDGET_PROPERTY, budget)
        setattr(scene, POINT_BUDGET_CONFIRMED_PROPERTY, False)
        setattr(scene, RETAIN_DENSE_PROPERTY, False)
    finally:
        scene.pop(_PROFILE_GUARD, None)


def _mark_profile_custom(scene, _context) -> None:
    if not scene.get(_PROFILE_GUARD, False):
        setattr(scene, PROFILE_PROPERTY, "Custom")


def _persistent(function):
    decorator = getattr(getattr(bpy.app, "handlers", None), "persistent", None)
    return decorator(function) if decorator is not None else function


@_persistent
def _recover_jobs_after_load(_unused) -> None:
    clear_model_coverage_guides()
    filepath = getattr(bpy.data, "filepath", "")
    if filepath:
        try:
            recover_jobs_for_blend(filepath)
        except Exception as exc:
            # Invalid project IPC is visible but never allowed to break file loading.
            report_job_recovery_error(exc)


def _job_ui_timer():
    _attempt_completed_result_auto_import()
    for window in getattr(bpy.context.window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    return 0.5


def _attempt_completed_result_auto_import() -> None:
    """Attempt only on a newly observed completion, never on a later idle tick."""

    global _last_completed_job_for_auto_import
    snapshot = get_job_snapshot()
    if snapshot.state != "succeeded" or not snapshot.job_id:
        return
    if snapshot.job_id == _last_completed_job_for_auto_import:
        return
    _last_completed_job_for_auto_import = snapshot.job_id
    filepath = str(getattr(bpy.data, "filepath", ""))
    if not filepath:
        return
    try:
        matches = tuple(
            item
            for item in discover_ready_results(filepath)
            if item.job_id == snapshot.job_id
        )
        if len(matches) != 1:
            set_import_status(
                "Automatic import not attempted: completed Job has no unique Ready Result"
            )
            return
        outcome = attempt_auto_import_once(
            matches[0],
            expected_job_id=snapshot.job_id,
            bpy_module=bpy,
        )
        if outcome is None:
            set_import_status(
                "Ready to Import: automatic import gates did not all pass"
            )
    except (ImportCapacityError, ResultImportError, OSError, MemoryError) as exc:
        set_import_status(f"Ready to Import: automatic import blocked: {exc}")
    except Exception as exc:
        set_import_status(
            "Ready to Import: automatic import failed and was rolled back: "
            f"{type(exc).__name__}: {exc}"
        )


def register() -> None:
    """Evaluate the host, then register only the non-mutating UI shell."""

    if _registered_classes:
        return

    global _last_completed_job_for_auto_import
    decision = probe_supported_host(tuple(bpy.app.version))
    set_host_decision(decision)
    try:
        for extension_class in CLASSES:
            bpy.utils.register_class(extension_class)
            _registered_classes.append(extension_class)
        scene_type = getattr(bpy.types, "Scene", None)
        if scene_type is not None and not hasattr(scene_type, CAPTURE_SOURCE_PROPERTY):
            setattr(
                scene_type,
                CAPTURE_SOURCE_PROPERTY,
                StringProperty(
                    name="Capture Source",
                    description="Scene-owned MP4 or MOV Job Draft path",
                    default="",
                ),
            )
        if scene_type is not None and not hasattr(scene_type, PROFILE_PROPERTY):
            setattr(
                scene_type,
                PROFILE_PROPERTY,
                EnumProperty(
                    name="Reconstruction Profile",
                    items=(
                        ("Draft", "Draft", "70% confidence, 1 iteration, 1M points"),
                        ("Balanced", "Balanced", "50% confidence, 4 iterations, 5M points"),
                        ("High", "High", "30% confidence, 4 iterations, 10M points"),
                        ("Custom", "Custom", "Explicit advanced settings"),
                    ),
                    default="Draft",
                    update=_apply_named_profile,
                ),
            )
            setattr(scene_type, CAMERA_ITERATIONS_PROPERTY, IntProperty(name="Camera Iterations", default=1, min=1, max=4, update=_mark_profile_custom))
            setattr(scene_type, CONFIDENCE_CUTOFF_PROPERTY, FloatProperty(name="Confidence Cutoff", default=70.0, min=0.0, max=100.0, update=_mark_profile_custom))
            setattr(scene_type, DEPTH_CUTOFF_PROPERTY, FloatProperty(name="Depth Cutoff", default=99.5, min=0.0, max=100.0, update=_mark_profile_custom))
            setattr(scene_type, POINT_BUDGET_PROPERTY, IntProperty(name="Import Point Budget", default=1_000_000, min=1, max=50_000_000, update=_mark_profile_custom))
            setattr(scene_type, POINT_BUDGET_CONFIRMED_PROPERTY, BoolProperty(name="Confirm Large Budget", default=False, update=_mark_profile_custom))
            setattr(scene_type, RETAIN_DENSE_PROPERTY, BoolProperty(name="Retain Dense Predictions", description="Retain finalized aligned depth and raw confidence in optional chunks", default=False, update=_mark_profile_custom))
        if scene_type is not None and not hasattr(scene_type, SKY_MASK_PROPERTY):
            setattr(
                scene_type,
                SKY_MASK_PROPERTY,
                BoolProperty(
                    name="Sky Masking",
                    description=(
                        "Filter sky point candidates on every frame without changing "
                        "Reconstruction Model input; not Dynamic Content removal"
                    ),
                    default=False,
                ),
            )
        handlers = getattr(bpy.app, "handlers", None)
        timers = getattr(bpy.app, "timers", None)
        if handlers is not None and _recover_jobs_after_load not in handlers.load_post:
            handlers.load_post.append(_recover_jobs_after_load)
        if timers is not None and not timers.is_registered(_job_ui_timer):
            timers.register(_job_ui_timer, first_interval=0.1, persistent=True)
        if hasattr(bpy, "data"):
            _recover_jobs_after_load(None)
            snapshot = get_job_snapshot()
            _last_completed_job_for_auto_import = (
                snapshot.job_id if snapshot.state == "succeeded" else None
            )
    except Exception:
        _unregister_scene_property()
        _unregister_classes()
        clear_host_decision()
        raise


def unregister() -> None:
    """Cleanly remove every registered class in reverse order."""

    global _last_completed_job_for_auto_import
    cancel_runtime_setup()
    cancel_model_setup()
    shutdown_gpu_capability()
    detach_job_monitor()
    clear_model_coverage_guides()
    _unregister_scene_property()
    handlers = getattr(bpy.app, "handlers", None)
    timers = getattr(bpy.app, "timers", None)
    if handlers is not None and _recover_jobs_after_load in handlers.load_post:
        handlers.load_post.remove(_recover_jobs_after_load)
    if timers is not None and timers.is_registered(_job_ui_timer):
        timers.unregister(_job_ui_timer)
    _unregister_classes()
    _last_completed_job_for_auto_import = None
    clear_host_decision()


def _unregister_classes() -> None:
    first_error = None
    while _registered_classes:
        extension_class = _registered_classes.pop()
        try:
            bpy.utils.unregister_class(extension_class)
        except Exception as exc:  # Finish cleanup before surfacing one error.
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _unregister_scene_property() -> None:
    scene_type = getattr(bpy.types, "Scene", None)
    if scene_type is not None:
        for name in (
            CAPTURE_SOURCE_PROPERTY,
            PROFILE_PROPERTY,
            CAMERA_ITERATIONS_PROPERTY,
            CONFIDENCE_CUTOFF_PROPERTY,
            DEPTH_CUTOFF_PROPERTY,
            POINT_BUDGET_PROPERTY,
            POINT_BUDGET_CONFIRMED_PROPERTY,
            RETAIN_DENSE_PROPERTY,
            SKY_MASK_PROPERTY,
        ):
            if hasattr(scene_type, name):
                delattr(scene_type, name)


__all__ = ["get_host_decision", "register", "unregister"]
