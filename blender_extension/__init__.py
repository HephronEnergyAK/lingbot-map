"""LingBot Map Reconstruction Blender Extension shell."""

from __future__ import annotations

import bpy
from bpy.props import StringProperty

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
    detach_job_monitor,
    recover_jobs_for_blend,
    report_job_recovery_error,
)


_registered_classes: list[type] = []


def _persistent(function):
    decorator = getattr(getattr(bpy.app, "handlers", None), "persistent", None)
    return decorator(function) if decorator is not None else function


@_persistent
def _recover_jobs_after_load(_unused) -> None:
    filepath = getattr(bpy.data, "filepath", "")
    if filepath:
        try:
            recover_jobs_for_blend(filepath)
        except Exception as exc:
            # Invalid project IPC is visible but never allowed to break file loading.
            report_job_recovery_error(exc)


def _job_ui_timer():
    for window in getattr(bpy.context.window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    return 0.5


def register() -> None:
    """Evaluate the host, then register only the non-mutating UI shell."""

    if _registered_classes:
        return

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
        handlers = getattr(bpy.app, "handlers", None)
        timers = getattr(bpy.app, "timers", None)
        if handlers is not None and _recover_jobs_after_load not in handlers.load_post:
            handlers.load_post.append(_recover_jobs_after_load)
        if timers is not None and not timers.is_registered(_job_ui_timer):
            timers.register(_job_ui_timer, first_interval=0.1, persistent=True)
        if hasattr(bpy, "data"):
            _recover_jobs_after_load(None)
    except Exception:
        _unregister_scene_property()
        _unregister_classes()
        clear_host_decision()
        raise


def unregister() -> None:
    """Cleanly remove every registered class in reverse order."""

    cancel_runtime_setup()
    cancel_model_setup()
    shutdown_gpu_capability()
    detach_job_monitor()
    _unregister_scene_property()
    handlers = getattr(bpy.app, "handlers", None)
    timers = getattr(bpy.app, "timers", None)
    if handlers is not None and _recover_jobs_after_load in handlers.load_post:
        handlers.load_post.remove(_recover_jobs_after_load)
    if timers is not None and timers.is_registered(_job_ui_timer):
        timers.unregister(_job_ui_timer)
    _unregister_classes()
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
    if scene_type is not None and hasattr(scene_type, CAPTURE_SOURCE_PROPERTY):
        delattr(scene_type, CAPTURE_SOURCE_PROPERTY)


__all__ = ["get_host_decision", "register", "unregister"]
