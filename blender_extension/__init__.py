"""LingBot Map Reconstruction Blender Extension shell."""

from __future__ import annotations

import bpy

from .host import probe_supported_host
from .runtime import cancel_runtime_setup, clear_host_decision, get_host_decision, set_host_decision
from .ui import CLASSES


_registered_classes: list[type] = []


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
    except Exception:
        _unregister_classes()
        clear_host_decision()
        raise


def unregister() -> None:
    """Cleanly remove every registered class in reverse order."""

    cancel_runtime_setup()
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


__all__ = ["get_host_decision", "register", "unregister"]
