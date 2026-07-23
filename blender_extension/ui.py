"""Blender Preferences and explicit lifecycle actions."""

from pathlib import Path

import bpy
from bpy.props import BoolProperty, StringProperty

from .runtime import (
    cancel_model_setup,
    cancel_runtime_setup,
    get_host_decision,
    get_model_setup_snapshot,
    get_setup_snapshot,
    start_model_download,
    start_model_import,
    start_runtime_setup,
)
from .model_store import ModelStore, bundled_model_catalog
from .runtime_setup import RuntimeSetupError, default_managed_root


class LINGBOTMAP_Preferences(bpy.types.AddonPreferences):
    """Machine-scoped deployment choices; never Project or Job truth."""

    bl_idname = __package__

    runtime_root: StringProperty(
        name="Managed Runtime Root",
        description="Machine-specific location for immutable Worker Runtimes and models",
        subtype="DIR_PATH",
        default="",
    )
    gpu_uuid: StringProperty(
        name="GPU UUID",
        description="Persistent machine-specific NVIDIA GPU selection",
        default="",
    )
    offline_setup: BoolProperty(
        name="Offline Setup",
        description="Require Setup to use only already verified local artifacts",
        default=False,
    )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "runtime_root")
        layout.prop(self, "gpu_uuid")
        layout.prop(self, "offline_setup")


class LINGBOTMAP_OT_setup_runtime(bpy.types.Operator):
    bl_idname = "lingbot_map.setup_runtime"
    bl_label = "Setup Worker Runtime"
    bl_description = "Explicitly provision the immutable Worker Runtime"

    @classmethod
    def poll(cls, _context):
        decision = get_host_decision()
        return bool(
            decision
            and decision.supported
            and get_setup_snapshot().state not in {"running", "cancelling"}
        )

    def execute(self, context):
        preferences = context.preferences.addons[__package__].preferences
        root = Path(preferences.runtime_root) if preferences.runtime_root.strip() else None
        try:
            start_runtime_setup(
                root,
                offline=bool(preferences.offline_setup),
                online_access=bool(bpy.app.online_access),
            )
        except RuntimeSetupError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_cancel_runtime_setup(bpy.types.Operator):
    bl_idname = "lingbot_map.cancel_runtime_setup"
    bl_label = "Cancel Runtime Setup"

    @classmethod
    def poll(cls, _context):
        return get_setup_snapshot().state in {"running", "cancelling"}

    def execute(self, _context):
        return {"FINISHED"} if cancel_runtime_setup() else {"CANCELLED"}


def _preferences(context):
    return context.preferences.addons[__package__].preferences


def _managed_root(preferences):
    return Path(preferences.runtime_root) if preferences.runtime_root.strip() else default_managed_root()


class LINGBOTMAP_OT_download_model(bpy.types.Operator):
    bl_idname = "lingbot_map.download_model"
    bl_label = "Download Catalogued Model"

    model_id: StringProperty(name="Model ID", default="")

    @classmethod
    def poll(cls, _context):
        decision = get_host_decision()
        return bool(
            decision
            and decision.supported
            and get_model_setup_snapshot().state not in {"running", "cancelling"}
        )

    def execute(self, context):
        preferences = _preferences(context)
        try:
            start_model_download(
                _managed_root(preferences),
                self.model_id,
                offline=bool(preferences.offline_setup),
                online_access=bool(bpy.app.online_access),
            )
        except RuntimeSetupError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_import_model(bpy.types.Operator):
    bl_idname = "lingbot_map.import_model"
    bl_label = "Import Exact Local Model"

    model_id: StringProperty(name="Model ID", default="")
    filepath: StringProperty(name="Model File", subtype="FILE_PATH", default="")
    filter_glob: StringProperty(default="*.pt;*.pth;*.onnx", options={"HIDDEN"})

    @classmethod
    def poll(cls, _context):
        decision = get_host_decision()
        return bool(
            decision
            and decision.supported
            and get_model_setup_snapshot().state not in {"running", "cancelling"}
        )

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        preferences = _preferences(context)
        try:
            start_model_import(
                _managed_root(preferences), self.model_id, Path(self.filepath)
            )
        except RuntimeSetupError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_cancel_model_setup(bpy.types.Operator):
    bl_idname = "lingbot_map.cancel_model_setup"
    bl_label = "Cancel Model Acquisition"

    @classmethod
    def poll(cls, _context):
        return get_model_setup_snapshot().state in {"running", "cancelling"}

    def execute(self, _context):
        return {"FINISHED"} if cancel_model_setup() else {"CANCELLED"}


class _LINGBOTMAP_LifecyclePanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LingBot Map"

    @staticmethod
    def _draw_host_status(layout):
        decision = get_host_decision()
        if decision is None:
            layout.label(text="Host support has not been evaluated", icon="ERROR")
        elif decision.supported:
            layout.label(text=decision.message, icon="CHECKMARK")
        else:
            layout.label(text="Unsupported Host", icon="ERROR")
            layout.label(text=decision.message)


class LINGBOTMAP_PT_setup(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_setup"
    bl_label = "Setup"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        self._draw_host_status(layout)
        layout.separator()
        snapshot = get_setup_snapshot()
        icon = "CHECKMARK" if snapshot.state == "ready" else "ERROR" if snapshot.state == "failed" else "INFO"
        layout.label(text=snapshot.message, icon=icon)
        if snapshot.state in {"running", "cancelling"}:
            layout.operator(LINGBOTMAP_OT_cancel_runtime_setup.bl_idname)
        else:
            layout.operator(LINGBOTMAP_OT_setup_runtime.bl_idname)
        preferences = _preferences(context)
        model_root = _managed_root(preferences)
        catalog = bundled_model_catalog()
        store = ModelStore(model_root, catalog)
        model_snapshot = get_model_setup_snapshot()
        for entry in catalog.entries:
            box = layout.box()
            role = "Reconstruction Model" if entry.role == "reconstruction" else "Optional Auxiliary Model"
            box.label(text=f"{role}: {entry.display_name}")
            source = entry.artifact.source_repository.removeprefix("https://")
            box.label(text=f"Source: {source} @ {entry.artifact.source_revision[:12]}")
            box.label(
                text=f"License: {entry.license_record.spdx_expression} ({entry.license_record.status})",
                icon="ERROR" if not entry.license_record.covers_weights else "CHECKMARK",
            )
            box.label(text=f"SHA-256: {entry.artifact.sha256}")
            box.label(text=f"Download size: {entry.artifact.length:,} bytes")
            status = store.quick_status(entry)
            box.label(text=f"Managed status: {status}")
            row = box.row(align=True)
            download = row.operator(
                LINGBOTMAP_OT_download_model.bl_idname,
                text="Check Offline" if preferences.offline_setup else "Download",
            )
            download.model_id = entry.id
            local_import = row.operator(LINGBOTMAP_OT_import_model.bl_idname, text="Import Local")
            local_import.model_id = entry.id
        if model_snapshot.state in {"running", "cancelling"}:
            layout.label(
                text=(
                    f"{model_snapshot.message}: "
                    f"{model_snapshot.completed:,}/{model_snapshot.total:,} bytes"
                ),
                icon="INFO",
            )
            layout.operator(LINGBOTMAP_OT_cancel_model_setup.bl_idname)
        elif model_snapshot.state in {"ready", "failed", "cancelled"}:
            layout.label(
                text=model_snapshot.message,
                icon="CHECKMARK" if model_snapshot.state == "ready" else "ERROR",
            )
        layout.label(text="GPU Profiles: Not tested")


class LINGBOTMAP_PT_reconstruct(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_reconstruct"
    bl_label = "Reconstruct"
    bl_order = 1

    def draw(self, _context):
        decision = get_host_decision()
        layout = self.layout
        layout.label(text="Configure one Capture Source and Reconstruction Profile")
        row = layout.row()
        row.enabled = bool(decision and decision.supported)
        row.label(text="Start Reconstruction is unavailable until Setup is complete")


class LINGBOTMAP_PT_active_job(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_active_job"
    bl_label = "Active Job"
    bl_order = 2

    def draw(self, _context):
        self.layout.label(text="No active Reconstruction Job")


class LINGBOTMAP_PT_results(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_results"
    bl_label = "Results"
    bl_order = 3

    def draw(self, _context):
        self.layout.label(text="No Reconstruction Results discovered")


class LINGBOTMAP_PT_diagnostics(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_diagnostics"
    bl_label = "Diagnostics"
    bl_order = 4

    def draw(self, _context):
        self.layout.label(text="No diagnostics")


CLASSES = (
    LINGBOTMAP_Preferences,
    LINGBOTMAP_OT_setup_runtime,
    LINGBOTMAP_OT_cancel_runtime_setup,
    LINGBOTMAP_OT_download_model,
    LINGBOTMAP_OT_import_model,
    LINGBOTMAP_OT_cancel_model_setup,
    LINGBOTMAP_PT_setup,
    LINGBOTMAP_PT_reconstruct,
    LINGBOTMAP_PT_active_job,
    LINGBOTMAP_PT_results,
    LINGBOTMAP_PT_diagnostics,
)
