"""Blender Preferences and non-mutating lifecycle panel shell."""

import bpy
from bpy.props import BoolProperty, StringProperty

from .runtime import get_host_decision


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

    def draw(self, _context):
        layout = self.layout
        self._draw_host_status(layout)
        layout.separator()
        layout.label(text="Worker Runtime: Not configured")
        layout.label(text="Reconstruction Model: Not configured")
        layout.label(text="Sky Auxiliary Model: Optional")
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
    LINGBOTMAP_PT_setup,
    LINGBOTMAP_PT_reconstruct,
    LINGBOTMAP_PT_active_job,
    LINGBOTMAP_PT_results,
    LINGBOTMAP_PT_diagnostics,
)
