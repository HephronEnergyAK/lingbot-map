"""Blender Preferences and explicit lifecycle actions."""

from dataclasses import asdict
import hashlib
import json
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
from .gpu_capability import (
    cancel_gpu_capability,
    discover_physical_gpus,
    get_capability_snapshot,
    select_gpu_uuid,
    start_gpu_capability,
)
from .job_lifecycle import (
    CAPTURE_SOURCE_PROPERTY,
    CAMERA_ITERATIONS_PROPERTY,
    CONFIDENCE_CUTOFF_PROPERTY,
    DEPTH_CUTOFF_PROPERTY,
    JobLifecycleError,
    POINT_BUDGET_CONFIRMED_PROPERTY,
    POINT_BUDGET_PROPERTY,
    PROFILE_PROPERTY,
    RETAIN_DENSE_PROPERTY,
    SKY_MASK_PROPERTY,
    cancel_active_job,
    capture_source_draft_path,
    ensure_unique_scene_uuid,
    get_job_snapshot,
    latest_successful_preflight,
    start_fixture_job,
    start_preflight_job,
    start_reconstruction_job,
)
from .results import discover_ready_results


PROFILE_DEFAULTS = {
    "Draft": (1, 70.0, 99.5, 1_000_000),
    "Balanced": (4, 50.0, 99.5, 5_000_000),
    "High": (4, 30.0, 99.5, 10_000_000),
}


def _capability_settings_sha256(name: str) -> str:
    camera, confidence, depth, budget = PROFILE_DEFAULTS[name]
    document = {
        "name": name,
        "camera_iterations": camera,
        "confidence_cutoff_percent": int(confidence),
        "import_point_budget": budget,
        "depth_cutoff_percent": depth,
        "image_size": 518,
        "patch_size": 14,
        "scale_frames": 8,
        "window_frames": 64,
        "attention_backend": "sdpa",
        "execution_mode": "eager",
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


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
        description="Persistent NVML physical GPU UUID; mutable CUDA ordinals are never stored",
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


class LINGBOTMAP_OT_test_gpu_profiles(bpy.types.Operator):
    bl_idname = "lingbot_map.test_gpu_profiles"
    bl_label = "Test GPU Profiles"
    bl_description = "Qualify Draft, Balanced, and High on one physical GPU UUID"

    @classmethod
    def poll(cls, _context):
        decision = get_host_decision()
        return bool(
            decision
            and decision.supported
            and get_capability_snapshot().state
            not in {"preparing", "running", "cancelling"}
        )

    def execute(self, context):
        preferences = _preferences(context)
        root = _managed_root(preferences)
        try:
            devices = discover_physical_gpus(root)
            selected = select_gpu_uuid(devices, preferences.gpu_uuid.strip())
            # Blender persists AddonPreferences; only an unambiguous physical UUID
            # may be selected automatically.
            preferences.gpu_uuid = selected
            start_gpu_capability(root, selected)
        except RuntimeSetupError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_cancel_gpu_profiles(bpy.types.Operator):
    bl_idname = "lingbot_map.cancel_gpu_profiles"
    bl_label = "Cancel GPU Profile Test"

    @classmethod
    def poll(cls, _context):
        return get_capability_snapshot().state in {"preparing", "running", "cancelling"}

    def execute(self, _context):
        return {"FINISHED"} if cancel_gpu_capability() else {"CANCELLED"}


class LINGBOTMAP_OT_run_fixture_job(bpy.types.Operator):
    bl_idname = "lingbot_map.run_fixture_job"
    bl_label = "Run Lifecycle Fixture Job"
    bl_description = "Exercise the complete isolated Blender-to-Worker lifecycle without neural inference"

    @classmethod
    def poll(cls, _context):
        decision = get_host_decision()
        return bool(decision and decision.supported and get_job_snapshot().state not in {
            "starting", "running", "reconnecting", "unresponsive", "cancelling"
        })

    def execute(self, context):
        try:
            blend_path = bpy.data.filepath
            if not blend_path:
                raise JobLifecycleError("Save the Blender file before launching a Job")
            scene_uuid = ensure_unique_scene_uuid(context.scene, tuple(bpy.data.scenes))
            if bpy.data.is_dirty:
                raise JobLifecycleError("Save the .blend after its Scene UUID or other changes before launch")
            preferences = _preferences(context)
            start_fixture_job(
                managed_root=_managed_root(preferences),
                blend_path=blend_path,
                scene_uuid=scene_uuid,
                scene_name=context.scene.name,
                timeline_start=context.scene.frame_current,
            )
        except (JobLifecycleError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_select_capture_source(bpy.types.Operator):
    bl_idname = "lingbot_map.select_capture_source"
    bl_label = "Choose Capture Source"
    bl_description = "Choose one local MP4 or MOV and store a scene-relative path when representable"

    filepath: StringProperty(name="Capture Source", subtype="FILE_PATH", default="")
    filter_glob: StringProperty(default="*.mp4;*.mov", options={"HIDDEN"})

    def invoke(self, context, _event):
        current = str(getattr(context.scene, CAPTURE_SOURCE_PROPERTY, "")).strip()
        if current and bpy.data.filepath:
            try:
                self.filepath = str(
                    capture_source_draft_path(current, bpy.data.filepath)
                )
            except JobLifecycleError:
                pass
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            if not bpy.data.filepath:
                raise JobLifecycleError("Save the Blender file before choosing a Capture Source")
            draft = capture_source_draft_path(self.filepath, bpy.data.filepath)
            setattr(context.scene, CAPTURE_SOURCE_PROPERTY, draft)
        except JobLifecycleError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_run_preflight_job(bpy.types.Operator):
    bl_idname = "lingbot_map.run_preflight_job"
    bl_label = "Preflight Capture Source"
    bl_description = "Decode and validate every presentation frame without retaining pixels"

    @classmethod
    def poll(cls, context):
        decision = get_host_decision()
        capture = str(getattr(getattr(context, "scene", None), CAPTURE_SOURCE_PROPERTY, "")).strip()
        return bool(decision and decision.supported and capture and get_job_snapshot().state not in {
            "starting", "running", "reconnecting", "unresponsive", "cancelling"
        })

    def execute(self, context):
        try:
            blend_path = bpy.data.filepath
            if not blend_path:
                raise JobLifecycleError("Save the Blender file before launching a Job")
            entered_draft = str(getattr(context.scene, CAPTURE_SOURCE_PROPERTY, ""))
            normalized_draft = capture_source_draft_path(entered_draft, blend_path)
            if normalized_draft != entered_draft:
                setattr(context.scene, CAPTURE_SOURCE_PROPERTY, normalized_draft)
                raise JobLifecycleError(
                    "Capture Source was normalized for this Scene. Save the .blend, then launch the Job again."
                )
            scene_uuid = ensure_unique_scene_uuid(context.scene, tuple(bpy.data.scenes))
            if bpy.data.is_dirty:
                raise JobLifecycleError("Save the .blend after its Scene UUID or other changes before launch")
            preferences = _preferences(context)
            start_preflight_job(
                managed_root=_managed_root(preferences),
                blend_path=blend_path,
                scene_uuid=scene_uuid,
                scene_name=context.scene.name,
                timeline_start=context.scene.frame_current,
                capture_draft_path=normalized_draft,
            )
        except (JobLifecycleError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_run_reconstruction_job(bpy.types.Operator):
    bl_idname = "lingbot_map.run_reconstruction_job"
    bl_label = "Reconstruct"
    bl_description = "Run qualified headless camera and depth reconstruction over every frame"

    @classmethod
    def poll(cls, context):
        decision = get_host_decision()
        capture = str(getattr(getattr(context, "scene", None), CAPTURE_SOURCE_PROPERTY, "")).strip()
        return bool(decision and decision.supported and capture and get_job_snapshot().state not in {
            "starting", "running", "reconnecting", "unresponsive", "cancelling"
        })

    def execute(self, context):
        try:
            blend_path = bpy.data.filepath
            if not blend_path:
                raise JobLifecycleError("Save the Blender file before launching a Job")
            scene = context.scene
            capture = str(getattr(scene, CAPTURE_SOURCE_PROPERTY, "")).strip()
            preflight = latest_successful_preflight(blend_path, capture)
            scene_uuid = ensure_unique_scene_uuid(scene, tuple(bpy.data.scenes))
            if bpy.data.is_dirty:
                raise JobLifecycleError("Save the .blend after its Scene UUID or other changes before launch")
            preferences = _preferences(context)
            managed_root = _managed_root(preferences)
            devices = discover_physical_gpus(managed_root)
            selected_uuid = select_gpu_uuid(devices, preferences.gpu_uuid.strip())
            selected_gpu = next(device for device in devices if device.uuid == selected_uuid)
            preferences.gpu_uuid = selected_uuid
            catalog = bundled_model_catalog()
            entries = tuple(entry for entry in catalog.entries if entry.role == "reconstruction")
            if len(entries) != 1:
                raise JobLifecycleError("Model Catalog must contain one Reconstruction Model")
            entry = entries[0]
            store = ModelStore(managed_root, catalog)
            model_path = store.validate(entry)
            sky_mask_enabled = bool(getattr(scene, SKY_MASK_PROPERTY))
            auxiliary_model = None
            if sky_mask_enabled:
                auxiliary_entries = tuple(
                    item for item in catalog.entries if item.role == "auxiliary"
                )
                if len(auxiliary_entries) != 1:
                    raise JobLifecycleError(
                        "Model Catalog must contain one Sky Mask Auxiliary Model"
                    )
                auxiliary_entry = auxiliary_entries[0]
                auxiliary_path = store.validate(auxiliary_entry)
                auxiliary_model = {
                    "catalog_version": catalog.version,
                    "id": auxiliary_entry.id,
                    "path": str(auxiliary_path),
                    "sha256": auxiliary_entry.artifact.sha256,
                }
            profile_name = str(getattr(scene, PROFILE_PROPERTY))
            camera_iterations = int(getattr(scene, CAMERA_ITERATIONS_PROPERTY))
            if camera_iterations > 4:
                raise JobLifecycleError(
                    "Custom camera iterations above four have no fixed v1 capability workload"
                )
            capability_name = (
                profile_name if profile_name in PROFILE_DEFAULTS
                else "Draft" if camera_iterations == 1 else "Balanced"
            )
            start_reconstruction_job(
                managed_root=managed_root,
                blend_path=blend_path,
                scene_uuid=scene_uuid,
                scene_name=scene.name,
                timeline_start=scene.frame_current,
                capture_draft_path=capture,
                profile_name=profile_name,
                camera_iterations=camera_iterations,
                confidence_cutoff_percent=float(getattr(scene, CONFIDENCE_CUTOFF_PROPERTY)),
                depth_cutoff_percent=float(getattr(scene, DEPTH_CUTOFF_PROPERTY)),
                import_point_budget=int(getattr(scene, POINT_BUDGET_PROPERTY)),
                point_budget_confirmed=bool(getattr(scene, POINT_BUDGET_CONFIRMED_PROPERTY)),
                retain_dense_predictions=bool(getattr(scene, RETAIN_DENSE_PROPERTY)),
                sky_mask_enabled=sky_mask_enabled,
                auxiliary_model=auxiliary_model,
                gpu=asdict(selected_gpu),
                capability_profile_name=capability_name,
                capability_profile_settings_sha256=_capability_settings_sha256(capability_name),
                model={
                    "catalog_version": catalog.version,
                    "id": entry.id,
                    "path": str(model_path),
                    "sha256": entry.artifact.sha256,
                },
                preflight_result=preflight,
            )
        except (JobLifecycleError, RuntimeSetupError, ValueError, StopIteration) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_cancel_active_job(bpy.types.Operator):
    bl_idname = "lingbot_map.cancel_active_job"
    bl_label = "Cancel Active Job"

    @classmethod
    def poll(cls, _context):
        return get_job_snapshot().state in {"starting", "running", "reconnecting", "unresponsive", "cancelling"}

    def execute(self, _context):
        try:
            return {"FINISHED"} if cancel_active_job() else {"CANCELLED"}
        except JobLifecycleError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


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
        capability = get_capability_snapshot()
        selected_uuid = getattr(preferences, "gpu_uuid", "").strip()
        layout.label(
            text=f"Physical GPU: {selected_uuid or 'not selected'}",
            icon="CHECKMARK" if selected_uuid else "INFO",
        )
        capability_icon = (
            "CHECKMARK" if capability.state == "succeeded"
            else "ERROR" if capability.state in {"failed", "blocked", "cancelled"}
            else "INFO"
        )
        layout.label(text=capability.message, icon=capability_icon)
        if capability.state in {"preparing", "running", "cancelling"}:
            if capability.total:
                layout.label(
                    text=(
                        f"{capability.phase or 'starting'}: "
                        f"{capability.completed}/{capability.total}"
                    )
                )
            layout.operator(LINGBOTMAP_OT_cancel_gpu_profiles.bl_idname)
        else:
            layout.operator(LINGBOTMAP_OT_test_gpu_profiles.bl_idname)
        for result in capability.results:
            identity = result.get("identity", {})
            name = identity.get("profile_name", "Unknown") if isinstance(identity, dict) else "Unknown"
            state = str(result.get("state", "unknown"))
            required = int(result.get("required_free_bytes", 0))
            peak = int(result.get("measured_peak_bytes", 0))
            layout.label(
                text=f"{name}: {state} (peak {peak:,} B; required {required:,} B)",
                icon="CHECKMARK" if state == "qualified" else "ERROR",
            )


class LINGBOTMAP_PT_reconstruct(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_reconstruct"
    bl_label = "Reconstruct"
    bl_order = 1

    def draw(self, context):
        decision = get_host_decision()
        layout = self.layout
        layout.label(text="Configure one Capture Source and Reconstruction Profile")
        layout.prop(context.scene, CAPTURE_SOURCE_PROPERTY, text="Capture Source")
        layout.operator(LINGBOTMAP_OT_select_capture_source.bl_idname)
        row = layout.row()
        row.enabled = bool(decision and decision.supported)
        row.operator(LINGBOTMAP_OT_run_preflight_job.bl_idname)
        layout.label(text="Preflight examines every frame without changing Scene FPS")
        layout.prop(context.scene, PROFILE_PROPERTY, text="Profile")
        box = layout.box()
        box.prop(context.scene, CAMERA_ITERATIONS_PROPERTY, text="Camera Iterations")
        box.prop(context.scene, CONFIDENCE_CUTOFF_PROPERTY, text="Confidence Cutoff %")
        box.prop(context.scene, DEPTH_CUTOFF_PROPERTY, text="Depth Cutoff %")
        box.prop(context.scene, POINT_BUDGET_PROPERTY, text="Import Point Budget")
        box.prop(context.scene, RETAIN_DENSE_PROPERTY, text="Retain Dense Predictions")
        box.prop(context.scene, SKY_MASK_PROPERTY, text="Sky Masking")
        box.label(text="Sky only; does not detect or remove Dynamic Content")
        box.label(text="Moving people and vehicles may still ghost or trail")
        if int(getattr(context.scene, POINT_BUDGET_PROPERTY)) > 10_000_000:
            box.prop(context.scene, POINT_BUDGET_CONFIRMED_PROPERTY, text="Confirm >10M Budget")
        run = layout.row()
        run.enabled = bool(decision and decision.supported)
        run.operator(LINGBOTMAP_OT_run_reconstruction_job.bl_idname)


class LINGBOTMAP_PT_active_job(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_active_job"
    bl_label = "Active Job"
    bl_order = 2

    def draw(self, _context):
        snapshot = get_job_snapshot()
        layout = self.layout
        icon = "CHECKMARK" if snapshot.state == "succeeded" else "ERROR" if snapshot.state in {
            "failed", "interrupted", "forced_termination", "protocol_error", "stale_identity"
        } else "INFO"
        layout.label(text=snapshot.message, icon=icon)
        if snapshot.job_id:
            layout.label(text=f"Job: {snapshot.job_id}")
        if snapshot.phase:
            layout.label(text=f"{snapshot.phase}: {snapshot.completed}/{snapshot.total}")
        if snapshot.eta_seconds is not None:
            layout.label(text=f"ETA: {snapshot.eta_seconds:.0f} s")
        if snapshot.state in {"starting", "running", "reconnecting", "unresponsive", "cancelling"}:
            layout.operator(LINGBOTMAP_OT_cancel_active_job.bl_idname)


class LINGBOTMAP_PT_results(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_results"
    bl_label = "Results"
    bl_order = 3

    def draw(self, context):
        blend_path = getattr(bpy.data, "filepath", "")
        scene_uuid = context.scene.get("lingbot_map_scene_uuid") if blend_path else None
        results = discover_ready_results(blend_path, scene_uuid=scene_uuid) if blend_path else ()
        if not results:
            self.layout.label(text="No Reconstruction Results discovered")
            return
        for result in results:
            box = self.layout.box()
            box.label(text="Ready", icon="CHECKMARK")
            box.label(text=f"{result.profile_name}: {result.point_count:,} points")
            box.label(text=f"{result.frame_count:,} frames · {result.result_id}")
            if result.alignment_boundary_count:
                if result.quality_warning_count:
                    box.label(
                        text=f"Quality Warning: {result.quality_warning_count} window boundaries",
                        icon="ERROR",
                    )
                    box.label(text=f"Worst boundary {result.worst_boundary}")
                else:
                    box.label(
                        text=f"Window alignment: {result.alignment_boundary_count} boundaries · no warnings"
                    )
            if result.dense_status == "available":
                box.label(text="Dense Predictions: available", icon="CHECKMARK")
            elif result.dense_status == "incompatible":
                box.label(text="Dense Predictions: incompatible version", icon="ERROR")
            elif result.dense_status == "unavailable":
                box.label(text="Dense Predictions: unavailable; core Result remains usable", icon="ERROR")
            if result.sky_masked:
                box.label(
                    text=f"Sky Masking: enabled ({result.sky_cache_status})",
                    icon="CHECKMARK",
                )
                if result.sky_count_above_95_percent:
                    box.label(
                        text=(
                            "Sky Mask warning: "
                            f"{result.sky_count_above_95_percent} frames exceed 95% sky"
                        ),
                        icon="ERROR",
                    )


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
    LINGBOTMAP_OT_test_gpu_profiles,
    LINGBOTMAP_OT_cancel_gpu_profiles,
    LINGBOTMAP_OT_run_fixture_job,
    LINGBOTMAP_OT_select_capture_source,
    LINGBOTMAP_OT_run_preflight_job,
    LINGBOTMAP_OT_run_reconstruction_job,
    LINGBOTMAP_OT_cancel_active_job,
    LINGBOTMAP_PT_setup,
    LINGBOTMAP_PT_reconstruct,
    LINGBOTMAP_PT_active_job,
    LINGBOTMAP_PT_results,
    LINGBOTMAP_PT_diagnostics,
)
