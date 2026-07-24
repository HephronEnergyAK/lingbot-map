"""Blender Preferences and explicit lifecycle actions."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid

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
    SCENE_UUID_SAVE_REQUIRED_PROPERTY,
    SKY_MASK_PROPERTY,
    cancel_active_job,
    capture_source_draft_path,
    duplicate_scene_uuid_groups,
    ensure_unique_scene_uuid,
    get_job_snapshot,
    latest_successful_preflight,
    normalized_blend_path,
    project_result_root,
    repair_duplicate_scene_uuids,
    start_fixture_job,
    start_preflight_job,
    start_reconstruction_job,
)
from .results import discover_ready_results
from .result_import import (
    ImportCapacityError,
    ResultImportCancelled,
    ResultImportError,
    detach_collection_copy,
    effective_disk_authority,
    get_import_status,
    find_reconstruction_camera,
    import_result,
    inspect_collection_ownership,
    relink_result_reference,
    relink_source_background,
    removal_inventory,
    remove_managed_version,
    resolve_duplicate_imports,
    result_reference_status,
    set_import_status,
    set_scene_resolution_to_source,
    set_source_background_visibility,
    toggle_model_coverage_guide,
    validate_result,
)


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
            scene_uuid = ensure_unique_scene_uuid(
                context.scene,
                tuple(bpy.data.scenes),
            )
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
            scene_uuid = ensure_unique_scene_uuid(
                context.scene,
                tuple(bpy.data.scenes),
            )
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
            scene_uuid = ensure_unique_scene_uuid(
                scene,
                tuple(bpy.data.scenes),
            )
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


class LINGBOTMAP_OT_import_result(bpy.types.Operator):
    """Explicitly import one fully validated Ready Result."""

    bl_idname = "lingbot_map.import_result"
    bl_label = "Import Reconstruction Result"
    bl_description = (
        "Validate and capacity-gate this Result, then transactionally import it"
    )

    result_directory: StringProperty(
        name="Result Directory",
        description="Exact published Reconstruction Result directory",
        subtype="DIR_PATH",
        options={"HIDDEN"},
    )

    def execute(self, context):
        try:
            outcome = import_result(
                self.result_directory,
                context.scene,
                context=context,
            )
        except ResultImportCancelled as exc:
            set_import_status(str(exc))
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        except (ImportCapacityError, ResultImportError, OSError, MemoryError) as exc:
            set_import_status(str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            message = f"Import failed and was rolled back: {type(exc).__name__}: {exc}"
            set_import_status(message)
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        self.report({"INFO"}, outcome.message)
        return {"FINISHED"}


def _scene_from_pointer(value):
    matches = [
        scene
        for scene in bpy.data.scenes
        if str(scene.as_pointer()) == str(value)
    ]
    if len(matches) != 1:
        raise ResultImportError("Selected Scene is no longer uniquely available")
    return matches[0]


def _collection_from_pointer(value):
    matches = [
        collection
        for collection in bpy.data.collections
        if str(collection.as_pointer()) == str(value)
    ]
    if len(matches) != 1:
        raise ResultImportError(
            "Selected Collection is no longer uniquely available"
        )
    return matches[0]


class LINGBOTMAP_OT_repair_scene_uuid(bpy.types.Operator):
    bl_idname = "lingbot_map.repair_duplicate_scene_uuid"
    bl_label = "Repair Duplicate Scene IDs"
    bl_description = (
        "Keep this explicitly selected Scene on the existing UUID and assign "
        "new UUIDs to every other duplicate; existing Jobs are not rebound"
    )

    duplicate_uuid: StringProperty(options={"HIDDEN"})
    keeper_pointer: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        return context.window_manager.invoke_confirm(self, _event)

    def execute(self, _context):
        try:
            keeper = _scene_from_pointer(self.keeper_pointer)
            repaired = repair_duplicate_scene_uuids(
                tuple(bpy.data.scenes),
                self.duplicate_uuid,
                keeper,
            )
        except (JobLifecycleError, ResultImportError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Repaired {len(repaired)} duplicate Scene IDs; save the .blend",
        )
        return {"FINISHED"}


class LINGBOTMAP_OT_import_result_into(bpy.types.Operator):
    bl_idname = "lingbot_map.import_result_into"
    bl_label = "Import Result Into Scene"
    bl_description = (
        "Explicitly import into another uniquely identified Scene while "
        "preserving the Result's original binding"
    )

    result_directory: StringProperty(options={"HIDDEN"}, subtype="DIR_PATH")
    target_scene_pointer: StringProperty(options={"HIDDEN"})
    confirmation_target: StringProperty(options={"HIDDEN"})
    confirmation_original: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        try:
            target = _scene_from_pointer(self.target_scene_pointer)
            document = validate_result(self.result_directory)
        except (ResultImportError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        original = document.manifest["target_scene"]
        self.confirmation_target = (
            f"{target.name} · "
            f"{target.get('lingbot_map_scene_uuid', 'no UUID')}"
        )
        self.confirmation_original = (
            f"{original['scene_name']} · {original['scene_uuid']}"
        )
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        self.layout.label(
            text="Confirm this explicit Result binding",
            icon="QUESTION",
        )
        self.layout.label(text=f"Original: {self.confirmation_original}")
        self.layout.label(text=f"Actual target: {self.confirmation_target}")
        self.layout.label(
            text="The original binding remains recorded and is not changed"
        )

    def execute(self, context):
        try:
            target = _scene_from_pointer(self.target_scene_pointer)
            outcome = import_result(
                self.result_directory,
                target,
                context=context if target is context.scene else None,
            )
        except (
            ImportCapacityError,
            ResultImportCancelled,
            ResultImportError,
            OSError,
            MemoryError,
        ) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, outcome.message)
        return {"FINISHED"}


class LINGBOTMAP_OT_import_external_result(bpy.types.Operator):
    bl_idname = "lingbot_map.import_external_result"
    bl_label = "Import External Result"
    bl_description = (
        "Validate a selected foreign Result and import it as a read-only disk "
        "reference without lifecycle authority"
    )

    filepath: StringProperty(
        name="External Result Directory", subtype="DIR_PATH"
    )
    target_scene_pointer: StringProperty(options={"HIDDEN"})
    confirmed: BoolProperty(default=False, options={"HIDDEN"})

    def invoke(self, context, _event):
        self.target_scene_pointer = str(context.scene.as_pointer())
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            target = _scene_from_pointer(self.target_scene_pointer)
            document = validate_result(self.filepath)
            current = normalized_blend_path(bpy.data.filepath)
            original = normalized_blend_path(
                document.manifest["target_scene"]["blend_path"]
            )
            expected_parent = project_result_root(original) / "results"
            selected = Path(self.filepath).resolve()
            if (
                str(current).casefold() == str(original).casefold()
                and str(selected.parent).casefold()
                == str(expected_parent).casefold()
            ):
                raise ResultImportError(
                    "This Result belongs to the current project; use Import"
                )
            if not self.confirmed:
                self.confirmed = True
                return context.window_manager.invoke_props_dialog(
                    self, width=520
                )
            outcome = import_result(
                document.ready.directory,
                target,
                context=context if target is context.scene else None,
            )
        except (
            ImportCapacityError,
            ResultImportCancelled,
            ResultImportError,
            OSError,
            MemoryError,
        ) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            outcome.message + "; external Result remains read-only on disk",
        )
        return {"FINISHED"}

    def draw(self, _context):
        self.layout.label(
            text="Import this completely validated external Result?",
            icon="QUESTION",
        )
        self.layout.label(text=str(self.filepath))
        self.layout.label(
            text="Its disk content remains read-only and is never deleted"
        )


class LINGBOTMAP_OT_relink_result_reference(bpy.types.Operator):
    bl_idname = "lingbot_map.relink_result_reference"
    bl_label = "Relink Result Reference"
    bl_description = (
        "Accept only a completely valid bundle with the exact recorded Result "
        "ID and manifest checksum"
    )

    collection_pointer: StringProperty(options={"HIDDEN"})
    filepath: StringProperty(name="Result Directory", subtype="DIR_PATH")

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            collection = _collection_from_pointer(self.collection_pointer)
            message = relink_result_reference(
                collection,
                context.scene,
                self.filepath,
                bpy.data.filepath,
            )
        except (ResultImportError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


class LINGBOTMAP_OT_detach_result_copy(bpy.types.Operator):
    bl_idname = "lingbot_map.detach_result_copy"
    bl_label = "Detach Copy"
    bl_description = (
        "Remove Extension identity and lifecycle authority without deleting "
        "or unlinking Blender data"
    )

    collection_pointer: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        return context.window_manager.invoke_confirm(self, _event)

    def execute(self, _context):
        try:
            collection = _collection_from_pointer(self.collection_pointer)
            detach_collection_copy(collection)
        except ResultImportError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Collection detached; all Blender data preserved")
        return {"FINISHED"}


class LINGBOTMAP_OT_resolve_duplicate_imports(bpy.types.Operator):
    bl_idname = "lingbot_map.resolve_duplicate_imports"
    bl_label = "Resolve Duplicate Imports"
    bl_description = (
        "Keep this explicitly selected managed Collection and detach every "
        "other duplicate non-destructively"
    )

    result_id: StringProperty(options={"HIDDEN"})
    keeper_pointer: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        return context.window_manager.invoke_confirm(self, _event)

    def execute(self, context):
        try:
            keeper = _collection_from_pointer(self.keeper_pointer)
            detached = resolve_duplicate_imports(
                context.scene, self.result_id, keeper
            )
        except ResultImportError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"}, f"Kept one managed instance; detached {detached} copies"
        )
        return {"FINISHED"}


class LINGBOTMAP_OT_remove_result_version(bpy.types.Operator):
    bl_idname = "lingbot_map.remove_result_version"
    bl_label = "Remove Version"
    bl_description = (
        "Remove only uniquely owned Blender datablocks; the disk Result remains"
    )

    collection_pointer: StringProperty(options={"HIDDEN"})
    inventory_name: StringProperty(options={"HIDDEN"})
    inventory_objects: StringProperty(options={"HIDDEN"})
    inventory_points: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        try:
            collection = _collection_from_pointer(self.collection_pointer)
            inventory = removal_inventory(collection, context.scene)
        except ResultImportError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.inventory_name = inventory.collection_name
        self.inventory_objects = str(inventory.object_count)
        self.inventory_points = str(inventory.point_count)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, _context):
        self.layout.label(
            text="Managed content may contain undetected user edits",
            icon="ERROR",
        )
        self.layout.label(text=f"Collection: {self.inventory_name}")
        self.layout.label(text=f"Objects: {self.inventory_objects}")
        self.layout.label(text=f"Points: {self.inventory_points}")
        self.layout.label(
            text="Only uniquely owned Blender datablocks are deleted; disk data stays"
        )

    def execute(self, context):
        try:
            collection = _collection_from_pointer(self.collection_pointer)
            inventory = remove_managed_version(
                collection, context.scene
            )
        except ResultImportError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Removed {inventory.collection_name}; shared data was preserved",
        )
        return {"FINISHED"}


def _imported_reconstruction_collections(scene):
    found = []
    pending = list(scene.collection.children)
    while pending:
        collection = pending.pop()
        pending.extend(collection.children)
        if (
            collection.get("lingbot_map_kind")
            == "reconstruction_collection"
            or collection.get("lingbot_map_result_id") is not None
        ):
            found.append(collection)
    return tuple(found)


def _imported_collection(scene, result_id):
    matches = [
        collection
        for collection in _imported_reconstruction_collections(scene)
        if collection.get("lingbot_map_result_id") == result_id
    ]
    if len(matches) != 1:
        raise ResultImportError(
            "Scene does not contain exactly one owned Collection for this Result"
        )
    inspection = inspect_collection_ownership(matches[0], scene)
    if inspection.status != "managed":
        raise ResultImportError(inspection.message)
    return matches[0]


class LINGBOTMAP_OT_use_reconstruction_camera(bpy.types.Operator):
    bl_idname = "lingbot_map.use_reconstruction_camera"
    bl_label = "Use Reconstruction Camera"
    bl_description = (
        "Explicitly make this imported camera the Scene camera for inspection"
    )

    result_id: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        try:
            collection = _imported_collection(
                context.scene, self.result_id
            )
            context.scene.camera = find_reconstruction_camera(collection)
            if (
                getattr(context, "area", None) is not None
                and context.area.type == "VIEW_3D"
            ):
                context.space_data.region_3d.view_perspective = "CAMERA"
        except (ResultImportError, AttributeError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Reconstruction Camera is active for inspection")
        return {"FINISHED"}


class LINGBOTMAP_OT_set_resolution_to_source(bpy.types.Operator):
    bl_idname = "lingbot_map.set_resolution_to_source"
    bl_label = "Set Scene Resolution to Source"
    bl_description = (
        "Revalidate source identity and alignment, then explicitly set exact "
        "display dimensions without changing FPS"
    )

    result_id: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        return context.window_manager.invoke_confirm(self, _event)

    def execute(self, context):
        try:
            collection = _imported_collection(
                context.scene, self.result_id
            )
            message = set_scene_resolution_to_source(
                collection, context.scene
            )
        except (ResultImportError, OSError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


class LINGBOTMAP_OT_relink_source_background(bpy.types.Operator):
    bl_idname = "lingbot_map.relink_source_background"
    bl_label = "Relink Source Background"
    bl_description = (
        "Choose an MP4 or MOV; only the exact checksum recorded by the Result "
        "is accepted"
    )

    result_id: StringProperty(options={"HIDDEN"})
    filepath: StringProperty(
        name="Capture Source", subtype="FILE_PATH", default=""
    )
    filter_glob: StringProperty(
        default="*.mp4;*.mov", options={"HIDDEN"}
    )

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            collection = _imported_collection(
                context.scene, self.result_id
            )
            message = relink_source_background(
                collection, context.scene, self.filepath
            )
        except (ResultImportError, OSError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


class LINGBOTMAP_OT_source_background_visibility(bpy.types.Operator):
    bl_idname = "lingbot_map.source_background_visibility"
    bl_label = "Set Source Background Visibility"

    result_id: StringProperty(options={"HIDDEN"})
    visible: BoolProperty(default=False, options={"HIDDEN"})

    def execute(self, context):
        try:
            collection = _imported_collection(
                context.scene, self.result_id
            )
            message = set_source_background_visibility(
                collection, self.visible
            )
        except (ResultImportError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


class LINGBOTMAP_OT_toggle_model_coverage(bpy.types.Operator):
    bl_idname = "lingbot_map.toggle_model_coverage"
    bl_label = "Toggle Model Crop Guide"
    bl_description = (
        "Toggle a temporary non-rendering Camera View overlay; no datablock "
        "or Result file is changed"
    )

    result_id: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        try:
            collection = _imported_collection(
                context.scene, self.result_id
            )
            shown = toggle_model_coverage_guide(context, collection)
        except (ResultImportError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            "Model Crop Guide shown" if shown else "Model Crop Guide hidden",
        )
        return {"FINISHED"}


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
        data = getattr(bpy, "data", None)
        for duplicate_uuid, scenes in duplicate_scene_uuid_groups(
            tuple(getattr(data, "scenes", ()))
        ).items():
            box = layout.box()
            box.label(
                text=f"Duplicate Scene ID: {duplicate_uuid}",
                icon="ERROR",
            )
            box.label(
                text="Choose the exact keeper; every other Scene gets a new ID"
            )
            for scene in scenes:
                repair = box.operator(
                    LINGBOTMAP_OT_repair_scene_uuid.bl_idname,
                    text=f"Keep {scene.name}",
                )
                repair.duplicate_uuid = duplicate_uuid
                repair.keeper_pointer = str(scene.as_pointer())
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
        external = self.layout.operator(
            LINGBOTMAP_OT_import_external_result.bl_idname,
            text="Import External Result",
            icon="IMPORT",
        )
        external.target_scene_pointer = str(context.scene.as_pointer())
        blend_path = getattr(bpy.data, "filepath", "")
        scene_uuid = context.scene.get("lingbot_map_scene_uuid") if blend_path else None
        results = discover_ready_results(blend_path, scene_uuid=scene_uuid) if blend_path else ()
        imported = _imported_reconstruction_collections(context.scene)
        imported_by_id = {}
        for collection in imported:
            imported_by_id.setdefault(
                collection.get("lingbot_map_result_id"), []
            ).append(collection)
        if not results and not imported:
            self.layout.label(text="No Reconstruction Results discovered")
            return
        self.layout.label(text=get_import_status())
        drawn_imported = set()
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
            claims = tuple(imported_by_id.get(result.result_id, ()))
            if not claims:
                action = box.operator(
                    LINGBOTMAP_OT_import_result.bl_idname,
                    text="Import",
                    icon="IMPORT",
                )
                action.result_directory = str(result.directory)
                self._draw_import_into_actions(
                    box, context.scene, result.directory
                )
            elif len(claims) > 1:
                box.label(
                    text="Duplicate Imported Identity",
                    icon="ERROR",
                )
                box.label(
                    text="Choose the exact managed keeper; copies are detached"
                )
                for collection in claims:
                    inspection = inspect_collection_ownership(
                        collection, context.scene
                    )
                    if inspection.status == "managed":
                        keep = box.operator(
                            LINGBOTMAP_OT_resolve_duplicate_imports.bl_idname,
                            text=f"Keep {collection.name}",
                        )
                        keep.result_id = result.result_id
                        keep.keeper_pointer = str(collection.as_pointer())
                    else:
                        box.label(
                            text=f"{collection.name}: {inspection.message}",
                            icon="ERROR",
                        )
            for collection in claims:
                drawn_imported.add(result.result_id)
                self._draw_managed_collection(
                    box,
                    context,
                    collection,
                    duplicate_count=len(claims),
                )
        for collection in imported:
            result_id = collection.get("lingbot_map_result_id")
            if result_id in drawn_imported:
                continue
            box = self.layout.box()
            claims = tuple(imported_by_id.get(result_id, ()))
            box.label(text=f"Imported {result_id}", icon="INFO")
            if len(claims) > 1:
                box.label(
                    text="Duplicate Imported Identity",
                    icon="ERROR",
                )
                for candidate in claims:
                    inspection = inspect_collection_ownership(
                        candidate, context.scene
                    )
                    if inspection.status == "managed":
                        keep = box.operator(
                            LINGBOTMAP_OT_resolve_duplicate_imports.bl_idname,
                            text=f"Keep {candidate.name}",
                        )
                        keep.result_id = str(result_id)
                        keep.keeper_pointer = str(candidate.as_pointer())
            self._draw_managed_collection(
                box,
                context,
                collection,
                duplicate_count=len(claims),
            )

    @staticmethod
    def _draw_import_into_actions(box, current_scene, result_directory):
        scene_uuids = []
        for scene in bpy.data.scenes:
            try:
                scene_uuids.append(
                    str(
                        uuid.UUID(
                            str(
                                scene.get(
                                    "lingbot_map_scene_uuid", ""
                                )
                            )
                        )
                    )
                )
            except ValueError:
                scene_uuids.append("")
        for scene in bpy.data.scenes:
            try:
                uuid_value = str(
                    uuid.UUID(
                        str(
                            scene.get(
                                "lingbot_map_scene_uuid", ""
                            )
                        )
                    )
                )
            except ValueError:
                uuid_value = ""
            if (
                scene is current_scene
                or not uuid_value
                or scene_uuids.count(uuid_value) != 1
                or bool(
                    scene.get(
                        SCENE_UUID_SAVE_REQUIRED_PROPERTY, False
                    )
                )
            ):
                continue
            action = box.operator(
                LINGBOTMAP_OT_import_result_into.bl_idname,
                text=f"Import Into {scene.name}",
            )
            action.result_directory = str(result_directory)
            action.target_scene_pointer = str(scene.as_pointer())

    @classmethod
    def _draw_managed_collection(
        cls, box, context, collection, *, duplicate_count
    ):
        inspection = inspect_collection_ownership(
            collection, context.scene
        )
        icon = "CHECKMARK" if inspection.status == "managed" else "ERROR"
        box.label(
            text=f"{collection.name}: {inspection.message}",
            icon=icon,
        )
        authority = effective_disk_authority(
            collection, bpy.data.filepath
        )
        box.label(text=f"Disk authority: {authority}")
        status, resolved = result_reference_status(
            collection, bpy.data.filepath
        )
        box.label(
            text=f"Result reference: {status}",
            icon="CHECKMARK" if status == "available" else "ERROR",
        )
        if resolved:
            box.label(text=resolved)
        if status != "available" and inspection.status == "managed":
            relink = box.operator(
                LINGBOTMAP_OT_relink_result_reference.bl_idname,
                text="Relink Result Reference",
            )
            relink.collection_pointer = str(collection.as_pointer())
        detach = box.operator(
            LINGBOTMAP_OT_detach_result_copy.bl_idname,
            text="Detach Copy",
        )
        detach.collection_pointer = str(collection.as_pointer())
        if inspection.status != "managed":
            box.label(
                text="Destructive actions disabled until ownership is consistent",
                icon="ERROR",
            )
            return
        if duplicate_count == 1:
            remove = box.operator(
                LINGBOTMAP_OT_remove_result_version.bl_idname,
                text="Remove Version",
                icon="TRASH",
            )
            remove.collection_pointer = str(collection.as_pointer())
        cls._draw_source_view(box, collection)

    @staticmethod
    def _draw_source_view(box, collection):
        result_id = str(collection.get("lingbot_map_result_id", ""))
        if not collection.get("lingbot_map_source_display_json"):
            box.label(
                text="Legacy Result: source-aligned inspection unavailable",
                icon="INFO",
            )
            return
        start = int(collection.get("lingbot_map_timeline_start", 0))
        count = int(collection.get("lingbot_map_frame_count", 0))
        box.label(
            text=f"Source-aligned frames: {start}–{start + max(0, count - 1)}"
        )
        status = str(
            collection.get(
                "lingbot_map_source_background_status", "unattached"
            )
        )
        box.label(
            text=f"Source Background: {status}",
            icon="CHECKMARK" if status == "attached-hidden" else "INFO",
        )
        camera = box.operator(
            LINGBOTMAP_OT_use_reconstruction_camera.bl_idname,
            text="Use Reconstruction Camera",
            icon="CAMERA_DATA",
        )
        camera.result_id = result_id
        resolution = box.operator(
            LINGBOTMAP_OT_set_resolution_to_source.bl_idname,
            text="Set Scene Resolution to Source",
        )
        resolution.result_id = result_id
        relink = box.operator(
            LINGBOTMAP_OT_relink_source_background.bl_idname,
            text="Relink Source Background",
            icon="FILE_MOVIE",
        )
        relink.result_id = result_id
        row = box.row(align=True)
        show = row.operator(
            LINGBOTMAP_OT_source_background_visibility.bl_idname,
            text="Show Background",
        )
        show.result_id = result_id
        show.visible = True
        hide = row.operator(
            LINGBOTMAP_OT_source_background_visibility.bl_idname,
            text="Hide Background",
        )
        hide.result_id = result_id
        hide.visible = False
        guide = box.operator(
            LINGBOTMAP_OT_toggle_model_coverage.bl_idname,
            text="Toggle Model Crop Guide",
        )
        guide.result_id = result_id


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
    LINGBOTMAP_OT_import_result,
    LINGBOTMAP_OT_repair_scene_uuid,
    LINGBOTMAP_OT_import_result_into,
    LINGBOTMAP_OT_import_external_result,
    LINGBOTMAP_OT_relink_result_reference,
    LINGBOTMAP_OT_detach_result_copy,
    LINGBOTMAP_OT_resolve_duplicate_imports,
    LINGBOTMAP_OT_remove_result_version,
    LINGBOTMAP_OT_use_reconstruction_camera,
    LINGBOTMAP_OT_set_resolution_to_source,
    LINGBOTMAP_OT_relink_source_background,
    LINGBOTMAP_OT_source_background_visibility,
    LINGBOTMAP_OT_toggle_model_coverage,
    LINGBOTMAP_PT_setup,
    LINGBOTMAP_PT_reconstruct,
    LINGBOTMAP_PT_active_job,
    LINGBOTMAP_PT_results,
    LINGBOTMAP_PT_diagnostics,
)
