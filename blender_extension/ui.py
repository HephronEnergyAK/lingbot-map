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
from .diagnostics import (
    DiagnosticReportError,
    build_portable_report,
    export_portable_report,
    retain_extension_diagnostic,
)
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
    ensure_project_layout,
    get_job_snapshot,
    latest_successful_preflight,
    normalized_blend_path,
    project_result_root,
    repair_duplicate_scene_uuids,
    start_fixture_job,
    start_preflight_job,
    start_reconstruction_job,
)
from .results import read_ready_result
from .project_lifecycle import (
    InventorySnapshot,
    LifecyclePlan,
    PartialDeletionError,
    PermanentDeletionSession,
    ProjectInventory,
    ProjectLifecycle,
    ProjectLifecycleError,
)
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

_project_inventory: ProjectInventory | None = None
_project_inventory_key = ""
_project_inventory_pages = {
    "jobs": 0,
    "results": 0,
    "diagnostics": 0,
    "trash": 0,
}
_pending_lifecycle_plans: dict[str, tuple[str, ProjectLifecycle, LifecyclePlan]] = {}
_permanent_delete_status = ""
_project_inventory_error = ""


def _project_key(blend_path: str) -> str:
    return str(normalized_blend_path(blend_path)).casefold()


def _retain_ui_diagnostic(
    *,
    category: str,
    state: str,
    error_code: str,
    phase: str,
    detail: object,
    scene=None,
) -> None:
    """Best-effort durable evidence; never mask the originating UI failure."""

    blend_path = str(getattr(bpy.data, "filepath", ""))
    if not blend_path:
        return
    current_scene = scene or getattr(
        getattr(bpy, "context", None),
        "scene",
        None,
    )
    target_scene = {
        "blend_path": blend_path,
        "scene_uuid": str(
            getattr(current_scene, "lingbot_map_scene_uuid", "")
        ),
        "scene_name": str(getattr(current_scene, "name", "")),
    }
    try:
        retain_extension_diagnostic(
            ensure_project_layout(blend_path),
            error_code=error_code,
            category=category,
            state=state,
            phase=phase,
            detail=detail,
            target_scene=target_scene,
        )
        refresh_project_inventory(blend_path)
    except (DiagnosticReportError, OSError, RuntimeError, ValueError):
        pass


def clear_project_inventory() -> None:
    global _project_inventory, _project_inventory_key
    global _permanent_delete_status
    global _project_inventory_error
    if _project_inventory is not None:
        _project_inventory.close()
    _project_inventory = None
    _project_inventory_key = ""
    _project_inventory_pages.update(
        {"jobs": 0, "results": 0, "diagnostics": 0, "trash": 0}
    )
    _pending_lifecycle_plans.clear()
    _permanent_delete_status = ""
    _project_inventory_error = ""


def _inventory_for(blend_path: str) -> ProjectInventory:
    global _project_inventory, _project_inventory_key
    key = _project_key(blend_path)
    if _project_inventory is None or key != _project_inventory_key:
        clear_project_inventory()
        _project_inventory = ProjectInventory(blend_path)
        _project_inventory_key = key
    return _project_inventory


def project_inventory_snapshot(
    blend_path: str,
) -> InventorySnapshot | None:
    if not blend_path:
        return None
    return _inventory_for(blend_path).snapshot()


def advance_project_inventory(blend_path: str) -> InventorySnapshot | None:
    global _project_inventory_error
    if not blend_path:
        clear_project_inventory()
        return None
    inventory = _inventory_for(blend_path)
    try:
        return inventory.advance()
    except (ProjectLifecycleError, OSError) as exc:
        _project_inventory_error = str(exc)
        inventory.close()
        return inventory.snapshot()


def refresh_project_inventory(blend_path: str) -> None:
    clear_project_inventory()
    if blend_path:
        _inventory_for(blend_path)


def _inventory_rows(
    snapshot: InventorySnapshot, category: str
) -> tuple:
    page = _project_inventory_pages.get(category, 0)
    return snapshot.page(category, page)


def _draw_inventory_page_controls(
    layout, category: str, total: int, label: str
) -> None:
    page = _project_inventory_pages.get(category, 0)
    if page > 0:
        previous = layout.operator(
            LINGBOTMAP_OT_load_more_project_items.bl_idname,
            text=f"Previous {label} Page",
        )
        previous.category = category
        previous.direction = "previous"
    if (page + 1) * 200 < total:
        following = layout.operator(
            LINGBOTMAP_OT_load_more_project_items.bl_idname,
            text=f"Next {label} Page",
        )
        following.category = category
        following.direction = "next"


def ready_results_for_completed_job(
    blend_path: str, job_id: str
) -> tuple | None:
    """Return exact Ready Results only after bounded inventory is complete."""

    snapshot = project_inventory_snapshot(blend_path)
    if snapshot is None or not snapshot.complete:
        return None
    if snapshot.jobs_abnormal or _project_inventory_error:
        return ()
    root = project_result_root(blend_path) / "results"
    ready = []
    for item in snapshot.items:
        if (
            item.category != "results"
            or item.status != "recognized"
            or item.job_id != job_id
        ):
            continue
        try:
            ready.append(read_ready_result(root / item.name))
        except (OSError, RuntimeError, ValueError):
            continue
    return tuple(ready)


def _store_lifecycle_plan(
    blend_path: str, lifecycle: ProjectLifecycle, plan: LifecyclePlan
) -> str:
    if len(_pending_lifecycle_plans) >= 64:
        _pending_lifecycle_plans.pop(next(iter(_pending_lifecycle_plans)))
    key = uuid.uuid4().hex
    _pending_lifecycle_plans[key] = (
        _project_key(blend_path),
        lifecycle,
        plan,
    )
    return key


def _consume_lifecycle_plan(
    key: str, blend_path: str
) -> tuple[ProjectLifecycle, LifecyclePlan]:
    try:
        expected, lifecycle, plan = _pending_lifecycle_plans.pop(key)
    except KeyError as exc:
        raise ProjectLifecycleError(
            "Lifecycle confirmation expired; inspect and confirm again"
        ) from exc
    if expected != _project_key(blend_path):
        raise ProjectLifecycleError(
            "The Blender file changed after confirmation"
        )
    return lifecycle, plan


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
            _retain_ui_diagnostic(
                category="import",
                state="cancelled",
                error_code="import.transaction.cancelled",
                phase="result-import",
                detail=exc,
                scene=context.scene,
            )
            set_import_status(str(exc))
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        except (ImportCapacityError, ResultImportError, OSError, MemoryError) as exc:
            _retain_ui_diagnostic(
                category="import",
                state="failed",
                error_code="import.transaction.failed",
                phase="result-import",
                detail=exc,
                scene=context.scene,
            )
            set_import_status(str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            message = f"Import failed and was rolled back: {type(exc).__name__}: {exc}"
            _retain_ui_diagnostic(
                category="import",
                state="failed",
                error_code="import.transaction.rollback-failed",
                phase="result-import",
                detail=message,
                scene=context.scene,
            )
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


def _new_lifecycle_plan(action: str, names: tuple[str, ...]):
    blend_path = str(getattr(bpy.data, "filepath", ""))
    if not blend_path:
        raise ProjectLifecycleError(
            "Save the Blender file before managing Project content"
        )
    lifecycle = ProjectLifecycle(blend_path)
    try:
        plan = lifecycle.plan(action, names)
    except (ProjectLifecycleError, OSError) as exc:
        _retain_ui_diagnostic(
            category="lifecycle",
            state="failed",
            error_code="lifecycle.plan.failed",
            phase=action,
            detail=exc,
        )
        raise
    key = _store_lifecycle_plan(blend_path, lifecycle, plan)
    return blend_path, plan, key


def _execute_lifecycle_plan(key: str):
    blend_path = str(getattr(bpy.data, "filepath", ""))
    lifecycle, plan = _consume_lifecycle_plan(key, blend_path)
    try:
        outcome = lifecycle.execute(plan)
    except (ProjectLifecycleError, OSError) as exc:
        _retain_ui_diagnostic(
            category="lifecycle",
            state="failed",
            error_code="lifecycle.execute.failed",
            phase=plan.action,
            detail=exc,
        )
        raise
    refresh_project_inventory(blend_path)
    return outcome


class LINGBOTMAP_OT_refresh_project_inventory(bpy.types.Operator):
    bl_idname = "lingbot_map.refresh_project_inventory"
    bl_label = "Refresh Project Inventory"
    bl_description = (
        "Restart bounded direct-child discovery without changing disk content"
    )

    def execute(self, _context):
        try:
            refresh_project_inventory(str(bpy.data.filepath))
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_load_more_project_items(bpy.types.Operator):
    bl_idname = "lingbot_map.load_more_project_items"
    bl_label = "Load Next Project Page"
    bl_description = "Display the next bounded page of at most 200 items"

    category: StringProperty(options={"HIDDEN"})
    direction: StringProperty(options={"HIDDEN"}, default="next")

    def execute(self, _context):
        if self.category not in _project_inventory_pages:
            self.report({"ERROR"}, "Unknown Project inventory category")
            return {"CANCELLED"}
        if self.direction == "previous":
            _project_inventory_pages[self.category] = max(
                0, _project_inventory_pages[self.category] - 1
            )
        elif self.direction == "next":
            _project_inventory_pages[self.category] += 1
        else:
            self.report({"ERROR"}, "Unknown Project page direction")
            return {"CANCELLED"}
        return {"FINISHED"}


class LINGBOTMAP_OT_trash_result(bpy.types.Operator):
    bl_idname = "lingbot_map.trash_result"
    bl_label = "Move Result to Project Trash"
    bl_description = (
        "Completely validate this owned Result, then atomically move it to "
        "this Project's recoverable Trash"
    )

    result_name: StringProperty(options={"HIDDEN"})
    plan_key: StringProperty(options={"HIDDEN"})
    result_id: StringProperty(options={"HIDDEN"})
    item_count: StringProperty(options={"HIDDEN"})
    file_count: StringProperty(options={"HIDDEN"})
    byte_count: StringProperty(options={"HIDDEN"})
    collection_names: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        try:
            _blend, plan, self.plan_key = _new_lifecycle_plan(
                "trash_result", (self.result_name,)
            )
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.result_id = str(plan.result_id or "")
        self.item_count = str(plan.item_count)
        self.file_count = str(plan.file_count)
        self.byte_count = str(plan.byte_count)
        names = [
            collection.name
            for collection in _imported_reconstruction_collections(
                context.scene
            )
            if collection.get("lingbot_map_result_id") == plan.result_id
        ]
        self.collection_names = ", ".join(names) or "none"
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        self.layout.label(
            text="Atomically move this owned Result to Project Trash?",
            icon="QUESTION",
        )
        self.layout.label(text=f"Result: {self.result_id}")
        self.layout.label(
            text=(
                f"Items: {self.item_count} · files: {self.file_count} · "
                f"bytes: {self.byte_count}"
            )
        )
        self.layout.label(
            text=f"Current Blender Collections: {self.collection_names}"
        )
        self.layout.label(
            text="Collections remain usable; their disk reference becomes unavailable"
        )
        self.layout.label(
            text="References in other .blend files cannot be discovered",
            icon="INFO",
        )

    def execute(self, _context):
        try:
            outcome = _execute_lifecycle_plan(self.plan_key)
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Moved {outcome.item_count} Result to Project Trash",
        )
        return {"FINISHED"}


class LINGBOTMAP_OT_trash_dense(bpy.types.Operator):
    bl_idname = "lingbot_map.trash_dense"
    bl_label = "Remove Dense Predictions"
    bl_description = (
        "Completely validate retained Dense Predictions, then atomically "
        "move only that component to Project Trash"
    )

    result_name: StringProperty(options={"HIDDEN"})
    plan_key: StringProperty(options={"HIDDEN"})
    result_id: StringProperty(options={"HIDDEN"})
    chunk_count: StringProperty(options={"HIDDEN"})
    file_count: StringProperty(options={"HIDDEN"})
    byte_count: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        try:
            _blend, plan, self.plan_key = _new_lifecycle_plan(
                "trash_dense", (self.result_name,)
            )
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.result_id = str(plan.result_id or "")
        self.chunk_count = str(plan.chunk_count)
        self.file_count = str(plan.file_count)
        self.byte_count = str(plan.byte_count)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, _context):
        self.layout.label(
            text="Move only Dense Predictions to Project Trash?",
            icon="QUESTION",
        )
        self.layout.label(text=f"Result: {self.result_id}")
        self.layout.label(
            text=(
                f"Chunks: {self.chunk_count} · files: {self.file_count} · "
                f"bytes: {self.byte_count}"
            )
        )
        self.layout.label(text="The core Result and Blender content remain usable")

    def execute(self, _context):
        try:
            _execute_lifecycle_plan(self.plan_key)
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Dense Predictions moved to Project Trash")
        return {"FINISHED"}


class LINGBOTMAP_OT_trash_diagnostic(bpy.types.Operator):
    bl_idname = "lingbot_map.trash_diagnostic"
    bl_label = "Move Diagnostic to Project Trash"
    bl_description = (
        "Explicitly move this terminal Diagnostic to recoverable Project Trash"
    )

    diagnostic_name: StringProperty(options={"HIDDEN"})
    diagnostic_names_json: StringProperty(options={"HIDDEN"})
    plan_key: StringProperty(options={"HIDDEN"})
    item_count: StringProperty(options={"HIDDEN"})
    file_count: StringProperty(options={"HIDDEN"})
    byte_count: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        try:
            if self.diagnostic_names_json:
                raw_names = json.loads(self.diagnostic_names_json)
                if not isinstance(raw_names, list):
                    raise ProjectLifecycleError(
                        "Diagnostic selection is invalid"
                    )
                names = tuple(raw_names)
            else:
                names = (self.diagnostic_name,)
            if (
                not 1 <= len(names) <= 200
                or not all(isinstance(name, str) for name in names)
            ):
                raise ProjectLifecycleError(
                    "Diagnostic selection is invalid"
                )
            _blend, plan, self.plan_key = _new_lifecycle_plan(
                "trash_diagnostics", names
            )
        except (
            json.JSONDecodeError,
            ProjectLifecycleError,
            OSError,
        ) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.item_count = str(plan.item_count)
        self.file_count = str(plan.file_count)
        self.byte_count = str(plan.byte_count)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, _context):
        self.layout.label(
            text="Diagnostics are retained unless you explicitly confirm",
            icon="QUESTION",
        )
        self.layout.label(text=f"Selected items: {self.item_count}")
        self.layout.label(
            text=f"Files: {self.file_count} · bytes: {self.byte_count}"
        )

    def execute(self, _context):
        try:
            outcome = _execute_lifecycle_plan(self.plan_key)
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Moved {outcome.item_count} Diagnostics to Project Trash",
        )
        return {"FINISHED"}


def _diagnostic_report_source(name: str) -> Path:
    blend_path = str(getattr(bpy.data, "filepath", ""))
    if not blend_path:
        raise DiagnosticReportError(
            "diagnostic-export-unsaved-project",
            "Save the Blender file before exporting Diagnostics",
        )
    return project_result_root(blend_path) / "diagnostics" / name


def _diagnostic_versions() -> dict[str, object]:
    return {
        "extension": "0.1.0",
        "blender": str(getattr(bpy.app, "version_string", "unknown")),
    }


class LINGBOTMAP_OT_export_diagnostic_report(bpy.types.Operator):
    bl_idname = "lingbot_map.export_diagnostic_report"
    bl_label = "Export Diagnostic Report"
    bl_description = (
        "Export a bounded allowlisted ZIP; sensitive identity is redacted "
        "unless explicitly enabled for this export"
    )

    diagnostic_name: StringProperty(options={"HIDDEN"})
    filepath: StringProperty(
        name="Diagnostic Report",
        subtype="FILE_PATH",
    )
    filter_glob: StringProperty(default="*.zip", options={"HIDDEN"})
    include_sensitive_identity: BoolProperty(
        name="Include unredacted identity for this export",
        description=(
            "Include local usernames, machine names, absolute paths, "
            "environment data, source identifiers, and full GPU UUIDs"
        ),
        default=False,
    )

    def invoke(self, context, _event):
        blend_path = Path(str(getattr(bpy.data, "filepath", "")))
        if not blend_path.name:
            self.report({"ERROR"}, "Save the Blender file before exporting")
            return {"CANCELLED"}
        # Blender may remember an operator's last-used properties. Sensitive
        # identity is nevertheless opt-in for every individual export.
        self.include_sensitive_identity = False
        self.filepath = str(
            blend_path.parent
            / f"{self.diagnostic_name}-diagnostic-report.zip"
        )
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, _context):
        self.layout.label(
            text=(
                "Default redaction: users, machine, absolute paths, "
                "environment, source identity, full GPU UUID"
            ),
            icon="INFO",
        )
        self.layout.prop(self, "include_sensitive_identity")
        if self.include_sensitive_identity:
            self.layout.label(
                text="This one report will contain sensitive local identity",
                icon="ERROR",
            )

    def execute(self, _context):
        try:
            outcome = export_portable_report(
                _diagnostic_report_source(self.diagnostic_name),
                self.filepath,
                redact=not self.include_sensitive_identity,
                versions=_diagnostic_versions(),
            )
        except (DiagnosticReportError, OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            (
                "Exported redacted Diagnostic Report"
                if not self.include_sensitive_identity
                else "Exported explicitly unredacted Diagnostic Report"
            )
            + f" ({outcome['length']} bytes)",
        )
        return {"FINISHED"}


class LINGBOTMAP_OT_copy_diagnostic_report(bpy.types.Operator):
    bl_idname = "lingbot_map.copy_diagnostic_report"
    bl_label = "Copy Diagnostic Report"
    bl_description = (
        "Copy only the bounded redacted report representation to the clipboard"
    )

    diagnostic_name: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        try:
            report = build_portable_report(
                _diagnostic_report_source(self.diagnostic_name),
                redact=True,
                versions=_diagnostic_versions(),
            )
        except (DiagnosticReportError, OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.window_manager.clipboard = report.clipboard_text
        self.report({"INFO"}, "Copied redacted Diagnostic Report")
        return {"FINISHED"}


class LINGBOTMAP_OT_restore_trash(bpy.types.Operator):
    bl_idname = "lingbot_map.restore_trash"
    bl_label = "Restore from Project Trash"
    bl_description = (
        "Restore to the exact owning destination only when identity is unique "
        "and nothing would be overwritten"
    )

    trash_name: StringProperty(options={"HIDDEN"})
    plan_key: StringProperty(options={"HIDDEN"})
    destination: StringProperty(options={"HIDDEN"})
    file_count: StringProperty(options={"HIDDEN"})
    byte_count: StringProperty(options={"HIDDEN"})

    def invoke(self, context, _event):
        try:
            _blend, plan, self.plan_key = _new_lifecycle_plan(
                "restore", (self.trash_name,)
            )
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.destination = str(plan.destinations[0])
        self.file_count = str(plan.file_count)
        self.byte_count = str(plan.byte_count)
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        self.layout.label(
            text="Restore this exact validated Trash item?",
            icon="QUESTION",
        )
        self.layout.label(text=f"Files: {self.file_count} · bytes: {self.byte_count}")
        self.layout.label(text=f"Destination: {self.destination}")
        self.layout.label(text="Existing destinations are never overwritten")

    def execute(self, _context):
        try:
            _execute_lifecycle_plan(self.plan_key)
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Project Trash item restored")
        return {"FINISHED"}


class LINGBOTMAP_OT_delete_trash(bpy.types.Operator):
    bl_idname = "lingbot_map.delete_trash"
    bl_label = "Permanently Delete Trash Item"
    bl_description = (
        "Permanently delete one fully inspected direct Project Trash child; "
        "press Esc between files to stop"
    )

    trash_name: StringProperty(options={"HIDDEN"})
    plan_key: StringProperty(options={"HIDDEN"})
    confirmation: StringProperty(name="Type DELETE", default="")
    item_count: StringProperty(options={"HIDDEN"})
    file_count: StringProperty(options={"HIDDEN"})
    byte_count: StringProperty(options={"HIDDEN"})

    _timer = None
    _session: PermanentDeletionSession | None = None

    def invoke(self, context, _event):
        try:
            _blend, plan, self.plan_key = _new_lifecycle_plan(
                "delete", (self.trash_name,)
            )
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.item_count = str(plan.item_count)
        self.file_count = str(plan.file_count)
        self.byte_count = str(plan.byte_count)
        self.confirmation = ""
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        self.layout.label(
            text="Permanent deletion cannot be undone",
            icon="ERROR",
        )
        self.layout.label(
            text=(
                f"Items: {self.item_count} · files: {self.file_count} · "
                f"bytes: {self.byte_count}"
            )
        )
        self.layout.prop(self, "confirmation")
        self.layout.label(text="After starting, press Esc between files to cancel")

    def _remove_timer(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def execute(self, context):
        global _permanent_delete_status
        try:
            lifecycle, plan = _consume_lifecycle_plan(
                self.plan_key, str(bpy.data.filepath)
            )
            self._session = lifecycle.begin_delete(
                plan, confirmation=self.confirmation
            )
        except (ProjectLifecycleError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _permanent_delete_status = (
            f"Deleting {plan.file_count} inspected file(s); Esc cancels "
            "between files"
        )
        self._timer = context.window_manager.event_timer_add(
            0.01, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _permanent_delete_status
        if event.type == "ESC":
            partial = False
            try:
                self._session.cancel()
            except PartialDeletionError as exc:
                partial = True
                self.report({"WARNING"}, str(exc))
            except ProjectLifecycleError as exc:
                self.report({"INFO"}, str(exc))
            self._remove_timer(context)
            refresh_project_inventory(str(bpy.data.filepath))
            _permanent_delete_status = (
                "Permanent deletion cancelled; inspect partial_delete items"
                if partial
                else "Permanent deletion cancelled before changing Trash"
            )
            self._session = None
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            outcome = self._session.step()
        except PartialDeletionError as exc:
            self._remove_timer(context)
            refresh_project_inventory(str(bpy.data.filepath))
            _permanent_delete_status = str(exc)
            self._session = None
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except (ProjectLifecycleError, OSError) as exc:
            self._remove_timer(context)
            refresh_project_inventory(str(bpy.data.filepath))
            _permanent_delete_status = str(exc)
            self._session = None
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _permanent_delete_status = (
            f"Permanent deletion: {self._session.completed_files}/"
            f"{self._session.total_files} files"
        )
        if outcome is None:
            return {"RUNNING_MODAL"}
        self._remove_timer(context)
        refresh_project_inventory(str(bpy.data.filepath))
        _permanent_delete_status = (
            f"Permanently deleted {outcome.item_count} Trash item(s)"
        )
        self._session = None
        self.report({"INFO"}, _permanent_delete_status)
        return {"FINISHED"}

    def cancel(self, context):
        global _permanent_delete_status
        message = ""
        if self._session is not None:
            try:
                self._session.cancel()
            except (PartialDeletionError, ProjectLifecycleError) as exc:
                message = str(exc)
            self._session = None
        self._remove_timer(context)
        refresh_project_inventory(str(getattr(bpy.data, "filepath", "")))
        _permanent_delete_status = message


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
        snapshot = project_inventory_snapshot(blend_path)
        self.layout.operator(
            LINGBOTMAP_OT_refresh_project_inventory.bl_idname,
            text="Refresh Project Inventory",
            icon="FILE_REFRESH",
        )
        if snapshot is None:
            self.layout.label(
                text="Save the Blender file to discover Project Results",
                icon="INFO",
            )
            results = ()
            lifecycle_actions_enabled = False
        else:
            state = "complete" if snapshot.complete else "scanning"
            self.layout.label(
                text=(
                    f"Project inventory: {state} · "
                    f"{snapshot.scanned_entries:,} direct entries"
                )
            )
            if _project_inventory_error:
                self.layout.label(
                    text=f"Inventory stopped safely: {_project_inventory_error}",
                    icon="ERROR",
                )
            if snapshot.jobs_abnormal:
                self.layout.label(
                    text=(
                        "Lifecycle actions disabled: more than 32 "
                        ".jobs entries"
                    ),
                    icon="ERROR",
                )
            lifecycle_actions_enabled = bool(
                not snapshot.jobs_abnormal
                and not _project_inventory_error
            )
            results = _inventory_rows(snapshot, "results")
            result_total = sum(
                item.category == "results" for item in snapshot.items
            )
            _draw_inventory_page_controls(
                self.layout, "results", result_total, "Results"
            )
        imported = _imported_reconstruction_collections(context.scene)
        imported_by_id = {}
        for collection in imported:
            imported_by_id.setdefault(
                collection.get("lingbot_map_result_id"), []
            ).append(collection)
        if (
            not results
            and not imported
            and (snapshot is None or snapshot.complete)
        ):
            self.layout.label(text="No Reconstruction Results discovered")
            return
        self.layout.label(text=get_import_status())
        drawn_imported = set()
        for result in results:
            box = self.layout.box()
            if result.status != "recognized":
                box.label(
                    text=f"Unrecognized: {result.name}",
                    icon="ERROR",
                )
                box.label(text=result.detail or "No action is available")
                continue
            box.label(
                text="Discovered Result · fully validates on action",
                icon="INFO",
            )
            box.label(
                text=(
                    f"{result.profile_name}: "
                    f"{int(result.point_count or 0):,} points"
                )
            )
            box.label(
                text=(
                    f"{int(result.frame_count or 0):,} frames · "
                    f"{result.result_id}"
                )
            )
            box.label(
                text=f"Target Scene: {result.scene_name} · {result.scene_uuid}"
            )
            result_directory = (
                project_result_root(blend_path)
                / "results"
                / result.name
            )
            claims = tuple(imported_by_id.get(result.result_id, ()))
            if not claims:
                if (
                    str(
                        context.scene.get(
                            "lingbot_map_scene_uuid", ""
                        )
                    )
                    == result.scene_uuid
                ):
                    action = box.operator(
                        LINGBOTMAP_OT_import_result.bl_idname,
                        text="Import",
                        icon="IMPORT",
                    )
                    action.result_directory = str(result_directory)
                self._draw_import_into_actions(
                    box, context.scene, result_directory
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
            if result.dense_status == "retained-unvalidated":
                box.label(
                    text="Dense Predictions: retained; validates on action",
                    icon="INFO",
                )
                if lifecycle_actions_enabled:
                    dense = box.operator(
                        LINGBOTMAP_OT_trash_dense.bl_idname,
                        text="Remove Dense Predictions",
                        icon="TRASH",
                    )
                    dense.result_name = result.name
            if lifecycle_actions_enabled:
                trash = box.operator(
                    LINGBOTMAP_OT_trash_result.bl_idname,
                    text="Move Result to Project Trash",
                    icon="TRASH",
                )
                trash.result_name = result.name
            else:
                box.label(
                    text="Project disk lifecycle actions are disabled",
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


def _draw_project_trash(layout, snapshot, lifecycle_actions_enabled):
    layout.separator()
    layout.label(text="Project Trash")
    if _permanent_delete_status:
        layout.label(
            text=_permanent_delete_status,
            icon=(
                "ERROR"
                if "partial" in _permanent_delete_status.casefold()
                else "INFO"
            ),
        )
    trash_items = _inventory_rows(snapshot, "trash")
    total = sum(item.category == "trash" for item in snapshot.items)
    if not trash_items:
        layout.label(
            text=(
                "Project Trash is empty"
                if snapshot.complete
                else "Scanning direct Project entries…"
            )
        )
    for item in trash_items:
        box = layout.box()
        if item.status == "unrecognized" or not item.ordinary:
            box.label(
                text=f"Unrecognized: {item.name}",
                icon="ERROR",
            )
            box.label(text=item.detail or "No action is available")
            continue
        partial = item.status.endswith("-partial_delete")
        box.label(
            text=f"{item.status}: {item.name}",
            icon="ERROR" if partial else "INFO",
        )
        if partial:
            box.label(
                text="Deletion was interrupted; inspect or retry",
                icon="ERROR",
            )
        elif lifecycle_actions_enabled:
            restore = box.operator(
                LINGBOTMAP_OT_restore_trash.bl_idname,
                text="Restore Exact Item",
            )
            restore.trash_name = item.name
        if lifecycle_actions_enabled:
            delete = box.operator(
                LINGBOTMAP_OT_delete_trash.bl_idname,
                text=(
                    "Retry Permanent Delete"
                    if partial
                    else "Permanently Delete"
                ),
                icon="TRASH",
            )
            delete.trash_name = item.name
        else:
            box.label(
                text="Project disk lifecycle actions are disabled",
                icon="ERROR",
            )
    _draw_inventory_page_controls(
        layout, "trash", total, "Trash"
    )


class LINGBOTMAP_PT_diagnostics(_LINGBOTMAP_LifecyclePanel, bpy.types.Panel):
    bl_idname = "LINGBOTMAP_PT_diagnostics"
    bl_label = "Diagnostics"
    bl_order = 4

    def draw(self, _context):
        blend_path = str(getattr(bpy.data, "filepath", ""))
        snapshot = project_inventory_snapshot(blend_path)
        if snapshot is None:
            self.layout.label(
                text="Save the Blender file to inspect Diagnostics"
            )
            return
        if _project_inventory_error:
            self.layout.label(
                text=f"Inventory stopped safely: {_project_inventory_error}",
                icon="ERROR",
            )
        lifecycle_actions_enabled = bool(
            not snapshot.jobs_abnormal
            and not _project_inventory_error
        )
        if snapshot.jobs_abnormal:
            self.layout.label(
                text="Abnormal Project: more than 32 .jobs entries",
                icon="ERROR",
            )
        unrecognized_jobs = tuple(
            item
            for item in snapshot.items
            if item.category == "jobs"
            and item.status != "recognized"
        )
        job_page = _project_inventory_pages["jobs"]
        visible_jobs = unrecognized_jobs[
            job_page * 200 : (job_page + 1) * 200
        ]
        for item in visible_jobs:
            box = self.layout.box()
            box.label(
                text=f"Unrecognized .jobs entry: {item.name}",
                icon="ERROR",
            )
            box.label(text=item.detail or "No lifecycle action is available")
        _draw_inventory_page_controls(
            self.layout,
            "jobs",
            len(unrecognized_jobs),
            "Unrecognized Jobs",
        )
        diagnostics = _inventory_rows(snapshot, "diagnostics")
        total = sum(
            item.category == "diagnostics" for item in snapshot.items
        )
        if not diagnostics:
            self.layout.label(
                text=(
                    "No Diagnostics"
                    if snapshot.complete
                    else "Scanning direct Project entries…"
                )
            )
        for item in diagnostics:
            box = self.layout.box()
            icon = "INFO" if item.status == "recognized" else "ERROR"
            box.label(text=item.name, icon=icon)
            if item.status != "recognized":
                box.label(text=item.detail or "Unrecognized; no action")
                continue
            box.label(text="Retained by default")
            report_actions = box.row(align=True)
            export = report_actions.operator(
                LINGBOTMAP_OT_export_diagnostic_report.bl_idname,
                text="Export Report",
                icon="EXPORT",
            )
            export.diagnostic_name = item.name
            copy = report_actions.operator(
                LINGBOTMAP_OT_copy_diagnostic_report.bl_idname,
                text="Copy Report",
                icon="COPYDOWN",
            )
            copy.diagnostic_name = item.name
            if lifecycle_actions_enabled:
                action = box.operator(
                    LINGBOTMAP_OT_trash_diagnostic.bl_idname,
                    text="Move to Project Trash",
                    icon="TRASH",
                )
                action.diagnostic_name = item.name
        recognized_diagnostics = tuple(
            item.name
            for item in diagnostics
            if item.status == "recognized"
        )[:200]
        if (
            lifecycle_actions_enabled
            and len(recognized_diagnostics) > 1
        ):
            multi = self.layout.operator(
                LINGBOTMAP_OT_trash_diagnostic.bl_idname,
                text=(
                    "Move First "
                    f"{len(recognized_diagnostics)} Displayed Diagnostics"
                ),
                icon="TRASH",
            )
            multi.diagnostic_names_json = json.dumps(
                recognized_diagnostics,
                separators=(",", ":"),
            )
        _draw_inventory_page_controls(
            self.layout, "diagnostics", total, "Diagnostics"
        )
        _draw_project_trash(
            self.layout, snapshot, lifecycle_actions_enabled
        )


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
    LINGBOTMAP_OT_refresh_project_inventory,
    LINGBOTMAP_OT_load_more_project_items,
    LINGBOTMAP_OT_trash_result,
    LINGBOTMAP_OT_trash_dense,
    LINGBOTMAP_OT_trash_diagnostic,
    LINGBOTMAP_OT_export_diagnostic_report,
    LINGBOTMAP_OT_copy_diagnostic_report,
    LINGBOTMAP_OT_restore_trash,
    LINGBOTMAP_OT_delete_trash,
    LINGBOTMAP_PT_setup,
    LINGBOTMAP_PT_reconstruct,
    LINGBOTMAP_PT_active_job,
    LINGBOTMAP_PT_results,
    LINGBOTMAP_PT_diagnostics,
)
