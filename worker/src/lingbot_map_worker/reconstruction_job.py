"""Qualified native-Windows Reconstruction Job for complete Capture Sources."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import threading
from typing import Any

from . import __version__
from .canonical_preprocessing import canonical_geometry
from .capability import (
    CapabilityCache,
    CapabilityIdentity,
    CapabilityLaunchGate,
    CapabilityStack,
    ModelFingerprint,
)
from .decoder import (
    DecodeContract,
    SourceIdentity,
    iter_capture_source_frames,
)
from .fixture_job import (
    FixtureJobError,
    StatusStore,
    _heartbeat_loop,
    _install_audit_policy,
    _publish_worker_identity,
    _set_below_normal_priority,
    _wait_for_control,
    validate_job_spec,
)
from .gpu_devices import NvmlDeviceProvider, PhysicalGpu, select_physical_gpu
from .gpu_profiles import (
    PROFILE_BY_NAME,
    ProfileSelection,
    ReconstructionProfile,
    inference_plan,
    resolve_profile,
)
from .ipc import read_json
from .human_log import HumanLogSession, write_terminal_diagnostic
from .model_store_compat import is_reparse_point
from .long_pipeline import WindowedReconstructionPipeline
from .production_model import (
    TorchPredictionDecoder,
    load_production_adapter,
    load_production_window_predictor,
)
from .provenance import result_provenance
from .result_bundle import ResultCancelled
from .result_pipeline import (
    IncrementalBundleResultSink,
    IncrementalResultRequest,
    ResultProfile,
)
from .result_resources import (
    SystemResourceProbe,
    estimate_fixture_memory_bytes,
    estimate_dense_buffer_bytes,
    estimate_sky_mask_buffer_bytes,
    estimate_window_alignment_memory_bytes,
    require_project_disk,
    require_worker_memory,
)
from .short_pipeline import (
    PipelineCancelled,
    ProgressReporter,
    ShortReconstructionPipeline,
    SourceFrame,
)
from .sky_masking import (
    SkyMaskCancelled,
    SkyMaskRequest,
    SkyMaskSession,
)
from .sleep_guard import WindowsExecutionState, WindowsSleepGuard
from .torch_capability import (
    EXPECTED_CUDA_VERSION,
    EXPECTED_TORCH_VERSION,
    classify_cuda_failure,
)


class ReconstructionJobError(FixtureJobError):
    pass


def _cancelled(job_dir: Path) -> bool:
    return (job_dir / "cancel.request").exists()


def _thread_budget() -> int:
    return max(1, min(8, int(os.cpu_count() or 1) - 2))


def _require_plain_managed_path(path: Path, managed_root: Path) -> None:
    current = path
    while True:
        if current.exists() and (current.is_symlink() or is_reparse_point(current)):
            raise ReconstructionJobError(f"Linked managed path is forbidden: {current}")
        if current == managed_root:
            return
        if current.parent == current or not current.is_relative_to(managed_root):
            raise ReconstructionJobError("Reconstruction Model escaped the managed root")
        current = current.parent


def _validate_catalogued_auxiliary(
    *,
    managed_root: Path,
    runtime_id: str,
    model: dict[str, Any],
) -> None:
    catalog_path = managed_root / "runtimes" / runtime_id / "model-catalog.json"
    _require_plain_managed_path(catalog_path, managed_root)
    document = read_json(catalog_path)
    if (
        not isinstance(document, dict)
        or set(document) != {"catalog_version", "models"}
        or document["catalog_version"] != model["catalog_version"]
        or not isinstance(document["models"], list)
    ):
        raise ReconstructionJobError("Worker Model Catalog identity is invalid")
    matches = [
        entry
        for entry in document["models"]
        if isinstance(entry, dict) and entry.get("id") == model["id"]
    ]
    if len(matches) != 1:
        raise ReconstructionJobError("Auxiliary Model is absent from the Worker Catalog")
    entry = matches[0]
    artifact = entry.get("artifact")
    if (
        entry.get("role") != "auxiliary"
        or entry.get("architecture") != "Sky segmentation ONNX CPU native-v1"
        or not isinstance(entry.get("input_contract"), dict)
        or entry["input_contract"].get("provider") != "CPUExecutionProvider"
        or entry["input_contract"].get("mask_rule")
        != "non-sky-confidence-gt-0.1-v1"
        or not isinstance(artifact, dict)
        or artifact.get("sha256") != model["sha256"]
    ):
        raise ReconstructionJobError("Auxiliary Model disagrees with the Worker Catalog")


def _selected_profile(raw: dict[str, Any]) -> tuple[Any, ReconstructionProfile]:
    selection = ProfileSelection(
        raw["name"],
        int(raw["camera_iterations"]),
        float(raw["confidence_cutoff_percent"]),
        float(raw["depth_cutoff_percent"]),
        int(raw["import_point_budget"]),
        bool(raw["point_budget_confirmed"]),
        bool(raw.get("retain_dense_predictions", False)),
    )
    resolved = resolve_profile(selection)
    if resolved.name != raw["name"]:
        raise ReconstructionJobError(
            f"Profile label {raw['name']} does not match its exact settings ({resolved.name})"
        )
    execution = ReconstructionProfile(
        resolved.name,
        resolved.camera_iterations,
        int(resolved.confidence_cutoff_percent),
        resolved.import_point_budget,
        float(resolved.depth_cutoff_percent),
    )
    return resolved, execution


def _terminal_destination(job: dict[str, Any], state: str) -> Path:
    return (
        Path(job["project_root"])
        / "diagnostics"
        / f"{job['job_id']}--reconstruction-{state}"
    )


def run_reconstruction_job(spec_path: Path, nonce: str) -> int:
    spec_path = Path(os.path.abspath(spec_path))
    if not spec_path.is_absolute() or spec_path.name != "job-spec.json":
        raise ReconstructionJobError("Reconstruction JobSpec must be an absolute job-spec.json path")
    job_dir = spec_path.parent
    job = validate_job_spec(read_json(spec_path))
    if "reconstruction" not in job:
        raise ReconstructionJobError("Reconstruction runner requires a reconstruction JobSpec")
    if job_dir.name != job["job_id"] or job_dir.parent.name != ".jobs":
        raise ReconstructionJobError("Reconstruction JobSpec is outside its bounded active directory")
    if Path(os.path.abspath(job["project_root"])) != job_dir.parent.parent:
        raise ReconstructionJobError("Project Result Root does not contain this Reconstruction Job")

    reconstruction = job["reconstruction"]
    source_raw = reconstruction["source"]
    preflight = reconstruction["preflight"]
    gpu_raw = reconstruction["gpu"]
    model = reconstruction["model"]
    sky_mask = reconstruction.get("sky_mask", {"enabled": False})
    sky_mask_enabled = bool(sky_mask["enabled"])
    auxiliary_model = sky_mask.get("model") if sky_mask_enabled else None
    managed_root = Path(os.path.abspath(reconstruction["managed_root"]))
    model_path = Path(os.path.abspath(model["path"]))
    if (
        not model_path.is_relative_to(managed_root)
    ):
        raise ReconstructionJobError("Reconstruction Model escaped the plain managed store")
    _require_plain_managed_path(model_path, managed_root)
    auxiliary_model_path: Path | None = None
    if auxiliary_model is not None:
        auxiliary_model_path = Path(os.path.abspath(auxiliary_model["path"]))
        if not auxiliary_model_path.is_relative_to(managed_root):
            raise ReconstructionJobError(
                "Sky Mask Auxiliary Model escaped the plain managed store"
            )
        _require_plain_managed_path(auxiliary_model_path, managed_root)
    resolved_profile, execution_profile = _selected_profile(reconstruction["profile"])
    retain_dense = bool(reconstruction["profile"].get("retain_dense_predictions", False))
    plan = inference_plan(int(preflight["frame_count"]))
    capability_profile = PROFILE_BY_NAME[gpu_raw["capability_profile_name"]]
    if (
        capability_profile.settings_sha256
        != gpu_raw["capability_profile_settings_sha256"]
        or capability_profile.camera_iterations < execution_profile.camera_iterations
    ):
        raise ReconstructionJobError(
            "Selected settings do not match the qualified camera-iteration workload"
        )

    _publish_worker_identity(job_dir, nonce)
    control = _wait_for_control(job_dir, job, nonce)
    total_frames = int(preflight["frame_count"])
    store = StatusStore(job_dir, job["job_id"], total_frames)
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(store, stop, float(reconstruction["heartbeat_interval_seconds"]), None),
        name="LingBotMap-Heartbeat",
        daemon=True,
    )
    state, message, error, return_code = (
        "failed", "Reconstruction Job failed", None, 1
    )
    caught_exception = None
    human_log = HumanLogSession(
        job_dir,
        on_discard=store.log_truncated,
        on_warning=store.structured_warning,
    ).start()
    try:
        _set_below_normal_priority()
        _install_audit_policy()
        heartbeat.start()
        budget = _thread_budget()
        decoder_threads = 1
        onnx_threads = 1
        torch_threads = max(
            1,
            budget - decoder_threads - (onnx_threads if sky_mask_enabled else 0),
        )
        source = Path(source_raw["absolute_path"])
        contract = DecodeContract(
            SourceIdentity(
                int(source_raw["size_bytes"]),
                int(source_raw["modification_time_ns"]),
                source_raw["sha256"],
            ),
            total_frames,
            int(preflight["video_stream_index"]),
            int(preflight["displayed_width"]),
            int(preflight["displayed_height"]),
            preflight["display_transform"],
            preflight["color_standard"],
            preflight["color_range"],
        )
        canonical_height, canonical_width, source_to_model, coverage = canonical_geometry(
            int(preflight["displayed_height"]), int(preflight["displayed_width"])
        )
        grid_pixels = canonical_height * canonical_width
        probe = SystemResourceProbe()
        estimated_memory = estimate_fixture_memory_bytes(
            total_frames, grid_pixels, execution_profile.import_point_budget
        )
        if retain_dense:
            estimated_memory += estimate_dense_buffer_bytes(total_frames, grid_pixels)
        if plan.mode == "windowed":
            estimated_memory += estimate_window_alignment_memory_bytes(grid_pixels)
        if sky_mask_enabled:
            estimated_memory += estimate_sky_mask_buffer_bytes(
                total_frames, grid_pixels
            )
        require_worker_memory(probe, estimated_memory)
        source_document = {
            "absolute_path": source_raw["absolute_path"],
            "scene_relative_path": source_raw["scene_relative_path"],
            "size_bytes": source_raw["size_bytes"],
            "modification_time_ns": source_raw["modification_time_ns"],
            "sha256": source_raw["sha256"],
        }
        warnings = ()
        if preflight["variable_frame_rate"]:
            warnings = ({
                "code": "variable-frame-rate",
                "message": "Source presentation timestamps are variable and were preserved exactly",
            },)
        guard_holder: dict[str, WindowsSleepGuard] = {}
        model_loaded = False

        gpu = PhysicalGpu(
            gpu_raw["uuid"],
            gpu_raw["name"],
            int(gpu_raw["total_memory"]),
            gpu_raw["driver_version"],
            tuple(gpu_raw["compute_capability"]),
        )
        devices = NvmlDeviceProvider()
        observed = select_physical_gpu(devices.discover(), gpu.uuid)
        if observed != gpu:
            raise ReconstructionJobError("Selected physical GPU identity changed after launch")
        stack = CapabilityStack(
            control["runtime_id"],
            reconstruction["worker_lock_sha256"],
            __version__,
            EXPECTED_TORCH_VERSION,
            EXPECTED_CUDA_VERSION,
        )
        identity = CapabilityIdentity(
            gpu.uuid,
            gpu.name,
            gpu.total_memory,
            gpu.driver_version,
            gpu.compute_capability,
            stack,
            ModelFingerprint(model["catalog_version"], model["id"], model["sha256"]),
            capability_profile.name,
            capability_profile.settings_sha256,
        )
        gate = CapabilityLaunchGate(
            CapabilityCache(managed_root / "capability-cache"),
            devices,
        )
        sky_session: SkyMaskSession | None = None
        if sky_mask_enabled:
            assert auxiliary_model is not None
            assert auxiliary_model_path is not None
            _validate_catalogued_auxiliary(
                managed_root=managed_root,
                runtime_id=control["runtime_id"],
                model=auxiliary_model,
            )
            sky_session = SkyMaskSession(
                SkyMaskRequest(
                    managed_root=managed_root,
                    source_sha256=source_raw["sha256"],
                    video_stream_index=int(preflight["video_stream_index"]),
                    display_transform=preflight["display_transform"],
                    color_standard=preflight["color_standard"],
                    color_range=preflight["color_range"],
                    frame_count=total_frames,
                    model_grid_shape=(canonical_height, canonical_width),
                    model_id=auxiliary_model["id"],
                    model_path=auxiliary_model_path,
                    model_sha256=auxiliary_model["sha256"],
                    worker_version=__version__,
                    onnx_threads=onnx_threads,
                    cancel=lambda: _cancelled(job_dir),
                )
            )

        def resource_check() -> None:
            require_worker_memory(probe, estimated_memory)
            require_project_disk(
                probe,
                Path(job["project_root"]),
                sink.estimated_remaining_bytes,
            )

        def cuda_check() -> None:
            nonlocal model_loaded
            current = select_physical_gpu(devices.discover(), gpu.uuid)
            if current != gpu:
                raise ReconstructionJobError("Physical GPU changed after system resume")
            if model_loaded:
                import torch
                torch.cuda.synchronize(torch.device("cuda:0"))

        progress: ProgressReporter

        def emit_progress(event) -> None:
            phase_total = event.total if event.total is not None else 1
            completed = min(int(event.completed), int(phase_total))
            store.emit(
                event.kind if event.kind != "resume" else "phase",
                event.phase,
                completed,
                (
                    f"{event.phase} {completed}/{phase_total}"
                    + (f"; ETA {event.eta_seconds:.1f}s" if event.eta_seconds is not None else "")
                ),
                total=int(phase_total),
                eta_seconds=event.eta_seconds,
                immediate=event.immediate,
            )

        progress = ProgressReporter(emit_progress)
        sleep_guard = WindowsSleepGuard(
            WindowsExecutionState(),
            resume_checks=(cuda_check, resource_check),
            on_resume=progress.resume,
        )
        guard_holder["guard"] = sleep_guard

        def provenance() -> dict[str, Any]:
            guard = guard_holder["guard"]
            return result_provenance(
                runtime_id=control["runtime_id"],
                worker_version=__version__,
                job_spec_sha256=control["job_spec"]["sha256"],
                model_id=model["id"],
                model_sha256=model["sha256"],
                source_sha256=source_raw["sha256"],
                profile_name=resolved_profile.name,
                camera_iterations=execution_profile.camera_iterations,
                confidence_cutoff_percent=resolved_profile.confidence_cutoff_percent,
                depth_cutoff_percent=resolved_profile.depth_cutoff_percent,
                import_point_budget=resolved_profile.import_point_budget,
                retain_dense_predictions=retain_dense,
                plan=plan,
                gpu={
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "total_memory": gpu.total_memory,
                    "driver_version": gpu.driver_version,
                    "compute_capability": list(gpu.compute_capability),
                    "torch_version": EXPECTED_TORCH_VERSION,
                    "cuda_version": EXPECTED_CUDA_VERSION,
                    "attention_backend": "sdpa",
                },
                suspension_count=guard.suspension_count,
                suspension_seconds=guard.suspension_seconds,
                decoder_threads=decoder_threads,
                onnx_threads=onnx_threads,
                torch_intraop_threads=torch_threads,
                torch_interop_threads=1,
            )

        sink = IncrementalBundleResultSink(
            IncrementalResultRequest(
                job_id=job["job_id"],
                project_root=Path(job["project_root"]),
                target_scene=job["target_scene"],
                timeline_start=job["timeline_start"],
                source=source_document,
                source_to_model=source_to_model,
                frame_count=total_frames,
                profile=ResultProfile(
                    resolved_profile.name,
                    float(resolved_profile.confidence_cutoff_percent),
                    float(resolved_profile.depth_cutoff_percent),
                    int(resolved_profile.import_point_budget),
                    float(reconstruction["initial_voxel_edge_length"]),
                    retain_dense,
                ),
                provenance={},
                warnings=warnings,
                created_utc=datetime.now(timezone.utc).isoformat(),
                model_grid_shape=(canonical_height, canonical_width),
                source_display={
                    "width": int(preflight["displayed_width"]),
                    "height": int(preflight["displayed_height"]),
                    "display_transform": preflight["display_transform"],
                },
                model_coverage={
                    "coordinate_space": "source-display-pixel-edges",
                    "polygon": [
                        [float(x), float(y)] for x, y in coverage
                    ],
                    "source_fraction": float(
                        (
                            (coverage[1][0] - coverage[0][0])
                            * (coverage[2][1] - coverage[1][1])
                        )
                        / (
                            int(preflight["displayed_width"])
                            * int(preflight["displayed_height"])
                        )
                    ),
                    "model_width": canonical_width,
                    "model_height": canonical_height,
                },
            ),
            prediction_decoder=(
                TorchPredictionDecoder()
                if plan.mode == "streaming"
                else lambda prediction, _canonical, _pts: prediction
            ),
            resource_probe=probe,
            cancel=lambda: _cancelled(job_dir),
            provenance_factory=provenance,
            sky_mask_session=sky_session,
        )

        def adapter_factory(current_plan, _profile, frame_shape):
            nonlocal model_loaded
            import torch
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(1)
            adapter = load_production_adapter(
                model_path=model_path,
                expected_sha256=model["sha256"],
                plan=current_plan,
                profile=execution_profile,
                frame_shape=frame_shape,
                cancel=lambda: _cancelled(job_dir),
                torch_module=torch,
            )
            model_loaded = True
            return adapter

        def predictor_factory(current_plan, _profile, _frame_shape):
            nonlocal model_loaded
            import torch
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(1)
            predictor = load_production_window_predictor(
                model_path=model_path,
                expected_sha256=model["sha256"],
                plan=current_plan,
                profile=execution_profile,
                cancel=lambda: _cancelled(job_dir),
                torch_module=torch,
            )
            model_loaded = True
            return predictor

        frame_source = (
            SourceFrame(item.frame_index, item.pts_seconds, item.srgb)
            for item in iter_capture_source_frames(
                source,
                contract,
                cancel=lambda: _cancelled(job_dir),
                thread_budget=decoder_threads,
            )
        )
        with gate.acquire(identity, nonce=nonce, job_id=job["job_id"]):
            pipeline = (
                ShortReconstructionPipeline(adapter_factory)
                if plan.mode == "streaming"
                else WindowedReconstructionPipeline(predictor_factory)
            )
            outcome = pipeline.run(
                frame_count=total_frames,
                frames=frame_source,
                profile=ProfileSelection(
                    resolved_profile.name,
                    execution_profile.camera_iterations,
                    float(resolved_profile.confidence_cutoff_percent),
                    float(resolved_profile.depth_cutoff_percent),
                    int(resolved_profile.import_point_budget),
                    True,
                    retain_dense,
                ),
                result_sink=sink,
                progress=progress,
                execution_guard=sleep_guard,
                health_check=lambda _phase, _index: (cuda_check(), resource_check()),
                cancel=lambda: _cancelled(job_dir),
            )
        state, message, return_code = (
            "succeeded",
            f"Ready Result {outcome.published.result_id} published",
            0,
        )
    except (PipelineCancelled, ResultCancelled, SkyMaskCancelled):
        state, message, return_code = "cancelled", "Reconstruction Job cancelled", 2
    except Exception as exc:
        caught_exception = exc
        try:
            classification = classify_cuda_failure(exc)
        except Exception:
            classification = None
        if classification and "gate" in locals() and "identity" in locals():
            gate.cache.invalidate_matching(identity, classification)
        error = f"{type(exc).__name__}: {exc}"[:16384]
    finally:
        if "sink" in locals() and state != "succeeded":
            try:
                sink.abort(state)
            except Exception as dense_abort_error:
                if caught_exception is None:
                    caught_exception = dense_abort_error
                if error is None:
                    error = (
                        f"{type(dense_abort_error).__name__}: {dense_abort_error}"
                    )[:16384]
        elif "sky_session" in locals() and sky_session is not None:
            try:
                sky_session.abort()
            except Exception as sky_abort_error:
                if caught_exception is None:
                    caught_exception = sky_abort_error
                if error is None:
                    error = (
                        f"{type(sky_abort_error).__name__}: {sky_abort_error}"
                    )[:16384]
        stop.set()
        if heartbeat.is_alive():
            heartbeat.join(timeout=6)
        error_code = f"pipeline.reconstruction.{state}"
        store.terminal(
            state,
            message,
            error=(
                None
                if state == "succeeded"
                else (f"{error_code}: {error}" if error else error_code)
            ),
        )
        human_log.close()
        if state != "succeeded":
            write_terminal_diagnostic(
                job_dir,
                job,
                error_code=error_code,
                state=state,
                phase=store.diagnostic_phase,
                detail=error or message,
                discarded_log_bytes=human_log.discarded_bytes,
                exception=caught_exception,
            )
    destination = _terminal_destination(job, state)
    if destination.exists():
        raise ReconstructionJobError(f"terminal diagnostics already exist: {destination.name}")
    os.replace(job_dir, destination)
    return return_code
