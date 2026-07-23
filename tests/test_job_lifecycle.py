from __future__ import annotations

import json
import io
import hashlib
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest import mock

if "bpy" not in sys.modules:
    bpy = ModuleType("bpy")
    bpy.app = SimpleNamespace(version=(5, 2, 1), online_access=False)
    bpy.types = SimpleNamespace(
        AddonPreferences=type("AddonPreferences", (), {}),
        Operator=type("Operator", (), {}),
        Panel=type("Panel", (), {}),
    )
    bpy.utils = SimpleNamespace(register_class=lambda _cls: None, unregister_class=lambda _cls: None)
    props = ModuleType("bpy.props")
    props.BoolProperty = lambda **kwargs: kwargs
    props.EnumProperty = lambda **kwargs: kwargs
    props.FloatProperty = lambda **kwargs: kwargs
    props.IntProperty = lambda **kwargs: kwargs
    props.StringProperty = lambda **kwargs: kwargs
    bpy.props = props
    sys.modules["bpy"] = bpy
    sys.modules["bpy.props"] = props

from blender_extension.ipc import (
    IpcError,
    MAX_DOCUMENT_BYTES,
    atomic_write_json,
    parse_json_bytes,
    parse_json_line,
    read_json,
)
from blender_extension.job_lifecycle import (
    JobController,
    JobLifecycleError,
    capture_source_draft_path,
    ensure_project_layout,
    ensure_unique_scene_uuid,
    project_result_root,
    scene_relative_capture_path,
    validate_control,
    validate_status,
    WorkerRecord,
)
from blender_extension.results import discover_ready_results

JOB_LIFECYCLE_MODULE = sys.modules[JobController.__module__]
IPC_MODULE = sys.modules[parse_json_bytes.__module__]


class Scene(dict):
    pass


class StrictIpcTests(unittest.TestCase):
    def test_rejects_bom_malformed_utf8_duplicate_nonfinite_and_oversize(self):
        invalid = (
            b"\xef\xbb\xbf{}",
            b'"\xff"',
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            b" " * (MAX_DOCUMENT_BYTES + 1),
        )
        for document in invalid:
            with self.subTest(document=document[:30]):
                with self.assertRaises(IpcError):
                    parse_json_bytes(document)

    def test_rejects_excessive_nesting_and_incomplete_jsonl(self):
        with self.assertRaises(IpcError):
            parse_json_bytes(("[" * 33 + "0" + "]" * 33).encode())
        with self.assertRaisesRegex(IpcError, "newline-terminated"):
            parse_json_line(b'{"schema_version":"1.0.0"}')

    def test_atomic_status_publication_retries_a_transient_windows_reader_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "status.json"
            real_replace = os.replace
            attempts = 0
            def flaky_replace(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("reader has status open")
                return real_replace(source, destination)
            with mock.patch.object(IPC_MODULE.os, "replace", side_effect=flaky_replace):
                atomic_write_json(target, {"schema_version": "1.0.0"})
            self.assertEqual(attempts, 2)
            self.assertEqual(read_json(target), {"schema_version": "1.0.0"})

    def test_control_and_status_reject_unknown_fields(self):
        control = {
            "schema_version": "1.0.0", "job_id": "job-" + "a" * 32,
            "job_spec": {"path": "job-spec.json", "sha256": "b" * 64},
            "runtime_id": "c" * 64,
            "worker": {"pid": 1, "creation_time": 1, "executable": "C:\\python.exe", "executable_sha256": "d" * 64, "nonce": "e" * 32},
            "event_schema_version": "1.0.0", "status_schema_version": "1.0.0",
            "cancel_request_path": "cancel.request",
            "target_scene": {"blend_path": "C:\\x.blend", "scene_uuid": "12345678-1234-1234-1234-123456789abc", "scene_name": "Scene"},
            "surprise": True,
        }
        with self.assertRaises(IpcError):
            validate_control(control)
        status = {
            "schema_version": "1.0.0", "job_id": "job-" + "a" * 32,
            "state": "running", "phase": "fixture", "heartbeat_sequence": 1,
            "heartbeat_utc": "2026-07-23T00:00:00+00:00", "worker_monotonic": 1.0,
            "progress_event_sequence": 1, "progress": {"completed": 0, "total": 1},
            "error": None, "surprise": True,
        }
        with self.assertRaises(IpcError):
            validate_status(status, status["job_id"])


class ProjectBindingTests(unittest.TestCase):
    def test_first_scene_uuid_requires_save_and_duplicates_are_rejected(self):
        scene = Scene()
        with self.assertRaisesRegex(JobLifecycleError, "assigned"):
            ensure_unique_scene_uuid(scene, (scene,))
        value = ensure_unique_scene_uuid(scene, (scene,))
        duplicate = Scene(scene)
        with self.assertRaisesRegex(JobLifecycleError, "duplicated"):
            ensure_unique_scene_uuid(scene, (scene, duplicate))
        self.assertEqual(value, scene["lingbot_map_scene_uuid"])

    def test_project_layout_is_bounded_and_sibling_to_blend(self):
        with tempfile.TemporaryDirectory() as temporary:
            blend = Path(temporary) / "target.blend"
            blend.touch()
            root = ensure_project_layout(blend)
            self.assertEqual(root, project_result_root(blend))
            self.assertEqual({item.name for item in root.iterdir()}, {".jobs", "results", "diagnostics"})

    def test_invalid_recovered_control_neither_moves_staging_nor_signals_a_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            blend = Path(temporary) / "target.blend"
            blend.touch()
            root = ensure_project_layout(blend)
            job_id = "job-" + "a" * 32
            job_dir = root / ".jobs" / job_id
            job_dir.mkdir()
            atomic_write_json(job_dir / "job-control.json", {"schema_version": "1.0.0", "pid": 4})
            controller = JobController()
            with mock.patch.object(JOB_LIFECYCLE_MODULE, "_terminate_exact") as terminate:
                with self.assertRaises(IpcError):
                    controller.recover_project(blend)
            terminate.assert_not_called()
            self.assertTrue(job_dir.is_dir())
            self.assertEqual(list((root / "diagnostics").iterdir()), [])

    def test_terminal_event_history_rejects_an_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_bytes(b'{"schema_version":"1.0.0"}')
            with self.assertRaisesRegex(IpcError, "newline-terminated"):
                JobController._validate_complete_events(path, "job-" + "a" * 32)

    def test_all_job_schema_contracts_begin_at_1_0_0(self):
        root = Path(__file__).resolve().parents[1]
        names = (
            "job-spec", "job-control", "job-event", "job-status",
            "preflight-result", "reconstruction-result",
        )
        for name in names:
            document = json.loads((root / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))
            self.assertEqual(document["properties"]["schema_version"]["const"], "1.0.0")
            self.assertFalse(document["additionalProperties"])

    def test_schema_catalog_hashes_every_identical_bundled_contract(self):
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / "schemas" / "catalog.json").read_text(encoding="utf-8"))
        bundled = json.loads((root / "blender_extension" / "runtime_bundle" / "schemas" / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog, bundled)
        for entry in catalog["contracts"]:
            source = root / "schemas" / entry["path"]
            runtime = root / "blender_extension" / "runtime_bundle" / "schemas" / entry["path"]
            self.assertEqual(source.read_bytes(), runtime.read_bytes())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), entry["sha256"])

    def test_launcher_uses_absolute_isolated_python_empty_cwd_and_allowlist(self):
        class FakeProcess:
            pid = 123
            stdout = io.BytesIO()
            def poll(self):
                return None
            def kill(self):
                pass
            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            empty = runtime / "empty-cwd"
            empty.mkdir(parents=True)
            python = runtime / "python.exe"
            python.touch()
            blend = root / "target.blend"
            blend.touch()
            captured = {}
            def popen(command, **kwargs):
                captured["command"] = command
                captured.update(kwargs)
                return FakeProcess()
            controller = JobController()
            controller._start_threads = lambda _active: None
            record = WorkerRecord(123, 456, str(python), "a" * 64, "b" * 32)
            with (
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_runtime_command", return_value=(runtime, python, "c" * 64, "d" * 64)),
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_wait_for_worker_record", return_value=record),
                mock.patch.object(JOB_LIFECYCLE_MODULE.subprocess, "Popen", side_effect=popen),
            ):
                controller.launch_fixture(
                    managed_root=root, blend_path=blend,
                    scene_uuid="12345678-1234-1234-1234-123456789abc",
                    scene_name="Scene", timeline_start=1,
                )
            self.assertTrue(Path(captured["command"][0]).is_absolute())
            self.assertEqual(captured["command"][1:3], ["-I", "-m"])
            self.assertEqual(captured["cwd"], empty)
            self.assertEqual(list(empty.iterdir()), [])
            environment = captured["env"]
            for forbidden in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "HTTP_PROXY", "HTTPS_PROXY"):
                self.assertNotIn(forbidden, environment)
            self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def test_capture_draft_is_relative_and_preflight_launch_freezes_both_identities(self):
        class FakeProcess:
            pid = 123
            stdout = io.BytesIO()
            def poll(self):
                return None
            def kill(self):
                pass
            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            (runtime / "empty-cwd").mkdir(parents=True)
            python = runtime / "python.exe"
            python.touch()
            blend = root / "target.blend"
            blend.touch()
            source = root / "capture.mp4"
            source.write_bytes(b"eight presentation frames")
            draft = capture_source_draft_path(source, blend)
            self.assertEqual(draft, "//capture.mp4")
            self.assertEqual(scene_relative_capture_path(source, blend), draft)
            captured = {}
            def popen(command, **kwargs):
                captured["command"] = command
                return FakeProcess()
            controller = JobController()
            controller._start_threads = lambda _active: None
            record = WorkerRecord(123, 456, str(python), "a" * 64, "b" * 32)
            with (
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_runtime_command", return_value=(runtime, python, "c" * 64, "d" * 64)),
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_wait_for_worker_record", return_value=record),
                mock.patch.object(JOB_LIFECYCLE_MODULE.subprocess, "Popen", side_effect=popen),
            ):
                job_id = controller.launch_preflight(
                    managed_root=root, blend_path=blend,
                    scene_uuid="12345678-1234-1234-1234-123456789abc",
                    scene_name="Scene", timeline_start=17,
                    capture_draft_path=draft,
                )
            spec_path = project_result_root(blend) / ".jobs" / job_id / "job-spec.json"
            frozen = read_json(spec_path)["capture_source"]
            self.assertEqual(frozen["draft_path"], draft)
            self.assertEqual(frozen["absolute_path"], str(source))
            self.assertEqual(frozen["scene_relative_path"], draft)
            self.assertEqual(frozen["size_bytes"], source.stat().st_size)
            self.assertEqual(captured["command"][4], "--preflight-job")
            source.write_bytes(b"replacement")
            self.assertEqual(read_json(spec_path)["capture_source"], frozen)

    def test_result_fixture_launch_freezes_profile_source_and_uses_dedicated_runner(self):
        class FakeProcess:
            pid = 123
            stdout = io.BytesIO()
            def poll(self):
                return None
            def kill(self):
                pass
            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            (runtime / "empty-cwd").mkdir(parents=True)
            python = runtime / "python.exe"
            python.touch()
            blend = root / "target.blend"
            blend.touch()
            source = root / "capture.mp4"
            source.write_bytes(b"result fixture identity")
            captured = {}
            def popen(command, **_kwargs):
                captured["command"] = command
                return FakeProcess()
            controller = JobController()
            controller._start_threads = lambda _active: None
            record = WorkerRecord(123, 456, str(python), "a" * 64, "b" * 32)
            with (
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_runtime_command", return_value=(runtime, python, "c" * 64, "d" * 64)),
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_wait_for_worker_record", return_value=record),
                mock.patch.object(JOB_LIFECYCLE_MODULE.subprocess, "Popen", side_effect=popen),
            ):
                job_id = controller.launch_result_fixture(
                    managed_root=root,
                    blend_path=blend,
                    scene_uuid="12345678-1234-1234-1234-123456789abc",
                    scene_name="Scene",
                    timeline_start=9,
                    capture_draft_path="//capture.mp4",
                    import_point_budget=17,
                )
            spec = read_json(project_result_root(blend) / ".jobs" / job_id / "job-spec.json")
            self.assertEqual(spec["result_fixture"]["size_bytes"], len(b"result fixture identity"))
            self.assertEqual(spec["result_fixture"]["import_point_budget"], 17)
            self.assertEqual(captured["command"][4], "--result-fixture-job")

    def test_reconstruction_launch_freezes_preflight_profile_gpu_model_and_cuda_uuid(self):
        class FakeProcess:
            pid = 123
            stdout = io.BytesIO()
            def poll(self): return None
            def kill(self): pass
            def wait(self, timeout=None): return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            (runtime / "empty-cwd").mkdir(parents=True)
            python = runtime / "python.exe"
            python.touch()
            blend = root / "target.blend"
            blend.touch()
            source = root / "capture.mp4"
            source.write_bytes(b"qualified source")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            preflight = {
                "source": {
                    "absolute_path": str(source), "size_bytes": source.stat().st_size,
                    "modification_time_ns": source.stat().st_mtime_ns, "sha256": source_hash,
                },
                "timing": {"frame_count": 321, "variable_frame_rate": True},
                "video": {
                    "stream_index": 0, "displayed_width": 1920, "displayed_height": 1080,
                    "display_transform": "identity",
                    "color": {"standard": "bt709", "range": "limited"},
                },
            }
            captured = {}
            def popen(command, **kwargs):
                captured["command"] = command
                captured.update(kwargs)
                return FakeProcess()
            controller = JobController()
            controller._start_threads = lambda _active: None
            record = WorkerRecord(123, 456, str(python), "a" * 64, "b" * 32)
            with (
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_runtime_command", return_value=(runtime, python, "c" * 64, "d" * 64)),
                mock.patch.object(JOB_LIFECYCLE_MODULE, "_wait_for_worker_record", return_value=record),
                mock.patch.object(JOB_LIFECYCLE_MODULE.subprocess, "Popen", side_effect=popen),
            ):
                job_id = controller.launch_reconstruction(
                    managed_root=root, blend_path=blend,
                    scene_uuid="12345678-1234-1234-1234-123456789abc",
                    scene_name="Scene", timeline_start=4,
                    capture_draft_path="//capture.mp4",
                    profile_name="Custom", camera_iterations=2,
                    confidence_cutoff_percent=42, depth_cutoff_percent=100,
                    import_point_budget=11_000_000, point_budget_confirmed=True,
                    gpu={
                        "uuid": "GPU-12345678", "name": "Ada", "total_memory": 24_000,
                        "driver_version": "999", "compute_capability": (8, 9),
                    },
                    capability_profile_name="Balanced",
                    capability_profile_settings_sha256="e" * 64,
                    model={
                        "catalog_version": "1.0.0", "id": "lingbot-map-long",
                        "path": str(root / "model.pt"), "sha256": "f" * 64,
                    },
                    preflight_result=preflight,
                )
            spec = read_json(project_result_root(blend) / ".jobs" / job_id / "job-spec.json")
            reconstruction = spec["reconstruction"]
            self.assertEqual(reconstruction["preflight"]["frame_count"], 321)
            self.assertEqual(reconstruction["profile"]["name"], "Custom")
            self.assertTrue(reconstruction["profile"]["point_budget_confirmed"])
            self.assertEqual(reconstruction["source"]["sha256"], source_hash)
            self.assertEqual(captured["command"][4], "--reconstruction-job")
            self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "GPU-12345678")

    def test_results_panel_discovery_only_returns_complete_matching_publications(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blend = root / "target.blend"
            blend.touch()
            results = ensure_project_layout(blend) / "results"
            directory = results / "20260723010203Z-11111111"
            directory.mkdir()
            arrays = {}
            names = {
                "positions", "colors", "confidence", "radius", "source_frame",
                "camera_to_world", "model_intrinsics", "source_intrinsics",
                "model_fov_radians", "source_pts_seconds", "source_to_model", "frame_type",
            }
            for name in names:
                path = directory / "arrays" / f"{name}.npy"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"npy")
                arrays[name] = {
                    "path": f"arrays/{name}.npy", "dtype": "|u1", "shape": [3],
                    "byte_length": 3, "sha256": "a" * 64,
                }
            log = directory / "logs" / "worker.log"
            log.parent.mkdir()
            log.write_bytes(b"")
            manifest = {
                "schema_version": "1.0.0", "result_id": "result-" + "2" * 32,
                "job_id": "job-" + "1" * 32, "created_utc": "2026-07-23T01:02:03+00:00",
                "target_scene": {"blend_path": str(blend), "scene_uuid": "12345678-1234-1234-1234-123456789abc", "scene_name": "Scene"},
                "timeline_start": 1, "source": {}, "contracts": {},
                "profile": {"name": "Fixture", "confidence_cutoff_percent": 50, "depth_cutoff_percent": 99.5, "import_point_budget": 8},
                "coordinate_system": {}, "counts": {"frames": 2, "points": 5},
                "arrays": arrays, "confidence_statistics": {}, "warnings": [], "provenance": {},
                "logs": [{"path": "logs/worker.log", "byte_length": 0, "sha256": "a" * 64}],
            }
            atomic_write_json(directory / "manifest.json", manifest)
            ready = discover_ready_results(
                blend, scene_uuid="12345678-1234-1234-1234-123456789abc"
            )
            self.assertEqual(len(ready), 1)
            self.assertEqual((ready[0].point_count, ready[0].frame_count), (5, 2))
            self.assertEqual(
                discover_ready_results(blend, scene_uuid="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                (),
            )
            arrays["positions"]["path"] = "../outside.npy"
            atomic_write_json(directory / "manifest.json", manifest)
            self.assertEqual(discover_ready_results(blend), ())


if __name__ == "__main__":
    unittest.main()
