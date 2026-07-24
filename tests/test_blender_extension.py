from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
import tempfile
import tomllib
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = ROOT / "blender_extension"


class FakeRegistration:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register_class(self, extension_class):
        self.registered.append(extension_class)

    def unregister_class(self, extension_class):
        self.unregistered.append(extension_class)


class FakeLayout:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.enabled = True

    def label(self, *, text, **_kwargs):
        self.events.append(("label", text))

    def operator(self, operator_id, *, text=None, **_kwargs):
        self.events.append(("operator", operator_id, text))
        return SimpleNamespace()

    def prop(self, owner, property_name, **_kwargs):
        self.events.append(("prop", property_name, getattr(owner, property_name, None)))

    def separator(self):
        self.events.append(("separator",))

    def box(self):
        return FakeLayout(self.events)

    def row(self, **_kwargs):
        return FakeLayout(self.events)


def install_fake_bpy(version=(5, 2, 1)):
    registration = FakeRegistration()
    class FakeFileImportMenu:
        callbacks = []

        @classmethod
        def append(cls, callback):
            cls.callbacks.append(callback)

        @classmethod
        def remove(cls, callback):
            cls.callbacks.remove(callback)

    bpy = ModuleType("bpy")
    bpy.app = SimpleNamespace(version=version, online_access=False)
    bpy.types = SimpleNamespace(
        AddonPreferences=type("AddonPreferences", (), {}),
        Operator=type("Operator", (), {}),
        Panel=type("Panel", (), {}),
        Scene=type("Scene", (), {}),
        TOPBAR_MT_file_import=FakeFileImportMenu,
    )
    bpy.utils = registration

    props = ModuleType("bpy.props")
    props.BoolProperty = lambda **kwargs: kwargs
    props.EnumProperty = lambda **kwargs: kwargs
    props.FloatProperty = lambda **kwargs: kwargs
    props.IntProperty = lambda **kwargs: kwargs
    props.StringProperty = lambda **kwargs: kwargs
    bpy.props = props
    sys.modules["bpy"] = bpy
    sys.modules["bpy.props"] = props
    return bpy, registration


def clear_extension_modules():
    for name in list(sys.modules):
        if name == "blender_extension" or name.startswith("blender_extension."):
            del sys.modules[name]


class BlenderExtensionManifestTests(unittest.TestCase):
    def test_manifest_has_permanent_identity_platform_and_only_three_permissions(self):
        manifest = tomllib.loads(
            (EXTENSION_ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["id"], "lingbot_map_reconstruction")
        self.assertEqual(manifest["version"], "0.1.0")
        self.assertEqual(manifest["name"], "LingBot Map Reconstruction")
        self.assertEqual(
            manifest["tagline"],
            "Reconstruct colored point clouds and animated cameras from video",
        )
        self.assertEqual(manifest["maintainer"], "LingBot Map Contributors")
        self.assertEqual(manifest["blender_version_min"], "5.2.0")
        self.assertEqual(manifest["blender_version_max"], "5.3.0")
        self.assertEqual(manifest["platforms"], ["windows-x64"])
        self.assertEqual(set(manifest["permissions"]), {"files", "network", "clipboard"})
        self.assertTrue(
            all(
                reason.isascii() and len(reason) <= 64 and not reason.endswith(".")
                for reason in manifest["permissions"].values()
            )
        )
        self.assertEqual(manifest["license"], ["SPDX:GPL-3.0-or-later"])

    def test_extension_zip_has_manifest_and_module_at_archive_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "lingbot_map_reconstruction-0.1.0.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(EXTENSION_ROOT.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts:
                        archive.write(path, path.relative_to(EXTENSION_ROOT).as_posix())

            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                self.assertIn("blender_manifest.toml", names)
                self.assertIn("__init__.py", names)
                self.assertIn("LICENSE.txt", names)
                self.assertNotIn("lingbot_map/model_adapter.py", names)
                self.assertFalse(any(name.endswith((".pt", ".pth", ".onnx")) for name in names))

    def test_extension_never_imports_worker_or_model_core(self):
        forbidden = []
        for path in EXTENSION_ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    if name == "worker" or name.startswith(("worker.", "lingbot_map")):
                        forbidden.append((path.name, name))
        self.assertEqual(forbidden, [])


class SupportedHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_extension_modules()
        install_fake_bpy()
        cls.host = importlib.import_module("blender_extension.host")

    @classmethod
    def tearDownClass(cls):
        clear_extension_modules()
        sys.modules.pop("bpy.props", None)
        sys.modules.pop("bpy", None)

    def facts(self, **overrides):
        values = {
            "operating_system": "Windows",
            "operating_system_release": "11",
            "native_architecture": "AMD64",
            "pointer_bits": 64,
            "blender_version": (5, 2, 1),
            "windows_build": 26100,
            "windows_product_type": 1,
            "compatibility_layer": None,
        }
        values.update(overrides)
        return self.host.HostFacts(**values)

    def test_accepts_windows_11_x64_workstation_and_blender_5_2(self):
        decision = self.host.evaluate_supported_host(self.facts())
        self.assertTrue(decision.supported)
        self.assertEqual(decision.code, "supported")

    def test_rejects_every_unsupported_host_dimension(self):
        cases = (
            ({"operating_system": "Linux"}, "unsupported_os"),
            ({"compatibility_layer": "Wine"}, "unsupported_compatibility_layer"),
            ({"native_architecture": "ARM64"}, "unsupported_architecture"),
            ({"pointer_bits": 32}, "unsupported_architecture"),
            ({"windows_product_type": 3}, "unsupported_windows_edition"),
            ({"operating_system_release": "10"}, "unsupported_windows_version"),
            ({"windows_build": 19045}, "unsupported_windows_version"),
            ({"blender_version": (5, 1, 9)}, "unsupported_blender_version"),
            ({"blender_version": (5, 3, 0)}, "unsupported_blender_version"),
        )
        for overrides, expected_code in cases:
            with self.subTest(overrides=overrides):
                decision = self.host.evaluate_supported_host(self.facts(**overrides))
                self.assertFalse(decision.supported)
                self.assertEqual(decision.code, expected_code)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        clear_extension_modules()
        self.bpy, self.registration = install_fake_bpy()
        self.extension = importlib.import_module("blender_extension")
        self.host = importlib.import_module("blender_extension.host")
        self.ui = importlib.import_module("blender_extension.ui")
        self.gpu_capability = importlib.import_module("blender_extension.gpu_capability")

    def tearDown(self):
        try:
            self.extension.unregister()
        except Exception:
            pass
        clear_extension_modules()
        sys.modules.pop("bpy.props", None)
        sys.modules.pop("bpy", None)

    def supported_decision(self, supported=True):
        facts = self.host.HostFacts(
            operating_system="Windows",
            operating_system_release="11",
            native_architecture="AMD64",
            pointer_bits=64,
            blender_version=(5, 2, 1),
            windows_build=26100,
            windows_product_type=1,
        )
        return self.host.HostDecision(
            supported=supported,
            code="supported" if supported else "unsupported_os",
            message="test decision",
            facts=facts,
        )

    def test_clean_enable_disable_and_reload(self):
        with mock.patch.object(
            self.extension, "probe_supported_host", return_value=self.supported_decision()
        ):
            self.extension.register()
        expected = list(self.extension.CLASSES)
        self.assertEqual(self.registration.registered, expected)
        self.assertTrue(self.extension.get_host_decision().supported)
        self.assertEqual(
            len(self.bpy.types.TOPBAR_MT_file_import.callbacks), 1
        )

        self.extension.unregister()
        self.assertEqual(self.registration.unregistered, list(reversed(expected)))
        self.assertIsNone(self.extension.get_host_decision())
        self.assertEqual(
            self.bpy.types.TOPBAR_MT_file_import.callbacks, []
        )

        self.extension = importlib.reload(self.extension)
        with mock.patch.object(
            self.extension, "probe_supported_host", return_value=self.supported_decision()
        ):
            self.extension.register()
        self.assertEqual(self.registration.registered[-len(expected) :], expected)

    def test_unsupported_host_registers_visible_shell_but_no_implicit_system_work(self):
        decision = self.supported_decision(supported=False)
        with (
            mock.patch.object(self.extension, "probe_supported_host", return_value=decision),
            mock.patch("os.mkdir", side_effect=AssertionError("storage mutation")),
            mock.patch("os.makedirs", side_effect=AssertionError("storage mutation")),
            mock.patch("socket.create_connection", side_effect=AssertionError("network access")),
        ):
            self.extension.register()

        self.assertEqual(len(self.registration.registered), len(self.extension.CLASSES))
        self.assertFalse(self.extension.get_host_decision().supported)

    def test_completed_job_refreshes_bounded_inventory_once_then_waits(self):
        with tempfile.TemporaryDirectory() as temporary:
            blend = Path(temporary) / "target.blend"
            blend.touch()
            self.bpy.data = SimpleNamespace(filepath=str(blend))
            snapshot = SimpleNamespace(
                state="succeeded",
                job_id="job-" + "a" * 32,
                target_blend=str(blend),
            )
            ready = SimpleNamespace(job_id=snapshot.job_id)
            with (
                mock.patch.object(
                    self.extension,
                    "get_job_snapshot",
                    return_value=snapshot,
                ),
                mock.patch.object(
                    self.extension,
                    "refresh_project_inventory",
                ) as refresh,
                mock.patch.object(
                    self.extension,
                    "ready_results_for_completed_job",
                    side_effect=[None, (ready,)],
                ),
                mock.patch.object(
                    self.extension,
                    "attempt_auto_import_once",
                    return_value=SimpleNamespace(),
                ) as attempt,
            ):
                self.extension._attempt_completed_result_auto_import()
                self.assertIsNone(
                    self.extension._last_completed_job_for_auto_import
                )
                self.extension._attempt_completed_result_auto_import()
            refresh.assert_called_once_with(str(blend))
            attempt.assert_called_once()
            self.assertEqual(
                self.extension._last_completed_job_for_auto_import,
                snapshot.job_id,
            )

    def test_completed_job_after_file_change_never_auto_imports_later(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = Path(temporary) / "current.blend"
            target = Path(temporary) / "target.blend"
            current.touch()
            target.touch()
            self.bpy.data = SimpleNamespace(filepath=str(current))
            snapshot = SimpleNamespace(
                state="succeeded",
                job_id="job-" + "b" * 32,
                target_blend=str(target),
            )
            with (
                mock.patch.object(
                    self.extension,
                    "get_job_snapshot",
                    return_value=snapshot,
                ),
                mock.patch.object(
                    self.extension,
                    "ready_results_for_completed_job",
                ) as ready,
                mock.patch.object(
                    self.extension, "attempt_auto_import_once"
                ) as attempt,
            ):
                self.extension._attempt_completed_result_auto_import()
            ready.assert_not_called()
            attempt.assert_not_called()
            self.assertEqual(
                self.extension._last_completed_job_for_auto_import,
                snapshot.job_id,
            )

    def test_lifecycle_panels_are_ordered_and_machine_settings_stay_in_preferences(self):
        classes = self.extension.CLASSES
        preferences = classes[0]
        operators = classes[1:-5]
        panels = classes[-5:]

        self.assertEqual(preferences.__name__, "LINGBOTMAP_Preferences")
        self.assertEqual(
            [operator.bl_idname for operator in operators],
            [
                "lingbot_map.setup_runtime",
                "lingbot_map.cancel_runtime_setup",
                "lingbot_map.download_model",
                "lingbot_map.import_model",
                "lingbot_map.cancel_model_setup",
                "lingbot_map.test_gpu_profiles",
                "lingbot_map.cancel_gpu_profiles",
                "lingbot_map.run_fixture_job",
                "lingbot_map.select_capture_source",
                "lingbot_map.run_preflight_job",
                "lingbot_map.run_reconstruction_job",
                "lingbot_map.cancel_active_job",
                "lingbot_map.import_result",
                "lingbot_map.repair_duplicate_scene_uuid",
                "lingbot_map.import_result_into",
                "lingbot_map.import_external_result",
                "lingbot_map.relink_result_reference",
                "lingbot_map.detach_result_copy",
                "lingbot_map.resolve_duplicate_imports",
                "lingbot_map.remove_result_version",
                "lingbot_map.use_reconstruction_camera",
                "lingbot_map.set_resolution_to_source",
                "lingbot_map.relink_source_background",
                "lingbot_map.source_background_visibility",
                "lingbot_map.toggle_model_coverage",
                "lingbot_map.refresh_project_inventory",
                "lingbot_map.load_more_project_items",
                "lingbot_map.trash_result",
                "lingbot_map.trash_dense",
                "lingbot_map.trash_diagnostic",
                "lingbot_map.export_diagnostic_report",
                "lingbot_map.copy_diagnostic_report",
                "lingbot_map.restore_trash",
                "lingbot_map.delete_trash",
            ],
        )
        self.assertEqual([panel.bl_order for panel in panels], [0, 1, 2, 3, 4])
        self.assertEqual(
            [panel.bl_label for panel in panels],
            [
                "Setup",
                "Reconstruct",
                "Active Job",
                "Results",
                "Diagnostics",
            ],
        )
        self.assertTrue(all(panel.bl_space_type == "VIEW_3D" for panel in panels))
        self.assertTrue(all(panel.bl_region_type == "UI" for panel in panels))
        self.assertTrue(all(panel.bl_category == "LingBot Map" for panel in panels))
        annotations = preferences.__annotations__
        self.assertEqual(set(annotations), {"runtime_root", "gpu_uuid", "offline_setup"})

    def test_copy_diagnostic_report_is_the_only_clipboard_write_and_is_redacted(self):
        source = (EXTENSION_ROOT / "ui.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(".clipboard ="), 1)
        self.assertIn(
            "class LINGBOTMAP_OT_copy_diagnostic_report",
            source,
        )
        operator_class = next(
            item
            for item in self.extension.CLASSES
            if item.__name__ == "LINGBOTMAP_OT_copy_diagnostic_report"
        )
        operator = operator_class()
        operator.diagnostic_name = "job-" + "a" * 32 + "--failed"
        operator.report = lambda *_args, **_kwargs: None
        window_manager = SimpleNamespace(clipboard="unchanged")
        report = SimpleNamespace(clipboard_text="redacted representation")
        with (
            mock.patch.object(
                self.ui,
                "_diagnostic_report_source",
                return_value=Path("diagnostic"),
            ),
            mock.patch.object(
                self.ui,
                "build_portable_report",
                return_value=report,
            ) as build,
        ):
            result = operator.execute(
                SimpleNamespace(window_manager=window_manager)
            )
        self.assertEqual(result, {"FINISHED"})
        self.assertEqual(
            window_manager.clipboard,
            "redacted representation",
        )
        self.assertTrue(build.call_args.kwargs["redact"])

    def test_failed_import_is_rolled_back_and_retained_with_stable_category(self):
        operator_class = next(
            item
            for item in self.extension.CLASSES
            if item.__name__ == "LINGBOTMAP_OT_import_result"
        )
        operator = operator_class()
        operator.result_directory = "result"
        operator.report = lambda *_args, **_kwargs: None
        scene = SimpleNamespace(name="Scene")
        with (
            mock.patch.object(
                self.ui,
                "import_result",
                side_effect=self.ui.ResultImportError(
                    "invalid result"
                ),
            ),
            mock.patch.object(
                self.ui,
                "_retain_ui_diagnostic",
            ) as retain,
            mock.patch.object(self.ui, "set_import_status"),
        ):
            result = operator.execute(
                SimpleNamespace(scene=scene)
            )
        self.assertEqual(result, {"CANCELLED"})
        self.assertEqual(
            retain.call_args.kwargs["error_code"],
            "import.transaction.failed",
        )
        self.assertEqual(
            retain.call_args.kwargs["category"],
            "import",
        )
        self.assertEqual(
            retain.call_args.kwargs["state"],
            "failed",
        )

    def test_unredacted_export_choice_resets_for_every_file_picker(self):
        operator_class = next(
            item
            for item in self.extension.CLASSES
            if item.__name__ == "LINGBOTMAP_OT_export_diagnostic_report"
        )
        operator = operator_class()
        operator.diagnostic_name = "job-" + "a" * 32 + "--failed"
        operator.include_sensitive_identity = True
        operator.report = lambda *_args, **_kwargs: None
        with tempfile.TemporaryDirectory() as temporary:
            blend = Path(temporary) / "target.blend"
            blend.touch()
            self.bpy.data = SimpleNamespace(filepath=str(blend))
            selected = []
            result = operator.invoke(
                SimpleNamespace(
                    window_manager=SimpleNamespace(
                        fileselect_add=selected.append
                    )
                ),
                None,
            )
        self.assertEqual(result, {"RUNNING_MODAL"})
        self.assertFalse(operator.include_sensitive_identity)
        self.assertEqual(selected, [operator])

    def test_setup_panel_shows_model_source_license_checksum_and_size_before_actions(self):
        with mock.patch.object(
            self.extension, "probe_supported_host", return_value=self.supported_decision()
        ):
            self.extension.register()
        setup_panel_class = next(
            item for item in self.extension.CLASSES if item.__name__ == "LINGBOTMAP_PT_setup"
        )
        panel = setup_panel_class()
        panel.layout = FakeLayout()
        preferences = SimpleNamespace(runtime_root="", gpu_uuid="", offline_setup=True)
        context = SimpleNamespace(
            preferences=SimpleNamespace(
                addons={"blender_extension": SimpleNamespace(preferences=preferences)}
            )
        )
        with mock.patch.dict("os.environ", {"LOCALAPPDATA": str(ROOT / ".test-local-app-data")}):
            panel.draw(context)
        labels = [event[1] for event in panel.layout.events if event[0] == "label"]
        operators = [event[1] for event in panel.layout.events if event[0] == "operator"]
        self.assertTrue(any("robbyant/lingbot-map" in label for label in labels))
        self.assertTrue(any("JianyuanWang/skyseg" in label for label in labels))
        self.assertTrue(any("Apache-2.0 (pending-weight-specific-confirmation)" in label for label in labels))
        self.assertTrue(any("MIT (pending-weight-specific-confirmation)" in label for label in labels))
        self.assertTrue(any("832bc82cbae0bc9b" in label for label in labels))
        self.assertTrue(any("ab9c34c64c3d8212" in label for label in labels))
        self.assertEqual(operators.count("lingbot_map.download_model"), 2)
        self.assertEqual(operators.count("lingbot_map.import_model"), 2)

    def test_reconstruct_panel_exposes_sky_mask_without_dynamic_content_claim(self):
        with mock.patch.object(
            self.extension, "probe_supported_host", return_value=self.supported_decision()
        ):
            self.extension.register()
        self.assertTrue(
            hasattr(self.bpy.types.Scene, "lingbot_map_sky_mask")
        )
        panel_class = next(
            item
            for item in self.extension.CLASSES
            if item.__name__ == "LINGBOTMAP_PT_reconstruct"
        )
        panel = panel_class()
        panel.layout = FakeLayout()
        scene = SimpleNamespace(
            lingbot_map_capture_source="//capture.mp4",
            lingbot_map_profile="Draft",
            lingbot_map_camera_iterations=1,
            lingbot_map_confidence_cutoff_percent=70.0,
            lingbot_map_depth_cutoff_percent=99.5,
            lingbot_map_import_point_budget=1_000_000,
            lingbot_map_retain_dense_predictions=False,
            lingbot_map_sky_mask=True,
        )
        panel.draw(SimpleNamespace(scene=scene))
        props = [event[1] for event in panel.layout.events if event[0] == "prop"]
        labels = [event[1] for event in panel.layout.events if event[0] == "label"]
        self.assertIn("lingbot_map_sky_mask", props)
        self.assertTrue(
            any("does not detect or remove Dynamic Content" in label for label in labels)
        )
        self.assertTrue(any("Moving people and vehicles" in label for label in labels))

    def test_results_draw_uses_snapshot_and_exposes_no_unrecognized_action(self):
        panel_class = next(
            item
            for item in self.extension.CLASSES
            if item.__name__ == "LINGBOTMAP_PT_results"
        )
        panel = panel_class()
        panel.layout = FakeLayout()

        class Scene(dict):
            collection = SimpleNamespace(children=())

            @staticmethod
            def as_pointer():
                return 7

        scene = Scene(
            lingbot_map_scene_uuid=(
                "12345678-1234-4321-8765-123456789abc"
            )
        )
        self.bpy.data = SimpleNamespace(
            filepath=r"C:\project\target.blend",
            scenes=(scene,),
        )
        snapshot = self.ui.InventorySnapshot(
            scanned_entries=1,
            complete=True,
            jobs_abnormal=False,
            items=(
                importlib.import_module(
                    "blender_extension.project_lifecycle"
                ).InventoryItem(
                    "results",
                    "malformed",
                    "unrecognized",
                    True,
                    detail="Malformed Result",
                ),
            ),
        )
        with (
            mock.patch.object(
                self.ui,
                "project_inventory_snapshot",
                return_value=snapshot,
            ),
            mock.patch(
                "os.scandir",
                side_effect=AssertionError("draw performed discovery"),
            ),
        ):
            panel.draw(SimpleNamespace(scene=scene))
        labels = [
            event[1]
            for event in panel.layout.events
            if event[0] == "label"
        ]
        operators = [
            event[1]
            for event in panel.layout.events
            if event[0] == "operator"
        ]
        self.assertTrue(
            any("Unrecognized: malformed" in label for label in labels)
        )
        self.assertNotIn("lingbot_map.trash_result", operators)
        self.assertNotIn("lingbot_map.import_result", operators)

    def test_gpu_test_persists_only_unambiguous_physical_uuid_before_nonmodal_start(self):
        with mock.patch.object(
            self.extension, "probe_supported_host", return_value=self.supported_decision()
        ):
            self.extension.register()
        operator_class = next(
            item for item in self.extension.CLASSES
            if item.__name__ == "LINGBOTMAP_OT_test_gpu_profiles"
        )
        operator = operator_class()
        preferences = SimpleNamespace(runtime_root="", gpu_uuid="", offline_setup=True)
        context = SimpleNamespace(
            preferences=SimpleNamespace(
                addons={"blender_extension": SimpleNamespace(preferences=preferences)}
            )
        )
        device = self.gpu_capability.GpuDevice(
            "GPU-12345678", "Test GPU", 16 * 1024**3, "610.47", (12, 0)
        )
        with (
            mock.patch.object(self.ui, "discover_physical_gpus", return_value=(device,)),
            mock.patch.object(self.ui, "start_gpu_capability") as start,
        ):
            self.assertEqual(operator.execute(context), {"FINISHED"})
        self.assertEqual(preferences.gpu_uuid, device.uuid)
        start.assert_called_once_with(mock.ANY, device.uuid)

    def test_import_into_confirmation_names_exact_original_and_actual_scenes(self):
        operator_class = next(
            item
            for item in self.extension.CLASSES
            if item.__name__ == "LINGBOTMAP_OT_import_result_into"
        )
        operator = operator_class()
        operator.target_scene_pointer = "42"
        operator.result_directory = r"C:\result"

        class Target(dict):
            name = "Chosen Target"

        target = Target(
            lingbot_map_scene_uuid="32345678-1234-4321-8765-123456789abc"
        )
        document = SimpleNamespace(
            manifest={
                "target_scene": {
                    "scene_name": "Original Scene",
                    "scene_uuid": "22345678-1234-4321-8765-123456789abc",
                }
            }
        )
        manager = SimpleNamespace(
            invoke_props_dialog=mock.Mock(return_value={"RUNNING_MODAL"})
        )
        with (
            mock.patch.object(self.ui, "_scene_from_pointer", return_value=target),
            mock.patch.object(self.ui, "validate_result", return_value=document),
        ):
            outcome = operator.invoke(
                SimpleNamespace(window_manager=manager), None
            )
        self.assertEqual(outcome, {"RUNNING_MODAL"})
        self.assertIn("Chosen Target", operator.confirmation_target)
        self.assertIn("Original Scene", operator.confirmation_original)
        manager.invoke_props_dialog.assert_called_once_with(
            operator, width=560
        )

    def test_failed_save_restores_scene_identity_save_gate(self):
        scene = {
            "lingbot_map_scene_uuid_save_required": True,
        }
        self.bpy.data = SimpleNamespace(scenes=(scene,))
        self.extension._clear_scene_identity_save_markers_before_save(None)
        self.assertNotIn(
            "lingbot_map_scene_uuid_save_required", scene
        )
        self.extension._restore_scene_identity_markers_after_failed_save(None)
        self.assertTrue(
            scene["lingbot_map_scene_uuid_save_required"]
        )
        self.extension._clear_scene_identity_save_markers_before_save(None)
        self.extension._finalize_scene_identity_markers_after_save(None)
        self.assertNotIn(
            "lingbot_map_scene_uuid_save_required", scene
        )

    def test_gpu_selection_rejects_ordinal_and_ambiguous_auto_selection(self):
        first = self.gpu_capability.GpuDevice(
            "GPU-12345678", "First", 1, "1", (1, 0)
        )
        second = self.gpu_capability.GpuDevice(
            "GPU-abcdefgh", "Second", 1, "1", (1, 0)
        )
        with self.assertRaisesRegex(self.gpu_capability.GpuCapabilityError, "physical UUID"):
            self.gpu_capability.select_gpu_uuid((first,), "0")
        with self.assertRaisesRegex(self.gpu_capability.GpuCapabilityError, "Multiple"):
            self.gpu_capability.select_gpu_uuid((first, second), "")

    def test_gpu_worker_environment_is_isolated_and_windows_getuser_safe(self):
        with mock.patch.dict(
            "os.environ",
            {
                "USERNAME": "poisoned-user",
                "PYTHONPATH": r"C:\poisoned",
                "CUDA_VISIBLE_DEVICES": "0",
                "LOCALAPPDATA": r"C:\local",
                "SystemRoot": r"C:\Windows",
            },
            clear=True,
        ):
            environment = self.gpu_capability._worker_environment()
        self.assertEqual(environment["USERNAME"], "LingBotMapWorker")
        self.assertEqual(environment["LOCALAPPDATA"], r"C:\local")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)

    def test_atomic_status_permission_race_is_transient_not_terminal(self):
        def racing_read(_path):
            try:
                raise PermissionError("atomic replace in progress")
            except PermissionError as exc:
                raise self.gpu_capability.GpuCapabilityError(
                    "Cannot parse GPU capability JSON"
                ) from exc

        with mock.patch.object(self.gpu_capability, "_read_json", side_effect=racing_read):
            with self.assertRaises(self.gpu_capability.TransientStatusRead):
                self.gpu_capability._status_snapshot(
                    Path("status.json"), "cap-test123", "GPU-12345678"
                )


if __name__ == "__main__":
    unittest.main()
