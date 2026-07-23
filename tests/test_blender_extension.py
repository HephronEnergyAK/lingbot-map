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


def install_fake_bpy(version=(5, 2, 1)):
    registration = FakeRegistration()
    bpy = ModuleType("bpy")
    bpy.app = SimpleNamespace(version=version, online_access=False)
    bpy.types = SimpleNamespace(
        AddonPreferences=type("AddonPreferences", (), {}),
        Panel=type("Panel", (), {}),
    )
    bpy.utils = registration

    props = ModuleType("bpy.props")
    props.BoolProperty = lambda **kwargs: kwargs
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

        self.extension.unregister()
        self.assertEqual(self.registration.unregistered, list(reversed(expected)))
        self.assertIsNone(self.extension.get_host_decision())

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

    def test_lifecycle_panels_are_ordered_and_machine_settings_stay_in_preferences(self):
        classes = self.extension.CLASSES
        preferences = classes[0]
        panels = classes[1:]

        self.assertEqual(preferences.__name__, "LINGBOTMAP_Preferences")
        self.assertEqual([panel.bl_order for panel in panels], [0, 1, 2, 3, 4])
        self.assertEqual(
            [panel.bl_label for panel in panels],
            ["Setup", "Reconstruct", "Active Job", "Results", "Diagnostics"],
        )
        self.assertTrue(all(panel.bl_space_type == "VIEW_3D" for panel in panels))
        self.assertTrue(all(panel.bl_region_type == "UI" for panel in panels))
        self.assertTrue(all(panel.bl_category == "LingBot Map" for panel in panels))
        annotations = preferences.__annotations__
        self.assertEqual(set(annotations), {"runtime_root", "gpu_uuid", "offline_setup"})


if __name__ == "__main__":
    unittest.main()
