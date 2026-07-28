from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMPORTER_PATH = ROOT / "blender_extension" / "result_import.py"


class ResultImportBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package_name = "lingbot_map_result_import_unit"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "blender_extension")]
        sys.modules[package_name] = package
        cls.module = importlib.import_module(package_name + ".result_import")

    def test_module_loads_without_numpy_or_blender_until_explicit_import(self):
        before_numpy = sys.modules.get("numpy")
        module = self.module
        self.assertIsNone(module.np)
        self.assertIs(sys.modules.get("numpy"), before_numpy)

    def test_importer_never_uses_global_undo_save_or_scene_global_assignments(self):
        source = IMPORTER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(IMPORTER_PATH))
        forbidden_calls = []
        forbidden_assignments = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.startswith("bpy.ops.") or name.startswith("bpy_module.ops."):
                    forbidden_calls.append(name)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    text = ast.unparse(target)
                    if any(
                        marker in text
                        for marker in (
                            ".render.engine",
                            ".render.fps",
                            ".view_settings",
                            ".camera",
                        )
                    ):
                        forbidden_assignments.append(text)
        self.assertEqual(forbidden_calls, [])
        self.assertEqual(forbidden_assignments, [])

    def test_capacity_model_uses_actual_points_headroom_and_four_gib_reserve(self):
        module = self.module
        point_count = 1_234_567
        estimate = (
            module.IMPORT_FIXED_BYTES
            + point_count * module.IMPORT_BYTES_PER_POINT
        )
        required = (estimate * 5 + 3) // 4 + 4 * 1024**3
        capacity = module.evaluate_import_capacity(
            point_count, available_probe=lambda: required
        )
        self.assertEqual(capacity.estimated_peak_bytes, estimate)
        self.assertEqual(capacity.required_available_bytes, required)
        self.assertTrue(capacity.allowed)
        with self.assertRaises(module.ImportCapacityError):
            module.evaluate_import_capacity(
                point_count, available_probe=lambda: required - 1
            )

    def test_every_interruptible_phase_is_named_and_commit_is_last(self):
        module = self.module
        self.assertEqual(module.IMPORT_PHASES[-1], "commit")
        self.assertEqual(len(module.IMPORT_PHASES), len(set(module.IMPORT_PHASES)))
        self.assertTrue(
            {
                "validation",
                "capacity",
                "positions",
                "color",
                "confidence",
                "radius",
                "source_frame",
                "material",
                "geometry_nodes",
                "ownership",
                "commit",
            }.issubset(module.IMPORT_PHASES)
        )


if __name__ == "__main__":
    unittest.main()
