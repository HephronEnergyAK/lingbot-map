from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package_name = "lingbot_map_source_view_unit"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "blender_extension")]
        sys.modules[package_name] = package
        cls.module = importlib.import_module(package_name + ".source_view")

    @staticmethod
    def manifest(transform="identity"):
        return {
            "source_display": {
                "width": 8,
                "height": 6,
                "display_transform": transform,
            },
            "model_coverage": {
                "coordinate_space": "source-display-pixel-edges",
                "polygon": [
                    [0.0, 1.0],
                    [8.0, 1.0],
                    [8.0, 5.0],
                    [0.0, 5.0],
                ],
                "source_fraction": 2.0 / 3.0,
                "model_width": 518,
                "model_height": 392,
            },
        }

    @staticmethod
    def source_to_model():
        return (
            (64.75, 0.0, 0.0),
            (0.0, 98.0, -98.0),
            (0.0, 0.0, 1.0),
        )

    def test_all_eight_display_transforms_have_exact_pixel_oracles(self):
        module = self.module
        expected = {
            "identity": (1, 0),
            "rotate_90_ccw": (0, 1),
            "rotate_180": (1, 2),
            "rotate_270_ccw": (2, 1),
            "reflect_x": (1, 0),
            "reflect_y": (1, 2),
            "reflect_main_diagonal": (0, 1),
            "reflect_anti_diagonal": (2, 1),
        }
        for transform, pixel in expected.items():
            with self.subTest(transform=transform):
                self.assertEqual(
                    module.transform_pixel(transform, 1, 0, 3, 3),
                    pixel,
                )
                contract = module.validate_source_view_contract(
                    self.manifest(transform), self.source_to_model()
                )
                self.assertIsNotNone(contract)
                self.assertIn(transform, module.BACKGROUND_MAPPINGS)
                expected_scale = 4.0 / 3.0 if transform in module.SWAPS_DIMENSIONS else 1.0
                self.assertAlmostEqual(
                    module.background_scale(contract), expected_scale
                )

    def test_contract_is_optional_as_a_pair_and_cross_validates_coverage(self):
        module = self.module
        self.assertIsNone(
            module.validate_source_view_contract({}, self.source_to_model())
        )
        document = self.manifest()
        contract = module.validate_source_view_contract(
            document, self.source_to_model()
        )
        self.assertEqual(contract.coverage_polygon[0], (0.0, 1.0))
        self.assertEqual(contract.coded_size, (8, 6))
        expected = (
            (10.0, 96.66666666666667),
            (110.0, 96.66666666666667),
            (110.0, 43.33333333333333),
            (10.0, 43.33333333333333),
        )
        actual = module.coverage_to_camera_border(
            contract, (10.0, 30.0, 110.0, 110.0)
        )
        for point, target in zip(actual, expected):
            self.assertAlmostEqual(point[0], target[0])
            self.assertAlmostEqual(point[1], target[1])

        missing = self.manifest()
        del missing["model_coverage"]
        with self.assertRaisesRegex(module.SourceViewError, "present together"):
            module.validate_source_view_contract(
                missing, self.source_to_model()
            )
        bad = self.manifest()
        bad["model_coverage"]["polygon"][2][1] = 4.0
        with self.assertRaisesRegex(module.SourceViewError, "source_to_model"):
            module.validate_source_view_contract(
                bad, self.source_to_model()
            )

    def test_scene_aspect_gate_is_exact_and_uses_pixel_aspect(self):
        module = self.module
        contract = module.validate_source_view_contract(
            self.manifest(), self.source_to_model()
        )
        scene = SimpleNamespace(
            render=SimpleNamespace(
                resolution_x=800,
                resolution_y=600,
                pixel_aspect_x=1.0,
                pixel_aspect_y=1.0,
            )
        )
        self.assertTrue(module.scene_aspect_matches(scene, contract))
        scene.render.resolution_x = 801
        self.assertFalse(module.scene_aspect_matches(scene, contract))
        scene.render.resolution_x = 400
        scene.render.pixel_aspect_x = 2.0
        self.assertTrue(module.scene_aspect_matches(scene, contract))

    def test_relink_accepts_checksum_not_recorded_mtime(self):
        module = self.module
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "capture.mp4"
            media.write_bytes(b"exact source bytes")
            source = {
                "absolute_path": str(root / "missing.mp4"),
                "scene_relative_path": "//capture.mp4",
                "size_bytes": media.stat().st_size,
                "modification_time_ns": 1,
                "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            }
            candidates = module.candidate_source_paths(
                source, current_blend_path=root / "scene.blend"
            )
            self.assertEqual(candidates[0], media)
            self.assertEqual(
                module.validate_source_media(media, source), media
            )
            media.write_bytes(b"different source")
            with self.assertRaisesRegex(module.SourceViewError, "checksum"):
                module.validate_source_media(media, source)


if __name__ == "__main__":
    unittest.main()
