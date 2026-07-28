from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_localization.py"


def load_validator():
    name = "_lingbot_localization_validator_tests"
    spec = importlib.util.spec_from_file_location(name, VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load localization validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LocalizationCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = load_validator()
        cls.catalogs, cls.localization = cls.validator._load_modules(ROOT)

    def test_catalogs_and_complete_manual_pass_release_validation(self):
        summary = self.validator.validate(ROOT)
        self.assertEqual(summary["manual_chapters"], 10)
        self.assertEqual(summary["manual_pages"], 20)
        self.assertGreaterEqual(summary["catalog_messages"], 250)
        self.assertGreaterEqual(summary["required_ui_messages"], 200)

    def test_locale_aliases_follow_traditional_chinese_and_default_to_english(self):
        for locale in ("zh_HANT", "zh-TW", "zh_HK", "zh_MO"):
            with self.subTest(locale=locale):
                self.assertEqual(
                    self.localization.normalize_locale(locale),
                    "zh_HANT",
                )
                self.assertEqual(
                    self.localization.translate("Setup", locale=locale),
                    "設定",
                )
        for locale in ("en_US", "fr_FR", "", None):
            with self.subTest(locale=locale):
                self.assertEqual(
                    self.localization.translate("Setup", locale=locale),
                    "Setup",
                )

    def test_current_blender_locale_and_unknown_messages_fall_back_safely(self):
        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(
                translations=SimpleNamespace(locale="zh_TW")
            )
        )
        self.assertEqual(
            self.localization.current_locale(fake_bpy),
            "zh_HANT",
        )
        self.assertEqual(
            self.localization.translate(
                "Uncatalogued machine detail",
                bpy_module=fake_bpy,
            ),
            "Uncatalogued machine detail",
        )

    def test_stable_error_code_is_never_translated(self):
        rendered = self.localization.present_error(
            "pipeline.preflight.failed",
            "來源無效",
            locale="zh_HANT",
        )
        self.assertIn("pipeline.preflight.failed", rendered)
        self.assertIn("來源無效", rendered)

    def test_placeholder_signature_drift_is_rejected(self):
        key = "Error {code}: {detail}"
        original = self.catalogs.TRADITIONAL_CHINESE[key]
        self.catalogs.TRADITIONAL_CHINESE[key] = "錯誤 {code}"
        try:
            with self.assertRaisesRegex(
                self.validator.LocalizationValidationError,
                "Placeholder mismatch",
            ):
                self.validator.validate_catalogs(
                    ROOT,
                    self.catalogs,
                    self.localization,
                )
        finally:
            self.catalogs.TRADITIONAL_CHINESE[key] = original


class OfflineManualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = load_validator()
        _catalogs, cls.localization = cls.validator._load_modules(ROOT)

    def copy_manual(self, destination: Path) -> Path:
        manual = destination / "manual"
        shutil.copytree(ROOT / "blender_extension" / "manual", manual)
        return manual

    def test_every_help_topic_resolves_to_one_packaged_local_file(self):
        for locale in ("en_US", "zh_HANT"):
            for topic in self.localization.MANUAL_TOPICS:
                with self.subTest(locale=locale, topic=topic):
                    uri = self.localization.manual_uri(
                        topic,
                        locale=locale,
                    )
                    self.assertTrue(uri.startswith("file:"))
                    self.assertNotIn("http:", uri)
                    self.assertNotIn("https:", uri)

    def test_missing_translation_page_falls_back_to_english(self):
        with tempfile.TemporaryDirectory() as temporary:
            manual = self.copy_manual(Path(temporary))
            (manual / "zh_HANT" / "capture.html").unlink()
            page, anchor = self.localization.resolve_manual_page(
                "capture",
                locale="zh_HANT",
                manual_root=manual,
            )
            self.assertEqual(page.parent.name, "en_US")
            self.assertEqual(anchor, "capture-guidance")

    def test_unknown_topic_and_linked_manual_page_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            manual = self.copy_manual(Path(temporary))
            with self.assertRaisesRegex(
                self.localization.LocalizationError,
                "Unknown offline Help topic",
            ):
                self.localization.resolve_manual_page(
                    "remote",
                    manual_root=manual,
                )

            page = manual / "en_US" / "index.html"
            alternate = manual / "en_US" / "alternate.html"
            page.replace(alternate)
            os.link(alternate, page)
            with self.assertRaisesRegex(
                self.localization.LocalizationError,
                "No packaged offline Help page",
            ):
                self.localization.resolve_manual_page(
                    "index",
                    locale="en_US",
                    manual_root=manual,
                )

    def test_open_manual_passes_only_the_local_file_uri_to_blender(self):
        calls = []
        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(
                translations=SimpleNamespace(locale="zh_HANT")
            ),
            ops=SimpleNamespace(
                wm=SimpleNamespace(
                    url_open=lambda **kwargs: calls.append(kwargs)
                    or {"FINISHED"}
                )
            ),
        )
        self.assertEqual(
            self.localization.open_manual("setup", bpy_module=fake_bpy),
            {"FINISHED"},
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["url"].startswith("file:"))
        self.assertIn("/zh_HANT/installation-setup.html#setup", calls[0]["url"])


class LocalizationReleaseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = load_validator()

    def copied_extension_root(self, temporary: str) -> Path:
        root = Path(temporary)
        shutil.copytree(
            ROOT / "blender_extension",
            root / "blender_extension",
        )
        return root

    def test_manual_gate_rejects_missing_pages_remote_links_and_bilingual_drift(self):
        cases = (
            (
                "missing chapter",
                lambda root: (
                    root
                    / "blender_extension"
                    / "manual"
                    / "zh_HANT"
                    / "profiles.html"
                ).unlink(),
                "Missing manual chapter",
            ),
            (
                "remote link",
                lambda root: self.replace(
                    root
                    / "blender_extension"
                    / "manual"
                    / "en_US"
                    / "index.html",
                    'href="capture.html"',
                    'href="https://example.invalid/manual"',
                ),
                "Non-local manual link",
            ),
            (
                "remote image",
                lambda root: self.replace(
                    root
                    / "blender_extension"
                    / "manual"
                    / "en_US"
                    / "index.html",
                    "</main>",
                    '<img src="https://example.invalid/pixel.png"></main>',
                ),
                "Non-local manual link",
            ),
            (
                "bilingual section drift",
                lambda root: self.replace(
                    root
                    / "blender_extension"
                    / "manual"
                    / "zh_HANT"
                    / "capture.html",
                    'data-section-id="capture-guidance"',
                    'data-section-id="capture-guidance-drift"',
                ),
                "Bilingual section drift",
            ),
            (
                "invalid error code",
                lambda root: self.replace(
                    root
                    / "blender_extension"
                    / "manual"
                    / "en_US"
                    / "errors-troubleshooting.html",
                    "<code data-error-code=\"setup.runtime.failed\">",
                    "<code data-error-code=\"translated.invalid.code\">",
                ),
                "Invalid manual error-code reference",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = self.copied_extension_root(temporary)
                    mutate(root)
                    with self.assertRaisesRegex(
                        self.validator.LocalizationValidationError,
                        message,
                    ):
                        self.validator.validate(root)

    @staticmethod
    def replace(path: Path, old: str, new: str) -> None:
        text = path.read_text("utf-8")
        if old not in text:
            raise AssertionError(f"Fixture marker missing in {path}: {old}")
        path.write_text(text.replace(old, new, 1), "utf-8")


if __name__ == "__main__":
    unittest.main()
