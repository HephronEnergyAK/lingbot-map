"""Run in Blender 5.2 to verify native locale selection and local Help."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import bpy


MODULE_NAME = "bl_ext.user_default.lingbot_map_reconstruction"


def main():
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    extension = importlib.import_module(MODULE_NAME)
    localization = importlib.import_module(
        f"{MODULE_NAME}.localization"
    )
    assert extension.get_host_decision() is not None
    bpy.ops.lingbot_map.open_offline_help.get_rna_type()

    view = bpy.context.preferences.view
    previous_language = view.language
    previous_interface = view.use_translate_interface
    try:
        view.use_translate_interface = True
        view.language = "zh_HANT"
        selected_locale = localization.current_locale(bpy)
        assert selected_locale == "zh_HANT", (
            bpy.app.translations.locale,
            selected_locale,
        )
        translated_setup = bpy.app.translations.pgettext_iface("Setup")
        translated_help = bpy.app.translations.pgettext_iface(
            "Open Offline Help"
        )
        assert translated_setup == "設定", translated_setup
        assert translated_help == "開啟離線說明", translated_help

        resolved = {}
        for topic, (_filename, expected_anchor) in (
            localization.MANUAL_TOPICS.items()
        ):
            uri = localization.manual_uri(topic, bpy_module=bpy)
            parsed = urlparse(uri)
            page = Path(unquote(parsed.path.lstrip("/")))
            assert parsed.scheme == "file", uri
            assert parsed.netloc in {"", "localhost"}, uri
            assert "/zh_HANT/" in uri.replace("\\", "/"), uri
            assert parsed.fragment == expected_anchor, (
                topic,
                uri,
            )
            assert page.is_file(), page
            resolved[topic] = {
                "page": page.name,
                "anchor": parsed.fragment,
            }

        assert localization.translate(
            "Setup",
            locale="fr_FR",
        ) == "Setup"
        try:
            localization.manual_uri(
                "unknown-topic",
                bpy_module=bpy,
            )
        except localization.LocalizationError:
            pass
        else:
            raise AssertionError("Unknown Help topic did not fail closed")
    finally:
        view.language = previous_language
        view.use_translate_interface = previous_interface

    print(
        "LINGBOT_MAP_BLENDER_LOCALIZATION_SMOKE="
        + json.dumps(
            {
                "blender_version": bpy.app.version_string,
                "locale": selected_locale,
                "setup": translated_setup,
                "help": translated_help,
                "topics": resolved,
                "network_navigation": False,
                "help_operator": "registered",
                "english_fallback": "passed",
                "unknown_topic": "rejected",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
