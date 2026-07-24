"""Blender-selected localization and strictly local offline Help routing."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Mapping
from urllib.parse import urlparse

from .locale_catalogs import (
    ENGLISH,
    ENGLISH_LOCALE,
    TRADITIONAL_CHINESE,
    TRADITIONAL_CHINESE_LOCALE,
)


TRANSLATION_OWNER = __name__
_registered = False

MANUAL_TOPICS: Mapping[str, tuple[str, str]] = {
    "index": ("index.html", "overview"),
    "setup": ("installation-setup.html", "setup"),
    "capture": ("capture.html", "capture-guidance"),
    "profiles": ("profiles.html", "profiles"),
    "reconstruction": ("reconstruction.html", "reconstruction"),
    "active_job": ("reconstruction.html", "cancellation"),
    "import": ("import-visualization.html", "import"),
    "results": ("result-lifecycle.html", "results"),
    "diagnostics": ("privacy-diagnostics.html", "diagnostic-export"),
    "errors": ("errors-troubleshooting.html", "stable-error-codes"),
    "licenses": ("model-licensing.html", "model-licenses"),
}

STABLE_ERROR_CODES = frozenset(
    {
        "setup.runtime.failed",
        "setup.runtime.cancelled",
        "setup.model.checksum-mismatch",
        "setup.model.failed-model-publication",
        "launch.worker.failed",
        "launch.worker.bootstrap-failed",
        "pipeline.preflight.failed",
        "pipeline.preflight.cancelled",
        "pipeline.reconstruction.failed",
        "pipeline.reconstruction.cancelled",
        "import.transaction.failed",
        "import.transaction.cancelled",
        "import.transaction.rollback-failed",
        "lifecycle.plan.failed",
        "lifecycle.execute.failed",
        "diagnostic-export-linked-file",
        "diagnostic-export-race",
        "diagnostic-export-file-too-large",
        "diagnostic-export-report-too-large",
        "diagnostic-export-write-failed",
    }
)


class LocalizationError(RuntimeError):
    """A bounded presentation-asset failure."""


def normalize_locale(locale: object) -> str:
    """Map Blender locale variants to one of the two shipped catalogs."""

    value = str(locale or "").replace("-", "_")
    folded = value.casefold()
    if (
        folded.startswith("zh_hant")
        or folded.startswith("zh_tw")
        or folded.startswith("zh_hk")
        or folded.startswith("zh_mo")
    ):
        return TRADITIONAL_CHINESE_LOCALE
    return ENGLISH_LOCALE


def current_locale(bpy_module=None) -> str:
    """Read Blender's selected language without keeping another preference."""

    if bpy_module is None:
        try:
            import bpy as bpy_module
        except ImportError:
            return ENGLISH_LOCALE
    translations = getattr(
        getattr(bpy_module, "app", None),
        "translations",
        None,
    )
    return normalize_locale(getattr(translations, "locale", ENGLISH_LOCALE))


def translate(
    message: str,
    *,
    locale: object | None = None,
    bpy_module=None,
    **parameters,
) -> str:
    """Translate one canonical English template with English fallback."""

    selected = (
        normalize_locale(locale)
        if locale is not None
        else current_locale(bpy_module)
    )
    canonical = ENGLISH.get(message, message)
    translated = (
        TRADITIONAL_CHINESE.get(message, canonical)
        if selected == TRADITIONAL_CHINESE_LOCALE
        else canonical
    )
    try:
        return translated.format(**parameters)
    except (KeyError, IndexError, ValueError) as exc:
        raise LocalizationError(
            f"Localization placeholder mismatch for {message!r}"
        ) from exc


def present_error(
    error_code: str,
    detail: object,
    *,
    locale: object | None = None,
    bpy_module=None,
) -> str:
    """Keep the stable code invariant while localizing its presentation."""

    return translate(
        "Error {code}: {detail}",
        locale=locale,
        bpy_module=bpy_module,
        code=str(error_code),
        detail=str(detail),
    )


def blender_translations() -> dict[str, dict[tuple[str, str], str]]:
    """Build Blender's official add-on translation dictionary."""

    translated: dict[tuple[str, str], str] = {}
    for english, traditional_chinese in TRADITIONAL_CHINESE.items():
        # Python-defined labels normally use the default context, while
        # operator labels may request Blender's Operator context.
        translated[("*", english)] = traditional_chinese
        translated[("Operator", english)] = traditional_chinese
    # Blender's current Traditional Chinese locale is zh_HANT.  The zh_TW
    # alias also keeps older or externally supplied locale values usable.
    return {
        TRADITIONAL_CHINESE_LOCALE: translated,
        "zh_TW": dict(translated),
    }


def register_translations(bpy_module=None) -> None:
    global _registered
    if _registered:
        return
    if bpy_module is None:
        import bpy as bpy_module
    translations = getattr(
        getattr(bpy_module, "app", None),
        "translations",
        None,
    )
    register = getattr(translations, "register", None)
    if register is None:
        return
    register(TRANSLATION_OWNER, blender_translations())
    _registered = True


def unregister_translations(bpy_module=None) -> None:
    global _registered
    if not _registered:
        return
    if bpy_module is None:
        import bpy as bpy_module
    translations = getattr(
        getattr(bpy_module, "app", None),
        "translations",
        None,
    )
    unregister = getattr(translations, "unregister", None)
    if unregister is not None:
        try:
            unregister(TRANSLATION_OWNER)
        except RuntimeError:
            pass
    _registered = False


def _is_reparse(path: Path) -> bool:
    try:
        value = os.lstat(path)
    except OSError:
        return False
    return bool(
        getattr(value, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _ordinary_directory(path: Path, label: str) -> None:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise LocalizationError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse(path)
    ):
        raise LocalizationError(f"{label} must be an ordinary directory")


def _ordinary_file(path: Path, label: str) -> None:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise LocalizationError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse(path)
        or value.st_nlink != 1
        or value.st_size > 2 * 1024 * 1024
    ):
        raise LocalizationError(f"{label} must be one bounded ordinary file")


def resolve_manual_page(
    topic: str,
    *,
    locale: object | None = None,
    bpy_module=None,
    manual_root: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve a fixed packaged page and anchor with English fallback."""

    try:
        filename, anchor = MANUAL_TOPICS[topic]
    except KeyError as exc:
        raise LocalizationError("Unknown offline Help topic") from exc
    root = (
        Path(manual_root)
        if manual_root is not None
        else Path(__file__).resolve().parent / "manual"
    )
    _ordinary_directory(root, "Offline manual root")
    selected = (
        normalize_locale(locale)
        if locale is not None
        else current_locale(bpy_module)
    )
    candidates = (
        (selected, ENGLISH_LOCALE)
        if selected != ENGLISH_LOCALE
        else (ENGLISH_LOCALE,)
    )
    for candidate_locale in candidates:
        locale_root = root / candidate_locale
        try:
            _ordinary_directory(locale_root, "Offline manual locale")
            page = locale_root / filename
            _ordinary_file(page, "Offline manual page")
        except LocalizationError:
            continue
        return page, anchor
    raise LocalizationError(
        f"No packaged offline Help page is available for {topic}"
    )


def manual_uri(
    topic: str,
    *,
    locale: object | None = None,
    bpy_module=None,
    manual_root: str | Path | None = None,
) -> str:
    page, anchor = resolve_manual_page(
        topic,
        locale=locale,
        bpy_module=bpy_module,
        manual_root=manual_root,
    )
    uri = f"{page.as_uri()}#{anchor}"
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise LocalizationError("Offline Help did not resolve to a local file")
    return uri


def open_manual(topic: str, *, bpy_module=None):
    """Open a fixed local file URI through Blender; no remote URL is accepted."""

    if bpy_module is None:
        import bpy as bpy_module
    uri = manual_uri(topic, bpy_module=bpy_module)
    if not uri.startswith("file:"):
        raise LocalizationError("Offline Help requires a file URI")
    return bpy_module.ops.wm.url_open(url=uri)
