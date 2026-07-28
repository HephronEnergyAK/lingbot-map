"""Release-gating validation for Extension localization and offline manuals."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import string
import sys
import types
from urllib.parse import unquote, urlparse


class LocalizationValidationError(RuntimeError):
    pass


def _load_modules(root: Path):
    package_root = root / "blender_extension"
    package = types.ModuleType("_lingbot_validation_extension")
    package.__path__ = [str(package_root)]
    sys.modules[package.__name__] = package

    def load(name: str):
        qualified = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified,
            package_root / f"{name}.py",
        )
        if spec is None or spec.loader is None:
            raise LocalizationValidationError(
                f"Could not load {name}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        return module

    catalogs = load("locale_catalogs")
    localization = load("localization")
    return catalogs, localization


def _placeholder_signature(template: str) -> tuple[tuple[str, str, str], ...]:
    result = []
    for _literal, field_name, format_spec, conversion in string.Formatter().parse(
        template
    ):
        if field_name is not None:
            if not field_name or any(
                marker in field_name for marker in (".", "[", "]")
            ):
                raise LocalizationValidationError(
                    f"Unsafe localization placeholder: {template!r}"
                )
            result.append(
                (field_name, conversion or "", format_spec or "")
            )
    return tuple(result)


def _literal_strings(value: ast.AST):
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        yield value.value
    elif isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        for child in value.elts:
            yield from _literal_strings(child)
    elif isinstance(value, ast.IfExp):
        yield from _literal_strings(value.body)
        yield from _literal_strings(value.orelse)


def required_ui_messages(root: Path) -> set[str]:
    required: set[str] = set()
    for relative in (
        "blender_extension/__init__.py",
        "blender_extension/ui.py",
    ):
        path = root / relative
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                names = [
                    target.id
                    for target in targets
                    if isinstance(target, ast.Name)
                ]
                if any(
                    name in {"bl_label", "bl_description"}
                    for name in names
                ):
                    required.update(_literal_strings(node.value))
            if not isinstance(node, ast.Call):
                continue
            function = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else ""
                )
            )
            keyword_names = (
                {"name", "description"}
                if function.endswith("Property")
                else (
                    {"text"}
                    if function in {"label", "operator", "prop"}
                    else set()
                )
            )
            for keyword in node.keywords:
                if keyword.arg not in keyword_names:
                    continue
                if isinstance(
                    keyword.value,
                    (ast.JoinedStr, ast.BinOp, ast.IfExp),
                ):
                    raise LocalizationValidationError(
                        f"Dynamic {function}.{keyword.arg} at "
                        f"{relative}:{node.lineno} must use translate()"
                    )
                required.update(_literal_strings(keyword.value))
            if function == "EnumProperty":
                for keyword in node.keywords:
                    if keyword.arg != "items":
                        continue
                    if not isinstance(
                        keyword.value,
                        (ast.Tuple, ast.List),
                    ):
                        raise LocalizationValidationError(
                            "Release EnumProperty items must be static"
                        )
                    for item in keyword.value.elts:
                        if not isinstance(item, (ast.Tuple, ast.List)):
                            continue
                        for value in item.elts[1:3]:
                            required.update(_literal_strings(value))
            if function == "report" and len(node.args) > 1:
                required.update(_literal_strings(node.args[1]))
            if function in {
                "set_setup_status",
                "set_import_status",
                "set_lifecycle_status",
                "tr",
            } and node.args:
                required.update(_literal_strings(node.args[0]))
    return required


def validate_catalogs(root: Path, catalogs, localization) -> dict[str, int]:
    english = catalogs.ENGLISH
    traditional = catalogs.TRADITIONAL_CHINESE
    if catalogs.CATALOG_SCHEMA_VERSION != "1.0.0":
        raise LocalizationValidationError(
            "Locale catalog schema must be 1.0.0"
        )
    if set(english) != set(traditional):
        raise LocalizationValidationError(
            "English and Traditional Chinese catalog keys drifted"
        )
    if any(key != value for key, value in english.items()):
        raise LocalizationValidationError(
            "English catalog is not canonical"
        )
    for message in sorted(english):
        if _placeholder_signature(english[message]) != _placeholder_signature(
            traditional[message]
        ):
            raise LocalizationValidationError(
                f"Placeholder mismatch for {message!r}"
            )
    required = required_ui_messages(root)
    missing = sorted(required - set(traditional))
    if missing:
        raise LocalizationValidationError(
            "Missing Traditional Chinese UI messages: "
            + repr(missing)
        )
    invariant_display = {"GPU UUID"}
    untranslated = sorted(
        message
        for message in required
        if traditional[message] == english[message]
        and message not in invariant_display
    )
    if untranslated:
        raise LocalizationValidationError(
            "Required UI messages remain untranslated: "
            + repr(untranslated)
        )
    machine_codes = set(localization.STABLE_ERROR_CODES)
    if machine_codes & set(traditional):
        raise LocalizationValidationError(
            "Stable error codes must not be catalog message IDs"
        )
    forbidden_imports = []
    for path in sorted((root / "worker").rglob("*.py")) + sorted(
        (root / "lingbot_map").rglob("*.py")
    ):
        text = path.read_text("utf-8")
        if "localization" in text or "locale_catalogs" in text:
            forbidden_imports.append(str(path.relative_to(root)))
    for path in sorted((root / "blender_extension").glob("*.py")):
        if path.name in {
            "__init__.py",
            "ui.py",
            "localization.py",
            "locale_catalogs.py",
        }:
            continue
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module or ""
            ).endswith(("localization", "locale_catalogs")):
                forbidden_imports.append(
                    str(path.relative_to(root))
                )
    if forbidden_imports:
        raise LocalizationValidationError(
            "Machine-contract modules import presentation localization: "
            + repr(forbidden_imports)
        )
    return {
        "catalog_messages": len(english),
        "required_ui_messages": len(required),
    }


@dataclass
class ParsedManualPage:
    lang: str = ""
    locale: str = ""
    chapter_id: str = ""
    sections: list[str] = field(default_factory=list)
    anchors: set[str] = field(default_factory=set)
    links: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    forbidden_elements: list[str] = field(default_factory=list)
    forbidden_attributes: list[str] = field(default_factory=list)


class _ManualParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.page = ParsedManualPage()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "html":
            self.page.lang = attributes.get("lang", "")
            self.page.locale = attributes.get("data-locale", "")
        if tag == "body":
            self.page.chapter_id = attributes.get(
                "data-chapter-id",
                "",
            )
        if "id" in attributes:
            self.page.anchors.add(attributes["id"])
        if tag == "section" and "data-section-id" in attributes:
            self.page.sections.append(attributes["data-section-id"])
        for attribute in (
            "href",
            "src",
            "action",
            "poster",
            "data",
            "formaction",
        ):
            link = attributes.get(attribute, "")
            if link:
                self.page.links.append(link)
        for attribute, value in attributes.items():
            if (
                attribute.startswith("on")
                or attribute == "srcset"
                or (
                    attribute == "style"
                    and "url(" in str(value or "").casefold()
                )
            ):
                self.page.forbidden_attributes.append(attribute)
        if (
            tag == "meta"
            and attributes.get("http-equiv", "").casefold() == "refresh"
        ):
            self.page.forbidden_elements.append("meta-refresh")
        if "data-error-code" in attributes:
            self.page.error_codes.append(
                attributes["data-error-code"]
            )
        if tag in {
            "script",
            "iframe",
            "object",
            "embed",
            "audio",
            "video",
            "base",
            "form",
        }:
            self.page.forbidden_elements.append(tag)


def _strict_json(path: Path):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LocalizationValidationError(
                    f"Duplicate JSON key in {path}: {key}"
                )
            result[key] = value
        return result

    return json.loads(
        path.read_text("utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            LocalizationValidationError(
                f"Non-finite JSON value in {path}: {value}"
            )
        ),
    )


def _manual_page(path: Path) -> ParsedManualPage:
    data = path.read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise LocalizationValidationError(
            f"Manual page is oversized: {path}"
        )
    parser = _ManualParser()
    try:
        parser.feed(data.decode("utf-8", errors="strict"))
        parser.close()
    except (UnicodeError, ValueError) as exc:
        raise LocalizationValidationError(
            f"Malformed manual page: {path}"
        ) from exc
    if parser.page.forbidden_elements:
        raise LocalizationValidationError(
            f"Active or embedded content in {path}: "
            + repr(parser.page.forbidden_elements)
        )
    if parser.page.forbidden_attributes:
        raise LocalizationValidationError(
            f"Active or remote-capable attributes in {path}: "
            + repr(parser.page.forbidden_attributes)
        )
    return parser.page


def _resolve_local_link(
    manual_root: Path,
    source: Path,
    link: str,
) -> tuple[Path, str]:
    parsed = urlparse(link)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.path.startswith(("/", "\\"))
    ):
        raise LocalizationValidationError(
            f"Non-local manual link in {source}: {link}"
        )
    path_text = unquote(parsed.path)
    target = source if not path_text else source.parent / path_text
    candidate = target.resolve()
    root = manual_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LocalizationValidationError(
            f"Manual link escaped package root in {source}: {link}"
        ) from exc
    if not candidate.is_file():
        raise LocalizationValidationError(
            f"Broken manual link in {source}: {link}"
        )
    return candidate, parsed.fragment


def validate_manual(root: Path, localization) -> dict[str, int]:
    manual_root = root / "blender_extension" / "manual"
    manifest = _strict_json(manual_root / "manifest.json")
    if set(manifest) != {"schema_version", "locales", "chapters"}:
        raise LocalizationValidationError(
            "Offline manual manifest fields are invalid"
        )
    if manifest["schema_version"] != "1.0.0":
        raise LocalizationValidationError(
            "Offline manual schema must be 1.0.0"
        )
    locales = manifest["locales"]
    if locales != ["en_US", "zh_HANT"]:
        raise LocalizationValidationError(
            "Offline manual locales must be en_US and zh_HANT"
        )
    chapters = manifest["chapters"]
    ids = [chapter.get("id") for chapter in chapters]
    files = [chapter.get("file") for chapter in chapters]
    if (
        len(chapters) != 10
        or len(set(ids)) != len(ids)
        or len(set(files)) != len(files)
    ):
        raise LocalizationValidationError(
            "Offline manual chapter inventory is invalid"
        )
    parsed_pages: dict[Path, ParsedManualPage] = {}
    stylesheet = (manual_root / "style.css").read_text("utf-8")
    if "@import" in stylesheet.casefold() or "url(" in stylesheet.casefold():
        raise LocalizationValidationError(
            "Offline manual stylesheet may not load external assets"
        )
    for chapter in chapters:
        if set(chapter) != {"id", "file", "sections"}:
            raise LocalizationValidationError(
                f"Manual chapter fields are invalid: {chapter!r}"
            )
        expected_sections = chapter["sections"]
        if (
            Path(chapter["file"]).name != chapter["file"]
            or not chapter["file"].endswith(".html")
        ):
            raise LocalizationValidationError(
                f"Manual chapter filename is invalid: {chapter['file']!r}"
            )
        if (
            not expected_sections
            or len(expected_sections) != len(set(expected_sections))
        ):
            raise LocalizationValidationError(
                f"Manual section inventory is invalid: {chapter['id']}"
            )
        error_reference_sets = []
        for locale in locales:
            path = manual_root / locale / chapter["file"]
            if not path.is_file():
                raise LocalizationValidationError(
                    f"Missing manual chapter: {locale}/{chapter['file']}"
                )
            page = _manual_page(path)
            parsed_pages[path.resolve()] = page
            if page.locale != locale:
                raise LocalizationValidationError(
                    f"Manual locale marker drifted: {path}"
                )
            expected_lang = "en" if locale == "en_US" else "zh-Hant"
            if page.lang != expected_lang:
                raise LocalizationValidationError(
                    f"Manual HTML language drifted: {path}"
                )
            if page.chapter_id != chapter["id"]:
                raise LocalizationValidationError(
                    f"Manual chapter ID drifted: {path}"
                )
            if page.sections != expected_sections:
                raise LocalizationValidationError(
                    f"Bilingual section drift in {path}: "
                    f"{page.sections!r} != {expected_sections!r}"
                )
            if not set(expected_sections) <= page.anchors:
                raise LocalizationValidationError(
                    f"Manual section anchors are incomplete: {path}"
                )
            invalid_codes = sorted(
                set(page.error_codes)
                - set(localization.STABLE_ERROR_CODES)
            )
            if invalid_codes:
                raise LocalizationValidationError(
                    f"Invalid manual error-code reference in {path}: "
                    + repr(invalid_codes)
                )
            error_reference_sets.append(tuple(page.error_codes))
        if error_reference_sets[0] != error_reference_sets[1]:
            raise LocalizationValidationError(
                f"Bilingual error-code references drifted: {chapter['id']}"
            )
    for source, page in list(parsed_pages.items()):
        for link in page.links:
            target, fragment = _resolve_local_link(
                manual_root,
                source,
                link,
            )
            if fragment:
                target_page = parsed_pages.get(target)
                if target_page is None:
                    target_page = _manual_page(target)
                    parsed_pages[target] = target_page
                if fragment not in target_page.anchors:
                    raise LocalizationValidationError(
                        f"Broken manual anchor in {source}: {link}"
                    )
    chapter_by_file = {
        chapter["file"]: chapter for chapter in chapters
    }
    for topic, (filename, anchor) in localization.MANUAL_TOPICS.items():
        chapter = chapter_by_file.get(filename)
        if chapter is None or anchor not in chapter["sections"]:
            raise LocalizationValidationError(
                f"Help topic does not resolve locally: {topic}"
            )
    disclosures = {
        "en_US": {
            "Dynamic Content",
            "Continuous Take",
            "Edited footage",
            "Fisheye",
            "Loop Closure",
            "Windows 11 x64",
            "Blender 5.2 LTS",
            "Cycles",
            "Ada",
            "weight-specific",
        },
        "zh_HANT": {
            "Dynamic Content",
            "Continuous Take",
            "剪輯過的影片",
            "fisheye",
            "Loop Closure",
            "Windows 11 x64",
            "Blender 5.2 LTS",
            "Cycles",
            "Ada",
            "權重專用",
        },
    }
    for locale in locales:
        combined = "\n".join(
            (manual_root / locale / chapter["file"]).read_text("utf-8")
            for chapter in chapters
        )
        missing = sorted(
            marker
            for marker in disclosures[locale]
            if marker.casefold() not in combined.casefold()
        )
        if missing:
            raise LocalizationValidationError(
                f"Required release disclosures missing in {locale}: "
                + repr(missing)
            )
    return {
        "manual_chapters": len(chapters),
        "manual_pages": len(chapters) * len(locales),
        "manual_links": sum(
            len(page.links) for page in parsed_pages.values()
        ),
    }


def validate(root: str | Path) -> dict[str, int]:
    repository = Path(root).resolve()
    catalogs, localization = _load_modules(repository)
    summary = {
        **validate_catalogs(repository, catalogs, localization),
        **validate_manual(repository, localization),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args(argv)
    summary = validate(arguments.root)
    print(
        "LINGBOT_MAP_LOCALIZATION_VALIDATION="
        + json.dumps(summary, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
