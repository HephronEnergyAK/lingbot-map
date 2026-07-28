"""Explicit full-model acquisition proof for the immutable Model Catalog."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "lingbot_model_store_integration"


def _load_modules(extension: Path):
    package = ModuleType(PACKAGE)
    package.__path__ = [str(extension)]
    sys.modules[PACKAGE] = package
    loaded = {}
    for short_name in ("runtime_setup", "model_store"):
        name = f"{PACKAGE}.{short_name}"
        spec = importlib.util.spec_from_file_location(
            name,
            extension / f"{short_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded[short_name] = module
    return loaded["runtime_setup"], loaded["model_store"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-root", type=Path, required=True)
    parser.add_argument(
        "--extension-root",
        type=Path,
        default=ROOT / "blender_extension",
    )
    parser.add_argument("--require-outside-checkout", action="store_true")
    parser.add_argument(
        "--local-source",
        type=Path,
        action="append",
        default=[],
    )
    arguments = parser.parse_args()
    extension_root = arguments.extension_root.resolve()
    if arguments.require_outside_checkout:
        try:
            extension_root.relative_to(ROOT)
        except ValueError:
            pass
        else:
            parser.error("installed Extension root must be outside the checkout")
    runtime_setup, model_store = _load_modules(extension_root)
    catalog = model_store.ModelCatalog(
        extension_root / "runtime_bundle" / "model-catalog.json"
    )
    store = model_store.ModelStore(arguments.managed_root, catalog)
    local_imports = []
    for source in arguments.local_source:
        imported_entry, imported_path = store.import_local(source)
        offline_imported = store.ensure_offline((imported_entry.id,))[0]
        local_imports.append(
            {
                "model_id": imported_entry.id,
                "path": str(imported_path),
                "offline_path": str(offline_imported),
                "same_path": imported_path == offline_imported,
                "length": imported_path.stat().st_size,
                "sha256": model_store.sha256_file(imported_path),
            }
        )
    entries = tuple(item for item in catalog.entries if item.role == "reconstruction")
    if len(entries) != 1:
        raise RuntimeError("expected exactly one Reconstruction Model")
    entry = entries[0]
    last_report = 0.0

    def progress(completed: int, total: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 5 or completed == total:
            print(f"MODEL_PROGRESS={completed}/{total}", flush=True)
            last_report = now

    path = store.acquire(
        entry.id,
        offline=False,
        online_access=True,
        cancellation=runtime_setup.CancellationToken(),
        progress=progress,
    )
    offline = store.ensure_offline((entry.id,))[0]
    print(
        json.dumps(
            {
                "model_id": entry.id,
                "path": str(path),
                "offline_path": str(offline),
                "same_path": path == offline,
                "extension_root": str(extension_root),
                "length": path.stat().st_size,
                "sha256": model_store.sha256_file(path),
                "local_imports": local_imports,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
