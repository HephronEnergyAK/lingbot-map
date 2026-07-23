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
EXTENSION = ROOT / "blender_extension"
PACKAGE = "lingbot_model_store_integration"


def _load_modules():
    package = ModuleType(PACKAGE)
    package.__path__ = [str(EXTENSION)]
    sys.modules[PACKAGE] = package
    loaded = {}
    for short_name in ("runtime_setup", "model_store"):
        name = f"{PACKAGE}.{short_name}"
        spec = importlib.util.spec_from_file_location(name, EXTENSION / f"{short_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded[short_name] = module
    return loaded["runtime_setup"], loaded["model_store"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-root", type=Path, required=True)
    arguments = parser.parse_args()
    runtime_setup, model_store = _load_modules()
    catalog = model_store.ModelCatalog(EXTENSION / "runtime_bundle" / "model-catalog.json")
    entries = tuple(item for item in catalog.entries if item.role == "reconstruction")
    if len(entries) != 1:
        raise RuntimeError("expected exactly one Reconstruction Model")
    entry = entries[0]
    store = model_store.ModelStore(arguments.managed_root, catalog)
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
                "length": path.stat().st_size,
                "sha256": model_store.sha256_file(path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
