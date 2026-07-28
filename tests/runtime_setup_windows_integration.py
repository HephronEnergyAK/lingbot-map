"""Real Windows/uv integration proof for issue #4.

Run explicitly because it downloads the two catalogued artifacts on first use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_runtime_setup(extension_root: Path):
    module_path = extension_root / "runtime_setup.py"
    spec = importlib.util.spec_from_file_location(
        "lingbot_runtime_setup_integration",
        module_path,
    )
    if spec is None:
        raise RuntimeError(f"Cannot load Runtime Setup from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-root", type=Path, required=True)
    parser.add_argument(
        "--extension-root",
        type=Path,
        default=ROOT / "blender_extension",
    )
    parser.add_argument("--require-outside-checkout", action="store_true")
    arguments = parser.parse_args()
    if os.name != "nt":
        parser.error("this integration proof is Windows-only")

    extension_root = arguments.extension_root.resolve()
    if arguments.require_outside_checkout:
        try:
            extension_root.relative_to(ROOT)
        except ValueError:
            pass
        else:
            parser.error("installed Extension root must be outside the checkout")
    runtime_setup = _load_runtime_setup(extension_root)
    bundle = runtime_setup.RuntimeBundle(extension_root / "runtime_bundle")
    installer = runtime_setup.RuntimeInstaller(arguments.managed_root, bundle)
    runtime_path = installer.setup(offline=False, online_access=True)

    poison = {
        "PATH": r"C:\external-python",
        "PYTHONPATH": r"C:\external-packages",
        "CONDA_PREFIX": r"C:\external-conda",
        "PIP_CONFIG_FILE": r"C:\external\pip.ini",
        "PIP_INDEX_URL": "https://invalid.example/simple",
        "UV_CONFIG_FILE": r"C:\external\uv.toml",
        "UV_INDEX": "https://invalid.example/simple",
        "HTTPS_PROXY": "http://127.0.0.1:1",
    }
    prior = {name: os.environ.get(name) for name in poison}
    os.environ.update(poison)
    try:
        offline_path = installer.setup(offline=True, online_access=False)
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    ready = json.loads((runtime_path / "READY.json").read_text(encoding="utf-8"))
    cache = {
        artifact.name: {
            "path": str(installer.artifact_store.path_for(artifact)),
            "verified": runtime_setup.verify_file(installer.artifact_store.path_for(artifact), artifact),
        }
        for artifact in bundle.artifacts
    }
    print(
        json.dumps(
            {
                "online_runtime": str(runtime_path),
                "offline_runtime": str(offline_path),
                "same_identity": runtime_path == offline_path,
                "extension_root": str(extension_root),
                "runtime_id": bundle.identity.runtime_id,
                "ready": ready,
                "artifacts": cache,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
