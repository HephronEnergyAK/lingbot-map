"""Real frozen-Runtime and NVIDIA GPU qualification proof for issue #6."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "lingbot_gpu_capability_integration"
TERMINAL = {"succeeded", "cancelled", "blocked", "failed"}


def _load_modules(extension: Path):
    package = ModuleType(PACKAGE)
    package.__path__ = [str(extension)]
    sys.modules[PACKAGE] = package
    loaded = {}
    for short_name in ("runtime_setup", "model_store", "gpu_capability"):
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
    return loaded["gpu_capability"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default="")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--extension-root",
        type=Path,
        default=ROOT / "blender_extension",
    )
    parser.add_argument("--require-outside-checkout", action="store_true")
    arguments = parser.parse_args()
    extension_root = arguments.extension_root.resolve()
    if arguments.require_outside_checkout:
        try:
            extension_root.relative_to(ROOT)
        except ValueError:
            pass
        else:
            parser.error("installed Extension root must be outside the checkout")
    capability = _load_modules(extension_root)
    devices = capability.discover_physical_gpus(arguments.managed_root)
    gpu_uuid = capability.select_gpu_uuid(devices, arguments.gpu_uuid)
    controller = capability.CapabilityController()
    controller.start(arguments.managed_root, gpu_uuid)
    deadline = time.monotonic() + arguments.timeout
    prior = None
    while True:
        snapshot = controller.snapshot()
        current = (
            snapshot.state, snapshot.phase, snapshot.completed, snapshot.total,
            snapshot.message,
        )
        if current != prior:
            print("CAPABILITY_PROGRESS=" + json.dumps(current), flush=True)
            prior = current
        if snapshot.state in TERMINAL:
            break
        if time.monotonic() >= deadline:
            controller.cancel()
            raise TimeoutError("GPU capability integration timed out and requested exact cancellation")
        time.sleep(0.25)
    result = {
        "state": snapshot.state,
        "message": snapshot.message,
        "test_id": snapshot.test_id,
        "gpu_uuid": snapshot.gpu_uuid,
        "phase": snapshot.phase,
        "completed": snapshot.completed,
        "total": snapshot.total,
        "devices": [device.__dict__ for device in devices],
        "results": list(snapshot.results),
        "extension_root": str(extension_root),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if snapshot.state != "succeeded":
        return 1
    names = [item["identity"]["profile_name"] for item in snapshot.results]
    if names != list(capability.PROFILE_NAMES):
        raise RuntimeError(f"Capability profiles differ from fixed suite: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
