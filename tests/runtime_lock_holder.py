"""Child helper for the cross-process Runtime lock test."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lingbot_runtime_setup_lock_holder", ROOT / "blender_extension" / "runtime_setup.py"
)
runtime_setup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime_setup
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime_setup)


def main() -> int:
    lock_path, ready_path, release_path = map(Path, sys.argv[1:4])
    with runtime_setup.RuntimeLock(lock_path, runtime_setup.current_process_identity()):
        ready_path.write_text("ready", encoding="ascii")
        deadline = time.monotonic() + 30
        while not release_path.exists():
            if time.monotonic() >= deadline:
                return 2
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
