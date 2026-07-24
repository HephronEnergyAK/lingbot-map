"""Real installed-Runtime proof for Dense Predictions Trash and restore."""

from __future__ import annotations

import json
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("blender_extension")
package.__path__ = [str(ROOT / "blender_extension")]
sys.modules["blender_extension"] = package

from blender_extension.project_lifecycle import ProjectLifecycle  # noqa: E402
from blender_extension.result_import import validate_result  # noqa: E402
from blender_extension.results import discover_ready_results  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2 or os.name != "nt":
        raise SystemExit(
            "usage: project_lifecycle_dense_windows_integration.py RUNTIME"
        )
    runtime = Path(sys.argv[1])
    worker_python = runtime / ".venv" / "Scripts" / "python.exe"
    with tempfile.TemporaryDirectory(
        prefix="lingbot-map-dense-lifecycle17-"
    ) as temporary:
        workspace = Path(temporary)
        project = workspace / "target.lingbot-map"
        completed = subprocess.run(
            [
                str(worker_python),
                "-I",
                str(
                    ROOT
                    / "tests"
                    / "dense_predictions_windows_integration.py"
                ),
                "--worker",
                str(project),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        worker_evidence = json.loads(completed.stdout)
        (project / ".jobs").mkdir()
        blend = workspace / "target.blend"
        ready = discover_ready_results(blend)
        assert len(ready) == 1
        result = ready[0]
        assert result.dense_status == "available"
        document = validate_result(result.directory)
        assert document.ready.result_id == worker_evidence["result_id"]
        del document
        gc.collect()
        lifecycle = ProjectLifecycle(blend)
        plan = lifecycle.plan(
            "trash_dense", [result.directory.name]
        )
        assert plan.result_id == worker_evidence["result_id"]
        assert plan.chunk_count == 4
        trashed = lifecycle.execute(plan).destinations[0]
        assert not (result.directory / "dense").exists()
        unavailable = validate_result(result.directory)
        assert unavailable.ready.dense_status == "unavailable"
        del unavailable
        gc.collect()
        restore = lifecycle.plan("restore", [trashed.name])
        lifecycle.execute(restore)
        restored = validate_result(result.directory)
        assert restored.ready.dense_status == "available"
        restored_id = restored.ready.result_id
        del restored
        gc.collect()
        print(
            json.dumps(
                {
                    "result_id": restored_id,
                    "dense_trash_restore": True,
                    "chunks": plan.chunk_count,
                    "files": plan.file_count,
                    "bytes": plan.byte_count,
                    "core_usable_while_trashed": True,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
