"""Real Windows proof for owned Result, Diagnostics, and Trash lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("blender_extension")
package.__path__ = [str(ROOT / "blender_extension")]
sys.modules["blender_extension"] = package

from blender_extension.job_lifecycle import JobController  # noqa: E402
from blender_extension.project_lifecycle import (  # noqa: E402
    PartialDeletionError,
    ProjectInventory,
    ProjectLifecycle,
)
from blender_extension.result_import import validate_result  # noqa: E402
from blender_extension.results import discover_ready_results  # noqa: E402


SCENE_UUID = "12345678-1234-4321-8765-123456789abc"


def _wait(controller: JobController, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if snapshot.state == "succeeded":
            return snapshot
        if snapshot.state in {
            "failed",
            "cancelled",
            "interrupted",
            "forced_termination",
            "protocol_error",
            "stale_identity",
        }:
            raise AssertionError(f"Fixture failed: {snapshot}")
        time.sleep(0.05)
    raise AssertionError(f"Fixture timed out: {controller.snapshot()}")


def main() -> int:
    if len(sys.argv) != 2 or os.name != "nt":
        raise SystemExit(
            "usage: project_lifecycle_windows_integration.py MANAGED_ROOT"
        )
    managed_root = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(
        prefix="lingbot-map-lifecycle17-"
    ) as temporary:
        workspace = Path(temporary)
        blend = workspace / "owned.blend"
        blend.touch()
        source = workspace / "capture.mp4"
        source.write_bytes(b"lifecycle fixture capture")
        controller = JobController()
        job_id = controller.launch_result_fixture(
            managed_root=managed_root,
            blend_path=blend,
            scene_uuid=SCENE_UUID,
            scene_name="Lifecycle Fixture Scene",
            timeline_start=7,
            capture_draft_path="//capture.mp4",
            confidence_cutoff_percent=50,
            depth_cutoff_percent=99.5,
            import_point_budget=8,
            initial_voxel_edge_length=0.01,
        )
        terminal = _wait(controller)
        diagnostic = Path(terminal.location)
        ready = discover_ready_results(blend, scene_uuid=SCENE_UUID)
        assert len(ready) == 1
        result = ready[0]
        lifecycle = ProjectLifecycle(blend)

        result_plan = lifecycle.plan(
            "trash_result", [result.directory.name]
        )
        assert result_plan.byte_count > 0
        trashed_result = lifecycle.execute(
            result_plan
        ).destinations[0]
        assert not result.directory.exists()
        assert validate_result(
            trashed_result, require_canonical_directory=False
        ).ready.result_id == result.result_id
        lifecycle.execute(
            lifecycle.plan("restore", [trashed_result.name])
        )
        assert validate_result(result.directory).ready.result_id == result.result_id

        diagnostic_plan = lifecycle.plan(
            "trash_diagnostics", [diagnostic.name]
        )
        trashed_diagnostic = lifecycle.execute(
            diagnostic_plan
        ).destinations[0]
        assert not diagnostic.exists()
        lifecycle.execute(
            lifecycle.plan("restore", [trashed_diagnostic.name])
        )
        assert diagnostic.is_dir()

        trashed_result = lifecycle.execute(
            lifecycle.plan("trash_result", [result.directory.name])
        ).destinations[0]
        delete_plan = lifecycle.plan("delete", [trashed_result.name])
        session = lifecycle.begin_delete(
            delete_plan, confirmation="DELETE"
        )
        assert session.step() is None
        try:
            session.cancel()
        except PartialDeletionError as exc:
            assert len(exc.partial_items) == 1
            partial_name = exc.partial_items[0]
        else:
            raise AssertionError("Interrupted deletion did not become partial")
        partial = trashed_result.parent / partial_name
        assert partial.is_dir()
        lifecycle.execute(
            lifecycle.plan("delete", [partial.name]),
            confirmation="DELETE",
        )
        assert not partial.exists()

        inventory = ProjectInventory(blend)
        deltas = []
        previous = 0
        while not inventory.snapshot().complete:
            snapshot = inventory.advance()
            deltas.append(snapshot.scanned_entries - previous)
            previous = snapshot.scanned_entries
        assert all(0 <= delta <= 50 for delta in deltas)
        assert not any(
            item.category == "trash" for item in snapshot.items
        )
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "result_id": result.result_id,
                    "result_trash_restore": True,
                    "diagnostic_trash_restore": True,
                    "partial_delete_retry": True,
                    "bounded_inventory": True,
                    "delete_items": delete_plan.item_count,
                    "delete_files": delete_plan.file_count,
                    "delete_bytes": delete_plan.byte_count,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
