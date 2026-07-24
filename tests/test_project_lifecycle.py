from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "lingbot_map_project_lifecycle_unit"


def _load_module():
    package = ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "blender_extension")]
    sys.modules[PACKAGE] = package
    return importlib.import_module(PACKAGE + ".project_lifecycle")


class ProjectLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.blend = self.base / "scene.blend"
        self.blend.write_bytes(b"blend")
        self.root = self.base / "scene.lingbot-map"
        for name in (".jobs", "results", "diagnostics"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.now = datetime(
            2026, 7, 23, 1, 2, 3, 456789, tzinfo=timezone.utc
        )
        self.lifecycle = self.module.ProjectLifecycle(
            self.blend, clock=lambda: self.now
        )
        self.result_id = "result-" + "1" * 32
        self.job_id = "job-" + "a" * 32
        self.result_name = "20260723010203Z-aaaaaaaa"

    def _document(self, *, target: Path | None = None, dense=None):
        manifest = {
            "result_id": self.result_id,
            "job_id": self.job_id,
            "created_utc": "2026-07-23T01:02:03Z",
            "target_scene": {
                "blend_path": str(target or self.blend),
                "scene_uuid": "scene",
                "scene_name": "Scene",
            },
        }
        if dense is not None:
            manifest["dense_predictions"] = dense
        return SimpleNamespace(
            ready=SimpleNamespace(
                result_id=self.result_id, job_id=self.job_id
            ),
            manifest=manifest,
        )

    def _result(self, *, dense=None) -> Path:
        result = self.root / "results" / self.result_name
        (result / "arrays").mkdir(parents=True)
        (result / "arrays" / "positions.npy").write_bytes(b"owned")
        manifest = self._document(dense=dense).manifest
        (result / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return result

    def _validator(self, document):
        return mock.patch.object(
            self.module,
            "validate_result",
            side_effect=lambda _path, **_kwargs: document,
        )

    def _dense(self, result: Path):
        dense = result / "dense"
        signals = {}
        for signal_name in ("depth", "depth_confidence"):
            signal_root = dense / signal_name
            signal_root.mkdir(parents=True)
            array = np.arange(12, dtype="<f4").reshape((2, 2, 3))
            chunk = signal_root / "00000000-00000001.npy"
            np.save(chunk, array, allow_pickle=False)
            payload = chunk.read_bytes()
            signals[signal_name] = {
                "dtype": "<f4",
                "shape": [2, 2, 3],
                "chunks": [
                    {
                        "path": (
                            f"{signal_name}/00000000-00000001.npy"
                        ),
                        "dtype": "<f4",
                        "shape": [2, 2, 3],
                        "frame_start": 0,
                        "frame_count": 2,
                        "byte_length": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        manifest = {
            "schema_version": "1.0.0",
            "component": "dense_predictions",
            "completion_state": "complete",
            "frame_count": 2,
            "model_grid": {"height": 2, "width": 3},
            "chunk_frame_limit": 64,
            "signals": signals,
            "provenance": {
                "job_id": self.job_id,
                "source_sha256": "2" * 64,
                "model_sha256": "3" * 64,
                "profile_settings_sha256": "4" * 64,
                "preprocessing_rule_version": "1.0.0",
                "alignment_rule_version": "1.0.0",
            },
        }
        raw = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        (dense / "manifest.json").write_bytes(raw)
        return {
            "schema_version": "1.0.0",
            "completion_state": "complete",
            "manifest_path": "dense/manifest.json",
            "manifest_byte_length": len(raw),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def test_inventory_is_direct_bounded_paged_and_non_authoritative(self):
        jobs = self.root / ".jobs"
        for index in range(451):
            job = jobs / f"job-{index:032x}"
            job.mkdir()
        nested = jobs / ("job-" + "f" * 32) / "never-scanned"
        nested.mkdir(parents=True)
        (nested / "payload").write_bytes(b"x")
        (jobs / "unrecognized").mkdir()
        (jobs / "linked").mkdir()
        result = self.root / "results" / self.result_name
        result.mkdir()
        (result / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "result_id": self.result_id,
                    "job_id": self.job_id,
                    "created_utc": "2026-07-23T01:02:03Z",
                    "target_scene": {
                        "blend_path": str(self.blend),
                        "scene_uuid": "scene",
                        "scene_name": "Scene",
                    },
                    "counts": {"frames": 2, "points": 3},
                    "profile": {"name": "Draft"},
                }
            ),
            encoding="utf-8",
        )
        (self.root / "results" / "malformed").mkdir()

        original_reparse = self.module.is_reparse_point

        def fake_reparse(path):
            return Path(path).name == "linked" or original_reparse(path)

        inventory = self.module.ProjectInventory(self.blend)
        deltas = []
        previous = 0
        with mock.patch.object(
            self.module, "is_reparse_point", side_effect=fake_reparse
        ):
            while not inventory.snapshot().complete:
                snapshot = inventory.advance()
                deltas.append(snapshot.scanned_entries - previous)
                previous = snapshot.scanned_entries

        self.assertTrue(all(0 <= delta <= 50 for delta in deltas))
        self.assertEqual(len(snapshot.page("jobs")), 200)
        self.assertTrue(snapshot.jobs_abnormal)
        self.assertEqual(snapshot.scanned_entries, 456)
        linked = next(item for item in snapshot.items if item.name == "linked")
        malformed = next(
            item for item in snapshot.items if item.name == "malformed"
        )
        recognized = next(
            item for item in snapshot.items if item.name == self.result_name
        )
        self.assertFalse(linked.ordinary)
        self.assertEqual(malformed.status, "unrecognized")
        self.assertEqual(recognized.status, "recognized")

    def test_result_trash_and_restore_preserve_exact_content(self):
        source = self._result()
        document = self._document()
        calls = []

        def validate(path, **kwargs):
            calls.append((Path(path).name, kwargs))
            return document

        with mock.patch.object(
            self.module, "validate_result", side_effect=validate
        ):
            plan = self.lifecycle.plan("trash_result", [self.result_name])
            before = plan.records
            outcome = self.lifecycle.execute(plan)
            trashed = outcome.destinations[0]
            self.assertFalse(source.exists())
            self.assertTrue(trashed.is_dir())
            restore = self.lifecycle.plan("restore", [trashed.name])
            self.lifecycle.execute(restore)

        self.assertTrue(source.is_dir())
        self.assertEqual(
            self.module._tree_records(source),
            before[0],
        )
        self.assertTrue(
            any(
                kwargs.get("require_canonical_directory") is False
                for _name, kwargs in calls
            )
        )

    def test_external_result_has_no_disk_authority(self):
        source = self._result()
        external = self._document(target=self.base / "other.blend")
        with self._validator(external):
            with self.assertRaisesRegex(
                self.module.ProjectLifecycleError,
                "External Result",
            ):
                self.lifecycle.plan("trash_result", [self.result_name])
        self.assertTrue(source.exists())

    def test_relative_target_binding_has_no_project_authority(self):
        source = self._result()
        relative = self._document(target=Path("target.blend"))
        with self._validator(relative):
            with self.assertRaisesRegex(
                self.module.ProjectLifecycleError,
                "not absolute",
            ):
                self.lifecycle.plan(
                    "trash_result", [self.result_name]
                )
        self.assertTrue(source.exists())

    def test_result_with_matching_staging_job_cannot_move(self):
        source = self._result()
        (self.root / ".jobs" / self.job_id).mkdir()
        with self._validator(self._document()):
            with self.assertRaisesRegex(
                self.module.ProjectLifecycleError,
                "active or staging",
            ):
                self.lifecycle.plan(
                    "trash_result", [self.result_name]
                )
        self.assertTrue(source.exists())

    def test_abnormal_job_inventory_disables_lifecycle_actions(self):
        self._result()
        for index in range(33):
            (self.root / ".jobs" / f"job-{index:032x}").mkdir()
        with self._validator(self._document()):
            with self.assertRaisesRegex(
                self.module.ProjectLifecycleError,
                "more than 32",
            ):
                self.lifecycle.plan(
                    "trash_result", [self.result_name]
                )

    def test_plan_detects_source_and_destination_replacement_races(self):
        source = self._result()
        with self._validator(self._document()):
            changed = self.lifecycle.plan(
                "trash_result", [self.result_name]
            )
            (source / "arrays" / "positions.npy").write_bytes(b"changed")
            with self.assertRaises(self.module.ProjectRaceError):
                self.lifecycle.execute(changed)
            self.assertTrue(source.exists())

            stable = self.lifecycle.plan(
                "trash_result", [self.result_name]
            )
            stable.destinations[0].mkdir(parents=True)
            marker = stable.destinations[0] / "do-not-overwrite"
            marker.write_bytes(b"foreign")
            with self.assertRaises(self.module.ProjectConflictError):
                self.lifecycle.execute(stable)
            self.assertEqual(marker.read_bytes(), b"foreign")
            self.assertTrue(source.exists())

    def test_result_change_during_complete_validation_is_rejected(self):
        source = self._result()

        def mutate(_path, **_kwargs):
            (source / "arrays" / "positions.npy").write_bytes(b"changed")
            return self._document()

        with mock.patch.object(
            self.module, "validate_result", side_effect=mutate
        ):
            with self.assertRaisesRegex(
                self.module.ProjectRaceError,
                "during complete validation",
            ):
                self.lifecycle.plan(
                    "trash_result", [self.result_name]
                )
        self.assertTrue(source.exists())

    def test_dense_trash_restore_and_conflict_are_exact(self):
        result = self._result()
        descriptor = self._dense(result)
        document = self._document(dense=descriptor)
        (result / "manifest.json").write_text(
            json.dumps(document.manifest), encoding="utf-8"
        )
        with self._validator(document):
            plan = self.lifecycle.plan(
                "trash_dense", [self.result_name]
            )
            self.assertEqual(plan.chunk_count, 2)
            trashed = self.lifecycle.execute(plan).destinations[0]
            self.assertFalse((result / "dense").exists())
            restore = self.lifecycle.plan("restore", [trashed.name])
            (result / "dense").mkdir()
            with self.assertRaises(self.module.ProjectConflictError):
                self.lifecycle.execute(restore)
            self.assertTrue(trashed.exists())
            (result / "dense").rmdir()
            self.lifecycle.execute(
                self.lifecycle.plan("restore", [trashed.name])
            )
        self.assertTrue((result / "dense" / "manifest.json").is_file())

    def test_dense_action_rechecks_owning_result_after_confirmation(self):
        result = self._result()
        descriptor = self._dense(result)
        document = self._document(dense=descriptor)
        (result / "manifest.json").write_text(
            json.dumps(document.manifest), encoding="utf-8"
        )
        with self._validator(document):
            plan = self.lifecycle.plan(
                "trash_dense", [self.result_name]
            )
            (result / "foreign").write_bytes(b"changed owner")
            with self.assertRaises(self.module.ProjectRaceError):
                self.lifecycle.execute(plan)
        self.assertTrue((result / "dense").is_dir())

    def test_multi_diagnostic_trash_and_cancellable_partial_delete(self):
        names = (
            self.job_id + "--failed",
            self.job_id + "--cancelled",
        )
        for name in names:
            directory = self.root / "diagnostics" / name
            directory.mkdir()
            (directory / "a.txt").write_bytes(b"a")
            (directory / "b.txt").write_bytes(b"bb")
        plan = self.lifecycle.plan("trash_diagnostics", names)
        self.assertEqual(plan.item_count, 2)
        trashed = self.lifecycle.execute(plan).destinations

        delete = self.lifecycle.plan(
            "delete", [item.name for item in trashed]
        )
        self.assertEqual(delete.file_count, 4)
        self.assertEqual(delete.byte_count, 6)
        with self.assertRaisesRegex(
            self.module.ProjectLifecycleError, "typing DELETE"
        ):
            self.lifecycle.execute(delete, confirmation="delete")

        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls == 2

        with self.assertRaises(self.module.PartialDeletionError) as raised:
            self.lifecycle.execute(
                delete, confirmation="DELETE", cancel=cancel
            )
        self.assertEqual(len(raised.exception.partial_items), 1)
        partial = raised.exception.partial_items[0]
        self.assertIn("--partial_delete", partial)
        self.assertTrue((self.root / ".trash" / partial).is_dir())
        retry = self.lifecycle.plan("delete", [partial])
        self.lifecycle.execute(retry, confirmation="DELETE")
        self.assertFalse((self.root / ".trash" / partial).exists())

    def test_terminal_diagnostic_action_refuses_matching_staging_job(self):
        name = self.job_id + "--interrupted"
        diagnostic = self.root / "diagnostics" / name
        diagnostic.mkdir()
        (self.root / ".jobs" / self.job_id).mkdir()
        inventory = self.module.ProjectInventory(self.blend)
        while not inventory.snapshot().complete:
            snapshot = inventory.advance()
        listed = next(
            item
            for item in snapshot.items
            if item.category == "diagnostics"
        )
        self.assertEqual(listed.status, "unrecognized")
        self.assertIn("active or staging", listed.detail)
        with self.assertRaisesRegex(
            self.module.ProjectLifecycleError, "active or staging"
        ):
            self.lifecycle.plan("trash_diagnostics", [name])
        self.assertTrue(diagnostic.exists())

    def test_diagnostic_restore_refuses_matching_staging_job(self):
        name = self.job_id + "--interrupted"
        diagnostic = self.root / "diagnostics" / name
        diagnostic.mkdir()
        trashed = self.lifecycle.execute(
            self.lifecycle.plan("trash_diagnostics", [name])
        ).destinations[0]
        (self.root / ".jobs" / self.job_id).mkdir()
        with self.assertRaisesRegex(
            self.module.ProjectConflictError,
            "active or staging",
        ):
            self.lifecycle.plan("restore", [trashed.name])
        self.assertTrue(trashed.exists())

    def test_diagnostic_trash_name_reserves_windows_partial_suffix_room(self):
        name = self.job_id + "--" + "x" * 158
        self.assertIsNotNone(
            self.module.DIAGNOSTIC_NAME.fullmatch(name)
        )
        trash_name = (
            f"{name}_diagnostic_{self.lifecycle._timestamp()}"
        )
        self.assertLessEqual(
            len(trash_name + "--partial_delete-" + "f" * 8),
            255,
        )

    def test_incremental_delete_session_remains_cancellable_between_files(self):
        trash = self.root / ".trash"
        item = trash / (
            f"{self.result_id}_result_20260723T010203456789Z"
        )
        item.mkdir(parents=True)
        (item / "a").write_bytes(b"a")
        (item / "b").write_bytes(b"b")
        plan = self.lifecycle.plan("delete", [item.name])
        session = self.lifecycle.begin_delete(
            plan, confirmation="DELETE"
        )
        self.assertIsNone(session.step(maximum_files=1))
        self.assertEqual(session.completed_files, 1)
        with self.assertRaises(self.module.PartialDeletionError):
            session.cancel()
        partial = next(trash.glob("*--partial_delete"))
        self.assertEqual({path.name for path in partial.iterdir()}, {"b"})

    def test_single_file_delete_finishes_without_cancellation_window(self):
        trash = self.root / ".trash"
        item = trash / (
            f"{self.result_id}_result_20260723T010203456789Z"
        )
        item.mkdir(parents=True)
        (item / "only").write_bytes(b"x")
        session = self.lifecycle.begin_delete(
            self.lifecycle.plan("delete", [item.name]),
            confirmation="DELETE",
        )
        outcome = session.step()
        self.assertEqual(outcome.item_count, 1)
        self.assertFalse(item.exists())
        session.cancel()

    def test_restore_never_overwrites_and_unsafe_names_never_escape(self):
        source = self._result()
        with self._validator(self._document()):
            trashed = self.lifecycle.execute(
                self.lifecycle.plan(
                    "trash_result", [self.result_name]
                )
            ).destinations[0]
            source.mkdir(parents=True)
            marker = source / "foreign"
            marker.write_bytes(b"keep")
            with self.assertRaises(self.module.ProjectConflictError):
                self.lifecycle.plan("restore", [trashed.name])
        self.assertEqual(marker.read_bytes(), b"keep")
        with self.assertRaises(self.module.ProjectLifecycleError):
            self.lifecycle.plan("delete", ["..\\outside"])

    def test_delete_rejects_reparse_content_before_confirmation(self):
        trash = self.root / ".trash"
        item = trash / (
            f"{self.result_id}_result_20260723T010203456789Z"
        )
        item.mkdir(parents=True)
        linked = item / "linked"
        linked.write_bytes(b"x")
        original_reparse = self.module.is_reparse_point

        def fake_reparse(path):
            return Path(path).name == "linked" or original_reparse(path)

        with mock.patch.object(
            self.module, "is_reparse_point", side_effect=fake_reparse
        ):
            with self.assertRaisesRegex(
                self.module.ProjectLifecycleError, "linked or reparse"
            ):
                self.lifecycle.plan("delete", [item.name])
        self.assertTrue(linked.exists())

    def test_unrecognized_trash_entry_is_visible_but_has_no_action(self):
        trash = self.root / ".trash"
        unknown = trash / "unknown"
        unknown.mkdir(parents=True)
        inventory = self.module.ProjectInventory(self.blend)
        while not inventory.snapshot().complete:
            snapshot = inventory.advance()
        item = next(
            candidate
            for candidate in snapshot.items
            if candidate.name == "unknown"
        )
        self.assertEqual(item.status, "unrecognized")
        with self.assertRaisesRegex(
            self.module.ProjectLifecycleError, "Unrecognized Trash"
        ):
            self.lifecycle.plan("delete", ["unknown"])
        self.assertTrue(unknown.exists())

    def test_delete_rechecks_ancestor_reparse_state_before_each_file(self):
        trash = self.root / ".trash"
        item = trash / (
            f"{self.result_id}_result_20260723T010203456789Z"
        )
        child = item / "child"
        child.mkdir(parents=True)
        payload = child / "payload"
        payload.write_bytes(b"keep")
        session = self.lifecycle.begin_delete(
            self.lifecycle.plan("delete", [item.name]),
            confirmation="DELETE",
        )
        original_reparse = self.module.is_reparse_point

        def replaced(path):
            return Path(path).name == "child" or original_reparse(path)

        with mock.patch.object(
            self.module, "is_reparse_point", side_effect=replaced
        ):
            with self.assertRaisesRegex(
                self.module.ProjectLifecycleError,
                "ordinary directory",
            ):
                session.step()
        self.assertEqual(payload.read_bytes(), b"keep")
        self.assertFalse(
            any(trash.glob("*--partial_delete"))
        )


if __name__ == "__main__":
    unittest.main()
