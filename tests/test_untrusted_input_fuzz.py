from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.run_untrusted_fuzz import (
    _minimize_failure,
    extension_ipc,
    run as run_fuzz,
)

from lingbot_map_worker import ipc as worker_ipc
from lingbot_map_worker import result_bundle as worker_result_bundle
from lingbot_map_worker.result_bundle import (
    ArrayContract,
    ResultBundleError,
    safe_relative_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _stat_with(value, **changes):
    fields = {
        name: getattr(value, name, 0)
        for name in (
            "st_mode",
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_file_attributes",
        )
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


class ParserPropertyAndFuzzTests(unittest.TestCase):
    def test_failure_minimizer_retains_a_one_minimal_reproducer(self):
        minimized = _minimize_failure(
            b"prefix-TRIGGER-suffix",
            lambda value: b"TRIGGER" in value,
        )
        self.assertEqual(minimized, b"TRIGGER")
        for index in range(len(minimized)):
            self.assertNotIn(
                b"TRIGGER",
                minimized[:index] + minimized[index + 1 :],
            )

    def test_reproducible_coverage_guided_json_and_npy_fuzz_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_fuzz(
                0x21C0DEC0,
                400,
                Path(temporary) / "failures",
            )
            self.assertEqual(result["failure_count"], 0)
            self.assertEqual(result["json"]["cases"], 400)
            self.assertGreater(result["json"]["rejected"], 0)
            self.assertGreater(result["json"]["coverage_lines"], 20)
            self.assertGreater(result["npy"]["rejected"], 0)
            self.assertLess(result["peak_traced_bytes"], 128 * 1024 * 1024)

    def test_ipc_readers_revalidate_handle_and_path_identity(self):
        for module in (extension_ipc, worker_ipc):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "status.json"
                path.write_text('{"value":1}', encoding="utf-8")
                before = path.lstat()
                changed = _stat_with(
                    before,
                    st_mtime_ns=before.st_mtime_ns + 1,
                )
                with mock.patch.object(
                    module.Path,
                    "lstat",
                    side_effect=(before, changed),
                ):
                    with self.assertRaisesRegex(
                        module.IpcError,
                        "changed while reading",
                    ):
                        module.read_json(path)

    def test_ipc_reparse_is_rejected_before_open(self):
        for module in (extension_ipc, worker_ipc):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "control.json"
                path.write_text("{}", encoding="utf-8")
                linked = _stat_with(
                    path.lstat(),
                    st_file_attributes=0x400,
                )
                with (
                    mock.patch.object(module.Path, "lstat", return_value=linked),
                    mock.patch.object(module.Path, "open") as opened,
                ):
                    with self.assertRaisesRegex(module.IpcError, "ordinary"):
                        module.read_json(path)
                    opened.assert_not_called()

    def test_result_paths_reject_traversal_absolute_and_linked_ancestors(self):
        unsafe = (
            "../outside",
            "/absolute",
            "C:/drive",
            "//server/share",
            r"arrays\positions.npy",
            "arrays/../positions.npy",
            "",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in unsafe:
                with self.subTest(value=value), self.assertRaises(ResultBundleError):
                    safe_relative_file(root, value)
            linked = root / "arrays"
            linked.mkdir()
            with mock.patch(
                "lingbot_map_worker.result_bundle.is_reparse_point",
                side_effect=lambda path: Path(path) == linked,
            ):
                with self.assertRaisesRegex(ResultBundleError, "reparse"):
                    safe_relative_file(root, "arrays/positions.npy")

    def test_worker_npy_replacement_at_load_seam_is_rejected(self):
        np = worker_result_bundle.np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "positions.npy"
            replacement = root / "replacement.npy"
            np.save(path, np.zeros((1, 3), dtype=np.dtype("<f4")))
            np.save(replacement, np.ones((1, 3), dtype=np.dtype("<f4")))
            _digest, _length, trusted_identity = (
                worker_result_bundle._sha256_file_identity(path)
            )
            original_load = np.load

            def replace_before_load(*args, **kwargs):
                os.replace(replacement, path)
                return original_load(path, **kwargs)

            with mock.patch.object(
                worker_result_bundle.np,
                "load",
                side_effect=replace_before_load,
            ):
                with self.assertRaisesRegex(
                    ResultBundleError,
                    "changed during consumption",
                ):
                    worker_result_bundle.validate_npy_file(
                        path,
                        ArrayContract("<f4", (1, 3)),
                        expected_identity=trusted_identity,
                    )

    def test_worker_unknown_result_major_never_dereferences_array_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary)
            (result / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "arrays": {
                            "positions": {"path": "../../outside.npy"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                worker_result_bundle,
                "safe_relative_file",
                side_effect=AssertionError("Result path dereferenced"),
            ):
                with self.assertRaisesRegex(
                    ResultBundleError,
                    "Incompatible Result",
                ):
                    worker_result_bundle.validate_result_bundle(result)
            self.assertTrue((result / "manifest.json").is_file())


@unittest.skipUnless(
    Path(sys.executable).name.lower().startswith("python"),
    "audit probes require the standalone Worker Python",
)
class WorkerAuditBoundaryTests(unittest.TestCase):
    def test_every_network_and_child_process_path_is_denied(self):
        cases = [
            "socket-connect",
            "socket-bind",
            "subprocess",
            "shell",
            "spawn",
            "multiprocessing",
        ]
        if os.name == "nt":
            cases.append("startfile")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case in cases:
                with self.subTest(case=case):
                    marker = root / f"{case}.marker"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "tests" / "worker_audit_probe.py"),
                            case,
                            str(marker),
                        ],
                        cwd=root,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        completed.stdout + completed.stderr,
                    )
                    record = json.loads(completed.stdout.splitlines()[-1])
                    self.assertEqual(record["state"], "denied")
                    self.assertFalse(record["marker_exists"])
                    self.assertFalse(marker.exists())


try:
    import bpy  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    bpy = None


@unittest.skipIf(bpy is None, "legacy Job state requires Blender-side modules")
class UnknownSchemaMajorTests(unittest.TestCase):
    def test_unknown_control_major_is_retained_without_identity_dereference(self):
        from blender_extension import job_lifecycle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blend = root / "scene.blend"
            blend.write_bytes(b"blend")
            project = root / "scene.lingbot-map"
            job_id = "job-" + "a" * 32
            job = project / ".jobs" / job_id
            job.mkdir(parents=True)
            (project / "results").mkdir()
            (project / "diagnostics").mkdir()
            (job / "job-control.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "worker": {
                            "pid": 4,
                            "executable": "C:/untrusted.exe",
                        },
                        "cancel_request_path": "../../signal",
                    }
                ),
                encoding="utf-8",
            )
            controller = job_lifecycle.JobController()
            with (
                mock.patch.object(
                    job_lifecycle,
                    "_record_from_control",
                    side_effect=AssertionError("identity dereferenced"),
                ),
                mock.patch.object(
                    job_lifecycle,
                    "_observed_record",
                    side_effect=AssertionError("process observed"),
                ),
                mock.patch.object(
                    job_lifecycle.os,
                    "kill",
                    side_effect=AssertionError("process signalled"),
                ),
            ):
                controller.recover_project(blend)
            snapshot = controller.snapshot()
            self.assertEqual(snapshot.state, "legacy_running")
            self.assertIn("Legacy Job Running", snapshot.message)
            self.assertTrue(job.is_dir())
            self.assertFalse(controller.has_active_job())
            self.assertFalse(controller.request_cancel())

    def test_unknown_result_major_is_retained_without_paths_or_datablocks(self):
        from blender_extension import result_import, results

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / ("20260724000000Z-" + "b" * 8)
            result.mkdir()
            manifest = {
                "schema_version": "2.0.0",
                "result_id": "result-" + "c" * 32,
                "job_id": "job-" + "d" * 32,
                "created_utc": "2026-07-24T00:00:00+00:00",
                "target_scene": {},
                "timeline_start": 1,
                "source": {},
                "contracts": {},
                "profile": {},
                "coordinate_system": {},
                "counts": {},
                "arrays": {
                    "positions": {
                        "path": "../../outside.npy",
                    }
                },
                "confidence_statistics": {},
                "warnings": [],
                "provenance": {},
                "logs": [{"path": "../../outside.log"}],
            }
            (result / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            before = {
                name: len(getattr(bpy.data, name))
                for name in (
                    "collections",
                    "objects",
                    "pointclouds",
                    "cameras",
                    "materials",
                    "curves",
                    "node_groups",
                    "actions",
                )
            }
            with mock.patch.object(
                results,
                "_ordinary_relative_file",
                side_effect=AssertionError("Result path dereferenced"),
            ):
                with self.assertRaisesRegex(
                    result_import.ResultImportError,
                    "Incompatible Result",
                ):
                    result_import.import_result(
                        result,
                        object(),
                    )
            after = {
                name: len(getattr(bpy.data, name))
                for name in before
            }
            self.assertEqual(after, before)
            self.assertTrue(result.is_dir())
            self.assertTrue((result / "manifest.json").is_file())

    def test_extension_npy_replacement_at_load_seam_is_rejected(self):
        from blender_extension import result_import

        result_import._require_numpy()
        np = result_import.np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = root / "arrays"
            arrays.mkdir()
            path = arrays / "positions.npy"
            replacement = arrays / "replacement.npy"
            np.save(path, np.zeros((1, 3), dtype=np.dtype("<f4")))
            np.save(replacement, np.ones((1, 3), dtype=np.dtype("<f4")))
            digest, length, _identity = result_import._stable_sha256_identity(
                path
            )
            descriptor = {
                "path": "arrays/positions.npy",
                "dtype": "<f4",
                "shape": [1, 3],
                "byte_length": length,
                "sha256": digest,
            }
            original_load = np.load

            def replace_before_load(*args, **kwargs):
                os.replace(replacement, path)
                return original_load(path, **kwargs)

            with mock.patch.object(
                result_import.np,
                "load",
                side_effect=replace_before_load,
            ):
                with self.assertRaisesRegex(
                    result_import.ResultImportError,
                    "changed during consumption",
                ):
                    result_import._array_descriptor(
                        root,
                        "positions",
                        descriptor,
                        cancel=None,
                    )


if __name__ == "__main__":
    unittest.main()
