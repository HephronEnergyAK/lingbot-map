from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lingbot_diagnostics_for_tests",
    ROOT / "blender_extension" / "diagnostics.py",
)
diagnostics = importlib.util.module_from_spec(SPEC)
sys_modules = __import__("sys").modules
sys_modules[SPEC.name] = diagnostics
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnostics)

DiagnosticReportError = diagnostics.DiagnosticReportError
MAX_CLIPBOARD_BYTES = diagnostics.MAX_CLIPBOARD_BYTES
REPORT_SCHEMA_VERSION = diagnostics.REPORT_SCHEMA_VERSION
build_portable_report = diagnostics.build_portable_report
diagnostic_record = diagnostics.diagnostic_record
export_portable_report = diagnostics.export_portable_report
retain_extension_diagnostic = diagnostics.retain_extension_diagnostic


JOB_ID = "job-0123456789abcdef0123456789abcdef"


class PortableDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.diagnostic = self.root / (
            f"{JOB_ID}--reconstruction-failed"
        )
        self.diagnostic.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_json(self, name: str, value) -> None:
        (self.diagnostic / name).write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _fixture(self) -> None:
        self._write_json(
            "diagnostic.json",
            {
                "schema_version": "1.0.0",
                "error_code": "pipeline.reconstruction.failed",
                "category": "pipeline",
                "state": "failed",
                "phase": "inference",
                "job_id": JOB_ID,
                "machine_name": "WORKSTATION-7",
                "username": "Alice",
                "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "environment": {
                    "SECRET_TOKEN": "never-export-this",
                    "PATH": r"C:\Users\Alice\bin",
                },
                "target_scene": {
                    "blend_path": r"D:\Private Project\map.blend",
                    "scene_uuid": "scene-safe",
                    "scene_name": "Map",
                },
            },
        )
        self._write_json(
            "job-spec.json",
            {
                "schema_version": "1.0.0",
                "job_id": JOB_ID,
                "project_root": r"D:\Private Project\map.lingbot-map",
                "source": {
                    "absolute_path": r"C:\Users\Alice\Videos\secret.mov",
                    "scene_relative_path": r"..\Videos\secret.mov",
                    "sha256": "a" * 64,
                    "size_bytes": 42,
                },
            },
        )
        (self.diagnostic / "events.jsonl").write_text(
            json.dumps(
                {
                    "kind": "error",
                    "message": (
                        r"WORKSTATION-7 C:\Users\Alice\Videos\secret.mov "
                        + "a" * 64
                    ),
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (self.diagnostic / "human.log").write_text(
            (
                r"Alice@WORKSTATION-7 opened "
                r"C:\Users\Alice\Videos\secret.mov; token never-export-this; "
                + "a" * 64
            ),
            encoding="utf-8",
        )
        # Every one of these is outside the versioned logical allowlist.
        for name, data in {
            "source.mov": b"source-media",
            "model.safetensors": b"model-weights",
            "points.npy": b"point-array",
            "depth.npy": b"dense-predictions",
            "mask.bin": b"sky-mask",
            "download.partial": b"partial-download",
            "arbitrary-stage.bin": b"staging",
        }.items():
            (self.diagnostic / name).write_bytes(data)
        (self.diagnostic / "dense").mkdir()
        (self.diagnostic / "dense" / "hidden.npy").write_bytes(b"dense")

    def test_default_report_redacts_identity_and_excludes_every_unlisted_file(self):
        self._fixture()
        report = build_portable_report(
            self.diagnostic,
            versions={"extension": "0.1.0", "blender": "5.2.1"},
        )
        names = {name for name, _data in report.entries}
        self.assertEqual(
            names,
            {
                "errors.json",
                "logs/human.log",
                "records/diagnostic.json",
                "records/events.jsonl",
                "records/job-spec.json",
                "report-manifest.json",
                "summary.txt",
                "versions.json",
            },
        )
        combined = b"\n".join(data for _name, data in report.entries)
        for secret in (
            b"Alice",
            b"WORKSTATION-7",
            b"never-export-this",
            b"Private Project",
            b"secret.mov",
            b"GPU-12345678-1234-1234-1234-123456789abc",
            b"a" * 64,
            b"source-media",
            b"model-weights",
            b"point-array",
            b"dense-predictions",
            b"sky-mask",
            b"partial-download",
            b"staging",
        ):
            self.assertNotIn(secret, combined)
        self.assertIn(b"pipeline.reconstruction.failed", combined)
        self.assertIn(b"[SOURCE-1]", combined)
        self.assertEqual(report.manifest["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(report.manifest["identity_mode"], "redacted")
        self.assertEqual(report.manifest["excluded_entry_count"], 8)
        self.assertLessEqual(
            len(report.clipboard_text.encode("utf-8")),
            MAX_CLIPBOARD_BYTES + 64,
        )

    def test_unredacted_choice_never_expands_allowlist(self):
        self._fixture()
        report = build_portable_report(self.diagnostic, redact=False)
        combined = b"\n".join(data for _name, data in report.entries)
        self.assertIn(b"Alice", combined)
        self.assertIn(b"WORKSTATION-7", combined)
        self.assertNotIn(b"source-media", combined)
        self.assertEqual(report.manifest["identity_mode"], "unredacted")

    def test_setup_diagnostic_name_exports_record_but_not_partial_content(self):
        setup = self.root / (
            "20260723T010203.123456Z-"
            "aaaaaaaaaaaaaaaa-bbbbbbbbbbbb"
        )
        setup.mkdir()
        (setup / "setup-diagnostic.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "error_code": "setup.runtime.cancelled",
                    "category": "setup",
                    "state": "cancelled",
                    "phase": "runtime-setup",
                    "environment": {
                        "TOKEN": "setup-environment-secret"
                    },
                }
            ),
            encoding="utf-8",
        )
        (setup / "download.partial").write_bytes(
            b"partial payload"
        )
        report = build_portable_report(setup)
        combined = b"\n".join(data for _name, data in report.entries)
        self.assertIn(b"setup.runtime.cancelled", combined)
        self.assertNotIn(b"setup-environment-secret", combined)
        self.assertNotIn(b"partial payload", combined)
        self.assertIn(
            "records/setup-diagnostic.json",
            {name for name, _data in report.entries},
        )

    def test_export_zip_is_deterministic_and_manifest_checks_every_entry(self):
        self._fixture()
        first = self.root / "first.zip"
        second = self.root / "second.zip"
        result_a = export_portable_report(
            self.diagnostic,
            first,
            versions={"extension": "0.1.0"},
        )
        result_b = export_portable_report(
            self.diagnostic,
            second,
            versions={"extension": "0.1.0"},
        )
        self.assertEqual(result_a["sha256"], result_b["sha256"])
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(
                archive.namelist(),
                sorted(archive.namelist()),
            )
            manifest = json.loads(
                archive.read("report-manifest.json")
            )
            self.assertNotIn(
                "report-manifest.json",
                manifest["included_sections"],
            )
            for entry in manifest["entries"]:
                data = archive.read(entry["path"])
                self.assertEqual(len(data), entry["length"])
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(),
                    entry["sha256"],
                )
            logical_golden = hashlib.sha256()
            for name in archive.namelist():
                data = archive.read(name)
                logical_golden.update(name.encode("utf-8"))
                logical_golden.update(b"\0")
                logical_golden.update(len(data).to_bytes(8, "big"))
                logical_golden.update(data)
            self.assertEqual(
                logical_golden.hexdigest(),
                "83d2f9b661e222fa8473dadeedf70f0259f24a2befb14f79332d956cc6b1b0d8",
            )

    def test_malformed_oversized_and_linked_allowlisted_files_are_not_followed(self):
        (self.diagnostic / "diagnostic.json").write_text(
            '{"duplicate":1,"duplicate":2}\n',
            encoding="utf-8",
        )
        (self.diagnostic / "job-spec.json").write_bytes(
            b"{" + b"x" * (1024 * 1024) + b"}"
        )
        outside = self.root / "outside.log"
        outside.write_text("must-not-follow", encoding="utf-8")
        link = self.diagnostic / "human.log"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Filesystem does not permit a test symlink")
        report = build_portable_report(self.diagnostic)
        names = {name for name, _data in report.entries}
        self.assertNotIn("records/diagnostic.json", names)
        self.assertNotIn("records/job-spec.json", names)
        self.assertNotIn("logs/human.log", names)
        errors = json.loads(dict(report.entries)["errors.json"])["errors"]
        self.assertEqual(
            {entry["code"] for entry in errors},
            {
                "diagnostic-export-malformed-record",
                "diagnostic-export-file-too-large",
                "diagnostic-export-linked-file",
            },
        )
        self.assertNotIn(
            b"must-not-follow",
            b"\n".join(data for _name, data in report.entries),
        )

    def test_source_directory_link_and_occupied_destination_fail_closed(self):
        linked = self.root / f"{JOB_ID}--linked"
        try:
            linked.symlink_to(self.diagnostic, target_is_directory=True)
        except OSError:
            self.skipTest("Filesystem does not permit a test symlink")
        with self.assertRaisesRegex(
            DiagnosticReportError,
            "diagnostic-export-linked-source",
        ):
            build_portable_report(linked)
        destination = self.root / "occupied.zip"
        destination.write_bytes(b"keep")
        with self.assertRaisesRegex(
            DiagnosticReportError,
            "diagnostic-export-destination-occupied",
        ):
            export_portable_report(self.diagnostic, destination)
        self.assertEqual(destination.read_bytes(), b"keep")

    def test_destination_created_during_publication_is_not_replaced(self):
        destination = self.root / "raced.zip"
        real_publish = diagnostics._publish_new_file

        def race(temporary, target):
            target.write_bytes(b"keep-race-winner")
            real_publish(temporary, target)

        with mock.patch.object(
            diagnostics,
            "_publish_new_file",
            side_effect=race,
        ):
            with self.assertRaisesRegex(
                DiagnosticReportError,
                "diagnostic-export-write-failed",
            ):
                export_portable_report(self.diagnostic, destination)
        self.assertEqual(
            destination.read_bytes(),
            b"keep-race-winner",
        )

    def test_binary_payload_renamed_as_human_log_is_not_exported(self):
        payload = b"\x89PNG\r\n\x1a\n\x00source-media-payload"
        (self.diagnostic / "human.log").write_bytes(payload)
        report = build_portable_report(self.diagnostic)
        self.assertNotIn(
            "logs/human.log",
            {name for name, _data in report.entries},
        )
        self.assertNotIn(
            payload,
            b"\n".join(data for _name, data in report.entries),
        )
        errors = json.loads(dict(report.entries)["errors.json"])
        self.assertEqual(
            errors["errors"][0]["code"],
            "diagnostic-export-malformed-record",
        )

    def test_file_handle_identity_race_fails_closed(self):
        payload = "must-not-export-from-swapped-handle"
        (self.diagnostic / "human.log").write_text(
            payload,
            encoding="utf-8",
        )
        real_fstat = os.fstat

        def swapped_handle(file_descriptor):
            value = real_fstat(file_descriptor)
            return SimpleNamespace(
                st_dev=value.st_dev,
                st_ino=value.st_ino + 1,
                st_size=value.st_size,
                st_mtime_ns=value.st_mtime_ns,
                st_mode=value.st_mode,
                st_nlink=value.st_nlink,
                st_file_attributes=getattr(
                    value,
                    "st_file_attributes",
                    0,
                ),
            )

        with mock.patch.object(
            diagnostics.os,
            "fstat",
            side_effect=swapped_handle,
        ):
            report = build_portable_report(self.diagnostic)
        self.assertNotIn(
            payload.encode("utf-8"),
            b"\n".join(data for _name, data in report.entries),
        )
        errors = json.loads(dict(report.entries)["errors.json"])
        self.assertEqual(
            errors["errors"][0]["code"],
            "diagnostic-export-race",
        )

    def test_stable_record_and_extension_retention_cover_all_failure_categories(self):
        for category in ("setup", "launch", "pipeline", "import", "lifecycle"):
            record = diagnostic_record(
                error_code=f"{category}.operation.failed",
                category=category,
                state="failed",
                phase="test",
                detail="bounded",
                timestamp=datetime(2026, 7, 23, tzinfo=timezone.utc),
            )
            self.assertEqual(record["error_code"], f"{category}.operation.failed")
            self.assertEqual(record["category"], category)
        project = self.root / "project.lingbot-map"
        (project / "diagnostics").mkdir(parents=True)
        retained = retain_extension_diagnostic(
            project,
            error_code="import.transaction.failed",
            category="import",
            state="failed",
            phase="result-import",
            detail="rollback completed",
        )
        record = json.loads(
            (retained / "diagnostic.json").read_text(encoding="utf-8")
        )
        self.assertRegex(retained.name, r"^job-[0-9a-f]{32}--")
        self.assertEqual(record["error_code"], "import.transaction.failed")

    def test_directory_scan_is_bounded(self):
        with mock.patch.object(
            diagnostics,
            "MAX_DIRECTORY_ENTRIES",
            3,
        ):
            for index in range(6):
                (self.diagnostic / f"unknown-{index}").write_bytes(b"x")
            report = build_portable_report(self.diagnostic)
        self.assertTrue(report.manifest["directory_scan_truncated"])
        self.assertGreaterEqual(report.manifest["excluded_entry_count"], 1)


if __name__ == "__main__":
    unittest.main()
