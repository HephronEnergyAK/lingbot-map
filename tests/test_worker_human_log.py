from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

from lingbot_map_worker.human_log import (  # noqa: E402
    HumanLogSession,
    RotatingHumanLog,
    write_bootstrap_diagnostic,
    write_terminal_diagnostic,
)
from lingbot_map_worker.ipc import atomic_write_json  # noqa: E402
from lingbot_map_worker.fixture_job import StatusStore  # noqa: E402


JOB_ID = "job-abcdefabcdefabcdefabcdefabcdefab"


class WorkerHumanLogTests(unittest.TestCase):
    def test_active_log_never_writes_through_a_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / "logs"
            root.mkdir()
            outside = workspace / "outside.log"
            outside.write_bytes(b"keep")
            os.link(outside, root / "human.log")
            with self.assertRaisesRegex(
                RuntimeError,
                "Linked or non-file",
            ):
                RotatingHumanLog(root)
            self.assertEqual(outside.read_bytes(), b"keep")

    def test_four_files_bound_and_structured_cumulative_discard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            discarded = []
            stream = RotatingHumanLog(
                root,
                file_bytes=8,
                file_count=4,
                on_discard=discarded.append,
            )
            stream.write(b"abcdefghijklmnopqrstuvwxyzABCDEFG")
            stream.close()
            paths = sorted(root.glob("human.log*"))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.stat().st_size <= 8 for path in paths))
            self.assertEqual(sum(path.stat().st_size for path in paths), 25)
            self.assertEqual(discarded, [8])
            self.assertEqual(stream.discarded_bytes, 8)
            self.assertEqual((root / "human.log").read_bytes(), b"G")

    def test_redirects_text_and_binary_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = HumanLogSession(
                root,
                on_discard=lambda _count: None,
                file_bytes=1024,
            ).start()
            try:
                print("third-party stdout")
                sys.stderr.write("third-party stderr\n")
                sys.stdout.buffer.write(b"binary output\n")
            finally:
                session.close()
            text = (root / "human.log").read_text(encoding="utf-8")
            self.assertIn("third-party stdout", text)
            self.assertIn("third-party stderr", text)
            self.assertIn("binary output", text)

    def test_discard_emits_immediate_structured_warning_and_status_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = StatusStore(root, JOB_ID, 10)
            store._protocol_stdout = io.BytesIO()
            store.log_truncated(12345)
            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            status = json.loads(
                (root / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(events[-1]["kind"], "warning")
            self.assertTrue(events[-1]["immediate"])
            self.assertIn("12345 cumulative bytes", events[-1]["message"])
            self.assertEqual(status["progress_event_sequence"], 1)

    def test_python_warning_is_kept_in_log_and_structured_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = StatusStore(root, JOB_ID, 1)
            store._protocol_stdout = io.BytesIO()
            session = HumanLogSession(
                root,
                on_discard=store.log_truncated,
                on_warning=store.structured_warning,
            ).start()
            try:
                warnings.warn(
                    "model adapter warning",
                    RuntimeWarning,
                )
            finally:
                session.close()
            log = (root / "human.log").read_text("utf-8")
            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(
                    "utf-8"
                ).splitlines()
            ]
            self.assertIn("model adapter warning", log)
            self.assertEqual(events[-1]["kind"], "warning")
            self.assertIn("RuntimeWarning", events[-1]["message"])

    def test_terminal_record_keeps_stable_code_identity_and_bounded_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / JOB_ID
            job_dir.mkdir()
            job = {
                "schema_version": "1.0.0",
                "job_id": JOB_ID,
                "target_scene": {
                    "blend_path": r"D:\Project\map.blend",
                    "scene_uuid": "78aa5f6b-f04f-4b60-b6af-dd69859b5acf",
                    "scene_name": "Scene",
                },
            }
            atomic_write_json(job_dir / "job-spec.json", job)
            try:
                raise RuntimeError("sensitive pipeline failure")
            except RuntimeError as exc:
                write_terminal_diagnostic(
                    job_dir,
                    job,
                    error_code="pipeline.reconstruction.failed",
                    state="failed",
                    phase="inference",
                    detail=exc,
                    discarded_log_bytes=123,
                    exception=exc,
                )
            record = json.loads(
                (job_dir / "diagnostic.json").read_text(encoding="utf-8")
            )
            exception = json.loads(
                (job_dir / "exception.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                record["error_code"],
                "pipeline.reconstruction.failed",
            )
            self.assertEqual(record["job_id"], JOB_ID)
            self.assertEqual(record["category"], "pipeline")
            self.assertTrue(record["human_log"]["truncated"])
            self.assertEqual(record["human_log"]["discarded_bytes"], 123)
            self.assertEqual(
                exception["error_code"],
                "pipeline.reconstruction.failed",
            )
            self.assertEqual(exception["type"], "RuntimeError")
            self.assertGreaterEqual(len(exception["traceback"]), 1)

    def test_bootstrap_failure_is_structured_before_status_store_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / ".jobs"
            job_dir = jobs / JOB_ID
            job_dir.mkdir(parents=True)
            spec = job_dir / "job-spec.json"
            spec.write_bytes(b"{malformed")
            try:
                raise ValueError("control envelope never arrived")
            except ValueError as exc:
                write_bootstrap_diagnostic(spec, exc)
            record = json.loads(
                (job_dir / "diagnostic.json").read_text("utf-8")
            )
            exception = json.loads(
                (job_dir / "exception.json").read_text("utf-8")
            )
            self.assertEqual(
                record["error_code"],
                "launch.worker.bootstrap-failed",
            )
            self.assertEqual(record["category"], "launch")
            self.assertEqual(record["phase"], "bootstrap")
            self.assertEqual(
                exception["error_code"],
                "launch.worker.bootstrap-failed",
            )


if __name__ == "__main__":
    unittest.main()
