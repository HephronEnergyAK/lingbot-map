from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile
import tempfile
import time
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "blender_extension" / "runtime_setup.py"
SPEC = importlib.util.spec_from_file_location("lingbot_runtime_setup_for_tests", MODULE_PATH)
runtime_setup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime_setup
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime_setup)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_uv_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("uv.exe", b"private uv")
    return buffer.getvalue()


def make_python_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        content = b"managed python"
        member = tarfile.TarInfo("python/python.exe")
        member.size = len(content)
        member.mode = 0o755
        archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


class FakeRunner:
    def __init__(self, fail_sync: Exception | None = None):
        self.calls = []
        self.fail_sync = fail_sync

    def run(self, command, *, cwd, env, cancellation):
        command = list(command)
        self.calls.append((command, Path(cwd), dict(env)))
        if command[-1] == "--version":
            return runtime_setup.subprocess.CompletedProcess(command, 0, "uv 0.11.16 (test build)\n", "")
        if "sync" in command:
            if self.fail_sync:
                raise self.fail_sync
            virtual_python = Path(cwd) / ".venv" / "Scripts" / "python.exe"
            virtual_python.parent.mkdir(parents=True)
            virtual_python.write_bytes(b"virtual python")
            (virtual_python.parents[1] / "pyvenv.cfg").write_text(
                f"home = {Path(cwd)}\\py\\python\n", encoding="utf-8"
            )
            return runtime_setup.subprocess.CompletedProcess(command, 0, "", "")
        if "lingbot_map_worker" in command:
            output = json.dumps(
                {"distribution": runtime_setup.WORKER_DISTRIBUTION, "version": runtime_setup.WORKER_VERSION}
            )
        elif "importlib.metadata" in command[-1]:
            output = json.dumps([[runtime_setup.WORKER_DISTRIBUTION, runtime_setup.WORKER_VERSION]])
        else:
            output = json.dumps(
                {"version": runtime_setup.PYTHON_VERSION, "bits": 64, "exe": str(Path(command[0]).resolve())}
            )
        return runtime_setup.subprocess.CompletedProcess(command, 0, output + "\n", "")


class RuntimeFixture:
    def __init__(self, root: Path):
        self.root = root
        self.bundle_root = root / "bundle"
        self.bundle_root.mkdir()
        self.artifact_bytes = {"uv": make_uv_archive(), "python": make_python_archive()}
        catalog = {"catalog_version": "1.0.0"}
        for name, data in self.artifact_bytes.items():
            filename = "uv.zip" if name == "uv" else "python.tar.gz"
            entry = {
                "filename": filename,
                "url": f"https://artifacts.example/{filename}",
                "length": len(data),
                "sha256": digest(data),
            }
            if name == "uv":
                entry["version"] = "0.11.16"
            else:
                entry.update({"version": "3.10.20", "key": "cpython-3.10.20-windows-x86_64-none"})
            catalog[name] = entry
        (self.bundle_root / "artifact-catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        inputs = {
            "pyproject.toml": "[project]\nname='runtime'\nversion='0.1.0'\n",
            "uv.lock": (
                "version = 1\n"
                "[[package]]\n"
                "name = 'lingbot-map-worker'\n"
                "version = '0.1.0'\n"
                "source = { path = 'wheels/lingbot_map_worker-0.1.0-py3-none-any.whl' }\n"
            ),
            "runtime-inventory.json": json.dumps(
                {
                    "schema_version": 1,
                    "platform": "windows-x64",
                    "python": "3.10.20",
                    "packages": [
                        {"name": "lingbot-map-worker", "version": "0.1.0"}
                    ],
                }
            ),
            "wheels/lingbot_map_worker-0.1.0-py3-none-any.whl": "wheel",
            "schemas/catalog.json": "{}",
            "model-catalog.json": "{}",
            "model-licenses/model-license.txt": "model license",
            "LICENSES/Apache-2.0.txt": "license",
            "NOTICES/worker.txt": "notice",
        }
        for relative, content in inputs.items():
            path = self.bundle_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def downloader(self, url: str, destination: Path) -> None:
        name = "uv" if url.endswith("uv.zip") else "python"
        destination.write_bytes(self.artifact_bytes[name])


class RuntimeIdentityTests(unittest.TestCase):
    def test_checked_in_bundle_pins_artifacts_lock_wheel_and_schema_catalog(self):
        bundle_root = ROOT / "blender_extension" / "runtime_bundle"
        bundle = runtime_setup.RuntimeBundle(bundle_root)
        artifacts = {artifact.name: artifact for artifact in bundle.artifacts}
        self.assertEqual(artifacts["uv"].metadata["version"], "0.11.16")
        self.assertEqual(artifacts["python"].metadata["version"], "3.10.20")
        self.assertEqual(
            artifacts["python"].metadata["key"], "cpython-3.10.20-windows-x86_64-none"
        )
        wheel = bundle_root / "wheels" / "lingbot_map_worker-0.1.0-py3-none-any.whl"
        lock = (bundle_root / "uv.lock").read_text(encoding="utf-8")
        self.assertIn(f"sha256:{runtime_setup.sha256_file(wheel)}", lock)
        inventory = json.loads(
            (bundle_root / "runtime-inventory.json").read_text(encoding="utf-8")
        )
        packages = {item["name"]: item["version"] for item in inventory["packages"]}
        self.assertEqual(packages["onnxruntime"], "1.23.2")
        self.assertNotIn("onnxruntime-gpu", packages)
        catalog = json.loads((bundle_root / "schemas" / "catalog.json").read_text(encoding="utf-8"))
        self.assertTrue(catalog["contracts"])
        for contract in catalog["contracts"]:
            self.assertEqual(
                runtime_setup.sha256_file(bundle_root / "schemas" / contract["path"]),
                contract["sha256"],
            )

    def test_identity_covers_lock_wheel_schemas_catalog_license_notice_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            first = runtime_setup.RuntimeBundle(fixture.bundle_root)
            paths = {item["path"] for item in first.identity.inputs}
            self.assertTrue(
                {
                    "artifact-catalog.json", "pyproject.toml", "uv.lock",
                    "runtime-inventory.json",
                    "wheels/lingbot_map_worker-0.1.0-py3-none-any.whl",
                    "schemas/catalog.json", "model-catalog.json",
                    "model-licenses/model-license.txt",
                    "LICENSES/Apache-2.0.txt", "NOTICES/worker.txt",
                }.issubset(paths)
            )
            (fixture.bundle_root / "schemas" / "catalog.json").write_text('{"changed":true}', encoding="utf-8")
            second = runtime_setup.RuntimeBundle(fixture.bundle_root)
            self.assertNotEqual(first.identity.runtime_id, second.identity.runtime_id)

    def test_catalog_rejects_unpinned_or_non_https_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            catalog_path = fixture.bundle_root / "artifact-catalog.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog["uv"]["url"] = "http://artifacts.example/uv.zip"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaisesRegex(runtime_setup.RuntimeSetupError, "HTTPS"):
                runtime_setup.RuntimeBundle(fixture.bundle_root)


class ArtifactPolicyTests(unittest.TestCase):
    def test_offline_reports_every_missing_artifact_without_calling_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            bundle = runtime_setup.RuntimeBundle(fixture.bundle_root)
            store = runtime_setup.ArtifactStore(Path(temporary) / "cache")
            network = mock.Mock(side_effect=AssertionError("network called"))
            with self.assertRaises(runtime_setup.MissingArtifactsError) as caught:
                store.acquire(bundle.artifacts, offline=True, online_access=True, downloader=network)
            self.assertEqual({item.name for item in caught.exception.missing}, {"python", "uv"})
            self.assertIn("Run explicit Online Setup once", str(caught.exception))
            network.assert_not_called()

    def test_online_requires_blender_online_access_and_verifies_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            bundle = runtime_setup.RuntimeBundle(fixture.bundle_root)
            store = runtime_setup.ArtifactStore(Path(temporary) / "cache")
            network = mock.Mock(wraps=fixture.downloader)
            with self.assertRaisesRegex(runtime_setup.RuntimeSetupError, "Online Access"):
                store.acquire(bundle.artifacts, offline=False, online_access=False, downloader=network)
            network.assert_not_called()
            paths = store.acquire(bundle.artifacts, offline=False, online_access=True, downloader=network)
            self.assertEqual(set(paths), {"python", "uv"})
            self.assertEqual(network.call_count, 2)
            network.reset_mock()
            store.acquire(bundle.artifacts, offline=True, online_access=False, downloader=network)
            network.assert_not_called()


class RuntimeInstallerTests(unittest.TestCase):
    def test_setup_is_frozen_isolated_same_parent_and_atomically_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            bundle = runtime_setup.RuntimeBundle(fixture.bundle_root)
            runner = FakeRunner()
            managed = Path(temporary) / "managed"
            poisoned = {
                "PATH": "external-python", "PYTHONPATH": "external-package",
                "CONDA_PREFIX": "external-conda", "PIP_INDEX_URL": "https://evil.invalid",
                "UV_INDEX": "https://evil.invalid", "HTTPS_PROXY": "https://evil.invalid",
            }
            with mock.patch.dict(runtime_setup.os.environ, poisoned, clear=False):
                installed = runtime_setup.RuntimeInstaller(
                    managed, bundle, downloader=fixture.downloader, runner=runner
                ).setup(offline=False, online_access=True)
            self.assertEqual(installed.parent, managed / "runtimes")
            self.assertEqual(installed.name, bundle.identity.runtime_id)
            self.assertEqual(
                runtime_setup.RuntimeInstaller(
                    managed, bundle, downloader=fixture.downloader, runner=runner
                ).validate_existing(),
                installed,
            )
            self.assertTrue((installed / "READY.json").is_file())
            self.assertIn(str(installed), (installed / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8"))
            self.assertFalse(any(installed.parent.glob(".staging-*")))
            sync_command, _, environment = next(call for call in runner.calls if "sync" in call[0])
            for argument in ("--frozen", "--no-dev", "--no-config", "--managed-python", "--no-python-downloads"):
                self.assertIn(argument, sync_command)
            self.assertNotIn("--no-index", sync_command)
            for name in poisoned:
                self.assertNotIn(name, environment)
            self.assertEqual(environment["UV_NO_CONFIG"], "1")
            self.assertNotIn("UV_OFFLINE", environment)
            self.assertEqual(environment["UV_CACHE_DIR"], str((managed / "uv-cache").resolve()))

    def test_offline_sync_stale_recovery_multiple_identities_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            managed = Path(temporary) / "managed"
            bundle1 = runtime_setup.RuntimeBundle(fixture.bundle_root)
            runner1 = FakeRunner()
            installer1 = runtime_setup.RuntimeInstaller(managed, bundle1, downloader=fixture.downloader, runner=runner1)
            first = installer1.setup(offline=False, online_access=True)
            stale = managed / "runtimes" / f".staging-{bundle1.identity.runtime_id[:16]}-dead-owner"
            stale.mkdir()
            (stale / ".runtime-id").write_text(bundle1.identity.runtime_id, encoding="ascii")
            (stale / "partial.txt").write_text("partial", encoding="utf-8")
            first_again = installer1.setup(offline=True, online_access=False)
            self.assertEqual(first, first_again)
            # Existing valid Runtime returns before stale recovery; stale belongs to a future retry only.
            self.assertTrue(stale.exists())

            (fixture.bundle_root / "model-catalog.json").write_text('{"version":2}', encoding="utf-8")
            bundle2 = runtime_setup.RuntimeBundle(fixture.bundle_root)
            stale2 = managed / "runtimes" / f".staging-{bundle2.identity.runtime_id[:16]}-dead-owner"
            stale2.mkdir()
            (stale2 / ".runtime-id").write_text(bundle2.identity.runtime_id, encoding="ascii")
            runner2 = FakeRunner()
            second = runtime_setup.RuntimeInstaller(
                managed, bundle2, downloader=fixture.downloader, runner=runner2
            ).setup(offline=True, online_access=False)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertNotEqual(first, second)
            self.assertFalse(stale2.exists())
            diagnostic = next((managed / "setup-diagnostics").glob("*"))
            record = json.loads((diagnostic / "setup-diagnostic.json").read_text(encoding="utf-8"))
            self.assertEqual(record["reason"], "stale-owner-recovery")
            sync = next(call for call in runner2.calls if "sync" in call[0])
            self.assertIn("--offline", sync[0])
            self.assertEqual(sync[2]["UV_OFFLINE"], "1")

    def test_cancellation_retains_staging_as_diagnostics_and_never_publishes(self):
        # Retained executable-shaped diagnostics can be released one tick late on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            fixture = RuntimeFixture(Path(temporary))
            bundle = runtime_setup.RuntimeBundle(fixture.bundle_root)
            runner = FakeRunner(runtime_setup.SetupCancelled("cancelled by test"))
            managed = Path(temporary) / "managed"
            installer = runtime_setup.RuntimeInstaller(managed, bundle, downloader=fixture.downloader, runner=runner)
            with self.assertRaises(runtime_setup.SetupCancelled):
                installer.setup(offline=False, online_access=True)
            self.assertFalse(installer.runtime_path.exists())
            diagnostics = list((managed / "setup-diagnostics").glob("*"))
            self.assertEqual(len(diagnostics), 1)
            record = json.loads((diagnostics[0] / "setup-diagnostic.json").read_text(encoding="utf-8"))
            self.assertEqual(record["reason"], "cancelled")

    def test_cleanup_requires_confirmation_and_preserves_extension_and_job_refs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [character * 64 for character in "abc"]
            for runtime_id in ids:
                (root / runtime_id).mkdir()
            with self.assertRaisesRegex(runtime_setup.RuntimeSetupError, "confirmation"):
                runtime_setup.cleanup_runtimes(
                    root, ids, installed_extension_refs=[ids[0]], active_job_refs=[ids[1]], confirmed=False
                )
            removed = runtime_setup.cleanup_runtimes(
                root, ids, installed_extension_refs=[ids[0]], active_job_refs=[ids[1]], confirmed=True
            )
            self.assertEqual(removed, (root.resolve() / ids[2],))
            self.assertTrue((root / ids[0]).is_dir())
            self.assertTrue((root / ids[1]).is_dir())


class LockAndArchiveTests(unittest.TestCase):
    def test_lock_records_exact_owner_and_rejects_a_second_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.lock"
            first = runtime_setup.ProcessIdentity(10, 20, "first.exe", "nonce-1")
            second = runtime_setup.ProcessIdentity(11, 21, "second.exe", "nonce-2")
            with runtime_setup.RuntimeLock(path, first):
                owner_path = path.with_suffix(".lock.owner.json")
                self.assertEqual(json.loads(owner_path.read_text(encoding="utf-8")), runtime_setup.asdict(first))
                with self.assertRaises(runtime_setup.RuntimeBusyError):
                    with runtime_setup.RuntimeLock(path, second):
                        pass

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "extract"
            destination.mkdir()
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("../escape.exe", b"bad")
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive:
                with self.assertRaises(runtime_setup.RuntimeSetupError):
                    runtime_setup._safe_extract_zip(archive, destination)

    def test_cross_process_lock_serializes_the_exact_runtime_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / ("a" * 64 + ".lock")
            ready_path = root / "ready"
            release_path = root / "release"
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "runtime_lock_holder.py"),
                    str(lock_path),
                    str(ready_path),
                    str(release_path),
                ]
            )
            try:
                deadline = time.monotonic() + 10
                while not ready_path.exists():
                    self.assertIsNone(child.poll(), "lock-holder child exited early")
                    self.assertLess(time.monotonic(), deadline, "lock-holder child timed out")
                    time.sleep(0.02)
                with self.assertRaises(runtime_setup.RuntimeBusyError):
                    with runtime_setup.RuntimeLock(lock_path, runtime_setup.current_process_identity()):
                        pass
                release_path.write_text("release", encoding="ascii")
                self.assertEqual(child.wait(timeout=10), 0)
                with runtime_setup.RuntimeLock(lock_path, runtime_setup.current_process_identity()):
                    pass
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
