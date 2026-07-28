from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = ROOT / "blender_extension"
PACKAGE_NAME = "lingbot_model_store_tests"
package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(EXTENSION_ROOT)]
sys.modules[PACKAGE_NAME] = package


def load_module(name: str):
    full_name = f"{PACKAGE_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, EXTENSION_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runtime_setup = load_module("runtime_setup")
model_store = load_module("model_store")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeResponse:
    def __init__(self, body: bytes, *, status: int, headers: dict[str, str], chunk_size=None):
        self._stream = io.BytesIO(body)
        self.status = status
        self.headers = headers
        self._chunk_size = chunk_size

    def read(self, size=-1):
        if self._chunk_size is not None and size > self._chunk_size:
            size = self._chunk_size
        return self._stream.read(size)

    def geturl(self):
        return "https://cdn.example/immutable-artifact"

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class FakeOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected network request")
        return self.responses.pop(0)


class CatalogFixture:
    def __init__(self, root: Path, content: bytes = b"catalogued-model-bytes"):
        self.root = root
        self.content = content
        self.digest = sha256(content)
        license_text = root / "model-licenses" / "MIT.txt"
        record = root / "model-licenses" / "record.txt"
        license_text.parent.mkdir(parents=True)
        license_text.write_text("MIT test license\n", encoding="utf-8")
        record.write_text("weight-specific test evidence\n", encoding="utf-8")
        revision = "a" * 40
        catalog = {
            "catalog_version": "1.0.0",
            "models": [
                {
                    "id": "fixture-model",
                    "display_name": "Fixture Model",
                    "role": "reconstruction",
                    "required": True,
                    "architecture": "fixture",
                    "input_contract": {"fixture": True},
                    "serialization": {
                        "format": "pytorch-zip-checkpoint",
                        "load_policy": "torch.load(weights_only=True)",
                        "arbitrary_pickle_allowed": False,
                    },
                    "artifact": {
                        "filename": "fixture.pt",
                        "length": len(content),
                        "sha256": self.digest,
                        "url": f"https://models.example/resolve/{revision}/fixture.pt",
                        "source_repository": "https://models.example/fixture",
                        "source_revision": revision,
                    },
                    "license_record": {
                        "status": "confirmed-for-test",
                        "spdx_expression": "MIT",
                        "covers_weights": True,
                        "captured_license_text_path": "model-licenses/MIT.txt",
                        "captured_license_text_sha256": model_store.sha256_file(license_text),
                        "record_path": "model-licenses/record.txt",
                        "record_sha256": model_store.sha256_file(record),
                        "copyright_or_attribution_source": "fixture author",
                        "evidence_basis": "fixture evidence",
                        "release_gate": "clear-for-test",
                    },
                }
            ],
        }
        self.catalog_path = root / "model-catalog.json"
        self.catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        self.catalog = model_store.ModelCatalog(self.catalog_path)
        self.entry = self.catalog.by_id("fixture-model")

    def full_response(self, content=None, *, chunk_size=None):
        body = self.content if content is None else content
        return FakeResponse(
            body,
            status=200,
            headers={
                "Content-Length": str(len(body)),
                "Accept-Ranges": "bytes",
                "ETag": '"fixture-etag"',
            },
            chunk_size=chunk_size,
        )

    def range_response(self, start: int, *, etag='"fixture-etag"'):
        body = self.content[start:]
        return FakeResponse(
            body,
            status=206,
            headers={
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{len(self.content) - 1}/{len(self.content)}",
                "Accept-Ranges": "bytes",
                "ETag": etag,
            },
        )


class CheckedInCatalogTests(unittest.TestCase):
    def test_catalog_pins_two_real_roles_and_pending_weight_license_gates(self):
        catalog = model_store.ModelCatalog(ROOT / "model-catalog.json")
        self.assertEqual(catalog.version, "1.0.0")
        self.assertEqual({entry.role for entry in catalog.entries}, {"reconstruction", "auxiliary"})
        lingbot = catalog.by_id("lingbot-map-long")
        self.assertEqual(lingbot.artifact.length, 4_632_303_465)
        self.assertEqual(
            lingbot.artifact.sha256,
            "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409",
        )
        skyseg = catalog.by_id("skyseg")
        self.assertEqual(skyseg.artifact.length, 175_997_079)
        self.assertEqual(
            skyseg.artifact.sha256,
            "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39",
        )
        for entry in catalog.entries:
            self.assertFalse(entry.license_record.covers_weights)
            self.assertEqual(entry.license_record.release_gate, "blocked-for-1.0")
            self.assertIn(entry.artifact.source_revision, entry.artifact.url)

    def test_extension_and_runtime_catalogs_and_license_records_are_identical(self):
        source_catalog = ROOT / "model-catalog.json"
        bundle = ROOT / "blender_extension" / "runtime_bundle"
        self.assertEqual(
            model_store.sha256_file(source_catalog),
            model_store.sha256_file(bundle / "model-catalog.json"),
        )
        for source in (ROOT / "model-licenses").iterdir():
            if source.is_file():
                self.assertEqual(
                    model_store.sha256_file(source),
                    model_store.sha256_file(bundle / "model-licenses" / source.name),
                )

    def test_offline_missing_report_names_every_requested_real_artifact(self):
        catalog = model_store.ModelCatalog(ROOT / "model-catalog.json")
        with tempfile.TemporaryDirectory() as temporary:
            store = model_store.ModelStore(Path(temporary), catalog, opener=FakeOpener())
            with self.assertRaises(model_store.MissingModelArtifactsError) as caught:
                store.ensure_offline(entry.id for entry in catalog.entries)
        self.assertEqual({entry.id for entry in caught.exception.entries}, {"lingbot-map-long", "skyseg"})
        for entry in catalog.entries:
            self.assertIn(entry.artifact.filename, str(caught.exception))
            self.assertIn(entry.artifact.sha256, str(caught.exception))


class ModelAcquisitionTests(unittest.TestCase):
    def test_offline_and_blender_online_access_fail_before_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            opener = FakeOpener()
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog, opener=opener)
            with self.assertRaises(model_store.MissingModelArtifactsError) as caught:
                store.acquire("fixture-model", offline=True, online_access=True)
            self.assertIn(fixture.digest, str(caught.exception))
            with self.assertRaisesRegex(model_store.ModelStoreError, "Online Access"):
                store.acquire("fixture-model", offline=False, online_access=False)
            self.assertEqual(opener.requests, [])

    def test_full_download_registers_only_after_checksum_and_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            opener = FakeOpener(fixture.full_response())
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog, opener=opener)
            artifact = store.acquire("fixture-model", offline=False, online_access=True)
            self.assertEqual(artifact.read_bytes(), fixture.content)
            self.assertEqual(store.quick_status(fixture.entry), "registered")
            self.assertEqual(store.validate(fixture.entry), artifact)
            self.assertFalse(any(store.models_root.glob(".staging-*")))
            registration = json.loads(
                (artifact.parent / "model-registration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(registration["catalog_sha256"], fixture.catalog.sha256)
            self.assertTrue((store.locks_root / f"{fixture.digest}.lock.owner.json").is_file())

    def test_valid_range_resume_uses_if_range_and_exact_content_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            start = 5
            opener = FakeOpener(fixture.range_response(start))
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog, opener=opener)
            store.downloads_root.mkdir(parents=True)
            # Simulate a process death after the last durable metadata checkpoint.
            store.partial_path(fixture.entry).write_bytes(fixture.content[: start + 2])
            store._write_resume_state(
                fixture.entry,
                model_store.ResumeState(
                    fixture.entry.artifact.url,
                    len(fixture.content),
                    start,
                    "ETag",
                    '"fixture-etag"',
                    True,
                ),
            )
            artifact = store.acquire("fixture-model", offline=False, online_access=True)
            self.assertEqual(artifact.read_bytes(), fixture.content)
            request = opener.requests[0][0]
            self.assertEqual(request.get_header("Range"), f"bytes={start}-")
            self.assertEqual(request.get_header("If-range"), '"fixture-etag"')

    def test_changed_validator_quarantines_partial_then_restarts_from_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            start = 4
            opener = FakeOpener(
                fixture.range_response(start, etag='"changed"'),
                fixture.full_response(),
            )
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog, opener=opener)
            store.downloads_root.mkdir(parents=True)
            store.partial_path(fixture.entry).write_bytes(fixture.content[:start])
            store._write_resume_state(
                fixture.entry,
                model_store.ResumeState(
                    fixture.entry.artifact.url,
                    len(fixture.content),
                    start,
                    "ETag",
                    '"fixture-etag"',
                    True,
                ),
            )
            artifact = store.acquire("fixture-model", offline=False, online_access=True)
            self.assertEqual(artifact.read_bytes(), fixture.content)
            reasons = [
                json.loads(path.read_text(encoding="utf-8"))["reason"]
                for path in store.diagnostics_root.glob("*/model-download-diagnostic.json")
            ]
            self.assertIn("invalid-resume-response", reasons)
            self.assertIsNone(opener.requests[1][0].get_header("Range"))

    def test_full_response_to_range_discards_partial_then_restarts_from_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            start = 4
            opener = FakeOpener(fixture.full_response(), fixture.full_response())
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog, opener=opener)
            store.downloads_root.mkdir(parents=True)
            store.partial_path(fixture.entry).write_bytes(fixture.content[:start])
            store._write_resume_state(
                fixture.entry,
                model_store.ResumeState(
                    fixture.entry.artifact.url,
                    len(fixture.content),
                    start,
                    "ETag",
                    '"fixture-etag"',
                    True,
                ),
            )

            artifact = store.acquire("fixture-model", offline=False, online_access=True)

            self.assertEqual(artifact.read_bytes(), fixture.content)
            self.assertEqual(opener.requests[0][0].get_header("Range"), f"bytes={start}-")
            self.assertIsNone(opener.requests[1][0].get_header("Range"))
            diagnostic = next(store.diagnostics_root.glob("*/model-download-diagnostic.json"))
            self.assertEqual(
                json.loads(diagnostic.read_text(encoding="utf-8"))["reason"],
                "invalid-resume-response",
            )

    def test_cancellation_retains_only_validated_partial_then_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog", b"0123456789abcdef")
            token = runtime_setup.CancellationToken()
            opener = FakeOpener(fixture.full_response(chunk_size=4))
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog, opener=opener)

            def cancel_after_first_chunk(completed, _total):
                if completed >= 4:
                    token.cancel()

            with self.assertRaises(runtime_setup.SetupCancelled):
                store.acquire(
                    "fixture-model",
                    offline=False,
                    online_access=True,
                    cancellation=token,
                    progress=cancel_after_first_chunk,
                )
            partial = store.partial_path(fixture.entry)
            self.assertEqual(partial.read_bytes(), fixture.content[:4])
            state = model_store.ResumeState(
                **json.loads(store.partial_metadata_path(fixture.entry).read_text(encoding="utf-8"))
            )
            self.assertEqual(state.completed, 4)
            store.opener = FakeOpener(fixture.range_response(4))
            artifact = store.acquire("fixture-model", offline=False, online_access=True)
            self.assertEqual(artifact.read_bytes(), fixture.content)

    def test_non_resumable_cancellation_is_quarantined_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog", b"0123456789abcdef")
            token = runtime_setup.CancellationToken()
            response = FakeResponse(
                fixture.content,
                status=200,
                headers={"Content-Length": str(len(fixture.content))},
                chunk_size=4,
            )
            store = model_store.ModelStore(
                Path(temporary) / "managed", fixture.catalog, opener=FakeOpener(response)
            )

            def cancel(completed, _total):
                if completed >= 4:
                    token.cancel()

            with self.assertRaises(runtime_setup.SetupCancelled):
                store.acquire(
                    "fixture-model", offline=False, online_access=True,
                    cancellation=token, progress=cancel,
                )
            self.assertFalse(store.partial_path(fixture.entry).exists())
            self.assertFalse(store.partial_metadata_path(fixture.entry).exists())
            record = next(store.diagnostics_root.glob("*/model-download-diagnostic.json"))
            self.assertEqual(
                json.loads(record.read_text(encoding="utf-8"))["reason"],
                "non-resumable-interruption",
            )

    def test_checksum_mismatch_is_quarantined_and_never_registered(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            corrupt = b"x" * len(fixture.content)
            store = model_store.ModelStore(
                Path(temporary) / "managed",
                fixture.catalog,
                opener=FakeOpener(fixture.full_response(corrupt)),
            )
            with self.assertRaisesRegex(model_store.ModelStoreError, "checksum mismatch"):
                store.acquire("fixture-model", offline=False, online_access=True)
            self.assertFalse(store.model_path(fixture.entry).exists())
            diagnostic = next(store.diagnostics_root.glob("*/model-download-diagnostic.json"))
            record = json.loads(diagnostic.read_text(encoding="utf-8"))
            self.assertEqual(record["reason"], "checksum-mismatch")
            self.assertEqual(
                record["error_code"],
                "setup.model.checksum-mismatch",
            )
            self.assertEqual(record["category"], "setup")


class LocalImportTests(unittest.TestCase):
    def test_known_file_is_copied_not_linked_then_revalidated(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            source = Path(temporary) / "selected.pt"
            source.write_bytes(fixture.content)
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog)
            entry, artifact = store.import_local(source, expected_model_id="fixture-model")
            self.assertEqual(entry, fixture.entry)
            self.assertEqual(artifact.read_bytes(), fixture.content)
            self.assertFalse(artifact.samefile(source))
            self.assertEqual(source.read_bytes(), fixture.content)

            second_document = json.loads(fixture.catalog_path.read_text(encoding="utf-8"))
            second_document["models"][0]["id"] = "fixture-model-v2"
            second_document["models"][0]["display_name"] = "Fixture Model v2"
            second_path = fixture.root / "model-catalog-v2.json"
            second_path.write_text(json.dumps(second_document), encoding="utf-8")
            second_catalog = model_store.ModelCatalog(second_path)
            second_entry = second_catalog.by_id("fixture-model-v2")
            second_store = model_store.ModelStore(store.managed_root, second_catalog)
            self.assertNotEqual(fixture.catalog.sha256, second_catalog.sha256)
            self.assertEqual(second_store.quick_status(second_entry), "registered")
            self.assertEqual(second_store.validate(second_entry), artifact)

    def test_unknown_pickle_is_hashed_and_rejected_without_deserialization(self):
        class CreateMarker:
            def __reduce__(self):
                return (Path.write_text, (Path(temporary) / "pickle-executed", "bad"))

        with tempfile.TemporaryDirectory() as temporary:
            fixture = CatalogFixture(Path(temporary) / "catalog")
            source = Path(temporary) / "unknown.pt"
            source.write_bytes(pickle.dumps(CreateMarker()))
            store = model_store.ModelStore(Path(temporary) / "managed", fixture.catalog)
            with self.assertRaisesRegex(model_store.UnsupportedModelError, "not present"):
                store.import_local(source)
            self.assertFalse((Path(temporary) / "pickle-executed").exists())
            self.assertFalse(store.models_root.exists())


if __name__ == "__main__":
    unittest.main()
