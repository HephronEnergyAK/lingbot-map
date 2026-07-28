from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import release_package


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "release" / "package-policy.json"
LICENSE_CATALOG_PATH = ROOT / "release" / "dependency-licenses.json"


def _metadata() -> release_package.BuildMetadata:
    return release_package.BuildMetadata(
        repository="HephronEnergyAK/lingbot-map",
        commit="1" * 40,
        tag="v0.1.0",
        workflow_ref=(
            "HephronEnergyAK/lingbot-map/"
            ".github/workflows/release.yml@refs/tags/v0.1.0"
        ),
        run_id="2201",
        run_attempt="1",
        created_utc="2026-07-28T00:00:00Z",
    )


def _rewrite_archive(
    source: Path,
    destination: Path,
    *,
    remove: str | None = None,
    replace: tuple[str, bytes] | None = None,
    add: tuple[str, bytes] | None = None,
) -> None:
    with zipfile.ZipFile(source) as original:
        entries = {
            info.filename: original.read(info)
            for info in original.infolist()
            if info.filename != remove
        }
    if replace is not None:
        entries[replace[0]] = replace[1]
    if add is not None:
        entries[add[0]] = add[1]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)


class ReleasePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        self.artifacts = release_package.assemble_release(
            ROOT,
            self.output,
            _metadata(),
            policy_path=POLICY_PATH,
            license_catalog_path=LICENSE_CATALOG_PATH,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, archive: Path | None = None):
        return release_package.validate_release(
            archive or self.artifacts.archive,
            policy_path=POLICY_PATH,
            license_catalog_path=LICENSE_CATALOG_PATH,
            expected_metadata=_metadata(),
        )

    def test_builds_and_revalidates_the_complete_final_archive(self):
        summary = self.validate()
        self.assertEqual(summary["version"], "0.1.0")
        self.assertEqual(summary["repository"], "HephronEnergyAK/lingbot-map")
        self.assertGreater(summary["files"], 50)

        policy = release_package.load_policy(POLICY_PATH)
        with zipfile.ZipFile(self.artifacts.archive) as archive:
            names = set(archive.namelist())
            self.assertEqual(names, set(policy.archive_files))
            self.assertIn(
                "runtime_bundle/wheels/"
                "lingbot_map_worker-0.1.0-py3-none-any.whl",
                names,
            )
            self.assertIn(
                "runtime_bundle/wheels/"
                "lingbot_map-0.1.0-py3-none-any.whl",
                names,
            )
            self.assertIn("runtime_bundle/uv.lock", names)
            self.assertIn("runtime_bundle/schemas/catalog.json", names)
            self.assertIn("runtime_bundle/model-catalog.json", names)
            self.assertIn("locale_catalogs.py", names)
            self.assertIn("manual/en_US/index.html", names)
            self.assertIn("manual/zh_HANT/index.html", names)
            self.assertIn("LICENSE.txt", names)
            self.assertIn("NOTICE.txt", names)
            self.assertIn("release/SBOM.spdx.json", names)
            self.assertIn("release/provenance.json", names)
            self.assertIn("release/package-manifest.json", names)
            self.assertFalse(
                any(
                    name.endswith((".pt", ".pth", ".onnx", ".mp4", ".mov"))
                    for name in names
                )
            )

            provenance = json.loads(
                archive.read("release/provenance.json")
            )
            self.assertEqual(provenance["source"]["commit"], "1" * 40)
            self.assertEqual(provenance["source"]["tag"], "v0.1.0")
            self.assertEqual(
                provenance["source"]["kind"],
                "git-archive-of-tagged-commit",
            )

            sbom = json.loads(archive.read("release/SBOM.spdx.json"))
            package_names = {
                package["name"] for package in sbom["packages"]
            }
            inventory = json.loads(
                archive.read("runtime_bundle/runtime-inventory.json")
            )
            self.assertTrue(
                {item["name"] for item in inventory["packages"]}
                <= package_names
            )
            self.assertTrue(
                all(
                    package["licenseDeclared"] != "NOASSERTION"
                    for package in sbom["packages"]
                )
            )
            package_manifest = json.loads(
                archive.read("release/package-manifest.json")
            )
            self.assertEqual(
                package_manifest["claims"]["engineering_qualification_scope"],
                release_package.ENGINEERING_QUALIFICATION_SCOPE,
            )
            self.assertTrue(
                all(
                    complete is False
                    for complete in package_manifest["claims"][
                        "stable_release_gates"
                    ].values()
                )
            )

    def test_external_sha_sbom_and_provenance_match_the_archive(self):
        archive_sha = hashlib.sha256(
            self.artifacts.archive.read_bytes()
        ).hexdigest()
        self.assertEqual(
            self.artifacts.sha256.read_text("ascii"),
            f"{archive_sha}  {self.artifacts.archive.name}\n",
        )
        external = json.loads(
            self.artifacts.provenance.read_text("utf-8")
        )
        self.assertEqual(external["subject"]["sha256"], archive_sha)
        self.assertEqual(
            external["subject"]["name"],
            self.artifacts.archive.name,
        )
        with zipfile.ZipFile(self.artifacts.archive) as archive:
            self.assertEqual(
                self.artifacts.sbom.read_bytes(),
                archive.read("release/SBOM.spdx.json"),
            )
        external["builder"]["run_id"] = "tampered"
        self.artifacts.provenance.write_text(
            json.dumps(external),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            release_package.ReleasePackageError,
            "published provenance companion is invalid",
        ):
            release_package.validate_companions(self.artifacts)

    def test_sbom_rejects_duplicate_or_extra_records(self):
        policy = release_package.load_policy(POLICY_PATH)
        with zipfile.ZipFile(self.artifacts.archive) as archive:
            entries = {
                info.filename: archive.read(info)
                for info in archive.infolist()
            }
        inventory = release_package._runtime_inventory(entries)
        catalog = release_package._load_license_catalog(
            LICENSE_CATALOG_PATH
        )
        records = release_package._validate_dependency_licenses(
            inventory,
            catalog,
        )
        wheel_hashes = release_package._validate_lock_and_wheels(
            entries,
            "0.1.0",
        )
        sbom = json.loads(entries["release/SBOM.spdx.json"])
        sbom["packages"].append(dict(sbom["packages"][-1]))
        entries["release/SBOM.spdx.json"] = json.dumps(sbom).encode("utf-8")
        with self.assertRaisesRegex(
            release_package.ReleasePackageError,
            "does not exactly describe",
        ):
            release_package._validate_sbom(
                entries,
                inventory,
                records,
                "0.1.0",
                policy,
                _metadata(),
                wheel_hashes,
            )

    def test_explicit_allowlist_covers_every_extension_package_input(self):
        policy = release_package.load_policy(POLICY_PATH)
        extension = ROOT / "blender_extension"
        actual = {
            path.relative_to(extension).as_posix()
            for path in extension.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.relative_to(extension).as_posix()
            != "runtime_bundle/wheels/.gitignore"
        }
        self.assertEqual(actual, set(policy.source_files))

    def test_application_wheels_match_the_tagged_source(self):
        policy = release_package.load_policy(POLICY_PATH)
        entries = release_package._read_source_files(ROOT, policy)
        release_package._validate_application_wheel_sources(
            ROOT,
            entries,
            "0.1.0",
        )
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary)
            shutil.copy2(ROOT / "LICENSE.txt", copied / "LICENSE.txt")
            shutil.copytree(ROOT / "lingbot_map", copied / "lingbot_map")
            (copied / "worker").mkdir()
            shutil.copy2(
                ROOT / "worker" / "NOTICE.txt",
                copied / "worker" / "NOTICE.txt",
            )
            shutil.copytree(
                ROOT / "worker" / "src" / "lingbot_map_worker",
                copied / "worker" / "src" / "lingbot_map_worker",
            )
            worker_ipc = (
                copied / "worker" / "src" / "lingbot_map_worker" / "ipc.py"
            )
            worker_ipc.write_bytes(worker_ipc.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                release_package.ReleasePackageError,
                "does not match tagged source",
            ):
                release_package._validate_application_wheel_sources(
                    copied,
                    entries,
                    "0.1.0",
                )
        wheel_path = (
            "runtime_bundle/wheels/"
            "lingbot_map_worker-0.1.0-py3-none-any.whl"
        )
        original_wheel = entries[wheel_path]
        with zipfile.ZipFile(
            io.BytesIO(original_wheel)
        ) as original:
            wheel_entries = {
                info.filename: original.read(info)
                for info in original.infolist()
            }
        wheel_entries["unreviewed.pth"] = b"import unreviewed\n"
        changed_wheel = io.BytesIO()
        with zipfile.ZipFile(
            changed_wheel,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as changed:
            for name, data in wheel_entries.items():
                changed.writestr(name, data)
        entries[wheel_path] = changed_wheel.getvalue()
        with self.assertRaisesRegex(
            release_package.ReleasePackageError,
            "install payload does not match tagged source",
        ):
            release_package._validate_application_wheel_sources(
                ROOT,
                entries,
                "0.1.0",
            )
        with zipfile.ZipFile(io.BytesIO(original_wheel)) as original:
            wheel_entries = {
                info.filename: original.read(info)
                for info in original.infolist()
            }
        wheel_metadata = (
            "lingbot_map_worker-0.1.0.dist-info/WHEEL"
        )
        wheel_entries[wheel_metadata] += b"\n"
        changed_wheel = io.BytesIO()
        with zipfile.ZipFile(
            changed_wheel,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as changed:
            for name, data in wheel_entries.items():
                changed.writestr(name, data)
        entries[wheel_path] = changed_wheel.getvalue()
        with self.assertRaisesRegex(
            release_package.ReleasePackageError,
            "RECORD mismatch",
        ):
            release_package._validate_application_wheel_sources(
                ROOT,
                entries,
                "0.1.0",
            )

    def test_runtime_inventory_matches_windows_lock_closure(self):
        policy = release_package.load_policy(POLICY_PATH)
        entries = release_package._read_source_files(ROOT, policy)
        inventory = release_package._runtime_inventory(entries)
        identities = {
            (package["name"], package["version"])
            for package in inventory["packages"]
        }
        self.assertIn(("hf-xet", "1.5.2"), identities)
        incomplete = {
            identity for identity in identities if identity[0] != "hf-xet"
        }
        with self.assertRaisesRegex(
            release_package.ReleasePackageError,
            "does not match the Windows x64 lock closure",
        ):
            release_package._validate_runtime_lock_inventory(
                entries,
                incomplete,
            )

    def test_stable_release_requires_every_explicit_gate(self):
        policy = release_package.load_policy(POLICY_PATH)
        with self.assertRaisesRegex(
            release_package.ReleasePackageError,
            "stable 1.0 release gates remain incomplete",
        ):
            release_package._validate_stable_release_gates(
                policy,
                stable_release=True,
            )

    def test_rejects_unexpected_missing_tampered_and_placeholder_content(self):
        policy = release_package.load_policy(POLICY_PATH)
        cases = [
            (
                {"add": ("LICENSE.TXT", b"ambiguous")},
                "case-insensitive duplicate",
            ),
            (
                {"add": ("weights/model.pt", b"model")},
                "archive entries",
            ),
            (
                {"remove": "manual/zh_HANT/index.html"},
                "archive entries",
            ),
            (
                {
                    "replace": (
                        "runtime_bundle/wheels/"
                        "lingbot_map_worker-0.1.0-py3-none-any.whl",
                        b"not a wheel",
                    )
                },
                "package manifest checksum",
            ),
            (
                {
                    "replace": (
                        "manual/en_US/index.html",
                        b"<html><body>TODO</body></html>",
                    )
                },
                "placeholder",
            ),
        ]
        self.assertIn(
            "runtime_bundle/wheels/"
            "lingbot_map_worker-0.1.0-py3-none-any.whl",
            policy.archive_files,
        )
        for index, (mutation, message) in enumerate(cases):
            with self.subTest(index=index):
                changed = self.output / f"changed-{index}.zip"
                _rewrite_archive(
                    self.artifacts.archive,
                    changed,
                    **mutation,
                )
                with self.assertRaisesRegex(
                    release_package.ReleasePackageError,
                    message,
                ):
                    self.validate(changed)

    def test_engineering_release_pipeline_rejects_stable_tags(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "source"
            shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns(
                ".git", ".venv", ".cache", ".tmp", "__pycache__"
            ))
            manifest_path = (
                copied / "blender_extension" / "blender_manifest.toml"
            )
            manifest_path.write_text(
                manifest_path.read_text("utf-8").replace(
                    'version = "0.1.0"',
                    'version = "1.0.0"',
                ),
                encoding="utf-8",
            )
            for relative in (
                "pyproject.toml",
                "worker/pyproject.toml",
                "blender_extension/runtime_bundle/pyproject.toml",
            ):
                path = copied / relative
                path.write_text(
                    path.read_text("utf-8").replace(
                        'version = "0.1.0"',
                        'version = "1.0.0"',
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(
                release_package.ReleasePackageError,
                "accepts only 0.x tags",
            ):
                release_package.assemble_release(
                    copied,
                    Path(temporary) / "out",
                    release_package.BuildMetadata(
                        **{
                            **_metadata().__dict__,
                            "tag": "v1.0.0",
                            "workflow_ref": (
                                "HephronEnergyAK/lingbot-map/"
                                ".github/workflows/release.yml@"
                                "refs/tags/v1.0.0"
                            ),
                        }
                    ),
                    policy_path=copied / "release/package-policy.json",
                    license_catalog_path=(
                        copied / "release/dependency-licenses.json"
                    ),
                )

    def test_local_cli_cannot_produce_a_release_package(self):
        with self.assertRaisesRegex(
            release_package.ReleasePackageError,
            "GitHub Actions",
        ):
            release_package.require_ci_environment({})

    def test_git_export_uses_only_the_tagged_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            exported = Path(temporary) / "exported"
            repository.mkdir()
            subprocess.run(
                ["git", "init", "-q", str(repository)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "core.autocrlf",
                    "false",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            (repository / "tracked.txt").write_text(
                "tracked\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "-C", str(repository), "add", "tracked.txt"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "tag", "v0.1.0"],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            (repository / "untracked.txt").write_text(
                "must not escape\n",
                encoding="utf-8",
            )

            created = release_package.export_tagged_tree(
                repository,
                commit,
                "v0.1.0",
                exported,
            )
            self.assertEqual(created, commit)
            self.assertTrue((exported / "tracked.txt").is_file())
            self.assertFalse((exported / "untracked.txt").exists())

    def test_ci_cli_builds_and_validates_only_the_committed_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            shutil.copy2(ROOT / "pyproject.toml", repository)
            shutil.copy2(ROOT / ".gitattributes", repository)
            shutil.copy2(ROOT / "LICENSE.txt", repository)
            shutil.copytree(ROOT / "lingbot_map", repository / "lingbot_map")
            shutil.copytree(
                ROOT / "blender_extension",
                repository / "blender_extension",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(ROOT / "release", repository / "release")
            (repository / "scripts").mkdir()
            for name in ("release_package.py", "validate_localization.py"):
                shutil.copy2(
                    ROOT / "scripts" / name,
                    repository / "scripts" / name,
                )
            (repository / "worker").mkdir()
            shutil.copy2(
                ROOT / "worker" / "pyproject.toml",
                repository / "worker" / "pyproject.toml",
            )
            shutil.copy2(
                ROOT / "worker" / "NOTICE.txt",
                repository / "worker" / "NOTICE.txt",
            )
            shutil.copytree(
                ROOT / "worker" / "src",
                repository / "worker" / "src",
            )
            workflow = repository / ".github" / "workflows"
            workflow.mkdir(parents=True)
            shutil.copy2(
                ROOT / ".github" / "workflows" / "release.yml",
                workflow / "release.yml",
            )
            subprocess.run(
                ["git", "init", "-q", str(repository)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "core.autocrlf",
                    "false",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "add", "-f", "."],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "release"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "tag", "v0.1.0"],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            output = root / "release-output"
            environment = {
                "GITHUB_ACTIONS": "true",
                "CI": "true",
                "GITHUB_REF_TYPE": "tag",
                "GITHUB_REPOSITORY": "HephronEnergyAK/lingbot-map",
                "GITHUB_SHA": commit,
                "GITHUB_REF_NAME": "v0.1.0",
                "GITHUB_WORKFLOW_REF": (
                    "HephronEnergyAK/lingbot-map/"
                    ".github/workflows/release.yml@refs/tags/v0.1.0"
                ),
                "GITHUB_RUN_ID": "2202",
                "GITHUB_RUN_ATTEMPT": "1",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                self.assertEqual(
                    release_package.main(
                        [
                            "build",
                            "--repository-root",
                            str(repository),
                            "--output-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
                archive = (
                    output / "lingbot_map_reconstruction-0.1.0.zip"
                )
                self.assertEqual(
                    release_package.main(
                        [
                            "validate",
                            "--repository-root",
                            str(repository),
                            "--archive",
                            str(archive),
                        ]
                    ),
                    0,
                )

    def test_release_workflow_is_tag_only_and_attests_published_assets(self):
        workflow = (
            ROOT / ".github" / "workflows" / "release.yml"
        ).read_text("utf-8")
        self.assertIn('tags:\n      - "v0.[0-9]*.[0-9]*"', workflow)
        self.assertNotIn("workflow_dispatch", workflow)
        self.assertIn("actions/attest@v4", workflow)
        self.assertIn("sbom-path:", workflow)
        self.assertIn("gh release create", workflow)
        self.assertIn(
            "Engineering release qualification scope: Windows 11 x64",
            workflow,
        )
        self.assertIn("does not claim Ada qualification", workflow)
        self.assertIn("--verify-tag", workflow)
        self.assertIn("git ls-remote --exit-code origin", workflow)
        self.assertIn("scripts/release_package.py build", workflow)
        self.assertIn("scripts/release_package.py validate", workflow)


if __name__ == "__main__":
    unittest.main()
