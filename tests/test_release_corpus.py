from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if np is not None:
    from scripts import build_release_corpus
    from scripts.calibrate_neural_oracle import (
        CalibrationError,
        _prepare_output,
    )
    from scripts.validate_neural_oracle import validate_neural_output
    from scripts.validate_release_corpus import (
        CorpusValidationError,
        _validate_generated,
        _validate_neural_oracles,
        load_json,
        validate,
    )


@unittest.skipIf(np is None, "Release corpus requires the pinned Worker Runtime")
class ReleaseCorpusTests(unittest.TestCase):
    def test_neural_calibration_refuses_nonempty_output_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            sentinel = output / "preserve.txt"
            sentinel.write_text("user data", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "new or empty"):
                _prepare_output(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")

    def test_structural_oracles_pass_and_release_gate_names_are_exact(self):
        result = validate()
        self.assertEqual(result["structural_validation"], "passed")
        self.assertFalse(result["release_ready"])
        self.assertEqual(
            [item["id"] for item in result["blockers"]],
            [
                "real-windowed-capture-rights",
                "ada-release-suite",
            ],
        )
        with self.assertRaisesRegex(
            CorpusValidationError,
            "real-windowed-capture-rights, ada-release-suite",
        ):
            validate(release=True)

    def test_small_media_generation_is_byte_deterministic_and_golden(self):
        goldens = load_json(ROOT / "release_corpus" / "generated-goldens.json")
        expected = {
            item["id"]: item for item in goldens["fixtures"]
        }["synthetic-boundary-8"]
        records = []
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for attempt in range(2):
                output = base / f"attempt-{attempt}"
                document = build_release_corpus.build(
                    output,
                    include_stress=False,
                    include_real=False,
                    only=frozenset({"synthetic-boundary-8"}),
                )
                record = document["fixtures"][0]
                for key, value in expected.items():
                    self.assertEqual(record.get(key), value, key)
                records.append(record)
            self.assertEqual(records[0], records[1])
            with self.assertRaisesRegex(RuntimeError, "must be empty"):
                build_release_corpus.build(
                    base / "attempt-0",
                    include_stress=False,
                    include_real=False,
                )

    def test_generated_manifest_requires_complete_ordinary_corpus(self):
        goldens = load_json(ROOT / "release_corpus" / "generated-goldens.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "only-one"
            build_release_corpus.build(
                output,
                include_stress=False,
                include_real=False,
                only=frozenset({"synthetic-boundary-8"}),
            )
            with self.assertRaisesRegex(
                CorpusValidationError, "ordinary generated corpus is incomplete"
            ):
                _validate_generated(
                    output / "generated-manifest.json",
                    goldens,
                    require_stress=False,
                )

    def test_neural_catalog_forbids_checksum_and_requires_calibrated_ranges(self):
        oracle = load_json(ROOT / "release_corpus" / "neural-oracles.json")
        _validate_neural_oracles(oracle)
        weakened = copy.deepcopy(oracle)
        weakened["cross_gpu_checksum_allowed"] = True
        with self.assertRaisesRegex(CorpusValidationError, "must be forbidden"):
            _validate_neural_oracles(weakened)
        invented = copy.deepcopy(oracle)
        invented["fixtures"]["real-windowed-above-3000"]["ranges"] = {}
        with self.assertRaisesRegex(CorpusValidationError, "must not invent"):
            _validate_neural_oracles(invented)
        uncalibrated = copy.deepcopy(oracle)
        uncalibrated["fixtures"]["real-courthouse-streaming-286"].pop(
            "calibration"
        )
        with self.assertRaisesRegex(CorpusValidationError, "calibration evidence"):
            _validate_neural_oracles(uncalibrated)


@unittest.skipIf(np is None, "Neural oracle requires the pinned Worker Runtime")
class NeuralOutputOracleTests(unittest.TestCase):
    FIXTURE_ID = "fixture-ready"
    SOURCE_SHA = "a" * 64

    def _write_output(self, root: Path) -> None:
        frames, height, width = 3, 2, 2
        cameras = np.repeat(
            np.eye(4, dtype="<f8")[None, :, :], frames, axis=0
        )
        cameras[1, 0, 3] = -0.5
        cameras[2, 0, 3] = -1.0
        intrinsics = np.repeat(
            np.array(
                ((2.0, 0.0, 0.5), (0.0, 2.0, 0.5), (0.0, 0.0, 1.0)),
                dtype="<f8",
            )[None, :, :],
            frames,
            axis=0,
        )
        arrays = {
            "world_to_camera_opencv.npy": cameras,
            "model_intrinsics.npy": intrinsics,
            "depth.npy": np.ascontiguousarray(
                np.arange(1, frames * height * width + 1, dtype=np.float32)
                .reshape(frames, height, width),
                dtype="<f4",
            ),
            "confidence.npy": np.ascontiguousarray(
                np.linspace(0.1, 0.9, frames * height * width, dtype=np.float32)
                .reshape(frames, height, width),
                dtype="<f4",
            ),
            "source_pts_seconds.npy": np.asarray(
                (0.0, 0.04, 0.08), dtype="<f8"
            ),
            "frame_type.npy": np.asarray((0, 1, 2), dtype="|u1"),
        }
        for name, array in arrays.items():
            np.save(root / name, array, allow_pickle=False)
        (root / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "fixture_id": self.FIXTURE_ID,
                    "source_sha256": self.SOURCE_SHA,
                    "runtime_id": "b" * 64,
                    "model_id": "fixture-model",
                    "model_sha256": "c" * 64,
                    "profile": "Draft",
                    "frame_count": frames,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_oracle(self, root: Path) -> Path:
        ranges = {
            "depth_p05": [1.0, 2.0],
            "depth_median": [6.0, 7.0],
            "depth_p95": [11.0, 12.0],
            "confidence_p05": [0.1, 0.2],
            "confidence_median": [0.4, 0.6],
            "confidence_p95": [0.8, 0.9],
            "camera_path_extent": [0.99, 1.01],
            "rotation_step_p95_degrees": [0.0, 0.0],
        }
        oracle = {
            "schema_version": "1.0.0",
            "cross_gpu_checksum_allowed": False,
            "invariant_tolerances": {
                "rigid_orthonormal_atol": 1e-6,
            },
            "range_metrics": list(ranges),
            "fixtures": {
                self.FIXTURE_ID: {
                    "status": "ready",
                    "expected_frame_count": 3,
                    "source_sha256": self.SOURCE_SHA,
                    "ranges": ranges,
                }
            },
        }
        path = root / "oracle.json"
        path.write_text(
            json.dumps(oracle, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def test_invariants_and_fixture_ranges_pass_without_output_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            self._write_output(output)
            oracle = self._write_oracle(root)
            result = validate_neural_output(
                output, self.FIXTURE_ID, oracle_path=oracle
            )
            self.assertEqual(result["frame_count"], 3)
            self.assertFalse(result["cross_gpu_checksum_used"])
            self.assertEqual(
                set(result["metrics"]),
                {
                    "depth_p05",
                    "depth_median",
                    "depth_p95",
                    "confidence_p05",
                    "confidence_median",
                    "confidence_p95",
                    "camera_path_extent",
                    "rotation_step_p95_degrees",
                },
            )

    def test_nonfinite_nonrigid_misaligned_and_out_of_range_fail(self):
        mutations = ("nonfinite", "nonrigid", "timestamps", "range")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "output"
                output.mkdir()
                self._write_output(output)
                oracle = self._write_oracle(root)
                if mutation == "nonfinite":
                    depth = np.load(output / "depth.npy", allow_pickle=False)
                    depth[0, 0, 0] = np.nan
                    np.save(output / "depth.npy", depth, allow_pickle=False)
                elif mutation == "nonrigid":
                    cameras = np.load(
                        output / "world_to_camera_opencv.npy",
                        allow_pickle=False,
                    )
                    cameras[1, 0, 0] = 2.0
                    np.save(
                        output / "world_to_camera_opencv.npy",
                        cameras,
                        allow_pickle=False,
                    )
                elif mutation == "timestamps":
                    np.save(
                        output / "source_pts_seconds.npy",
                        np.asarray((0.0, 0.04, 0.04), dtype="<f8"),
                        allow_pickle=False,
                    )
                else:
                    document = load_json(oracle)
                    document["fixtures"][self.FIXTURE_ID]["ranges"][
                        "camera_path_extent"
                    ] = [2.0, 3.0]
                    oracle.write_text(
                        json.dumps(document, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                with self.assertRaises(CorpusValidationError):
                    validate_neural_output(
                        output, self.FIXTURE_ID, oracle_path=oracle
                    )

    def test_production_pending_fixture_refuses_uncalibrated_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(
                CorpusValidationError, "not calibrated"
            ):
                validate_neural_output(
                    output, "real-windowed-above-3000"
                )


if __name__ == "__main__":
    unittest.main()
