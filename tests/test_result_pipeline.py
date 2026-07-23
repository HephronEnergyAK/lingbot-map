from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:  # Blender-side tests intentionally omit the Worker stack.
    np = None


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

if np is not None:
    from lingbot_map_worker.ipc import atomic_write_json
    from lingbot_map_worker.point_reducer import PointCandidate, PointReducer
    from lingbot_map_worker.result_bundle import (
        ArrayContract,
        ResultBundleError,
        ResultCancelled,
        safe_relative_file,
        validate_npy_file,
        validate_result_bundle,
    )
    from lingbot_map_worker.result_pipeline import (
        AlignedPrediction,
        ResultBuildRequest,
        ResultProfile,
        build_reconstruction_result,
    )
    from lingbot_map_worker.result_resources import (
        DISK_RESERVE_BYTES,
        MEMORY_RESERVE_BYTES,
        FixedResourceProbe,
        ResourceGateError,
        require_project_disk,
        require_worker_memory,
        with_headroom,
    )
    from lingbot_map_worker.provenance import result_provenance


SHA = "a" * 64


def _prediction(
    index: int,
    *,
    depth: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    w2c: np.ndarray | None = None,
) -> AlignedPrediction:
    depth = depth if depth is not None else np.array(((1.0, 2.0), (3.0, 4.0)), dtype="<f4")
    confidence = confidence if confidence is not None else np.array(((1.0, 2.0), (3.0, 4.0)), dtype="<f4")
    return AlignedPrediction(
        frame_index=index,
        frame_type=index % 3,
        source_pts_seconds=index * 0.04,
        world_to_camera_opencv=np.ascontiguousarray(
            np.eye(4, dtype="<f8") if w2c is None else w2c
        ),
        model_intrinsics=np.array(((2.0, 0.0, 0.5), (0.0, 2.0, 0.5), (0.0, 0.0, 1.0)), dtype="<f8"),
        depth=np.ascontiguousarray(depth, dtype="<f4"),
        confidence=np.ascontiguousarray(confidence, dtype="<f4"),
        rgb=np.array(
            (((10, 11, 12), (20, 21, 22)), ((30, 31, 32), (40, 41, 42))),
            dtype="|u1",
        ),
    )


class ResultFixture:
    def __init__(self, root: Path, **profile):
        self.root = root
        self.project = root / "target.lingbot-map"
        (self.project / "results").mkdir(parents=True)
        (self.project / "diagnostics").mkdir()
        self.blend = root / "target.blend"
        self.blend.touch()
        self.source_path = root / "capture.mp4"
        self.source_path.write_bytes(b"immutable fixture capture")
        selected_profile = ResultProfile(
            profile.pop("name", "Balanced"),
            profile.pop("confidence_cutoff_percent", 50.0),
            profile.pop("depth_cutoff_percent", 99.5),
            profile.pop("import_point_budget", 8),
            profile.pop("initial_voxel_edge_length", 0.01),
        )
        self.request = ResultBuildRequest(
            job_id="job-" + "1" * 32,
            project_root=self.project,
            target_scene={
                "blend_path": str(self.blend),
                "scene_uuid": "12345678-1234-1234-1234-123456789abc",
                "scene_name": "Scene",
            },
            timeline_start=17,
            source={
                "absolute_path": str(self.source_path),
                "scene_relative_path": "//capture.mp4",
                "size_bytes": self.source_path.stat().st_size,
                "modification_time_ns": self.source_path.stat().st_mtime_ns,
                "sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            },
            source_to_model=np.eye(3, dtype="<f8"),
            predictions=(_prediction(0),),
            profile=selected_profile,
            provenance=result_provenance(
                runtime_id="2" * 64,
                worker_version="0.1.0",
                job_spec_sha256="3" * 64,
                model_id="fixture-model",
                model_sha256="4" * 64,
                source_sha256=hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
                profile_name=selected_profile.name,
                camera_iterations=4,
                confidence_cutoff_percent=selected_profile.confidence_cutoff_percent,
                depth_cutoff_percent=selected_profile.depth_cutoff_percent,
                import_point_budget=selected_profile.import_point_budget,
                plan=None,
                gpu=None,
                suspension_count=0,
                suspension_seconds=0,
                fixture=True,
            ),
            created_utc="2026-07-23T01:02:03+00:00",
            result_id="result-" + "5" * 32,
        )
        if profile:
            raise AssertionError(profile)
        self.probe = FixedResourceProbe(20 * 1024**3, 20 * 1024**3)

    def build(self, cancel=lambda: False):
        return build_reconstruction_result(
            self.request, resource_probe=self.probe, cancel=cancel
        )


@unittest.skipIf(np is None, "Worker NumPy stack is not installed")
class FilteringAndCoordinateTests(unittest.TestCase):
    def test_filtering_precedes_unprojection_and_depth_uses_confidence_survivors(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResultFixture(Path(temporary), confidence_cutoff_percent=50, depth_cutoff_percent=50)
            depth = np.array(((np.nan, -1.0), (2.0, 100.0)), dtype="<f4")
            confidence = np.array(((100.0, 100.0), (1.0, 2.0)), dtype="<f4")
            fixture.request = ResultBuildRequest(
                **{**fixture.request.__dict__, "predictions": (_prediction(0, depth=depth, confidence=confidence),)}
            )
            outcome = fixture.build()
            stats = outcome.filters[0]
            self.assertEqual(
                (stats.valid_depth_count, stats.confidence_retained_count, stats.depth_retained_count),
                (2, 1, 1),
            )
            result = outcome.published.directory
            np.testing.assert_array_equal(np.load(result / "arrays/source_frame.npy"), (0,))
            self.assertTrue(np.isfinite(np.load(result / "arrays/positions.npy")).all())

    def test_reconstruction_frame_intrinsics_and_timestamps_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResultFixture(Path(temporary), confidence_cutoff_percent=0, depth_cutoff_percent=100)
            source_to_model = np.array(
                ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0)),
                dtype="<f8",
            )
            fixture.request = ResultBuildRequest(
                **{**fixture.request.__dict__, "source_to_model": source_to_model}
            )
            outcome = fixture.build()
            result = outcome.published.directory
            camera = np.load(result / "arrays/camera_to_world.npy")
            expected = np.array(
                ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)),
                dtype="<f4",
            )
            np.testing.assert_allclose(camera[0], expected)
            np.testing.assert_allclose(
                source_to_model @ np.load(result / "arrays/source_intrinsics.npy")[0],
                np.load(result / "arrays/model_intrinsics.npy")[0],
            )
            np.testing.assert_array_equal(np.load(result / "arrays/source_pts_seconds.npy"), (0.0,))
            positions = np.load(result / "arrays/positions.npy")
            self.assertTrue(bool((positions[:, 1] > 0).all()), "OpenCV forward must become Reconstruction +Y")
            validate_result_bundle(result)


@unittest.skipIf(np is None, "Worker NumPy stack is not installed")
class PointReducerTests(unittest.TestCase):
    def test_exact_doubling_ties_and_lexicographic_output(self):
        reducer = PointReducer(2, initial_edge_length=1.0)
        reducer.add_many(
            (
                PointCandidate((2.1, 0.1, 0.1), (1, 1, 1), 0.5, 1, 0),
                PointCandidate((0.1, 0.1, 0.1), (2, 2, 2), 0.8, 1, 7),
                PointCandidate((1.1, 0.1, 0.1), (3, 3, 3), 0.8, 0, 9),
            )
        )
        reduced = reducer.finish()
        self.assertEqual(reducer.maximum_occupied_entries, 3)
        self.assertEqual(reduced.edge_length, 2.0)
        self.assertEqual(reduced.voxel_coordinates, ((0, 0, 0), (1, 0, 0)))
        self.assertEqual(reduced.source_frame.tolist(), [0, 1])
        self.assertEqual(reduced.colors.tolist(), [[3, 3, 3], [1, 1, 1]])
        self.assertEqual(reduced.radius.tolist(), [1.0, 1.0])

    def test_never_holds_more_than_budget_plus_one_entries(self):
        reducer = PointReducer(8, initial_edge_length=0.001)
        for pixel in range(500):
            reducer.add(PointCandidate((pixel + 0.1, 0.1, 0.1), (0, 0, 0), 1.0, 0, pixel))
        self.assertLessEqual(reducer.maximum_occupied_entries, 9)
        self.assertLessEqual(reducer.occupied_entries, 8)


@unittest.skipIf(np is None, "Worker NumPy stack is not installed")
class ResourceGateTests(unittest.TestCase):
    def test_disk_and_physical_memory_boundaries_are_exact(self):
        estimate = 17
        disk_required = with_headroom(estimate) + DISK_RESERVE_BYTES
        memory_required = with_headroom(estimate) + MEMORY_RESERVE_BYTES
        self.assertEqual(
            require_project_disk(FixedResourceProbe(disk_required, 0), Path.cwd(), estimate),
            disk_required,
        )
        self.assertEqual(
            require_worker_memory(FixedResourceProbe(0, memory_required), estimate),
            memory_required,
        )
        with self.assertRaises(ResourceGateError):
            require_project_disk(FixedResourceProbe(disk_required - 1, 0), Path.cwd(), estimate)
        with self.assertRaises(ResourceGateError):
            require_worker_memory(FixedResourceProbe(0, memory_required - 1), estimate)

    def test_gate_failure_leaves_no_partial_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResultFixture(Path(temporary))
            fixture.probe.disk_bytes = 0
            with self.assertRaises(ResourceGateError):
                fixture.build()
            self.assertEqual(list((fixture.project / "results").iterdir()), [])


@unittest.skipIf(np is None, "Worker NumPy stack is not installed")
class SafeBundleTests(unittest.TestCase):
    def test_paths_reject_traversal_absolute_drive_unc_and_backslash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in ("../x", "/x", "C:/x", "//host/share", "arrays\\x.npy", "."):
                with self.subTest(value=value), self.assertRaises(ResultBundleError):
                    safe_relative_file(root, value)

    def test_npy_rejects_fortran_object_wrong_dtype_shape_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "fortran.npy": np.asfortranarray(np.ones((2, 3), dtype="<f4")),
                "object.npy": np.array([object()], dtype=object),
                "dtype.npy": np.ones((2, 3), dtype="<f8"),
                "shape.npy": np.ones((3, 2), dtype="<f4"),
            }
            for name, array in cases.items():
                path = root / name
                np.save(path, array, allow_pickle=True)
                with self.subTest(name=name), self.assertRaises(ResultBundleError):
                    validate_npy_file(path, ArrayContract("<f4", (2, 3)))
            trailing = root / "trailing.npy"
            np.save(trailing, np.ones((2, 3), dtype="<f4"), allow_pickle=False)
            with trailing.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaises(ResultBundleError):
                validate_npy_file(trailing, ArrayContract("<f4", (2, 3)))

    def test_oversized_npy_header_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hostile.npy"
            path.write_bytes(b"\x93NUMPY\x02\x00" + struct.pack("<I", 65537))
            with self.assertRaisesRegex(ResultBundleError, "64 KiB"):
                validate_npy_file(path, ArrayContract("<f4", (1,)))

    def test_manifest_checksum_cross_semantics_and_undeclared_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResultFixture(Path(temporary))
            result = fixture.build().published.directory
            manifest_path = result / "manifest.json"
            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            variants = []
            checksum = copy.deepcopy(original)
            checksum["arrays"]["positions"]["sha256"] = "0" * 64
            variants.append(checksum)
            provenance = copy.deepcopy(original)
            provenance["provenance"]["source_sha256"] = "0" * 64
            variants.append(provenance)
            for document in variants:
                atomic_write_json(manifest_path, document)
                with self.assertRaises(ResultBundleError):
                    validate_result_bundle(result)
            atomic_write_json(manifest_path, original)
            (result / "undeclared.bin").write_bytes(b"x")
            with self.assertRaisesRegex(ResultBundleError, "undeclared"):
                validate_result_bundle(result)

    def test_failure_and_precommit_cancel_publish_no_result_but_retain_diagnostics(self):
        for mode in ("cancel", "failure"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = ResultFixture(Path(temporary))
                if mode == "cancel":
                    with self.assertRaises(ResultCancelled):
                        fixture.build(cancel=lambda: True)
                    # Cancellation before construction creates no staging content.
                    self.assertEqual(list((fixture.project / "diagnostics").iterdir()), [])
                else:
                    calls = 0
                    def cancel_late():
                        nonlocal calls
                        calls += 1
                        return calls >= 4
                    with self.assertRaises(ResultCancelled):
                        fixture.build(cancel=cancel_late)
                    diagnostic = next((fixture.project / "diagnostics").iterdir())
                    self.assertTrue((diagnostic / "failure.json").is_file())
                self.assertEqual(list((fixture.project / "results").iterdir()), [])

    def test_commit_is_one_same_parent_rename_and_late_cancel_cannot_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResultFixture(Path(temporary))
            real_replace = os.replace
            committed = False
            result_renames = []
            def observed_replace(source, destination):
                nonlocal committed
                source_path, destination_path = Path(source), Path(destination)
                if source_path.parent == fixture.project / "results":
                    result_renames.append((source_path, destination_path))
                    value = real_replace(source, destination)
                    committed = True
                    return value
                return real_replace(source, destination)
            with mock.patch("lingbot_map_worker.result_bundle.os.replace", side_effect=observed_replace):
                outcome = fixture.build(cancel=lambda: committed)
            self.assertEqual(len(result_renames), 1)
            self.assertEqual(result_renames[0][0].parent, result_renames[0][1].parent)
            self.assertTrue(outcome.published.directory.is_dir())
            self.assertEqual(list((fixture.project / "diagnostics").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
