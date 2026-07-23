from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
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
    from lingbot_map_worker.dense_predictions import (
        DensePredictionWriter,
        DensePredictionsError,
        DensePredictionsIncompatible,
        validate_dense_component,
    )
    from lingbot_map_worker.result_pipeline import (
        AlignedPrediction,
        IncrementalBundleResultSink,
        IncrementalResultRequest,
        ResultBuildRequest,
        ResultProfile,
        _filter_frame,
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
    from lingbot_map_worker.sky_masking import SkyMaskError, SkyMaskOutcome


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

    def test_sky_eligibility_precedes_confidence_and_depth_percentiles(self):
        frame = _prediction(
            0,
            depth=np.array(((1.0, 2.0), (3.0, 4.0)), dtype="<f4"),
            confidence=np.array(((1.0, 2.0), (100.0, 200.0)), dtype="<f4"),
        )
        eligible = np.array(((1, 1), (0, 0)), dtype="|u1")
        retained, statistics = _filter_frame(
            frame,
            ResultProfile("Custom", 50, 100, 8, 0.01),
            eligible,
        )
        np.testing.assert_array_equal(retained, ((False, True), (False, False)))
        self.assertEqual(statistics.valid_depth_count, 2)
        self.assertEqual(statistics.confidence_threshold, 1.5)

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


class FakeSkyMaskSession:
    def __init__(self, masks, *, fail_frame=None):
        self.masks = tuple(masks)
        self.fail_frame = fail_frame
        self.prepared = []
        self.consumed = []
        self.aborted = False

    def prepare(self, frame_index, canonical):
        self.prepared.append((frame_index, canonical))

    def mask_for(self, frame_index, target_shape):
        if frame_index == self.fail_frame:
            raise SkyMaskError(f"missing Sky Mask frame {frame_index}")
        mask = self.masks[frame_index]
        if mask.shape != target_shape:
            raise SkyMaskError("shape mismatch")
        self.consumed.append(frame_index)
        fraction = np.float32(1.0 - np.count_nonzero(mask) / mask.size)
        return mask, fraction

    def finish(self):
        if len(self.consumed) != len(self.masks):
            raise SkyMaskError("incomplete Sky Mask set")
        fractions = np.ascontiguousarray(
            [1.0 - np.count_nonzero(mask) / mask.size for mask in self.masks],
            dtype="<f4",
        )
        return SkyMaskOutcome(
            fractions,
            {
                "enabled": True,
                "model_id": "skyseg",
                "model_sha256": "9" * 64,
                "rule_version": "non-sky-confidence-gt-0.1-v1",
                "preprocessing_version": "skyseg-imagenet-320-bilinear-v1",
                "provider": "CPUExecutionProvider",
                "onnxruntime_version": "1.23.2",
                "batch_size": 1,
                "onnx_threads": 1,
                "cache_key": "8" * 64,
                "cache_status": "generated",
            },
            (),
        )

    def abort(self):
        self.aborted = True


@unittest.skipIf(np is None, "Worker NumPy stack is not installed")
class SkyMaskResultTests(unittest.TestCase):
    def _sink(self, root: Path, sky_session):
        project = root / "target.lingbot-map"
        (project / "results").mkdir(parents=True)
        (project / "diagnostics").mkdir()
        blend = root / "target.blend"
        blend.touch()
        source = root / "capture.mp4"
        source.write_bytes(b"sky-result-fixture")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        provenance = result_provenance(
            runtime_id="2" * 64,
            worker_version="0.1.0",
            job_spec_sha256="3" * 64,
            model_id="fixture-model",
            model_sha256="4" * 64,
            source_sha256=source_sha,
            profile_name="Custom",
            camera_iterations=1,
            confidence_cutoff_percent=50,
            depth_cutoff_percent=100,
            import_point_budget=8,
            plan=None,
            gpu=None,
            suspension_count=0,
            suspension_seconds=0,
            fixture=True,
        )
        request = IncrementalResultRequest(
            job_id="job-" + "7" * 32,
            project_root=project,
            target_scene={
                "blend_path": str(blend),
                "scene_uuid": "12345678-1234-1234-1234-123456789abc",
                "scene_name": "Scene",
            },
            timeline_start=1,
            source={
                "absolute_path": str(source),
                "scene_relative_path": "//capture.mp4",
                "size_bytes": source.stat().st_size,
                "modification_time_ns": source.stat().st_mtime_ns,
                "sha256": source_sha,
            },
            source_to_model=np.eye(3, dtype="<f8"),
            frame_count=2,
            profile=ResultProfile("Custom", 50, 100, 8, 0.01),
            provenance=provenance,
            created_utc="2026-07-23T07:00:00+00:00",
            result_id="result-" + "8" * 32,
            model_grid_shape=(2, 2),
        )
        return (
            IncrementalBundleResultSink(
                request,
                prediction_decoder=lambda prediction, _canonical, _pts: prediction,
                resource_probe=FixedResourceProbe(20 * 1024**3, 20 * 1024**3),
                cancel=lambda: False,
                sky_mask_session=sky_session,
            ),
            project,
        )

    def test_result_contains_only_sky_fractions_summary_and_provenance(self):
        masks = (
            np.array(((1, 1), (0, 0)), dtype="|u1"),
            np.array(((0, 0), (0, 0)), dtype="|u1"),
        )
        sky = FakeSkyMaskSession(masks)
        with tempfile.TemporaryDirectory() as temporary:
            sink, _project = self._sink(Path(temporary), sky)
            for index in range(2):
                canonical = SimpleNamespace(color_rgb=np.zeros((2, 2, 3), dtype="|u1"))
                sink.prepare_frame(index, canonical)
                sink.accept(_prediction(index), canonical, index * 0.04)
            outcome = sink.finish()
            result = outcome.published.directory
            manifest = validate_result_bundle(result)
            self.assertEqual(sky.prepared[0][0], 0)
            self.assertEqual(sky.consumed, [0, 1])
            np.testing.assert_allclose(
                np.load(result / "arrays" / "sky_fraction.npy"), (0.5, 1.0)
            )
            self.assertEqual(
                manifest["sky_statistics"]["count_above_95_percent"], 1
            )
            self.assertEqual(
                manifest["provenance"]["sky_masking"]["cache_key"], "8" * 64
            )
            self.assertEqual(
                {item["role"] for item in manifest["provenance"]["models"]},
                {"fixture", "auxiliary"},
            )
            self.assertFalse(any("mask" in path.name for path in result.rglob("*.npy") if path.name != "sky_fraction.npy"))

    def test_missing_frame_fails_before_result_publication(self):
        sky = FakeSkyMaskSession(
            (
                np.ones((2, 2), dtype="|u1"),
                np.ones((2, 2), dtype="|u1"),
            ),
            fail_frame=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            sink, project = self._sink(Path(temporary), sky)
            canonical = SimpleNamespace(color_rgb=np.zeros((2, 2, 3), dtype="|u1"))
            sink.prepare_frame(0, canonical)
            sink.accept(_prediction(0), canonical, 0.0)
            sink.prepare_frame(1, canonical)
            with self.assertRaisesRegex(SkyMaskError, "missing Sky Mask"):
                sink.accept(_prediction(1), canonical, 0.04)
            sink.abort("failed")
            self.assertTrue(sky.aborted)
            self.assertFalse(any((project / "results").iterdir()))


@unittest.skipIf(np is None, "Worker NumPy stack is not installed")
class DensePredictionTests(unittest.TestCase):
    def _provenance(self, source_sha: str, *, retained: bool = True):
        return result_provenance(
            runtime_id="2" * 64,
            worker_version="0.1.0",
            job_spec_sha256="3" * 64,
            model_id="fixture-model",
            model_sha256="4" * 64,
            source_sha256=source_sha,
            profile_name="Custom" if retained else "Draft",
            camera_iterations=1,
            confidence_cutoff_percent=70,
            depth_cutoff_percent=99.5,
            import_point_budget=8,
            plan=None,
            gpu=None,
            suspension_count=0,
            suspension_seconds=0,
            fixture=True,
            retain_dense_predictions=retained,
        )

    def _dense_result(
        self, root: Path, frame_count: int = 65, *, cancel_on_finish: bool = False
    ):
        project = root / "target.lingbot-map"
        (project / "results").mkdir(parents=True)
        (project / "diagnostics").mkdir()
        blend = root / "target.blend"
        blend.touch()
        source = root / "capture.mp4"
        source.write_bytes(b"dense-fixture")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        request = IncrementalResultRequest(
            job_id="job-" + "d" * 32,
            project_root=project,
            target_scene={
                "blend_path": str(blend),
                "scene_uuid": "12345678-1234-1234-1234-123456789abc",
                "scene_name": "Scene",
            },
            timeline_start=1,
            source={
                "absolute_path": str(source),
                "scene_relative_path": "//capture.mp4",
                "size_bytes": source.stat().st_size,
                "modification_time_ns": source.stat().st_mtime_ns,
                "sha256": source_sha,
            },
            source_to_model=np.eye(3, dtype="<f8"),
            frame_count=frame_count,
            profile=ResultProfile("Custom", 70, 99.5, 8, 0.01, True),
            provenance=self._provenance(source_sha),
            created_utc="2026-07-23T06:00:00+00:00",
            result_id="result-" + "e" * 32,
            model_grid_shape=(2, 2),
        )
        sink = IncrementalBundleResultSink(
            request,
            prediction_decoder=lambda prediction, _canonical, _pts: prediction,
            resource_probe=FixedResourceProbe(20 * 1024**3, 20 * 1024**3),
            cancel=lambda: cancel_on_finish and sink.finalized_frame_count == frame_count,
        )
        for index in range(frame_count):
            sink.accept(_prediction(index), None, index * 0.04)
        return sink.finish(), project

    def test_writer_uses_exact_64_frame_boundaries_and_separate_float32_signals(self):
        for frame_count, expected in ((64, [64]), (65, [64, 1]), (128, [64, 64])):
            with self.subTest(frame_count=frame_count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project = root / "target.lingbot-map"
                (project / "results").mkdir(parents=True)
                (project / "diagnostics").mkdir()
                writer = DensePredictionWriter(
                    project_root=project,
                    job_id="job-" + "a" * 32,
                    frame_count=frame_count,
                    grid_shape=(2, 3),
                    disk_check=lambda _remaining: None,
                )
                for index in range(frame_count):
                    writer.accept(
                        index,
                        np.full((2, 3), index + 1, dtype="<f4"),
                        np.full((2, 3), index + 2, dtype="<f4"),
                    )
                    self.assertLessEqual(writer.buffered_frames, 63)
                artifact = writer.finish(self._provenance("5" * 64))
                manifest = json.loads(
                    (artifact.staging_directory / "manifest.json").read_text(encoding="utf-8")
                )
                for name in ("depth", "depth_confidence"):
                    chunks = manifest["signals"][name]["chunks"]
                    self.assertEqual([chunk["frame_count"] for chunk in chunks], expected)
                    self.assertTrue(all(chunk["dtype"] == "<f4" for chunk in chunks))
                    self.assertEqual(manifest["signals"][name]["shape"], [frame_count, 2, 3])

    def test_core_validation_survives_missing_or_corrupt_dense_content(self):
        for mutation in ("missing-manifest", "corrupt-chunk"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                outcome, _project = self._dense_result(Path(temporary))
                result = outcome.published.directory
                descriptor = outcome.published.manifest["dense_predictions"]
                validate_dense_component(result, descriptor)
                if mutation == "missing-manifest":
                    (result / "dense" / "manifest.json").unlink()
                else:
                    chunk = next((result / "dense" / "depth").glob("*.npy"))
                    with chunk.open("ab") as stream:
                        stream.write(b"corrupt")
                validate_result_bundle(result)
                with self.assertRaises(DensePredictionsError):
                    validate_dense_component(result, descriptor)

    def test_dense_schema_compatibility_is_independent_from_core_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            outcome, _project = self._dense_result(Path(temporary), frame_count=8)
            result = outcome.published.directory
            manifest_path = result / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dense_predictions"]["schema_version"] = "2.0.0"
            atomic_write_json(manifest_path, manifest)
            validate_result_bundle(result)
            with self.assertRaises(DensePredictionsIncompatible):
                validate_dense_component(result, manifest["dense_predictions"])

    def test_disk_exhaustion_and_cancellation_publish_no_partial_dense_component(self):
        class ExhaustingProbe(FixedResourceProbe):
            def __init__(self):
                super().__init__(20 * 1024**3, 20 * 1024**3)
                self.calls = 0
            def available_disk_bytes(self, path):
                self.calls += 1
                # Constructor + 64 per-frame core checks pass; the first chunk write fails.
                return super().available_disk_bytes(path) if self.calls <= 66 else 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "target.lingbot-map"
            (project / "results").mkdir(parents=True)
            (project / "diagnostics").mkdir()
            blend = root / "target.blend"
            blend.touch()
            source = root / "capture.mp4"
            source.write_bytes(b"disk-fixture")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            request = IncrementalResultRequest(
                job_id="job-" + "f" * 32,
                project_root=project,
                target_scene={"blend_path": str(blend), "scene_uuid": "12345678-1234-1234-1234-123456789abc", "scene_name": "Scene"},
                timeline_start=1,
                source={"absolute_path": str(source), "scene_relative_path": "//capture.mp4", "size_bytes": source.stat().st_size, "modification_time_ns": source.stat().st_mtime_ns, "sha256": source_sha},
                source_to_model=np.eye(3, dtype="<f8"),
                frame_count=64,
                profile=ResultProfile("Custom", 70, 99.5, 8, 0.01, True),
                provenance=self._provenance(source_sha),
                model_grid_shape=(2, 2),
            )
            sink = IncrementalBundleResultSink(
                request,
                prediction_decoder=lambda prediction, _canonical, _pts: prediction,
                resource_probe=ExhaustingProbe(),
                cancel=lambda: False,
            )
            with self.assertRaises(ResourceGateError):
                for index in range(64):
                    sink.accept(_prediction(index), None, index * 0.04)
            diagnostic = sink.abort("failed")
            self.assertIsNotNone(diagnostic)
            self.assertFalse(any((project / "results").iterdir()))
            self.assertEqual(
                json.loads((diagnostic / "incomplete.json").read_text(encoding="utf-8"))["completion_state"],
                "incomplete",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ResultCancelled):
                self._dense_result(root, frame_count=8, cancel_on_finish=True)
            project = root / "target.lingbot-map"
            self.assertFalse(any((project / "results").iterdir()))
            self.assertTrue(any((project / "diagnostics").iterdir()))


if __name__ == "__main__":
    unittest.main()
