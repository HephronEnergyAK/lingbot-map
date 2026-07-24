"""Validate the versioned release corpus and all deterministic oracle layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

MAX_JSON_BYTES = 2 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}")
ORDINARY_IDS = {
    "synthetic-boundary-8",
    "synthetic-boundary-320",
    "synthetic-boundary-321",
    "synthetic-boundary-3000",
    "synthetic-boundary-3001",
    "synthetic-vfr-8",
    "synthetic-color-bt709-limited",
    "synthetic-color-bt709-full",
    "synthetic-color-bt601-limited",
    "synthetic-color-bt601-full",
    "real-courthouse-streaming-286",
}


class CorpusValidationError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    if not _ordinary_file(path):
        raise CorpusValidationError(f"JSON must be one ordinary file: {path}")
    data = path.read_bytes()
    if len(data) > MAX_JSON_BYTES:
        raise CorpusValidationError(f"JSON document is oversized: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusValidationError(f"JSON is not UTF-8: {path}") from exc
    if text.startswith("\ufeff"):
        raise CorpusValidationError(f"JSON must not contain a BOM: {path}")

    def reject_constant(value: str) -> None:
        raise CorpusValidationError(f"JSON contains non-finite number: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CorpusValidationError(f"invalid JSON: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse = bool(
            attributes
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        return path.is_file() and not path.is_symlink() and not reparse
    except OSError:
        return False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_exact_keys(
    value: Any, expected: Iterable[str], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise CorpusValidationError(
            f"{label} fields differ: expected {sorted(expected)!r}"
        )
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise CorpusValidationError(f"{label} is not a lowercase SHA-256")
    return value


def _ordered_source_manifest_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _test_id_exists(test_id: str) -> bool:
    parts = test_id.split(".")
    if len(parts) < 4 or parts[0] != "tests":
        return False
    module_path = ROOT.joinpath(*parts[:-2]).with_suffix(".py")
    if not module_path.is_file():
        return False
    text = module_path.read_text(encoding="utf-8")
    class_name, method_name = parts[-2:]
    return (
        re.search(rf"^class {re.escape(class_name)}\b", text, re.MULTILINE)
        is not None
        and re.search(
            rf"^\s+def {re.escape(method_name)}\s*\(",
            text,
            re.MULTILINE,
        )
        is not None
    )


def _walk_test_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "test_id":
                if not isinstance(child, str):
                    raise CorpusValidationError("test_id must be a string")
                yield child
            elif key == "test_ids":
                if not isinstance(child, list) or not all(
                    isinstance(item, str) for item in child
                ):
                    raise CorpusValidationError(
                        "test_ids must be an array of strings"
                    )
                yield from child
            else:
                yield from _walk_test_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_test_ids(child)


def _validate_schema_catalog(exact: dict[str, Any]) -> None:
    oracle = exact["schema_catalog"]
    catalog_path = ROOT / oracle["path"]
    catalog = load_json(catalog_path)
    if catalog.get("catalog_version") != oracle["catalog_version"]:
        raise CorpusValidationError("schema catalog version drifted")
    contracts = catalog.get("contracts")
    if not isinstance(contracts, list) or len(contracts) != oracle["contract_count"]:
        raise CorpusValidationError("schema catalog contract count drifted")
    seen = set()
    for contract in contracts:
        if not isinstance(contract, dict):
            raise CorpusValidationError("schema contract entry is not an object")
        identifier = contract.get("id")
        if not isinstance(identifier, str) or identifier in seen:
            raise CorpusValidationError("schema contract id is invalid or duplicated")
        seen.add(identifier)
        path = ROOT / "schemas" / str(contract.get("path"))
        expected = _require_sha(
            contract.get("sha256"), label=f"schema {identifier}"
        )
        if not path.is_file() or _sha256(path) != expected:
            raise CorpusValidationError(
                f"schema contract checksum drifted: {identifier}"
            )
        runtime_copy = (
            ROOT
            / "blender_extension"
            / "runtime_bundle"
            / "schemas"
            / path.name
        )
        if not runtime_copy.is_file() or _sha256(runtime_copy) != expected:
            raise CorpusValidationError(
                f"bundled schema checksum drifted: {identifier}"
            )


def _validate_exact_oracles(exact: dict[str, Any]) -> None:
    try:
        import numpy as np
        from lingbot_map_worker.canonical_preprocessing import canonicalize_srgb
        from lingbot_map_worker.decoder import TRANSFORMS, apply_display_transform
        from lingbot_map_worker.gpu_profiles import inference_plan
        from lingbot_map_worker.point_reducer import PointCandidate, PointReducer
        from lingbot_map_worker.result_pipeline import (
            AlignedPrediction,
            _camera_arrays,
            _normalization,
        )
    except ModuleNotFoundError as exc:
        raise CorpusValidationError(
            "exact oracle validation requires the pinned Worker Runtime"
        ) from exc

    display_input = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    by_name = {transform.name: (matrix, transform) for matrix, transform in TRANSFORMS.items()}
    if len(by_name) != 8:
        raise CorpusValidationError("decoder does not expose exactly eight transforms")
    for oracle in exact["display_transforms"]:
        name = oracle["name"]
        if name not in by_name:
            raise CorpusValidationError(f"unknown Display Transform oracle: {name}")
        matrix, transform = by_name[name]
        if list(matrix) != oracle["matrix"]:
            raise CorpusValidationError(f"Display Transform matrix drifted: {name}")
        output = np.ascontiguousarray(
            apply_display_transform(display_input, transform)
        )
        if list(output.shape) != oracle["shape"] or _sha256_bytes(
            output.tobytes(order="C")
        ) != oracle["sha256"]:
            raise CorpusValidationError(f"Display Transform pixels drifted: {name}")

    for oracle in exact["inference_plans"]:
        plan = inference_plan(oracle["frame_count"])
        if (plan.mode, plan.keyframe_interval) != (
            oracle["mode"],
            oracle["keyframe_interval"],
        ):
            raise CorpusValidationError(
                f"inference plan drifted at {oracle['frame_count']} frames"
            )

    preprocessing = exact["canonical_preprocessing"]
    canonical = canonicalize_srgb(
        np.arange(96, dtype=np.uint8).reshape(4, 8, 3)
    )
    if (
        list(canonical.color_rgb.shape) != preprocessing["color_shape"]
        or list(canonical.model_input.shape) != preprocessing["model_shape"]
        or _sha256_bytes(canonical.color_rgb.tobytes(order="C"))
        != preprocessing["color_sha256"]
        or _sha256_bytes(canonical.model_input.tobytes(order="C"))
        != preprocessing["model_sha256"]
        or not np.array_equal(
            canonical.source_to_model,
            np.asarray(preprocessing["source_to_model"], dtype="<f8"),
        )
        or [list(point) for point in canonical.coverage_polygon]
        != preprocessing["coverage_polygon"]
    ):
        raise CorpusValidationError("canonical preprocessing oracle drifted")

    coordinate = exact["coordinate_conversion"]
    prediction = AlignedPrediction(
        frame_index=0,
        frame_type=0,
        source_pts_seconds=0.0,
        world_to_camera_opencv=np.asarray(
            coordinate["world_to_camera_opencv"], dtype="<f8"
        ),
        model_intrinsics=np.asarray(
            coordinate["model_intrinsics"], dtype="<f8"
        ),
        depth=np.ones((2, 2), dtype="<f4"),
        confidence=np.ones((2, 2), dtype="<f4"),
        rgb=np.zeros((2, 2, 3), dtype="|u1"),
    )
    normalization, opencv_c2w = _normalization((prediction,))
    arrays = _camera_arrays(
        (prediction,),
        normalization,
        opencv_c2w,
        np.asarray(coordinate["source_to_model"], dtype="<f8"),
        2,
        2,
    )
    if not np.array_equal(
        arrays["camera_to_world"][0],
        np.asarray(coordinate["camera_to_world_reconstruction"], dtype="<f4"),
    ) or not np.array_equal(
        arrays["source_intrinsics"][0],
        np.asarray(coordinate["source_intrinsics"], dtype="<f4"),
    ):
        raise CorpusValidationError("coordinate conversion oracle drifted")

    reducer_oracle = exact["point_reducer"]
    reducer = PointReducer(
        reducer_oracle["budget"],
        initial_edge_length=reducer_oracle["initial_edge_length"],
        origin=tuple(reducer_oracle["origin"]),
    )
    reducer.add_many(
        PointCandidate(
            tuple(item["position"]),
            tuple(item["color"]),
            item["confidence"],
            item["source_frame"],
            item["pixel_index"],
        )
        for item in reducer_oracle["candidates"]
    )
    reduced = reducer.finish()
    reducer_values = {
        "edge_length": reduced.edge_length,
        "maximum_occupied_entries": reducer.maximum_occupied_entries,
        "voxel_coordinates": [list(value) for value in reduced.voxel_coordinates],
        "positions": reduced.positions.tolist(),
        "colors": reduced.colors.tolist(),
        "confidence": reduced.confidence.tolist(),
        "source_frame": reduced.source_frame.tolist(),
        "radius": reduced.radius.tolist(),
    }
    for key, value in reducer_values.items():
        if value != reducer_oracle[key]:
            raise CorpusValidationError(f"Point Reducer oracle drifted: {key}")

    _validate_schema_catalog(exact)
    for test_id in _walk_test_ids(exact["safe_bundle"]):
        if not _test_id_exists(test_id):
            raise CorpusValidationError(f"safe-bundle test id is stale: {test_id}")


def _contains_key(value: Any, pattern: re.Pattern[str]) -> bool:
    if isinstance(value, dict):
        return any(
            pattern.search(str(key)) or _contains_key(child, pattern)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(child, pattern) for child in value)
    return False


def _validate_neural_oracles(neural: dict[str, Any]) -> None:
    if neural.get("schema_version") != "1.0.0":
        raise CorpusValidationError("neural oracle schema version drifted")
    if neural.get("cross_gpu_checksum_allowed") is not False:
        raise CorpusValidationError("cross-GPU neural checksum must be forbidden")
    if neural.get("invariant_tolerances") != {
        "rigid_orthonormal_atol": 1e-6
    }:
        raise CorpusValidationError("neural invariant tolerances drifted")
    if _contains_key(
        neural.get("fixtures"),
        re.compile(r"(cross.?gpu|output).*(sha|checksum)", re.IGNORECASE),
    ):
        raise CorpusValidationError("neural fixture contains a cross-GPU checksum")
    metrics = neural.get("range_metrics")
    fixtures = neural.get("fixtures")
    if (
        not isinstance(metrics, list)
        or len(metrics) != len(set(metrics))
        or not isinstance(fixtures, dict)
        or not fixtures
    ):
        raise CorpusValidationError("neural range catalog is malformed")
    for fixture_id, fixture in fixtures.items():
        status = fixture.get("status")
        ranges = fixture.get("ranges")
        if status == "ready":
            if not isinstance(ranges, dict) or set(ranges) != set(metrics):
                raise CorpusValidationError(
                    f"ready neural fixture lacks exact metric ranges: {fixture_id}"
                )
            for metric, bounds in ranges.items():
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or any(
                        isinstance(bound, bool)
                        or not isinstance(bound, (int, float))
                        or not math.isfinite(float(bound))
                        for bound in bounds
                    )
                    or float(bounds[0]) > float(bounds[1])
                ):
                    raise CorpusValidationError(
                        f"ready neural fixture has malformed range: "
                        f"{fixture_id}: {metric}"
                    )
            calibration = fixture.get("calibration")
            if (
                not isinstance(calibration, dict)
                or set(calibration.get("observed_metrics", {})) != set(metrics)
                or "qualified_hardware_evidence" not in calibration
                or "envelope_policy" not in calibration
            ):
                raise CorpusValidationError(
                    f"ready neural fixture lacks calibration evidence: {fixture_id}"
                )
        elif ranges is not None:
            raise CorpusValidationError(
                f"pending neural fixture must not invent ranges: {fixture_id}"
            )


def _validate_generated(
    generated_path: Path,
    goldens: dict[str, Any],
    *,
    require_stress: bool,
) -> set[str]:
    generated = load_json(generated_path)
    _require_exact_keys(
        generated,
        {
            "schema_version",
            "generator_id",
            "toolchain",
            "stress_included",
            "personal_capture_sources",
            "fixtures",
        },
        label="generated manifest",
    )
    if (
        generated["schema_version"] != "1.0.0"
        or generated["generator_id"] != goldens["generator_id"]
        or generated["toolchain"] != goldens["toolchain"]
        or generated["personal_capture_sources"] is not False
    ):
        raise CorpusValidationError("generated corpus identity or toolchain drifted")
    expected = {item["id"]: item for item in goldens["fixtures"]}
    actual_ids: set[str] = set()
    for record in generated["fixtures"]:
        fixture_id = record.get("id")
        if fixture_id in actual_ids or fixture_id not in expected:
            raise CorpusValidationError(
                f"generated fixture id is unknown or duplicated: {fixture_id}"
            )
        actual_ids.add(fixture_id)
        for key, value in expected[fixture_id].items():
            if record.get(key) != value:
                raise CorpusValidationError(
                    f"generated fixture drifted: {fixture_id}.{key}"
                )
        media = generated_path.parent / record["file"]
        if (
            not _ordinary_file(media)
            or media.stat().st_size != record["bytes"]
            or _sha256(media) != record["sha256"]
        ):
            raise CorpusValidationError(
                f"generated fixture file is absent or changed: {fixture_id}"
            )
    if not ORDINARY_IDS.issubset(actual_ids):
        missing = sorted(ORDINARY_IDS - actual_ids)
        raise CorpusValidationError(
            f"ordinary generated corpus is incomplete: {missing!r}"
        )
    has_stress = "synthetic-stress-25000" in actual_ids
    if generated["stress_included"] is not has_stress:
        raise CorpusValidationError("stress flag disagrees with generated fixtures")
    if require_stress and not has_stress:
        raise CorpusValidationError("milestone validation requires stress fixture")
    return actual_ids


def validate(
    *,
    generated_manifest: Path | None = None,
    require_stress: bool = False,
    release: bool = False,
) -> dict[str, Any]:
    manifest = load_json(ROOT / "release_corpus" / "manifest.json")
    exact = load_json(ROOT / manifest["oracles"]["exact"])
    neural = load_json(ROOT / manifest["oracles"]["neural"])
    goldens = load_json(ROOT / manifest["oracles"]["generated_decoder"])
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("privacy", {}).get("personal_capture_sources") is not False
        or manifest.get("storage", {}).get("checked_in_media") is not False
    ):
        raise CorpusValidationError("corpus privacy or version contract drifted")
    coverage = manifest["synthetic_coverage"]
    if coverage["boundary_frame_counts"] != [8, 320, 321, 3000, 3001]:
        raise CorpusValidationError("boundary frame-count inventory drifted")
    if len(coverage["display_transforms"]) != 8 or len(
        set(coverage["display_transforms"])
    ) != 8:
        raise CorpusValidationError("Display Transform inventory is incomplete")
    colors = {
        (item["standard"], item["range"])
        for item in coverage["color_descriptions"]
    }
    if colors != {
        ("bt709", "limited"),
        ("bt709", "full"),
        ("bt601", "limited"),
        ("bt601", "full"),
    }:
        raise CorpusValidationError("accepted color inventory drifted")
    rejected = {item["id"] for item in coverage["rejected_media_classes"]}
    required_rejected = {
        "interlaced",
        "alpha",
        "non-square-pixels",
        "unsupported-hdr",
        "changing-stream-properties",
        "ambiguous-video-tracks",
        "corrupt-frame",
        "non-monotonic-timestamp",
        "fewer-than-eight-frames",
        "unsupported-display-matrix",
        "edited-take-declaration",
    }
    if rejected != required_rejected:
        raise CorpusValidationError("rejected media-class inventory drifted")
    for test_id in set(_walk_test_ids(manifest)):
        if not _test_id_exists(test_id):
            raise CorpusValidationError(f"corpus test id is stale: {test_id}")

    real = {item["id"]: item for item in manifest["real_captures"]}
    courthouse = real["real-courthouse-streaming-286"]
    source_paths = tuple((ROOT / "example" / "courthouse").glob("*.png"))
    if (
        courthouse["status"] != "ready"
        or courthouse["classification"] != "streaming"
        or courthouse["presentation_frames"] != 286
        or len(source_paths) != 286
        or _ordered_source_manifest_hash(source_paths)
        != courthouse["provenance"]["ordered_name_and_sha256_manifest"]
        or _sha256(ROOT / courthouse["license"]["evidence_path"])
        != courthouse["license"]["evidence_sha256"]
        or courthouse["license"]["spdx"] != "Apache-2.0"
    ):
        raise CorpusValidationError(
            "real courthouse provenance or license evidence drifted"
        )
    stress = manifest["stress_capture"]
    if (
        stress["presentation_frames"] != 25000
        or stress["cadence"] != "release-candidate-or-milestone-only"
    ):
        raise CorpusValidationError("stress cadence or size drifted")
    visual = manifest["oracles"]["blender_visual"]
    if (
        visual["blender_version"] != "5.2"
        or set(visual["coverage"])
        != {"animated-camera", "source-background", "model-coverage"}
        or visual["display_transforms"] != 8
        or visual["tolerance_display_pixels"] != 1.0
        or visual["requires_visible_gui"] is not True
    ):
        raise CorpusValidationError("Blender visual oracle contract drifted")
    for path_key in (
        "fixture_builder",
        "windows_runner",
        "isolated_preferences",
        "blender_driver",
    ):
        if not (ROOT / visual[path_key]).is_file():
            raise CorpusValidationError(
                f"Blender visual oracle is missing: {visual[path_key]}"
            )

    _validate_exact_oracles(exact)
    _validate_neural_oracles(neural)
    fixture_ids: set[str] = set()
    if generated_manifest is not None:
        fixture_ids = _validate_generated(
            generated_manifest.resolve(),
            goldens,
            require_stress=require_stress,
        )
    elif require_stress:
        raise CorpusValidationError(
            "--require-stress requires --generated-manifest"
        )

    blockers = [
        {
            "id": gate["id"],
            "status": gate["status"],
            "resolution": gate["resolution"],
        }
        for gate in manifest["release_gates"]
        if gate["status"] != "ready"
    ]
    result = {
        "schema_version": "1.0.0",
        "structural_validation": "passed",
        "generated_fixtures": sorted(fixture_ids),
        "release_ready": not blockers,
        "blockers": blockers,
    }
    if release and blockers:
        raise CorpusValidationError(
            "release gates remain blocked: "
            + ", ".join(item["id"] for item in blockers)
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-manifest", type=Path)
    parser.add_argument("--require-stress", action="store_true")
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = validate(
            generated_manifest=arguments.generated_manifest,
            require_stress=arguments.require_stress,
            release=arguments.release,
        )
    except CorpusValidationError as exc:
        print(f"LINGBOT_MAP_RELEASE_CORPUS_ERROR={exc}", file=sys.stderr)
        return 2
    print(
        "LINGBOT_MAP_RELEASE_CORPUS_VALIDATION="
        + json.dumps(result, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
