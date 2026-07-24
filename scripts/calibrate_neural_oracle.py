"""Capture raw pinned-model outputs for release-oracle range calibration.

This is an engineering calibration tool, not a release validator.  It writes
the exact array contract consumed by ``validate_neural_oracle.py`` plus a
separate hardware/metric report.  The output directory must be new and empty
so an interrupted or mixed run cannot be mistaken for a complete calibration.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "worker" / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from lingbot_map_worker.canonical_preprocessing import canonical_geometry  # noqa: E402
from lingbot_map_worker.decoder import (  # noqa: E402
    DecodeContract,
    iter_capture_source_frames,
    preflight_capture_source,
)
from lingbot_map_worker.gpu_profiles import (  # noqa: E402
    PROFILE_BY_NAME,
    ProfileSelection,
)
from lingbot_map_worker.gpu_devices import (  # noqa: E402
    NvmlDeviceProvider,
    select_physical_gpu,
)
from lingbot_map_worker.production_model import (  # noqa: E402
    TorchPredictionDecoder,
    load_production_adapter,
)
from lingbot_map_worker.short_pipeline import (  # noqa: E402
    NullExecutionGuard,
    ProgressReporter,
    ShortReconstructionPipeline,
    SourceFrame,
)
from scripts.validate_neural_oracle import _metrics  # noqa: E402


ARRAY_FILENAMES = {
    "world_to_camera": "world_to_camera_opencv.npy",
    "intrinsics": "model_intrinsics.npy",
    "depth": "depth.npy",
    "confidence": "confidence.npy",
    "timestamps": "source_pts_seconds.npy",
    "frame_type": "frame_type.npy",
}


class CalibrationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CalibrationError(f"{label} must be one lowercase SHA-256")
    return value


def _prepare_output(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if path.exists():
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        linked = path.is_symlink() or bool(
            attributes
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        if linked or not path.is_dir() or any(path.iterdir()):
            raise CalibrationError("calibration output must be a new or empty plain directory")
    else:
        path.mkdir(parents=True)
    return path


class RawArraySink:
    def __init__(
        self,
        output: Path,
        *,
        frame_count: int,
        height: int,
        width: int,
    ) -> None:
        self.output = output
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.next_index = 0
        self.decoder = TorchPredictionDecoder()
        self.arrays = {
            "world_to_camera": np.lib.format.open_memmap(
                output / ARRAY_FILENAMES["world_to_camera"],
                mode="w+",
                dtype="<f8",
                shape=(frame_count, 4, 4),
            ),
            "intrinsics": np.lib.format.open_memmap(
                output / ARRAY_FILENAMES["intrinsics"],
                mode="w+",
                dtype="<f8",
                shape=(frame_count, 3, 3),
            ),
            "depth": np.lib.format.open_memmap(
                output / ARRAY_FILENAMES["depth"],
                mode="w+",
                dtype="<f4",
                shape=(frame_count, height, width),
            ),
            "confidence": np.lib.format.open_memmap(
                output / ARRAY_FILENAMES["confidence"],
                mode="w+",
                dtype="<f4",
                shape=(frame_count, height, width),
            ),
            "timestamps": np.lib.format.open_memmap(
                output / ARRAY_FILENAMES["timestamps"],
                mode="w+",
                dtype="<f8",
                shape=(frame_count,),
            ),
            "frame_type": np.lib.format.open_memmap(
                output / ARRAY_FILENAMES["frame_type"],
                mode="w+",
                dtype="|u1",
                shape=(frame_count,),
            ),
        }

    def prepare_frame(self, _frame_index: int, _canonical: Any) -> None:
        return None

    def accept(self, prediction: Any, canonical: Any, pts_seconds: float) -> None:
        aligned = self.decoder(prediction, canonical, pts_seconds)
        if aligned.frame_index != self.next_index:
            raise CalibrationError("model output is not contiguous and presentation-aligned")
        if aligned.depth.shape != (self.height, self.width):
            raise CalibrationError("model depth shape changed from canonical geometry")
        if aligned.confidence.shape != aligned.depth.shape:
            raise CalibrationError("model confidence shape differs from depth")
        index = self.next_index
        self.arrays["world_to_camera"][index] = aligned.world_to_camera_opencv
        self.arrays["intrinsics"][index] = aligned.model_intrinsics
        self.arrays["depth"][index] = aligned.depth
        self.arrays["confidence"][index] = aligned.confidence
        self.arrays["timestamps"][index] = aligned.source_pts_seconds
        self.arrays["frame_type"][index] = aligned.frame_type
        self.next_index += 1

    def finish(self) -> int:
        if self.next_index != self.frame_count:
            raise CalibrationError(
                f"model emitted {self.next_index} of {self.frame_count} frames"
            )
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()
        return self.next_index

    def abort(self) -> None:
        self.arrays.clear()
        gc.collect()


def calibrate(arguments: argparse.Namespace) -> dict[str, Any]:
    source = Path(arguments.source).resolve()
    model = Path(arguments.model).resolve()
    output = _prepare_output(arguments.output)
    report_path = Path(arguments.report).resolve()
    if report_path.exists() or report_path.is_symlink():
        raise CalibrationError("calibration report path must not already exist")
    expected_source = _require_sha256(arguments.source_sha256, "source SHA-256")
    expected_model = _require_sha256(arguments.model_sha256, "model SHA-256")
    runtime_id = _require_sha256(arguments.runtime_id, "runtime id")
    if _sha256(model) != expected_model:
        raise CalibrationError("model checksum differs before calibration")

    preflight = preflight_capture_source(source, thread_budget=1)
    frame_count = len(preflight.timestamps_seconds)
    if preflight.identity.sha256 != expected_source:
        raise CalibrationError("source checksum differs before calibration")
    profile = PROFILE_BY_NAME[arguments.profile]
    height, width, _source_to_model, _coverage = canonical_geometry(
        preflight.displayed_height,
        preflight.displayed_width,
    )
    contract = DecodeContract(
        preflight.identity,
        frame_count,
        preflight.video_stream_index,
        preflight.displayed_width,
        preflight.displayed_height,
        preflight.display_transform.name,
        preflight.color.standard,
        preflight.color.range,
    )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CalibrationError("calibration worker must see exactly one CUDA GPU")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_gpu:
        raise CalibrationError("calibration requires one physical GPU UUID")
    provider = NvmlDeviceProvider()
    physical_gpu = select_physical_gpu(provider.discover(), visible_gpu)
    torch.set_num_threads(max(1, min(6, int(os.cpu_count() or 1) - 2)))
    torch.set_num_interop_threads(1)
    gpu = torch.cuda.get_device_properties(0)
    sink = RawArraySink(
        output,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    started = time.monotonic()

    def emit(event: Any) -> None:
        if event.immediate or (
            event.kind == "progress"
            and (event.completed == frame_count or event.completed % 10 == 0)
        ):
            print(
                "LINGBOT_MAP_CALIBRATION_PROGRESS="
                + json.dumps(
                    {
                        "phase": event.phase,
                        "completed": event.completed,
                        "total": event.total,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    def adapter_factory(plan: Any, _resolved: Any, frame_shape: tuple[int, int, int]) -> Any:
        return load_production_adapter(
            model_path=model,
            expected_sha256=expected_model,
            plan=plan,
            profile=profile,
            frame_shape=frame_shape,
            cancel=lambda: False,
            torch_module=torch,
        )

    frames = (
        SourceFrame(item.frame_index, item.pts_seconds, item.srgb)
        for item in iter_capture_source_frames(
            source,
            contract,
            thread_budget=1,
        )
    )
    pipeline = ShortReconstructionPipeline(adapter_factory)
    try:
        pipeline.run(
            frame_count=frame_count,
            frames=frames,
            profile=ProfileSelection(
                profile.name,
                profile.camera_iterations,
                float(profile.confidence_cutoff_percent),
                float(profile.depth_cutoff_percent),
                profile.import_point_budget,
                True,
                False,
            ),
            result_sink=sink,
            progress=ProgressReporter(emit),
            execution_guard=NullExecutionGuard(),
            health_check=lambda _phase, _index: None,
            cancel=lambda: False,
        )
    except Exception:
        sink.abort()
        raise
    elapsed = time.monotonic() - started
    provenance = {
        "schema_version": "1.0.0",
        "fixture_id": arguments.fixture_id,
        "source_sha256": expected_source,
        "runtime_id": runtime_id,
        "model_id": arguments.model_id,
        "model_sha256": expected_model,
        "profile": profile.name,
        "frame_count": frame_count,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    arrays = {
        name: np.load(output / filename, allow_pickle=False, mmap_mode="r")
        for name, filename in (
            ("world_to_camera_opencv.npy", ARRAY_FILENAMES["world_to_camera"]),
            ("model_intrinsics.npy", ARRAY_FILENAMES["intrinsics"]),
            ("depth.npy", ARRAY_FILENAMES["depth"]),
            ("confidence.npy", ARRAY_FILENAMES["confidence"]),
            ("source_pts_seconds.npy", ARRAY_FILENAMES["timestamps"]),
            ("frame_type.npy", ARRAY_FILENAMES["frame_type"]),
        )
    }
    report = {
        "schema_version": "1.0.0",
        "purpose": "engineering-range-calibration-not-a-cross-gpu-checksum",
        "fixture_id": arguments.fixture_id,
        "source_sha256": expected_source,
        "model_sha256": expected_model,
        "runtime_id": runtime_id,
        "profile": profile.name,
        "frame_count": frame_count,
        "model_grid": [height, width],
        "metrics": _metrics(arrays),
        "hardware": {
            "gpu_uuid": physical_gpu.uuid,
            "gpu_name": physical_gpu.name,
            "gpu_total_memory": physical_gpu.total_memory,
            "compute_capability": list(physical_gpu.compute_capability),
            "driver": physical_gpu.driver_version,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
        },
        "elapsed_seconds": elapsed,
        "cross_gpu_checksum_used": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="lingbot-map-long")
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_BY_NAME), default="Draft")
    arguments = parser.parse_args(argv)
    try:
        report = calibrate(arguments)
    except Exception as exc:
        print(
            f"LINGBOT_MAP_NEURAL_CALIBRATION_ERROR={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        "LINGBOT_MAP_NEURAL_CALIBRATION="
        + json.dumps(report, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
