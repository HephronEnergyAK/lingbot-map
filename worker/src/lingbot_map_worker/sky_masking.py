"""Bounded all-or-fail Sky Masking with a content-addressed shared cache."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping
import uuid

import numpy as np
from PIL import Image

from .canonical_preprocessing import CanonicalImage, PREPROCESSING_RULE_VERSION
from .ipc import atomic_write_json, read_json, require_exact_object, require_text
from .model_store_compat import is_reparse_point
from .result_bundle import ArrayContract, validate_npy_file


SKY_MASK_RULE_VERSION = "non-sky-confidence-gt-0.1-v1"
SKY_SEGMENTATION_PREPROCESSING_VERSION = "skyseg-imagenet-320-bilinear-v1"
SKY_MASK_CACHE_SCHEMA_VERSION = "1.0.0"
SKY_MASK_PROVIDER = "CPUExecutionProvider"
SKY_MASK_BATCH_SIZE = 1
SKY_MASK_INPUT_SIZE = (320, 320)
SKY_MASK_THRESHOLD = np.float32(0.1)
SKY_MASK_CHUNK_FRAMES = 64
SKY_MASK_MAX_IN_FLIGHT = 64
SKY_MASK_CACHE_MANIFEST_MAX_BYTES = 4 * 1024 * 1024

CancelCheck = Callable[[], bool]


class SkyMaskError(RuntimeError):
    pass


class SkyMaskCancelled(SkyMaskError):
    pass


@dataclass(frozen=True)
class SkyMaskRequest:
    managed_root: Path
    source_sha256: str
    video_stream_index: int
    display_transform: str
    color_standard: str
    color_range: str
    frame_count: int
    model_grid_shape: tuple[int, int]
    model_id: str
    model_path: Path
    model_sha256: str
    worker_version: str
    onnx_threads: int
    cancel: CancelCheck


@dataclass(frozen=True)
class SkyMaskOutcome:
    sky_fraction: np.ndarray
    provenance: Mapping[str, Any]
    warnings: tuple[Mapping[str, str], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_directory(path: Path, *, label: str) -> None:
    if not path.is_dir() or path.is_symlink() or is_reparse_point(path):
        raise SkyMaskError(f"{label} is not an ordinary directory")


def _plain_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink() or is_reparse_point(path):
        raise SkyMaskError(f"{label} is not an ordinary file")


def _canonical_identity(document: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    identity = dict(document)
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return identity, hashlib.sha256(encoded).hexdigest()


class _CacheLock:
    """Short OS-held per-checksum lock; cache hits retain it while reading."""

    def __init__(self, path: Path, cancel: CancelCheck) -> None:
        self.path = path
        self.cancel = cancel
        self._stream: Any = None

    def acquire(self) -> "_CacheLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _plain_directory(self.path.parent, label="Sky Mask cache lock root")
        if self.path.exists() and (
            self.path.is_symlink() or is_reparse_point(self.path)
        ):
            raise SkyMaskError("Sky Mask cache lock is linked")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        else:
            try:
                os.write(descriptor, b"\0")
            finally:
                os.close(descriptor)
        deadline = time.monotonic() + 60.0
        while True:
            if self.cancel():
                raise SkyMaskCancelled("cancelled while waiting for Sky Mask cache lock")
            stream = self.path.open("r+b")
            try:
                self._lock(stream)
            except BlockingIOError:
                stream.close()
                if time.monotonic() >= deadline:
                    raise SkyMaskError("Sky Mask cache lock remained busy for 60 seconds")
                time.sleep(0.02)
                continue
            self._stream = stream
            return self

    @staticmethod
    def _lock(stream: Any) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError(str(exc)) from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise BlockingIOError(str(exc)) from exc

    @staticmethod
    def _unlock(stream: Any) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def release(self) -> None:
        if self._stream is None:
            return
        self._unlock(self._stream)
        self._stream.close()
        self._stream = None

    def __enter__(self) -> "_CacheLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


@dataclass(frozen=True)
class _CacheChunk:
    start: int
    count: int
    path: str
    byte_length: int
    sha256: str


class _CacheReader:
    def __init__(
        self,
        root: Path,
        chunks: tuple[_CacheChunk, ...],
        grid_shape: tuple[int, int],
    ) -> None:
        self.root = root
        self.chunks = chunks
        self.grid_shape = grid_shape
        self._loaded_index = -1
        self._loaded: np.ndarray | None = None

    def mask(self, frame_index: int) -> np.ndarray:
        chunk_index = frame_index // SKY_MASK_CHUNK_FRAMES
        if not 0 <= chunk_index < len(self.chunks):
            raise SkyMaskError(f"Sky Mask cache lacks frame {frame_index}")
        chunk = self.chunks[chunk_index]
        if not chunk.start <= frame_index < chunk.start + chunk.count:
            raise SkyMaskError(f"Sky Mask cache chunk does not cover frame {frame_index}")
        if self._loaded_index != chunk_index:
            path = self.root / Path(chunk.path)
            self._loaded = validate_npy_file(
                path,
                ArrayContract(
                    "|u1", (chunk.count, self.grid_shape[0], self.grid_shape[1])
                ),
            )
            self._loaded_index = chunk_index
        assert self._loaded is not None
        return np.ascontiguousarray(self._loaded[frame_index - chunk.start], dtype="|u1")


def _cache_chunks(
    root: Path,
    *,
    identity: Mapping[str, Any],
    cache_key: str,
    frame_count: int,
    grid_shape: tuple[int, int],
) -> tuple[_CacheChunk, ...]:
    _plain_directory(root, label="Sky Mask cache entry")
    manifest_path = root / "manifest.json"
    _plain_file(manifest_path, label="Sky Mask cache manifest")
    manifest = read_json(
        manifest_path, maximum=SKY_MASK_CACHE_MANIFEST_MAX_BYTES
    )
    manifest = require_exact_object(
        manifest,
        {
            "schema_version",
            "cache_key",
            "completion_state",
            "identity",
            "frame_count",
            "grid_shape",
            "chunk_frames",
            "chunks",
        },
        label="Sky Mask cache manifest",
    )
    if (
        manifest["schema_version"] != SKY_MASK_CACHE_SCHEMA_VERSION
        or manifest["cache_key"] != cache_key
        or manifest["completion_state"] != "complete"
        or manifest["identity"] != dict(identity)
        or manifest["frame_count"] != frame_count
        or manifest["grid_shape"] != list(grid_shape)
        or manifest["chunk_frames"] != SKY_MASK_CHUNK_FRAMES
        or not isinstance(manifest["chunks"], list)
    ):
        raise SkyMaskError("Sky Mask cache manifest identity is invalid")
    chunks: list[_CacheChunk] = []
    expected_start = 0
    declared = {"manifest.json"}
    for raw in manifest["chunks"]:
        item = require_exact_object(
            raw,
            {"start", "count", "path", "dtype", "shape", "byte_length", "sha256"},
            label="Sky Mask cache chunk",
        )
        start, count = item["start"], item["count"]
        expected_count = min(SKY_MASK_CHUNK_FRAMES, frame_count - expected_start)
        expected_path = (
            f"chunks/{expected_start:08d}-{expected_start + expected_count - 1:08d}.npy"
        )
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or start != expected_start
            or count != expected_count
            or item["path"] != expected_path
            or item["dtype"] != "|u1"
            or item["shape"] != [count, grid_shape[0], grid_shape[1]]
            or isinstance(item["byte_length"], bool)
            or not isinstance(item["byte_length"], int)
            or item["byte_length"] < 1
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
        ):
            raise SkyMaskError("Sky Mask cache chunk descriptor is invalid")
        path = root / Path(expected_path)
        _plain_file(path, label="Sky Mask cache chunk")
        before = path.stat()
        array = validate_npy_file(
            path, ArrayContract("|u1", (count, grid_shape[0], grid_shape[1]))
        )
        if (
            before.st_size != item["byte_length"]
            or _sha256_file(path) != item["sha256"]
            or not bool(np.isin(array, (0, 1)).all())
        ):
            raise SkyMaskError("Sky Mask cache chunk checksum or values are invalid")
        chunks.append(
            _CacheChunk(start, count, expected_path, before.st_size, item["sha256"])
        )
        declared.add(expected_path)
        expected_start += count
    if expected_start != frame_count:
        raise SkyMaskError("Sky Mask cache does not contain every source frame")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or is_reparse_point(path):
            raise SkyMaskError("Sky Mask cache contains linked content")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != declared:
        raise SkyMaskError("Sky Mask cache contains undeclared or missing files")
    return tuple(chunks)


def _quarantine_corrupt(cache_root: Path, entry: Path, cache_key: str) -> Path:
    corrupt_root = cache_root / "corrupt"
    corrupt_root.mkdir(parents=True, exist_ok=True)
    _plain_directory(corrupt_root, label="Sky Mask corrupt cache root")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = corrupt_root / f"{cache_key}-{stamp}-{uuid.uuid4().hex[:8]}"
    os.replace(entry, destination)
    return destination


def _remove_owned_staging(path: Path, cache_root: Path) -> None:
    if (
        path.exists()
        and path.parent == cache_root
        and path.name.startswith(".staging-")
        and path.is_dir()
        and not path.is_symlink()
        and not is_reparse_point(path)
    ):
        shutil.rmtree(path)


class SkyMaskSession:
    """Deep module for bounded segmentation, cache lifecycle, and Result evidence."""

    def __init__(
        self,
        request: SkyMaskRequest,
        *,
        runtime_module: Any | None = None,
    ) -> None:
        self.request = request
        self._validate_request()
        self.runtime = runtime_module or self._import_runtime()
        self.runtime_version = require_text(
            getattr(self.runtime, "__version__", None),
            label="ONNX Runtime version",
            maximum=128,
        )
        self.session = self._create_session()
        self.input_name, self.output_name = self._validate_session()
        self._probe_session()
        self.identity, self.cache_key = _canonical_identity(
            {
                "source_sha256": request.source_sha256,
                "video_stream_index": request.video_stream_index,
                "display_transform": request.display_transform,
                "color_conversion": {
                    "standard": request.color_standard,
                    "range": request.color_range,
                },
                "canonical_preprocessing_rule_version": PREPROCESSING_RULE_VERSION,
                "sky_preprocessing_version": SKY_SEGMENTATION_PREPROCESSING_VERSION,
                "model_id": request.model_id,
                "model_sha256": request.model_sha256,
                "mask_rule_version": SKY_MASK_RULE_VERSION,
                "worker_version": request.worker_version,
                "onnxruntime_version": self.runtime_version,
                "provider": SKY_MASK_PROVIDER,
                "batch_size": SKY_MASK_BATCH_SIZE,
                "grid_shape": list(request.model_grid_shape),
            }
        )
        self.cache_root = request.managed_root / "cache" / "sky-masks"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        _plain_directory(self.cache_root, label="Sky Mask cache root")
        for name in ("locks", "corrupt"):
            directory = self.cache_root / name
            directory.mkdir(exist_ok=True)
            _plain_directory(directory, label=f"Sky Mask cache {name} root")
        self.final_cache = self.cache_root / self.cache_key
        self.staging = self.cache_root / (
            f".staging-{self.cache_key[:16]}-{uuid.uuid4().hex}"
        )
        self.reader: _CacheReader | None = None
        self._retained_hit_lock: _CacheLock | None = None
        self._corrupt_quarantined = False
        self._cache_write_failed = False
        self._cache_failure_message: str | None = None
        self._cache_status = "generated"
        self._discover_cache()
        self._executor: ThreadPoolExecutor | None = None
        if self.reader is None:
            self.staging.mkdir()
            _plain_directory(self.staging, label="Sky Mask cache staging")
            (self.staging / "chunks").mkdir()
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="LingBotMap-SkyMask"
            )
        self._futures: "OrderedDict[int, Future[np.ndarray]]" = OrderedDict()
        self._prepared = 0
        self._consumed = 0
        self._fractions: list[np.float32] = []
        self._chunk_start = 0
        self._chunk_masks: list[np.ndarray] = []
        self._chunk_descriptors: list[dict[str, Any]] = []
        self.maximum_in_flight = 0
        self._finished = False
        self._aborted = False

    @staticmethod
    def _import_runtime() -> Any:
        try:
            import onnxruntime
        except ImportError as exc:
            raise SkyMaskError("CPU ONNX Runtime is not installed") from exc
        try:
            from importlib import metadata

            cpu_version = metadata.version("onnxruntime")
            try:
                metadata.version("onnxruntime-gpu")
            except metadata.PackageNotFoundError:
                pass
            else:
                raise SkyMaskError("onnxruntime-gpu is forbidden in the native Worker")
            if cpu_version != onnxruntime.__version__:
                raise SkyMaskError("ONNX Runtime package and module versions disagree")
        except SkyMaskError:
            raise
        except Exception as exc:
            raise SkyMaskError(f"Cannot verify CPU ONNX Runtime package: {exc}") from exc
        return onnxruntime

    def _validate_request(self) -> None:
        request = self.request
        managed_root = Path(os.path.abspath(request.managed_root))
        model_path = Path(os.path.abspath(request.model_path))
        _plain_directory(managed_root, label="managed Runtime root")
        _plain_file(model_path, label="SkySeg Auxiliary Model")
        if not model_path.is_relative_to(managed_root):
            raise SkyMaskError("SkySeg Auxiliary Model escaped the managed Runtime root")
        if _sha256_file(model_path) != request.model_sha256:
            raise SkyMaskError("SkySeg Auxiliary Model checksum is invalid")
        if (
            request.frame_count < 1
            or len(request.model_grid_shape) != 2
            or min(request.model_grid_shape) < 1
            or request.onnx_threads < 1
            or request.onnx_threads > 8
            or request.video_stream_index < 0
            or not callable(request.cancel)
        ):
            raise SkyMaskError("Sky Mask execution request is invalid")
        for value, label in (
            (request.source_sha256, "source checksum"),
            (request.model_sha256, "model checksum"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SkyMaskError(f"Sky Mask {label} is invalid")
        if request.model_id != "skyseg":
            raise SkyMaskError("Sky Masking requires the catalogued skyseg model")
        if request.color_standard not in {"bt601", "bt709"} or request.color_range not in {
            "limited",
            "full",
        }:
            raise SkyMaskError("Sky Mask color identity is invalid")

    def _create_session(self) -> Any:
        options = self.runtime.SessionOptions()
        options.intra_op_num_threads = int(self.request.onnx_threads)
        options.inter_op_num_threads = 1
        options.execution_mode = self.runtime.ExecutionMode.ORT_SEQUENTIAL
        try:
            session = self.runtime.InferenceSession(
                str(self.request.model_path),
                sess_options=options,
                providers=[SKY_MASK_PROVIDER],
            )
        except Exception as exc:
            raise SkyMaskError(f"Cannot create bounded CPU SkySeg session: {exc}") from exc
        disable = getattr(session, "disable_fallback", None)
        if not callable(disable):
            raise SkyMaskError("ONNX Runtime session cannot disable Provider fallback")
        disable()
        return session

    def _validate_session(self) -> tuple[str, str]:
        providers = self.session.get_providers()
        if providers != [SKY_MASK_PROVIDER]:
            raise SkyMaskError(
                f"SkySeg session selected unexpected Providers: {providers!r}"
            )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise SkyMaskError("SkySeg must expose exactly one input and an output")
        input_meta = inputs[0]
        output_meta = outputs[0]
        input_name = require_text(
            getattr(input_meta, "name", None), label="SkySeg input name", maximum=256
        )
        output_name = require_text(
            getattr(output_meta, "name", None), label="SkySeg output name", maximum=256
        )
        input_type = getattr(input_meta, "type", "tensor(float)")
        if input_type != "tensor(float)":
            raise SkyMaskError("SkySeg input must be float32")
        if getattr(output_meta, "type", "tensor(float)") != "tensor(float)":
            raise SkyMaskError("SkySeg primary output must be float32")
        if getattr(output_meta, "shape", [1, 1, 320, 320]) != [1, 1, 320, 320]:
            raise SkyMaskError("SkySeg primary output shape is not native-v1")
        return input_name, output_name

    def _probe_session(self) -> None:
        probe = np.zeros(
            (SKY_MASK_BATCH_SIZE, 3, SKY_MASK_INPUT_SIZE[1], SKY_MASK_INPUT_SIZE[0]),
            dtype="<f4",
        )
        self._run_session(probe)

    def _discover_cache(self) -> None:
        lock = _CacheLock(
            self.cache_root / "locks" / f"{self.cache_key}.lock",
            self.request.cancel,
        ).acquire()
        retain = False
        try:
            if not self.final_cache.exists():
                return
            try:
                chunks = _cache_chunks(
                    self.final_cache,
                    identity=self.identity,
                    cache_key=self.cache_key,
                    frame_count=self.request.frame_count,
                    grid_shape=self.request.model_grid_shape,
                )
            except Exception:
                _quarantine_corrupt(self.cache_root, self.final_cache, self.cache_key)
                self._corrupt_quarantined = True
                self._cache_status = "regenerated"
                return
            self.reader = _CacheReader(
                self.final_cache, chunks, self.request.model_grid_shape
            )
            self._cache_status = "hit"
            self._retained_hit_lock = lock
            retain = True
        finally:
            if not retain:
                lock.release()

    def prepare(self, frame_index: int, canonical: CanonicalImage) -> None:
        if self._finished or self._aborted:
            raise SkyMaskError("Sky Mask session is not accepting frames")
        if frame_index != self._prepared or frame_index >= self.request.frame_count:
            raise SkyMaskError("Sky Mask frames must be prepared contiguously")
        rgb = canonical.color_rgb
        expected_shape = (*self.request.model_grid_shape, 3)
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype.str != "|u1"
            or rgb.shape != expected_shape
            or not rgb.flags.c_contiguous
        ):
            raise SkyMaskError("Sky Mask requires the canonical uint8 RGB model grid")
        if self.request.cancel():
            raise SkyMaskCancelled(f"cancelled before Sky Mask frame {frame_index}")
        if self.reader is None:
            if len(self._futures) >= SKY_MASK_MAX_IN_FLIGHT:
                raise SkyMaskError("Sky Mask Pipeline exceeded its bounded 64-frame queue")
            assert self._executor is not None
            self._futures[frame_index] = self._executor.submit(
                self._infer_eligible, rgb
            )
            self.maximum_in_flight = max(
                self.maximum_in_flight, len(self._futures)
            )
        self._prepared += 1

    def mask_for(
        self, frame_index: int, target_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.float32]:
        if self._finished or self._aborted:
            raise SkyMaskError("Sky Mask session cannot provide another mask")
        if (
            frame_index != self._consumed
            or frame_index >= self._prepared
            or target_shape != self.request.model_grid_shape
        ):
            raise SkyMaskError("Sky Mask consumption order or shape is invalid")
        if self.request.cancel():
            raise SkyMaskCancelled(f"cancelled before Sky Mask frame {frame_index}")
        if self.reader is not None:
            mask = self.reader.mask(frame_index)
        else:
            try:
                future = self._futures.pop(frame_index)
            except KeyError as exc:
                raise SkyMaskError(f"Sky Mask frame {frame_index} is missing") from exc
            try:
                mask = future.result()
            except SkyMaskCancelled:
                raise
            except Exception as exc:
                raise SkyMaskError(
                    f"Sky Mask generation failed at frame {frame_index}: {exc}"
                ) from exc
            self._record_generated_mask(mask)
        if (
            mask.dtype.str != "|u1"
            or mask.shape != target_shape
            or not mask.flags.c_contiguous
            or not bool(np.isin(mask, (0, 1)).all())
        ):
            raise SkyMaskError(f"Sky Mask frame {frame_index} is invalid")
        fraction = np.float32(1.0 - (float(np.count_nonzero(mask)) / mask.size))
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise SkyMaskError(f"Sky fraction for frame {frame_index} is invalid")
        self._fractions.append(fraction)
        self._consumed += 1
        return mask, fraction

    def _infer_eligible(self, rgb: np.ndarray) -> np.ndarray:
        if self.request.cancel():
            raise SkyMaskCancelled("cancelled before SkySeg inference")
        resized = Image.fromarray(rgb, mode="RGB").resize(
            SKY_MASK_INPUT_SIZE, Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.float32)
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
        normalized = (pixels / np.float32(255.0) - mean) / std
        batch = np.ascontiguousarray(
            normalized.transpose(2, 0, 1)[None, ...], dtype="<f4"
        )
        non_sky = self._run_session(batch)
        target_height, target_width = self.request.model_grid_shape
        continuous = np.asarray(
            Image.fromarray(non_sky, mode="F").resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        if (
            continuous.shape != self.request.model_grid_shape
            or not np.isfinite(continuous).all()
        ):
            raise SkyMaskError("SkySeg bilinear output is invalid")
        eligible = np.ascontiguousarray(continuous > SKY_MASK_THRESHOLD, dtype="|u1")
        if self.request.cancel():
            raise SkyMaskCancelled("cancelled after SkySeg inference")
        return eligible

    def _run_session(self, batch: np.ndarray) -> np.ndarray:
        if (
            batch.dtype.str != "<f4"
            or batch.shape
            != (
                SKY_MASK_BATCH_SIZE,
                3,
                SKY_MASK_INPUT_SIZE[1],
                SKY_MASK_INPUT_SIZE[0],
            )
            or not batch.flags.c_contiguous
        ):
            raise SkyMaskError("SkySeg input batch violates the fixed native-v1 contract")
        try:
            outputs = self.session.run(
                [self.output_name], {self.input_name: batch}
            )
        except Exception as exc:
            raise SkyMaskError(f"SkySeg ONNX inference failed: {exc}") from exc
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise SkyMaskError("SkySeg returned an invalid output list")
        raw = np.asarray(outputs[0])
        while raw.ndim > 2 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.shape != (SKY_MASK_INPUT_SIZE[1], SKY_MASK_INPUT_SIZE[0]):
            raise SkyMaskError(f"SkySeg returned invalid output shape {raw.shape}")
        raw = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise SkyMaskError("SkySeg returned non-finite confidence")
        minimum = float(raw.min())
        maximum = float(raw.max())
        denominator = max(maximum - minimum, 1e-8)
        sky = (raw - np.float32(minimum)) / np.float32(denominator)
        return np.ascontiguousarray(np.float32(1.0) - sky, dtype="<f4")

    def _record_generated_mask(self, mask: np.ndarray) -> None:
        if self._cache_write_failed:
            return
        self._chunk_masks.append(mask)
        if len(self._chunk_masks) == SKY_MASK_CHUNK_FRAMES:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self._chunk_masks or self._cache_write_failed:
            return
        start = self._chunk_start
        count = len(self._chunk_masks)
        relative = f"chunks/{start:08d}-{start + count - 1:08d}.npy"
        path = self.staging / Path(relative)
        array = np.ascontiguousarray(np.stack(self._chunk_masks), dtype="|u1")
        try:
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            validate_npy_file(
                path,
                ArrayContract(
                    "|u1",
                    (count, self.request.model_grid_shape[0], self.request.model_grid_shape[1]),
                ),
            )
            self._chunk_descriptors.append(
                {
                    "start": start,
                    "count": count,
                    "path": relative,
                    "dtype": "|u1",
                    "shape": [
                        count,
                        self.request.model_grid_shape[0],
                        self.request.model_grid_shape[1],
                    ],
                    "byte_length": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
            self._chunk_start += count
            self._chunk_masks.clear()
        except Exception as exc:
            self._mark_cache_write_failed(exc)

    def _mark_cache_write_failed(self, exc: Exception) -> None:
        self._cache_write_failed = True
        self._cache_failure_message = f"{type(exc).__name__}: {exc}"[:1024]
        self._chunk_masks.clear()
        try:
            _remove_owned_staging(self.staging, self.cache_root)
        except OSError:
            pass

    def finish(self) -> SkyMaskOutcome:
        if self._finished:
            raise SkyMaskError("Sky Mask finish is not repeatable")
        self._finished = True
        try:
            if (
                self._prepared != self.request.frame_count
                or self._consumed != self.request.frame_count
                or self._futures
                or len(self._fractions) != self.request.frame_count
            ):
                raise SkyMaskError(
                    f"Sky Masking completed {self._consumed} of "
                    f"{self.request.frame_count} source frames"
                )
            if self.reader is None:
                self._flush_chunk()
                if not self._cache_write_failed:
                    self._publish_generated_cache()
            fractions = np.ascontiguousarray(self._fractions, dtype="<f4")
            if (
                fractions.shape != (self.request.frame_count,)
                or not np.isfinite(fractions).all()
                or not bool(((fractions >= 0) & (fractions <= 1)).all())
            ):
                raise SkyMaskError("Sky Mask fractions are incomplete or invalid")
            warnings: list[Mapping[str, str]] = []
            if self._corrupt_quarantined:
                warnings.append(
                    {
                        "code": "sky_mask_cache_corrupt",
                        "message": (
                            "A corrupt Sky Mask cache entry was quarantined and "
                            "regenerated once"
                        ),
                    }
                )
            if self._cache_write_failed:
                base = (
                    "regenerated" if self._corrupt_quarantined else "generated"
                )
                self._cache_status = f"{base}-cache-write-failed"
                warnings.append(
                    {
                        "code": "cache_write_failed",
                        "message": (
                            "Complete Sky Masks were used, but the reusable cache "
                            f"could not be published: {self._cache_failure_message}"
                        )[:16384],
                    }
                )
            return SkyMaskOutcome(
                fractions,
                {
                    "enabled": True,
                    "model_id": self.request.model_id,
                    "model_sha256": self.request.model_sha256,
                    "rule_version": SKY_MASK_RULE_VERSION,
                    "preprocessing_version": SKY_SEGMENTATION_PREPROCESSING_VERSION,
                    "provider": SKY_MASK_PROVIDER,
                    "onnxruntime_version": self.runtime_version,
                    "batch_size": SKY_MASK_BATCH_SIZE,
                    "onnx_threads": self.request.onnx_threads,
                    "cache_key": self.cache_key,
                    "cache_status": self._cache_status,
                },
                tuple(warnings),
            )
        finally:
            self._shutdown_executor()
            if self._retained_hit_lock is not None:
                self._retained_hit_lock.release()
                self._retained_hit_lock = None

    def _publish_generated_cache(self) -> None:
        manifest = {
            "schema_version": SKY_MASK_CACHE_SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "completion_state": "complete",
            "identity": self.identity,
            "frame_count": self.request.frame_count,
            "grid_shape": list(self.request.model_grid_shape),
            "chunk_frames": SKY_MASK_CHUNK_FRAMES,
            "chunks": self._chunk_descriptors,
        }
        try:
            atomic_write_json(self.staging / "manifest.json", manifest)
            _cache_chunks(
                self.staging,
                identity=self.identity,
                cache_key=self.cache_key,
                frame_count=self.request.frame_count,
                grid_shape=self.request.model_grid_shape,
            )
            with _CacheLock(
                self.cache_root / "locks" / f"{self.cache_key}.lock",
                self.request.cancel,
            ):
                if self.final_cache.exists():
                    try:
                        _cache_chunks(
                            self.final_cache,
                            identity=self.identity,
                            cache_key=self.cache_key,
                            frame_count=self.request.frame_count,
                            grid_shape=self.request.model_grid_shape,
                        )
                    except Exception:
                        _quarantine_corrupt(
                            self.cache_root, self.final_cache, self.cache_key
                        )
                    else:
                        _remove_owned_staging(self.staging, self.cache_root)
                        self._cache_status = "race-reused"
                        return
                os.replace(self.staging, self.final_cache)
                _cache_chunks(
                    self.final_cache,
                    identity=self.identity,
                    cache_key=self.cache_key,
                    frame_count=self.request.frame_count,
                    grid_shape=self.request.model_grid_shape,
                )
            self._cache_status = (
                "regenerated" if self._corrupt_quarantined else "generated"
            )
        except SkyMaskCancelled:
            raise
        except Exception as exc:
            self._mark_cache_write_failed(exc)

    def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        for future in self._futures.values():
            future.cancel()
        self._futures.clear()
        self._shutdown_executor()
        if self._retained_hit_lock is not None:
            self._retained_hit_lock.release()
            self._retained_hit_lock = None
        try:
            _remove_owned_staging(self.staging, self.cache_root)
        except OSError:
            pass

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
