from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
sys.path.insert(0, str(WORKER_SOURCE))

from lingbot_map_worker.capability import (  # noqa: E402
    CapabilityCache,
    CapabilityCancelled,
    CapabilityGpuFailure,
    CapabilityIdentity,
    CapabilityLaunchGate,
    CapabilityResult,
    CapabilityStack,
    CapabilitySuite,
    InsufficientFreeVram,
    Measurement,
    ModelFingerprint,
    capability_identity,
)
from lingbot_map_worker.gpu_devices import (  # noqa: E402
    DeviceSelectionError,
    PhysicalGpu,
    select_physical_gpu,
)
from lingbot_map_worker.gpu_lease import (  # noqa: E402
    AmbiguousGpuLeaseError,
    GpuBusyError,
    GpuLease,
    LeaseOwner,
    ProcessFacts,
)
from lingbot_map_worker.gpu_profiles import PROFILES, profiles_by_name  # noqa: E402
from lingbot_map_worker.torch_capability import TorchCapabilityWorkload  # noqa: E402
from lingbot_map_worker.capability_cli import (  # noqa: E402
    CapabilityRequestError,
    _read_request,
)


RUNTIME_ID = "1" * 64
LOCK_SHA = "2" * 64
MODEL_SHA = "3" * 64
GPU_UUID = "GPU-3dd69ad9-06be-796b-5f6a-0a9159cd288c"


def device(total=1_000) -> PhysicalGpu:
    return PhysicalGpu(GPU_UUID, "Fixture GPU", total, "610.47", (12, 0))


def stack() -> CapabilityStack:
    return CapabilityStack(
        RUNTIME_ID,
        LOCK_SHA,
        "0.1.0",
        "2.11.0+cu130",
        "13.0",
    )


def model() -> ModelFingerprint:
    return ModelFingerprint("1.0.0", "fixture-model", MODEL_SHA)


class ProfileAndDeviceTests(unittest.TestCase):
    def test_named_profiles_preserve_fixed_model_workloads_and_product_settings(self):
        self.assertEqual([item.name for item in PROFILES], ["Draft", "Balanced", "High"])
        self.assertEqual([item.camera_iterations for item in PROFILES], [1, 4, 4])
        self.assertEqual([item.confidence_cutoff_percent for item in PROFILES], [70, 50, 30])
        self.assertEqual([item.import_point_budget for item in PROFILES], [1_000_000, 5_000_000, 10_000_000])
        for profile in PROFILES:
            self.assertEqual(
                (profile.image_size, profile.patch_size, profile.scale_frames, profile.window_frames),
                (518, 14, 8, 64),
            )
            self.assertEqual((profile.attention_backend, profile.execution_mode), ("sdpa", "eager"))
        self.assertEqual(PROFILES[1].workload_key, PROFILES[2].workload_key)
        self.assertNotEqual(PROFILES[1].settings_sha256, PROFILES[2].settings_sha256)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            profiles_by_name(("Draft", "Draft"))

    def test_device_selection_uses_only_unambiguous_physical_uuid(self):
        first = device()
        second = PhysicalGpu("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "Other", 2_000, "610.47", (8, 9))
        self.assertEqual(select_physical_gpu((first,), None), first)
        self.assertEqual(select_physical_gpu((first, second), second.uuid), second)
        with self.assertRaisesRegex(DeviceSelectionError, "Multiple"):
            select_physical_gpu((first, second), None)
        with self.assertRaisesRegex(DeviceSelectionError, "unavailable"):
            select_physical_gpu((first,), second.uuid)


class CapabilityRequestTests(unittest.TestCase):
    def test_request_requires_absolute_plain_checksum_addressed_paths_and_exact_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            managed = Path(temporary).resolve()
            model_path = managed / "models" / MODEL_SHA / "artifact"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"model")
            test_root = managed / "capability-tests" / "cap-valid123"
            test_root.mkdir(parents=True)
            document = {
                "schema_version": 1,
                "test_id": "cap-valid123",
                "nonce": "nonce-valid123",
                "runtime_id": RUNTIME_ID,
                "worker_lock_sha256": LOCK_SHA,
                "managed_root": str(managed),
                "gpu_uuid": GPU_UUID,
                "model": {
                    "catalog_version": "1.0.0",
                    "id": "fixture-model",
                    "sha256": MODEL_SHA,
                    "path": str(model_path),
                },
                "profiles": ["Draft"],
            }
            request = test_root / "request.json"
            request.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(_read_request(request).managed_root, managed)

            outside = managed / "outside-request.json"
            outside.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(CapabilityRequestError, "exact test identity"):
                _read_request(outside)

            document["managed_root"] = "relative-root"
            request.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(CapabilityRequestError, "must be absolute"):
                _read_request(request)


class FakeLeaseHandle:
    def __init__(self, state):
        self.state = state

    def read(self, maximum_bytes):
        if len(self.state.content) > maximum_bytes:
            raise AssertionError("oversized fixture")
        return self.state.content

    def replace(self, content):
        self.state.content = content

    def close(self):
        self.state.busy = False


class FakeHandles:
    def __init__(self, state):
        self.state = state

    def try_acquire(self, _path):
        if self.state.busy:
            return None
        self.state.busy = True
        return FakeLeaseHandle(self.state)

    def read_shared(self, _path, maximum_bytes):
        return self.state.content[:maximum_bytes]


class FakeProbe:
    def __init__(self, current, observed=None):
        self.current_facts = current
        self.observed = {} if observed is None else observed

    def current(self):
        return self.current_facts

    def observe(self, pid):
        return self.observed.get(pid)


def facts(pid=100) -> ProcessFacts:
    return ProcessFacts(pid, pid * 10, f"C:\\Runtime\\python-{pid}.exe", f"{pid:064x}"[-64:])


class GpuLeaseTests(unittest.TestCase):
    def test_exact_live_owner_is_busy_and_different_uuid_is_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = SimpleNamespace(content=b"", busy=False)
            owner_facts = facts(100)
            owner_probe = FakeProbe(owner_facts, {100: owner_facts})
            first = GpuLease(
                GPU_UUID,
                nonce="nonce-owner",
                runtime_id=RUNTIME_ID,
                action_kind="capability-test",
                action_id="test-owner",
                local_app_data=Path(temporary),
                probe=owner_probe,
                handles=FakeHandles(state),
            )
            with first:
                contender = GpuLease(
                    GPU_UUID,
                    nonce="nonce-contender",
                    runtime_id=RUNTIME_ID,
                    action_kind="capability-test",
                    action_id="test-contender",
                    local_app_data=Path(temporary),
                    probe=FakeProbe(facts(200), {100: owner_facts}),
                    handles=FakeHandles(state),
                )
                with self.assertRaises(GpuBusyError) as caught:
                    contender.__enter__()
                self.assertEqual(caught.exception.owner["pid"], 100)
            self.assertEqual(state.content, b"")

    def test_valid_dead_owner_is_reclaimed_but_ambiguous_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous_facts = facts(100)
            previous = LeaseOwner(
                previous_facts.pid,
                previous_facts.creation_time,
                previous_facts.executable,
                previous_facts.executable_sha256,
                1,
                GPU_UUID,
                "old-nonce",
                RUNTIME_ID,
                "capability-test",
                "old-test",
            )
            state = SimpleNamespace(
                content=(json.dumps(asdict(previous), sort_keys=True) + "\n").encode(),
                busy=False,
            )
            lease = GpuLease(
                GPU_UUID,
                nonce="new-nonce",
                runtime_id=RUNTIME_ID,
                action_kind="capability-test",
                action_id="new-test",
                local_app_data=Path(temporary),
                probe=FakeProbe(facts(200), {}),
                handles=FakeHandles(state),
            )
            with lease:
                self.assertIn(b'"pid":200', state.content)
            state.content = b"not-json"
            with self.assertRaises(AmbiguousGpuLeaseError):
                GpuLease(
                    GPU_UUID,
                    nonce="third-nonce",
                    runtime_id=RUNTIME_ID,
                    action_kind="capability-test",
                    action_id="third-test",
                    local_app_data=Path(temporary),
                    probe=FakeProbe(facts(300), {}),
                    handles=FakeHandles(state),
                ).__enter__()

    def test_live_metadata_without_os_handle_is_ambiguous_not_stolen(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous_facts = facts(100)
            previous = LeaseOwner(
                previous_facts.pid, previous_facts.creation_time, previous_facts.executable,
                previous_facts.executable_sha256, 1, GPU_UUID, "old-nonce", RUNTIME_ID,
                "capability-test", "old-test",
            )
            state = SimpleNamespace(
                content=(json.dumps(asdict(previous), sort_keys=True) + "\n").encode(),
                busy=False,
            )
            with self.assertRaises(AmbiguousGpuLeaseError):
                GpuLease(
                    GPU_UUID,
                    nonce="new-nonce",
                    runtime_id=RUNTIME_ID,
                    action_kind="capability-test",
                    action_id="new-test",
                    local_app_data=Path(temporary),
                    probe=FakeProbe(facts(200), {100: previous_facts}),
                    handles=FakeHandles(state),
                ).__enter__()

    @unittest.skipUnless(sys.platform == "win32", "native GPU Lease is Windows-only")
    def test_real_cross_process_lease_serializes_and_recovers_owner_death(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready"
            release = root / "release"
            holder = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "gpu_lease_holder.py"),
                    str(root),
                    GPU_UUID,
                    str(ready),
                    str(release),
                ]
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and holder.poll() is None:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.02)
                self.assertIsNone(holder.poll())
                contender = GpuLease(
                    GPU_UUID,
                    nonce="contender-nonce",
                    runtime_id=RUNTIME_ID,
                    action_kind="capability-test",
                    action_id="contender-test",
                    local_app_data=root,
                )
                with self.assertRaises(GpuBusyError):
                    contender.__enter__()

                second_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                with GpuLease(
                    second_uuid,
                    nonce="independent-nonce",
                    runtime_id=RUNTIME_ID,
                    action_kind="capability-test",
                    action_id="independent-test",
                    local_app_data=root,
                ):
                    pass

                holder.terminate()
                holder.wait(timeout=10)
                with GpuLease(
                    GPU_UUID,
                    nonce="recovery-nonce",
                    runtime_id=RUNTIME_ID,
                    action_kind="capability-test",
                    action_id="recovery-test",
                    local_app_data=root,
                ):
                    pass
            finally:
                if holder.poll() is None:
                    holder.terminate()
                    holder.wait(timeout=10)


class NullLease:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class FakeWorkload:
    def __init__(self, peaks, failure=None):
        self.peaks = peaks
        self.failure = failure
        self.measured = []

    def stack(self):
        return stack()

    def measure(self, profile, *, cancelled, progress):
        if cancelled():
            raise CapabilityCancelled("cancelled")
        self.measured.append(profile.name)
        if self.failure:
            raise RuntimeError(self.failure)
        progress(profile.name, profile.window_frames, profile.window_frames)
        return Measurement(self.peaks[profile.camera_iterations])

    def classify_gpu_failure(self, exception):
        return "cuda-device-loss" if "device lost" in str(exception) else None


class CapabilityTests(unittest.TestCase):
    def test_subset_profiles_are_cached_by_complete_identity_and_shared_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CapabilityCache(Path(temporary) / "cache")
            workload = FakeWorkload({1: 800, 4: 900})
            results = CapabilitySuite(
                device=device(1_000),
                model=model(),
                workload=workload,
                cache=cache,
                runtime_id=RUNTIME_ID,
                action_id="capability-1",
                nonce="nonce-1",
                lease_factory=NullLease,
            ).run(PROFILES)
            self.assertEqual(workload.measured, ["Draft", "Balanced"])
            self.assertEqual([result.state for result in results], ["qualified", "unqualified", "unqualified"])
            self.assertEqual([result.required_free_bytes for result in results], [960, 1080, 1080])
            self.assertEqual(len({result.identity.sha256 for result in results}), 3)
            for result in results:
                self.assertEqual(cache.read(result.identity), result)

    def test_cancellation_creates_no_success_or_failure_cache_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CapabilityCache(Path(temporary) / "cache")
            suite = CapabilitySuite(
                device=device(), model=model(), workload=FakeWorkload({1: 10, 4: 20}),
                cache=cache, runtime_id=RUNTIME_ID, action_id="cancel-test", nonce="nonce",
                lease_factory=NullLease,
            )
            with self.assertRaises(CapabilityCancelled):
                suite.run(PROFILES, cancelled=lambda: True)
            self.assertFalse(cache.root.exists())

    def test_cancellation_after_one_measurement_publishes_no_partial_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CapabilityCache(Path(temporary) / "cache")
            workload = FakeWorkload({1: 10, 4: 20})
            calls = 0

            def cancelled():
                nonlocal calls
                calls += 1
                return calls >= 3

            suite = CapabilitySuite(
                device=device(), model=model(), workload=workload,
                cache=cache, runtime_id=RUNTIME_ID, action_id="partial-cancel",
                nonce="nonce", lease_factory=NullLease,
            )
            with self.assertRaises(CapabilityCancelled):
                suite.run(PROFILES[:2], cancelled=cancelled)
            self.assertFalse(cache.root.exists())

    def test_gpu_failure_invalidates_matching_cache_without_retry_or_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CapabilityCache(Path(temporary) / "cache")
            identity = capability_identity(device(), stack(), model(), PROFILES[0])
            result = CapabilityResult(identity, "qualified", 100, 120, 1_000, "old", "2026-01-01T00:00:00+00:00")
            cache.write(result)
            workload = FakeWorkload({}, failure="CUDA device lost")
            suite = CapabilitySuite(
                device=device(), model=model(), workload=workload, cache=cache,
                runtime_id=RUNTIME_ID, action_id="failure-test", nonce="nonce",
                lease_factory=NullLease,
            )
            with self.assertRaises(CapabilityGpuFailure) as caught:
                suite.run((PROFILES[0],))
            self.assertEqual(caught.exception.classification, "cuda-device-loss")
            self.assertIsNone(cache.read(identity))
            self.assertEqual(len(tuple(cache.invalidated_root.iterdir())), 1)
            self.assertEqual(workload.measured, ["Draft"])

    def test_launch_gate_rechecks_current_free_vram_without_degrading_or_queueing(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CapabilityCache(Path(temporary) / "cache")
            identity = capability_identity(device(), stack(), model(), PROFILES[0])
            result = CapabilityResult(identity, "qualified", 800, 960, 1_000, "test", "2026-01-01T00:00:00+00:00")
            cache.write(result)
            provider = SimpleNamespace(memory_info=lambda _uuid: (959, 1_000))
            gate = CapabilityLaunchGate(cache, provider, lease_factory=NullLease)
            with self.assertRaisesRegex(InsufficientFreeVram, "explicitly retry"):
                with gate.acquire(identity, nonce="job-nonce", job_id="job-1"):
                    pass
            provider.memory_info = lambda _uuid: (960, 1_000)
            with gate.acquire(identity, nonce="job-nonce-2", job_id="job-2") as accepted:
                self.assertEqual(accepted, result)


class TorchFaultCleanupTests(unittest.TestCase):
    def test_device_loss_closes_backend_and_clears_cuda_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.pt"
            model_path.write_bytes(b"fixture")
            digest = hashlib.sha256(b"fixture").hexdigest()
            empty_cache_calls = []

            class FakeCuda:
                class OutOfMemoryError(RuntimeError):
                    pass

                @staticmethod
                def is_available(): return True
                @staticmethod
                def device_count(): return 1
                @staticmethod
                def set_device(_device): return None
                @staticmethod
                def manual_seed_all(_seed): return None
                @staticmethod
                def empty_cache(): empty_cache_calls.append(True)
                @staticmethod
                def reset_peak_memory_stats(_device): return None
                @staticmethod
                def get_device_capability(_device): return (12, 0)
                @staticmethod
                def synchronize(_device): return None

            fake_torch = SimpleNamespace(
                __version__="2.11.0+cu130",
                version=SimpleNamespace(cuda="13.0"),
                cuda=FakeCuda,
                backends=SimpleNamespace(
                    cudnn=SimpleNamespace(benchmark=True, deterministic=False)
                ),
                device=lambda value: value,
                manual_seed=lambda _seed: None,
                use_deterministic_algorithms=lambda *_args, **_kwargs: None,
                load=lambda *_args, **_kwargs: {"weight": object()},
                is_tensor=lambda _value: True,
                zeros=lambda *_args, **_kwargs: object(),
                float32=object(),
                bfloat16=object(),
                float16=object(),
            )

            fake_model = SimpleNamespace(
                load_state_dict=lambda _state, strict=False: ([], []),
                to=lambda _device: fake_model,
                eval=lambda: fake_model,
            )

            class FakeBatch:
                def release(self): return None

            class FakeBackend:
                def __init__(self):
                    self.closed = False
                    self.predict_calls = 0

                def begin(self, _scale): return FakeBatch()

                def predict(self, _frame, *, persist_keyframe):
                    self.predict_calls += 1
                    raise RuntimeError("CUDA device lost")

                def close(self): self.closed = True

            backend = FakeBackend()
            adapter = ModuleType("lingbot_map.model_adapter")
            adapter.build_native_windows_gct_model = lambda **_kwargs: fake_model
            adapter.GCTStreamBackend = lambda *_args, **_kwargs: backend
            package = ModuleType("lingbot_map")
            package.__path__ = []
            workload = TorchCapabilityWorkload(
                model_path=model_path,
                model_sha256=digest,
                runtime_id=RUNTIME_ID,
                worker_lock_sha256=LOCK_SHA,
                torch_module=fake_torch,
            )
            with mock.patch.dict(
                sys.modules,
                {"lingbot_map": package, "lingbot_map.model_adapter": adapter},
            ):
                with self.assertRaisesRegex(RuntimeError, "device lost") as caught:
                    workload.measure(
                        PROFILES[0], cancelled=lambda: False,
                        progress=lambda *_args: None,
                    )
            self.assertEqual(workload.classify_gpu_failure(caught.exception), "cuda-device-loss")
            self.assertEqual(backend.predict_calls, 1)
            self.assertTrue(backend.closed)
            self.assertGreaterEqual(len(empty_cache_calls), 2)

    def test_cuda_oom_reset_and_tdr_are_classified_fail_closed(self):
        fake_torch = SimpleNamespace(
            __version__="2.11.0+cu130",
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(OutOfMemoryError=MemoryError),
        )
        workload = TorchCapabilityWorkload(
            model_path=Path("model.pt"), model_sha256=MODEL_SHA,
            runtime_id=RUNTIME_ID, worker_lock_sha256=LOCK_SHA,
            torch_module=fake_torch,
        )
        self.assertEqual(workload.classify_gpu_failure(MemoryError()), "cuda-out-of-memory")
        self.assertEqual(
            workload.classify_gpu_failure(RuntimeError("CUDA driver reset")),
            "cuda-driver-reset",
        )
        self.assertEqual(
            workload.classify_gpu_failure(RuntimeError("CUDA launch timeout")),
            "windows-tdr",
        )


if __name__ == "__main__":
    unittest.main()
