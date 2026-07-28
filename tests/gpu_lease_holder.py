"""Cross-process Windows GPU Lease holder used by integration tests."""

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "src"))

from lingbot_map_worker.gpu_lease import GpuLease


def main() -> None:
    local_app_data, gpu_uuid, ready, release = map(Path, sys.argv[1:5])
    with GpuLease(
        str(gpu_uuid),
        nonce="holder-nonce",
        runtime_id="1" * 64,
        action_kind="capability-test",
        action_id="holder-test",
        local_app_data=local_app_data,
    ):
        ready.write_text("ready", encoding="ascii")
        while not release.exists():
            time.sleep(0.02)


if __name__ == "__main__":
    main()
