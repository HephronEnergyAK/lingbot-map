"""Exercise the real NVML free-VRAM launch gate against a cached profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-root", type=Path, required=True)
    parser.add_argument("--profile", default="Balanced")
    parser.add_argument("--runtime-id", required=True)
    arguments = parser.parse_args()
    from lingbot_map_worker.capability import (
        CapabilityCache,
        CapabilityLaunchGate,
        InsufficientFreeVram,
        _identity_from_json,
    )
    from lingbot_map_worker.gpu_devices import NvmlDeviceProvider

    cache = CapabilityCache(arguments.managed_root / "capability-cache")
    matches = []
    for path in sorted(cache.root.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        identity = _identity_from_json(document["identity"])
        if (
            identity.profile_name == arguments.profile
            and identity.stack.runtime_id == arguments.runtime_id
        ):
            matches.append(identity)
    if len(matches) != 1:
        raise RuntimeError(f"expected one cached {arguments.profile} identity, got {len(matches)}")
    identity = matches[0]
    devices = NvmlDeviceProvider()
    free, total = devices.memory_info(identity.gpu_uuid)
    gate = CapabilityLaunchGate(cache, devices)
    try:
        with gate.acquire(identity, nonce="launch-gate-probe", job_id="launch-gate-probe"):
            outcome = "accepted"
    except InsufficientFreeVram as exc:
        outcome = "refused-insufficient-free-vram"
        error = str(exc)
    else:
        error = ""
    print(
        json.dumps(
            {
                "outcome": outcome,
                "gpu_uuid": identity.gpu_uuid,
                "profile": identity.profile_name,
                "free_bytes": free,
                "total_bytes": total,
                "required_free_bytes": cache.read(identity).required_free_bytes,
                "error": error,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
