"""Headless Worker entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import identity


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -I -m lingbot_map_worker")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Print the installed Worker distribution identity and exit",
    )
    parser.add_argument(
        "--discover-gpus",
        action="store_true",
        help="Print strict physical NVIDIA GPU identities without initializing CUDA",
    )
    parser.add_argument(
        "--capability-test",
        type=Path,
        metavar="REQUEST",
        help="Run one validated GPU capability-test request",
    )
    parser.add_argument(
        "--fixture-job",
        type=Path,
        metavar="JOB_SPEC",
        help="Run one deterministic lifecycle fixture from an immutable JobSpec",
    )
    parser.add_argument(
        "--preflight-job",
        type=Path,
        metavar="JOB_SPEC",
        help="Preflight one complete Capture Source from an immutable JobSpec",
    )
    parser.add_argument("--job-nonce", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.identity:
        print(json.dumps(identity(), sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.discover_gpus:
        from .capability_cli import discover_gpus

        return discover_gpus()
    if arguments.capability_test is not None:
        from .capability_cli import run_capability_test

        return run_capability_test(arguments.capability_test)
    if arguments.fixture_job is not None:
        if not arguments.job_nonce:
            parser.error("--fixture-job requires --job-nonce")
        from .fixture_job import run_fixture_job

        return run_fixture_job(arguments.fixture_job, arguments.job_nonce)
    if arguments.preflight_job is not None:
        if not arguments.job_nonce:
            parser.error("--preflight-job requires --job-nonce")
        from .preflight_job import run_preflight_job

        return run_preflight_job(arguments.preflight_job, arguments.job_nonce)
    parser.error("a validated Job Control Envelope is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
