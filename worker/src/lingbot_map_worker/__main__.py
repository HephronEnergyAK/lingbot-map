"""Headless Worker entry point; Job execution is added by later tickets."""

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
    parser.error("a validated Job Control Envelope is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
