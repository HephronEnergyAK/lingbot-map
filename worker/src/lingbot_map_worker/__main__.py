"""Headless Worker entry point; Job execution is added by later tickets."""

from __future__ import annotations

import argparse
import json

from . import identity


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -I -m lingbot_map_worker")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Print the installed Worker distribution identity and exit",
    )
    arguments = parser.parse_args()
    if arguments.identity:
        print(json.dumps(identity(), sort_keys=True, separators=(",", ":")))
        return 0
    parser.error("a validated Job Control Envelope is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
