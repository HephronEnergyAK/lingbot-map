"""Isolated process probes for the permanent Worker audit hook."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "worker" / "src"
if str(WORKER) not in sys.path:
    sys.path.insert(0, str(WORKER))

from lingbot_map_worker.fixture_job import _install_audit_policy


def _write_marker(path: str) -> None:
    Path(path).write_text("escaped", encoding="utf-8")


def _attempt(case: str, marker: Path) -> None:
    child = (
        "from pathlib import Path;"
        f"Path({str(marker)!r}).write_text('escaped',encoding='utf-8')"
    )
    if case == "socket-connect":
        with socket.socket() as stream:
            stream.connect(("127.0.0.1", 9))
    elif case == "socket-bind":
        with socket.socket() as stream:
            stream.bind(("127.0.0.1", 0))
    elif case == "subprocess":
        subprocess.run([sys.executable, "-c", child], check=False)
    elif case == "shell":
        os.system(f'"{sys.executable}" -c "{child}"')
    elif case == "spawn":
        os.spawnv(os.P_WAIT, sys.executable, (sys.executable, "-c", child))
    elif case == "multiprocessing":
        process = multiprocessing.get_context("spawn").Process(
            target=_write_marker,
            args=(str(marker),),
        )
        process.start()
        process.join()
    elif case == "startfile":
        if not hasattr(os, "startfile"):
            raise RuntimeError("startfile is unavailable")
        os.startfile(str(marker.with_suffix(".missing")))
    else:
        raise ValueError(f"unknown audit probe: {case}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("marker", type=Path)
    arguments = parser.parse_args(argv)
    _install_audit_policy()
    try:
        _attempt(arguments.case, arguments.marker)
    except PermissionError as exc:
        result = {
            "case": arguments.case,
            "state": "denied",
            "detail": str(exc),
            "marker_exists": arguments.marker.exists(),
        }
    except Exception as exc:
        result = {
            "case": arguments.case,
            "state": "unexpected-error",
            "detail": f"{type(exc).__name__}: {exc}",
            "marker_exists": arguments.marker.exists(),
        }
    else:
        result = {
            "case": arguments.case,
            "state": "escaped",
            "detail": None,
            "marker_exists": arguments.marker.exists(),
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "denied" and not result["marker_exists"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
