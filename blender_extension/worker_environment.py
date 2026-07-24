"""Pure construction of the finite Worker's environment allowlist."""

from __future__ import annotations

import os


def worker_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP", "LOCALAPPDATA"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            # getpass.getuser() imports POSIX-only pwd when every user-name
            # variable is absent. Use a non-authoritative fixed value rather
            # than inheriting user-controlled identity into the Worker.
            "USERNAME": "LingBotMapWorker",
        }
    )
    return environment
