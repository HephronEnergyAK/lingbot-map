"""Prove the launcher's ``-I`` and environment allowlist against poison state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("blender_extension")
package.__path__ = [str(ROOT / "blender_extension")]
sys.modules["blender_extension"] = package

from blender_extension.worker_environment import (
    worker_environment as _worker_environment,
)


FORBIDDEN = {
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PIP_CONFIG_FILE",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "UV_CONFIG_FILE",
    "UV_INDEX",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
}


def main() -> int:
    if os.name != "nt":
        print("LINGBOT_MAP_WORKER_ISOLATION=skipped-non-windows")
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        cwd = root / "poison-cwd"
        pythonpath = root / "poison-pythonpath"
        cwd.mkdir()
        pythonpath.mkdir()
        for directory in (cwd, pythonpath):
            (directory / "lingbot_poison.py").write_text(
                "raise RuntimeError('poison imported')\n",
                encoding="utf-8",
            )
        (cwd / "sitecustomize.py").write_text(
            "raise RuntimeError('sitecustomize imported')\n",
            encoding="utf-8",
        )
        for name in ("pyvenv.cfg", "pip.ini", "uv.toml"):
            (cwd / name).write_text("poison=true\n", encoding="utf-8")
        poisoned = dict(os.environ)
        poisoned.update({name: str(root / name) for name in FORBIDDEN})
        poisoned["PYTHONPATH"] = str(pythonpath)
        with mock.patch.dict(os.environ, poisoned, clear=True):
            environment = _worker_environment()
        code = (
            "import importlib.util,json,os,site,sys;"
            "forbidden=" + repr(sorted(FORBIDDEN)) + ";"
            "result={"
            "'isolated':sys.flags.isolated,"
            "'user_site':site.ENABLE_USER_SITE,"
            "'poison_spec':importlib.util.find_spec('lingbot_poison'),"
            "'forbidden_present':[k for k in forbidden if k in os.environ],"
            "'cwd_on_path':os.getcwd() in [os.path.abspath(p) for p in sys.path if p]"
            "};"
            "print(json.dumps(result,sort_keys=True))"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        result = json.loads(completed.stdout.splitlines()[-1])
        if result != {
            "cwd_on_path": False,
            "forbidden_present": [],
            "isolated": 1,
            "poison_spec": None,
            "user_site": False,
        }:
            raise RuntimeError(f"Worker isolation contract failed: {result!r}")
    print("LINGBOT_MAP_WORKER_ISOLATION=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
