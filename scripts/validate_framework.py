"""Lightweight validation for the integration scaffold."""

from __future__ import annotations

import subprocess
import sys
import os


def main() -> int:
    commands = [
        [sys.executable, "-m", "compileall", "-q", "drivewm"],
        [sys.executable, "-m", "drivewm.cli", "list"],
        [sys.executable, "-m", "py_compile", "scripts/train_cogvideox_lora_nuscenes.py"],
    ]
    env = {
        **os.environ,
        "PYTHONPYCACHEPREFIX": os.environ.get("PYTHONPYCACHEPREFIX", "/tmp/drivewm_pycache"),
    }
    for command in commands:
        completed = subprocess.run(command, check=False, env=env)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
