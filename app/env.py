# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Small `.env` loader for local app entrypoints."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_env(project_root: str | Path, *, filename: str = ".env") -> Path:
    """Load environment variables from ``project_root/filename`` if it exists.

    Existing process environment variables take precedence over values stored in
    the file so shell-level overrides continue to work.
    """

    env_path = Path(project_root) / filename
    if not env_path.exists():
        return env_path

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    return env_path
