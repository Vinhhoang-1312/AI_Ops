# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Thin Streamlit entrypoint for the IT support copilot demo."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.env import load_project_env
from app.streamlit_demo import main


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOGGER = logging.getLogger("agentic_it_support.streamlit_entrypoint")


def _configure_logging() -> None:
    level_name = os.getenv("ITS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT, force=False)
    LOGGER.setLevel(level)


def _main() -> None:
    load_project_env(PROJECT_ROOT)
    _configure_logging()
    LOGGER.info(
        "Starting Streamlit entrypoint cwd=%s script=%s python=%s log_level=%s",
        Path.cwd(),
        Path(__file__).resolve(),
        sys.version.split()[0],
        logging.getLevelName(LOGGER.level),
    )
    main()


_main()
