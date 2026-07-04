# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Download the MiniLM NLI model used by the Router.

Example:
    python -m router_agent.src.download_router_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from router_agent.src.model_store import (  # noqa: E402
    NLI_ROUTER_MODEL_DIR,
    NLI_ROUTER_MODEL_ID,
    download_router_model,
    is_router_model_downloaded,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Router NLI MiniLM model")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload of files that already exist in the local snapshot.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(NLI_ROUTER_MODEL_DIR),
        help="Directory where the model snapshot should be stored.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if is_router_model_downloaded(model_dir) and not args.force:
        print(f"Router model already available at {model_dir}")
        return

    print(f"Downloading {NLI_ROUTER_MODEL_ID} to {model_dir}...")
    downloaded_path = download_router_model(model_dir=model_dir, force_download=args.force)
    print(f"Router model ready at {downloaded_path}")


if __name__ == "__main__":
    main()
