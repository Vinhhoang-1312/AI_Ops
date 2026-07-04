# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Local model storage helpers for the Router component."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_MODELS_DIR = PROJECT_ROOT / "data" / "pretrained-models"
NLI_ROUTER_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_ROUTER_MODEL_DIR = PRETRAINED_MODELS_DIR / "cross-encoder" / "nli-MiniLM2-L6-H768"
NLI_ROUTER_OPENVINO_DIR = NLI_ROUTER_MODEL_DIR / "openvino"
NLI_ROUTER_OPENVINO_MODEL_XML = NLI_ROUTER_OPENVINO_DIR / "model.xml"
NLI_ROUTER_OPENVINO_MANIFEST = NLI_ROUTER_OPENVINO_DIR / "manifest.json"


def router_model_path(model_dir: Optional[str | Path] = None) -> Path:
    """Return the local filesystem path for the Router NLI model."""
    return Path(model_dir) if model_dir is not None else NLI_ROUTER_MODEL_DIR


def router_openvino_model_path(model_dir: Optional[str | Path] = None) -> Path:
    """Return the OpenVINO IR XML path for the Router model."""
    base_dir = router_model_path(model_dir)
    return base_dir / "openvino" / "model.xml"


def is_router_model_downloaded(model_dir: Optional[str | Path] = None) -> bool:
    """Return True when the local Router model snapshot has config and weights."""
    path = router_model_path(model_dir)
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_tokenizer = any(
        (path / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json")
    )
    has_weights = any(any(path.glob(pattern)) for pattern in ("*.safetensors", "*.bin"))
    return has_config and has_tokenizer and has_weights


def is_router_openvino_model_available(model_dir: Optional[str | Path] = None) -> bool:
    """Return True when the converted OpenVINO Router IR exists."""
    xml_path = router_openvino_model_path(model_dir)
    bin_path = xml_path.with_suffix(".bin")
    return xml_path.exists() and bin_path.exists()


def download_router_model(
    *,
    model_id: str = NLI_ROUTER_MODEL_ID,
    model_dir: Optional[str | Path] = None,
    force_download: bool = False,
) -> Path:
    """Download the Router NLI model from Hugging Face into data/pretrained-models."""
    target_dir = router_model_path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Install `huggingface-hub` or run `pip install -r requirements.txt` "
            "before downloading the Router model."
        ) from exc

    common_kwargs = {
        "repo_id": model_id,
        "local_dir": str(target_dir),
        "force_download": force_download,
        "allow_patterns": [
            "config.json",
            "merges.txt",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
        "ignore_patterns": ["onnx/*", "openvino/*"],
    }
    try:
        snapshot_download(local_dir_use_symlinks=False, **common_kwargs)
    except TypeError:  # pragma: no cover - newer huggingface_hub removed this arg
        snapshot_download(**common_kwargs)

    if not is_router_model_downloaded(target_dir):
        raise RuntimeError(f"Downloaded Router model snapshot is incomplete: {target_dir}")
    return target_dir


__all__ = [
    "NLI_ROUTER_MODEL_DIR",
    "NLI_ROUTER_MODEL_ID",
    "NLI_ROUTER_OPENVINO_DIR",
    "NLI_ROUTER_OPENVINO_MANIFEST",
    "NLI_ROUTER_OPENVINO_MODEL_XML",
    "PRETRAINED_MODELS_DIR",
    "download_router_model",
    "is_router_model_downloaded",
    "is_router_openvino_model_available",
    "router_model_path",
    "router_openvino_model_path",
]
