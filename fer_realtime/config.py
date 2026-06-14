"""Project-local defaults for OpenVINO realtime inference."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_OPENVINO_MODEL_PATH = PROJECT_ROOT / "models" / "fer_expression_yolo26n_openvino"
DEFAULT_OPENVINO_MODEL_PATH = PUBLISHED_OPENVINO_MODEL_PATH
# Backward-compatible name for existing UI/session code. It now points to OpenVINO IR.
DEFAULT_MODEL_PATH = DEFAULT_OPENVINO_MODEL_PATH
ASSETS_DIR = PROJECT_ROOT / "assets"
ROBOT_SVG_PATH = ASSETS_DIR / "support_robot.svg"
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_DB_PATH = DATA_DIR / "fer_realtime_history.sqlite3"

IMG_SIZE = 224
DEFAULT_SAMPLE_RATE_HZ = 3.0
DEFAULT_AVERAGE_WINDOW = 3

FER_CLASS_NAMES = (
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "sleepy",
    "surprise",
)
