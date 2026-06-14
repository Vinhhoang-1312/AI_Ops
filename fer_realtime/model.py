"""OpenVINO model loading and frame inference for realtime FER."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .config import DEFAULT_OPENVINO_MODEL_PATH, FER_CLASS_NAMES, IMG_SIZE


@dataclass(frozen=True)
class FaceRegion:
    x: int
    y: int
    w: int
    h: int
    detected: bool


@dataclass(frozen=True)
class InferenceResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    top_k: list[tuple[str, float]]
    latency_ms: float
    device: str


def validate_openvino_model_path(model_path: str | Path) -> Path:
    """Return an OpenVINO IR XML path from a model directory or explicit XML path."""
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"OpenVINO model not found: {path}")

    if path.is_dir():
        xml_files = sorted(path.glob("*.xml"))
        if not xml_files:
            raise ValueError(f"OpenVINO model directory has no .xml file: {path}")
        if len(xml_files) > 1:
            preferred = path / "best.xml"
            return preferred if preferred.exists() else xml_files[0]
        return xml_files[0]

    if path.suffix.lower() != ".xml":
        raise ValueError(
            "OpenVINO realtime inference expects an IR .xml file or a directory containing one. "
            "Use models/fer_expression_yolo26n_openvino after running the training notebook."
        )
    return path


def select_openvino_device(preferred: str = "AUTO") -> tuple[str, str]:
    """Normalize an OpenVINO device name and return a display string."""
    choice = (preferred or "AUTO").strip().upper()
    if choice == "AUTO":
        return "AUTO", "OpenVINO AUTO"
    if choice in {"CPU", "GPU", "NPU"} or choice.startswith(("GPU.", "NPU.")):
        return choice, f"OpenVINO {choice}"
    raise ValueError("device must be one of: AUTO, CPU, GPU, NPU, GPU.<index>, or NPU.<index>.")


class FaceCropper:
    """Detect faces and return square crops for the classifier."""

    def __init__(self, enabled: bool = True, margin: float = 0.18) -> None:
        self.enabled = enabled
        self.margin = float(margin)
        self._cascade: Any | None = None

    def crop(self, frame_bgr: Any) -> tuple[Any | None, FaceRegion | None]:
        crops = self.crop_all(frame_bgr, max_faces=1)
        if not crops:
            return None, None
        return crops[0]

    def crop_all(self, frame_bgr: Any, max_faces: int | None = None) -> list[tuple[Any, FaceRegion]]:
        if not self.enabled:
            h, w = frame_bgr.shape[:2]
            return [(frame_bgr, FaceRegion(0, 0, w, h, detected=False))]

        cascade = self._load_cascade()
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return []

        sorted_faces = sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)
        selected_faces = sorted_faces[:max_faces] if max_faces is not None else sorted_faces
        crops: list[tuple[Any, FaceRegion]] = []
        for x, y, w, h in selected_faces:
            crop_region = self._square_with_margin(int(x), int(y), int(w), int(h), frame_bgr.shape[:2])
            x1, y1, side = crop_region
            crop = frame_bgr[y1 : y1 + side, x1 : x1 + side]
            if crop.size == 0:
                continue
            crops.append((crop, FaceRegion(x1, y1, side, side, detected=True)))
        return crops

    def _load_cascade(self) -> Any:
        if self._cascade is not None:
            return self._cascade

        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load OpenCV face cascade: {cascade_path}")
        self._cascade = cascade
        return cascade

    def _square_with_margin(self, x: int, y: int, w: int, h: int, frame_shape: tuple[int, int]) -> tuple[int, int, int]:
        frame_h, frame_w = frame_shape
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        side = int(round(max(w, h) * (1.0 + 2.0 * self.margin)))
        side = max(side, 48)
        side = min(side, frame_w, frame_h)

        x1 = int(round(center_x - side / 2.0))
        y1 = int(round(center_y - side / 2.0))
        x1 = max(0, min(x1, frame_w - side))
        y1 = max(0, min(y1, frame_h - side))
        return x1, y1, side


class OpenVINOExpressionClassifier:
    """Lazy OpenVINO classifier wrapper for BGR frames."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_OPENVINO_MODEL_PATH,
        imgsz: int = IMG_SIZE,
        device: str = "AUTO",
        top_k: int = 3,
        enable_dynamic_batch: bool = True,
    ) -> None:
        self.model_path = validate_openvino_model_path(model_path)
        self.imgsz = int(imgsz)
        self.top_k = int(top_k)
        self.device, self.device_name = select_openvino_device(device)
        self.enable_dynamic_batch = bool(enable_dynamic_batch)
        self.class_names = _load_openvino_names(self.model_path) or {idx: name for idx, name in enumerate(FER_CLASS_NAMES)}
        self._core: Any | None = None
        self._compiled_model: Any | None = None
        self._input_layer: Any | None = None
        self._output_layer: Any | None = None
        self._dynamic_batch_enabled = False

    def load(self) -> None:
        if self._compiled_model is not None:
            return

        try:
            from openvino import Core
        except ImportError as exc:
            raise ImportError(
                "OpenVINO is required for realtime inference. Install it in the active environment with "
                "python -m pip install openvino==2025.0.0."
            ) from exc

        core = Core()
        model = self._read_model(core, try_dynamic_batch=self.enable_dynamic_batch)
        compiled = core.compile_model(model, self.device)

        self._core = core
        self._compiled_model = compiled
        self._input_layer = compiled.inputs[0]
        self._output_layer = compiled.outputs[0]
        self.device_name = self._compiled_device_name()

    def predict_frame(self, frame_bgr: Any) -> InferenceResult:
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: Sequence[Any]) -> list[InferenceResult]:
        if not frames_bgr:
            return []

        self.load()
        assert self._compiled_model is not None

        tensors = [self._preprocess_frame(frame) for frame in frames_bgr]
        start = perf_counter()
        raw_outputs = self._infer_tensors(tensors)
        per_image_latency_ms = ((perf_counter() - start) * 1000.0) / len(frames_bgr)
        return [self._parse_vector(raw, per_image_latency_ms) for raw in raw_outputs]

    def _read_model(self, core: Any, try_dynamic_batch: bool) -> Any:
        model = core.read_model(str(self.model_path))
        if not try_dynamic_batch:
            return model

        try:
            dynamic_model = core.read_model(str(self.model_path))
            input_name = dynamic_model.inputs[0].get_any_name()
            dynamic_model.reshape({input_name: [-1, 3, self.imgsz, self.imgsz]})
            self._dynamic_batch_enabled = True
            return dynamic_model
        except Exception:
            self._dynamic_batch_enabled = False
            return model

    def _preprocess_frame(self, frame_bgr: Any) -> Any:
        import cv2
        import numpy as np

        resized = cv2.resize(frame_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype("float32") / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        return tensor[None, :, :, :]

    def _infer_tensors(self, tensors: Sequence[Any]) -> list[Any]:
        import numpy as np

        if self._dynamic_batch_enabled and len(tensors) > 1:
            try:
                batched = np.concatenate(list(tensors), axis=0)
                raw = self._infer_array(batched)
                raw_array = np.asarray(raw)
                if raw_array.ndim >= 2 and raw_array.shape[0] == len(tensors):
                    return [raw_array[idx] for idx in range(len(tensors))]
            except Exception:
                pass

        raw_outputs = []
        for tensor in tensors:
            raw = np.asarray(self._infer_array(tensor))
            raw_outputs.append(raw[0] if raw.ndim >= 2 and raw.shape[0] == 1 else raw)
        return raw_outputs

    def _infer_array(self, tensor: Any) -> Any:
        assert self._compiled_model is not None
        assert self._input_layer is not None
        assert self._output_layer is not None

        infer_request = self._compiled_model.create_infer_request()
        outputs = infer_request.infer({self._input_layer: tensor})
        return outputs[self._output_layer]

    def _parse_vector(self, vector: Any, latency_ms: float) -> InferenceResult:
        probabilities = _probabilities_from_output(vector, self.class_names)
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[: self.top_k]

        return InferenceResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            top_k=ranked,
            latency_ms=latency_ms,
            device=self.device_name,
        )

    def _compiled_device_name(self) -> str:
        if self._compiled_model is None:
            return self.device_name
        try:
            execution_devices = self._compiled_model.get_property("EXECUTION_DEVICES")
            if execution_devices:
                return "OpenVINO " + "+".join(str(device) for device in execution_devices)
        except Exception:
            pass
        return f"OpenVINO {self.device}"


def _load_openvino_names(xml_path: Path) -> dict[int, str]:
    metadata_path = xml_path.with_name("metadata.yaml")
    if not metadata_path.exists():
        return {}

    names: dict[int, str] = {}
    in_names = False
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            continue
        if in_names and line and not line.startswith((" ", "\t")):
            break
        if not in_names or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip().isdigit():
            names[int(key.strip())] = value.strip().strip("'\"")
    return names


def _probabilities_from_output(vector: Any, names: dict[int, str]) -> dict[str, float]:
    import numpy as np

    values = np.asarray(vector, dtype="float32").reshape(-1)
    if values.size == 0:
        raise ValueError("OpenVINO model returned an empty output vector.")

    if np.all(np.isfinite(values)) and np.all(values >= 0.0) and 0.98 <= float(values.sum()) <= 1.02:
        probabilities = values / max(float(values.sum()), 1e-12)
    else:
        shifted = values - float(values.max())
        exp_values = np.exp(np.clip(shifted, -80.0, 80.0))
        probabilities = exp_values / max(float(exp_values.sum()), 1e-12)

    return {names.get(idx, str(idx)): float(value) for idx, value in enumerate(probabilities)}
