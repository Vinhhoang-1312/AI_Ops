"""Thread-safe realtime analyzer for camera-frame expression recognition."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from .config import DEFAULT_AVERAGE_WINDOW, DEFAULT_OPENVINO_MODEL_PATH, DEFAULT_SAMPLE_RATE_HZ, IMG_SIZE
from .model import FaceCropper, FaceRegion, OpenVINOExpressionClassifier
from .smoothing import FrameSampler, ProbabilityAverager
from .tracking import FaceTracker


@dataclass(frozen=True)
class FaceExpression:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]
    face_region: FaceRegion
    latency_ms: float = 0.0
    device: str = ""
    track_id: int | None = None


@dataclass(frozen=True)
class RealtimeState:
    label: str | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    top_k: list[tuple[str, float]] = field(default_factory=list)
    face_region: FaceRegion | None = None
    faces: list[FaceExpression] = field(default_factory=list)
    face_detected: bool = False
    sample_count: int = 0
    latency_ms: float = 0.0
    device: str = ""
    updated_at: float = 0.0
    status: str = "waiting"


class RealtimeEmotionAnalyzer:
    """Sample frames at 3 FPS and average the latest 3 probability vectors."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_OPENVINO_MODEL_PATH,
        imgsz: int = IMG_SIZE,
        device: str = "AUTO",
        sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
        average_window: int = DEFAULT_AVERAGE_WINDOW,
        face_crop: bool = True,
    ) -> None:
        self.classifier = OpenVINOExpressionClassifier(model_path=model_path, imgsz=imgsz, device=device)
        self.cropper = FaceCropper(enabled=face_crop)
        self.sampler = FrameSampler(sample_rate_hz=sample_rate_hz)
        self.averager = ProbabilityAverager(window_size=average_window)
        self.tracker = FaceTracker()
        self._state = RealtimeState(device=self.classifier.device_name)
        self._lock = Lock()

    def process_frame(self, frame_bgr: Any) -> Any:
        """Return an annotated frame while updating the latest analysis state."""
        annotated = frame_bgr.copy()
        now = monotonic()

        if self.sampler.should_sample(now):
            self._analyze_sample(frame_bgr, now)

        with self._lock:
            state = self._state
        return draw_overlay(annotated, state)

    def get_state(self) -> RealtimeState:
        with self._lock:
            return self._state

    def reset(self) -> None:
        with self._lock:
            self.sampler.reset()
            self.averager.clear()
            self.tracker.reset()
            self._state = RealtimeState(device=self.classifier.device_name)

    def _analyze_sample(self, frame_bgr: Any, now: float) -> None:
        try:
            face_crops = self.cropper.crop_all(frame_bgr)
            if not face_crops:
                self.tracker.assign([])
                with self._lock:
                    self._state = RealtimeState(
                        device=self.classifier.device_name,
                        updated_at=now,
                        status="no_face",
                    )
                return

            crops = [crop for crop, _region in face_crops]
            predictions = self.classifier.predict_batch(crops)
            first_prediction = predictions[0]
            probabilities = first_prediction.probabilities
            sample_count = len(predictions)

            if len(predictions) == 1:
                averaged = self.averager.update(first_prediction.probabilities)
                probabilities = averaged.probabilities
                sample_count = averaged.sample_count
                first_top_k = sorted(averaged.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
                face_results = [
                    FaceExpression(
                        label=averaged.label,
                        confidence=averaged.confidence,
                        top_k=first_top_k,
                        face_region=face_crops[0][1],
                        latency_ms=first_prediction.latency_ms,
                        device=first_prediction.device,
                    )
                ]
            else:
                self.averager.clear()
                face_results = [
                    FaceExpression(
                        label=prediction.label,
                        confidence=prediction.confidence,
                        top_k=prediction.top_k,
                        face_region=face_region,
                        latency_ms=prediction.latency_ms,
                        device=prediction.device,
                    )
                    for prediction, (_crop, face_region) in zip(predictions, face_crops)
                ]

            face_results = _with_tracking_ids(face_results, self.tracker)
            first = face_results[0]

            with self._lock:
                self._state = RealtimeState(
                    label=first.label,
                    confidence=first.confidence,
                    probabilities=probabilities,
                    top_k=first.top_k,
                    face_region=first.face_region,
                    faces=face_results,
                    face_detected=True,
                    sample_count=sample_count,
                    latency_ms=first.latency_ms,
                    device=first.device,
                    updated_at=now,
                    status="ok",
                )
        except Exception as exc:
            with self._lock:
                self._state = RealtimeState(
                    device=self.classifier.device_name,
                    updated_at=now,
                    status=f"error: {exc}",
                )


def draw_overlay(frame_bgr: Any, state: RealtimeState) -> Any:
    """Draw lightweight per-face labels with ASCII text for OpenCV compatibility."""
    import cv2

    h, w = frame_bgr.shape[:2]
    faces = state.faces or _legacy_face_list(state)
    for face in faces:
        region = face.face_region
        if not region.detected:
            continue
        x2, y2 = region.x + region.w, region.y + region.h
        cv2.rectangle(frame_bgr, (region.x, region.y), (x2, y2), (16, 118, 111), 2)
        identity = f"FACE #{face.track_id}" if face.track_id is not None else "FACE"
        label = f"{identity} {face.label.upper()} {face.confidence * 100:.0f}%"
        _draw_label(frame_bgr, label, region.x, max(0, region.y - 8))

    status = f"faces={len(faces)} | {state.device} | {state.latency_ms:.0f} ms"
    if state.status == "no_face":
        status = "NO FACE | Move face into camera"
    elif state.status.startswith("error:"):
        status = f"INFERENCE ERROR | {state.status[:70]}"

    cv2.putText(
        frame_bgr,
        status,
        (14, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )
    return frame_bgr


def _legacy_face_list(state: RealtimeState) -> list[FaceExpression]:
    if not state.face_region or not state.label:
        return []
    return [
        FaceExpression(
            label=state.label,
            confidence=state.confidence,
            top_k=state.top_k,
            face_region=state.face_region,
            latency_ms=state.latency_ms,
            device=state.device,
            track_id=1,
        )
    ]


def _with_tracking_ids(faces: list[FaceExpression], tracker: FaceTracker) -> list[FaceExpression]:
    track_ids = tracker.assign([face.face_region for face in faces])
    return [replace(face, track_id=track_id) for face, track_id in zip(faces, track_ids)]


def _draw_label(frame_bgr: Any, text: str, x: int, y: int) -> None:
    import cv2

    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    tw, th = text_size
    y = max(th + baseline + 4, y)
    cv2.rectangle(frame_bgr, (x, y - th - baseline - 6), (x + tw + 10, y + 2), (18, 25, 33), -1)
    cv2.putText(frame_bgr, text, (x + 5, y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
