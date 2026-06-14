"""Thread-safe realtime analyzer used by Streamlit/WebRTC callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from .config import DEFAULT_AVERAGE_WINDOW, DEFAULT_OPENVINO_MODEL_PATH, DEFAULT_SAMPLE_RATE_HZ, IMG_SIZE
from .emotion_policy import AgentCue, cue_for_expression
from .model import FaceCropper, FaceRegion, OpenVINOExpressionClassifier
from .smoothing import FrameSampler, ProbabilityAverager


@dataclass(frozen=True)
class RealtimeState:
    label: str | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    top_k: list[tuple[str, float]] = field(default_factory=list)
    cue: AgentCue = field(default_factory=lambda: cue_for_expression(None))
    face_region: FaceRegion | None = None
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
            self._state = RealtimeState(device=self.classifier.device_name)

    def _analyze_sample(self, frame_bgr: Any, now: float) -> None:
        try:
            crop, face_region = self.cropper.crop(frame_bgr)
            if crop is None:
                with self._lock:
                    self._state = RealtimeState(
                        cue=cue_for_expression(None),
                        device=self.classifier.device_name,
                        updated_at=now,
                        status="no_face",
                    )
                return

            prediction = self.classifier.predict_frame(crop)
            averaged = self.averager.update(prediction.probabilities)
            top_k = sorted(averaged.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
            cue = cue_for_expression(averaged.label, averaged.confidence)

            with self._lock:
                self._state = RealtimeState(
                    label=averaged.label,
                    confidence=averaged.confidence,
                    probabilities=averaged.probabilities,
                    top_k=top_k,
                    cue=cue,
                    face_region=face_region,
                    face_detected=bool(face_region and face_region.detected),
                    sample_count=averaged.sample_count,
                    latency_ms=prediction.latency_ms,
                    device=prediction.device,
                    updated_at=now,
                    status="ok",
                )
        except Exception as exc:
            with self._lock:
                self._state = RealtimeState(
                    cue=cue_for_expression(None),
                    device=self.classifier.device_name,
                    updated_at=now,
                    status=f"error: {exc}",
                )


def draw_overlay(frame_bgr: Any, state: RealtimeState) -> Any:
    """Draw lightweight camera overlay with ASCII text for OpenCV compatibility."""
    import cv2

    h, w = frame_bgr.shape[:2]
    if state.face_region and state.face_region.detected:
        region = state.face_region
        cv2.rectangle(frame_bgr, (region.x, region.y), (region.x + region.w, region.y + region.h), (16, 118, 111), 2)

    if state.top_k:
        top_two = state.top_k[:2]
        text = " | ".join(f"{label.upper()} {score * 100:.0f}%" for label, score in top_two)
    else:
        label = state.label.upper() if state.label else "WAITING"
        confidence = f"{state.confidence * 100:.0f}%" if state.label else ""
        text = f"{label} {confidence}".strip()
    subtitle = "3 fps sampling | avg latest 3 frames"
    if state.status == "no_face":
        text = "NO FACE"
        subtitle = "Move face into camera"
    elif state.status.startswith("error:"):
        text = "INFERENCE ERROR"
        subtitle = state.status[:70]

    pad = 14
    title_scale = 0.68 if len(text) > 24 else 0.82
    cv2.rectangle(frame_bgr, (0, 0), (w, 72), (18, 25, 33), -1)
    cv2.putText(frame_bgr, text, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, subtitle, (pad, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (196, 207, 218), 1, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        f"{state.device} | {state.latency_ms:.0f} ms",
        (max(pad, w - 250), h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )
    return frame_bgr
