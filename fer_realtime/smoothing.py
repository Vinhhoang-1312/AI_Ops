"""Frame sampling and probability averaging for stable realtime FER output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Mapping


@dataclass(frozen=True)
class AveragedPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    sample_count: int


class FrameSampler:
    """Gate high-FPS video so inference runs at a fixed sample rate."""

    def __init__(self, sample_rate_hz: float = 3.0) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        self.interval_seconds = 1.0 / float(sample_rate_hz)
        self._last_sample_time: float | None = None

    def should_sample(self, now: float | None = None) -> bool:
        now = monotonic() if now is None else float(now)
        if self._last_sample_time is None:
            self._last_sample_time = now
            return True
        if now - self._last_sample_time >= self.interval_seconds:
            self._last_sample_time = now
            return True
        return False

    def reset(self) -> None:
        self._last_sample_time = None


class ProbabilityAverager:
    """Average the latest N probability vectors, preserving class keys."""

    def __init__(self, window_size: int = 3) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self._window: deque[dict[str, float]] = deque(maxlen=int(window_size))

    @property
    def sample_count(self) -> int:
        return len(self._window)

    def clear(self) -> None:
        self._window.clear()

    def update(self, probabilities: Mapping[str, float]) -> AveragedPrediction:
        cleaned = {str(k): float(v) for k, v in probabilities.items()}
        if not cleaned:
            raise ValueError("probabilities must not be empty.")

        self._window.append(cleaned)
        averaged = self.current_probabilities()
        label, confidence = max(averaged.items(), key=lambda item: item[1])
        return AveragedPrediction(
            label=label,
            confidence=confidence,
            probabilities=averaged,
            sample_count=len(self._window),
        )

    def current_probabilities(self) -> dict[str, float]:
        if not self._window:
            return {}

        keys = sorted({key for sample in self._window for key in sample})
        averaged = {}
        for key in keys:
            averaged[key] = sum(sample.get(key, 0.0) for sample in self._window) / len(self._window)
        total = sum(averaged.values())
        if total > 0:
            averaged = {key: value / total for key, value in averaged.items()}
        return averaged
