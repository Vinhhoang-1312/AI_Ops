"""Three-thread IP-camera ingest pipeline: capture, gRPC inference, visualize."""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from fer_realtime.analyzer import FaceExpression, RealtimeState, draw_overlay
from fer_realtime.model import FaceRegion
from fer_realtime.smoothing import ProbabilityAverager
from fer_realtime.tracking import FaceTracker

from .client import OpenVINOGrpcClient


@dataclass(frozen=True)
class IngestConfig:
    stream_url: str
    model_server_address: str = "127.0.0.1:50051"
    model_timeout_seconds: float = 90.0
    batch_size: int = 16
    batch_timeout_seconds: float = 0.25
    frame_queue_size: int = 64
    result_queue_size: int = 64
    average_window: int = 3
    face_crop: bool = True
    show_window: bool = True
    window_name: str = "FER OpenVINO gRPC Ingest"


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    frame_bgr: Any
    captured_at: float


@dataclass(frozen=True)
class VisualFrame:
    captured: CapturedFrame
    state: RealtimeState


class StreamIngestPipeline:
    def __init__(self, config: IngestConfig) -> None:
        self.config = config
        self.client = OpenVINOGrpcClient(config.model_server_address, timeout_seconds=config.model_timeout_seconds)
        self.frame_queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=config.frame_queue_size)
        self.visual_queue: queue.Queue[VisualFrame] = queue.Queue(maxsize=config.result_queue_size)
        self.averager = ProbabilityAverager(window_size=config.average_window)
        self.tracker = FaceTracker()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self._latest_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_summary = "waiting"
        self._latest_state = RealtimeState()
        self._latest_frame_id: int | None = None

    def run(self) -> None:
        self.start()
        try:
            self.join_forever()
        finally:
            self.stop()
            for thread in self.threads:
                thread.join(timeout=2.0)

    def start(self) -> None:
        if self.threads:
            return
        self.threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._inference_loop, name="inference", daemon=True),
            threading.Thread(target=self._visualize_loop, name="visualize", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def join_forever(self) -> None:
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()

    def latest_jpeg(self) -> bytes | None:
        with self._latest_lock:
            return self._latest_jpeg

    def latest_summary(self) -> str:
        with self._latest_lock:
            return self._latest_summary

    def latest_state_snapshot(self) -> dict[str, Any]:
        with self._latest_lock:
            return _state_payload(self._latest_frame_id, self._latest_state, self._latest_summary)

    def latest_state(self) -> RealtimeState:
        with self._latest_lock:
            return self._latest_state

    def _capture_loop(self) -> None:
        import cv2

        frame_id = 0
        source = _opencv_source(self.config.stream_url)
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(source) if isinstance(source, int) else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self._set_summary(f"camera_not_open stream={self.config.stream_url}")
                time.sleep(2.0)
                continue

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                self._put_latest(self.frame_queue, CapturedFrame(frame_id, frame, time.time()))
                frame_id += 1
            cap.release()
            time.sleep(1.0)

    def _inference_loop(self) -> None:
        while not self.stop_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue

            try:
                payload = self.client.infer_batch(
                    [item.frame_bgr for item in batch],
                    [item.frame_id for item in batch],
                    face_crop=self.config.face_crop,
                )
                result_by_id = {str(item["frame_id"]): item for item in payload.get("results", [])}
                for captured in batch:
                    result = result_by_id.get(str(captured.frame_id), {})
                    state = self._state_from_result(result)
                    self._put_latest(self.visual_queue, VisualFrame(captured, state))
            except Exception as exc:
                state = RealtimeState(status=f"error: {exc}")
                self._put_latest(self.visual_queue, VisualFrame(batch[-1], state))

    def _visualize_loop(self) -> None:
        import cv2

        while not self.stop_event.is_set():
            try:
                item = self.visual_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            annotated = draw_overlay(item.captured.frame_bgr.copy(), item.state)
            self._publish_latest(annotated, _state_summary(item.captured.frame_id, item.state), item.state, item.captured.frame_id)
            if self.config.show_window:
                cv2.imshow(self.config.window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop()

        cv2.destroyAllWindows()

    def _collect_batch(self) -> list[CapturedFrame]:
        batch = []
        deadline = time.monotonic() + self.config.batch_timeout_seconds
        while len(batch) < self.config.batch_size and not self.stop_event.is_set():
            timeout = max(0.0, deadline - time.monotonic())
            try:
                batch.append(self.frame_queue.get(timeout=timeout))
            except queue.Empty:
                break
        return batch

    def _state_from_result(self, result: dict[str, Any]) -> RealtimeState:
        if result.get("status") != "ok":
            self.tracker.assign([])
            return RealtimeState(status=str(result.get("status", "error")))

        faces = [_face_expression_from_dict(item) for item in result.get("faces", []) if isinstance(item, dict)]
        faces = [face for face in faces if face is not None]
        if faces:
            faces = _with_tracking_ids(faces, self.tracker)
            first = faces[0]
            return RealtimeState(
                label=first.label,
                confidence=first.confidence,
                probabilities={str(k): float(v) for k, v in result.get("probabilities", {}).items()},
                top_k=first.top_k,
                face_region=first.face_region,
                faces=faces,
                face_detected=True,
                sample_count=len(faces),
                latency_ms=first.latency_ms,
                device=first.device,
                updated_at=time.monotonic(),
                status="ok",
            )

        probabilities = {str(k): float(v) for k, v in result.get("probabilities", {}).items()}
        self.tracker.assign([])
        averaged = self.averager.update(probabilities)
        top_k = sorted(averaged.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
        label = averaged.label
        confidence = averaged.confidence
        face_region = _face_region_from_dict(result.get("face_region"))
        return RealtimeState(
            label=label,
            confidence=confidence,
            probabilities=averaged.probabilities,
            top_k=top_k,
            face_region=face_region,
            face_detected=bool(face_region and face_region.detected),
            sample_count=averaged.sample_count,
            latency_ms=float(result.get("latency_ms", 0.0)),
            device=str(result.get("device", "")),
            updated_at=time.monotonic(),
            status="ok",
        )

    @staticmethod
    def _put_latest(target_queue: queue.Queue, item: Any) -> None:
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
            target_queue.put_nowait(item)

    def _publish_latest(self, frame_bgr: Any, summary: str, state: RealtimeState, frame_id: int) -> None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self._latest_lock:
            self._latest_jpeg = encoded.tobytes()
            self._latest_summary = summary
            self._latest_state = state
            self._latest_frame_id = frame_id

    def _set_summary(self, summary: str) -> None:
        with self._latest_lock:
            self._latest_summary = summary


def config_from_env() -> IngestConfig:
    return IngestConfig(
        stream_url=os.environ.get("IP_CAMERA_URL", "0"),
        model_server_address=os.environ.get("MODEL_SERVER_ADDRESS", "127.0.0.1:50051"),
        model_timeout_seconds=float(os.environ.get("MODEL_SERVER_TIMEOUT_SECONDS", "90")),
        batch_size=int(os.environ.get("INGEST_BATCH_SIZE", "16")),
        batch_timeout_seconds=float(os.environ.get("INGEST_BATCH_TIMEOUT_SECONDS", "0.25")),
        frame_queue_size=int(os.environ.get("INGEST_FRAME_QUEUE_SIZE", "64")),
        result_queue_size=int(os.environ.get("INGEST_RESULT_QUEUE_SIZE", "64")),
        average_window=int(os.environ.get("INGEST_AVERAGE_WINDOW", "3")),
        face_crop=os.environ.get("INGEST_FACE_CROP", "true").lower() in {"1", "true", "yes", "on"},
        show_window=os.environ.get("INGEST_SHOW_WINDOW", "false").lower() in {"1", "true", "yes", "on"},
    )


def main() -> None:
    pipeline = StreamIngestPipeline(config_from_env())
    signal.signal(signal.SIGTERM, lambda *_: pipeline.stop())
    pipeline.run()


def _opencv_source(source: str) -> int | str:
    stripped = str(source).strip()
    return int(stripped) if stripped.isdigit() else stripped


def _face_region_from_dict(value: Any) -> FaceRegion | None:
    if not isinstance(value, dict):
        return None
    return FaceRegion(
        x=int(value.get("x", 0)),
        y=int(value.get("y", 0)),
        w=int(value.get("w", 0)),
        h=int(value.get("h", 0)),
        detected=bool(value.get("detected", False)),
    )


def _face_expression_from_dict(value: dict[str, Any]) -> FaceExpression | None:
    region = _face_region_from_dict(value.get("face_region"))
    if region is None:
        return None
    top_k = [(str(label), float(score)) for label, score in value.get("top_k", [])]
    return FaceExpression(
        label=str(value.get("label", "unknown")),
        confidence=float(value.get("confidence", 0.0)),
        top_k=top_k,
        face_region=region,
        latency_ms=float(value.get("latency_ms", 0.0)),
        device=str(value.get("device", "")),
        track_id=_optional_int(value.get("track_id")),
    )


def _state_summary(frame_id: int, state: RealtimeState) -> str:
    if state.faces:
        labels = " | ".join(
            f"face#{face.track_id if face.track_id is not None else idx + 1}:{face.label}={face.confidence * 100:.1f}%"
            for idx, face in enumerate(state.faces[:3])
        )
        return f"frame={frame_id} faces={len(state.faces)} {labels} latency_ms={state.latency_ms:.1f}"
    if not state.top_k:
        return f"frame={frame_id} status={state.status}"
    top_two = " | ".join(f"{label}={score * 100:.1f}%" for label, score in state.top_k[:2])
    return f"frame={frame_id} {top_two} latency_ms={state.latency_ms:.1f}"


def _state_payload(frame_id: int | None, state: RealtimeState, summary: str) -> dict[str, Any]:
    top_k = [{"label": label, "score": score} for label, score in state.top_k[:3]]
    faces = [
        {
            "label": face.label,
            "track_id": face.track_id,
            "confidence": face.confidence,
            "top_k": [{"label": label, "score": score} for label, score in face.top_k[:3]],
            "face_region": {
                "x": face.face_region.x,
                "y": face.face_region.y,
                "w": face.face_region.w,
                "h": face.face_region.h,
                "detected": face.face_region.detected,
            },
            "latency_ms": face.latency_ms,
            "device": face.device,
        }
        for face in state.faces
    ]

    return {
        "frame_id": frame_id,
        "status": state.status,
        "summary": summary,
        "label": state.label,
        "confidence": state.confidence,
        "top_k": top_k,
        "faces": faces,
        "sample_count": state.sample_count,
        "latency_ms": state.latency_ms,
        "device": state.device,
        "face_detected": state.face_detected,
    }


def _with_tracking_ids(faces: list[FaceExpression], tracker: FaceTracker) -> list[FaceExpression]:
    track_ids = tracker.assign([face.face_region for face in faces])
    return [replace(face, track_id=track_id) for face, track_id in zip(faces, track_ids)]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
