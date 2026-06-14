"""gRPC client used by stream ingest to call the OpenVINO model server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .codec import encode_bgr_to_jpeg_b64
from .protocol import HEALTH_PATH, INFER_PATH, dumps, loads


@dataclass(frozen=True)
class OpenVINOGrpcClient:
    address: str = "127.0.0.1:50051"
    timeout_seconds: float = 30.0
    jpeg_quality: int = 85
    max_message_mb: int = 64

    def infer_batch(self, frames_bgr: Sequence[Any], frame_ids: Sequence[int | str], face_crop: bool = True) -> dict[str, Any]:
        if not frames_bgr:
            return {"ok": True, "batch_size": 0, "predicted": 0, "results": []}
        if len(frames_bgr) != len(frame_ids):
            raise ValueError("frames_bgr and frame_ids must have the same length.")

        instances = []
        for frame, frame_id in zip(frames_bgr, frame_ids):
            instances.append(
                {
                    "frame_id": str(frame_id),
                    "image_b64": encode_bgr_to_jpeg_b64(frame, quality=self.jpeg_quality),
                }
            )

        return self._call(INFER_PATH, {"instances": instances, "face_crop": bool(face_crop)})

    def health(self) -> dict[str, Any]:
        return self._call(HEALTH_PATH, {})

    def _call(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import grpc

        channel_options = [
            ("grpc.max_send_message_length", self.max_message_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", self.max_message_mb * 1024 * 1024),
        ]
        with grpc.insecure_channel(self.address, options=channel_options) as channel:
            method = channel.unary_unary(path, request_serializer=dumps, response_deserializer=loads)
            return method(payload, timeout=self.timeout_seconds)
