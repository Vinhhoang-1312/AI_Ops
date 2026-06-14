"""OpenVINO FER model server exposed through gRPC over HTTP/2."""

from __future__ import annotations

import argparse
import logging
import signal
from concurrent import futures
from pathlib import Path
from threading import Lock
from typing import Any

from fer_realtime.config import DEFAULT_OPENVINO_MODEL_PATH, IMG_SIZE
from fer_realtime.model import FaceCropper, FaceRegion, OpenVINOExpressionClassifier

from .codec import decode_jpeg_b64_to_bgr
from .protocol import HEALTH_METHOD, INFER_METHOD, SERVICE_NAME, dumps, loads

LOGGER = logging.getLogger(__name__)


class ExpressionService:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_OPENVINO_MODEL_PATH,
        device: str = "AUTO",
        imgsz: int = IMG_SIZE,
        top_k: int = 3,
        face_margin: float = 0.18,
    ) -> None:
        self.classifier = OpenVINOExpressionClassifier(model_path=model_path, imgsz=imgsz, device=device, top_k=top_k)
        self.cropper = FaceCropper(enabled=True, margin=face_margin)
        self._infer_lock = Lock()

    def health(self, request: dict[str, Any], context: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "runtime": "OpenVINO",
            "model_path": _ascii_safe(str(self.classifier.model_path)),
            "device": self.classifier.device_name,
        }

    def infer_batch(self, request: dict[str, Any], context: Any) -> dict[str, Any]:
        instances = request.get("instances", [])
        if not isinstance(instances, list):
            return {"ok": False, "error": "instances must be a list.", "results": []}

        face_crop = bool(request.get("face_crop", True))
        crops: list[Any] = []
        pending: list[tuple[int, str, FaceRegion | None]] = []
        results: list[dict[str, Any] | None] = [None] * len(instances)

        for idx, instance in enumerate(instances):
            frame_id = str(instance.get("frame_id", idx)) if isinstance(instance, dict) else str(idx)
            try:
                image_b64 = instance["image_b64"]
                frame = decode_jpeg_b64_to_bgr(str(image_b64))
                crop, face_region = self._crop_frame(frame, enabled=face_crop)
                if crop is None:
                    results[idx] = {"frame_id": frame_id, "status": "no_face"}
                    continue
                crops.append(crop)
                pending.append((idx, frame_id, face_region))
            except Exception as exc:
                results[idx] = {"frame_id": frame_id, "status": "decode_error", "error": str(exc)}

        if crops:
            with self._infer_lock:
                predictions = self.classifier.predict_batch(crops)
            for prediction, (idx, frame_id, face_region) in zip(predictions, pending):
                results[idx] = {
                    "frame_id": frame_id,
                    "status": "ok",
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "probabilities": prediction.probabilities,
                    "top_k": prediction.top_k,
                    "latency_ms": prediction.latency_ms,
                    "device": prediction.device,
                    "face_region": _face_region_dict(face_region),
                }

        normalized_results = [item for item in results if item is not None]
        return {
            "ok": True,
            "runtime": "OpenVINO",
            "batch_size": len(instances),
            "predicted": sum(1 for item in normalized_results if item.get("status") == "ok"),
            "results": normalized_results,
            "device": self.classifier.device_name,
        }

    def _crop_frame(self, frame_bgr: Any, enabled: bool) -> tuple[Any | None, FaceRegion | None]:
        if not enabled:
            h, w = frame_bgr.shape[:2]
            return frame_bgr, FaceRegion(0, 0, w, h, detected=False)
        return self.cropper.crop(frame_bgr)


def create_grpc_server(service: ExpressionService, address: str, max_workers: int = 4) -> Any:
    import grpc

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    method_handlers = {
        INFER_METHOD: grpc.unary_unary_rpc_method_handler(
            service.infer_batch,
            request_deserializer=loads,
            response_serializer=dumps,
        ),
        HEALTH_METHOD: grpc.unary_unary_rpc_method_handler(
            service.health,
            request_deserializer=loads,
            response_serializer=dumps,
        ),
    }
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE_NAME, method_handlers),))
    server.add_insecure_port(address)
    return server


def serve(
    model_path: str | Path = DEFAULT_OPENVINO_MODEL_PATH,
    device: str = "AUTO",
    address: str = "127.0.0.1:50051",
    max_workers: int = 4,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    service = ExpressionService(model_path=model_path, device=device)
    server = create_grpc_server(service, address=address, max_workers=max_workers)
    server.start()
    LOGGER.info("OpenVINO gRPC model server listening on %s", address)
    LOGGER.info("Model: %s | Device: %s", _ascii_safe(str(service.classifier.model_path)), service.classifier.device_name)
    signal.signal(signal.SIGTERM, lambda *_: server.stop(grace=2))
    server.wait_for_termination()


def _face_region_dict(region: FaceRegion | None) -> dict[str, Any] | None:
    if region is None:
        return None
    return {
        "x": region.x,
        "y": region.y,
        "w": region.w,
        "h": region.h,
        "detected": region.detected,
    }


def _ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="backslashreplace").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenVINO FER gRPC model server.")
    parser.add_argument("--model", default=str(DEFAULT_OPENVINO_MODEL_PATH), help="OpenVINO model directory or .xml path.")
    parser.add_argument("--device", default="AUTO", help="OpenVINO device: AUTO, CPU, GPU, NPU, GPU.0, ...")
    parser.add_argument("--address", default="127.0.0.1:50051", help="gRPC bind address.")
    parser.add_argument("--max-workers", type=int, default=4, help="gRPC worker threads.")
    args = parser.parse_args()
    serve(model_path=args.model, device=args.device, address=args.address, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
