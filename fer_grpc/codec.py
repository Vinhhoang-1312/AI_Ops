"""Image encode/decode helpers shared by gRPC client and server."""

from __future__ import annotations

import base64
from typing import Any


def encode_bgr_to_jpeg_b64(frame_bgr: Any, quality: int = 85) -> str:
    import cv2

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_params)
    if not ok:
        raise ValueError("Could not encode frame as JPEG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_jpeg_b64_to_bgr(image_b64: str) -> Any:
    import cv2
    import numpy as np

    image_bytes = base64.b64decode(image_b64.encode("ascii"), validate=True)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG image.")
    return frame
