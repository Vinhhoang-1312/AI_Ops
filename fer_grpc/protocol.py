"""Small JSON-over-gRPC protocol helpers.

gRPC supplies the HTTP/2 transport, deadlines, and channel behavior. The payload
is UTF-8 JSON so this lab can stay lightweight without generated protobuf files.
"""

from __future__ import annotations

import json
from typing import Any

SERVICE_NAME = "fer.ExpressionService"
INFER_METHOD = "InferBatch"
HEALTH_METHOD = "Health"
INFER_PATH = f"/{SERVICE_NAME}/{INFER_METHOD}"
HEALTH_PATH = f"/{SERVICE_NAME}/{HEALTH_METHOD}"


def dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def loads(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {}
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gRPC payload must be a JSON object.")
    return value
