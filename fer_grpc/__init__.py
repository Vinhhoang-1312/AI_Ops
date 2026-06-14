"""OpenVINO gRPC serving and IP-camera ingest helpers."""

from .client import OpenVINOGrpcClient
from .stream_ingest import IngestConfig, StreamIngestPipeline

__all__ = [
    "IngestConfig",
    "OpenVINOGrpcClient",
    "StreamIngestPipeline",
]
