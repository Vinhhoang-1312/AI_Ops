import unittest
from unittest.mock import patch

from fer_grpc.client import OpenVINOGrpcClient
from fer_grpc.protocol import dumps, loads
from fer_grpc import stream_ingest_server
from fer_grpc.stream_ingest import IngestConfig, _opencv_source


class GrpcProtocolTest(unittest.TestCase):
    def test_json_payload_roundtrip(self):
        payload = {"instances": [{"frame_id": "1", "image_b64": "abc"}], "face_crop": True}

        self.assertEqual(loads(dumps(payload)), payload)


class GrpcClientTest(unittest.TestCase):
    def test_empty_batch_short_circuits_without_grpc(self):
        client = OpenVINOGrpcClient("example.invalid:50051")

        payload = client.infer_batch([], [])

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["batch_size"], 0)
        self.assertEqual(payload["results"], [])


class IngestConfigTest(unittest.TestCase):
    def test_default_batch_size_is_sixteen(self):
        config = IngestConfig(stream_url="rtsp://camera/stream")

        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.model_server_address, "127.0.0.1:50051")
        self.assertEqual(config.model_timeout_seconds, 90.0)

    def test_local_camera_index_is_supported(self):
        self.assertEqual(_opencv_source("0"), 0)
        self.assertEqual(_opencv_source("http://192.168.1.50:8080/video"), "http://192.168.1.50:8080/video")


class StreamIngestServerTest(unittest.TestCase):
    def test_save_endpoint_persists_latest_pipeline_state(self):
        class DummyPipeline:
            def latest_state(self):
                return type(
                    "DummyState",
                    (),
                    {
                        "status": "ok",
                        "label": "neutral",
                        "confidence": 0.8,
                        "sample_count": 3,
                        "latency_ms": 12.0,
                        "device": "OpenVINO CPU",
                        "top_k": [("neutral", 0.8), ("sad", 0.2)],
                        "probabilities": {"neutral": 0.8, "sad": 0.2},
                    },
                )()

        original_pipeline = stream_ingest_server.pipeline
        stream_ingest_server.pipeline = DummyPipeline()
        try:
            with patch("fer_grpc.stream_ingest_server.save_state_snapshot", return_value=42):
                payload = stream_ingest_server.save()
        finally:
            stream_ingest_server.pipeline = original_pipeline

        self.assertEqual(payload, {"ok": True, "id": 42})


if __name__ == "__main__":
    unittest.main()
