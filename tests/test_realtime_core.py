import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from fer_realtime.emotion_policy import cue_for_expression
from fer_realtime.history import fetch_recent_snapshots, save_state_snapshot
from fer_realtime.model import OpenVINOExpressionClassifier, validate_openvino_model_path
from fer_realtime.smoothing import FrameSampler, ProbabilityAverager


@dataclass
class DummyState:
    label: str = "neutral"
    confidence: float = 0.8
    sample_count: int = 3
    latency_ms: float = 12.5
    device: str = "CPU"
    status: str = "ok"
    top_k: list[tuple[str, float]] = field(default_factory=lambda: [("neutral", 0.8), ("happy", 0.2)])
    probabilities: dict[str, float] = field(default_factory=lambda: {"neutral": 0.8, "happy": 0.2})


class FrameSamplerTest(unittest.TestCase):
    def test_samples_at_requested_interval(self):
        sampler = FrameSampler(sample_rate_hz=3)

        self.assertTrue(sampler.should_sample(now=0.0))
        self.assertFalse(sampler.should_sample(now=0.1))
        self.assertTrue(sampler.should_sample(now=0.34))


class ProbabilityAveragerTest(unittest.TestCase):
    def test_averages_latest_three_vectors(self):
        averager = ProbabilityAverager(window_size=3)

        averager.update({"sad": 0.8, "happy": 0.2})
        averager.update({"sad": 0.6, "happy": 0.4})
        result = averager.update({"sad": 0.1, "happy": 0.9})

        self.assertEqual(result.sample_count, 3)
        self.assertAlmostEqual(result.probabilities["sad"], 0.5)
        self.assertAlmostEqual(result.probabilities["happy"], 0.5)

    def test_drops_oldest_vector(self):
        averager = ProbabilityAverager(window_size=3)

        averager.update({"sad": 1.0, "happy": 0.0})
        averager.update({"sad": 0.0, "happy": 1.0})
        averager.update({"sad": 0.0, "happy": 1.0})
        result = averager.update({"sad": 0.0, "happy": 1.0})

        self.assertEqual(result.label, "happy")
        self.assertAlmostEqual(result.probabilities["happy"], 1.0)


class PolicyTest(unittest.TestCase):
    def test_sad_maps_to_empathy_cue(self):
        cue = cue_for_expression("sad", confidence=0.91)

        self.assertEqual(cue.label, "sad")
        self.assertIn("dong cam", cue.action.lower())


class OpenVINOModelValidationTest(unittest.TestCase):
    def test_accepts_openvino_directory_and_xml_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "best.xml"
            xml_path.write_text("<xml />", encoding="utf-8")

            self.assertEqual(validate_openvino_model_path(tmp), xml_path.resolve())
            self.assertEqual(validate_openvino_model_path(xml_path), xml_path.resolve())

    def test_rejects_pytorch_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "best.pt"
            pt_path.write_bytes(b"placeholder")

            with self.assertRaises(ValueError):
                validate_openvino_model_path(pt_path)

    def test_empty_predict_batch_does_not_import_openvino(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "best.xml"
            xml_path.write_text("<xml />", encoding="utf-8")
            classifier = OpenVINOExpressionClassifier(model_path=xml_path, device="CPU")

            self.assertEqual(classifier.predict_batch([]), [])


class HistoryTest(unittest.TestCase):
    def test_saves_and_fetches_top_two_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.sqlite3"

            row_id = save_state_snapshot(DummyState(), source="test", db_path=db_path)
            rows = fetch_recent_snapshots(limit=1, db_path=db_path)

            self.assertEqual(row_id, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["top_emotions"][0][0], "neutral")
            self.assertEqual(rows[0]["cue_sections"][1]["label"], "happy")


if __name__ == "__main__":
    unittest.main()
