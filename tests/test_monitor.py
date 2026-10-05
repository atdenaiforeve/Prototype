import tempfile
import unittest
from pathlib import Path

from monitor import ReasoningMonitor


class ReasoningMonitorTests(unittest.TestCase):
    def test_records_structured_reasoning_event(self):
        monitor = ReasoningMonitor(lambda ids: f"token-{ids[0]}", top_k=2)
        event = monitor.record(
            step=0,
            context_length=4,
            token_id=7,
            probability=0.75,
            entropy=0.4,
            top_candidates=[{"token_id": 7, "text": "token-7", "probability": 0.75}],
        )

        self.assertEqual(event.token_text, "token-7")
        self.assertEqual(event.token_id, 7)
        self.assertEqual(len(monitor.events), 1)
        self.assertGreater(monitor.summary()["average_probability"], 0.0)

    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reasoning.jsonl"
            monitor = ReasoningMonitor(lambda ids: str(ids[0]), path=path)
            monitor.record(
                step=1,
                context_length=2,
                token_id=3,
                probability=1.0,
                entropy=0.0,
                top_candidates=[],
            )
            self.assertTrue(path.exists())
            self.assertIn('"token_id": 3', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
