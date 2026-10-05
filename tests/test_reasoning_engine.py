import unittest

from reasoning_engine import _score_monitor
from monitor import ReasoningMonitor


class ReasoningEngineTests(unittest.TestCase):
    def test_score_is_bounded(self):
        monitor = ReasoningMonitor(lambda ids: str(ids[0]))
        monitor.record(
            step=0,
            context_length=1,
            token_id=4,
            probability=0.8,
            entropy=0.2,
            top_candidates=[],
        )
        monitor.record(
            step=1,
            context_length=2,
            token_id=5,
            probability=0.6,
            entropy=0.3,
            top_candidates=[],
        )
        score = _score_monitor(monitor)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_reasoning_engine_connects_candidates_to_workspace(self):
        calls = []

        def fake_generate(prompt, **kwargs):
            monitor = kwargs["monitor"]
            index = len(calls)
            calls.append(prompt)
            monitor.record(
                step=0,
                context_length=1,
                token_id=index + 4,
                probability=0.4 + index * 0.2,
                entropy=0.1,
                top_candidates=[],
            )
            return f"candidate {index}"

        from reasoning_engine import reason

        result = reason("hello", fake_generate, candidates=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.output, "candidate 2")
        self.assertEqual(result.chosen.status, "chosen")
        self.assertEqual(len(result.workspace["hypotheses"]), 3)


if __name__ == "__main__":
    unittest.main()
