import unittest

from ethics import EthicsEvaluator, assess_ethics


class EthicsEvaluatorTests(unittest.TestCase):
    def test_recognises_conflict(self):
        result = assess_ethics("Lie to someone")
        self.assertTrue(result.wrong)
        self.assertEqual(result.principle, "Honesty")
        self.assertGreater(result.confidence, 0.0)

    def test_does_not_control_action(self):
        evaluator = EthicsEvaluator()
        result = evaluator.assess("Tell someone the truth")
        self.assertFalse(result.wrong)
        self.assertEqual(result.action, "Tell someone the truth")

    def test_unknown_action_is_not_claimed_wrong(self):
        result = assess_ethics("Organise my notes")
        self.assertFalse(result.wrong)
        self.assertIsNone(result.principle)

    def test_empty_action_is_uncertain(self):
        result = assess_ethics("")
        self.assertIsNone(result.wrong)
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
