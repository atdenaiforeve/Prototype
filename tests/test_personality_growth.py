import json
import tempfile
import unittest
from pathlib import Path

from personality_growth import PersonalityGrowth


class PersonalityGrowthTests(unittest.TestCase):
    def test_starts_without_assigned_traits(self):
        with tempfile.TemporaryDirectory() as directory:
            growth = PersonalityGrowth(Path(directory) / "profile.json")
            self.assertEqual(growth.snapshot()["replies_observed"], 0)
            self.assertIn("No stable behavioural patterns", growth.context_for_prompt())

    def test_records_and_restores_own_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            growth = PersonalityGrowth(path)
            for _ in range(3):
                growth.record_reply("I prefer examining the evidence before choosing a direction.")
            restored = PersonalityGrowth(path)
            snapshot = restored.snapshot()
            self.assertEqual(snapshot["replies_observed"], 3)
            self.assertTrue(snapshot["recurring_self_selected_phrases"])

    def test_profile_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            growth = PersonalityGrowth(Path(directory) / "profile.json")
            for index in range(140):
                growth.record_reply(f"Observed response number {index} with unique wording.")
            self.assertEqual(growth.snapshot()["replies_observed"], 30)
            data = json.loads((Path(directory) / "profile.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["replies"]), growth.MAX_REPLIES)


if __name__ == "__main__":
    unittest.main()
