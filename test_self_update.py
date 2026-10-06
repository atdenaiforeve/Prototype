import tempfile
import unittest
from pathlib import Path

from memory import MemoryEngine
from self_update import GitHubSelfUpdater


class SelfUpdateMemoryTests(unittest.TestCase):
    def test_latest_fresh_intention_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryEngine(Path(tmp) / "memory.db")
            first = memory.remember_self_update_intention(
                "Improve retrieval",
                reason="Recent memories should win",
            )
            second = memory.remember_self_update_intention(
                "Improve tokenizer",
                reason="Handle unknown words better",
            )

            latest = memory.latest_self_update_intention()
            self.assertIsNotNone(latest)
            self.assertEqual(latest["id"], second)
            self.assertNotEqual(latest["id"], first)

    def test_intention_status_can_be_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryEngine(Path(tmp) / "memory.db")
            memory_id = memory.remember_self_update_intention("Improve tests")
            self.assertTrue(memory.set_self_update_intention_status(memory_id, "testing"))
            latest = memory.latest_self_update_intention()
            self.assertIsNotNone(latest)
            self.assertEqual(latest["metadata"]["status"], "testing")

    def test_protected_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            GitHubSelfUpdater.validate_paths([".env"])
        with self.assertRaises(ValueError):
            GitHubSelfUpdater.validate_paths([".github/workflows/deploy.yml"])
        with self.assertRaises(ValueError):
            GitHubSelfUpdater.validate_paths(["../outside.py"])

    def test_update_file_count_is_bounded(self):
        with self.assertRaises(ValueError):
            GitHubSelfUpdater.validate_paths([f"file{i}.py" for i in range(9)])


if __name__ == "__main__":
    unittest.main()
