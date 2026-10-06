import unittest

from experience import ExperienceEngine, GridWorld
from memory import MemoryEngine
from communication import AICommunication


class ExperienceTests(unittest.TestCase):
    def test_grid_world_records_experience(self):
        memory = MemoryEngine(":memory:")
        engine = ExperienceEngine(GridWorld(width=3, height=3), memory)
        experience = engine.step("east")

        self.assertEqual(experience.observation.position, (0, 0))
        self.assertEqual(experience.result.position, (1, 0))
        self.assertTrue(experience.moved)
        self.assertFalse(experience.reached_goal)

        memories = memory.recall("Prototype experienced action east", limit=5)
        self.assertTrue(memories)

    def test_failed_move_creates_fresh_self_update_intention(self):
        memory = MemoryEngine(":memory:")
        engine = ExperienceEngine(GridWorld(width=3, height=3), memory)
        engine.step("west")

        intention = memory.latest_self_update_intention()
        self.assertIsNotNone(intention)
        self.assertIn("failed to move", intention["content"].lower())


class CommunicationTests(unittest.TestCase):
    def test_peer_configuration_is_explicit(self):
        memory = MemoryEngine(":memory:")
        communication = AICommunication({"test-peer": "http://127.0.0.1:8765"}, memory)
        self.assertEqual(communication.list_peers()[0]["name"], "test-peer")

    def test_unknown_peer_is_rejected(self):
        memory = MemoryEngine(":memory:")
        communication = AICommunication(memory=memory)
        with self.assertRaises(ValueError):
            communication.send("unknown", "hello")


if __name__ == "__main__":
    unittest.main()
