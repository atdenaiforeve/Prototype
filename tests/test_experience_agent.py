import unittest

from experience import ExperienceEngine, GridWorld
from experience_agent import ExperienceAgent
from memory import MemoryEngine


class ExperienceAgentTests(unittest.TestCase):
    def test_model_action_is_used(self):
        memory = MemoryEngine(":memory:")
        engine = ExperienceEngine(GridWorld(width=3, height=3), memory)
        agent = ExperienceAgent(engine, lambda prompt, **kwargs: "east")
        result = agent.step()

        self.assertEqual(result["chosen_action"], "east")
        self.assertEqual(result["experience"]["result"]["position"], (1, 0))

    def test_invalid_model_output_falls_back_and_records_intention(self):
        memory = MemoryEngine(":memory:")
        engine = ExperienceEngine(GridWorld(width=3, height=3), memory)
        agent = ExperienceAgent(engine, lambda prompt, **kwargs: "banana")
        result = agent.step()

        self.assertIn(result["chosen_action"], result["experience"]["observation"]["available_actions"])
        self.assertIsNotNone(memory.latest_self_update_intention())


if __name__ == "__main__":
    unittest.main()
