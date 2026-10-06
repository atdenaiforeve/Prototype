import unittest
from pathlib import Path
from unittest.mock import patch

from memory import MemoryEngine

from self_model import SelfModel


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class FakeModel:
    training = False
    vocab_size = 100
    embedding_size = 16
    hidden_size = 32
    context_size = 8
    num_layers = 2
    num_heads = 2
    dropout = 0.1

    def parameters(self):
        return [FakeParameter(10), FakeParameter(20)]


class SelfModelTests(unittest.TestCase):
    def test_self_model_observes_generation_model_and_code(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            (tmp_path / "generation.json").write_text(
                '{"generation": 7}', encoding="utf-8"
            )
            (tmp_path / "example.py").write_text("print('x')", encoding="utf-8")

            observation = SelfModel(tmp_path).observe(model=FakeModel())

            self.assertEqual(observation.generation, 7)
            self.assertEqual(observation.model["parameters"], 30)
            self.assertEqual(observation.model["context_size"], 8)
            self.assertEqual(observation.code["python_file_count"], 1)
            self.assertEqual(observation.code["python_files"][0]["path"], "example.py")

    def test_self_model_save_and_load(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            model = SelfModel(tmp_path)
            observation = model.observe(state={"mode": "running"})
            output = tmp_path / "self_model.json"

            model.save(observation, output)
            loaded = SelfModel.load(output)

            self.assertEqual(loaded["identity"]["name"], "Prototype")
            self.assertEqual(loaded["observation"]["state"]["mode"], "running")

    def test_self_model_is_observational(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            model = SelfModel(Path(directory))
            observation = model.observe()
            snapshot = model.snapshot(observation)

            self.assertNotIn("approve", snapshot)
            self.assertNotIn("veto", snapshot)

    def test_record_and_retrieve_self_update_intention(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            memory_root = Path(directory)
            model = SelfModel(memory_root)
            memory = MemoryEngine(memory_root / "memory.db")
            with patch("memory.get_memory", return_value=memory):
                intention_id = model.record_self_update_intention(
                    "Improve the reasoning monitor",
                    reason="The monitor needs clearer signals.",
                    freshness_seconds=3600,
                )
                latest = model.latest_self_update_intention()

            self.assertEqual(latest["id"], intention_id)
            self.assertEqual(latest["content"], "Improve the reasoning monitor")
            self.assertEqual(
                latest["metadata"]["reason"],
                "The monitor needs clearer signals.",
            )


if __name__ == "__main__":
    unittest.main()
