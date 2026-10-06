from pathlib import Path

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


def test_self_model_observes_generation_model_and_code(tmp_path: Path) -> None:
    (tmp_path / "generation.json").write_text(
        '{"generation": 7}', encoding="utf-8"
    )
    (tmp_path / "example.py").write_text("print('x')", encoding="utf-8")

    observation = SelfModel(tmp_path).observe(model=FakeModel())

    assert observation.generation == 7
    assert observation.model["parameters"] == 30
    assert observation.model["context_size"] == 8
    assert observation.code["python_file_count"] == 1
    assert observation.code["python_files"][0]["path"] == "example.py"


def test_self_model_save_and_load(tmp_path: Path) -> None:
    model = SelfModel(tmp_path)
    observation = model.observe(state={"mode": "running"})
    output = tmp_path / "self_model.json"

    model.save(observation, output)
    loaded = SelfModel.load(output)

    assert loaded["identity"]["name"] == "Prototype"
    assert loaded["observation"]["state"]["mode"] == "running"


def test_self_model_is_observational(tmp_path: Path) -> None:
    model = SelfModel(tmp_path)
    observation = model.observe()
    snapshot = model.snapshot(observation)

    assert "approve" not in snapshot
    assert "veto" not in snapshot
