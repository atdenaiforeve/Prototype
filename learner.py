from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("learner_model.json")


class LanguageLearner:
    """A small trainable text model built from scratch.

    This is an intentionally simple first learning engine. It learns word
    transition patterns from examples and predicts likely next words.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.vocabulary: Counter[str] = Counter()
        self.transitions: dict[str, Counter[str]] = {}
        self.examples_seen = 0
        self.load()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.lower().split()

    def train(self, text: str) -> None:
        tokens = self.tokenize(text)
        if len(tokens) < 2:
            return

        self.vocabulary.update(tokens)

        for current_word, next_word in zip(tokens, tokens[1:]):
            if current_word not in self.transitions:
                self.transitions[current_word] = Counter()
            self.transitions[current_word][next_word] += 1

        self.examples_seen += 1

    def predict_next(self, word: str, count: int = 5) -> list[dict]:
        word = word.lower().strip()
        count = max(1, min(int(count), 20))

        choices = self.transitions.get(word)
        if not choices:
            return []

        total = sum(choices.values())
        ranked = choices.most_common(count)

        return [
            {
                "word": next_word,
                "confidence": round(amount / total, 4),
                "seen": amount,
            }
            for next_word, amount in ranked
        ]

    def generate(self, start: str, length: int = 10) -> str:
        words = self.tokenize(start)
        if not words:
            return ""

        length = max(1, min(int(length), 100))

        while len(words) < length:
            predictions = self.predict_next(words[-1], count=10)
            if not predictions:
                break

            # Sample using learned frequencies rather than always choosing
            # the most common word.
            candidates = [item["word"] for item in predictions]
            weights = [item["seen"] for item in predictions]
            words.append(random.choices(candidates, weights=weights, k=1)[0])

        return " ".join(words)

    def save(self) -> None:
        data = {
            "version": 1,
            "examples_seen": self.examples_seen,
            "vocabulary": dict(self.vocabulary),
            "transitions": {
                word: dict(next_words)
                for word, next_words in self.transitions.items()
            },
        }

        self.model_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.model_path.exists():
            return

        try:
            data = json.loads(self.model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        self.examples_seen = int(data.get("examples_seen", 0))
        self.vocabulary = Counter(data.get("vocabulary", {}))
        self.transitions = {
            word: Counter(next_words)
            for word, next_words in data.get("transitions", {}).items()
        }

    def stats(self) -> dict:
        return {
            "examples_seen": self.examples_seen,
            "vocabulary_size": len(self.vocabulary),
            "learned_transitions": sum(
                len(next_words) for next_words in self.transitions.values()
            ),
        }


if __name__ == "__main__":
    learner = LanguageLearner()

    training_data = [
        "the cat sits on the mat",
        "the cat likes food",
        "the dog sits on the mat",
        "the dog likes food",
    ]

    for example in training_data:
        learner.train(example)

    learner.save()

    print("Prototype learner:", learner.stats())
    print("Next words after 'the':", learner.predict_next("the"))
    print("Example generation:", learner.generate("the", 12))
