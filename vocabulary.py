from __future__ import annotations

from pathlib import Path

from tokenizer import Tokenizer


class Vocabulary:
    """Convenience layer for the model's token-to-ID vocabulary."""

    def __init__(self, path: Path | str = "vocabulary.json") -> None:
        self.tokenizer = Tokenizer(path)

    def learn_text(self, text: str) -> list[str]:
        """Rebuild the vocabulary from the supplied training text."""
        return self.tokenizer.learn(text)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def token_id(self, token: str) -> int:
        return self.tokenizer.token_to_id.get(token, self.tokenizer.token_to_id["<UNK>"])

    def token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token.get(int(token_id), "<UNK>")

    @property
    def size(self) -> int:
        return self.tokenizer.vocabulary_size

    def save(self) -> None:
        self.tokenizer.save()


if __name__ == "__main__":
    vocabulary = Vocabulary()

    vocabulary.learn_text(
        "Prototype is learning language. Language is made from tokens."
    )

    vocabulary.save()

    ids = vocabulary.encode("Prototype is learning.")
    print("Vocabulary size:", vocabulary.size)
    print("Encoded:", ids)
    print("Decoded:", vocabulary.decode(ids))
