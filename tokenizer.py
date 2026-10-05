from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
DEFAULT_WORD_LIST = Path(__file__).with_name("english_words.txt")


class Tokenizer:
    """Simple word-and-punctuation tokenizer for the first model."""

    def __init__(
        self,
        model_path: Path | str = "vocabulary.json",
        word_list_path: Path | str = DEFAULT_WORD_LIST,
    ) -> None:
        self.model_path = Path(model_path)
        self.word_list_path = Path(word_list_path)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.frequencies: Counter[str] = Counter()
        self._reset_special_tokens()
        self.load()
        self.load_starting_vocabulary()

    def _reset_special_tokens(self) -> None:
        self.token_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}

    def tokenize(self, text: str) -> list[str]:
        """Convert text into word/punctuation tokens."""
        return TOKEN_PATTERN.findall(text)

    def learn(self, text: str) -> list[str]:
        """Learn vocabulary entries from text and return its tokens."""
        tokens = self.tokenize(text)
        self.frequencies.update(tokens)
        self._rebuild_ids()
        return tokens

    def load_starting_vocabulary(self) -> None:
        """Load the bundled English word list without pretending it was learned."""
        if not self.word_list_path.exists():
            return

        try:
            lines = self.word_list_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return

        for line in lines:
            word = line.strip().lower()
            if not word or word.startswith("#"):
                continue
            if TOKEN_PATTERN.fullmatch(word):
                self._add_token(word)

    def _add_token(self, token: str) -> None:
        if token in self.token_to_id:
            return

        next_id = max(self.id_to_token, default=-1) + 1
        self.token_to_id[token] = next_id
        self.id_to_token[next_id] = token

    def _rebuild_ids(self) -> None:
        for token in sorted(self.frequencies):
            self._add_token(token)

    def encode(self, text: str, add_boundaries: bool = True) -> list[int]:
        """Convert text into token IDs."""
        tokens = self.tokenize(text)
        ids = []

        if add_boundaries:
            ids.append(self.token_to_id["<BOS>"])

        for token in tokens:
            ids.append(self.token_to_id.get(token.lower(), self.token_to_id["<UNK>"]))

        if add_boundaries:
            ids.append(self.token_to_id["<EOS>"])

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token IDs back into readable text."""
        special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        tokens = [
            self.id_to_token.get(int(token_id), "<UNK>")
            for token_id in ids
        ]

        if skip_special:
            tokens = [token for token in tokens if token not in special]

        text = " ".join(tokens)
        return re.sub(r"\s+([.,!?;:)\]])", r"\1", text)

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)

    def save(self) -> None:
        data = {
            "version": 2,
            "token_to_id": self.token_to_id,
            "frequencies": dict(self.frequencies),
        }
        self.model_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.model_path.exists():
            return

        try:
            data = json.loads(self.model_path.read_text(encoding="utf-8"))
            token_to_id = {
                str(token): int(token_id)
                for token, token_id in data.get("token_to_id", {}).items()
            }
            frequencies = Counter(data.get("frequencies", {}))
        except (OSError, ValueError, json.JSONDecodeError):
            return

        if not all(
            token in token_to_id
            for token in ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
        ):
            return

        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}
        self.frequencies = frequencies


if __name__ == "__main__":
    tokenizer = Tokenizer()

    examples = [
        "The cat sits on the mat.",
        "The dog sits on the mat.",
        "Prototype learns language.",
    ]

    for example in examples:
        tokenizer.learn(example)

    tokenizer.save()

    encoded = tokenizer.encode("The cat learns.")
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Tokens:", tokenizer.tokenize("The cat learns."))
    print("Token IDs:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
