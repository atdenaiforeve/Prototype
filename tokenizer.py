"""Subword tokenizer for Prototype.

Prototype uses a small byte-independent BPE-style subword tokenizer. It learns
frequent character-pair merges from the training corpus, capped at 8,192
vocabulary entries. This lets unseen words be represented by smaller pieces
instead of requiring a whole-word vocabulary entry.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

DEFAULT_WORD_LIST = Path(__file__).with_name("english_words.txt")


class Tokenizer:
    """Learn and apply a compact BPE-style subword vocabulary."""

    def __init__(
        self,
        model_path: Path | str = "vocabulary.json",
        word_list_path: Path | str = DEFAULT_WORD_LIST,
        vocab_limit: int = 8192,
        min_frequency: int = 2,
    ) -> None:
        if vocab_limit < len(SPECIAL_TOKENS) + 16:
            raise ValueError("vocab_limit is too small")
        self.model_path = Path(model_path)
        self.word_list_path = Path(word_list_path)
        self.vocab_limit = vocab_limit
        self.min_frequency = max(1, min_frequency)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.frequencies: Counter[str] = Counter()
        self.merges: list[tuple[str, str]] = []
        self._reset_special_tokens()
        self.load()

    def _reset_special_tokens(self) -> None:
        self.token_to_id = dict(SPECIAL_TOKENS)
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}

    def _add_token(self, token: str) -> bool:
        if token in self.token_to_id:
            return False
        if len(self.token_to_id) >= self.vocab_limit:
            return False
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return True

    def _word_sequences(self, text: str) -> Counter[tuple[str, ...]]:
        words = TOKEN_PATTERN.findall(text.lower())
        sequences: Counter[tuple[str, ...]] = Counter()
        for word in words:
            if not word:
                continue
            sequences[tuple(word)] += 1
        return sequences

    @staticmethod
    def _pair_counts(sequences: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
        pairs: Counter[tuple[str, str]] = Counter()
        for symbols, count in sequences.items():
            for left, right in zip(symbols, symbols[1:]):
                pairs[(left, right)] += count
        return pairs

    @staticmethod
    def _merge_sequence(
        symbols: tuple[str, ...],
        pair: tuple[str, str],
        merged: str,
    ) -> tuple[str, ...]:
        output: list[str] = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                output.append(merged)
                index += 2
            else:
                output.append(symbols[index])
                index += 1
        return tuple(output)

    def learn(self, text: str) -> list[str]:
        """Learn a subword vocabulary from text and return encoded pieces."""
        self._reset_special_tokens()
        self.frequencies.clear()
        self.merges.clear()

        sequences = self._word_sequences(text)

        # Seed the vocabulary with every character/punctuation symbol we see.
        symbols = sorted({symbol for word in sequences for symbol in word})
        for symbol in symbols:
            self._add_token(symbol)

        # Add frequent merged pieces until the configured vocabulary limit.
        while len(self.token_to_id) < self.vocab_limit:
            pairs = self._pair_counts(sequences)
            candidates = [
                (count, pair)
                for pair, count in pairs.items()
                if count >= self.min_frequency and pair[0] != "" and pair[1] != ""
            ]
            if not candidates:
                break

            _, pair = max(candidates, key=lambda item: (item[0], item[1]))
            merged = pair[0] + pair[1]
            if not self._add_token(merged):
                break

            self.merges.append(pair)
            sequences = Counter(
                {
                    self._merge_sequence(word, pair, merged): count
                    for word, count in sequences.items()
                }
            )

        # Frequencies are useful metadata, but the tokenizer no longer depends
        # on whole-word entries.
        self.frequencies.update(TOKEN_PATTERN.findall(text.lower()))
        return self.tokenize(text)

    def _apply_merges(self, word: str) -> list[str]:
        pieces = list(word)
        merge_map = {pair: pair[0] + pair[1] for pair in self.merges}
        for pair in self.merges:
            merged = merge_map[pair]
            output: list[str] = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and (pieces[index], pieces[index + 1]) == pair:
                    output.append(merged)
                    index += 2
                else:
                    output.append(pieces[index])
                    index += 1
            pieces = output
        return pieces

    def tokenize(self, text: str) -> list[str]:
        """Convert text into learned subword/punctuation pieces."""
        output: list[str] = []
        for token in TOKEN_PATTERN.findall(text.lower()):
            if token.isalnum() or "_" in token:
                output.extend(self._apply_merges(token))
            else:
                output.append(token)
        return output

    def encode(self, text: str, add_boundaries: bool = True) -> list[int]:
        ids: list[int] = []
        if add_boundaries:
            ids.append(SPECIAL_TOKENS["<BOS>"])

        for token in self.tokenize(text):
            ids.append(self.token_to_id.get(token, SPECIAL_TOKENS["<UNK>"]))

        if add_boundaries:
            ids.append(SPECIAL_TOKENS["<EOS>"])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = [
            self.id_to_token.get(int(token_id), "<UNK>")
            for token_id in ids
        ]
        if skip_special:
            tokens = [
                token for token in tokens
                if token not in SPECIAL_TOKENS
            ]

        text = ""
        for token in tokens:
            if not text:
                text = token
            elif token.isalnum() or "_" in token:
                # Character/subword pieces are joined without spaces.
                text += token
            elif token in ".,!?;:)]}%":
                text += token
            else:
                text += " " + token
        return text

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)

    def save(self) -> None:
        data = {
            "version": 3,
            "vocab_limit": self.vocab_limit,
            "min_frequency": self.min_frequency,
            "token_to_id": self.token_to_id,
            "frequencies": dict(self.frequencies),
            "merges": [list(pair) for pair in self.merges],
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
            if not all(token in token_to_id for token in SPECIAL_TOKENS):
                return
            self.token_to_id = token_to_id
            self.id_to_token = {value: key for key, value in token_to_id.items()}
            self.frequencies = Counter(data.get("frequencies", {}))
            self.merges = [
                (str(pair[0]), str(pair[1]))
                for pair in data.get("merges", [])
                if isinstance(pair, list) and len(pair) == 2
            ]
            self.vocab_limit = int(data.get("vocab_limit", self.vocab_limit))
            self.min_frequency = int(data.get("min_frequency", self.min_frequency))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._reset_special_tokens()


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.learn("The cat sits on the mat. Prototype learns language.")
    tokenizer.save()
    encoded = tokenizer.encode("The cats learn language.")
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Tokens:", tokenizer.tokenize("The cats learn language."))
    print("Token IDs:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
