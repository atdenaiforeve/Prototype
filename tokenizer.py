"""Subword tokenizer for Prototype.

Prototype uses a small byte-independent BPE-style subword tokenizer. It learns
frequent character-pair merges from the training corpus.
"""

from __future__ import annotations

import heapq
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
WORD_BOUNDARY = "▁"


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
            if word:
                sequences[tuple(WORD_BOUNDARY + word)] += 1
        return sequences

    @staticmethod
    def _pair_counts(
        sequences: Counter[tuple[str, ...]],
    ) -> Counter[tuple[str, str]]:
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
            if (
                index + 1 < len(symbols)
                and (symbols[index], symbols[index + 1]) == pair
            ):
                output.append(merged)
                index += 2
            else:
                output.append(symbols[index])
                index += 1
        return tuple(output)

    def learn(self, text: str) -> list[str]:
        """Rebuild the vocabulary using heap/indexed BPE pair updates.

        The old implementation recomputed every pair in the whole corpus after
        every merge. That is simple but becomes very slow as the vocabulary
        grows. This implementation keeps a max-heap of candidate pairs and an
        index of the sequences containing each pair, so only sequences affected
        by the selected merge are updated.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("training text must not be empty")

        self._reset_special_tokens()
        self.frequencies.clear()
        self.merges.clear()

        sequences = self._word_sequences(text)

        symbols = sorted(
            {symbol for word in sequences for symbol in word}
        )
        for symbol in symbols:
            self._add_token(symbol)

        # Weighted pair counts avoid rescanning the entire corpus to find the
        # most frequent merge. The heap uses lazy deletion: stale entries are
        # ignored when their stored count no longer matches pair_counts.
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_to_sequences: dict[
            tuple[str, str],
            set[tuple[str, ...]],
        ] = {}

        for sequence, count in sequences.items():
            for pair in zip(sequence, sequence[1:]):
                pair_counts[pair] += count
                pair_to_sequences.setdefault(pair, set()).add(sequence)

        merge_heap = [
            (-count, pair)
            for pair, count in pair_counts.items()
        ]
        heapq.heapify(merge_heap)

        while len(self.token_to_id) < self.vocab_limit:
            pair: tuple[str, str] | None = None

            while merge_heap:
                negative_count, candidate = heapq.heappop(merge_heap)
                count = -negative_count

                if (
                    pair_counts.get(candidate, 0) == count
                    and count >= self.min_frequency
                ):
                    pair = candidate
                    break

            if pair is None:
                break

            merged = pair[0] + pair[1]
            if not self._add_token(merged):
                break

            affected_sequences = list(
                pair_to_sequences.get(pair, ())
            )
            if not affected_sequences:
                break

            for old_sequence in affected_sequences:
                count = sequences.pop(old_sequence, 0)
                if not count:
                    continue

                # Remove the old sequence's contribution from the global
                # pair counts and occurrence index.
                old_pairs = Counter(
                    zip(old_sequence, old_sequence[1:])
                )
                for old_pair, occurrences in old_pairs.items():
                    new_count = (
                        pair_counts.get(old_pair, 0)
                        - occurrences * count
                    )

                    sequence_index = pair_to_sequences.get(old_pair)
                    if sequence_index is not None:
                        sequence_index.discard(old_sequence)
                        if not sequence_index:
                            pair_to_sequences.pop(old_pair, None)

                    if new_count > 0:
                        pair_counts[old_pair] = new_count
                        heapq.heappush(
                            merge_heap,
                            (-new_count, old_pair),
                        )
                    else:
                        pair_counts.pop(old_pair, None)

                new_sequence = self._merge_sequence(
                    old_sequence,
                    pair,
                    merged,
                )
                sequences[new_sequence] += count

                # Add only the pairs created by the changed sequence.
                new_pairs = Counter(
                    zip(new_sequence, new_sequence[1:])
                )
                for new_pair, occurrences in new_pairs.items():
                    new_count = (
                        pair_counts.get(new_pair, 0)
                        + occurrences * count
                    )
                    pair_counts[new_pair] = new_count
                    pair_to_sequences.setdefault(
                        new_pair,
                        set(),
                    ).add(new_sequence)
                    heapq.heappush(
                        merge_heap,
                        (-new_count, new_pair),
                    )

            self.merges.append(pair)

            if len(self.merges) % 500 == 0:
                print(
                    f"tokenizer merges: {len(self.merges)} "
                    f"| vocabulary: {self.vocabulary_size}"
                )

        self.frequencies.update(
            TOKEN_PATTERN.findall(text.lower())
        )

        return self.tokenize(text)

    def _apply_merges(self, word: str) -> list[str]:
        pieces = list(WORD_BOUNDARY + word)

        for pair in self.merges:
            merged = pair[0] + pair[1]
            output: list[str] = []
            index = 0

            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and (pieces[index], pieces[index + 1]) == pair
                ):
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

    def encode(
        self,
        text: str,
        add_boundaries: bool = True,
    ) -> list[int]:
        ids: list[int] = []

        if add_boundaries:
            ids.append(SPECIAL_TOKENS["<BOS>"])

        for token in self.tokenize(text):
            ids.append(
                self.token_to_id.get(
                    token,
                    SPECIAL_TOKENS["<UNK>"],
                )
            )

        if add_boundaries:
            ids.append(SPECIAL_TOKENS["<EOS>"])

        return ids

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
    ) -> str:
        tokens = [
            self.id_to_token.get(
                int(token_id),
                "<UNK>",
            )
            for token_id in ids
        ]

        if skip_special:
            tokens = [
                token
                for token in tokens
                if token not in SPECIAL_TOKENS
            ]

        text = ""

        for token in tokens:
            boundary = token.startswith(
                WORD_BOUNDARY
            )

            piece = (
                token[len(WORD_BOUNDARY):]
                if boundary
                else token
            )

            if boundary:
                if text:
                    text += " "
                text += piece
            elif not text:
                text = piece
            elif piece.isalnum() or "_" in piece:
                text += piece
            elif piece in ".,!?;:)]}%":
                text += piece
            else:
                text += " " + piece

        return text

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)

    def save(self) -> None:
        data = {
            "version": 5,
            "vocab_limit": self.vocab_limit,
            "min_frequency": self.min_frequency,
            "token_to_id": self.token_to_id,
            "frequencies": dict(self.frequencies),
            "merges": [list(pair) for pair in self.merges],
        }

        self.model_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.model_path.exists():
            return

        try:
            data = json.loads(
                self.model_path.read_text(
                    encoding="utf-8"
                )
            )

            token_to_id = {
                str(token): int(token_id)
                for token, token_id
                in data.get("token_to_id", {}).items()
            }

            if any(
                token_to_id.get(token) != token_id
                for token, token_id
                in SPECIAL_TOKENS.items()
            ):
                return

            ids = list(token_to_id.values())

            if (
                len(ids) != len(set(ids))
                or sorted(ids) != list(range(len(ids)))
            ):
                return

            saved_limit = int(
                data.get(
                    "vocab_limit",
                    self.vocab_limit,
                )
            )

            if len(token_to_id) > saved_limit:
                return

            self.token_to_id = token_to_id
            self.id_to_token = {
                value: key
                for key, value
                in token_to_id.items()
            }

            self.frequencies = Counter(
                data.get(
                    "frequencies",
                    {},
                )
            )

            self.merges = [
                (
                    str(pair[0]),
                    str(pair[1]),
                )
                for pair in data.get("merges", [])
                if (
                    isinstance(pair, list)
                    and len(pair) == 2
                )
            ]

            self.vocab_limit = saved_limit

            self.min_frequency = max(
                1,
                int(
                    data.get(
                        "min_frequency",
                        self.min_frequency,
                    )
                ),
            )

            if (
                self.vocab_limit
                < len(SPECIAL_TOKENS)
            ):
                raise ValueError(
                    "saved vocab_limit is too small"
                )

        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            self._reset_special_tokens()


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.learn(
        "The cat sits on the mat. "
        "Prototype learns language."
    )
    tokenizer.save()

    encoded = tokenizer.encode(
        "The cats learn language."
    )

    print(
        "Vocabulary size:",
        tokenizer.vocabulary_size,
    )
    print(
        "Tokens:",
        tokenizer.tokenize(
            "The cats learn language."
        ),
    )
    print(
        "Token IDs:",
        encoded,
    )
    print(
        "Decoded:",
        tokenizer.decode(encoded),
    )
