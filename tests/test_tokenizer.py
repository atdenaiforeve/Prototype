"""Regression tests for Prototype's subword tokenizer.

Run with:
    python -m unittest tests.test_tokenizer
"""

import tempfile
import unittest
from pathlib import Path

from tokenizer import SPECIAL_TOKENS, Tokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.corpus = (
            "Prototype learns language. "
            "The model learns words, punctuation, and patterns. "
            "Cats climb trees and dogs run. "
            "Programming uses computers and mathematics. "
            "The earth has weather, plants, oceans, and mountains. "
            "Space contains stars, planets, and galaxies."
        )
        self.tokenizer = Tokenizer(
            model_path=Path(tempfile.gettempdir()) / "prototype_test_tokenizer.json",
            vocab_limit=256,
            min_frequency=2,
        )
        self.tokenizer.model_path.unlink(missing_ok=True)
        self.tokenizer.learn(self.corpus)

    def test_vocabulary_stays_within_limit(self):
        self.assertLessEqual(self.tokenizer.vocabulary_size, 256)
        self.assertGreaterEqual(
            self.tokenizer.vocabulary_size,
            len(SPECIAL_TOKENS),
        )

    def test_encode_ids_are_valid(self):
        ids = self.tokenizer.encode("Prototype learns language.")
        self.assertGreater(len(ids), 2)
        self.assertTrue(all(0 <= token_id < self.tokenizer.vocabulary_size for token_id in ids))

    def test_punctuation_is_preserved(self):
        tokens = self.tokenizer.tokenize("Hello, world!")
        self.assertIn(",", tokens)
        self.assertIn("!", tokens)

    def test_unseen_word_does_not_require_whole_word_token(self):
        tokens = self.tokenizer.tokenize("prototyping")
        self.assertNotIn("prototyping", self.tokenizer.token_to_id)
        self.assertTrue(tokens)
        self.assertTrue(all(token in self.tokenizer.token_to_id for token in tokens))

    def test_decode_is_readable(self):
        text = "The cats run."
        decoded = self.tokenizer.decode(self.tokenizer.encode(text))
        self.assertIn("the", decoded)
        self.assertIn("cats", decoded)
        self.assertIn("run", decoded)
        self.assertTrue(decoded.endswith("."))

    def test_save_and_reload_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            first = Tokenizer(model_path=path, vocab_limit=256, min_frequency=2)
            first.learn(self.corpus)
            before_tokens = first.tokenize("Prototype learns language.")
            before_ids = first.encode("Prototype learns language.")
            first.save()

            second = Tokenizer(model_path=path, vocab_limit=256, min_frequency=2)
            self.assertEqual(before_tokens, second.tokenize("Prototype learns language."))
            self.assertEqual(before_ids, second.encode("Prototype learns language."))

    def test_word_boundaries_are_not_lost(self):
        decoded = self.tokenizer.decode(self.tokenizer.encode("cats climb trees"))
        self.assertIn("cats climb trees", decoded)


if __name__ == "__main__":
    unittest.main()
