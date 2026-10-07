import unittest
from pathlib import Path
import tempfile

import torch

from train_model import load_training_folder, make_windows, split_token_ids


class TrainingDataTests(unittest.TestCase):
    def test_make_windows_aligns_next_token_targets(self):
        inputs, targets = make_windows([0, 1, 2, 3, 4], context_size=3)
        self.assertEqual(inputs.shape, targets.shape)
        self.assertEqual(inputs.shape, (2, 3))
        self.assertEqual(inputs.tolist(), [[0, 1, 2], [1, 2, 3]])
        self.assertEqual(targets.tolist(), [[1, 2, 3], [2, 3, 4]])

    def test_make_windows_rejects_invalid_stride(self):
        with self.assertRaises(ValueError):
            make_windows([0, 1, 2, 3], context_size=2, stride=0)

    def test_split_happens_before_windowing(self):
        tokens = list(range(20))
        train, validation = split_token_ids(
            tokens,
            context_size=4,
            validation_fraction=0.25,
        )
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertEqual(train[-1] + 1, validation[0])
        self.assertEqual(set(train).intersection(validation), set())

    def test_split_requires_two_independent_segments(self):
        with self.assertRaises(ValueError):
            split_token_ids([0, 1, 2, 3, 4, 5], context_size=3)

    def test_window_outputs_are_long_tensors(self):
        inputs, targets = make_windows([0, 1, 2, 3], context_size=2)
        self.assertEqual(inputs.dtype, torch.long)
        self.assertEqual(targets.dtype, torch.long)

    def test_load_training_folder_reads_all_txt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.txt").write_text("beta", encoding="utf-8")

            text, files = load_training_folder(root)

            self.assertEqual(
                [path.name for path in files],
                ["a.txt", "b.txt"],
            )
            self.assertIn("alpha", text)
            self.assertIn("beta", text)


if __name__ == "__main__":
    unittest.main()
