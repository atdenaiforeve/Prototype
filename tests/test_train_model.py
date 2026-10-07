import unittest
from pathlib import Path
import tempfile

from train_model import load_training_folder


class TrainingFolderTests(unittest.TestCase):
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
