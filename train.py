"""Convenience entry point for training Prototype.

The actual neural-language-model trainer lives in train_model.py. This wrapper
keeps `python train.py` as the obvious command while avoiding the legacy
interactive word-table learner.
"""

from train_model import main


if __name__ == "__main__":
    main()
