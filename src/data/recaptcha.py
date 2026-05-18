import string
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import datamodule

CHARACTERS = string.ascii_uppercase + string.digits
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARACTERS)}


class RecaptchaDataset(Dataset):
    def __init__(self, image_paths, width: int, height: int):
        self._paths = list(image_paths)
        self._width = width
        self._height = height

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        path = Path(self._paths[idx])
        word = path.parent.name.upper()
        label = np.array([CHAR_TO_IDX[c] for c in word], dtype=np.int64)
        img = Image.open(path).convert("RGB").resize((self._width, self._height))
        x = np.array(img, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return x, label


class RecaptchaDataModule(datamodule.DataModule):
    """DataModule for reCAPTCHA word images from the Recaptcha dataset.

    Loads the full-word images from generated/segmented_words/, one per word,
    with the word text (parent directory name) as the label.
    """

    def __init__(self, *args, data_dir: str = "data/recaptcha",
                 width: int = 200, height: int = 100,
                 test_split: float = 0.2, **kwargs):
        self._recaptcha_dir = Path(data_dir)
        self._width = width
        self._height = height
        self._test_split = test_split
        super().__init__(*args, **kwargs)

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        seg_words_dir = self._recaptcha_dir / "generated" / "segmented_words"
        paths = sorted(
            p for p in seg_words_dir.glob("*/0_*.png")
            if not p.name.startswith("._")
        )
        if not paths:
            raise FileNotFoundError(
                f"No word images found under {seg_words_dir}"
            )
        n_test = max(1, int(len(paths) * self._test_split))
        return (
            RecaptchaDataset(paths[:-n_test], self._width, self._height),
            RecaptchaDataset(paths[-n_test:], self._width, self._height),
        )

    @property
    def shape(self) -> Tuple:
        return (3, self._height, self._width)
