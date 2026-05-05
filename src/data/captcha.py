import string
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import datamodule

CHARACTERS = string.ascii_uppercase + string.digits
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARACTERS)}


class CaptchaDataset(Dataset):
    def __init__(self, image_paths, width: int, height: int):
        self._paths = list(image_paths)
        self._width = width
        self._height = height

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        path = self._paths[idx]
        char = Path(path).stem.split("_")[1]
        label = CHAR_TO_IDX[char]
        img = Image.open(path).convert("RGB").resize((self._width, self._height))
        x = np.array(img, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        return x, np.array(label)


class CaptchaDataModule(datamodule.DataModule):
    """DataModule for single-character CAPTCHA PNG images produced by captcha-dataset."""

    def __init__(self, *args, image_dir: str = "output", width: int = 160,
                 height: int = 60, test_split: float = 0.2, **kwargs):
        self._image_dir = Path(image_dir)
        self._width = width
        self._height = height
        self._test_split = test_split
        super().__init__(*args, **kwargs)

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        paths = sorted(self._image_dir.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG files found in {self._image_dir}")
        n_test = max(1, int(len(paths) * self._test_split))
        return (
            CaptchaDataset(paths[:-n_test], self._width, self._height),
            CaptchaDataset(paths[-n_test:], self._width, self._height),
        )

    @property
    def shape(self) -> Tuple:
        return (3, self._height, self._width)
