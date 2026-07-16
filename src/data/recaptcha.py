from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import datamodule
from .captcha import encode_label


def parse_word(path) -> str:
    """Read the caption for a word image from its parent directory name."""
    return Path(path).parent.name.upper()


class RecaptchaDataset(Dataset):
    def __init__(self, image_paths, width: int, height: int, num_chars: int,
                 with_targets: bool = True):
        self._paths = list(image_paths)
        self._width = width
        self._height = height
        self._num_chars = num_chars
        self._with_targets = with_targets

    def __len__(self):
        return len(self._paths)

    def raw_label(self, idx) -> str:
        """Retrieve the caption for one item from disk, as a string."""
        return parse_word(self._paths[int(idx)])

    def __getitem__(self, idx):
        img = Image.open(self._paths[idx]).convert("RGB")
        img = img.resize((self._width, self._height))
        x = np.array(img, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        if not self._with_targets:
            return (x,)
        return x, encode_label(self.raw_label(idx), self._num_chars)


class RecaptchaDataModule(datamodule.DataModule):
    """DataModule for reCAPTCHA word images from the Recaptcha dataset.

    Loads the full-word images from generated/segmented_words/, one per word,
    with the word text (parent directory name) as the label. Batched targets
    have static shape (batch, num_chars), padded with ``PAD_IDX``; recover
    lengths with ``label_lengths`` and strings with ``decode_label``. Pass
    ``num_chars=None`` to infer the width from the longest word on disk, or
    ``with_targets=False`` to batch images alone and fetch captions on demand
    via ``RecaptchaDataset.raw_label``.
    """

    def __init__(self, *args, data_dir: str = "data/recaptcha",
                 width: int = 200, height: int = 100,
                 num_chars: Optional[int] = None, test_split: float = 0.2,
                 with_targets: bool = True, **kwargs):
        self._recaptcha_dir = Path(data_dir)
        self._width = width
        self._height = height
        self._num_chars = num_chars
        self._test_split = test_split
        self._with_targets = with_targets
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

        max_len = max(len(parse_word(p)) for p in paths)
        if self._num_chars is None:
            self._num_chars = max_len
        elif max_len > self._num_chars:
            raise ValueError(
                f"num_chars={self._num_chars}, but {seg_words_dir} holds "
                f"words up to {max_len} characters long"
            )

        n_test = max(1, int(len(paths) * self._test_split))
        self.train_dataset = RecaptchaDataset(
            paths[:-n_test], self._width, self._height, self._num_chars,
            with_targets=self._with_targets,
        )
        self.test_dataset = RecaptchaDataset(
            paths[-n_test:], self._width, self._height, self._num_chars,
            with_targets=self._with_targets,
        )
        return self.train_dataset, self.test_dataset

    @property
    def num_chars(self) -> int:
        return self._num_chars

    @property
    def shape(self) -> Tuple:
        return (3, self._height, self._width)
