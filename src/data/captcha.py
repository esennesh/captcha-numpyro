import string
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import datamodule

CHARACTERS = string.ascii_uppercase + string.digits
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARACTERS)}
NULL_SENTINEL = "NULL"
PAD_IDX = -1


def parse_label(path) -> str:
    """Read the caption string out of a captcha filename ({id}_{LABEL}.png)."""
    raw_label = Path(path).stem.split("_", 1)[1]
    return "" if raw_label == NULL_SENTINEL else raw_label


def encode_label(chars: str, num_chars: int) -> np.ndarray:
    """Encode a caption as a fixed-width int64 array, padded with PAD_IDX.

    Fixed-width padding (rather than padding to the longest sequence in each
    batch) keeps batch shapes static, so jitted JAX code compiles once.
    """
    if len(chars) > num_chars:
        raise ValueError(
            f"Label {chars!r} has {len(chars)} characters, but num_chars is "
            f"{num_chars}"
        )
    label = np.full((num_chars,), PAD_IDX, dtype=np.int64)
    for i, c in enumerate(chars):
        label[i] = CHAR_TO_IDX[c]
    return label


def decode_label(label) -> str:
    """Invert encode_label, stripping padding."""
    return "".join(CHARACTERS[i] for i in np.asarray(label) if i != PAD_IDX)


def label_lengths(labels) -> np.ndarray:
    """Number of real (non-pad) characters, over the trailing axis."""
    return (np.asarray(labels) != PAD_IDX).sum(axis=-1)


class CaptchaDataset(Dataset):
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
        return parse_label(self._paths[int(idx)])

    def __getitem__(self, idx):
        img = Image.open(self._paths[idx]).convert("RGB")
        img = img.resize((self._width, self._height))
        x = np.array(img, dtype=np.float32) / 255.0
        if not self._with_targets:
            return (x,)
        return x, encode_label(self.raw_label(idx), self._num_chars)


class CaptchaDataModule(datamodule.DataModule):
    """DataModule for CAPTCHA PNG images produced by captcha-dataset.

    Captions live in the filenames ({id}_{LABEL}.png, with NULL for the empty
    string) and may have any length up to ``num_chars``. Batched targets have
    static shape (batch, num_chars), padded with ``PAD_IDX``; recover lengths
    with ``label_lengths`` and strings with ``decode_label``. Pass
    ``num_chars=None`` to infer the width from the longest caption on disk,
    or ``with_targets=False`` to batch images alone and fetch captions
    on demand via ``CaptchaDataset.raw_label``.
    """

    def __init__(self, *args, image_dir: str = "output", width: int = 160,
                 height: int = 60, num_chars: Optional[int] = None,
                 test_split: float = 0.2, with_targets: bool = True,
                 **kwargs):
        self._image_dir = Path(image_dir)
        self._width = width
        self._height = height
        self._num_chars = num_chars
        self._test_split = test_split
        self._with_targets = with_targets
        super().__init__(*args, **kwargs)

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        paths = sorted(self._image_dir.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG files found in {self._image_dir}")

        max_len = max(len(parse_label(p)) for p in paths)
        if self._num_chars is None:
            self._num_chars = max_len
        elif max_len > self._num_chars:
            raise ValueError(
                f"num_chars={self._num_chars}, but {self._image_dir} holds "
                f"captions up to {max_len} characters long"
            )

        n_test = max(1, int(len(paths) * self._test_split))
        self.train_dataset = CaptchaDataset(
            paths[:-n_test], self._width, self._height, self._num_chars,
            with_targets=self._with_targets,
        )
        self.test_dataset = CaptchaDataset(
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
