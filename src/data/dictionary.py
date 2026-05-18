from pathlib import Path
from typing import Dict

from flax import nnx
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from PIL import Image

from src import utils


def load_dictionary(path: Path, mode="RGB", transform=None) -> dict[str, np.ndarray]:
    """Load a flat directory of PNGs into a dict of arrays keyed by filename stem."""
    if transform is None:
        transform = lambda x: x

    dictionary = {}
    for p in sorted(Path(path).glob("*.png")):
        dictionary[p.stem] = np.array(Image.open(p).convert(mode))
        if len(dictionary[p.stem].shape) == 2:
            dictionary[p.stem] = dictionary[p.stem][..., np.newaxis]
        dictionary[p.stem] = transform(dictionary[p.stem])
    return dictionary


def load_nested_dictionary(
    path: Path, mode="RGB", transform=None
) -> dict[tuple[str, str, int, int], np.ndarray]:
    """Load a two-level directory of PNGs into a dict keyed by (char, font, size, idx).

    Expects the structure {path}/{char}/{char}_{font}_{size}_{idx}.png, e.g.
    training_characters/A/A_georgia_52_0.png -> key ('A', 'georgia', 52, 0).
    """
    if transform is None:
        transform = lambda x: x

    dictionary = {}
    for p in sorted(Path(path).glob("*/*.png")):
        if p.name.startswith("._"):
            continue
        char, *font_parts, size_str, idx_str = p.stem.split("_")
        key = (char, "_".join(font_parts), int(size_str), int(idx_str))
        arr = np.array(Image.open(p).convert(mode))
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        dictionary[key] = transform(arr)
    return dictionary


@nnx.dataclass
class ShapeDictionary(nnx.Pytree):
    shapes: Float[Array, "K H W C"] = nnx.data()
    targets: Dict[str | tuple[str, str, int, int], int] = nnx.data()

    @classmethod
    def load(cls, path: str):
        def transform(img):
            img = np.array(img, dtype=jnp.float32) / 255.
            return 1. - img

        shapes = load_dictionary(path, mode="L", transform=transform)
        return cls(shapes=jnp.stack(tuple(shapes.values()), axis=0),
                   targets={i: k for i, k in enumerate(shapes)})

    @classmethod
    def load_nested(cls, path: str):
        def transform(img):
            img = np.array(img, dtype=jnp.float32) / 255.
            return 1. - img

        shapes = load_nested_dictionary(path, mode="L", transform=transform)
        return cls(shapes=jnp.stack(tuple(shapes.values()), axis=0),
                   targets={key: i for i, key in enumerate(shapes)})

    @classmethod
    def randomize(cls, k: int, h: int, w: int, *, rngs: nnx.Rngs):
        return cls(shapes=rngs.normal(shape=(k, h, w, 1)),
                   targets={str(c): c for c in range(k)})
