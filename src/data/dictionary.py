from pathlib import Path
from typing import Dict

from flax import nnx
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from PIL import Image

from src import utils


def load_dictionary(path: Path, mode="RGB", transform=None) -> dict[str, np.ndarray]:
    """Load a directory of PNG images into a dict of numpy arrays keyed by stem."""
    if transform is None:
        transform = lambda x: x

    dictionary = {}
    for p in sorted(Path(path).glob("*.png")):
        dictionary[p.stem] = np.array(Image.open(p).convert(mode))
        if len(dictionary[p.stem].shape) == 2:
            dictionary[p.stem] = dictionary[p.stem][..., np.newaxis]
        dictionary[p.stem] = transform(dictionary[p.stem])
    return dictionary


@nnx.dataclass
class ShapeDictionary(nnx.Pytree):
    shapes: Float[Array, "K H W C"] = nnx.data()
    targets: Dict[str, int] = nnx.data()

    @classmethod
    def load(cls, path: str):
        def transform(img):
            img = np.array(img, dtype=jnp.float32) / 255.
            return 1. - img

        shapes = load_dictionary(path, mode="L", transform=transform)
        return cls(shapes=jnp.stack(tuple(shapes.values()), axis=0),
                   targets={i: k for i, k in enumerate(shapes)})

    @classmethod
    def randomize(cls, k: int, h: int, w: int, *, rngs: nnx.Rngs):
        return cls(shapes=rngs.normal(shape=(k, h, w, 1)),
                   targets={str(c): c for c in range(k)})
