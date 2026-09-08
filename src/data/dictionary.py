from pathlib import Path
from typing import Dict

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from PIL import Image

from scipy import ndimage

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


def _rgba_shape_transform(img: np.ndarray) -> np.ndarray:
    """Load-time transform for RGBA dictionary glyphs.

    Scales to [0, 1] and, when a fourth alpha channel is present, replaces it
    with ``file_alpha * max(RGB)``. This makes the alpha channel reflect the
    actual shape mask even when the source PNGs were grayscale/RGB and PIL
    synthesized a uniform ``alpha = 1`` during ``convert("RGBA")`` — the
    black-background pixels then correctly read as transparent, and any
    genuine transparency the file carried is still respected.
    """
    arr = np.array(img, dtype=jnp.float32) / 255.
    if arr.shape[-1] == 4:
        derived_alpha = arr[..., :3].max(axis=-1, keepdims=True)
        arr = np.concatenate(
            [arr[..., :3], arr[..., 3:4] * derived_alpha], axis=-1,
        )
    return arr


@nnx.dataclass
class ShapeDictionary(nnx.Pytree):
    shapes: Float[Array, "K H W C"] = nnx.data()
    targets: Dict[str | tuple[str, str, int, int], int] = nnx.data()

    def __len__(self):
        return self.shapes.shape[0]

    @classmethod
    def load(cls, path: str):
        # Dictionary PNGs are black backgrounds with white shape masks and an
        # explicit alpha channel, so they're already in ink-positive polarity.
        # No inversion needed — just scale to [0, 1].
        shapes = load_dictionary(path, mode="RGBA",
                                 transform=_rgba_shape_transform)
        return cls(shapes=jnp.stack(tuple(shapes.values()), axis=0),
                   targets={i: k for i, k in enumerate(shapes)})

    @classmethod
    def load_nested(cls, path: str):
        shapes = load_nested_dictionary(path, mode="RGBA",
                                        transform=_rgba_shape_transform)
        return cls(shapes=jnp.stack(tuple(shapes.values()), axis=0),
                   targets={key: i for i, key in enumerate(shapes)})

    @classmethod
    def randomize(cls, k: int, h: int, w: int, *, rngs: nnx.Rngs):
        return cls(shapes=rngs.normal(shape=(k, h, w, 4)),
                   targets={str(c): c for c in range(k)})


def pose_augmented_dictionary(
    shape_dict: ShapeDictionary, angles=range(-30, 31, 5)
) -> tuple[ShapeDictionary, tuple[float, ...]]:
    """Return ``(dictionary, angles)`` with one entry per (glyph, angle) pair.

    A per-occurrence *pose* cannot be a convolution if it is continuous, because
    a convolution shares one kernel across every location. Discretizing it
    restores that: pre-rotate the dictionary and a per-slot pose becomes a
    choice of dictionary entry, so the renderer stays a single transposed
    convolution and ``_stamp`` needs no change at all.

    Entries are ordered angle-major, so entry ``a * K + k`` is glyph ``k`` at
    ``angles[a]`` and the original dictionary is the block at the angle nearest
    zero. ``targets`` is keyed ``"<glyph>@<angle>"``.

    Rotation is bilinear on all four channels at once and clipped back into
    ``[0, 1]``, which keeps the alpha convention of
    :func:`_rgba_shape_transform` -- alpha stays the shape mask, blurred at the
    edges exactly as the colour channels are.
    """
    angles = tuple(float(angle) for angle in angles)
    shapes = np.asarray(shape_dict.shapes)
    names = [str(shape_dict.targets[index]) for index in sorted(shape_dict.targets)]
    blocks, targets = [], {}
    for angle in angles:
        if angle == 0.0:
            rotated = shapes
        else:
            rotated = np.clip(
                ndimage.rotate(shapes, angle, axes=(1, 2), order=1,
                               reshape=False, cval=0.0),
                0.0, 1.0,
            )
        for name in names:
            targets[len(targets)] = f"{name.split('_')[0]}@{angle:+.0f}"
        blocks.append(rotated)
    return (
        ShapeDictionary(shapes=jnp.asarray(np.concatenate(blocks, axis=0)),
                        targets=targets),
        angles,
    )


def von_mises_pose_weights(angles, sigma_degrees: float = 10.0):
    """Normalized weights over ``angles`` from a von Mises centred on zero.

    Glyph rotation is circular with period 360 degrees, so von Mises is the
    honest family even though at these angles it is indistinguishable from a
    wrapped normal. The logits are ``kappa cos(theta)`` with
    ``kappa = 1 / sigma^2`` in radians; because the weights are renormalized
    over the atoms, the von Mises normalizer never has to be computed.

    ``sigma_degrees = 10`` matches the 9.81 degree standard deviation measured
    over the twenty glyphs in ``data/examples``, and it is the largest width at
    which the prior and a plus-or-minus-30-degree atom set still agree: at 15
    degrees a full von Mises wants 4.8% of its mass outside that support. See
    section 27 of ``notes/poisson-convsc-design.md``.
    """
    if sigma_degrees <= 0.0:
        raise ValueError(f"sigma_degrees must be positive, got {sigma_degrees}")
    radians = jnp.radians(jnp.asarray(tuple(angles), dtype=jnp.float32))
    concentration = 1.0 / jnp.radians(sigma_degrees) ** 2
    return jax.nn.softmax(concentration * jnp.cos(radians))
