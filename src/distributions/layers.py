r"""Composable image layers that carry the region labelling in their lattice.

One layer equals an image and its pixel labels. The pixel labels induce a graph
with an edge between two pixels whenever they carry the same count, while pixels
not sharing the same count lack edges between them.

We route everything through a count field in order to avoid combining graphs as
actual graphs or adjacency matrices. Instead, the edges in the graph come as a
function only of the counts.
"""

from enum import Enum
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Optional, Sequence


__all__ = ["AlphaFormat", "Layer"]

# Claude pointed out that a pixel with an intermediate alpha is often an
# antialiased way of blending two layers where the higher one rasterizes an edge
# and so the lower layer partially shows through from behind it. We therefore
# define the median of possible alpha values as the place to put edges in layer
# counts and adjacency matrices.
POTENTIAL_EDGE_ALPHA = 0.5

class AlphaFormat(Enum):
    STRAIGHT = 0
    PREMULTIPLIED = 1
    CLOSENESS_PREMULTIPLIED = 2

def alpha_to_closeness(alpha, eps: float = 1e-3):
    """Optical depth ``-log(1 - alpha)``, clipped because alpha = 1 diverges.

    ``_ink_kernel`` uses the same ``1 - 1e-3`` clip, which caps a single fully
    covered stamp at a depth of 6.908.
    """
    return -jnp.log1p(-jnp.clip(alpha, 0., 1. - eps))

def closeness_to_alpha(tau):
    return 1. - jnp.exp(-tau)

@nnx.dataclass
class Layer(nnx.Pytree):
    """An image, its coverage, and the region count of every pixel.

    Leading dimensions batch freely: every field carries ``(*batch, H, W)``
    or ``(*batch, H, W, C)``, and every operation here is elementwise over the
    leading axes.

    What each ``format`` stores, given a straight colour ``c`` and coverage
    ``alpha``, with closeness ``tau = -log(1 - alpha)``:

    ==================  =====================  ====================
    format              ``image``              ``coverage``
    ==================  =====================  ====================
    ``STRAIGHT``        ``c``                  ``alpha``
    ``PREMULTIPLIED``   ``alpha * c``          ``alpha``
    ``CLOSENESS_PREMULTIPLIED``   ``tau * c``            ``tau``
    ==================  =====================  ====================

    All three round-trip exactly, with two unavoidable exceptions. At
    ``alpha = 0`` both premultiplied forms scale the colour to zero and cannot
    return it, exactly as premultiplied alpha never can. And ``alpha = 1`` is
    clipped to ``1 - 1e-3`` by :func:`alpha_to_closeness`, because the closeness
    of full coverage is infinite.

    :param count: accumulated region count in ``[0, ∞)``, ``(*batch, H, W)``.
    :param coverage: coverage, ``(H, W)``.  In [0, 1] when STRAIGHT or
    PREMULTIPLIED, but logarithmic in [0, ∞) when CLOSENESS_PREMULTIPLIED.
    :param image: premultiplied colour, ``(H, W, C)``.

    Adjacency is a function of ``count`` alone.
    """

    count: Int[Array, "*batch H W"] = nnx.data()
    coverage: Float[Array, "*batch H W"] = nnx.data()
    image: Float[Array, "*batch H W C"] = nnx.data()
    format: AlphaFormat = nnx.data()

    # -- algebra ------------------------------------------------------

    def __add__(self, other: "Layer") -> "Layer":
        """Emission: coverage and colour accumulate, clipped at full.

        Commutative and associative, because saturating addition on labels is.
        """
        l, r = self.to_straight(), other.to_straight()
        composite = Layer(coverage=jnp.clip(l.coverage + r.coverage, 0.0, 1.0),
                          count=self.count + other.count,
                          format=AlphaFormat.STRAIGHT,
                          image=jnp.clip(l.image + r.image, 0.0, 1.0))
        match self.format:
            case AlphaFormat.STRAIGHT:
                return composite
            case AlphaFormat.PREMULTIPLIED:
                return composite.to_premultiplied()
            case AlphaFormat.CLOSENESS_PREMULTIPLIED:
                return composite.to_closeness_premultiplied()

    def __matmul__(self, below: "Layer") -> "Layer":
        """``self`` occludes ``below``, by conversion to premultiplied color.

        Occlusion means the top layer *owns* the overlap, so its count wins
        rather than the two adding. OVER is thus non-commutative on the counts,
        exactly as it is on the images.
        """
        above = self.to_premultiplied()
        below = below.to_premultiplied()

        composite = Layer(count=jnp.where(above.count > 0, above.count,
                                          below.count),
                          coverage=above.coverage +\
                                   below.coverage * (1. - above.coverage),
                          format=AlphaFormat.PREMULTIPLIED,
                          image=above.image +
                                below.image * (1. - above.coverage[..., None]))
        match self.format:
            case AlphaFormat.STRAIGHT:
                return composite.to_straight()
            case AlphaFormat.PREMULTIPLIED:
                return composite
            case AlphaFormat.CLOSENESS_PREMULTIPLIED:
                return composite.to_closeness_premultiplied()

    def __mul__(self, other: "Layer") -> "Layer":
        """Alpha-blending by addition in hue-and-closeness space, marked as a
        multiplication to denote convolution together (multiplication in the
        space of functions approximated by pixel grids).

        ``(ht', t') = (ht_l + ht_r, t_l + t_r)
        """
        l, r = self.to_closeness_premultiplied(), other.to_closeness_premultiplied()
        composite = Layer(count=self.count + other.count,
                          coverage=l.coverage + r.coverage,
                          format=AlphaFormat.CLOSENESS_PREMULTIPLIED,
                          image=l.image + r.image)
        match self.format:
            case AlphaFormat.STRAIGHT:
                return composite.to_straight()
            case AlphaFormat.PREMULTIPLIED:
                return composite.to_premultiplied()
            case AlphaFormat.CLOSENESS_PREMULTIPLIED:
                return composite

    def __or__(self, other: "Layer") -> "Layer":
        """Probabilistic OR, the ``1 - prod(1 - .)``.

        Exact on straight images, where pixel channels act like probabilities.
        """
        l, r = self.to_straight(), other.to_straight()
        composite = Layer(coverage=1.0 - (1.0 - l.coverage) * (1.0 - r.coverage),
                          count=l.count + r.count, format=AlphaFormat.STRAIGHT,
                          image=1.0 - (1.0 - l.image) * (1.0 - r.image))
        match self.format:
            case AlphaFormat.STRAIGHT:
                return composite
            case AlphaFormat.PREMULTIPLIED:
                return composite.to_premultiplied()
            case AlphaFormat.CLOSENESS_PREMULTIPLIED:
                return composite.to_closeness_premultiplied()

    # -- construction -----------------------------------------------------

    @classmethod
    def blank(cls, height: int, width: int, channels: int = 3,
              batch_shape: Sequence[int] = (), dtype=jnp.float32) -> "Layer":
        """Empty paper: no colour, no coverage, a count of zero everywhere."""
        lead = tuple(batch_shape)
        return cls(count=jnp.zeros(lead + (height, width), jnp.int32),
                   coverage=jnp.zeros(lead + (height, width), dtype),
                   format=AlphaFormat.STRAIGHT,
                   image=jnp.zeros(lead + (height, width, channels), dtype))

    @classmethod
    def colored_mask(cls, alpha: Float[Array, "H W"], color: Float[Array, "C"],
                     level: float = POTENTIAL_EDGE_ALPHA) -> "Layer":
        """One stamped shape of a single colour, labelled at its outline."""
        premultiplied = alpha[..., jnp.newaxis] * color
        count = jnp.where(alpha >= level, 1, 0).astype(jnp.int32)
        return cls(count=count, coverage=alpha,
                   format=AlphaFormat.PREMULTIPLIED, image=premultiplied)

    @classmethod
    def premultiplied(cls, image: Float[Array, "*batch H W C"],
                      level: float=POTENTIAL_EDGE_ALPHA) -> "Layer":
        """Just a regular old image as a single layer."""
        alpha = image[..., -1]
        count = jnp.where(alpha >= level, 1, 0).astype(jnp.int32)
        return cls(count=count, coverage=alpha, format=alpha_format,
                   image=image)

    def over_background(self, background: float = 1.0
                        ) -> Float[Array, "*batch H W C"]:
        """Composite onto a flat background, which is what a likelihood sees.

        Alpha compositing is ``alpha * colour + (1 - alpha) * background``, and
        ``alpha * colour`` is exactly the PREMULTIPLIED image. Compositing the
        STRAIGHT image instead adds the background to a full-strength colour,
        which brightens every partially covered pixel and can leave the result
        above 1: at ``alpha = 0.5`` with colour ``(0.2, 0.24, 0.55)`` on white it
        returned ``(0.70, 0.74, 1.05)`` against a correct ``(0.60, 0.62, 0.775)``.
        """
        premultiplied = self.to_premultiplied()
        uncovered = 1.0 - premultiplied.coverage[..., jnp.newaxis]
        return premultiplied.image + uncovered * background

    @classmethod
    def straight(cls, image: Float[Array, "*batch H W C"],
                 level: float=POTENTIAL_EDGE_ALPHA) -> "Layer":
        """Just a regular old image as a single layer."""
        if image.shape[-1] == 4:
            alpha = image[..., -1]
            count = jnp.where(alpha >= level, 1, 0).astype(jnp.int32)
            return cls(count=count, coverage=alpha, format=AlphaFormat.STRAIGHT,
                       image=image)
        alpha = jnp.ones(image.shape[:-1])
        count = jnp.zeros_like(alpha, dtype=jnp.int32)
        return cls(count=count, coverage=alpha, format=AlphaFormat.STRAIGHT,
                   image=image)

    # -- conversion between formats ---------------------------------------

    def to_closeness_premultiplied(self):
        if self.format == AlphaFormat.STRAIGHT:
            closeness = alpha_to_closeness(self.coverage)
            # The colour is premultiplied by CLOSENESS, not normalised to unit
            # peak, so that ``to_straight`` divides it straight back out and the
            # round trip is exact. ``_ink_kernel`` does normalise to unit peak,
            # because for those glyph masks ``alpha == max(rgb)`` and
            # premultiplying by raw rgb would apply the anti-alias ramp twice.
            # A Layer carries colour and coverage in separate fields, so it has
            # no such double counting and can afford to keep the value.
            return Layer(count=self.count, coverage=closeness,
                         image=self.image * closeness[..., jnp.newaxis],
                         format=AlphaFormat.CLOSENESS_PREMULTIPLIED)
        return self.to_straight().to_closeness_premultiplied()

    def to_premultiplied(self):
        if self.format == AlphaFormat.STRAIGHT:
            return Layer(count=self.count, coverage=self.coverage,
                         image=self.image * self.coverage[..., jnp.newaxis],
                         format=AlphaFormat.PREMULTIPLIED)
        return self.to_straight().to_premultiplied()

    def to_straight(self, eps: float = 1e-6) -> "Layer":
        """Un-premultiply, for display. Undefined where alpha is 0, so 0 there."""
        match self.format:
            case AlphaFormat.STRAIGHT:
                return self
            case AlphaFormat.PREMULTIPLIED:
                a = self.coverage[..., jnp.newaxis]
                alpha = self.coverage
                image = jnp.where(a > eps, self.image / jnp.clip(a, eps, None),
                                  0.0)
            case AlphaFormat.CLOSENESS_PREMULTIPLIED:
                alpha = closeness_to_alpha(self.coverage)
                closeness = self.coverage[..., jnp.newaxis]
                image = jnp.where(closeness > eps,
                                  self.image / jnp.clip(closeness, eps, None),
                                  0.0)

        return Layer(count=self.count, coverage=alpha, image=image,
                     format=AlphaFormat.STRAIGHT)

    # -- diagnostics ------------------------------------------------------

    def region_sizes(self, regions: Optional[int]=3) -> Int[Array, "R"]:
        """Add up the sizes of the different count-regions in the image, capping
        at 3 regions by default in case this function is jitted.
        """
        flat = self.count.reshape((-1, self.height * self.width))
        sizes = jax.vmap(lambda c: jnp.bincount(c, length=regions))(flat)
        return sizes.reshape(self.count.shape[:-2] + (regions,))

    def region_crossing_edges(self) -> Array:
        """How many edges linking a pixel to its neighbor cross region
        boundaries. Each such edge induces a conditional dependency between the
        pixels it connects when we construct a GMRF precision matrix.
        """
        vertical = (self.count[..., :-1, :] != self.count[..., 1:, :])
        horizontal = (self.count[..., :, :-1] != self.count[..., :, 1:])
        return (vertical.sum(axis=(-2, -1)) + horizontal.sum(axis=(-2, -1)))

    # -- Graphical structure ----------------------------------------------

    @property
    def edge_masks(self):
        """Surviving-edge indicators, as ``(vertical, horizontal)``.

        Shapes ``(*batch, H-1, W)`` and ``(*batch, H, W-1)``, in that order, so
        that ``kv, kh = layer.edge_masks`` lines up with the argument order of
        ``_apply_precision(v, tau, kv, kh)``.
        """
        mask_v = jnp.where(self.count[..., :-1, :] == self.count[..., 1:, :],
                           1., 0.)
        mask_h = jnp.where(self.count[..., :, :-1] == self.count[..., :, 1:],
                           1., 0.)
        return mask_v, mask_h

    # -- shape ------------------------------------------------------------

    @property
    def channels(self) -> int:
        return self.image.shape[-1]

    @property
    def height(self) -> int:
        return self.coverage.shape[-2]

    @property
    def width(self) -> int:
        return self.coverage.shape[-1]
