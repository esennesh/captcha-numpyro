"""SPAIR-style encoder for the CAPTCHA generative model's latent surface.

A single stride-1 SAME-padding CNN backbone produces image-resolution
features; three composition-based heads (placement, colour, background)
share those features and emit the variational distributions for the
model's sample sites:

  - ``ShapePlacer``        -> ``z_count``, ``z_mark``     (per-patch)
  - ``ColorFinder``        -> ``color``                   (global)
  - ``BackgroundEncoder``  -> ``bg``                       (global, optional)

The placement head's patch projection is a single ``(kh, kw)`` VALID conv
that maps the backbone features to exactly the placement grid shape
``(H', W') = ((H - kh)//stride + 1, (W - kw)//stride + 1)`` -- the same
geometry as ``PoissonGatedSlabPrior`` on the generative side, so each
per-patch posterior parameter sees exactly one glyph's worth of input
pixels and aligns spatially with the corresponding prior cell.

The guide structurally mirrors the prior: ``z_mark``'s posterior is a
``GatedSpikeAndSlab`` gated by the encoder's *own* ``z_count`` sample,
keeping model and guide log-probs consistent at empty patches.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from flax import nnx
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import nnx_module

from src.distributions import GatedSpikeAndSlab, OneHotCategorical


def softplus_inverse(x):
    return jnp.log(jnp.expm1(x))


class Backbone(nnx.Module):
    """Shared image-resolution CNN backbone.

    Stride-1 SAME-padding so spatial dimensions are preserved, letting
    downstream patch / global heads pick their own pooling geometry.
    """

    def __init__(self, in_channels: int = 3,
                 hidden_dims: Tuple[int, ...] = (16, 32, 64),
                 *, rngs: nnx.Rngs):
        layers = []
        ch_in = in_channels
        for ch_out in hidden_dims:
            layers.append(nnx.Conv(ch_in, ch_out, (3, 3),
                                   padding="SAME", rngs=rngs))
            layers.append(nnx.relu)
            ch_in = ch_out
        self.layers = nnx.Sequential(*layers)
        self.out_channels = hidden_dims[-1]

    def __call__(
        self, images: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H W C_out"]:
        return self.layers(images)

class ShapePlacer(nnx.Module):
    """SPAIR-style per-patch placement encoder.

    Takes the shared backbone features, projects them to the placement
    grid resolution via a single ``(kh, kw)`` VALID conv, then emits
    per-patch ``z_count`` (Poisson) and ``z_mark`` (gated spike-and-slab)
    sample sites.
    """

    def __init__(self, backbone_channels: int = 64, count_init: float = 1.,
                 feat_dim: int = 64, img_h: int = 80, img_w: int = 180,
                 kh: int = 60, kw: int = 60, num_features: int = 36,
                 stride: int = 1, *, rngs: nnx.Rngs):
        self.kw, self.kh = kw, kh
        self.stride = stride
        self.height = (img_h - kh) // stride + 1
        self.width  = (img_w - kw) // stride + 1
        self.num_features = num_features
        count_rate_init = count_init / (self.height * self.width *
                                        self.num_features)
        count_bias_init = float(softplus_inverse(jnp.asarray(count_rate_init)))

        def count_head_bias_init(key, shape, dtype=jnp.float32):
            return jnp.full(shape, count_bias_init, dtype)

        # Patch projection: (kh, kw) VALID conv, stride=stride. Output
        # shape matches the placement prior's grid exactly, and the
        # kh×kw receptive field per output position is one glyph's
        # worth of input pixels.
        self.to_patches = nnx.Sequential(
            nnx.Conv(backbone_channels, feat_dim, (kh, kw), padding="VALID",
                     strides=(stride, stride), rngs=rngs),
            nnx.relu
        )

        # Per-patch heads (1×1 convs over the (H', W', feat_dim) map).
        self.count_head = nnx.Conv(feat_dim, self.num_features, (1, 1),
                                   bias_init=count_head_bias_init,
                                   kernel_init=jax.nn.initializers.zeros,
                                   rngs=rngs)

    def __call__(
        self, features: Float[Array, "B H W C_feat"]
    ) -> Tuple[Array, Array]:
        B = features.shape[0]
        patches = self.to_patches(features)    # (B, H', W', feat_dim)

        # Poisson rate for the count. softplus ensures nonnegativity;
        # squeeze the trailing channel of size 1.
        rate = jax.nn.softplus(self.count_head(patches))  # (B, H', W', K)
        rate = jnp.moveaxis(rate, -1, -3) # (B, K, H', W')
        return numpyro.sample("z_count", dist.Poisson(rate).to_event(3))

class ColorFinder(nnx.Module):
    """Global RGB encoder.

    Global-average-pools the backbone features, then emits ``color``
    (3-d Normal with sigmoid-loc, softplus-scale).
    """

    def __init__(self, backbone_channels: int = 64,
                 hidden_dim: int = 128, *, rngs: nnx.Rngs):
        self.head = nnx.Sequential(
            nnx.Linear(backbone_channels, hidden_dim, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dim, 3 * 2, rngs=rngs),
        )

    def __call__(
        self, features: Float[Array, "B H W C_feat"]
    ) -> Float[Array, "B 3"]:
        x = features.mean(axis=(1, 2))                # (B, C_feat)
        params = self.head(x).reshape(-1, 2, 3)       # (B, 2, 3)
        return numpyro.sample(
            "color",
            dist.Normal(
                jax.nn.sigmoid(params[:, 0]),
                jax.nn.softplus(params[:, 1]),
            ).to_event(1),
        )


class BackgroundEncoder(nnx.Module):
    """Global background-latent encoder.

    Global-average-pools the backbone features, then emits ``bg``
    (Normal with diagonal scale).
    """

    def __init__(self, backbone_channels: int = 64,
                 embedding_dim: int = 50, hidden_dim: int = 128,
                 *, rngs: nnx.Rngs):
        self.embedding_dim = embedding_dim
        self.head = nnx.Sequential(
            nnx.Linear(backbone_channels, hidden_dim, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dim, embedding_dim * 2, rngs=rngs),
        )

    def __call__(
        self, features: Float[Array, "B H W C_feat"]
    ) -> Float[Array, "B D"]:
        x = features.mean(axis=(1, 2))                              # (B, C_feat)
        params = self.head(x).reshape(-1, 2, self.embedding_dim)    # (B, 2, D)
        return numpyro.sample(
            "bg",
            dist.Normal(
                params[:, 0],
                jax.nn.softplus(params[:, 1]),
            ).to_event(1),
        )


def captcha_guide(images, backbone: Backbone, placements: ShapePlacer,
                  color_finder: ColorFinder,
                  backgrounder: Optional[BackgroundEncoder] = None):
    """Guide function mirroring ``captcha_model``.

    Runs the shared backbone once and feeds its features to each head.
    The ``bg`` site is only emitted when a ``BackgroundEncoder`` is
    provided, matching the model's conditional ``backgrounder``.
    """
    backbone = nnx_module("backbone_q", backbone)
    placements = nnx_module("placements_q", placements)
    color_finder = nnx_module("color_finder_q", color_finder)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_q", backgrounder)

    with numpyro.plate("batch", images.shape[0]):
        features = backbone(images)
        color_finder(features)
        if backgrounder is not None:
            backgrounder(features)
        placements(features)
