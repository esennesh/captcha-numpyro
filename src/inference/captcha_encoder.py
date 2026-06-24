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

from src.data.dictionary import ShapeDictionary
from src.distributions import (ConcreteLogits, GatedSpikeAndSlab,
                               OneHotCategorical)


def _as_nhwc_dictionary(shapes: Array) -> Array:
    shapes = jnp.asarray(shapes)
    if shapes.ndim != 4:
        raise ValueError(f"Expected dictionary shape (K, H, W, C), got {shapes.shape}")
    if shapes.shape[-1] in (1, 3, 4):
        return shapes
    if shapes.shape[1] in (1, 3, 4):
        return jnp.moveaxis(shapes, 1, -1)
    raise ValueError(
        "Could not infer dictionary channel axis; expected either "
        f"(K, H, W, C) or (K, C, H, W), got {shapes.shape}"
    )


def _dictionary_conv_scores(images: Array, shapes: Array, stride: int) -> Array:
    dictionary = _as_nhwc_dictionary(shapes)
    images = _match_dictionary_channels(images, dictionary.shape[-1])
    kernel = jnp.moveaxis(dictionary, 0, -1)
    return jax.lax.conv_general_dilated(
        images, kernel, (stride, stride), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def _match_dictionary_channels(images: Array, channels: int) -> Array:
    image_channels = images.shape[-1]
    if image_channels == channels:
        return 1.0 - images
    if channels == 1:
        return 1.0 - images.mean(axis=-1, keepdims=True)
    if image_channels == 1:
        images = 1.0 - images
        return jnp.broadcast_to(images, images.shape[:-1] + (channels,))
    if image_channels < channels:
        # Pad missing trailing channels (e.g. RGB -> RGBA) with 1's so the
        # subsequent (1 - x) inversion zeroes them out and they contribute
        # nothing to the dictionary-matching score.
        pad_shape = images.shape[:-1] + (channels - image_channels,)
        images = jnp.concatenate(
            [images, jnp.ones(pad_shape, dtype=images.dtype)], axis=-1,
        )
        return 1.0 - images
    raise ValueError(
        f"Cannot match image channels {image_channels} to dictionary channels {channels}"
    )


def _valid_num_groups(num_features: int, max_groups: int = 32) -> int:
    for num_groups in range(min(max_groups, num_features), 0, -1):
        if num_features % num_groups == 0:
            return num_groups
    return 1


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


class MarioNetteBackbone(nnx.Module):
    """MarioNette-style convolutional encoder stem.

    The original paper uses partial-convolution downsampling blocks with Group
    Normalization and Leaky ReLU. For the Captcha model we keep stride 1 by
    default so the later VALID patch projection can produce the exact same
    anchor grid as the generative dictionary convolution.
    """

    def __init__(self, hidden_dims: Tuple[int, ...] = (16, 32, 64),
                 in_channels: int = 3, max_groups: int = 32,
                 *, rngs: nnx.Rngs):
        channels = in_channels
        layers = []
        for hidden_dim in hidden_dims:
            num_groups = _valid_num_groups(hidden_dim, max_groups)
            layers.append(nnx.Conv(channels, hidden_dim, (3, 3),
                                   padding="SAME", rngs=rngs))
            layers.append(nnx.GroupNorm(hidden_dim, num_groups=num_groups,
                                        rngs=rngs))
            layers.append(nnx.leaky_relu)
            channels = hidden_dim
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

    def __init__(self, backbone_channels: int = 64, kw: int = 60, kh: int = 60,
                 img_w: int = 180, img_h: int = 80, num_features: int = 36,
                 stride: int = 1, feat_dim: int = 64, *, rngs: nnx.Rngs):
        self.kw, self.kh = kw, kh
        self.stride = stride
        self.height = (img_h - kh) // stride + 1
        self.width  = (img_w - kw) // stride + 1
        self.num_features = num_features

        # Patch projection: (kh, kw) VALID conv, stride=stride. Output
        # shape matches the placement prior's grid exactly, and the
        # kh×kw receptive field per output position is one glyph's
        # worth of input pixels.
        self.to_patches = nnx.Conv(
            backbone_channels, feat_dim, (kh, kw),
            padding="VALID", strides=(stride, stride), rngs=rngs,
        )

        # Per-patch heads (1×1 convs over the (H', W', feat_dim) map).
        self.count_head = nnx.Conv(feat_dim, 1, (1, 1), rngs=rngs)
        self.mark_head  = nnx.Conv(feat_dim, num_features, (1, 1), rngs=rngs)

    def __call__(
        self, features: Float[Array, "B H W C_feat"]
    ) -> Tuple[Array, Array]:
        B = features.shape[0]
        patches = nnx.relu(self.to_patches(features))    # (B, H', W', feat_dim)

        # Poisson rate for the count. softplus ensures nonnegativity;
        # squeeze the trailing channel of size 1.
        rate = jax.nn.softplus(self.count_head(patches)[..., 0])  # (B, H', W')
        z_count = numpyro.sample(
            "z_count", dist.Poisson(rate).to_event(2),
        )

        # Mark: properly gated spike-and-slab on the encoder side, exactly
        # mirroring the prior so model/guide log-probs match at empty patches.
        K = self.num_features
        mark_logits = self.mark_head(patches)                   # (B, H', W', K)
        spike = dist.Delta(jnp.zeros((B, 1, 1, K)), event_dim=1)
        slab  = OneHotCategorical(logits=mark_logits)
        z_mark = numpyro.sample(
            "z_mark",
            GatedSpikeAndSlab(z_count, spike, slab).to_event(2),
        )
        return z_count, z_mark


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


class MarioNetteColorFinder(nnx.Module):
    """Global RGB guide with support matching the model's Uniform prior."""

    def __init__(self, backbone_channels: int = 64,
                 hidden_dim: int = 128, min_concentration: float = 1e-3,
                 *, rngs: nnx.Rngs):
        self.head = nnx.Sequential(
            nnx.Linear(backbone_channels, hidden_dim, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dim, 3 * 2, rngs=rngs),
        )
        self.min_concentration = min_concentration

    def __call__(
        self, features: Float[Array, "B H W C_feat"]
    ) -> Float[Array, "B 3"]:
        x = features.mean(axis=(1, 2))
        params = self.head(x).reshape(-1, 2, 3)
        concentration0 = jax.nn.softplus(params[:, 0]) + self.min_concentration
        concentration1 = jax.nn.softplus(params[:, 1]) + self.min_concentration
        return numpyro.sample(
            "color", dist.Beta(concentration1, concentration0).to_event(1),
        )


class MarioNettePlacer(nnx.Module):
    """Guide for MarioNette-style switch and dictionary-match mark latents.

    The guide factorizes across anchors as

        q(z_switch, z_mark | x) = prod_j q(z_switch[j] | x)
                                        q(z_mark[j] | x),

    where

        q(z_switch[j] | x) = RelaxedBernoulli(logits=eta[j])
        q(z_mark[j] | x)   = Concrete(logits=lambda[j, :]).

    A ``(kh, kw)`` VALID convolution maps image-resolution backbone features to
    the same anchor grid as the model's dictionary convolution:

        H' = floor((H - kh) / stride) + 1,
        W' = floor((W - kw) / stride) + 1.

    The switch logits are amortized from encoder features and answer whether
    anything is present at anchor ``j``. The mark logits are direct dictionary
    convolution scores and answer which glyph best matches the observed image
    patch at that anchor:

        lambda[j, i] = score_scale * sum_{u, v, c} x[j + (u, v), c] d_i[u, v, c].
    """

    def __init__(self, shape_dict: ShapeDictionary,
                 backbone_channels: int = 64, feat_dim: int = 64,
                 hidden_dim: int = 128, img_h: int = 60,
                 img_w: int = 160, kh: int = 60, kw: int = 60,
                 mark_temperature: float = 0.5, score_scale: float = 1.0,
                 stride: int = 1, switch_bias: float = -4.0,
                 switch_temperature: float = 0.5, *, rngs: nnx.Rngs):
        self.feat_dim = feat_dim
        self.height = (img_h - kh) // stride + 1
        self.hidden_dim = hidden_dim
        self.kh = kh
        self.kw = kw
        self.mark_temperature = mark_temperature
        self.patch_norm = nnx.LayerNorm(feat_dim, rngs=rngs)
        self.score_scale = score_scale
        self.shape_dict = shape_dict
        self.stride = stride
        self.switch_bias = switch_bias
        self.switch_head = nnx.Conv(hidden_dim, 1, (1, 1), rngs=rngs)
        self.switch_hidden = nnx.Conv(feat_dim, hidden_dim, (1, 1), rngs=rngs)
        self.switch_norm = nnx.GroupNorm(
            hidden_dim, num_groups=_valid_num_groups(hidden_dim), rngs=rngs,
        )
        self.switch_temperature = switch_temperature
        self.to_patches = nnx.Conv(
            backbone_channels, feat_dim, (kh, kw),
            padding="VALID", strides=(stride, stride), rngs=rngs,
        )
        self.width = (img_w - kw) // stride + 1

    @property
    def num_features(self) -> int:
        return self.shape_dict.shapes.shape[0]

    def __call__(
        self, images: Float[Array, "B H W C_in"],
        features: Float[Array, "B H W C_feat"],
    ) -> Tuple[Array, Array]:
        mark_logits = self.score_scale * _dictionary_conv_scores(
            images, self.shape_dict.shapes, self.stride,
        )
        patches = self.patch_norm(self.to_patches(features))
        switch_hidden = nnx.leaky_relu(self.switch_norm(
            self.switch_hidden(patches),
        ))
        switch_logits = self.switch_bias + self.switch_head(switch_hidden)[..., 0]

        z_mark = numpyro.sample(
            "z_mark", ConcreteLogits(
                temperature=self.mark_temperature, logits=mark_logits,
            ).to_event(2),
        )
        z_switch = numpyro.sample(
            "z_switch", dist.RelaxedBernoulli(
                temperature=self.switch_temperature, logits=switch_logits,
            ).to_event(2),
        )
        return z_switch, z_mark


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


def marionette_captcha_guide(
    images, backbone: Backbone, placements: MarioNettePlacer,
    color_finder: MarioNetteColorFinder,
    backgrounder: Optional[BackgroundEncoder] = None,
):
    """Guide function mirroring ``marionette_captcha_model``."""
    backbone = nnx_module("backbone_q", backbone)
    color_finder = nnx_module("color_finder_q", color_finder)
    placements = nnx_module("placements_q", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_q", backgrounder)

    with numpyro.plate("batch", images.shape[0]):
        features = backbone(images)
        color_finder(features)
        if backgrounder is not None:
            backgrounder(features)
        placements(images, features)
