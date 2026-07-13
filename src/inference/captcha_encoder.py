"""MarioNette-style guide for the CAPTCHA generative model's latent surface.

A convolutional backbone produces image-resolution features; three
composition-based heads (placement, colour, background) share those features
and emit the variational distributions for the model's sample sites:

  - ``MarioNettePlacer``      -> ``z_switch``, ``z_mark``   (per-anchor)
  - ``MarioNetteColorFinder`` -> ``color``                  (global)
  - ``BackgroundEncoder``     -> ``bg``                      (global, optional)

The placement head's patch projection is a single ``(kh, kw)`` VALID conv
that maps the backbone features to exactly the anchor grid shape
``(H', W') = ((H - kh)//stride + 1, (W - kw)//stride + 1)`` -- the same
geometry as ``BayesianMarioNettePlacements`` on the generative side, so each
per-anchor posterior parameter sees exactly one glyph's worth of input pixels
and aligns spatially with the corresponding prior cell.
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
from src.distributions import Concrete


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

def _center_scores(scores, axis=-1):
    loc = scores.mean(axis=axis, keepdims=True)
    std = scores.std(axis=(-1, -2, -3), keepdims=True)

    return (scores - loc) / jnp.maximum(std, 1e-6)

def _dictionary_conv_scores(images: Array, shapes: Array, stride: int) -> Array:
    dictionary = _as_nhwc_dictionary(shapes)
    images = _match_dictionary_channels(images, dictionary.shape[-1])
    kernel = jnp.moveaxis(dictionary, 0, -1)
    return jax.lax.conv_general_dilated(images, kernel, (stride, stride),
                                        "VALID",
                                        dimension_numbers=("NHWC", "HWIO",
                                                           "NHWC"))

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

        lambda[j, i] = sum_{u, v, c} x[j + (u, v), c] d_i[u, v, c].
    """

    def __init__(self, shape_dict: ShapeDictionary,
                 backbone_channels: int = 64, feat_dim: int = 64,
                 hidden_dim: int = 128, img_h: int = 60,
                 img_w: int = 160, kh: int = 60, kw: int = 60,
                 mark_temperature: float = 0.5, stride: int = 1,
                 switch_bias: float = -4.0, switch_temperature: float = 0.5, *,
                 rngs: nnx.Rngs):
        self.feat_dim = feat_dim
        self.height = (img_h - kh) // stride + 1
        self.hidden_dim = hidden_dim
        self.kh = kh
        self.kw = kw
        # Fixed hyperparameter, not a learnable parameter (see the note in
        # BayesianMarioNettePlacements): learning the relaxation temperature
        # lets the optimizer exploit the unbounded relaxed-density KL.
        self.mark_temperature = mark_temperature
        self.shape_dict = shape_dict
        self.stride = stride
        self.switch_bias = switch_bias
        # Pointwise stack over the K glyph-score channels: it maps
        # ``scores`` (B, H', W', K) -> (B, H', W', 1) without touching the
        # spatial axes, so the switch logits stay on the anchor grid.
        self.switch_predictor = nnx.Sequential(
            nnx.Conv(self.num_features, feat_dim, (1, 1), rngs=rngs),
            nnx.LayerNorm(feat_dim, rngs=rngs), nnx.leaky_relu,
            nnx.Conv(feat_dim, hidden_dim, (1, 1), rngs=rngs),
            nnx.GroupNorm(hidden_dim, num_groups=_valid_num_groups(hidden_dim),
                          rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(hidden_dim, 1, (1, 1), rngs=rngs)
        )
        self.switch_temperature = switch_temperature
        self.width = (img_w - kw) // stride + 1

    @property
    def num_features(self) -> int:
        return len(self.shape_dict)

    def __call__(self, images: Float[Array, "B H W C_in"]) -> Tuple[Array, Array]:
        scores = _dictionary_conv_scores(images, self.shape_dict.shapes,
                                         self.stride)
        scores = _center_scores(scores)
        mark_dist = Concrete(temperature=self.mark_temperature,
                             logits=scores)
        z_mark = numpyro.sample(
            "z_mark", mark_dist.to_event(2),
        )

        # Per-anchor confidence that *some* glyph is present: treat the scores
        # as parameters to a Dirichlet distribution and take its concentration.
        # The pointwise predictor learns a residual correction on top of this
        # concentration signal.
        numpyro.deterministic("scores", scores)
        # confidence = jnp.exp(scores).sum(axis=-1, keepdims=True)
        uncertainty = mark_dist.entropy()[..., jnp.newaxis]
        numpyro.deterministic("switch_uncertainty", uncertainty)
        switch_logits = (self.switch_predictor(scores) - uncertainty +
                         self.switch_bias).squeeze(-1)
        numpyro.deterministic("switch_logits", switch_logits)
        z_switch = numpyro.sample(
            "z_switch", dist.RelaxedBernoulli(
                temperature=self.switch_temperature, logits=switch_logits,
            ).to_event(2)
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


def marionette_captcha_guide(
    images, backbone: MarioNetteBackbone, placements: MarioNettePlacer,
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
        placements(images)
