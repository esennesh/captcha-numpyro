from flax import nnx
from flax.nnx.nn.linear import canonicalize_padding
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import math
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import nnx_module
from typing import Optional

from src.data.dictionary import ShapeDictionary
from src.distributions import ConcreteLogits
from src import utils


def over(bg, fg):
    """
    Combines a foreground and background image layer.
    Both inputs should have a shape of (H, W, 4) containing RGBA channels.
    """
    # Split RGB and Alpha channels
    fg_rgb, fg_alpha = fg[..., :3], fg[..., 3:4]
    bg_rgb, bg_alpha = bg[..., :3], bg[..., 3:4]

    # Calculate the combined alpha channel
    out_alpha = fg_alpha + bg_alpha * (1.0 - fg_alpha)

    # Prevent division by zero if both alphas are 0
    safe_alpha = jnp.where(out_alpha == 0.0, 1.0, out_alpha)

    # Calculate the composited RGB colors
    out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1.0 - fg_alpha)
    out_rgb = out_rgb / safe_alpha

    # Combine back into RGBA and return
    return jnp.concatenate([out_rgb, out_alpha], axis=-1)


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


def _logmeanexp(x: Array, axis: int) -> Array:
    return jax.nn.logsumexp(x, axis=axis) - math.log(x.shape[axis])


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


def _render_dictionary_placements(activations: Array, shapes: Array,
                                  stride: int) -> Array:
    dictionary = _as_nhwc_dictionary(shapes)

    def render_one(activation, glyph):
        # With transpose_kernel=True the supplied kernel has the layout of the
        # *forward* conv it transposes: HWIO with I = deconv-output channels
        # (= dictionary channels C) and O = deconv-input channels (= 1 here).
        kernel = glyph[..., jnp.newaxis]
        return jax.lax.conv_transpose(
            activation[..., jnp.newaxis], kernel, (stride, stride), "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            transpose_kernel=True,
        )

    rendered = jax.vmap(render_one, in_axes=(1, 0), out_axes=1)(
        activations, dictionary,
    )
    return jnp.clip(rendered.sum(axis=1), 0., None)


class TopographicPoisson(nnx.Module):
    def __init__(self, kw: int=60, kh: int=60, img_w: int=180, img_h: int=80,
                 num_features: int=36, stride: int=1, rate_init: float=1., *,
                 rngs: nnx.Rngs):
        self.height = (img_h - kh) // stride + 1
        self.width  = (img_w - kw) // stride + 1
        self._num_features = num_features

        # Learnable per-patch base log-rate. exp(-6) ≈ 0.0025 per patch;
        # at 21 * 121 = 2541 patches this gives ~6 expected placements per
        # image — a sensible captcha default. Tune with log_rate_init.
        log_rate_init = jnp.log(
            rate_init / (self.height * self.width * self.num_features)
        )
        self.u_init = nnx.Param(jnp.full((self.num_features, self.height,
                                          self.width),
                                log_rate_init))

    @property
    def num_features(self) -> int:
        return self._num_features

    def __call__(self, rngs=None):
        u = self.u_init # (K, H, W)
        # Per-patch, per-character Poisson presence/intensity. -> (..., K, H, W)
        return numpyro.sample("z_count", dist.Poisson(jnp.exp(u)).to_event(3))

class ExternalKernelConvTranspose(nnx.ConvTranspose):
    def __init__(self, *args, **kwargs):
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.kernel = nnx.data(None)

    def __call__(
        self,
        inputs: Float[Array, "*batch height width in_features"],
        kernel: Float[Array, "kernel_height kernel_width in_features out_features"],
    ) -> Float[Array, "*batch out_height out_width out_features"]:
        def maybe_broadcast(x):
            if x is None:
                x = 1
            if isinstance(x, int):
                return (x,) * len(self.kernel_size)
            return tuple(x)

        num_batch_dimensions = inputs.ndim - (len(self.kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            inputs = jnp.reshape(inputs, (-1,) +\
                     inputs.shape[num_batch_dimensions:])

        strides = maybe_broadcast(self.strides)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        if self.mask is not None:
            kernel = kernel * self.mask

        padding_lax = canonicalize_padding(self.padding, len(self.kernel_size))
        if padding_lax == 'CIRCULAR':
            padding_lax = 'VALID'

        inputs, kernel = self.promote_dtype((inputs, kernel), dtype=self.dtype)

        y = jax.lax.conv_transpose(
            inputs, kernel, strides, padding_lax,
            rhs_dilation=kernel_dilation,
            transpose_kernel=self.transpose_kernel,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        )
        y = jnp.clip(y, 0., None)

        if self.padding == 'CIRCULAR':
            scaled_x_dims = [x * s for x, s in zip(jnp.shape(inputs)[1:-1],
                                                   strides)]
            size_diffs = [-(y_dim - x_dim) % (2 * x_dim)
                          for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)]
            if self.transpose_kernel:
                pad_fn = lambda d: (d // 2, (d + 1) // 2)
            else:
                pad_fn = lambda d: ((d + 1) // 2, d // 2)
            y = jnp.pad(y, [(0, 0)] + [pad_fn(d) for d in size_diffs] +\
                           [(0, 0)])
            for i in range(1, y.ndim - 1):
                y = y.reshape(y.shape[:i] + (-1, scaled_x_dims[i - 1]) +\
                    y.shape[i + 1:])
                y = y.sum(axis=i)

        if num_batch_dimensions != 1:
            y = jnp.reshape(y, input_batch_shape + y.shape[1:])

        return y

class ShapeConvTranspose(nnx.Module):
    def __init__(self, shape_dict: ShapeDictionary, *, rngs: nnx.Rngs,
                 **kwargs):
        channels = shape_dict.shapes.shape[1]
        kernel_size = shape_dict.shapes.shape[-2:]

        self.deconv = ExternalKernelConvTranspose(in_features=1,
                                                  out_features=channels,
                                                  kernel_size=kernel_size,
                                                  **kwargs, rngs=rngs)
        self.shape_dict = shape_dict

    def __call__(self, activations: Array, kernels: Optional[Array] = None,
                 rngs=None):
        if kernels is None:
            kernels = self.shape_dict.shapes
        deconv = jax.vmap(jax.vmap(self.deconv))
        kernel_array = jnp.broadcast_to(
            kernels[jnp.newaxis, ..., jnp.newaxis],
            activations.shape[:1] + kernels.shape + (1,)
        )
        return deconv(activations[..., jnp.newaxis], kernel_array)

class BayesianMarioNettePlacements(nnx.Module):
    """Bayesian MarioNette-style glyph placements from dictionary matches.

    For image ``x``, glyph dictionary ``d_i``, and anchor ``j = (h, w)``,
    this module computes scores

        s[j, i] = sum_{u, v, c} x[h + u, w + v, c] d_i[u, v, c].

    The anchor switch and glyph identity are then explicit random variables:

        z_switch[j] ~ RelaxedBernoulli(logits=b + a logmeanexp_i(s[j, i]))
        z_mark[j]   ~ Concrete(logits=s[j, :]).

    Rendering uses ``z_switch[j] * z_mark[j, i]`` as the dictionary placement
    activation and applies no transformations beyond anchored translation.
    """

    def __init__(self, shape_dict: ShapeDictionary, alpha_sharpness: float = 5.0,
                 mark_temperature: float = 0.5, score_scale: float = 1.0,
                 stride: int = 1, switch_bias: float = -6.0,
                 switch_scale: float = 1.0, switch_temperature: float = 0.5, *,
                 rngs: Optional[nnx.Rngs] = None):
        del rngs
        self.alpha_sharpness = alpha_sharpness
        self.mark_temperature = mark_temperature
        self.score_scale = score_scale
        self.shape_dict = shape_dict
        self.stride = stride
        self.switch_bias = switch_bias
        self.switch_scale = switch_scale
        self.switch_temperature = switch_temperature

    @property
    def num_features(self) -> int:
        return self.shape_dict.shapes.shape[0]

    def score_logits(self, images: Array) -> Array:
        scores = _dictionary_conv_scores(images, self.shape_dict.shapes,
                                         self.stride)
        return self.score_scale * scores

    def switch_logits(self, score_logits: Array) -> Array:
        evidence = _logmeanexp(score_logits, axis=-1)
        return self.switch_bias + self.switch_scale * evidence

    def __call__(self, images: Array, rngs=None):
        del rngs
        mark_logits = self.score_logits(images)
        switch_logits = self.switch_logits(mark_logits)

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

        activations = jnp.moveaxis(z_mark * z_switch[..., jnp.newaxis], -1, 1)
        rendered = _render_dictionary_placements(
            activations, self.shape_dict.shapes, self.stride,
        )
        # Pick the alpha source by inspecting the rendered output's channel
        # layout: an RGBA dictionary carries an explicit alpha in the 4th
        # channel; a single-channel dictionary uses its only channel; an RGB
        # dictionary has no native alpha so we fall back to the channel max
        # as a brightness proxy.
        if rendered.shape[-1] == 1:
            alpha_source = rendered
            rgb_raw = rendered
        elif rendered.shape[-1] == 4:
            alpha_source = rendered[..., 3:4]
            rgb_raw = rendered[..., :3]
        else:  # 3
            alpha_source = rendered.max(axis=-1, keepdims=True)
            rgb_raw = rendered

        # Normalize RGB by the alpha source so the compositing colour is the
        # intrinsic ink colour at each pixel (bounded to [0, 1]) rather than
        # the raw sum over overlapping placements. Otherwise the alpha
        # channel — which saturates via Beer-Lambert — reaches ~1 in regions
        # where the linear RGB sum is still small, and Porter-Duff over
        # paints a dark opaque fringe.
        rgb = jnp.clip(rgb_raw / jnp.maximum(alpha_source, 1e-6), 0., 1.)
        if rendered.shape[-1] == 1:
            rgb = jnp.broadcast_to(rgb, rgb.shape[:-1] + (3,))
        alpha = 1.0 - jnp.exp(-self.alpha_sharpness * alpha_source)
        return jnp.concatenate((rgb, alpha), axis=-1)

class ShapePlacements(nnx.Module):
    """Depth-softmax compositing of glyph placements.

    Each source cell ``c = (k, h', w')`` in the sampled ``wheres`` Poisson
    activation field is treated as a placement whose distance from an imagined
    background decreases as its activation count ``a_c`` grows. The compositing
    weight is ``w_c = a_c ** beta``, and the per-pixel output is an
    activation-weighted average over all placements whose glyph footprint
    covers that pixel:

        color(p) = sum_k conv_transpose(a_k**beta, K_k)(p)
        alpha(p) = 1 - exp(- alpha_sharpness * color(p))

    where ``K_k`` is the glyph kernel for class ``k`` and ``M_k = (K_k > 0)``
    is its binary support mask. ``beta -> infty`` hardens toward winner-take-all
    (closest placement wins per pixel), ``beta = 1`` gives a count-weighted
    blend, and any ``beta > 0`` makes zero-activation cells drop out.
    """

    def __init__(self, prior, shaper: ShapeConvTranspose,
                 num_hiddens=64, beta: float = 1.0,
                 alpha_sharpness: float = 5.0, *, rngs: nnx.Rngs):
        self.prior = prior
        self.shaper = shaper
        self.beta = beta
        self.alpha_sharpness = alpha_sharpness
        assert len(self.shaper.shape_dict.shapes) == prior.num_features

    def __call__(self, rngs=None):
        # The prior emits its own sample sites and returns the packed
        # (..., K, H', W') activation tensor.
        wheres = self.prior(rngs=rngs)

        # Per-source-cell compositing weight a_c ** beta.
        weights = wheres ** self.beta                       # (B, K, H', W')

        # Numerator: conv_transpose with glyph kernels K_k, summed across K.
        # Denominator: conv_transpose with binary masks M_k = (K_k > 0).
        shapes = self.shaper.shape_dict.shapes
        images = self.shaper(weights)                # (B, K, H, W, 1)
        masks = images > 0.
        alphas = masks / (masks.sum(axis=-4, keepdims=True) + 1)
        gray = (images * alphas).sum(axis=-4)        # (B, H, W, 1)

        # Beer-Lambert-style smooth alpha; saturates as coverage accumulates.
        alpha = 1.0 - jnp.exp(-self.alpha_sharpness * gray)

        # Grayscale glyph -> RGB by broadcast, then concat alpha for RGBA.
        rgb = jnp.broadcast_to(gray, gray.shape[:-1] + (3,))
        return jnp.concatenate((rgb, alpha), axis=-1)       # (B, H, W, 4)

class BackgroundDecoder(nnx.Module):
    def __init__(self, embedding_dim: int=50, height=60, hiddens=400, width=160,
                 *, rngs: nnx.Rngs):
        self.bg_shape = (height, width)
        self.decoder = nnx.Sequential(
            nnx.Linear(embedding_dim, hiddens, rngs=rngs), nnx.silu,
            nnx.Linear(hiddens, height * width, rngs=rngs), nnx.sigmoid
        )
        self.embedding_dim = embedding_dim

    def __call__(self, rngs=None):
        loc = jnp.zeros((self.embedding_dim,))
        scale = jnp.ones_like(loc)
        z_bg = numpyro.sample("bg", dist.Normal(loc, scale).to_event(1))
        background = self.decoder(z_bg)
        background = jnp.where(background > 0., background,
                               jnp.ones_like(background))
        return jnp.reshape(background, z_bg.shape[:-1] + self.bg_shape + (1,))

def generate_captcha(placements: ShapePlacements,
                     backgrounder: Optional[BackgroundDecoder]=None):
    rgb_prior = dist.Uniform(jnp.zeros((3,)), jnp.ones((3,)))
    color = numpyro.sample("color", rgb_prior.to_event(1))
    color = color[:, jnp.newaxis, jnp.newaxis, :]
    color = jnp.concatenate((color, jnp.ones(color.shape[:-1] + (1,))), axis=-1)

    foreground = placements()
    foreground = foreground * color
    if backgrounder is not None:
        background = backgrounder() * color
    else:
        background = jnp.ones_like(foreground)

    return utils.soft_clamp(over(background, foreground)[..., :-1], 0., 1.)

def generate_marionette_captcha(images, placements: BayesianMarioNettePlacements,
                                backgrounder: Optional[BackgroundDecoder]=None):
    rgb_prior = dist.Uniform(jnp.zeros((3,)), jnp.ones((3,)))
    color = numpyro.sample("color", rgb_prior.to_event(1))
    color = color[:, jnp.newaxis, jnp.newaxis, :]
    color = jnp.concatenate((color, jnp.ones(color.shape[:-1] + (1,))), axis=-1)

    foreground = placements(images) * color
    if backgrounder is not None:
        background = backgrounder() * color
    else:
        background = jnp.ones_like(foreground)

    return utils.soft_clamp(over(background, foreground)[..., :-1], 0., 1.)

def captcha_model(images, placements: ShapePlacements,
                  backgrounder: Optional[BackgroundDecoder]=None, scale=None):
    placements = nnx_module("placements_p", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_p", backgrounder)
    if scale is None:
        scale = jnp.exp(numpyro.param("log_scale", jnp.zeros(())))
    with numpyro.plate("batch", images.shape[0]):
        prediction = generate_captcha(placements, backgrounder)
        return numpyro.sample("obs", dist.Normal(prediction, scale).to_event(3),
                              obs=images)

def marionette_captcha_model(images, placements: BayesianMarioNettePlacements,
                             backgrounder: Optional[BackgroundDecoder]=None,
                             scale=None):
    placements = nnx_module("placements_p", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_p", backgrounder)
    if scale is None:
        scale = jnp.exp(numpyro.param("log_scale", jnp.zeros(())))
    with numpyro.plate("batch", images.shape[0]):
        prediction = generate_marionette_captcha(images, placements,
                                                 backgrounder)
        prediction = jnp.moveaxis(prediction, -1, -3)
        images = jnp.moveaxis(images, -1, -3)
        return numpyro.sample("obs", dist.Normal(prediction, scale).to_event(3),
                              obs=images)
