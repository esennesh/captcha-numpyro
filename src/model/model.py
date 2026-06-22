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

class SparsePatchPrior(nnx.Module):
    def __init__(self, kw: int=60, kh: int=60, img_w: int=180, img_h: int=80,
                 num_features: int=36, stride: int=1, spikes_init: float=1., *,
                 rngs: nnx.Rngs):
        self.height = (img_h - kh) // stride + 1
        self.width  = (img_w - kw) // stride + 1
        self._num_features = num_features

        # Learnable per-patch base log-rate. exp(-6) ≈ 0.0025 per patch;
        # at 21 * 121 = 2541 patches this gives ~6 expected placements per
        # image — a sensible captcha default. Tune with log_rate_init.
        log_spikes_init = jnp.log(
            spikes_init / (self.height * self.width * self.num_features)
        )
        self.u_0 = nnx.Param(jnp.full((self.num_features, self.height,
                                       self.width),
                                      log_spikes_init))

    @property
    def num_features(self) -> int:
        return self._num_features

    def __call__(self, rngs=None):
        rate = jnp.exp(-self.u_0) # (K, H, W)
        # Per-patch, per-character Poisson presence/intensity. -> (..., K, H, W)
        return numpyro.sample("z_count", dist.Exponential(rate).to_event(3))

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
        rgba = jnp.concatenate((rgb, alpha), axis=-1)       # (B, H, W, 4)
        log_variance = weights.sum(axis=(-3, -2, -1))     # (B,)
        return rgba, jnp.exp(log_variance)

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
    rgb_prior = dist.Beta(jnp.ones((3,)), jnp.ones((3,)))
    color = numpyro.sample("color", rgb_prior.to_event(1))
    color = color[:, jnp.newaxis, jnp.newaxis, :]
    color = jnp.concatenate((color, jnp.ones(color.shape[:-1] + (1,))), axis=-1)

    foreground, dvar = placements()
    foreground = foreground * color
    dvar = dvar[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
    if backgrounder is not None:
        background = backgrounder() * color
    else:
        background = jnp.ones_like(foreground)

    composite = over(background, foreground)[..., :-1]
    composite = jnp.moveaxis(composite, -1, -3)
    return composite, dvar * (composite > 0).astype(dvar.dtype)

def captcha_model(images, placements: ShapePlacements,
                  backgrounder: Optional[BackgroundDecoder]=None, scale=None):
    placements = nnx_module("placements_p", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_p", backgrounder)
    if scale is None:
        scale = jnp.exp(numpyro.param("log_scale", jnp.zeros(())))
    with numpyro.plate("batch", images.shape[0]):
        prediction, dvar = generate_captcha(placements, backgrounder)
        scale = jnp.sqrt(scale ** 2 + dvar)
        numpyro.deterministic("salience_map",
                              utils.gaussian_variance_image(prediction, scale,
                                                            image_layout="HWC"))
        return numpyro.sample("obs", dist.Normal(prediction, scale).to_event(3),
                              obs=images)
