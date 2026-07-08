from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array
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

    def __init__(self, shape_dict: ShapeDictionary, alpha_sharpness: float=5.0,
                 expected_switches: int=1, img_h: int = 60, img_w: int = 160,
                 kh: int = 60, kw: int = 60, mark_temperature: float = 0.1,
                 stride: int = 1, switch_temperature: float = 0.1, *,
                 rngs: Optional[nnx.Rngs]=None):
        del rngs
        self.alpha_sharpness = alpha_sharpness
        self.expected_switches = expected_switches
        self.height = (img_h - kh) // stride + 1
        self.mark_temperature = nnx.Param(jnp.array(mark_temperature))
        self.shape_dict = shape_dict
        self.stride = stride
        self.switch_temperature = nnx.Param(jnp.array(switch_temperature))
        self.width = (img_w - kw) // stride + 1

    @property
    def num_features(self) -> int:
        return self.shape_dict.shapes.shape[0]

    def __call__(self, rngs=None):
        del rngs

        z_mark = numpyro.sample(
            "z_mark", ConcreteLogits(
                logits=jnp.zeros((1, self.height, self.width,
                                  len(self.shape_dict))),
                temperature=self.mark_temperature
            ).to_event(2)
        )
        expected_logit = math.log(self.expected_switches) -\
                         math.log(self.height * self.width -
                                  self.expected_switches)
        z_switch = numpyro.sample(
            "z_switch", dist.RelaxedBernoulli(
                logits=jnp.ones((1, self.height, self.width)) * expected_logit,
                temperature=self.switch_temperature
            ).to_event(2)
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

def generate_marionette_captcha(images, placements: BayesianMarioNettePlacements,
                                backgrounder: Optional[BackgroundDecoder]=None):
    rgb_prior = dist.Uniform(jnp.zeros((3,)), jnp.ones((3,)))
    color = numpyro.sample("color", rgb_prior.to_event(1))
    color = color[:, jnp.newaxis, jnp.newaxis, :]
    color = jnp.concatenate((color, jnp.ones(color.shape[:-1] + (1,))), axis=-1)

    foreground = placements() * color
    if backgrounder is not None:
        background = backgrounder() * color
    else:
        background = jnp.ones_like(foreground)

    return utils.soft_clamp(over(background, foreground)[..., :-1], 0., 1.)

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
