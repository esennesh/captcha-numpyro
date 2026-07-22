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
from src.distributions import ConcreteLogits, SpatialMixtureSameFamily
from src import utils


def screen_blend(layers, axis=0, logits=False):
    if logits:
        layers = jax.nn.sigmoid(layers)
    p = 1.0 - jnp.prod(1.0 - layers, axis=axis)
    if logits:
        return jax.scipy.special.logit(p)
    return p

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
    """Render each feature's placements separately, ``(N, K, out_h, out_w, C)``.

    This is the pre-reduction result of :func:`_render_dictionary_placements`:
    the sum over the feature axis (axis 1) recovers the composited render.
    """
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

    return jnp.clip(jax.vmap(render_one, in_axes=(1, 0), out_axes=1)(
        activations, dictionary,
    ), 0., None)

def _dictionary_support_masks(shapes: Array, threshold: float = 0.0) -> Array:
    """Per-glyph 0/1 support masks, shaped ``(K, kh, kw, 1)``.

    A glyph "touches" a pixel wherever its alpha support exceeds ``threshold``.
    The alpha source follows the same convention as
    :meth:`BayesianMarioNettePlacements.render`: the explicit alpha channel for
    RGBA glyphs, the sole channel for single-channel glyphs, and the per-pixel
    channel max as a brightness proxy for RGB glyphs.
    """
    dictionary = _as_nhwc_dictionary(shapes)
    if dictionary.shape[-1] == 4:
        alpha = dictionary[..., 3:4]
    elif dictionary.shape[-1] == 1:
        alpha = dictionary
    else:
        alpha = dictionary.max(axis=-1, keepdims=True)
    return (alpha > threshold).astype(dictionary.dtype)


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
        # Relaxation temperatures are fixed hyperparameters, not learnable
        # parameters: a learnable temperature on the reparameterized relaxed
        # ELBO lets the optimizer game the (unbounded) relaxed-density KL.
        self.mark_temperature = mark_temperature
        self.shape_dict = shape_dict
        self.stride = stride
        self.switch_temperature = switch_temperature
        self.width = (img_w - kw) // stride + 1

    @property
    def num_features(self) -> int:
        return self.shape_dict.shapes.shape[0]

    def sample_latents(self):
        """Draw the per-anchor switch and glyph-identity latents.

        Returns ``(z_switch, z_mark)`` with shapes ``(1, H, W)`` and
        ``(1, H, W, K)`` (broadcast over any enclosing plate). These are the
        ``numpyro.sample`` sites ``"z_switch"`` and ``"z_mark"`` that
        :meth:`__call__` also draws; call this once and thread the results
        through :meth:`render` and :meth:`glyph_coverage` so both consume the
        same samples rather than re-sampling the sites.
        """
        mark_temperature = numpyro.param("mark_temperature",
                                         jnp.log(self.mark_temperature))
        mark_temperature = jnp.exp(mark_temperature)
        z_mark = numpyro.sample(
            "z_mark", ConcreteLogits(
                logits=jnp.zeros((1, self.height, self.width,
                                  len(self.shape_dict))),
                temperature=mark_temperature
            ).to_event(2)
        )

        expected_logit = math.log(self.expected_switches) -\
                         math.log(self.height * self.width -
                                  self.expected_switches)
        switch_temperature = numpyro.param("switch_log_temperature",
                                           jnp.log(self.switch_temperature))
        switch_temperature = jnp.exp(switch_temperature)
        z_switch = numpyro.sample(
            "z_switch", dist.RelaxedBernoulli(
                logits=jnp.ones((1, self.height, self.width)) * expected_logit,
                temperature=switch_temperature
            ).to_event(2)
        )
        return z_switch, z_mark

    def glyph_coverage(self, z_switch, z_mark, *, threshold: float = 0.0):
        """Per-glyph soft coverage, shape ``(..., out_h, out_w, K)``.

        Entry ``[..., i]`` is ``sum_j z_switch[j] * z_mark[j, i] * mask_i[p - j]``:
        how much glyph ``i``, as placed by the active switches, covers each output
        pixel. It reuses the exact placement geometry of :meth:`render` (anchored
        ``conv_transpose`` at ``self.stride``) but renders each glyph's 0/1 support
        mask instead of its intensity, so the value counts overlapping placements
        rather than summing their ink.

        Because ``z_switch``/``z_mark`` are relaxed (continuous in ``[0, 1]``) the
        counts are soft, and they are differentiable in the latents.

        :param z_switch: per-anchor switch activations, shape ``(..., H, W)``.
        :param z_mark: per-anchor glyph identities, shape ``(..., H, W, K)``.
        :param threshold: a glyph touches a pixel where its alpha support
            exceeds this value.
        :return: per-glyph coverage, shape ``(..., out_h, out_w, K)``.
        """
        masks = _dictionary_support_masks(self.shape_dict.shapes, threshold)
        activations = jnp.moveaxis(z_mark * z_switch[..., jnp.newaxis], -1, 1)
        per_feature = _render_dictionary_placements(activations, masks,
                                                    self.stride)
        return jnp.clip(per_feature, 0., None)  # (..., K, out_h, out_w, 1)

    def render(self, z_switch, z_mark):
        """Composite the RGBA foreground implied by the given latents."""
        activations = jnp.moveaxis(z_mark * z_switch[..., jnp.newaxis], -1, 1)
        return _render_dictionary_placements(activations,
                                             self.shape_dict.shapes,
                                             self.stride)

    def __call__(self, rngs=None):
        del rngs
        z_switch, z_mark = self.sample_latents()
        rgba = self.render(z_switch, z_mark)
        coverage = self.glyph_coverage(z_switch, z_mark)
        return rgba, coverage

class PoissonMarkedPlacements(nnx.Module):
    """Marked-Poisson-process glyph placements, without relaxations.

    Anchors compete to fire through a normalized intensity field: a total
    firing mass ``z_rate ~ Gamma`` (whose prior mean is the expected glyph
    count) is allocated across anchors by a sparse ``z_where ~ Dirichlet``,
    giving per-anchor rates ``lambda_j = z_rate * z_where_j``. Glyph
    identities compete within each patch through an explicit per-anchor mark
    simplex ``z_mark[j] ~ Dirichlet(mark_concentration * ones(K))``, sparse
    when ``mark_concentration < 1`` so marks genuinely compete to fire.

    Discrete firing is never sampled. The per-(anchor, glyph) intensities
    ``lambda_j * rho_jm`` are the sufficient statistics of the process:
    by the marking and superposition theorems the pixelwise mixture
    downstream is its exact marginalization, with mixture weights given by
    the first-arrival race over intensities and the Beer-Lambert
    ``exp(-coverage)`` background term equal to the Poisson void
    probability. Every latent is Gamma/Dirichlet/Normal -- no temperatures,
    parameter-free supports, valid KLs, and a safely mirrorable mean field.
    """

    def __init__(self, shape_dict: ShapeDictionary,
                 count_concentration: float=2., expected_count: float=1.,
                 img_h: int=60, img_w: int=160, kh: int=60, kw: int=60,
                 mark_concentration: float=0.5, stride: int=1,
                 where_concentration: float=0.5, *,
                 rngs: Optional[nnx.Rngs]=None):
        del rngs
        self.count_concentration = count_concentration
        self.expected_count = expected_count
        self.height = (img_h - kh) // stride + 1
        self.mark_concentration = mark_concentration
        self.shape_dict = shape_dict
        self.stride = stride
        self.where_concentration = where_concentration
        self.width = (img_w - kw) // stride + 1

    @property
    def num_features(self) -> int:
        return self.shape_dict.shapes.shape[0]

    def sample_latents(self):
        """Draw the intensity field and mark simplices.

        Returns ``(intensity, marks)`` with shapes ``(..., H, W)`` and
        ``(..., H, W, K)``; their product is the per-(anchor, glyph) firing
        intensity that drives both rendering and coverage.
        """
        z_rate = numpyro.sample("z_rate", dist.Gamma(
            self.count_concentration,
            self.count_concentration / self.expected_count,
        ))
        concentration = jnp.full((self.height * self.width,),
                                 self.where_concentration)
        z_where = numpyro.sample("z_where", dist.Dirichlet(concentration))
        marks = numpyro.sample("z_mark", dist.Dirichlet(jnp.full(
            (1, self.height, self.width, len(self.shape_dict)),
            self.mark_concentration,
        )).to_event(2))
        intensity = z_rate[..., jnp.newaxis] * z_where
        intensity = intensity.reshape(z_where.shape[:-1] +
                                      (self.height, self.width))
        return intensity, marks

    def glyph_coverage(self, activations, *, threshold: float = 0.0):
        """Intensity-weighted support coverage, ``(..., K, out_h, out_w, 1)``.

        Entry ``[..., m, x, 1]`` is ``sum_j activations[j, m] * mask_m(x - j)``:
        the total firing intensity claiming pixel ``x`` with glyph ``m``.
        """
        masks = _dictionary_support_masks(self.shape_dict.shapes, threshold)
        per_feature = _render_dictionary_placements(activations, masks,
                                                    self.stride)
        return jnp.clip(per_feature, 0., None)

    def render(self, activations, coverage):
        """Intensity-weighted mean sprite appearance per glyph.

        The raw render scales sprite ink with the (unbounded) intensity;
        dividing by the identically-weighted support coverage makes the
        mixture-component means scale-invariant in the intensities, leaving
        the intensities to act only through the mixture weights.
        """
        ink = _render_dictionary_placements(activations,
                                            self.shape_dict.shapes,
                                            self.stride)
        return jnp.clip(ink, 0., None) / jnp.clip(coverage, 1e-6, None)

    def __call__(self, rngs=None):
        del rngs
        intensity, marks = self.sample_latents()
        activations = jnp.moveaxis(marks * intensity[..., jnp.newaxis], -1, 1)
        coverage = self.glyph_coverage(activations)
        rgba = self.render(activations, coverage)
        return rgba, coverage

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

def generate_marionette_captcha(placements: BayesianMarioNettePlacements,
                                backgrounder: Optional[BackgroundDecoder]=None):
    # Beta(1, 1) is the uniform density on [0, 1], but unlike Uniform its
    # parameters carry plain positive constraints and its support is fixed:
    # a mean-field guide can mirror it with learnable concentrations without
    # ever proposing colors outside the prior's support.
    rgb_prior = dist.Beta(jnp.ones((3,)), jnp.ones((3,)))
    color = numpyro.sample("color", rgb_prior.to_event(1))
    color = color[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

    rgba, coverage = placements()
    canvas_shape = list(rgba.shape)
    canvas_shape[-4] = 1
    canvas_shape = tuple(canvas_shape)
    foreground = rgba[..., :-1] * color
    if backgrounder is not None:
        background = backgrounder() * color
    else:
        background = jnp.ones(canvas_shape[:-1] + (3,))

    prediction = jnp.concatenate((background, foreground), axis=-4)

    # ``coverage`` is a soft count of overlapping placements and can exceed 1, so
    # the background "survival" weight -- the probability that no glyph covers the
    # pixel -- is the Beer-Lambert / Poisson zero-count term exp(-sum coverage).
    # This stays in (0, 1] by construction (never negative, unlike prod(1 -
    # coverage) once a count exceeds 1), agrees with the product form to first
    # order, and keeps the mixture weights a valid non-negative simplex.
    transmittance = jnp.exp(-jnp.sum(coverage, axis=-4, keepdims=True))
    coverage = jnp.concatenate((transmittance, coverage), axis=-4)
    return prediction, coverage

def marionette_captcha_model(images, placements: BayesianMarioNettePlacements,
                             backgrounder: Optional[BackgroundDecoder]=None,
                             scale=None):
    # One Gaussian mixture component per dictionary glyph, plus a background
    # component (index 0); each carries its own learnable log-scale. Read the
    # glyph count from the raw module before wrapping, since the nnx_module
    # wrapper exposes only ``__call__``.
    n_components = placements.num_features + 1
    placements = nnx_module("placements_p", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_p", backgrounder)

    if scale is None:
        scale = jnp.exp(numpyro.param("log_scale", jnp.zeros((1,))))
    # Per-component scale, shaped to broadcast over the trailing channel axis of
    # ``prediction`` (whose component axis is -2 and channel axis is -1).
    scale = jnp.broadcast_to(jnp.asarray(scale), (n_components,))[:, jnp.newaxis]

    batch_size = images.shape[0] if images is not None else 1
    with numpyro.plate("batch", batch_size):
        prediction, coverage = generate_marionette_captcha(placements,
                                                           backgrounder)
        # Keep everything in NHWC so the color channel is the *event* of the
        # component distribution, not a batch dim of the mixture. If the channel
        # stayed in the mixture batch, the assignment categorical would fire
        # independently per channel and pick a different component for R, G and B
        # at the same pixel -- producing saturated per-channel speckle wherever
        # the coverage weights are split. Folding the channel into the component
        # event makes a single per-pixel assignment select the whole RGB vector,
        # so colors stay correlated within a pixel even when the latents are soft.
        prediction = jnp.moveaxis(prediction, -4, -2)  # (B, H, W, K, C)
        coverage = jnp.moveaxis(coverage[..., 0], -3, -1)  # (B, H, W, K)

        # The coverage gives mixture weights, so normalize it to a simplex. Empty
        # pixels legitimately place zero weight on glyph components; the mixture's
        # linear-space log_prob handles zero weights exactly (finite gradients),
        # so no epsilon floor is needed.
        coverage = coverage / coverage.sum(axis=-1, keepdims=True)
        likelihood = SpatialMixtureSameFamily(
            dist.Categorical(probs=coverage),          # batch (B, H, W)
            dist.Normal(prediction, scale).to_event(1),  # batch (B, H, W, K), event (C,)
            reinterpreted_batch_ndims=2,               # fold (H, W); event -> (H, W, C)
        )
        if images is not None:
            numpyro.deterministic("residual", (images - likelihood.mean) ** 2)
        return numpyro.sample("obs", likelihood, obs=images)
