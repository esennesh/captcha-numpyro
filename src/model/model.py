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


def _dictionary_alpha_rgb(shapes: Array) -> tuple[Array, Array]:
    """Split a dictionary into ``(alpha, rgb)``, shaped ``(K, kh, kw, 1|3)``.

    The alpha convention follows :func:`_dictionary_support_masks`: the explicit
    alpha channel for RGBA glyphs, the sole channel for single-channel glyphs,
    and the per-pixel channel max as a brightness proxy for RGB glyphs.
    """
    dictionary = _as_nhwc_dictionary(shapes)
    if dictionary.shape[-1] == 4:
        alpha, rgb = dictionary[..., 3:4], dictionary[..., :3]
    elif dictionary.shape[-1] == 1:
        alpha = dictionary
        rgb = jnp.broadcast_to(dictionary, dictionary.shape[:-1] + (3,))
    else:
        alpha, rgb = dictionary.max(axis=-1, keepdims=True), dictionary
    return alpha, rgb


def _ink_kernel(shapes: Array) -> Array:
    """The stamping kernel, shaped ``(kh, kw, 4, K)`` for ``conv_transpose``.

    Channels 0:3 carry *premultiplied* colour ``alpha_k * rgb_k`` and channel 3
    carries ``alpha_k`` alone, so a single transposed convolution of the count
    field emits both the colour numerator and the optical depth that normalizes
    it (see :meth:`PoissonConvPlacements.ink_field`).

    The layout is HWIO as ``conv_transpose(..., transpose_kernel=True)`` wants
    it: ``I`` is the deconvolution's *output* channels (the 4 ink channels) and
    ``O`` its *input* channels (the ``K`` dictionary features).
    """
    alpha, rgb = _dictionary_alpha_rgb(shapes)
    # Premultiply by the glyph's *hue*, not its raw RGB. `_rgba_shape_transform`
    # derives alpha as ``file_alpha * max(RGB)`` while leaving RGB untouched, so
    # on a glyph's support ``rgb.max(-1)`` equals ``alpha`` exactly: the
    # anti-alias ramp is present in both channels. Premultiplying by raw RGB and
    # then dividing back out (``tint = premult / depth`` in
    # :func:`generate_poisson_convsc`) recovers that ramp *as the foreground
    # colour*, so the anti-aliasing gets applied twice -- once as opacity and
    # once as colour. Measured on glyph 'A', edge pixels came out at
    # ``tint = 0.14`` against 0.996 at the core, i.e. the foreground was
    # 0.14 * color rather than color.
    #
    # Invisible for a single stamp (0.015 too light at the edge) but severe once
    # opacity accumulates: at four overlapping spikes the composite landed at
    # 0.610 where a correct alpha-composite gives 0.860 -- a dark fringe around
    # every glyph, and the converged model runs at a total rate near 4.
    #
    # Normalizing to unit hue puts the ramp in alpha alone, so
    # ``1 - exp(-tau)`` applies it exactly once. A white mask gives hue == 1 and
    # ``fg == color``; a genuinely coloured anti-aliased glyph gives its pure
    # chromaticity, with the ramp still entirely in alpha.
    peak = rgb.max(axis=-1, keepdims=True)
    hue = jnp.where(peak > 0., rgb / jnp.clip(peak, 1e-6, None), 1.)
    ink = jnp.concatenate((hue * alpha, alpha), axis=-1)  # (K, kh, kw, 4)
    return jnp.moveaxis(ink, 0, -1)


def _stamp(counts: Array, kernel: Array) -> Array:
    """Add ``counts[..., y, x, k]`` copies of glyph ``k`` *centred* at ``(y, x)``.

    ``counts`` is ``(..., H, W, K)`` at image resolution and the result is
    ``(..., H, W, C)``: the count field and the image it renders are the same
    grid, so a translation of one is a translation of the other. That is the
    equivariance ``poisson_hesc.py`` checks, and §1 of
    ``notes/poisson-convsc-design.md`` explains why the whole design rests on it.

    Rather than trusting a padding mode to place an even-sized kernel, this
    renders the full ``(H + kh - 1, W + kw - 1)`` support and crops explicitly.
    The convention: a unit spike at ``(y, x)`` lays the glyph's ``(kh, kw)``
    frame down with its top-left corner at ``(y - (kh-1)//2, x - (kw-1)//2)``,
    unflipped. For an odd kernel dimension the frame centre is exactly ``y``;
    for an even one -- as here, 38 x 26 -- the frame centre necessarily falls
    between pixels, half a pixel past ``y``, since a box with an even side has
    no centre pixel. Sub-pixel offsets (§6 of the design note) absorb that when
    they land; nothing here depends on it, because *equivariance* holds exactly
    either way and that is the property the design uses.
    """
    kh, kw = kernel.shape[:2]
    height, width = counts.shape[-3:-1]
    # conv_transpose wants exactly one batch dim; fold any particle/plate dims.
    leading = counts.shape[:-3]
    folded = counts.reshape((-1,) + counts.shape[-3:]).astype(kernel.dtype)
    full = jax.lax.conv_transpose(
        folded, kernel, (1, 1), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"), transpose_kernel=True,
    )
    top, left = (kh - 1) // 2, (kw - 1) // 2
    cropped = full[:, top:top + height, left:left + width]
    return cropped.reshape(leading + cropped.shape[1:])


class PoissonConvPlacements(nnx.Module):
    """Poisson convolutional sparse coding: integer counts stamped as opacity.

    The latent is an integer activation field at *image* resolution,

        a[k, y, x] ~ Poisson(rate),

    where ``(y, x)`` is the pixel the glyph's centre lands on. Stamping it
    through the dictionary gives the **optical depth**

        tau(p) = sum_{k,y,x} a[k,y,x] * alpha_k(p - (y,x)),

    a single transposed convolution, and the count enters the image through
    ``tau`` and nowhere else: more spikes at a site mean a more opaque stamp,
    saturating at ``1 - exp(-tau)``, never a different hue. Opacity is the one
    quantity where "counts add" is meaningful -- alpha composes by addition in
    log-transmittance, colour does not compose by addition at all.

    Non-negativity of ``tau`` is free (counts are non-negative, alpha lies in
    ``[0, 1]``), so no constraint is needed on the dictionary and glyphs cannot
    cancel each other. That is what a Poisson rate downstream would require, and
    it is why a learnable dictionary will need a non-negativity
    reparameterization.

    Unlike :class:`PoissonMarkedPlacements` there is no allocation simplex: an
    explicit count per (glyph, location) leaves nothing to allocate, so both
    Dirichlets -- and with them the unbounded sparse-simplex density and the
    2364-deep stick-breaking recorded in ``notes/minsum-session-2026-07-29.md``
    -- are gone. The firing rate is a plain learnable parameter rather than a
    latent: with a single dictionary layer there is no second level of features
    for a rate prior to be informative about, so being Bayesian about it would
    only add an unidentified scalar and a KL term competing with the likelihood
    over sparsity. That changes when a layer-1 rate field is predicted top-down.
    """

    def __init__(self, shape_dict: ShapeDictionary, expected_count: float=1.,
                 img_h: int=80, img_w: int=80, *,
                 rngs: Optional[nnx.Rngs]=None):
        del rngs
        self.expected_count = expected_count
        self.height = img_h
        self.shape_dict = shape_dict
        self.width = img_w

    @property
    def num_features(self) -> int:
        return self.shape_dict.shapes.shape[0]

    @property
    def num_sites(self) -> int:
        return self.height * self.width * self.num_features

    @property
    def ink_kernel(self) -> Array:
        return _ink_kernel(self.shape_dict.shapes)

    def sample_counts(self):
        """Draw the integer activation field, shape ``(1, H, W, K)``.

        ``expected_count`` only *initializes* the rate, which then learns
        freely. A scalar keeps the prior a homogeneous marked Poisson process
        and so exactly translation-invariant; ``(K,)`` would still be safe (it
        learns per-glyph frequency), but a per-site rate would destroy that
        invariance and with it the equivariance :func:`_stamp` provides.
        """
        log_rate = numpyro.param(
            "log_rate", jnp.log(self.expected_count / self.num_sites)
        )
        rate = jnp.broadcast_to(jnp.exp(log_rate),
                                (1, self.height, self.width, self.num_features))
        return numpyro.sample("a", dist.Poisson(rate).to_event(3))

    def ink_field(self, counts) -> Array:
        """Stamp the counts into an ``(..., H, W, 4)`` ink field.

        Channels 0:3 hold premultiplied colour and channel 3 the optical depth.
        This pair is the *only* interface the likelihood consumes, so warped or
        genuinely deformable renderers (§6 of the design note) can replace this
        method without anything downstream changing.
        """
        return jnp.clip(_stamp(counts, self.ink_kernel), 0., None)

    def __call__(self, rngs=None):
        del rngs
        return self.ink_field(self.sample_counts())


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
        # The z_rate shape is safely learnable: z_rate lives in the interior
        # of its support, where the Gamma density is bounded. The Dirichlet
        # concentrations must stay FIXED hyperparameters: adapted allocation
        # samples pin to the simplex boundary, where the sparse Dirichlet
        # density is unbounded in the concentration -- learning it drives
        # every concentration toward zero (KL -> -inf, then NaN).
        count_concentration = numpyro.param(
            "count_concentration", jnp.asarray(self.count_concentration),
            constraint=dist.constraints.positive
        )
        z_rate = numpyro.sample("z_rate", dist.Gamma(
            count_concentration, count_concentration / self.expected_count,
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
                             plot_mean=False, scale=None):
    # One Gaussian mixture component per dictionary glyph, plus a background
    # component (index 0); each carries its own learnable log-scale. Read the
    # glyph count from the raw module before wrapping, since the nnx_module
    # wrapper exposes only ``__call__``.
    n_components = placements.num_features + 1
    placements = nnx_module("placements_p", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_p", backgrounder)

    if scale is None:
        # Initialize sharp (sigma = 0.1, the regime where per-batch inference
        # demonstrably works) rather than at sigma = 1, which is so forgiving
        # that adaptation places no ink for the thousands of steps it takes
        # the learned scale to crawl down.
        scale = jnp.exp(numpyro.param("log_scale",
                                      jnp.log(0.1) * jnp.ones((1,))))
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
        # The coverage-weighted mixture mean is what the model "expects" the
        # image to look like. A single ``obs`` draw is a per-pixel categorical
        # over the background and the K glyph components, so it dithers even a
        # firmly placed glyph (a pixel with unit coverage still picks the
        # background with probability exp(-1) / (exp(-1) + 1)); plot this site
        # rather than a sample to inspect the geometry.
        mean = likelihood.mean
        if plot_mean:
            numpyro.deterministic("mean", mean)
        if images is not None:
            numpyro.deterministic("residual", (images - mean) ** 2)
        return numpyro.sample("obs", likelihood, obs=images)

def _ink_scale(opacity, schedule: str, sigma_bg: float, sigma_ink_init: float):
    """Per-pixel Gaussian scale as a function of opacity ``A``.

    Which *direction* ink should move the variance is an open empirical
    question, so the map is pluggable:

    ``endpoints``  ``(1 - A) sigma_bg^2 + A sigma_ink^2`` -- interpolates between
        two scales, so the optimizer decides whether ink buys slack or steepness
        rather than the config deciding in advance. Bounded at both ends. This is
        also exactly the form the two-component mixture produces on its own.
    ``affine``     ``sigma_bg^2 + k A``          -- slack where the ink is.
    ``edge``       ``sigma_bg^2 + k A (1 - A)``  -- slack only at partial coverage.
    ``inverse``    ``sigma_bg^2 + k / (A + eps)`` -- steepness where the ink is.
        Singular at ``A = 0`` and assigns its *largest* variance to blank paper;
        ``sigma_bg^2 + k (1 - A)`` is the bounded version of the same direction.

    ``sigma_bg`` is a fixed constant, never learnable. The captcha backgrounds
    are bit-identical to pure white in 95.3% of pixels, so a learnable
    background scale has an unbounded optimum at zero and the optimizer will
    find it.
    """
    variance_bg = sigma_bg ** 2
    if schedule == "endpoints":
        sigma_ink = jnp.exp(numpyro.param("log_sigma_ink",
                                          jnp.log(sigma_ink_init)))
        variance = (1. - opacity) * variance_bg + opacity * sigma_ink ** 2
    elif schedule in ("affine", "edge", "inverse"):
        coefficient = jnp.exp(numpyro.param(
            "log_ink_variance", jnp.log(sigma_ink_init ** 2)
        ))
        if schedule == "affine":
            variance = variance_bg + coefficient * opacity
        elif schedule == "edge":
            variance = variance_bg + coefficient * opacity * (1. - opacity)
        else:
            variance = variance_bg + coefficient / (opacity + 1e-3)
    else:
        raise ValueError(
            f"Unknown ink variance schedule {schedule!r}; expected one of "
            "'endpoints', 'affine', 'edge', 'inverse'."
        )
    return jnp.sqrt(variance)


def _observation_df(opacity, observation_df, learn_df: bool=False,
                    df_couples_to: str="opacity", depth=None):
    """Per-pixel Student-t degrees of freedom, or ``None`` for a Normal.

    ``observation_df`` accepts:

    ``None``
        Normal observations.
    a scalar
        a flat Student-t at that many degrees of freedom.
    a pair ``(nu_bg, nu_ink)``
        ``nu(A) = (1 - A) nu_bg + A nu_ink``, the same endpoint interpolation
        :func:`_ink_scale` uses for the scale.

    The pair form exists because ``nu`` is a statement about *uncertainty in the
    scale*, and ours is wildly asymmetric. Student-t is a scale mixture of
    normals, ``x | w ~ N(mu, sigma^2 / w)`` with ``w ~ Gamma(nu/2, nu/2)``, so
    ``nu`` is an inverse measure of how heterogeneous the per-pixel noise scale
    is believed to be -- equivalently, the conjugate Normal-Inverse-Gamma
    posterior predictive is a Student-t whose df is the pseudo-count behind the
    variance estimate. We effectively *know* ``sigma_bg`` (background residual is
    exactly zero; 0.01 was picked for optimisation comfort against a 0.0011
    quantization floor) and have never estimated ``sigma_ink`` at all. So heavy
    tails belong where the scale is unknown -- ink, anti-aliased edges,
    sub-pixel placement -- and near-Gaussian tails where it is known.

    Interpolating the *excess* over 2 rather than ``nu`` itself keeps
    ``nu > 2``, and with it a finite variance, by construction.

    Coupling to opacity rather than to the spike count is deliberate. A
    count-dependent ``nu`` would give the counts a second channel into the
    likelihood through the log-normalizer -- violating the invariant of §2 that
    counts reach the image only through optical depth -- and its incentive
    points the wrong way: since ``-log p ~ ((nu+1)/2) log(r^2 / nu sigma^2)``
    for a large residual, anything that raises ``nu`` raises the penalty, so a
    model free to lower ``nu`` by *removing spikes* will do exactly that.
    """
    if observation_df is None:
        return None
    # Validate on the Python values: observation_df is a static config quantity,
    # and comparing a traced array here would raise under jit.
    if isinstance(observation_df, (int, float)):
        raw = [float(observation_df)]
    else:
        raw = [float(v) for v in observation_df]
    if len(raw) not in (1, 2):
        raise ValueError("observation_df must be None, a scalar, or a "
                         f"(nu_bg, nu_ink) pair; got {observation_df!r}")
    if any(v <= 2. for v in raw):
        raise ValueError("observation_df entries must exceed 2 for the "
                         f"variance to be finite; got {observation_df!r}")
    excess = jnp.asarray([v - 2. for v in raw])
    if learn_df:
        # Fixed by default, on the same reasoning as sigma_bg: anything the
        # model can adjust to forgive its own errors, it eventually will.
        excess = jnp.exp(numpyro.param("log_df_excess", jnp.log(excess)))
    if excess.shape[0] == 1:
        return 2. + jnp.broadcast_to(excess[0], opacity.shape)
    if df_couples_to == "opacity":
        return 2. + (1. - opacity) * excess[0] + opacity * excess[1]
    if df_couples_to == "depth":
        # nu = nu_0 + kappa * tau, the pair read as (nu_0, kappa). Unlike the
        # opacity form this does not saturate: a pixel under many overlapping
        # stamps keeps gaining degrees of freedom, which is the literal reading
        # of "df counts the observations behind this pixel".
        if depth is None:
            raise ValueError("df_couples_to='depth' needs the optical depth")
        return 2. + excess[0] + excess[1] * depth
    raise ValueError(f"df_couples_to must be 'opacity' or 'depth', got "
                     f"{df_couples_to!r}")


def generate_poisson_convsc(placements, backgrounder: Optional[BackgroundDecoder]=None,
                            ambient_depth: float=1e-4):
    """Composite the ink field over the background into ``(foreground, background, opacity)``.

    Returns per-pixel foreground colour, background colour and opacity
    ``A = 1 - exp(-tau)``, all ``(..., H, W, ·)``. Both likelihoods in
    :func:`poisson_convsc_model` are built from exactly this triple.

    ``ambient_depth`` is a small optical-depth floor added before computing the
    opacity -- the same device as the ``EPS`` added to ``ahat0`` in
    ``poisson_hesc.py``. It matters more than its size suggests, for two
    reasons.

    Numerically, it keeps the mixture weight on the foreground component
    strictly positive. ``MixtureSameFamily.log_prob`` stabilises by shifting
    with ``m = max_k log p_k(x)`` taken over *all* components, including
    zero-weight ones; at an inkless pixel whose data looks like ink, the
    zero-weight foreground attains that max and the weighted background term
    underflows to zero, giving ``log 0 = -inf``. A positive floor removes the
    ``-inf`` that ``notes/minsum-session-2026-07-29.md`` Finding 3 records as
    fatal to the objective.

    Modelling-wise, it caps the cost of an unexplained ink pixel at roughly
    ``-log(ambient_depth)`` nats rather than letting the confident background
    component charge a quadratic ``(x - 1)^2 / 2 sigma_bg^2``. The gradient in
    the opacity is then ``~1/A``, which is large exactly where ink is missing:
    a strong, bounded signal to place a glyph rather than an enormous one.
    """
    # Beta(1, 1) is the uniform density on [0, 1], but unlike Uniform its
    # parameters carry plain positive constraints and its support is fixed, so a
    # mean-field guide can mirror it without proposing colours off-support.
    color = numpyro.sample("color", dist.Beta(jnp.ones((3,)),
                                              jnp.ones((3,))).to_event(1))
    color = color[..., jnp.newaxis, jnp.newaxis, :]

    ink = placements()
    depth = ink[..., 3:]
    # Premultiplied colour divided by optical depth is the depth-weighted mean
    # ink colour. A Poisson process has no depth ordering -- its points are
    # exchangeable -- so a weighted average is the right answer here, not an
    # ordered "over". With a white dictionary this is identically 1.
    #
    # Where there is no ink the ratio is 0/0 and the mean ink colour is simply
    # undefined; fall back to 1, so the foreground component reads "if this
    # pixel were ink, it would be this image's ink colour". Falling back to 0
    # (black) instead would make the foreground hypothesis wrong in a way that
    # depends on the glyph dictionary rather than on the image.
    tint = jnp.where(depth > 1e-6, ink[..., :3] / jnp.clip(depth, 1e-6, None),
                     1.)
    foreground = tint * color

    if backgrounder is not None:
        background = backgrounder()
    else:
        background = jnp.ones(depth.shape[:-1] + (1,))
    background = jnp.broadcast_to(background, foreground.shape)

    # Beer-Lambert: 1 - exp(-tau) is the probability that at least one glyph
    # covers the pixel, i.e. one minus the Poisson void probability. It is the
    # only channel through which the counts reach the image.
    opacity = -jnp.expm1(-(depth + ambient_depth))
    return foreground, background, opacity, depth


def poisson_convsc_model(images, placements: PoissonConvPlacements,
                         backgrounder: Optional[BackgroundDecoder]=None,
                         ambient_depth: float=1e-4,
                         df_couples_to: str="opacity",
                         learn_df: bool=False,
                         likelihood: str="blend",
                         observation_df=None,
                         plot_mean: bool=False,
                         sigma_bg: float=0.01, sigma_ink_init: float=0.04,
                         variance_schedule: str="endpoints"):
    """Poisson convolutional sparse coding over the captcha dictionary.

    Two likelihoods, both reading only the ink field and both with the same
    mean, selected by ``likelihood``:

    ``"mixture"``
        A two-component per-pixel mixture, background and foreground, with
        weights ``(1 - A, A)``. Those already sum to one -- ``1 - A = exp(-tau)``
        *is* the void probability -- so unlike the 37-component version in
        :func:`marionette_captcha_model` they need no renormalization.
    ``"blend"`` (default)
        One composited layer, ``A * fg + (1 - A) * bg``, whose per-pixel scale
        comes from :func:`_ink_scale`.

    **The blend is the default because the mixture cannot represent this data.**
    The mixture's gradient properties are genuinely better -- its derivative in
    ``A`` is ``r_fg / A - r_bg / (1 - A)``, a ratio of posterior
    responsibilities that survives vanishing foreground/background contrast
    where the blend's ``(x - mean)(fg - bg) / sigma^2`` does not -- but a
    two-component mixture places no probability mass *between* its components,
    and 95.1% of this dataset's ink pixels are intermediate, anti-aliased
    values. A half-covered pixel scores ``log p = -116`` under the mixture
    against ``+7.9`` under the blend, so the ELBO correctly concludes that
    placing a glyph is harmful. Use ``"mixture"`` only for near-binary coverage.

    ``observation_df`` selects the tail -- ``None`` for a Normal, a scalar for a
    flat Student-t, or a ``(nu_bg, nu_ink)`` pair to interpolate the degrees of
    freedom with opacity exactly as the scale is interpolated (see
    :func:`_observation_df`). ``learn_df`` makes the endpoints learnable.
    Either way the scale is rescaled so it still means a standard deviation and
    only the tail changes. This matters
    because the Normal's penalty is quadratic and unbounded: with
    ``sigma_bg = 0.01`` a blank render costs on the order of ``1e6`` nats, which
    makes the ELBO optimise *miss-avoidance* rather than render accuracy and
    pushes the firing rate above where reconstruction is best. A Student-t's
    penalty grows logarithmically in the residual, so a missed glyph is
    expensive without being catastrophic.
    """
    placements = nnx_module("placements_p", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_p", backgrounder)

    batch_size = images.shape[0] if images is not None else 1
    with numpyro.plate("batch", batch_size):
        foreground, background, opacity, depth = generate_poisson_convsc(
            placements, backgrounder, ambient_depth
        )
        if likelihood == "mixture":
            sigma_ink = jnp.exp(numpyro.param("log_sigma_ink",
                                              jnp.log(sigma_ink_init)))
            scales = jnp.stack((jnp.asarray(sigma_bg), sigma_ink))[:, jnp.newaxis]
            weights = jnp.concatenate((1. - opacity, opacity), axis=-1)
            means = jnp.stack((background, foreground), axis=-2)
            observation = SpatialMixtureSameFamily(
                dist.Categorical(probs=weights),             # batch (B, H, W)
                dist.Normal(means, scales).to_event(1),      # batch (B, H, W, 2)
                reinterpreted_batch_ndims=2,                 # fold (H, W)
            )
        elif likelihood == "blend":
            mean = opacity * foreground + (1. - opacity) * background
            scale = _ink_scale(opacity, variance_schedule, sigma_bg,
                               sigma_ink_init)
            df = _observation_df(opacity, observation_df, learn_df,
                                 df_couples_to, depth)
            if df is None:
                observation = dist.Normal(mean, scale).to_event(3)
            else:
                # StudentT's variance is scale^2 * df/(df - 2), so rescale to
                # keep `scale` meaning a standard deviation: the whole
                # ink-dependent variance schedule then transfers unchanged and
                # only the *tail* differs. (The cost is that a heavier tail
                # narrows the core at fixed variance; dropping this factor would
                # instead preserve the core and inflate the variance.)
                observation = dist.StudentT(
                    df, mean, scale * jnp.sqrt((df - 2.) / df)
                ).to_event(3)
        else:
            raise ValueError(f"Unknown likelihood {likelihood!r}; expected "
                             "'mixture' or 'blend'.")

        mean = observation.mean
        if plot_mean:
            numpyro.deterministic("mean", mean)
        if images is not None:
            numpyro.deterministic("residual", (images - mean) ** 2)
        return numpyro.sample("obs", observation, obs=images)
