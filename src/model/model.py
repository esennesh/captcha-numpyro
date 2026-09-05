import math

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from flax import nnx
from jaxtyping import Array
from numpyro.contrib.module import nnx_module

from src.data.dictionary import ShapeDictionary
from src.distributions import (
    CompleteGaussianMrf,
    LocallyScaledGaussianMrf,
    SecondOrderGaussianMrf,
    SpatialMixtureSameFamily,
)
from src.distributions.layers import (
    POTENTIAL_EDGE_ALPHA,
    AlphaFormat,
    Layer,
)
from src.model.diffeomorphism import (
    affine_free_velocity,
    boundary_taper,
    diffeomorphic_warp,
    translation_free_velocity,
)

# Which image modes ``TexturedDiffeomorphicPoissonConvPlacements`` conditions
# out of its velocity field. See the ``warp_modes`` argument.
_WARP_CONDITIONERS = {
    "affine": affine_free_velocity,
    "translation": translation_free_velocity,
}


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


def _dictionary_support_masks(shapes: Array, threshold: float = 0.0) -> Array:
    """Per-glyph 0/1 support masks, shaped ``(K, kh, kw, 1)``.

    A glyph "touches" a pixel wherever its alpha support exceeds ``threshold``.
    The alpha source comes from the explicit alpha channel for RGBA glyphs, the
    sole channel for single-channel glyphs, and the per-pixel channel max as a
    brightness proxy for RGB glyphs.
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


def _ink_kernel(shapes: Array, color_field: Array | None = None) -> Array:
    """The stamping kernel, shaped ``(kh, kw, 4, K)`` for ``conv_transpose``.

    Channels 0:3 carry *premultiplied* colour ``alpha_k * rgb_k`` and channel 3
    carries ``alpha_k`` alone, so a single transposed convolution of the count
    field emits both the colour numerator and the optical depth that normalizes
    it (see :meth:`PoissonConvPlacements.ink_field`). An optional canonical RGB
    multiplier can be batched and is shared across every dictionary entry in
    each batch member.

    The layout is HWOI as ``conv_transpose(..., transpose_kernel=True)`` wants
    it: ``O`` is the deconvolution's output channels (the 4 ink channels) and
    ``I`` its input channels (the ``K`` dictionary features).
    """
    alpha, rgb = _dictionary_alpha_rgb(shapes)
    # Normalizing here takes us out of premultiplied alpha and gives us just the
    # hue of the nonzero pixels, while preserving the information necessary for
    # anti-aliasing edges and such later.
    peak = rgb.max(axis=-1, keepdims=True)
    hue = jnp.where(peak > 0., rgb / jnp.clip(peak, 1e-6, None), 1.)

    # We calculate the negative-log transmittance tau = -log(1 - α), because
    # convolving will add it in the stamping process to recover alpha
    # compositing when we combine the stamped layers. One stamp gives
    # A = 1 - (1 - α)^1 = α; n stamps give 1 - (1 - α)^n; and α = 1 gives A = 1.
    # This lets increasing spike-counts bring a glyph "closer" to the camera
    # without over-opacifying anti-aliased shape edges. We clip α into
    # [0., 0.999] for numerical stability at large values.
    tau = -jnp.log1p(-jnp.clip(alpha, 0., 1. - 1e-3))
    ink = jnp.concatenate((hue * tau, tau), axis=-1)  # (K, kh, kw, 4)
    ink = jnp.moveaxis(ink, 0, -1)
    if color_field is None:
        return ink

    color_field = jnp.asarray(color_field)
    if color_field.ndim < 3 or color_field.shape[-3:] != (
        ink.shape[0], ink.shape[1], 3
    ):
        raise ValueError(
            "color_field needs trailing shape (kh, kw, 3), got "
            f"{color_field.shape} for kernel shape {ink.shape[:2]}"
        )
    colored = ink[..., :3, :] * color_field[..., jnp.newaxis]
    depth = jnp.broadcast_to(
        ink[..., 3:4, :], color_field.shape[:-3] + ink[..., 3:4, :].shape
    )
    return jnp.concatenate((colored, depth), axis=-2)


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
    if kernel.ndim < 4:
        raise ValueError(
            "kernel needs trailing shape (kh, kw, C, K), "
            f"got {kernel.shape}"
        )
    if kernel.ndim > 4:
        counts_leading = counts.shape[:-3]
        kernel_leading = kernel.shape[:-4]
        leading = jax.lax.broadcast_shapes(counts_leading, kernel_leading)
        count_shape = counts.shape[-3:]
        kernel_shape = kernel.shape[-4:]
        total = max(1, math.prod(leading))
        counts = jnp.broadcast_to(counts, leading + count_shape).reshape(
            (total,) + count_shape
        )
        kernel = jnp.broadcast_to(kernel, leading + kernel_shape).reshape(
            (total,) + kernel_shape
        )
        stamped = jax.vmap(_stamp)(counts, kernel)
        return stamped.reshape(leading + stamped.shape[-3:])

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
    """

    def __init__(self, shape_dict: ShapeDictionary, expected_count: float=1.,
                 img_h: int=80, img_w: int=80, *,
                 rngs: nnx.Rngs | None=None):
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

    def ink_field(self, counts, color_field: Array | None = None) -> Array:
        """Stamp the counts into an ``(..., H, W, 4)`` ink field.

        Channels 0:3 hold premultiplied colour and channel 3 the optical depth.
        This pair is the *only* interface the likelihood consumes, so warped or
        genuinely deformable renderers (§6 of the design note) can replace this
        method without anything downstream changing.
        """
        kernel = (
            self.ink_kernel
            if color_field is None
            else _ink_kernel(self.shape_dict.shapes, color_field)
        )
        return jnp.clip(_stamp(counts, kernel), 0., None)

    def __call__(self, color: Array | None = None, rngs=None):
        del color
        del rngs
        return self.ink_field(self.sample_counts())


class TexturedDiffeomorphicPoissonConvPlacements(PoissonConvPlacements):
    r"""Stamp one shared canonical texture and warp the complete foreground.

    For an image-level baseline colour ``c_0`` and canonical texture ``r``,

    .. math::

        r &\sim \mathcal N(0,Q_{\mathrm{tex}}^{-1}\otimes I_3),\\
        c(u) &= \operatorname{sigmoid}(\operatorname{logit}c_0+r(u)).

    The renderer stamps the baseline-relative multiplier

    .. math::

        m(u)=\frac{c(u)}{c_0}
            =\frac{\exp r(u)}{1+c_0(\exp r(u)-1)}

    through every dictionary kernel and retains the former final multiplication
    by ``c_0``. Thus there is one texture field per image, not one per glyph
    class or occurrence, and ``r=0`` recovers the old renderer exactly. The
    four resulting ink channels are then pulled back through one whole-image
    diffeomorphism:

    .. math::

        u &\sim \mathcal N(0,Q_{\mathrm{warp}}^{-1}\otimes I_2),\\
        v &= WR\,\operatorname{ConditionAffineFree}(u),\\
        \phi &= \exp(v),\\
        I(p) &= I_0(\phi^{-1}(p)).

    Colour numerators and optical depth are warped together.  Background
    compositing remains downstream, so the deformation moves foreground ink
    only.  ``u`` is the normalized latent; conditioning, resizing, and
    exponentiation are deterministic transformations and require no additional
    Jacobian factor in the joint density.

    ``ConditionTranslationFree`` is the default conditioning: translation is
    redundant with the count field's site index, but rotation, scale, and shear
    have no other latent in this renderer, so conditioning them out removes
    them from the model. ``warp_modes="affine"`` restores the former
    :func:`~src.model.diffeomorphism.affine_free_velocity` behaviour; see its
    docstring for the measured cost.
    """

    def __init__(
        self,
        shape_dict: ShapeDictionary,
        expected_count: float = 1.0,
        img_h: int = 80,
        img_w: int = 80,
        *,
        cg_iters: int = 300,
        rngs: nnx.Rngs | None = None,
        texture_bond_precision: float = 4.0,
        texture_element_precision: float = 1.0,
        warp_bond_precision: float = 0.4,
        warp_coarse_height: int = 10,
        warp_coarse_width: int = 10,
        warp_element_precision: float = 0.1,
        warp_modes: str = "translation",
        warp_scale: float = 5.0,
        warp_squaring_steps: int = 7,
    ):
        super().__init__(
            shape_dict,
            expected_count,
            img_h,
            img_w,
            rngs=rngs,
        )
        if warp_modes not in _WARP_CONDITIONERS:
            raise ValueError(
                f"Unknown warp_modes {warp_modes!r}; expected one of "
                f"{sorted(_WARP_CONDITIONERS)}."
            )
        self.cg_iters = cg_iters
        self.texture_bond_precision = texture_bond_precision
        self.texture_element_precision = texture_element_precision
        self.warp_bond_precision = warp_bond_precision
        self.warp_coarse_height = warp_coarse_height
        self.warp_coarse_width = warp_coarse_width
        self.warp_element_precision = warp_element_precision
        self.warp_modes = warp_modes
        self.warp_scale = warp_scale
        self.warp_squaring_steps = warp_squaring_steps

    def __call__(self, color: Array | None = None, rngs=None):
        if color is None:
            raise ValueError(
                "TexturedDiffeomorphicPoissonConvPlacements needs an RGB "
                "baseline color"
            )
        del rngs
        counts = self.sample_counts()
        modulation = self.color_modulation(color)
        return self.warp_ink(self.ink_field(counts, modulation))

    def color_modulation(self, color: Array) -> Array:
        """Sample canonical texture and return its baseline-relative colour."""
        texture = numpyro.sample("color_texture", self.texture_prior())
        baseline = color[..., jnp.newaxis, jnp.newaxis, :]
        exponential = jnp.exp(texture)
        return exponential / (
            1.0 + baseline * jnp.expm1(texture)
        )

    def texture_prior(self) -> SecondOrderGaussianMrf:
        """Return the normalized proper prior over canonical RGB texture."""
        height, width = self.ink_kernel.shape[:2]
        return SecondOrderGaussianMrf(
            jnp.zeros((height, width, 3), dtype=self.ink_kernel.dtype),
            self.texture_element_precision
            * jnp.ones((height, width), dtype=self.ink_kernel.dtype),
            jnp.asarray(
                self.texture_bond_precision, dtype=self.ink_kernel.dtype
            ),
            cg_iters=self.cg_iters,
        )

    def velocity_prior(self) -> SecondOrderGaussianMrf:
        """Return the normalized proper prior over coarse image velocity."""
        shape = (self.warp_coarse_height, self.warp_coarse_width)
        return SecondOrderGaussianMrf(
            jnp.zeros((*shape, 2), dtype=self.ink_kernel.dtype),
            self.warp_element_precision
            * jnp.ones(shape, dtype=self.ink_kernel.dtype),
            jnp.asarray(self.warp_bond_precision, dtype=self.ink_kernel.dtype),
            cg_iters=self.cg_iters,
        )

    def warp_ink(self, ink: Array) -> Array:
        """Sample one affine-free image flow and pull back foreground ink."""
        prior = self.velocity_prior()
        raw_velocity = numpyro.sample("warp_velocity", prior)
        leading = jax.lax.broadcast_shapes(
            ink.shape[:-3], raw_velocity.shape[:-3]
        )
        coarse_shape = raw_velocity.shape[-3:]
        ink_shape = ink.shape[-3:]
        total = max(1, math.prod(leading))
        inks = jnp.broadcast_to(ink, leading + ink_shape).reshape(
            (total,) + ink_shape
        )
        raw_velocities = jnp.broadcast_to(
            raw_velocity, leading + coarse_shape
        ).reshape((total,) + coarse_shape)
        window = boundary_taper(
            (self.height, self.width), dtype=ink.dtype
        )
        condition = _WARP_CONDITIONERS[self.warp_modes]

        def warp_one(image, raw):
            velocity = self.warp_scale * condition(
                raw,
                prior.solve_precision,
                (self.height, self.width),
                window=window,
            )
            return diffeomorphic_warp(
                image,
                velocity,
                squaring_steps=self.warp_squaring_steps,
            )

        warped = jax.vmap(warp_one)(inks, raw_velocities)
        return warped.reshape(leading + ink_shape)


class PaperGmrfBackground(nnx.Module):
    r"""A coarse Gaussian random field of paper colour, in logit space.

    .. math::

        m_b &\sim \mathcal N(\operatorname{logit}b_0,\sigma_b^2 I_C),\\
        f &\sim \mathcal N(0,Q_{\mathrm{paper}}^{-1}\otimes I_C),\\
        b(p) &= \operatorname{sigmoid}(m_b+(Rf)(p)),

    where ``R`` resizes the coarse lattice to the image. The global logit
    ``m_b`` carries the overall paper colour and the field carries smooth
    variation across the sheet, exactly the split
    :class:`TexturedDiffeomorphicPoissonConvPlacements` uses for ink: pushing
    an overall shift through the field would charge one element-precision
    penalty per lattice site, while a ``C``-vector can shift it directly.

    This exists because a *fixed* white paper is not a harmless simplification.
    The captcha paper in ``data/examples`` departs from white by 0.04 to 0.07
    per channel, which at ``sigma_bg = 0.01`` is four to seven standard
    deviations at each of 6,400 pixels. The cheapest escape the count field has
    is to tile the canvas with glyphs and repaint the paper, and it is worth
    tens of thousands of nats: on ``0000_LJ`` the model's own log joint scores
    49 stamps of tiling at -21,091 against -69,192 for the correct two glyphs.
    Sampling the paper removes that incentive rather than penalizing it.

    The resize is deterministic, so the joint density needs no extra Jacobian
    factor; ``f`` carries the normalized :class:`CompleteGaussianMrf` density on
    the coarse lattice.
    """

    def __init__(
        self,
        img_h: int = 80,
        img_w: int = 80,
        channels: int = 3,
        *,
        bond_precision: float = 1.0,
        cg_iters: int = 300,
        coarse_height: int = 8,
        coarse_width: int = 8,
        element_precision: float = 0.25,
        logit_scale: float = 0.35,
        paper_color: tuple[float, ...] = (0.94, 0.95, 0.98),
        resize_method: str = "cubic",
        rngs: nnx.Rngs | None = None,
    ):
        del rngs
        paper_color = jnp.asarray(paper_color)
        if paper_color.shape != (channels,):
            raise ValueError(
                f"paper_color needs {channels} entries, got {paper_color.shape}"
            )
        self.bond_precision = bond_precision
        self.cg_iters = cg_iters
        self.channels = channels
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        self.element_precision = element_precision
        self.height = img_h
        self.logit_loc = jnp.log(paper_color) - jnp.log1p(-paper_color)
        self.logit_scale = logit_scale
        self.resize_method = resize_method
        self.width = img_w

    def field_prior(self):
        """Return the normalized proper prior over the coarse paper field."""
        shape = (self.coarse_height, self.coarse_width)
        dtype = self.logit_loc.dtype
        return CompleteGaussianMrf(
            jnp.zeros((*shape, self.channels), dtype=dtype),
            self.element_precision * jnp.ones(shape, dtype=dtype),
            jnp.asarray(self.bond_precision, dtype=dtype),
            cg_iters=self.cg_iters,
        )

    def resize_field(self, field: Array) -> Array:
        """Resize ``(..., h, w, C)`` to image resolution, folding batch dims."""
        leading = field.shape[:-3]
        total = max(1, math.prod(leading))
        folded = field.reshape((total,) + field.shape[-3:])
        resized = jax.vmap(
            lambda one: jax.image.resize(
                one,
                (self.height, self.width, self.channels),
                method=self.resize_method,
            )
        )(folded)
        return resized.reshape(leading + resized.shape[-3:])

    def __call__(self, rngs=None):
        del rngs
        logit = numpyro.sample(
            "paper_logit",
            dist.Normal(self.logit_loc, self.logit_scale).to_event(1),
        )
        field = numpyro.sample("paper_field", self.field_prior())
        return jax.nn.sigmoid(
            logit[..., jnp.newaxis, jnp.newaxis, :] + self.resize_field(field)
        )


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

    The pair form was motivated by ``nu`` as a statement about *uncertainty in
    the scale*: Student-t is a scale mixture of normals,
    ``x | w ~ N(mu, sigma^2 / w)`` with ``w ~ Gamma(nu/2, nu/2)``, so ``nu``
    measures how heterogeneous the per-pixel noise scale is believed to be. Since
    ``sigma_bg`` is effectively known and ``sigma_ink`` was never estimated, heavy
    tails looked like they belonged on ink.

    **That argument is wrong and §20 measured it wrong.** The catastrophic
    residuals are not ink pixels whose scale is poorly known; they are pixels
    where the model placed *no* ink and the data has ink. Those have ``A ~ 0``, so
    a schedule with heavy tails on ink hands them the *light* tail -- the
    forgiving tail lands exactly where the outliers are not. §21 reversed it:
    ``nu`` should **rise** with coverage, fat-tailed on blank paper and
    near-Normal on opaque ink, which is what ``[3.0, 10.0]`` now expresses.

    Interpolating the *excess* over 2 rather than ``nu`` itself keeps
    ``nu > 2``, and with it a finite variance, by construction. Note this applies
    to *both* entries, so a ``depth`` pair reads as ``(nu_0, kappa + 2)``:
    ``[2.5, 4.0]`` is ``nu = 2.5 + 2.0 tau``.

    An earlier version of this docstring argued that coupling to the spike count
    was unsound, on the grounds that ``-log p ~ ((nu+1)/2) log(r^2/nu sigma^2)``
    at large residual, so raising ``nu`` raises the penalty and a model free to
    lower ``nu`` by removing spikes would do so. The premise is the wrong limit:
    95% of these pixels sit at *near-zero* residual, where the Normal beats the t
    (-3.2 against -2.8 nats at ``r = 0.01``), so a higher ``nu`` is a *reward*.
    The concern about counts reaching the likelihood through the log-normalizer as
    well as through ``tau`` is real, though, and it is one reason §24 prefers the
    opacity coupling -- ``A`` is a bounded function of ``tau``, so the second
    channel saturates instead of growing without limit.
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


def generate_poisson_convsc(placements, backgrounder: BackgroundDecoder | None=None,
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
    ink = placements(color)
    depth = ink[..., 3:]
    # Premultiplied colour divided by optical depth is the depth-weighted mean
    # ink colour. A Poisson process has no depth ordering -- its points are
    # exchangeable -- so a weighted average is the right answer here, not an
    # ordered "over". With a white dictionary this is identically 1.
    #
    # Where there is no ink the ratio is 0/0 and the mean ink colour is simply
    # undefined; fall back to the image-level baseline, so the foreground
    # component still reads "if this pixel were ink, it would have this image's
    # ink colour". Falling back to black would make the hypothesis wrong in a
    # way that depends on the glyph dictionary rather than on the image.
    fallback = jnp.ones_like(color[..., jnp.newaxis, jnp.newaxis, :])
    tint = jnp.where(
        depth > 1e-6,
        ink[..., :3] / jnp.clip(depth, 1e-6, None),
        fallback,
    )
    foreground = tint * color[..., jnp.newaxis, jnp.newaxis, :]

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


def _ink_layer(ink: Array, color: Array, ambient_depth: float = 1e-4) -> Layer:
    """Wrap a stamped ink field as a foreground :class:`Layer`.

    ``ink`` is the ``(..., H, W, 4)`` pair :meth:`PoissonConvPlacements.ink_field`
    emits: premultiplied colour in channels 0:3 and optical depth in channel 3.
    The layer carries straight-alpha coverage ``A = 1 - exp(-tau)``, the
    premultiplied image ``A c``, and the integer region count that
    :class:`~src.distributions.spatial.PartitionedGaussianMrf` needs.

    The count is ``1`` wherever coverage reaches
    :data:`~src.distributions.layers.POTENTIAL_EDGE_ALPHA` and ``0`` elsewhere,
    following :mod:`src.distributions.layers`: an intermediate alpha is an
    anti-aliased edge, so the median coverage is where a region boundary
    belongs. It is a threshold, hence flat in the counts, which is why the
    Gamma degrees of freedom this feeds are a *labelling* rather than another
    channel for the count field to inflate.

    This deliberately repeats the tint algebra of
    :func:`generate_poisson_convsc` instead of sharing it: that function's
    inkless fallback and opacity floor are calibrated against the ``blend``
    likelihood (§§4, 5 and 17 of the design note) and changing them to suit the
    layer algebra would silently re-tune it.
    """
    depth = ink[..., 3:]
    coverage = -jnp.expm1(-(depth + ambient_depth))
    fallback = jnp.ones_like(color[..., jnp.newaxis, jnp.newaxis, :])
    tint = jnp.where(
        depth > 1e-6,
        ink[..., :3] / jnp.clip(depth, 1e-6, None),
        fallback,
    )
    foreground = tint * color[..., jnp.newaxis, jnp.newaxis, :]
    return Layer(
        count=jnp.where(
            coverage[..., 0] >= POTENTIAL_EDGE_ALPHA, 1, 0
        ).astype(jnp.int32),
        coverage=coverage[..., 0],
        format=AlphaFormat.PREMULTIPLIED,
        image=coverage * foreground,
    )


def composite_poisson_convsc(placements, backgrounder=None,
                             ambient_depth: float=1e-4) -> Layer:
    """Composite the ink layer *over* the paper layer and return one Layer.

    This is the two-field counterpart of :func:`generate_poisson_convsc`: the
    glyph field and the paper field are each their own random field, and
    ``over`` -- :meth:`src.distributions.layers.Layer.__matmul__` -- combines
    them. Because the paper covers the sheet, the composite's coverage is one
    everywhere and its image is exactly ``A c + (1 - A) b``. Its *count* is not
    constant: it is one under ink and zero on bare paper, which is the
    labelling the partitioned GMRF cuts its lattice bonds along and the
    labelling the Gamma degrees of freedom read.

    ``backgrounder=None`` falls back to fixed white paper, which reproduces the
    former mean. See :class:`PaperGmrfBackground` for why that fallback costs
    tens of thousands of nats on real captcha paper.
    """
    color = numpyro.sample("color", dist.Beta(jnp.ones((3,)),
                                              jnp.ones((3,))).to_event(1))
    glyph = _ink_layer(placements(color), color, ambient_depth)
    if backgrounder is not None:
        background = backgrounder()
    else:
        background = jnp.ones(glyph.image.shape[:-1] + (1,))
    background = jnp.broadcast_to(background, glyph.image.shape)
    return glyph @ Layer.straight(background)


def poisson_convsc_model(images, placements: PoissonConvPlacements,
                         backgrounder: BackgroundDecoder | None=None,
                         ambient_depth: float=1e-4,
                         cg_iters: int=300,
                         df_couples_to: str="opacity",
                         gmrf_bond_precision: float=100.,
                         gmrf_element_precision: float=50.,
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
        *is* the void probability -- they need no renormalization.
    ``"blend"`` (default)
        One composited layer, ``A * fg + (1 - A) * bg``, whose per-pixel scale
        comes from :func:`_ink_scale`.
    ``"gmrf"``
        A Gamma-Normal partitioned GMRF over the whole image, from
        :func:`composite_poisson_convsc` and
        :class:`~src.distributions.spatial.LocallyScaledGaussianMrf`. The mean
        is the same ``over`` composite; the residual is a Gaussian field whose
        lattice bonds are cut wherever the region count changes, scaled pixel by
        pixel by a latent precision ``lambda_p ~ Gamma(nu_p/2, nu_p/2)`` with
        ``nu_p = count_p + 1``. Bare paper therefore gets ``nu = 1``, a very
        dispersed precision that can fall to ~0.03 and absorb speckles and
        distractor strokes the renderer does not represent, while ink gets
        ``nu = 2`` and is held closer to the render. ``sigma_bg``,
        ``sigma_ink_init``, ``variance_schedule``, ``observation_df``,
        ``df_couples_to`` and ``learn_df`` are all unused in this branch, which
        gets its tails from the Gamma mixing variable instead. See sections 3
        and 4 of ``notebooks/partitioned_gmrf.ipynb``.

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
        if likelihood == "gmrf":
            composite = composite_poisson_convsc(
                placements, backgrounder, ambient_depth
            )
            # nu = 1 on bare paper, 2 under ink. The Gamma is its own latent,
            # so a pixel the renderer cannot explain is downweighted rather
            # than dragging the paper field or the count field towards it.
            degrees_of_freedom = composite.count.astype(
                composite.image.dtype
            ) + 1.
            local_precision = numpyro.sample(
                "local_precision",
                dist.Gamma(
                    0.5 * degrees_of_freedom, 0.5 * degrees_of_freedom
                ).to_event(2),
            )
            observation = LocallyScaledGaussianMrf(
                composite,
                gmrf_element_precision * jnp.ones(
                    composite.image.shape[:-1], dtype=composite.image.dtype
                ),
                jnp.asarray(gmrf_bond_precision, dtype=composite.image.dtype),
                local_precision,
                cg_iters=cg_iters,
            )
            mean = observation.mean
            if plot_mean:
                numpyro.deterministic("mean", mean)
            if images is not None:
                numpyro.deterministic("residual", (images - mean) ** 2)
            return numpyro.sample("obs", observation, obs=images)

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
                             "'blend', 'gmrf' or 'mixture'.")

        mean = observation.mean
        if plot_mean:
            numpyro.deterministic("mean", mean)
        if images is not None:
            numpyro.deterministic("residual", (images - mean) ** 2)
        return numpyro.sample("obs", observation, obs=images)
