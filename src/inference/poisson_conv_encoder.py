"""Convolutional proposal for the Poisson convolutional sparse coding model.

``poisson_convsc_model`` puts an integer activation field ``a[k, y, x]`` at
*image* resolution, so the proposal over it is an ordinary image-to-image
network: a stack of dilated convolutions from ``(B, H, W, 3)`` to
``(B, H, W, K)`` log-rates, plus a head for the global ink colour.

Both heads read the *image* for their location and the backbone only for a
residual -- the rate head through a matched filter, the colour head through a
coverage-weighted average. That is not a stylistic choice in either case: see
:class:`PoissonRateHead` on the cold start and :class:`InkColorFinder` on the
colour signal the normalized feature stack does not carry.

That is the payoff of anchoring the generative model at image resolution. The
MarioNette guide in :mod:`src.inference.captcha_encoder` has to spend a
paragraph justifying a VALID ``(kh, kw)`` convolution chosen so its head lands
on exactly the ``(H - kh)//stride + 1`` anchor grid, "so each per-anchor
posterior parameter sees exactly one glyph's worth of input pixels and aligns
spatially with the corresponding prior cell". Here the proposal grid *is* the
image grid, so there is no alignment argument to make and no geometry to keep
in sync with the generative side.

Two properties the architecture has to preserve, both from §7 of
``notes/poisson-convsc-design.md``:

* **Receptive field at least the glyph frame.** Deciding whether glyph ``k`` is
  centred at pixel ``p`` requires seeing all of ``p``'s ``38 x 26``
  neighbourhood. The default dilation ladder ``(1, 2, 4, 8, 16)`` of ``3 x 3``
  convolutions gives ``1 + 2*(1+2+4+8+16) = 63`` in each axis, comfortably
  clear of it. A shallow local stack would still train, but it would learn
  rates that respond to strokes rather than to whole characters -- a failure
  that is easy to miss, hence :attr:`PoissonConvBackbone.receptive_field`.
* **Fully convolutional.** Then ``q`` is translation-equivariant exactly as the
  prior is: translating the image translates the proposed rate field.

``dist.Poisson`` has ``has_rsample = False``, so ``ELBOTracer`` routes the ``a``
site through its score-function surrogate with a leave-one-out baseline. No
relaxation is involved anywhere in this guide.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from flax import nnx
from jaxtyping import Array, Float
from numpyro.contrib.module import nnx_module

from src.data.dictionary import ShapeDictionary
from src.distributions import SecondOrderGaussianMrf
from src.inference.captcha_encoder import _valid_num_groups
from src.model.model import _dictionary_alpha_rgb


def _alpha_kernel(shapes: Array) -> Array:
    """Glyph alpha masks as a correlation kernel, ``(kh, kw, 1, K)``."""
    alpha, _ = _dictionary_alpha_rgb(shapes)
    return jnp.transpose(alpha[..., 0], (1, 2, 0))[:, :, jnp.newaxis, :]


def _matched_filter(evidence: Array, kernel: Array) -> Array:
    """The adjoint of :func:`src.model.model._stamp`.

    ``score[q, k] = <evidence, glyph_k centred at q>``, at image resolution and
    on exactly the grid the generative stamp uses. Being the adjoint is the
    whole point: ``Phi^T x`` is the matched filter *for this decoder*, so a peak
    at ``(k, y, x)`` is a direct statement about the latent site of the same
    name. A correlation on any other grid would need its own alignment argument.

    This is also the first step of every convolutional sparse coding algorithm
    -- matching pursuit and ISTA both begin by correlating the residual with the
    dictionary -- so the guide is being initialised at the classical solution
    rather than at noise.
    """
    kh, kw = kernel.shape[:2]
    top, left = (kh - 1) // 2, (kw - 1) // 2
    padded = jnp.pad(evidence, ((0, 0), (top, kh - 1 - top),
                                (left, kw - 1 - left), (0, 0)))
    return jax.lax.conv_general_dilated(
        padded, kernel, (1, 1), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


class PoissonConvBackbone(nnx.Module):
    """Dilated fully-convolutional stem, image resolution in and out.

    Every layer is stride 1 with ``SAME`` padding, so the spatial grid is never
    resampled and the module stays translation-equivariant. Dilation, not
    downsampling, is what buys the receptive field: downsampling would break
    the equivariance that makes the count field and the image the same grid.
    """

    def __init__(self, hidden_dims: tuple[int, ...]=(32, 64, 64, 64, 64),
                 dilations: tuple[int, ...]=(1, 2, 4, 8, 16),
                 in_channels: int=3, max_groups: int=32, *, rngs: nnx.Rngs):
        if len(hidden_dims) != len(dilations):
            raise ValueError(
                f"hidden_dims has {len(hidden_dims)} entries but dilations has "
                f"{len(dilations)}; they index the same layers."
            )
        channels, layers = in_channels, []
        for hidden_dim, dilation in zip(hidden_dims, dilations):
            layers += [
                nnx.Conv(channels, hidden_dim, (3, 3), padding="SAME",
                         kernel_dilation=(dilation, dilation), rngs=rngs),
                nnx.GroupNorm(hidden_dim,
                              num_groups=_valid_num_groups(hidden_dim, max_groups),
                              rngs=rngs),
                nnx.leaky_relu,
            ]
            channels = hidden_dim
        self.dilations = tuple(dilations)
        self.layers = nnx.Sequential(*layers)
        self.out_channels = hidden_dims[-1]

    @property
    def receptive_field(self) -> int:
        """Side length seen by one output pixel, for 3x3 kernels at stride 1."""
        return 1 + 2 * sum(self.dilations)

    def __call__(self, images: Float[Array, "B H W C_in"]
                 ) -> Float[Array, "B H W C_out"]:
        return self.layers(images)


class PoissonRateHead(nnx.Module):
    """Amortized mean-field proposal over the integer activation field.

        q(a | x) = prod_{k,y,x} Poisson(a[k,y,x] ; lambda_hat[k,y,x](x))

    The rate field is parameterized as a *total* times a *shape*::

        log lambda_hat = log_total + log_softmax(match_gain * z + head(features))

    where ``z`` is the per-image standardized matched-filter score field
    (:func:`_matched_filter`, the adjoint of the generative stamp) and ``head``
    is a ``1 x 1`` convolution on backbone features. Both extra parameters are
    scalars and both are learnable.

    Three things this buys.

    **It is calibrated by construction.** ``log_softmax`` over the site axes
    makes the rates sum to ``exp(log_total)``, initialised at
    ``expected_count``. A flat score field therefore reproduces the prior rate
    ``expected_count / (H W K)`` exactly; a peaked one redistributes the same
    total mass without changing it. The total is then free to learn, which it
    must be -- saturating a glyph wants one or two spikes, not exactly one.

    **It breaks the cold start.** ``head`` is zero-initialised, so at step 0 the
    proposal *is* the matched filter at temperature ``match_gain``. This matters
    because the score-function estimator cannot bootstrap itself: a spike drawn
    uniformly over 230,400 sites lands usefully with probability ~1e-3, so the
    leave-one-out baseline spends its time ranking equally-wrong samples and the
    proposal learns only the marginal (§11 of the design note). Starting from
    ``Phi^T x`` hands RLOO samples worth ranking.

    **The learned part is a residual.** ``head`` corrects the matched filter
    rather than replacing it, which is the right division of labour: the
    correlation is exact for clean, undistorted glyphs and degrades under the
    deformation the model is eventually meant to handle.

    Note this is the same ``total x allocation`` factorisation that
    :class:`~src.model.model.PoissonMarkedPlacements` used for ``z_rate`` and
    ``z_where`` -- but here it lives in the *guide*, where a simplex is a
    parameterization, not in the model, where it was a Dirichlet prior with an
    unbounded density on the boundary.
    """

    def __init__(self, shape_dict: ShapeDictionary, backbone_channels: int=64,
                 expected_count: float=1., img_h: int=80, img_w: int=80,
                 match_gain: float=2., max_log_rate: float=5.,
                 min_log_rate: float=-20., *, rngs: nnx.Rngs):
        self.expected_count = expected_count
        self.height = img_h
        self.match_gain = match_gain
        self.max_log_rate = max_log_rate
        self.min_log_rate = min_log_rate
        self.shape_dict = shape_dict
        self.width = img_w
        self.head = nnx.Conv(
            backbone_channels, len(shape_dict), (1, 1),
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(), rngs=rngs,
        )
        self.prior_log_rate = float(
            jnp.log(expected_count / (img_h * img_w * len(shape_dict)))
        )

    @property
    def num_features(self) -> int:
        return len(self.shape_dict)

    def scores(self, images: Float[Array, "B H W 3"]) -> Array:
        """Standardized matched-filter score field, ``(B, H, W, K)``.

        Evidence is the per-pixel ink mass ``max_c (1 - x_c)``, which is
        colour-invariant -- it fires the same for blue ink as for black, so the
        detector never has to guess the ``color`` latent. This assumes a white
        background, which holds exactly here: 95.3% of pixels in this dataset
        are bit-identical to ``(1, 1, 1)``.

        Dividing by ``||alpha_k||`` is the matching-pursuit criterion, and stops
        physically larger glyphs from outscoring smaller ones purely on support.
        """
        kernel = _alpha_kernel(self.shape_dict.shapes)
        ink = (1.0 - images).max(axis=-1, keepdims=True)
        raw = _matched_filter(ink, kernel) / jnp.sqrt(
            (kernel ** 2).sum(axis=(0, 1, 2))
        )
        flat = raw.reshape(raw.shape[0], -1)
        centred = raw - flat.mean(-1)[:, None, None, None]
        # Standardize so match_gain is denominated in standard deviations of the
        # score field, making its initial value portable across datasets.
        return centred / jnp.maximum(flat.std(-1)[:, None, None, None], 1e-6)

    def __call__(self, features: Float[Array, "B H W C_feat"],
                 images: Float[Array, "B H W 3"]) -> Array:
        log_total = numpyro.param("log_total_q",
                                  jnp.log(jnp.asarray(self.expected_count)))
        match_gain = jnp.exp(numpyro.param("log_match_gain_q",
                                           jnp.log(jnp.asarray(self.match_gain))))
        logits = match_gain * self.scores(images) + self.head(features)
        numpyro.deterministic("match_scores", logits)
        # Clipping in log space, not a softplus: the score-function gradient
        # cares about d log(lambda_hat) / d logit, which is exactly 1 under exp
        # at every rate. The clip only bounds the tails -- exp(5) caps a site at
        # ~148 expected spikes.
        log_rate = jnp.clip(
            log_total + jax.nn.log_softmax(logits.reshape(logits.shape[0], -1),
                                           axis=-1).reshape(logits.shape),
            self.min_log_rate, self.max_log_rate,
        )
        numpyro.deterministic("log_rate_q", log_rate)
        return numpyro.sample("a", dist.Poisson(jnp.exp(log_rate)).to_event(3))


class ForegroundFieldGuide(nnx.Module):
    r"""Correlated amortized proposals for texture and raw warp velocity.

    The canonical texture location is decoded from globally pooled image
    features, while the image-coordinate velocity location is decoded from the
    same features after resizing them to the coarse velocity lattice.  Both
    proposal covariances retain the model's proper second-order GMRF form:

    .. math::

        q_\phi(r\mid x)
        &=\mathcal N(r;m_{\phi,r}(x),Q_{\phi,r}^{-1}),\\
        q_\phi(u\mid x)
        &=\mathcal N(u;m_{\phi,u}(x),Q_{\phi,u}^{-1}).

    Their scalar element and bond precisions are learned.  The velocity guide
    proposes the unconstrained coarse ``u`` sampled by the model; affine
    conditioning remains the same deterministic model transformation for every
    proposal sample.
    """

    def __init__(
        self,
        shape_dict: ShapeDictionary,
        backbone_channels: int = 64,
        *,
        cg_iters: int = 300,
        rngs: nnx.Rngs,
        texture_bond_precision: float = 4.0,
        texture_element_precision: float = 1.0,
        warp_bond_precision: float = 0.4,
        warp_coarse_height: int = 10,
        warp_coarse_width: int = 10,
        warp_element_precision: float = 0.1,
    ):
        dictionary = jnp.asarray(shape_dict.shapes)
        if dictionary.ndim != 4:
            raise ValueError(
                "shape_dict.shapes needs shape (K, kh, kw, C), "
                f"got {dictionary.shape}"
            )
        self.cg_iters = cg_iters
        self.kernel_height = dictionary.shape[1]
        self.kernel_width = dictionary.shape[2]
        self.texture_bond_precision = texture_bond_precision
        self.texture_element_precision = texture_element_precision
        self.texture_head = nnx.Linear(
            backbone_channels,
            self.kernel_height * self.kernel_width * 3,
            bias_init=nnx.initializers.zeros_init(),
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.velocity_head = nnx.Conv(
            backbone_channels,
            2,
            (1, 1),
            bias_init=nnx.initializers.zeros_init(),
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.warp_bond_precision = warp_bond_precision
        self.warp_coarse_height = warp_coarse_height
        self.warp_coarse_width = warp_coarse_width
        self.warp_element_precision = warp_element_precision

    def __call__(self, features: Float[Array, "B H W C_feat"]):
        texture_loc = self.texture_head(features.mean(axis=(1, 2))).reshape(
            features.shape[0], self.kernel_height, self.kernel_width, 3
        )
        texture_bond_precision = jnp.exp(numpyro.param(
            "log_color_texture_bond_precision_q",
            jnp.log(jnp.asarray(self.texture_bond_precision)),
        ))
        texture_element_precision = jnp.exp(numpyro.param(
            "log_color_texture_element_precision_q",
            jnp.log(jnp.asarray(self.texture_element_precision)),
        ))
        numpyro.sample(
            "color_texture",
            SecondOrderGaussianMrf(
                texture_loc,
                texture_element_precision
                * jnp.ones((self.kernel_height, self.kernel_width)),
                texture_bond_precision,
                cg_iters=self.cg_iters,
            ),
        )

        coarse_shape = (self.warp_coarse_height, self.warp_coarse_width)
        resized = jax.image.resize(
            features,
            (features.shape[0], *coarse_shape, features.shape[-1]),
            method="cubic",
        )
        velocity_loc = self.velocity_head(resized)
        warp_bond_precision = jnp.exp(numpyro.param(
            "log_warp_bond_precision_q",
            jnp.log(jnp.asarray(self.warp_bond_precision)),
        ))
        warp_element_precision = jnp.exp(numpyro.param(
            "log_warp_element_precision_q",
            jnp.log(jnp.asarray(self.warp_element_precision)),
        ))
        numpyro.sample(
            "warp_velocity",
            SecondOrderGaussianMrf(
                velocity_loc,
                warp_element_precision * jnp.ones(coarse_shape),
                warp_bond_precision,
                cg_iters=self.cg_iters,
            ),
        )


class InkColorFinder(nnx.Module):
    """Amortized ``q(color)``, located by a closed-form estimate off the image.

    Same division of labour as :class:`PoissonRateHead`: a statistic computed
    directly from the pixels sets the location, and the backbone features supply
    only a residual.

    This replaces :class:`~src.inference.captcha_encoder.MarioNetteColorFinder`,
    which reads the colour out of ``backbone(x).mean(axis=(1, 2))`` alone. That
    head is not *incapable* -- on the 2026-08-03 run it recovered the ink colour
    to 0.027 mean absolute error, against a per-channel spread of 0.20-0.24, so
    it was very nearly right. It is **fragile**, and on the 2026-08-04 run it
    collapsed: ``E_q[color] = (0.522, 0.531, 0.517)`` for every image, identical
    to five decimals across twelve held-out images and bit-identical across six
    synthetic recolourings of one. Reconstructions came out grey, measured chroma
    0.02 against 0.39-0.69 in the data, with identification and placement
    perfect.

    The trigger was elsewhere: the rate ran to 5.3 effective spikes per glyph,
    painting fringe pixels at ``A = 0.94`` against a true 0.408, and
    ``sigma_ink`` inflated to 0.200 -- at which point the colour term stopped
    paying. That inflation is *caused* by the grey rather than causing it: with
    the colour pinned to this head's own output the profiled optimum for
    ``sigma_ink`` is 0.187, and with the colour correct it is 0.006. But the
    collapse was available *because* of how this head
    reads its input, and it is not recoverable by training: a least-squares probe
    from those 64 pooled numbers to the true ink colour, fit on 150 images and
    scored on 70, gets held-out ``R^2 = (-0.27, -0.76, -0.28)`` on the collapsed
    backbone -- worse than predicting the dataset mean, so the constant really
    was its optimum once it got there. Two properties make it a one-way door:

    * ``_valid_num_groups(32, 32) == 32``, so the backbone's first ``GroupNorm``
      is instance norm. A first-layer map is ``a * mask + b`` with the colour
      amplitude in ``a``; normalizing per channel over space returns
      ``sign(a) * (mask - mean) / std(mask)``, independent of ``|a|`` and ``b``.
      Cross-channel amplitude ratios are hue, and only their signs survive.
      Measured over the six recolourings, pooled-feature spread falls from 0.0127
      at the input to 0.0015 after the stack; with the norms removed it rises to
      0.119 instead. Colour reaches the head as a ~1% perturbation on features
      whose job is something else.
    * Nothing pushes back hard enough to climb out. :meth:`PoissonRateHead.scores`
      uses deliberately colour-invariant evidence, so the backbone's one strong
      gradient wants colour invariance. Adam's second moments on the collapsed run
      put the gradient RMS at 9.1 on the rate head against 8.5e-3 on this head's
      input layer -- three orders of magnitude, and ``clip_by_global_norm``
      preserves the ratio.

    Reading the colour off the pixels removes the failure mode rather than
    re-tuning around it: the estimate below is exact to 0.002 with no fitting at
    all, so there is no amortization gap left to collapse. Features are kept as an
    input because a zero-initialised residual can only help, and because glyph
    overlap is the one thing the closed form cannot see.
    """

    def __init__(self, backbone_channels: int=64,
                 concentration_init: float=100., core_sharpness: float=8.,
                 hidden_dim: int=128, max_log_concentration: float=11.5,
                 min_concentration: float=1e-3,
                 min_log_concentration: float=-2.3, prior_count: float=1e-2,
                 *, rngs: nnx.Rngs):
        self.concentration_init = concentration_init
        self.core_sharpness = core_sharpness
        self.max_log_concentration = max_log_concentration
        self.min_concentration = min_concentration
        self.min_log_concentration = min_log_concentration
        self.prior_count = prior_count
        # Zero-initialised output, so at step 0 the guide *is* the closed-form
        # estimator at concentration_init and everything learned is a
        # correction -- the same cold-start argument PoissonRateHead makes for
        # its matched filter.
        self.head = nnx.Sequential(
            nnx.Linear(backbone_channels + 4, hidden_dim, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dim, 4, kernel_init=nnx.initializers.zeros_init(),
                       bias_init=nnx.initializers.zeros_init(), rngs=rngs),
        )

    def estimate(self, images: Float[Array, "B H W 3"]
                 ) -> tuple[Float[Array, "B 3"], Float[Array, "B 1"]]:
        """Closed-form ink colour and ink mass, ``(B, 3)`` and ``(B, 1)``.

        Inverting the model's own compositing over a white background: with
        ``x_c = A c_c + (1 - A)``, a pixel reports the ink colour undiluted
        exactly where ``A = 1``, so the estimator is the image averaged with
        weights ``A_hat ** core_sharpness``, where ``A_hat = max_c (1 - x_c)``
        proxies the coverage. Raising it to a power concentrates the average on
        the glyph cores, which is where the colour is least diluted by paper.

        ``A_hat`` underestimates ``A`` by ``1 - min_c c_c``, so a per-pixel
        inversion ``1 - (1 - x_c) / A_hat`` would drive every colour to full
        saturation -- grey ink would come out black. Weighting instead of
        dividing avoids that: as ``core_sharpness`` grows the weight concentrates
        on the fully covered pixels, where no correction is needed at all.

        The one assumption is that glyph cores reach ``A = 1``. Note that
        ``max_p A_hat`` is *not* a test of it -- it maxes out at
        ``1 - min_c c_c``, so its dataset median of 0.808 says the inks are not
        pure primaries, not that coverage is partial. The check that matters is
        the fit itself: 0.002 mean absolute error against the core-pixel colour
        over 88 images, which an unmet assumption would not survive. The white
        background it also assumes is the same one
        :meth:`PoissonRateHead.scores` already relies on.

        A ``prior_count`` of paper at colour 0.5 is mixed in so the ratio stays
        defined when a captcha carries no ink and the colour is genuinely
        unidentified -- the estimate falls back to the prior mean, with the
        concentration head free to go broad. This is defensive rather than
        load-bearing on *this* dataset, where all 5000 images carry ink and only
        0.2% peak below coverage 0.3, but a blank canvas is exactly the input a
        ratio estimator has to survive.

        Every operation is a weighted sum over pixels, so this is exactly as
        translation-equivariant as the rest of the guide.
        """
        coverage = (1. - images).max(axis=-1, keepdims=True)
        weight = coverage ** self.core_sharpness
        mass = weight.sum(axis=(1, 2))                            # (B, 1)
        color = ((weight * images).sum(axis=(1, 2)) + 0.5 * self.prior_count) / (
            mass + self.prior_count
        )
        return color, mass

    def __call__(self, features: Float[Array, "B H W C_feat"],
                 images: Float[Array, "B H W 3"]) -> Float[Array, "B 3"]:
        estimate, mass = self.estimate(images)
        # The ink mass and the estimate are handed to the head explicitly: both
        # are absolute per-image scales, which is the class of statistic the
        # GroupNorm stack destroys, so pooled features cannot supply them. The
        # mass is what tells the head whether to be confident at all.
        summary = jnp.concatenate(
            (features.mean(axis=(1, 2)), estimate,
             jnp.log(mass + self.prior_count)), axis=-1,
        )
        correction = self.head(summary)

        # Parameterized as (mean, concentration) rather than (c1, c0) so the
        # closed form lands on the mean alone and the residual is a logit shift.
        mean = jax.nn.sigmoid(
            jax.scipy.special.logit(jnp.clip(estimate, 1e-4, 1. - 1e-4))
            + correction[..., :3]
        )
        # Clipped for the same reason PoissonRateHead clips its log-rate: only to
        # bound the tails. exp(11.5) = 99k caps the posterior sd at ~0.0016, and
        # exp(-2.3) = 0.1 floors it at a Beta more spread out than uniform.
        log_concentration = jnp.clip(
            numpyro.param("log_color_concentration_q",
                          jnp.log(jnp.asarray(self.concentration_init)))
            + correction[..., 3:],
            self.min_log_concentration, self.max_log_concentration,
        )
        concentration = jnp.exp(log_concentration)
        numpyro.deterministic("color_estimate_q", estimate)
        return numpyro.sample(
            "color",
            dist.Beta(mean * concentration + self.min_concentration,
                      (1. - mean) * concentration + self.min_concentration
                      ).to_event(1),
        )


def poisson_convsc_guide(images, backbone: PoissonConvBackbone,
                         placements: PoissonRateHead,
                         color_finder: InkColorFinder,
                         fields: ForegroundFieldGuide | None=None,
                         backgrounder: nnx.Module | None=None):
    """Guide mirroring :func:`src.model.model.poisson_convsc_model`.

    Samples the model's ``a``, ``color``, ``color_texture``, and
    ``warp_velocity`` sites inside the same ``batch`` plate.  Passing
    ``fields=None`` retains compatibility with an untextured, unwarped
    :class:`~src.model.model.PoissonConvPlacements`.
    """
    backbone = nnx_module("backbone_q", backbone)
    color_finder = nnx_module("color_finder_q", color_finder)
    if fields is not None:
        fields = nnx_module("fields_q", fields)
    placements = nnx_module("placements_q", placements)
    if backgrounder is not None:
        backgrounder = nnx_module("backgrounder_q", backgrounder)

    with numpyro.plate("batch", images.shape[0]):
        features = backbone(images)
        color_finder(features, images)
        if fields is not None:
            fields(features)
        if backgrounder is not None:
            backgrounder(features)
        placements(features, images)
