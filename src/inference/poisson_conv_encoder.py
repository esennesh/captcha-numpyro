"""Convolutional proposal for the Poisson convolutional sparse coding model.

``poisson_convsc_model`` puts an integer activation field ``a[k, y, x]`` at
*image* resolution, so the proposal over it is an ordinary image-to-image
network: a stack of dilated convolutions from ``(B, H, W, 3)`` to
``(B, H, W, K)`` log-rates, plus a pooled head for the global ink colour.

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

from typing import Optional, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import nnx_module

from src.data.dictionary import ShapeDictionary
from src.inference.captcha_encoder import (MarioNetteColorFinder,
                                           _valid_num_groups)
from src.model.model import _dictionary_alpha_rgb


def _alpha_kernel(shapes: Array) -> Array:
    """Glyph alpha masks as a correlation kernel, ``(kh, kw, 1, K)``."""
    alpha, _ = _dictionary_alpha_rgb(shapes)
    return jnp.transpose(alpha[..., 0], (1, 2, 0))[:, :, jnp.newaxis, :]


def _matched_filter(evidence: Array, kernel: Array) -> Array:
    """Correlate ``evidence`` against each glyph on the generative stamp's grid.

    ``score[q, k] = <evidence, glyph_k centred at q>``, at image resolution and
    on exactly the grid :func:`src.model.model._stamp` uses. The shared geometry
    is the point: a peak at ``(k, y, x)`` is a direct statement about the latent
    site of the same name, where a correlation on any other grid would need its
    own alignment argument.

    This is called with the glyphs' raw ``alpha`` (see
    :func:`_alpha_kernel`), which makes it the exact adjoint of a *coverage*
    stamp. ``_stamp`` itself convolves ``-log(1 - alpha)`` so that the
    downstream ``1 - exp(-tau)`` is exact alpha compositing, so the two are no
    longer adjoint in the strict sense -- identical convolution structure and
    crop, different radial weighting. Coverage is the better detection statistic
    anyway, since image ink *is* a coverage fraction: on this dataset the argmax
    of the score field recovers the correct glyph identity 100% of the time.
    Switch this kernel to ``-log(1 - alpha)`` if exact adjointness is ever
    wanted for its own sake.

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

    def __init__(self, hidden_dims: Tuple[int, ...]=(32, 64, 64, 64, 64),
                 dilations: Tuple[int, ...]=(1, 2, 4, 8, 16),
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


def poisson_convsc_guide(images, backbone: PoissonConvBackbone,
                         placements: PoissonRateHead,
                         color_finder: MarioNetteColorFinder,
                         backgrounder: Optional[nnx.Module]=None):
    """Guide mirroring :func:`src.model.model.poisson_convsc_model`.

    Samples the same two latent sites the model does -- ``a`` with event shape
    ``(H, W, K)`` and ``color`` with event shape ``(3,)`` -- inside the same
    ``batch`` plate.
    """
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
        placements(features, images)
