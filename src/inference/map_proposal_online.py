r"""Online importance sampling with a fitted MAP-centered proposal.

For one observation ``x``, :class:`~numpyro.contrib.map_proposal.AutoMAPProposal`
first finds a mode of the DSGD-smoothed full target,

.. math::

    \widetilde z^*
    = \mathop{\rm argmax}_{\widetilde z}
      \log \gamma_{\theta,\eta}(\widetilde z; x),

and fits a factorized proposal :math:`q_\phi(z\mid x)` around that mode.  Exact
discrete count samples and continuous samples from the fitted proposal are then
importance weighted with the original, unsmoothed generative model:

.. math::

    z^{(s)} &\sim q_\phi(z\mid x), \\
    \ell_s &= \log \gamma_\theta(z^{(s)};x)
              - \log q_\phi(z^{(s)}\mid x), \\
    \widetilde w_s &=
        \frac{\exp \ell_s}{\sum_r \exp \ell_r}, \\
    \mathbb E_{\pi_\theta(z\mid x)}[f(z)]
        &\approx \sum_s \widetilde w_s f(z^{(s)}).

The proposal is fitted once and frozen while drawing the importance particles.
Every spatial location and dictionary identity remains represented in the
Poisson count tensor throughout fitting and sampling.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.contrib.diag_sgd import SmoothedCount
from numpyro.contrib.map_proposal import AutoMAPProposal
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.util import get_importance_trace

from src.inference.count_relaxation import patch_count_log_pmf

# Repair the Gamma-count log PMF before any relaxed count is differentiated.
# Without it the proposal phase silently reports NaN losses and never moves;
# see :mod:`src.inference.count_relaxation`. A no-op once upstream carries the
# fix.
patch_count_log_pmf()

# The DSGD relaxation of a count site has support ``(0, inf)``, so its
# unconstrained representation is a logarithm and no positive count can be
# initialized at zero. This floor is the smallest count the initializer will
# hand back.
_MINIMUM_RELAXED_COUNT = 1e-12


def _relaxed_count_base(distribution):
    """The discrete law behind a DSGD count relaxation, or ``None``."""
    if isinstance(distribution, SmoothedCount):
        return distribution.base_dist
    base = getattr(distribution, "base_dist", None)
    return None if base is None else _relaxed_count_base(base)


def init_to_count_mean(site=None, *, fallback=init_to_uniform):
    r"""Initialize a DSGD-relaxed count site at its own prior mean.

    ``AutoMAPProposal`` otherwise falls back to
    :func:`~numpyro.infer.initialization.init_to_uniform`, and
    :class:`~numpyro.contrib.diag_sgd.SmoothedCount` has support ``(0, inf)``,
    so the unconstrained representation is a logarithm and the default draws
    ``exp(U(-2, 2))`` -- a mean of 1.81 -- at *every* count coordinate. For an
    ``80 x 80`` image and a 36-glyph dictionary that is 417,081 glyph stamps.

    That point is not merely far from the mode, it is a point where the
    optimizer has nothing to follow. Counts reach the image only through the
    optical depth ``tau``, and 417,081 stamps put ``tau`` at 7.1e4 per pixel, so
    ``A = 1 - exp(-tau)`` is 1.0 and ``dA/dtau = exp(-tau)`` is *exactly zero in
    float32* at every pixel. The likelihood gradient on all 230,400 count
    coordinates vanishes and only the Poisson prior pushes, uniformly downward,
    one Adam step size per step.

    The prior mean is the natural cure and it restores the gradient: at a
    homogeneous rate of ``4/230400`` the same render has ``tau = 0.68`` per
    pixel and ``dA/dtau`` up to 0.81. It is also not a warm start -- it uses the
    model's own rate and nothing derived from the observation, so no matched
    filter decides which sites begin with count mass.

    Sites without a count relaxation, such as the paper field or the local
    Gamma precisions, go to ``fallback``.
    """
    if site is None:
        return functools.partial(init_to_count_mean, fallback=fallback)
    if site["type"] == "sample" and not site["is_observed"]:
        base = _relaxed_count_base(site["fn"])
        if base is not None:
            sample_shape = site["kwargs"].get("sample_shape") or ()
            mean = jnp.broadcast_to(
                base.mean, sample_shape + site["fn"].shape()
            )
            return jnp.clip(mean, _MINIMUM_RELAXED_COUNT, None)
    return fallback(site)


class MAPProposalCaptchaResult(NamedTuple):
    """Outputs from MAP fitting and self-normalized importance sampling."""

    dispersion_converged: jax.Array
    dispersion_losses: jax.Array
    dispersion_num_steps: jax.Array
    effective_sample_size: jax.Array
    log_weights: jax.Array
    map_converged: jax.Array
    map_losses: jax.Array
    map_num_steps: jax.Array
    map_values: dict[str, jax.Array]
    normalized_weights: jax.Array
    reconstructions: jax.Array
    samples: dict[str, jax.Array]
    weighted_counts: jax.Array
    weighted_reconstruction: jax.Array


def _sum_sample_log_probs(model_trace, guide_trace):
    """Return ``log gamma_theta(z; x) - log q_phi(z | x)``."""
    guide_log_prob = sum(
        jnp.sum(site["log_prob"])
        for site in guide_trace.values()
        if site["type"] == "sample"
    )
    model_log_prob = sum(
        jnp.sum(site["log_prob"])
        for site in model_trace.values()
        if site["type"] == "sample"
    )
    return model_log_prob - guide_log_prob


class MAPProposalCaptchaInference:
    """Fit a MAP proposal and importance-sample one observed CAPTCHA."""

    def __init__(
        self,
        model,
        *,
        discrete_temperature: float = 0.1,
        dsgd_kwargs: dict | None = None,
        init_dispersion: float = 0.1,
        init_loc_fn=init_to_count_mean,
        map_max_steps: int = 1000,
        map_optimizer=None,
        map_tolerance: float = 1e-5,
        num_dispersion_particles: int = 8,
        num_importance_samples: int = 64,
        proposal_max_steps: int = 1000,
        proposal_optimizer=None,
        proposal_tolerance: float = 1e-3,
        termination_check_interval: int = 50,
        termination_patience: int = 5,
    ):
        if num_importance_samples < 1:
            raise ValueError("num_importance_samples needs to be positive")
        self.discrete_temperature = discrete_temperature
        self.dsgd_kwargs = dsgd_kwargs
        self.init_dispersion = init_dispersion
        self.init_loc_fn = init_loc_fn
        self.map_max_steps = map_max_steps
        self.map_optimizer = map_optimizer
        self.map_tolerance = map_tolerance
        self.model = model
        self.num_dispersion_particles = num_dispersion_particles
        self.num_importance_samples = num_importance_samples
        self.proposal_max_steps = proposal_max_steps
        self.proposal_optimizer = proposal_optimizer
        self.proposal_tolerance = proposal_tolerance
        self.termination_check_interval = termination_check_interval
        self.termination_patience = termination_patience

    def __call__(self, rng_key: jax.Array, images: jax.Array):
        """Fit once, draw exact proposal samples, and normalize their weights."""
        fit_key, importance_key = jax.random.split(rng_key)
        guide = AutoMAPProposal(
            self.model,
            discrete_temperature=self.discrete_temperature,
            dsgd_kwargs=self.dsgd_kwargs,
            init_dispersion=self.init_dispersion,
            init_loc_fn=self.init_loc_fn,
            map_max_steps=self.map_max_steps,
            map_optimizer=self.map_optimizer,
            map_tolerance=self.map_tolerance,
            num_dispersion_particles=self.num_dispersion_particles,
            proposal_max_steps=self.proposal_max_steps,
            proposal_optimizer=self.proposal_optimizer,
            proposal_tolerance=self.proposal_tolerance,
            termination_check_interval=self.termination_check_interval,
            termination_patience=self.termination_patience,
        )
        fit_result = guide.fit(fit_key, images)

        def importance_particle(key):
            guide_key, model_key = jax.random.split(key)
            model_trace, guide_trace = get_importance_trace(
                handlers.seed(self.model, model_key),
                handlers.seed(guide, guide_key),
                (images,),
                {"plot_mean": True},
                {},
            )
            samples = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "sample"
            }
            return (
                _sum_sample_log_probs(model_trace, guide_trace),
                model_trace["mean"]["value"],
                samples,
            )

        keys = jax.random.split(importance_key, self.num_importance_samples)
        log_weights, reconstructions, samples = jax.vmap(importance_particle)(keys)
        normalized_weights = jax.nn.softmax(log_weights)
        count_weight_shape = (self.num_importance_samples,) + (1,) * (
            samples["a"].ndim - 1
        )
        reconstruction_weight_shape = (self.num_importance_samples,) + (
            1,
        ) * (reconstructions.ndim - 1)
        weighted_counts = jnp.sum(
            normalized_weights.reshape(count_weight_shape) * samples["a"],
            axis=0,
        )
        weighted_reconstruction = jnp.sum(
            normalized_weights.reshape(reconstruction_weight_shape)
            * reconstructions,
            axis=0,
        )
        effective_sample_size = 1.0 / jnp.sum(normalized_weights**2)
        proposal_result = fit_result.proposal_result
        dispersion_converged = (
            True if proposal_result is None else proposal_result.converged
        )
        dispersion_losses = (
            jnp.empty((0,)) if proposal_result is None else proposal_result.losses
        )
        dispersion_num_steps = (
            0 if proposal_result is None else proposal_result.num_steps
        )
        return MAPProposalCaptchaResult(
            dispersion_converged=jnp.asarray(dispersion_converged),
            dispersion_losses=dispersion_losses,
            dispersion_num_steps=jnp.asarray(dispersion_num_steps),
            effective_sample_size=effective_sample_size,
            log_weights=log_weights,
            map_converged=jnp.asarray(fit_result.map_result.converged),
            map_losses=fit_result.map_result.losses,
            map_num_steps=jnp.asarray(fit_result.map_result.num_steps),
            map_values=fit_result.map_estimate,
            normalized_weights=normalized_weights,
            reconstructions=reconstructions,
            samples=samples,
            weighted_counts=weighted_counts,
            weighted_reconstruction=weighted_reconstruction,
        )
