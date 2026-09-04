r"""Online importance sampling with a fitted MAP-centered proposal.

For one observation ``x``, :class:`~numpyro.contrib.map_proposal.AutoMAPProposal`
first finds a mode of the DSGD-smoothed, candidate-restricted target,

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
As in :mod:`src.inference.online`, this targets the exact joint density on an
observation-specific candidate support, rather than the full count lattice.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.contrib.map_proposal import AutoMAPProposal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import get_importance_trace

from src.inference.online import restrict_poisson_model


class MAPProposalCaptchaResult(NamedTuple):
    """Outputs from MAP fitting and self-normalized importance sampling."""

    candidate_sites: jax.Array
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
    weighted_reconstruction: jax.Array


def _initial_guide_values(model, candidate_sites, images, count_mass):
    """Construct a coherent, neutral starting CAPTCHA explanation."""
    candidate_numpy = np.asarray(candidate_sites)
    first_indices = []
    locations = set()
    for index, (y, x, _) in enumerate(candidate_numpy):
        location = (int(y), int(x))
        if location not in locations:
            first_indices.append(index)
            locations.add(location)

    placements = model.keywords["placements"]
    minimum_count = jnp.minimum(
        jnp.asarray(1e-3, dtype=images.dtype),
        count_mass / (2.0 * candidate_sites.shape[0]),
    )
    candidate_counts = jnp.full(
        (1, candidate_sites.shape[0]), minimum_count, dtype=images.dtype
    )
    remaining_count = count_mass - minimum_count * (
        candidate_sites.shape[0] - len(first_indices)
    )
    candidate_counts = candidate_counts.at[0, jnp.asarray(first_indices)].set(
        remaining_count / len(first_indices)
    )
    color = jnp.clip(
        jnp.quantile(images, 0.01, axis=(0, 1, 2)), 1e-3, 1.0 - 1e-3
    )[jnp.newaxis]
    color_texture = jnp.zeros(
        (1, *placements.ink_kernel.shape[:2], 3), dtype=images.dtype
    )
    warp_velocity = jnp.zeros(
        (
            1,
            placements.warp_coarse_height,
            placements.warp_coarse_width,
            2,
        ),
        dtype=images.dtype,
    )
    return {
        "candidate_counts": candidate_counts,
        "color": color,
        "color_texture": color_texture,
        "warp_velocity": warp_velocity,
    }


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
        initial_count_mass: float | None = None,
        map_max_steps: int = 1000,
        map_optimizer=None,
        map_tolerance: float = 1e-5,
        min_distance: float = 6.0,
        num_dispersion_particles: int = 8,
        num_importance_samples: int = 64,
        num_locations: int = 4,
        proposal_max_steps: int = 1000,
        proposal_optimizer=None,
        proposal_tolerance: float = 1e-3,
        termination_check_interval: int = 50,
        termination_patience: int = 5,
    ):
        if initial_count_mass is not None and initial_count_mass <= 0:
            raise ValueError("initial_count_mass needs to be positive")
        if num_importance_samples < 1:
            raise ValueError("num_importance_samples needs to be positive")
        self.discrete_temperature = discrete_temperature
        self.dsgd_kwargs = dsgd_kwargs
        self.init_dispersion = init_dispersion
        self.initial_count_mass = initial_count_mass
        self.map_max_steps = map_max_steps
        self.map_optimizer = map_optimizer
        self.map_tolerance = map_tolerance
        self.min_distance = min_distance
        self.model = model
        self.num_dispersion_particles = num_dispersion_particles
        self.num_importance_samples = num_importance_samples
        self.num_locations = num_locations
        self.proposal_max_steps = proposal_max_steps
        self.proposal_optimizer = proposal_optimizer
        self.proposal_tolerance = proposal_tolerance
        self.termination_check_interval = termination_check_interval
        self.termination_patience = termination_patience

    def __call__(self, rng_key: jax.Array, images: jax.Array):
        """Fit once, draw exact proposal samples, and normalize their weights."""
        fit_key, importance_key = jax.random.split(rng_key)
        model, candidate_sites = restrict_poisson_model(
            self.model,
            images,
            min_distance=self.min_distance,
            num_locations=self.num_locations,
        )
        initial_count_mass = (
            self.model.keywords["placements"].expected_count
            if self.initial_count_mass is None
            else self.initial_count_mass
        )
        initial_values = _initial_guide_values(
            model, candidate_sites, images, initial_count_mass
        )
        guide = AutoMAPProposal(
            model,
            discrete_temperature=self.discrete_temperature,
            dsgd_kwargs=self.dsgd_kwargs,
            init_dispersion=self.init_dispersion,
            init_loc_fn=init_to_value(values=initial_values),
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
                handlers.seed(model, model_key),
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
        reconstruction_weight_shape = (self.num_importance_samples,) + (
            1,
        ) * (reconstructions.ndim - 1)
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
            candidate_sites=candidate_sites,
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
            weighted_reconstruction=weighted_reconstruction,
        )
