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

from contextlib import ExitStack
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.contrib.map_proposal import AutoMAPProposal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import get_importance_trace

from src.inference.online import restrict_poisson_model


class MAPProposalCaptchaResult(NamedTuple):
    """Outputs from MAP fitting and self-normalized importance sampling."""

    candidate_sites: jax.Array
    dispersion_converged: jax.Array
    effective_sample_size: jax.Array
    log_weights: jax.Array
    map_converged: jax.Array
    map_values: dict[str, jax.Array]
    normalized_weights: jax.Array
    reconstructions: jax.Array
    samples: dict[str, jax.Array]
    weighted_reconstruction: jax.Array


def _frozen_guide(guide: AutoMAPProposal):
    """Expose the fitted proposal without triggering another optimization."""

    def sample(*args, **kwargs):
        plates = guide._create_plates(*args, **kwargs)
        prototype_trace = guide.prototype_trace
        if prototype_trace is None:
            raise RuntimeError("AutoMAPProposal has not been fitted")

        result = {}
        for name, site in prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])
                result[name] = numpyro.sample(
                    name,
                    guide._get_proposal(name, guide._proposal_params[name]),
                )
        return result

    return sample


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


def _stabilize_fitted_guide(guide: AutoMAPProposal) -> None:
    r"""Replace unusable optimizer outputs with full-support initial values.

    BFGS is allowed to stop before convergence and can produce non-finite or
    exponentially overflowing log-parameters.  Replacing only those invalid
    proposal coordinates does not change :math:`\gamma_\theta`; importance
    weighting remains valid for any resulting full-support
    :math:`q_\phi(z\mid x)`.
    """
    guide._map_locs = {
        name: jnp.where(jnp.isfinite(value), value, guide._init_locs[name])
        for name, value in guide._map_locs.items()
    }
    initial_parameters = guide._initial_proposal_params()
    stabilized_parameters = {}
    for name, parameters in guide._proposal_params.items():
        stabilized_parameters[name] = {}
        for parameter_name, value in parameters.items():
            fallback = initial_parameters[name][parameter_name]
            valid = jnp.isfinite(value) & (value >= -12.0) & (value <= 12.0)
            stabilized_parameters[name][parameter_name] = jnp.where(
                valid, value, fallback
            )
    guide._proposal_params = stabilized_parameters


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
        classes_per_location: int = 3,
        discrete_temperature: float = 0.1,
        dsgd_kwargs: dict | None = None,
        init_dispersion: float = 0.1,
        initial_count_mass: float | None = None,
        min_distance: float = 6.0,
        num_candidates: int = 12,
        num_dispersion_particles: int = 8,
        num_importance_samples: int = 64,
        optimizer_options: dict | None = None,
    ):
        if initial_count_mass is not None and initial_count_mass <= 0:
            raise ValueError("initial_count_mass needs to be positive")
        if num_importance_samples < 1:
            raise ValueError("num_importance_samples needs to be positive")
        self.classes_per_location = classes_per_location
        self.discrete_temperature = discrete_temperature
        self.dsgd_kwargs = dsgd_kwargs
        self.init_dispersion = init_dispersion
        self.initial_count_mass = initial_count_mass
        self.min_distance = min_distance
        self.model = model
        self.num_candidates = num_candidates
        self.num_dispersion_particles = num_dispersion_particles
        self.num_importance_samples = num_importance_samples
        self.optimizer_options = optimizer_options

    def __call__(self, rng_key: jax.Array, images: jax.Array):
        """Fit once, draw exact proposal samples, and normalize their weights."""
        fit_key, importance_key = jax.random.split(rng_key)
        model, candidate_sites = restrict_poisson_model(
            self.model,
            images,
            classes_per_location=self.classes_per_location,
            min_distance=self.min_distance,
            num_candidates=self.num_candidates,
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
            num_dispersion_particles=self.num_dispersion_particles,
            optimizer_options=self.optimizer_options,
        )
        guide.find_map(fit_key, images)
        _stabilize_fitted_guide(guide)
        map_values = {
            name: guide._transforms[name](value)
            for name, value in guide._map_locs.items()
        }
        frozen_guide = _frozen_guide(guide)

        def importance_particle(key):
            guide_key, model_key = jax.random.split(key)
            model_trace, guide_trace = get_importance_trace(
                handlers.seed(model, model_key),
                handlers.seed(frozen_guide, guide_key),
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
        dispersion_converged = (
            True
            if guide.dispersion_result is None
            else guide.dispersion_result.success
        )
        return MAPProposalCaptchaResult(
            candidate_sites=candidate_sites,
            dispersion_converged=jnp.asarray(dispersion_converged),
            effective_sample_size=effective_sample_size,
            log_weights=log_weights,
            map_converged=jnp.asarray(guide.map_result.success),
            map_values=map_values,
            normalized_weights=normalized_weights,
            reconstructions=reconstructions,
            samples=samples,
            weighted_reconstruction=weighted_reconstruction,
        )
