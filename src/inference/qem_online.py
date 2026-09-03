r"""Online moment matching for the candidate-restricted CAPTCHA posterior.

For shortlisted counts ``a_S``, baseline color ``c_0``, canonical texture
``r``, and raw velocity ``u``, QEM maintains a mean-field approximation

.. math::

    q_\phi(z)=q_\phi(a_S)q_\phi(c_0)q_\phi(r)q_\phi(u).

At iteration ``t`` it draws ``K`` alternatives per site, uses MPIW to contract
the resulting ``K**4`` combinations, and estimates each sufficient-statistic
mean with self-normalized marginal weights:

.. math::

    \widehat m_i
    =\sum_{k=1}^K \widetilde w_{ik}T_i(z_{ik}),\qquad
    m_{i,t}=\lambda_t m_{i,t-1}+(1-\lambda_t)\widehat m_i.

The M-step reconstructs each proposal by moment matching.  The registrations
below add the model's Beta color family and a fixed-covariance GMRF family to
NumPyro's exponential-family registry.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.special import digamma, polygamma
from numpyro import handlers
from numpyro.contrib.qem import QEM, QEMRunResult
from numpyro.distributions import Beta
from numpyro.distributions.exp_family import (
    base_distribution,
    canonical_params,
    from_mean_params,
    mean_params,
    sufficient_statistics,
)
from numpyro.distributions.transforms import biject_to
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoExponentialFamily

from src.distributions import SecondOrderGaussianMrf
from src.inference.online import restrict_poisson_model


class OnlineAutoExponentialFamily(AutoExponentialFamily):
    """Auto exponential-family guide that permits unbounded count sites.

    The QEM branch registers Poisson moment matching, but its generic
    ``initialize_model`` path rejects non-enumerable discrete sites before the
    guide can use that registration. QEM needs only a prototype trace and the
    prior-family parameters, so a plain seeded trace is sufficient here and
    avoids continuous unconstraining of the Poisson value.
    """

    def _setup_prototype(self, *args, **kwargs):
        if self.prototype_trace is not None:
            return
        rng_key = numpyro.prng_key()
        with handlers.block():
            self.prototype_trace = handlers.trace(
                handlers.seed(self.model, rng_key)
            ).get_trace(*args, **kwargs)

        self._prototype_frames = {}
        for name, site in self.prototype_trace.items():
            if site["type"] == "plate":
                self._prototype_frame_full_sizes[name] = site["args"][0]
            elif site["type"] == "sample" and not site["is_observed"]:
                for frame in site["cond_indep_stack"]:
                    self._prototype_frames.setdefault(frame.name, frame)

        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            base = base_distribution(site["fn"])
            self._base_dists[name] = base
            self._reinterpreted_ndims[name] = site["fn"].event_dim - base.event_dim
            lead_shape = jnp.shape(site["value"])[
                : jnp.ndim(site["value"]) - base.event_dim
            ]
            initial = {}
            for argument, value in canonical_params(base).items():
                transform = biject_to(base.arg_constraints[argument])
                unconstrained = transform.inv(value)
                event_ndim = transform.domain.event_dim
                event_shape = jnp.shape(unconstrained)[
                    jnp.ndim(unconstrained) - event_ndim :
                ]
                initial[argument] = jnp.broadcast_to(
                    unconstrained, lead_shape + event_shape
                )
            self._init_params[name] = initial


class QEMCaptchaResult(NamedTuple):
    """Posterior proposal draws and diagnostics from one online QEM fit."""

    candidate_sites: jax.Array
    qem_result: QEMRunResult
    reconstructions: jax.Array
    samples: dict[str, jax.Array]


def _beta_from_mean_params(distribution: Beta, params: dict) -> Beta:
    concentration1, concentration0 = _solve_beta_parameters(
        params["logx"],
        params["log1mx"],
        distribution.concentration1,
        distribution.concentration0,
    )
    return Beta(concentration1, concentration0)


def _beta_mean_params(distribution: Beta) -> dict[str, jax.Array]:
    concentration = distribution.concentration1 + distribution.concentration0
    return {
        "log1mx": digamma(distribution.concentration0) - digamma(concentration),
        "logx": digamma(distribution.concentration1) - digamma(concentration),
    }


def _beta_sufficient_statistics(
    distribution: Beta, value: jax.Array
) -> dict[str, jax.Array]:
    del distribution
    return {"log1mx": jnp.log1p(-value), "logx": jnp.log(value)}


def _gmrf_from_mean_params(
    distribution: SecondOrderGaussianMrf, params: dict
) -> SecondOrderGaussianMrf:
    return SecondOrderGaussianMrf(
        params["x"],
        distribution.element_precision,
        distribution.bond_precision,
        cg_iters=distribution.cg_iters,
        edge_masks=(distribution._mask_v, distribution._mask_h),
    )


def _gmrf_mean_params(
    distribution: SecondOrderGaussianMrf,
) -> dict[str, jax.Array]:
    return {"x": distribution.mean}


def _gmrf_sufficient_statistics(
    distribution: SecondOrderGaussianMrf, value: jax.Array
) -> dict[str, jax.Array]:
    del distribution
    return {"x": value}


def _initial_candidate_mean(
    candidate_mean: jax.Array,
    candidate_sites: jax.Array,
    alternative_fraction: float,
    count_mass: float,
) -> jax.Array:
    """Put most initial mass on the best class at each spatial location."""
    candidate_numpy = np.asarray(candidate_sites)
    first_indices = []
    locations = set()
    for index, (y, x, _) in enumerate(candidate_numpy):
        location = (int(y), int(x))
        if location not in locations:
            first_indices.append(index)
            locations.add(location)

    num_alternatives = candidate_sites.shape[0] - len(first_indices)
    if num_alternatives:
        alternative_mean = (
            alternative_fraction * count_mass / num_alternatives
        )
        initial_mean = jnp.full_like(candidate_mean, alternative_mean)
        primary_mean = (
            (1.0 - alternative_fraction) * count_mass / len(first_indices)
        )
    else:
        initial_mean = jnp.zeros_like(candidate_mean)
        primary_mean = count_mass / len(first_indices)
    return initial_mean.at[..., jnp.asarray(first_indices)].set(primary_mean)


def _solve_beta_parameters(
    expected_log_x: jax.Array,
    expected_log1m_x: jax.Array,
    initial_concentration1: jax.Array,
    initial_concentration0: jax.Array,
    *,
    iterations: int = 40,
) -> tuple[jax.Array, jax.Array]:
    r"""Invert the Beta mean map with damped Newton iterations.

    The Beta sufficient statistics and mean parameters are

    .. math::

        T(x)&=(\log x,\log(1-x)),\\
        m&=(\psi(\alpha)-\psi(\alpha+\beta),
             \psi(\beta)-\psi(\alpha+\beta)).

    Newton updates act on ``(log(alpha), log(beta))`` to preserve positivity.
    """
    initial = jnp.stack(
        (
            jnp.log(jnp.broadcast_to(initial_concentration1, expected_log_x.shape)),
            jnp.log(jnp.broadcast_to(initial_concentration0, expected_log_x.shape)),
        ),
        axis=-1,
    )

    def update(_, logarithms):
        concentration1, concentration0 = jnp.moveaxis(
            jnp.exp(logarithms), -1, 0
        )
        concentration = concentration1 + concentration0
        common = polygamma(1, concentration)
        residual1 = (
            digamma(concentration1) - digamma(concentration) - expected_log_x
        )
        residual0 = (
            digamma(concentration0) - digamma(concentration) - expected_log1m_x
        )
        jacobian11 = concentration1 * (polygamma(1, concentration1) - common)
        jacobian12 = -concentration0 * common
        jacobian21 = -concentration1 * common
        jacobian22 = concentration0 * (polygamma(1, concentration0) - common)
        determinant = jacobian11 * jacobian22 - jacobian12 * jacobian21
        delta1 = (jacobian22 * residual1 - jacobian12 * residual0) / determinant
        delta0 = (-jacobian21 * residual1 + jacobian11 * residual0) / determinant
        delta = jnp.stack((delta1, delta0), axis=-1)
        return jnp.clip(logarithms - 0.5 * delta, -12.0, 12.0)

    solution = jax.lax.fori_loop(0, iterations, update, initial)
    return tuple(jnp.moveaxis(jnp.exp(solution), -1, 0))


def register_qem_families() -> None:
    """Register the project-specific exponential families idempotently."""
    from_mean_params.register(Beta)(_beta_from_mean_params)
    from_mean_params.register(SecondOrderGaussianMrf)(_gmrf_from_mean_params)
    mean_params.register(Beta)(_beta_mean_params)
    mean_params.register(SecondOrderGaussianMrf)(_gmrf_mean_params)
    sufficient_statistics.register(Beta)(_beta_sufficient_statistics)
    sufficient_statistics.register(SecondOrderGaussianMrf)(
        _gmrf_sufficient_statistics
    )


class QEMCaptchaInference:
    """Fit one observation-specific QEM posterior and draw its final proposal."""

    def __init__(
        self,
        model,
        *,
        classes_per_location: int = 3,
        forget: float | None = None,
        initial_alternative_fraction: float = 0.2,
        initial_count_mass: float | None = None,
        min_distance: float = 6.0,
        num_candidates: int = 12,
        num_posterior_samples: int = 8,
        num_samples: int = 4,
        num_steps: int = 20,
        schedule_power: float = 1.0,
        serial_sites: tuple[str, ...] = (
            "candidate_counts",
            "color",
            "color_texture",
            "warp_velocity",
        ),
    ):
        if not 0.0 <= initial_alternative_fraction < 1.0:
            raise ValueError("initial_alternative_fraction needs to be in [0, 1)")
        if initial_count_mass is not None and initial_count_mass <= 0:
            raise ValueError("initial_count_mass needs to be positive")
        if num_posterior_samples < 1:
            raise ValueError("num_posterior_samples needs to be positive")
        if num_samples < 1:
            raise ValueError("num_samples needs to be positive")
        if num_steps < 1:
            raise ValueError("num_steps needs to be positive")
        self.classes_per_location = classes_per_location
        self.forget = forget
        self.initial_alternative_fraction = initial_alternative_fraction
        self.initial_count_mass = initial_count_mass
        self.min_distance = min_distance
        self.model = model
        self.num_candidates = num_candidates
        self.num_posterior_samples = num_posterior_samples
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.schedule_power = schedule_power
        self.serial_sites = serial_sites
        register_qem_families()

    def __call__(self, rng_key: jax.Array, images: jax.Array) -> QEMCaptchaResult:
        fit_key, predict_key, sample_key = jax.random.split(rng_key, 3)
        model, sites = restrict_poisson_model(
            self.model,
            images,
            classes_per_location=self.classes_per_location,
            min_distance=self.min_distance,
            num_candidates=self.num_candidates,
        )
        qem_model = handlers.block(model, hide=("omitted_count_mass",))
        guide = OnlineAutoExponentialFamily(qem_model)
        qem = QEM(
            qem_model,
            guide,
            self.num_samples,
            forget=self.forget,
            schedule_power=self.schedule_power,
            serial_sites=self.serial_sites,
        )
        state = qem.init(fit_key, images)
        initial_count_mass = (
            self.model.keywords["placements"].expected_count
            if self.initial_count_mass is None
            else self.initial_count_mass
        )
        initial_means = state.mean_params.copy()
        candidate_mean = initial_means["candidate_counts"]["x"]
        initial_means["candidate_counts"] = {
            "x": _initial_candidate_mean(
                candidate_mean,
                sites,
                self.initial_alternative_fraction,
                initial_count_mass,
            )
        }
        state = state._replace(mean_params=initial_means)
        log_marginals = []
        for _ in range(self.num_steps):
            state, log_marginal = qem.update(state, images)
            log_marginals.append(log_marginal)
        qem_result = QEMRunResult(
            params=qem.get_params(state),
            state=state,
            log_marginals=jnp.stack(log_marginals),
        )
        samples = guide.sample_posterior(
            sample_key,
            qem_result.params,
            images,
            sample_shape=(self.num_posterior_samples,),
        )
        reconstructions = Predictive(
            model,
            posterior_samples=samples,
            return_sites=("mean",),
            batch_ndims=1,
        )(predict_key, images, plot_mean=True)["mean"]
        return QEMCaptchaResult(
            candidate_sites=sites,
            qem_result=qem_result,
            reconstructions=reconstructions,
            samples=samples,
        )


register_qem_families()
