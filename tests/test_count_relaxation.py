"""Checks for the Gamma-count log-PMF repair.

The defect is a NaN *derivative* with a correct value, which is the worst
combination: `optimizer.eval_and_stable_update` discards the update and the
only symptom is a loss history of NaN.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpyro.distributions as dist
from jax.scipy.special import gammainc, gammaincc
from numpyro.contrib.diag_sgd import anchored_relaxed_count
from numpyro.distributions.util import logdiffexp

from src.inference.count_relaxation import (
    PROBE_CONCENTRATION,
    PROBE_RATE,
    _floored_gamma_count_log_pmf,
    patch_count_log_pmf,
    upstream_is_floored,
)

CONCENTRATIONS = (0.5, 1.0, 2.718, 7.389, 54.6, 500.0)
RATES = (1e-5, 1e-3, 0.1, 1.0, 4.0)


def _default_dtype():
    return jnp.zeros(()).dtype


def _log_tiny():
    """Where the floor bites, in log space, at the precision actually in use."""
    return float(jnp.log(jnp.finfo(_default_dtype()).tiny))


def _boundary_concentration():
    """A concentration whose log PMF is finite but whose derivative is not.

    ``tests/test_diffeomorphism`` enables x64 globally, and x64 only *moves*
    the underflow boundary rather than removing it, so this point has to follow
    the active precision.
    """
    return 7.389 if _default_dtype() == jnp.float32 else 54.6


@pytest.fixture(scope="module", autouse=True)
def patched():
    patch_count_log_pmf()


def test_patch_is_idempotent_and_self_disabling():
    assert patch_count_log_pmf() is True
    assert patch_count_log_pmf() is True
    assert upstream_is_floored()


@pytest.mark.parametrize("concentration", CONCENTRATIONS)
@pytest.mark.parametrize("rate", RATES)
def test_relaxed_count_gradient_is_finite(concentration, rate):
    uniforms = jax.random.uniform(jax.random.key(0), (256,))

    def mean_relaxed_count(parameters):
        distribution = dist.GammaCount(
            concentration=jnp.exp(parameters[0]), rate=jnp.exp(parameters[1])
        )
        return jnp.mean(
            anchored_relaxed_count(
                distribution, uniforms, 0.1, width=8, max_count=16
            )
        )

    parameters = jnp.asarray([jnp.log(concentration), jnp.log(rate)])
    assert np.isfinite(float(mean_relaxed_count(parameters)))
    gradient = jax.grad(mean_relaxed_count)(parameters)
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize("concentration", CONCENTRATIONS)
@pytest.mark.parametrize("rate", RATES)
def _unfloored_gamma_count_log_pmf(value, concentration, rate):
    """The upstream formula, inline, so the comparison cannot depend on whether
    the patch has already been installed by another test module."""
    count = jnp.where(value == 0, 1.0, value)
    left_shape = concentration * count
    right_shape = concentration * (count + 1.0)
    scaled_rate = concentration * rate
    lower_left = gammainc(left_shape, scaled_rate)
    lower_right = gammainc(right_shape, scaled_rate)
    upper_left = gammaincc(left_shape, scaled_rate)
    upper_right = gammaincc(right_shape, scaled_rate)
    use_lower = lower_left < upper_right
    lower = logdiffexp(
        jnp.log(jnp.where(use_lower, lower_left, 1.0)),
        jnp.log(jnp.where(use_lower, lower_right, 0.5)),
    )
    upper = logdiffexp(
        jnp.log(jnp.where(use_lower, 1.0, upper_right)),
        jnp.log(jnp.where(use_lower, 0.5, upper_left)),
    )
    positive = jnp.where(use_lower, lower, upper)
    zero = jnp.log(gammaincc(concentration, scaled_rate))
    return jnp.where(value == 0, zero, positive)


@pytest.mark.parametrize("concentration", CONCENTRATIONS)
@pytest.mark.parametrize("rate", RATES)
def test_values_match_upstream_wherever_upstream_is_finite(concentration, rate):
    """The floor may only touch tails that have underflowed.

    Flooring at the smallest *normal* float, rather than only at exact zero,
    keeps ``1 / x`` finite in the derivative -- the reciprocal of a denormal is
    already ``inf``. The price is a discrepancy within about eight nats of
    ``log(finfo.tiny)``, which is a probability near 1e-35 in float32.
    """
    counts = jnp.arange(0.0, 10.0)
    upstream = np.asarray(
        _unfloored_gamma_count_log_pmf(
            counts, jnp.asarray(concentration), jnp.asarray(rate)
        )
    )
    patched_values = np.asarray(
        _floored_gamma_count_log_pmf(
            counts, jnp.asarray(concentration), jnp.asarray(rate)
        )
    )
    # Agreement is exact wherever upstream is comfortably above the floor.
    representable = np.isfinite(upstream) & (upstream > _log_tiny() + 8.0)
    if representable.any():
        np.testing.assert_allclose(
            patched_values[representable],
            upstream[representable],
            rtol=0,
            atol=1e-4,
        )
    finite = np.isfinite(upstream)
    if finite.any():
        assert np.max(np.abs(patched_values[finite] - upstream[finite])) < 0.05


def test_the_defect_is_a_nan_derivative_at_a_correct_value():
    """Both halves matter: the value is right, which is why it went unnoticed.

    :func:`upstream_is_floored` probes further out, where the value is already
    ``-inf``; this needs the boundary step, where the value is still right.
    """
    concentration, rate = jnp.asarray(_boundary_concentration()), PROBE_RATE

    def unfloored(log_rate):
        return jnp.sum(
            _unfloored_gamma_count_log_pmf(
                jnp.asarray([1.0]), concentration, jnp.exp(log_rate)
            )
        )

    def floored(log_rate):
        return jnp.sum(
            _floored_gamma_count_log_pmf(
                jnp.asarray([1.0]), concentration, jnp.exp(log_rate)
            )
        )

    argument = jnp.log(jnp.asarray(rate))
    np.testing.assert_allclose(
        float(unfloored(argument)), float(floored(argument)), atol=1e-3
    )
    assert not np.isfinite(float(jax.grad(unfloored)(argument)))
    assert np.isfinite(float(jax.grad(floored)(argument)))
