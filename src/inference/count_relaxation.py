r"""A numerical repair for the DSGD count relaxation's Gamma-count log PMF.

:func:`numpyro.contrib.diag_sgd._gamma_count_log_pmf` evaluates

.. math::

    \log\left[P(\alpha z,\alpha\lambda)-P(\alpha(z+1),\alpha\lambda)\right]

by taking the logarithm of each incomplete-gamma tail and differencing them
with ``logdiffexp``. It already switches between the lower and upper tails to
avoid cancellation, and it already guards the *branch selection* with the
double-``where`` idiom JAX requires. What it does not guard is underflow: once
a tail is smaller than the smallest representable float, ``jnp.log`` of it is
``-inf``, and although ``logdiffexp`` returns the right value there, its
derivative is ``NaN``.

The consequence is silent and total. ``AutoMAPProposal``'s proposal phase
differentiates the relaxed count, gets ``NaN``, and
``optimizer.eval_and_stable_update`` discards every update -- so the fitted
dispersions never move, while the only visible symptom is a loss history of
``NaN``. It bites at ordinary settings: whether the gradient is finite at rate
``1e-3``, in float32, as a function of the fitted GammaCount concentration and
the recurrence window ``width``:

===============  =====  =====  =====  =====
concentration    W = 2  W = 3  W = 4  W = 8
===============  =====  =====  =====  =====
1.0              yes    yes    yes    yes
1.65             yes    yes    yes    NaN
2.72             yes    yes    NaN    NaN
4.48             NaN    NaN    NaN    NaN
===============  =====  =====  =====  =====

The concentration is the axis that matters, and it is *fitted*: it starts at
one and the proposal objective pushes it up, because an underdispersed count
proposal is what reduces importance-weight variance. No choice of ``width``
survives that, which is why narrowing the recurrence window is not a fix.

Enabling float64 only postpones it. At rate ``1e-3`` a concentration of 7.4 is
already past the float32 boundary and 54.6 is past the float64 one, both with a
finite value and a ``NaN`` derivative, so the repair is needed at either
precision and so is a probe that detects it at either.

The repair keeps the tails in linear space long enough to floor them at the
smallest positive normal before taking a logarithm, so ``logdiffexp`` never
receives an infinity and no derivative can become ``NaN``. Where only the
subtrahend has underflowed -- the common boundary case, and the one that made
the gradient ``NaN`` -- the result keeps its full dynamic range. Where both
tails have underflowed the mass is negligible by construction and the log PMF
is reported as ``log(tiny)``, about ``-87`` in float32, with zero gradient.

This belonged upstream in the ``codex/map-proposal`` branch of
https://github.com/esennesh/numpyro. :func:`patch_count_log_pmf` is a temporary
shim: it probes for the defect rather than reading a version, so it becomes a
no-op once upstream carries the floor, and it is idempotent.

**Upstream carries it as of commit ``4ca6f62`` (checked 2026-09-09), so this
module is now inert.** The floor moved into ``GammaCount.log_prob`` in
``numpyro/distributions/discrete.py`` -- the same four tails maxed at
``finfo.tiny`` before ``jnp.log``, plus the ``logdiffexp(x, x)`` case -- and the
same commit *deleted* ``diag_sgd._gamma_count_log_pmf``, every ``_idx_spec``
family having been rerouted through its own distribution's ``log_prob``. That
deletion is why :func:`patch_count_log_pmf` reads the attribute with
``getattr``: a bare access raises ``AttributeError`` against a current
checkout, and this module is imported for effect by
:mod:`src.inference.map_proposal_online`, so the failure would land at import
time. The whole module can be retired once the pin no longer reaches a commit
that predates the fix.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaincc
from numpyro.contrib import diag_sgd
from numpyro.distributions import GammaCount
from numpyro.distributions.util import logdiffexp

__all__ = ["patch_count_log_pmf", "upstream_is_floored"]

_PATCH_MARKER = "_captcha_numpyro_floored"


def _floored_gamma_count_log_pmf(value, concentration, rate):
    """``_gamma_count_log_pmf`` with underflowed tails floored before ``log``."""
    count = jnp.where(value == 0, 1.0, value)
    left_shape = concentration * count
    right_shape = concentration * (count + 1.0)
    scaled_rate = concentration * rate

    lower_left = gammainc(left_shape, scaled_rate)
    lower_right = gammainc(right_shape, scaled_rate)
    upper_left = gammaincc(left_shape, scaled_rate)
    upper_right = gammaincc(right_shape, scaled_rate)
    use_lower = lower_left < upper_right

    # Pick whichever pair is the small one, exactly as upstream does.
    larger = jnp.where(use_lower, lower_left, upper_right)
    smaller = jnp.where(use_lower, lower_right, upper_left)

    tiny = jnp.finfo(jnp.result_type(larger, float)).tiny
    log_larger = jnp.log(jnp.clip(larger, tiny, None))
    log_smaller = jnp.log(jnp.clip(smaller, tiny, None))

    # `logdiffexp` needs a strict inequality and must never see an infinity, so
    # substitute a finite, ordered pair on the degenerate branch and discard
    # its value afterwards. Both operands are finite in every branch, which is
    # what keeps the derivative finite too.
    degenerate = log_larger <= log_smaller
    safe_smaller = jnp.where(degenerate, log_larger - 1.0, log_smaller)
    positive_log_prob = jnp.where(
        degenerate, jnp.log(tiny), logdiffexp(log_larger, safe_smaller)
    )

    zero_tail = gammaincc(concentration, scaled_rate)
    zero_log_prob = jnp.log(jnp.clip(zero_tail, tiny, None))
    return jnp.where(value == 0, zero_log_prob, positive_log_prob)


#: A concentration and rate whose Gamma-count tail underflows in float32 *and*
#: in float64, so the probe below detects the defect at either precision.
#: Enabling x64 only moves the boundary -- a concentration of 54.6 at this rate
#: is already past it in float64 -- so a probe tuned to float32 would silently
#: decide the defect was absent under x64 and skip the repair.
PROBE_CONCENTRATION = 500.0
PROBE_RATE = 1e-3


def upstream_is_floored() -> bool:
    """Whether the Gamma-count log PMF already survives an underflowed tail.

    Probes the failure directly rather than reading a version number.
    """

    def log_pmf(log_rate):
        distribution = GammaCount(
            concentration=jnp.asarray(PROBE_CONCENTRATION), rate=jnp.exp(log_rate)
        )
        return jnp.sum(
            diag_sgd._count_log_pmf(
                distribution, jnp.asarray([1.0]), trailing_ndims=1
            )
        )

    gradient = jax.grad(log_pmf)(jnp.log(jnp.asarray(PROBE_RATE)))
    return bool(jnp.isfinite(gradient))


def patch_count_log_pmf() -> bool:
    """Install the floored log PMF. Returns whether it was needed.

    Idempotent, and a no-op once upstream carries the floor itself.
    """
    # Read the attribute defensively. The upstream fix landed on
    # `codex/map-proposal` as a floor inside `GammaCount.log_prob`, and the
    # same commit deleted `diag_sgd._gamma_count_log_pmf` entirely, so a bare
    # attribute access raises `AttributeError` against a current checkout --
    # at import time, since `map_proposal_online` calls this on import.
    upstream = getattr(diag_sgd, "_gamma_count_log_pmf", None)
    # Our own marker has to be checked before the probe: once the patch is in,
    # the probe passes, and answering "not needed" would contradict the return
    # contract on the second call.
    if upstream is not None and getattr(upstream, _PATCH_MARKER, False):
        return True
    if upstream_is_floored():
        return False
    if upstream is None:
        raise RuntimeError(
            "the Gamma-count log PMF still has a NaN gradient at an "
            "underflowed tail, but `diag_sgd._gamma_count_log_pmf` is gone, so "
            "this patch has nothing to replace. The upstream floor has moved; "
            "see notes/diag-sgd-zero-inflation-issue.md."
        )
    setattr(_floored_gamma_count_log_pmf, _PATCH_MARKER, True)
    diag_sgd._gamma_count_log_pmf = _floored_gamma_count_log_pmf
    return True
