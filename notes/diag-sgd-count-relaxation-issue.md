# `diag_sgd`: the count relaxation has a NaN gradient at a correct value, and a silent truncation

*Draft issue for `esennesh/numpyro`, branch `codex/map-proposal`. Not filed yet.*

## Summary

`anchored_relaxed_count` returns the right value and a NaN derivative once the
Gamma-count PMF underflows. Nothing warns you. In `AutoMAPProposal` the effect
is that the proposal-fitting phase runs its whole step budget, changes no
parameter, and reports a loss history of NaN, because
`_NumPyroOptim.eval_and_stable_update` correctly discards every non-finite
update.

A one-line floor stops the NaN, and I have a patch that does. I do not think it
is the right fix, because the same saturation reappears in three other places in
the same code path, each with its own ad-hoc guard, and because a second and
independent defect nearby is also silent.

## Reproduction

```python
import jax, jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.contrib.diag_sgd import _count_log_pmf, anchored_relaxed_count

CONCENTRATION, RATE = 7.389, 1e-3   # float32; use 54.6 under jax_enable_x64

def log_pmf(log_rate):
    d = dist.GammaCount(concentration=jnp.asarray(CONCENTRATION), rate=jnp.exp(log_rate))
    return jnp.sum(_count_log_pmf(d, jnp.asarray([1.0]), trailing_ndims=1))

def mean_relaxed_count(log_rate):
    d = dist.GammaCount(concentration=jnp.asarray(CONCENTRATION), rate=jnp.exp(log_rate))
    u = jax.random.uniform(jax.random.key(0), (256,))
    return jnp.mean(anchored_relaxed_count(d, u, 0.1, width=8, max_count=16))

argument = jnp.log(jnp.asarray(RATE))
print(log_pmf(argument), jax.grad(log_pmf)(argument))
print(mean_relaxed_count(argument), jax.grad(mean_relaxed_count)(argument))
```

```
-45.589123 nan
0.0 nan
```

## The mechanism

`_gamma_count_log_pmf` needs `log(A - B)` for two incomplete-gamma tails, and
computes it as `logdiffexp(log A, log B)`, which is
`log A + log1p(-exp(log B - log A))`.

Let `B` underflow. In float32 the smallest normal number is `1.18e-38`, so a
tail of `1e-45` stores as exactly `0.0`.

* Forward: `log B` is `-inf`, `exp(log B - log A)` is `0`, `log1p(0)` is `0`,
  and the result is `log A`. That is the correct limit: if `B` is negligible
  then `log(A - B)` is `log A`.
* Backward: reverse mode does not know `B` was negligible. It applies
  `d log(y) / dy = 1 / y` at `y = 0`, gets `inf`, and multiplies it by the zero
  sensitivity of the result to `log B`. `0 * inf` is NaN, and the NaN then
  propagates through every term it touches.

So the value is a limit that JAX evaluates correctly, and the derivative is an
indeterminate form that JAX evaluates as NaN.

Guarding after the operation does not help, because both `jnp.where` branches
are evaluated and differentiated and `0 * NaN` is still NaN. The value has to be
made safe before it reaches `jnp.log`. The function already applies that
discipline to the lower/upper branch selection — that is what the four `safe_*`
values are for. It does not apply it to underflow.

## Why this is easy to miss

Three things hide it.

1. **The value is correct.** Any check on the loss, the relaxed sample, or the
   log density passes.
2. **`eval_and_stable_update` is doing its job.** It keeps the old parameters on
   a non-finite gradient, which is the right behaviour and which converts a
   crash into a silent no-op.
3. **The trigger is a fitted quantity, not a configured one.** The axis that
   matters is the GammaCount *concentration*. `AutoMAPProposal` initializes it
   at one, where nothing is wrong, and then the proposal objective pushes it up,
   because an underdispersed count proposal is what lowers importance-weight
   variance. Whether the relaxed count's gradient is finite, at rate `1e-3` in
   float32:

   | concentration | 1.00 | 1.65 | 2.72 | 4.48 |
   |---|---|---|---|---|
   | `width = 2` | yes | yes | yes | NaN |
   | `width = 4` | yes | yes | NaN | NaN |
   | `width = 8` | yes | NaN | NaN | NaN |

   No recurrence window survives a concentration the fitter will reach, so
   narrowing `width` only postpones the failure past a short smoke run. I lost
   a day to that.

Enabling `jax_enable_x64` moves the boundary rather than removing it: at rate
`1e-3`, concentration 7.4 is past it in float32 and 54.6 is past it in float64,
both with a finite value and a NaN derivative. Anything that probes for this
defect has to probe far enough out to trigger at either precision.

## Why a local floor is not the whole fix

I floored the four tails at `jnp.finfo(...).tiny` before `jnp.log` and added a
`_safe_logdiffexp` for the case where both operands land on the floor together
(`logdiffexp(x, x)` is `-inf` with a `0 / 0` derivative). That removes every
NaN I can produce. It also makes visible that the surrounding recurrence has the
same saturation problem in three more places, each repaired differently:

1. **The ratio recurrence goes flat.** After the floor, `_pmf_ratio_down`
   returns `1 / tiny` at the boundary step and then exactly `1.0` for every step
   beyond it, because both log PMFs sit on the floor and the existing
   `jnp.where(jnp.isnan(...), 0.0, ...)` guard turns the difference into zero.
   The window past the underflow point is therefore flat, and carries neither
   mass nor gradient. Forward this is harmless, since the mass really is
   negligible; it does mean the effective window is narrower than `width`, in a
   way that depends on the parameters and is not reported.

2. **`_count_cdf` saturates to exactly 0 or 1 with a zero derivative.**
   `center_cdf` is *not* wrapped in `stop_gradient` — only the binary anchor is
   — so it feeds `sigmoid((u - cdf) / eta)` directly. At concentration 54.6 and
   rate `1e-3` the CDF is `1.0` and its gradient is `-0`. That is a silent loss
   of gradient signal rather than a NaN, but it has the same root cause.

3. **The guards are inconsistent.** `_pmf_ratio_down`'s GammaCount branch uses
   `jnp.where(jnp.isnan(...))`; its `ZeroInflatedProbs` branch uses a
   `max_log = log(finfo.max) - 2` clamp, with a comment explaining exactly this
   hazard; `anchored_relaxed_count` uses `jnp.clip(cdf, 0.0, 1.0)`. Three
   different ad-hoc repairs of one problem, and the GammaCount branch is the one
   that is missing the clamp its neighbour already has.

There is also a design question a floor forces but does not answer.
`_gamma_count_log_pmf` is reached both from the recurrence *and* from
`_idx_spec`, that is from `SmoothedCount.log_prob`, which is a density in the
objective. Flooring it changes that density in the far tail — by at most
0.05 nats, at log densities within about eight nats of `log(finfo.tiny)` — and
it is not obvious that the recurrence's floor and the density's floor should be
the same number.

## A third issue: the zero-inflation indicator's continuous extension removes the spike

This one is a design decision rather than a numerical defect, and it is the
same decision as the paragraph above -- what the density should be *between*
the integers -- in a place where it costs 12 nats instead of 0.05.

`_idx_spec`'s `ZeroInflatedProbs` branch needs a continuous stand-in for
`1[k == 0]`, and uses

```python
bump = jnp.clip(1.0 - k, 0.0, 1.0)
```

which is exact on the integers, as the comment says. Off them it spreads the
gate's entire point mass **linearly across the whole unit interval**. Since the
gate carries essentially all the probability, the smeared spike then hides the
base family completely on `(0, 1)`. Measured for
`ZeroInflatedPoisson(gate=1 - 2/230400, rate=0.5)` under
`SmoothedCount.log_prob`:

| `a` | relaxed ZIP | smeared spike, `log(1 - a)` | base family + `log(1 - gate)` |
|---|---|---|---|
| 0.001 | -0.0010 | -0.0010 | -12.15 |
| 0.1 | -0.1054 | -0.1054 | -12.17 |
| 0.5 | -0.6931 | -0.6931 | -12.38 |
| 0.9 | -2.3026 | -2.3026 | -12.74 |
| 0.999 | -6.9051 | -6.9078 | -12.85 |
| 1.0 | -12.8451 | -- | -12.85 |
| 2.0 | -14.2314 | -- | -14.23 |

So the relaxed density is the smeared spike on `(0, 1)`, to four decimals, and
the base family shifted down by `log(1 - gate) = -11.654` from `a = 1` upward,
to within 0.0025 nats. The spike is not lost; it is stretched into a ramp that
covers the unit interval, and the slab survives unchanged behind a wall at 1.

**Why that matters.** Zero inflation exists to separate *whether* a site fires
from *how much* it carries, so that an occurrence costs the same whether its
contribution is large or tiny. Under this extension it does the opposite. The
marginal price is `1 / (1 - a)` nats per unit, so **1.00 nats per unit near
zero**, against **11.77** for a plain Poisson at a matched expected count.
Relaxing a zero-inflated count is therefore about twelve times *cheaper* per
unit than relaxing the Poisson it was meant to sparsify, and the entire gate
charge sits in a cliff immediately below `a = 1` -- the derivative of
`SmoothedCount.log_prob` at `a = 1.0` is `-189458`. For a first-order method the
shape is backwards: the region coefficients actually occupy is a gentle
downhill slope that invites faint marks, and the barrier that should stop them
is parked where nothing goes.

**This is not an internal inconsistency, which I first thought it was.**
`_count_log_pmf`'s branch for the same family uses a hard `jnp.where(value == 0,
...)`, which disagrees with the bump by up to 12 nats off the integers -- but it
is only ever *evaluated* at integers. `_count_anchor_binary` searches over
integers, so `center_float` is integer-valued, and `_pmf_ratio_down` is
documented as taking integer `value`. The pmf and the continuous extension
agree wherever both are defined, which is what correctness requires.

**Suggested resolution.** Make the width of the bump explicit rather than
implicitly 1:

```python
bump = jnp.clip(1.0 - k / zero_width, 0.0, 1.0)
```

`zero_width = 1.0` reproduces today's behaviour exactly, and a small value
recovers the fixed per-occurrence charge, at the price of a steeper density near
zero. Whatever the default, the docstring should say that zero inflation confers
no sparsity in the relaxed density at width 1, because a user reaching for
`ZeroInflatedPoisson` is almost always reaching for exactly that sparsity.

## A second, independent defect: `max_count` truncates silently

`_count_anchor_binary` searches `[0, max_count]`, so a count above that bound
cannot be anchored. The relaxation then returns a wrong value, with a finite
gradient and no warning. Poisson-equivalent GammaCount (concentration 1),
`width = 8`, mean over 20,000 uniforms:

| rate | true mean | relaxed `E[z]`, `max_count = 16` | relaxed `E[z]`, `max_count = 256` |
|---|---|---|---|
| 8 | 8.0 | 7.97 | — |
| 16 | 16.0 | 15.95 | 15.97 |
| 20 | 20.0 | 19.70 | 19.97 |
| 30 | 30.0 | **24.59** | 29.97 |
| 64 | 64.0 | **25.00** | 63.97 |

The failure begins near `max_count + width` and then pins. Raising `max_count`
fixes it exactly, so this is purely the bound. Since `max_count` looks like a
performance knob — the docstring calls it an "upper search bound" — it is easy
to set it too low and never learn. An error, or at least a saturation flag on
the return, would help. So would documenting the requirement as something like
`max_count` above a high quantile of the count distribution rather than a
number chosen for speed.

## Suggested direction

The window is currently reconstructed in linear space: `cumprod` of PMF ratios
for the masses, `cumsum` of masses for the CDF, anchored on a CDF binary search.
Every stage of that has a regime where an underflowed quantity has to be
repaired by hand, which is why the guards accumulated.

Carrying the window in log space would remove the class rather than the
instance. The masses become a `cumsum` of log ratios, with no `cumprod`;
underflow becomes a large negative number instead of a zero, so `-inf` never
appears in an intermediate and there is nothing to guard.

The one place that does not translate directly is the smoothing step itself.
`sigmoid((u - cdf) / eta)` needs `u - cdf` as a difference of probabilities, not
as a ratio, so the CDF has to come back to linear space at that point. That is
doable — accumulate `log_cdf` with `logaddexp` and exponentiate once, or form
`u - cdf` through a `log1p`/`expm1` identity — but it is a real design decision
about where the boundary sits, and it is why I am filing this rather than
sending only the patch.

## Interim patch

`gamma-count-log-pmf-nan-gradient.patch` in this directory floors the tails
before `jnp.log` and adds `_safe_logdiffexp` (`.mbox.patch` is the same change
in `git am` format). Verified over 7 concentrations, 5 rates, and 3 window
widths, against the unpatched function:

| | float32 | float64 |
|---|---|---|
| NaN relaxed-count gradients, before | 75/105 | 48/105 |
| NaN relaxed-count gradients, after | 0/105 | 0/105 |
| worst log-PMF difference where the old value is representable | 1.5e-5 | 5.6e-9 |
| worst gradient difference where both are finite | 0 | 0 |

The last row is the one that matters for accepting it: where the current code
already worked, the patched version is identical. It only changes behaviour
where the old answer was NaN. It does not address the flat ratio tail, the
saturated CDF, the inconsistent guards, or `max_count`.

## Environment

* `numpyro` `0.21.0`, `esennesh/numpyro` branch `codex/map-proposal`, commit
  `8a11fcafb41576be25f347a379816ac5754ddd28`
* `jax` 0.7.x, CPU and CUDA, macOS and Linux, float32 by default
* Reached through `AutoMAPProposal` on a model with a
  230,400-coordinate Poisson count field at rate `1.7e-5`
