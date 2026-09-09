# `diag_sgd`: the zero-inflation continuation removes the sparsity zero inflation is for

*Draft issue for `esennesh/numpyro`, branch `codex/map-proposal`, commit
`4ca6f62`. Not filed yet.*

*Supersedes an earlier draft about `_gamma_count_log_pmf`'s NaN gradient at an
underflowed Gamma-count tail. That defect is **fixed** on this commit: the four
incomplete-gamma tails are now floored at `finfo.tiny` before `jnp.log` inside
`GammaCount.log_prob`, `_gamma_count_log_pmf` is gone from `diag_sgd`, and every
`_idx_spec` family delegates to its distribution's own `log_prob`, so there is
one definition per family rather than two. Re-measured: the old reproduction
returns a gradient of 7.3825 where it returned NaN, and all 18 cells of the old
(concentration, width) failure grid are finite. Two items from that draft
survive in weakened form and are recorded at the end.*

## Summary

`_idx_spec`'s `ZeroInflatedProbs` branch continues `1[k == 0]` to non-integer
relaxed counts as `bump = clip(1 - k, 0, 1)`. This is a deliberate choice, it
agrees with `log_prob` on the integers, and `test_idx_spec_zero_inflated_smooths_
the_indicator` pins it.

Its consequence looks unintended. Because the gate carries nearly all the
probability, a linear bump spreads the point mass across the whole unit
interval, so the relaxed density on `(0, 1)` is the smeared spike and not the
base family. The marginal price becomes `1 / (1 - a)` nats per unit -- **1.00
nats per unit near zero** -- where the un-inflated Poisson at a matched expected
count charges **11.77**. Relaxing a zero-inflated count is therefore about
twelve times *cheaper* per unit than relaxing the family it was meant to
sparsify, and a user reaching for `ZeroInflatedPoisson` is usually reaching for
exactly that sparsity.

The whole gate charge is still in the density. It is compressed into the last
thousandth before `a = 1`, where a first-order method never arrives.

## Reproduction

```python
import math
import jax, jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.contrib.diag_sgd import SmoothedCount

GATE, BASE_RATE, FIELD = 1.0 - 2.0 / 230400.0, 0.5, 230400.0

zero_inflated = SmoothedCount(
    dist.ZeroInflatedPoisson(gate=jnp.asarray(GATE), rate=jnp.asarray(BASE_RATE)),
    eta=0.1, width=4, max_count=8,
)
# the same expected total count over the field, without the gate
plain = SmoothedCount(
    dist.Poisson(jnp.asarray(1.0 / FIELD)), eta=0.1, width=4, max_count=8
)
price = lambda d, a: float(-jax.grad(lambda x: d.log_prob(x))(jnp.asarray(a)))

for a in (0.001, 0.1, 0.5, 1.0):
    print(a, float(zero_inflated.log_prob(jnp.asarray(a))),
          math.log1p(-a) if a < 1 else None, price(zero_inflated, a))
print("plain:", price(plain, 0.001))
```

## Measurements

`gate = 1 - 2/230400`, `rate = 0.5`; the plain Poisson at rate `1/230400` has
the same expected total of 1.000 over a 230,400-site field and the same price
for one whole occurrence to within half a nat. `log(1 - gate) = -11.6544`.

| `a` | relaxed ZIP | `log(1 - a)` | base family + `log(1-gate)` | plain Poisson | ZIP price, nats/unit |
|---|---|---|---|---|---|
| 0.000 | 0.0000 | 0.0000 | −12.1544 | 0.0000 | 0.50 |
| 0.001 | −0.0010 | −0.0010 | −12.1545 | −0.0118 | 1.00 |
| 0.010 | −0.0101 | −0.0101 | −12.1557 | −0.1178 | 1.01 |
| 0.100 | −0.1054 | −0.1054 | −12.1739 | −1.1849 | 1.11 |
| 0.500 | −0.6931 | −0.6931 | −12.3802 | −6.0530 | 2.00 |
| 0.900 | −2.3026 | −2.3026 | −12.7393 | −11.0738 | 10.00 |
| 0.999 | −6.9051 | −6.9078 | −12.8465 | −12.3348 | 997.38 |
| 1.000 | −12.8451 | -- | −12.8476 | −12.3476 | **189458.03** |
| 1.500 | −13.4763 | -- | −13.4788 | −18.8060 | 1.40 |
| 2.000 | −14.2314 | -- | −14.2339 | −25.3883 | 1.62 |
| 3.000 | −16.0231 | -- | −16.0256 | −38.8345 | 1.95 |

Reading the columns:

* On `(0, 1)` the relaxed density equals `log(1 - a)` to four decimals. It is
  the smeared spike.
* The base family sits about 12 nats below it across that whole interval, so it
  contributes nothing there. The spike is not lost; it is stretched over the
  slab.
* From `a = 1` upward the relaxed density equals the base family shifted down by
  `log(1 - gate)`, to within 0.0025 nats. The slab survives intact, behind a
  wall.
* The plain Poisson column is *lower* than the ZIP column for every `a < 0.9`.
  Zero-inflating this site makes a faint count cheaper than not inflating it.

## Why the shape is the problem

The price of a faint mark should not depend on how faint it is -- that is what
the gate is for, and what separates "does this site fire" from "how much does it
carry". Under this continuation the price is proportional to the amount, which
is the behaviour zero inflation is chosen to avoid.

For a first-order optimizer the shape is also backwards. The region a
coefficient occupies in practice is a gentle downhill slope that rewards adding
more faint marks, and the barrier that should stop them is at `a = 1`, an order
of magnitude beyond where any coefficient sits. The derivative there is
`-189458`, so the barrier is not merely misplaced but numerically hostile.

Concretely, from a convolutional sparse-coding model with a 230,400-coordinate
count field: 1,682 coefficients holding a total amplitude of 2.0 score **1.2
nats better** than two coefficients at 1.0 under the plain Poisson. Zero
inflation is the natural repair and would make the diffuse solution cost about
19,600 nats more than the sparse one. Under this continuation it would instead
make the diffuse solution *cheaper* still.

## What this is not

It is **not** an inconsistency between two code paths, which is what I first
took it for. `_count_log_pmf`'s branch for the same family uses a hard
`jnp.where(value == 0, ...)` and disagrees with the bump by up to 12 nats off
the integers -- but it is only ever *evaluated* at integers.
`_count_anchor_binary` searches over integers, so `center_float` is
integer-valued, and `_pmf_ratio_down` is documented as taking integer `value`.
The pmf and the continuation agree wherever both are defined, which is what
correctness requires. The continuation is a free choice, and this issue is about
which choice to make.

## Suggested resolution

Make the width of the bump explicit rather than implicitly 1:

```python
bump = jnp.clip(1.0 - k / zero_width, 0.0, 1.0)
```

`zero_width = 1.0` reproduces today's behaviour exactly, so nothing breaks, and
a small width recovers a per-occurrence charge that does not shrink with the
amount. The trade is a steeper density near zero, and where to sit on it is a
genuine modelling decision -- which is the argument for exposing it rather than
fixing it at either end.

Whatever the default, the docstring should say that at width 1 the relaxed
density gives zero inflation no sparsity advantage over its own base family. The
present docstring and comment explain that the bump is exact on the integers and
continuous in between, both true, and neither hints at this.

## The two items carried over from the previous draft

* **`_count_cdf` still saturates with a zero derivative.** At concentration 54.6
  and rate `1e-3`, `_count_cdf(d, 0.0)` is exactly 1.0 and its derivative with
  respect to `log rate` is `-0`. The consequence that mattered has been
  addressed -- a survival-function path was added and
  `test_anchored_relaxed_count_gradient_survives_cdf_saturation` covers it -- so
  this is now a latent sharp edge rather than a live defect.
* **`max_count` truncation is now detectable but still not announced.** The
  truncation itself is unchanged: Poisson-equivalent GammaCount, `width = 8`,
  mean over 20,000 uniforms at `max_count = 16` gives 24.59 for a true mean of
  30 and 25.00 for a true mean of 64. The new public `count_anchor_saturates`
  tracks it well -- 0.3% of draws saturating corresponds to an error of 0.04,
  99.6% to an error of 5.41 -- but nothing warns unless you think to ask, and
  `max_count` still reads like a performance knob.

Also still open from that draft: carrying the recurrence window in log space,
which would remove the class of underflow repairs rather than the instances.

## Environment

* `numpyro` `0.21.0`, `esennesh/numpyro` branch `codex/map-proposal`, commit
  `4ca6f62`
* `jax` 0.10.2, CPU, float32 by default
* Reached through `AutoMAPProposal` on a model with a 230,400-coordinate count
  field at an expected total count of 1
