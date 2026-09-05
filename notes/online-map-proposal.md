# Online CAPTCHA inference with a MAP proposal

This branch uses the experimental
[`AutoMAPProposal`](https://github.com/esennesh/numpyro/tree/codex/map-proposal)
to fit an observation-specific proposal and then correct it with
self-normalized importance sampling.

## Full count support

The latent count field retains all image locations and all glyph identities,

$$
a\in\mathbb N_0^{H\times W\times K}.
$$

For an $80\times80$ image and a 36-template dictionary, this is
$80\cdot80\cdot36=230{,}400$ count coordinates. No matched filter,
non-maximum suppression, or observation-dependent support restriction is
applied. Consequently every location and glyph identity can receive posterior
mass.

Writing the remaining latents as the baseline color $c_0$, canonical texture
$r$, image-level warp velocity $u$, paper logit $m_b$, coarse paper field $f$,
and per-pixel observation precision $\lambda$, the full latent state is

$$
z=(a,c_0,r,u,m_b,f,\lambda).
$$

The unsmoothed inference target is therefore the original joint density,

$$
\gamma_\theta(z;x)=p_\theta(x,z)
=p_\theta(x\mid z)\,p_\theta(\lambda\mid a,u)\,
p_\theta(a)p_\theta(c_0)p_\theta(r)p_\theta(u)p_\theta(m_b)p_\theta(f).
$$

The three fields after $u$ are new; `configs/model/poisson_convsc_gmrf`
replaces the fixed white paper and the scheduled Student-t with a paper GMRF
and a Gamma-Normal likelihood. Section 26 of `notes/poisson-convsc-design.md`
records why: under fixed white paper, tiling the canvas with glyphs was the
target's global mode, so a *better* MAP fit produced a worse picture.

There is also no image-derived warm-start. Removing it prevents an external
matched-filter decision from deciding which sites begin with appreciable count
mass, and it does not remove the homogeneous Poisson occupancy cost from the
target.

The count field is nevertheless initialized at its own prior mean rather than
at `AutoMAPProposal`'s generic support-aware default, and that is an optimizer
choice with no effect on $\gamma_\theta$: it uses the model's rate and nothing
derived from the observation. See "Three bugs the diagnostics were hiding"
below for why the default cannot be used here.

## Fitted proposal

For the full latent state

$$
z=(a,c_0,r,u),
$$

the guide first replaces count sites with the DSGD relaxation at temperature
$\eta$ and finds

$$
\widetilde z^*
=\mathop{\mathrm{argmax}}_{\widetilde z}
  \log\gamma_{\theta,\eta}(\widetilde z;x).
$$

It then fits a factorized proposal $q_\phi(z\mid x)$ around that mode. Both
phases use an iterative NumPyro or Optax optimizer. For example, Adam updates a
parameter vector $v_t$ from a stochastic loss gradient $g_t$ as

$$
\begin{aligned}
g_t &= \nabla_v \widehat{\mathcal L}_t(v_t),\\
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t,\\
r_t &= \beta_2r_{t-1}+(1-\beta_2)g_t^2,\\
v_{t+1} &= v_t-\rho
\frac{\widehat m_t}{\sqrt{\widehat r_t}+\epsilon}.
\end{aligned}
$$

For the MAP phase, $v$ is the unconstrained relaxed latent vector and

$$
\mathcal L_{\mathrm{MAP}}(v)
=-\log\gamma_{\theta,\eta}(T(v);x),
$$

where $T$ maps unconstrained values to each site's support. For the dispersion
phase, $v=\phi$ and each step uses fresh Monte Carlo draws to estimate

$$
\mathbb E_{q_\phi(\widetilde z\mid x)}
\left[
  \log q_\phi(\widetilde z\mid x)
  -\log\gamma_{\theta,\eta}(\widetilde z;x)
\right].
$$

Continuous sites use transformed Normal factors centered at the MAP. Count
sites use exact GammaCount factors after fitting, so the particles evaluated by
the original model contain integer counts.

The two maximum-step settings are safety caps. After each interval of $K$
steps, the fitter computes the average loss $\overline{\mathcal L}_j$. An
improvement over the best previous interval $b_{j-1}$ is meaningful when

$$
b_{j-1}-\overline{\mathcal L}_j
>\tau(1+|b_{j-1}|).
$$

The phase reports convergence after $P$ consecutive checks without a
meaningful improvement. Its returned result also contains the complete loss
history and actual step count.

## Importance correction

The implementation calls `AutoMAPProposal.fit` exactly once. The returned
`MAPProposalResult` holds the MAP locations, proposal parameters, loss
histories, step counts, and convergence flags. Subsequent guide calls only draw
from that reusable fitted state and perform no optimization. For each particle,

$$
z^{(s)}&\sim q_\phi(z\mid x),\\
\ell_s&=\log\gamma_\theta(z^{(s)};x)
       -\log q_\phi(z^{(s)}\mid x),\\
\widetilde w_s&=
\frac{\exp\ell_s}{\sum_r\exp\ell_r}.
$$

The reported image is the self-normalized estimator

$$
\widehat{\mathbb E}_{\pi_\theta}
[\mu_\theta(z)\mid x]
=\sum_s\widetilde w_s\mu_\theta(z^{(s)}).
$$

The script reports convergence and step counts for both optimization stages so
that a capped smoke run is not mistaken for a fitted posterior. The loss
histories are available on the inference result as `map_losses` and
`dispersion_losses`.

The effective sample size

$$
\operatorname{ESS}=\frac{1}{\sum_s\widetilde w_s^2}
$$

is a basic degeneracy diagnostic, and here it is mostly a statement about
dimension rather than about fit quality. The latent state is about 233,000
dimensional. A count site carrying a glyph has $\lambda_{\text{prior}}
= 1.7\times10^{-5}$ against a proposal rate near one, so each unit change in
its count moves the log weight by
$\log(\lambda_{\text{prior}}/\lambda_{q})\approx-11$ nats. A factorized
proposal over that many coordinates gives an ESS near one however well it is
fitted, and 64 particles cannot repair it. Read an ESS of one as "this is a
reweighting of essentially one mode", which does not say the mode is wrong.

## Three bugs the diagnostics were hiding

Measured 2026-09-04 against `data/examples/0000_LJ`, which reported an expected
total glyph count of 342,414 and an ESS of 1.00 out of 64.

### The MAP phase could not move, because its gradient was zero

`AutoMAPProposal` falls back to `init_to_uniform` for count sites.
`SmoothedCount.support` is `positive`, so the unconstrained representation is a
logarithm and the default draws $\exp(U(-2,2))$ — a mean of 1.81 — at *every*
coordinate. For $80\times80\times36$ that is **417,081 glyph stamps**.

Counts reach the image only through the optical depth, and 417,081 stamps put
$\tau$ at $7.1\times10^4$ per pixel. Then $A = 1-e^{-\tau}$ is 1.000000 and
$\mathrm dA/\mathrm d\tau = e^{-\tau}$ is **exactly zero in float32** at
every pixel. The likelihood contributes no gradient to any of the 230,400 count
coordinates; only the Poisson prior pushes, uniformly downward, and Adam walks
every coordinate down by one step size per step.

The reported total confirms it exactly. Twenty steps at 0.01 is 0.197 in log
space, and $342{,}414/417{,}081 = 0.821 = e^{-0.197}$. The published run was
the initialization minus twenty Adam steps and nothing else.

`init_to_count_mean` starts the field at the model's own rate, where the same
render has $\tau = 0.68$ per pixel and $\mathrm dA/\mathrm d\tau$ up to 0.81.

### The step budget was two orders of magnitude short

From the prior mean at $\log(4/230400) = -10.96$, raising the handful of true
sites to a count near one is a journey of about 11 in log space, and Adam moves
at most one step size per step. At the former 0.01 that needs at least 1,100
steps; the config allowed 200 and the notebook 20. `configs/online/map_proposal`
now uses 0.05 for 1,500 steps.

### The proposal phase never had a valid gradient at all

`_gamma_count_log_pmf` evaluates

$$
\log\left[P(\alpha z,\alpha\lambda)-P(\alpha(z+1),\alpha\lambda)\right]
$$

by taking the logarithm of each incomplete-gamma tail and differencing them
with `logdiffexp`. It switches between the lower and upper tails to avoid
cancellation, and it guards that *branch selection* with the double-`where`
idiom JAX requires. It does not guard underflow. Once a tail is smaller than
the smallest representable float, `jnp.log` of it is $-\infty$; `logdiffexp`
still returns the right **value**, and its **derivative** is NaN.

That combination is the worst one. `eval_and_stable_update` discards every
NaN update, so the fitted dispersions silently never move, and the only visible
symptom is a loss history of NaN.

The axis that matters is the fitted GammaCount **concentration**, not the rate
and not the recurrence window. It starts at one, and the proposal objective
pushes it up, because an underdispersed count proposal is what reduces
importance-weight variance. Whether the relaxed count's gradient was finite at
rate $10^{-3}$, in float32:

| concentration | 1.00 | 1.65 | 2.72 | 4.48 |
|---|---|---|---|---|
| $W=2$ | yes | yes | yes | NaN |
| $W=4$ | yes | yes | NaN | NaN |
| $W=8$ | yes | NaN | NaN | NaN |

So **no window width survives a concentration the fitter was always going to
reach.** A first attempt at this note blamed float32 overflow in the tail
`cumprod` and prescribed `width: 4`; that reading was wrong. It happened to
predict the boundary of a short smoke run, because a narrow window reaches the
underflow later, but it only postponed the failure, and the pod run hit it as
soon as the concentration climbed.

`src/inference/count_relaxation.py` keeps the tails in linear space long enough
to floor them at the smallest positive *normal* float before taking a
logarithm, so `logdiffexp` never receives an infinity. Flooring at the smallest
normal rather than at exact zero is deliberate: the reciprocal of a denormal is
already $\infty$, and `1/x` appears in the derivative of `log`. The price is a
discrepancy of at most 0.011 nats, confined to log densities below about $-79$,
i.e. probabilities near $10^{-35}$. After the repair every combination up to
concentration 500 and $W=32$ has a finite gradient.

`patch_count_log_pmf` probes for the defect rather than reading a version, so
it becomes a no-op the moment the
[fork](https://github.com/esennesh/numpyro/tree/codex/map-proposal) carries the
floor itself, which is where it belongs.

Enabling float64 does **not** fix this, only postpone it: at rate $10^{-3}$ a
concentration of 7.4 is already past the float32 boundary and 54.6 is past the
float64 one, both with a finite value and a NaN derivative. That matters for
more than advice, because `tests/test_diffeomorphism` enables x64 globally at
import; a defect probe tuned to float32 would decide the bug was absent behind
it and skip the repair. The probe uses a concentration of 500, which underflows
at either precision.

`width` is now chosen for the window's cost rather than for correctness: it is
materialized per site, so 230,400 sites times $2W$ terms is 14.7M floats per
particle at $W=32$ against 3.7M at $W=8$, and the relaxed mean is identical for
every rate up to 4.

### Two reporting artifacts

`termination_check_interval` defaults to 50. The notebook ran 20 steps, so
exactly one convergence check happened and `converged=False` was guaranteed
regardless of the fit. The notebook now warns when the interval exceeds the
steps taken, and asserts that both loss histories are finite.

The notebook also built `MAPProposalCaptchaInference` by hand rather than from
`configs/online/map_proposal`, so it had drifted to a tenth of the script's
step budget. Both now compose the same two configs.

## Run

Synchronize this branch's dependency and fit one image:

```shell
uv sync
uv run python scripts/online_map_proposal.py path/to/captcha.png
```

The default optimizes all 230,400 count coordinates for an $80\times80$ image
and 36-template dictionary, against `configs/model/poisson_convsc_gmrf`. Every
setting comes from `configs/online/map_proposal`; the command-line flags
override only what is passed. Reducing `--map-max-steps` and
`--proposal-max-steps` is useful for a compile smoke test but not for assessing
recovery — the MAP budget in particular is derived from how far the count field
has to travel in log space.
