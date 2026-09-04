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
$r$, and image-level warp velocity $u$, the full latent state is

$$
z=(a,c_0,r,u).
$$

The unsmoothed inference target is therefore the original joint density,

$$
\gamma_\theta(z;x)=p_\theta(x,z)
=p_\theta(x\mid z)p_\theta(a)p_\theta(c_0)p_\theta(r)p_\theta(u).
$$

There is also no image-derived warm-start. `AutoMAPProposal` uses its generic,
support-aware initialization in the unconstrained representation. Removing
the warm-start prevents an external matched-filter decision from deciding
which sites begin with appreciable count mass. It does not remove the
homogeneous Poisson occupancy cost from the target.

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

is a basic degeneracy diagnostic. An ESS near one, or one normalized weight
near one, says that more particles or a better proposal are needed; it does not
by itself certify approximation quality.

## Run

Synchronize this branch's dependency and fit one image:

```shell
uv sync
uv run python scripts/online_map_proposal.py path/to/captcha.png
```

The default optimizes all 230,400 count coordinates for an $80\times80$ image
and 36-template dictionary. It uses at most two hundred Adam steps in each
fit, eight dispersion-fitting particles, and sixty-four importance particles.
Reducing `--map-max-steps` and `--proposal-max-steps` is useful for a compile
smoke test but not for assessing recovery.
