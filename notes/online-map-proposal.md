# Online CAPTCHA inference with a MAP proposal

This branch uses the experimental
[`AutoMAPProposal`](https://github.com/esennesh/numpyro/tree/codex/map-proposal)
to fit an observation-specific proposal and then correct it with
self-normalized importance sampling.

## Restricted count support

The full activation tensor has $80\cdot80\cdot36=230{,}400$ coordinates.
Optimizing every count coordinate together with the continuous fields would be
impractical. Before optimization, an alpha-mask matched filter selects a small
set $S$ of $(y,x,k)$ candidates.
Spatial non-maximum suppression keeps separated locations, while several glyph
identities remain possible at each location.

The candidate model is the full model conditioned on $a_i=0$ for
$i\notin S$:

$$
p_\theta(a_S,a_{\neg S}=0)
=\prod_{i\in S}\operatorname{Poisson}(a_i;\lambda)
 \exp\{-\lambda(|\mathcal I|-|S|)\}.
$$

Unlike a heuristic replacement prior, the retained factor means this is the
original joint density on the restricted support. Candidate selection is still
an inference approximation: a true glyph omitted from $S$ cannot be recovered.
The canonical texture and whole-image affine-free warp remain latent in full.

The original homogeneous model rate is only about $4/230{,}400$ per site. An
optimizer initialized from that rate starts effectively on the all-zero count
boundary, even though its search now contains only a few sites. At each selected
location, candidates are ordered by their alpha-mask score. The guide puts the
configured expected count mass on the best initial class at each location and
puts the other relaxed counts just inside their positive support. It initializes
the color from the darkest image percentile and starts texture and velocity at
zero. This is one coherent, unwarped CAPTCHA explanation rather than a
translucent superposition of every class alternative. It changes only the
optimizer's initial point; every objective evaluation still uses the original
$p_\theta(a_S,a_{\neg S}=0)$ above.

## Fitted proposal

For the restricted latent vector

$$
z=(a_S,c_0,r,u),
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

The default uses twelve candidate sites, at most two hundred Adam steps in each
fit, eight dispersion-fitting particles, and sixty-four importance particles.
Reducing `--map-max-steps` and `--proposal-max-steps` is useful for a compile
smoke test but not for assessing recovery.
