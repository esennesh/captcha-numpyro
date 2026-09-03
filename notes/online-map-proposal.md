# Online CAPTCHA inference with a MAP proposal

This branch uses the experimental
[`AutoMAPProposal`](https://github.com/esennesh/numpyro/tree/codex/map-proposal)
to fit an observation-specific proposal and then correct it with
self-normalized importance sampling.

## Restricted count support

The full activation tensor has $80\cdot80\cdot36=230{,}400$ coordinates. A
dense BFGS solve over that count tensor and the continuous fields would require
an impractical dense inverse-Hessian approximation. Before optimization, an
alpha-mask matched filter selects a small set $S$ of $(y,x,k)$ candidates.
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

It then fits a factorized proposal $q_\phi(z\mid x)$ around that mode by
minimizing a fixed-randomness Monte Carlo estimate of

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

## Importance correction

The implementation freezes the fitted guide before drawing particles; calling
`AutoMAPProposal` directly would refit it for every call. For each particle,

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

If BFGS stops with a non-finite or exponentially overflowing proposal
log-parameter, that coordinate falls back to the guide's initial full-support
dispersion. This changes only $q_\phi$, not $\gamma_\theta$, so the importance
ratio remains the required correction. The script reports convergence of both
optimization stages so that such a smoke run is not mistaken for a fitted
posterior.

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

The default uses twelve candidate sites, twenty BFGS iterations, eight
dispersion-fitting particles, and sixty-four importance particles. Reducing
`--maxiter` is useful for a compile smoke test but not for assessing recovery.
