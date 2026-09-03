# Online CAPTCHA inference with QEM

This branch uses the experimental QEM implementation at
[`feature/qem`](https://github.com/esennesh/numpyro/tree/feature/qem).
It performs observation-specific, gradient-free moment matching rather than
training an amortized encoder.

## Restricted count support

The full activation tensor has $80\cdot80\cdot36=230{,}400$ coordinates. A
single QEM alternative for that entire tensor almost never lands near a useful
CAPTCHA explanation. Before inference, alpha-normalized matched filtering of
image darkness against dictionary masks selects a small set $S$ of
$(y,x,k)$ candidates. Several glyph identities are retained at each
non-maximum-suppressed spatial location.

The candidate model is the full model conditioned on $a_i=0$ for
$i\notin S$:

$$
p_\theta(a_S,a_{\neg S}=0)
=\prod_{i\in S}\operatorname{Poisson}(a_i;\lambda)
 \exp\{-\lambda(|\mathcal I|-|S|)\}.
$$

The omitted-count factor is constant in the online latents. MPIW currently
cannot pack that auxiliary plate factor, so the QEM contraction hides it. This
changes the reported `log P_MP(x)` by a constant but changes neither normalized
importance weights nor posterior moments.

The model still assigns every candidate its original homogeneous rate
$\lambda$. At each selected location, candidates are ordered by their alpha-mask
score. The initial proposal assigns a fraction $1-\rho$ of the configured count
mass to the best class at each location and distributes the exploration fraction
$\rho$ over the other retained classes:

$$
m_i^{(0)}=
\begin{cases}
(1-\rho)M/L,&i\text{ is a location's best class},\\
\rho M/(|S|-L),&\text{otherwise},
\end{cases}
$$

where $M$ is the configured initial count mass and $L$ the number of spatial
locations. Without this proposal-only initialization, nearly every first-round
count vector is zero because the model rate was originally spread over 230,400
sites. The original $p_\theta$ is unchanged.

## QEM update

For

$$
z=(a_S,c_0,r,u),
$$

QEM uses

$$
q_\phi(z)
=q_\phi(a_S)q_\phi(c_0)q_\phi(r)q_\phi(u).
$$

With $K$ proposals per site, MPIW contracts the $K^4$ combinations and returns
the marginal self-normalized weights $\widetilde w_{ik}$. Each E-step estimates

$$
\widehat m_i=\sum_{k=1}^K\widetilde w_{ik}T_i(z_{ik}),
$$

and the running state is

$$
m_{i,t}=\lambda_t m_{i,t-1}+(1-\lambda_t)\widehat m_i.
$$

The M-step rebuilds $q_\phi$ from these mean parameters. Poisson is supplied by
the QEM branch. This project adds exact Beta sufficient statistics

$$
T(c)=(\log c,\log(1-c))
$$

and treats each GMRF as the fixed-covariance exponential family whose sufficient
statistic is the field itself. Thus QEM updates GMRF means while retaining their
sparse second-order covariances.

## Run

Synchronize this branch's dependency and fit one image:

```shell
uv sync
uv run python scripts/online_qem.py path/to/captcha.png
```

The defaults use twelve candidate sites, four alternatives per latent site,
and twenty QEM iterations. Increase `--num-samples` cautiously: the conceptual
Cartesian grid grows as $K^4$, even though serial contraction bounds some
intermediates.
