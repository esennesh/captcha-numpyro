# Diffeomorphic glyph prototype

## Generative construction

For one glyph occurrence, let the latent stationary velocity be

$$
z_v=v\in\mathbb R^{H_v\times W_v\times2}.
$$

The two coordinate channels are ordered `(y, x)` and measured in pixels of the
rendered canonical frame. We place the normalized second-order GMRF prior

$$
\begin{aligned}
A_v &= \eta_v I + \rho_v\mathcal L,\\
Q_v &= A_v^\top A_v,\\
p_\theta(v) &= \mathcal N\!\left(v;0,Q_v^{-1}\otimes I_2\right).
\end{aligned}
$$

The prototype samples this field on a coarse lattice, interpolates it to the
glyph frame, and multiplies it by a sine window that fixes the frame boundary.
These are linear maps of a Gaussian variable, so the resulting full-resolution
velocity is still a Gaussian random field, albeit with zero variance exactly on
the boundary.

## Scaling and squaring

The stationary velocity defines the continuous flow

$$
\frac{d}{dt}\phi_t(u)=v(\phi_t(u)),\qquad \phi_0(u)=u,
$$

whose time-one map is $\phi=\exp(v)$. Write a grid map as a displacement
$\phi(u)=u+d(u)$. If two maps have displacements $d_\phi$ and $d_\psi$, their
composition has displacement

$$
d_{\phi\circ\psi}(u)
=d_\psi(u)+d_\phi\!\left(u+d_\psi(u)\right).
$$

The second term is evaluated by bilinear interpolation. Scaling and squaring
starts with the small Euler map

$$
d_0(u)=\frac{v(u)}{2^J}
$$

and repeatedly self-composes it:

$$
d_{j+1}(u)=d_j(u)+d_j\!\left(u+d_j(u)\right),
\qquad j=0,\ldots,J-1.
$$

The resulting displacement approximates $\exp(v)$. Because the velocity is
stationary, the inverse is obtained without fitting a second field:

$$
\phi^{-1}=\exp(-v).
$$

## Rendering

The renderer uses backward sampling. For a canonical glyph bitmap $g$ and an
output pixel $p$,

$$
\widetilde g(p)=g(\phi^{-1}(p)).
$$

Backward sampling assigns every output pixel a source coordinate and therefore
does not create the holes associated with pushing source pixels forward.
`diffeomorphic_warp` is differentiable almost everywhere with respect to $v$,
so $v$ can participate directly in MAP or variational inference. Because the
model retains $v$ as the latent and treats $\phi=\exp(v)$ as deterministic, its
joint density contains $p_\theta(v)$ and does not require a change-of-variables
determinant for the exponential map.

The continuous construction is diffeomorphic for a sufficiently regular
velocity. Its pixel-grid approximation should nevertheless be checked with

$$
\min_u \det D\phi(u)>0
$$

and with the inverse-consistency residual

$$
\left\|\phi\circ\phi^{-1}-\operatorname{id}\right\|.
$$

Run the example with

```shell
python scripts/diffeomorphic_glyph.py
```

It displays the canonical glyph, sampled velocity, deformed coordinate grid,
backward-warped glyph, and discrete Jacobian determinant.

## Sparse occurrence renderer

This remains a useful correctness reference, but the current prototype direction
uses the fixed-shape whole-image field in the next section.

The reference scene renderer represents the active set explicitly. Occurrence
$i$ has amplitude $a_i$, integer image center $s_i$, dictionary index $k_i$,
and its own velocity $v_i$. Its total closeness-premultiplied ink field is

$$
I(p)
=\sum_{i=1}^N
a_i K_{k_i}\!\left(\exp(-v_i)(p-s_i)\right),
$$

where all four kernel channels are warped together: the first three carry
$\tau c$ and the fourth carries optical depth $\tau$. Their sums therefore
retain the existing renderer's compositing semantics,

$$
\tau(p)=I_4(p),\qquad
\alpha(p)=1-\exp[-\tau(p)],\qquad
c(p)=\frac{I_{1:3}(p)}{\tau(p)}.
$$

The implementation `vmap`s the warp over the $N$ active occurrences and then
scatter-adds the resulting $H_g\times W_g$ patches with canvas-edge clipping.
It costs $O(NH_gW_g)$ storage and work outside the scaling-and-squaring
compositions, rather than materializing an $H\times W\times K$ field of warped
kernels. At zero velocity it agrees exactly with `_stamp`, including its
even-kernel centering convention and overlapping amplitudes.

The current reference renderer deliberately keeps centers on integer image
sites, matching the Poisson count lattice. Velocities and amplitudes are
differentiable. Continuous placement offsets can later be absorbed into each
occurrence's backward sampling grid or handled by the eventual deformable
transposed-convolution implementation.

Run the two-occurrence example with

```shell
python scripts/sparse_diffeomorphic_scene.py
```

## Whole-image foreground flow

The simpler scene model retains the ordinary convolutional renderer and places
one velocity field over the image:

$$
\begin{aligned}
I_0 &= \operatorname{Stamp}(a,K),\\
\phi &= \exp(v),\\
I(p) &= I_0(\phi^{-1}(p)).
\end{aligned}
$$

Only the four foreground ink channels are pulled back through $\phi^{-1}$.
For a fixed paper field $b(p)$, the final mean remains

$$
\mu_\theta(p)
=\alpha_I(p)c_I(p)+[1-\alpha_I(p)]b(p),
$$

so the paper does not move. The velocity latent has a fixed shape independent
of the number of active dictionary coefficients, and `_stamp` remains one
optimized transposed convolution.

### Removing affine components

**Correction, 2026-09-04: the premise below is false for the convolutional
renderer, and only translation should be removed. See "Which modes are actually
redundant" after the derivation.**

Translation, rotation, scale, and shear are redundant with glyph placement and
pose latents. Let the coarse velocity channel have the normalized prior

$$
p_\theta(u)=\mathcal N(u;0,Q^{-1}),
$$

let $R\in\mathbb R^{HW\times hw}$ be the coarse-to-image resize, let $W$ be a
diagonal boundary window, and let $B\in\mathbb R^{HW\times3}$ contain the
normalized image-coordinate functions

$$
B=[\mathbf 1, y, x]
$$

with $y,x\in[-1,1]$. The desired image velocity $v=WRu_\perp$ is affine-free
when

$$
B^\top v=B^\top WRu_\perp=0.
$$

Pulling those constraints back to the coarse lattice gives

$$
C=R^\top WB\in\mathbb R^{hw\times3},
\qquad C^\top u_\perp=0.
$$

Rather than subtracting the ordinary least-squares affine fit, we condition the
GMRF on these three linear constraints per velocity channel. Given an
unconstrained draw $u\sim p_\theta(u)$, define

$$
\begin{aligned}
X &= Q^{-1}C,\\
M &= C^\top X=C^\top Q^{-1}C,\\
u_\perp &= u-XM^{-1}C^\top u,\\
v &= WRu_\perp.
\end{aligned}
$$

The resulting conditional Gaussian has covariance

$$
\operatorname{Cov}(u_\perp)
=Q^{-1}-Q^{-1}C(C^\top Q^{-1}C)^{-1}C^\top Q^{-1}.
$$

The three scalar modes in each of two coordinate channels are the two
translations and the four coefficients of a linear $2\times2$ map, encompassing
rotation, scale, and shear. The sine window additionally makes $v=0$ on the
image boundary. The implementation obtains $C$ with the transpose of automatic
differentiation's resize map, computes $X$ with three sparse
`SecondOrderGaussianMrf.solve_precision` right-hand sides, and solves only the
dense $3\times3$ system $M$.

### Which modes are actually redundant

The premise above holds for `sparse_diffeomorphic_stamp`, which takes a
per-occurrence velocity and therefore does carry a pose latent.
`TexturedDiffeomorphicPoissonConvPlacements` does not. Its only placement
latent is the count field $a$, which supplies a *site index* and nothing else.
Translation is genuinely redundant with that index; rotation, scale, and shear
have no other latent at all, so conditioning them out removes them from the
model.

`data/examples` contains them. Correlating the dictionary against each image
over rotations and scales, rotating the template raises the matched-filter
score by 70% on `0000_LJ` (both glyphs at about 25 degrees), 48% on `0001_JA`
(one glyph at $-20$ degrees, the other at $+5$), 17% on `0003_7Z`, and 16% on
`0004_7D`. The five older examples are upright and gain nothing, which is why
this went unnoticed.

The projection is also worse than a deletion, because a *local* deformation
carries a nonzero global affine moment and the constraint forces a
compensating counter-deformation elsewhere in the image. Least-squares fitting
each target velocity field through the coarse-to-image map and reading off the
prior cost, on the $10\times10$ lattice at `warp_scale = 1.5`:

| target | affine-free | translation-free |
|---|---|---|
| one glyph rotated 25 degrees | 2294 nats | **99 nats** |
| `0001_JA`, $-20$ and $+5$ degrees | 1258 nats | **596 nats** |
| `0000_LJ`, $+25$ and $+25$ degrees | 5920 nats | **462 nats** |

Under the affine projection an *isolated* local rotation costs more than a
two-glyph differential one. The cost ordering of local deformations is
inverted, which is the opposite of what a deformation prior is for. The
relative residual is 1% or better in every row, so the coarse lattice
represents these fields perfectly well; the whole difference is prior cost.

`warp_scale` was raised 1.5 to 5.0 at the same time. Cost falls as
$1/\text{scale}^2$, so the two measured deformations land near 50 nats. At 1.5
the prior gave local rotations of standard deviation 3.8 degrees and a 95th
percentile displacement of 1.9 px, against the roughly 8 px needed to turn a
38 px glyph by 25 degrees.

Measured end to end on `0000_LJ`, scoring the correct two-glyph explanation
with the model's own log joint, with and without the fitted rotation:

| conditioning | scale | warp prior cost | change in log joint |
|---|---|---|---|
| affine-free | 1.5 | 5920 | **$-2660$** |
| translation-free | 1.5 | 462 | $+2802$ |
| translation-free | 5.0 | 42 | **$+3223$** |

So under the former settings the model was right to refuse the rotation: the
projection charged more for it than the likelihood would pay. `warp_modes`
selects between the two conditionings and defaults to `"translation"`.

This is affine-free by construction and respects the GMRF covariance geometry.
It is also a linear transformation of the original Gaussian draw. The induced
density over $u_\perp$ or $v$ is necessarily singular in the ambient space,
because it lives on an exactly constrained subspace. In a generative program we
can retain the proper, normalized latent density $p_\theta(u)$ and make
$u_\perp$ and $v$ deterministic; the discarded affine coordinates then remain
uninformed auxiliary randomness. A reduced-coordinate parameterization could
remove that redundancy, but a dense null-space basis would sacrifice the GMRF's
sparse Markov structure.

Run this version with

```shell
python scripts/global_diffeomorphic_scene.py
```

## Main-model integration

`TexturedDiffeomorphicPoissonConvPlacements` installs the prototype in the
Poisson CAPTCHA generative model. For each observed image, the normalized
latent factors are

$$
\begin{aligned}
p_\theta(a)
&=\prod_{k,y,x}\operatorname{Poisson}(a_{kyx};\lambda),\\
p_\theta(c_0)
&=\prod_{j=1}^3\operatorname{Beta}(c_{0j};1,1),\\
p_\theta(r)
&=\mathcal N(r;0,Q_{\mathrm{tex}}^{-1}\otimes I_3),\\
p_\theta(u)
&=\mathcal N(u;0,Q_{\mathrm{warp}}^{-1}\otimes I_2).
\end{aligned}
$$

The deterministic renderer is

$$
\begin{aligned}
c(s)&=\operatorname{sigmoid}(\operatorname{logit}c_0+r(s)),\\
m(s)&=\frac{c(s)}{c_0}
     =\frac{\exp r(s)}{1+c_0(\exp r(s)-1)},\\
I_0&=\operatorname{Stamp}(a,K\odot m),\\
u_\perp&=\operatorname{ConditionTranslationFree}(u),\\
v&=WRu_\perp,\\
\phi&=\exp(v),\\
I(p)&=I_0(\phi^{-1}(p)).
\end{aligned}
$$

Here `color_texture` is one canonical field shared across the dictionary, while
`warp_velocity` is one image-coordinate field shared by every foreground glyph.
Only after the four ink channels have been warped does the existing likelihood
recover foreground colour, multiply it by the baseline $c_0$, and composite it
over paper. The baseline-relative $m$ preserves the old compositor and makes
$r=0$ exactly its former global-colour path. Setting both fields to zero
therefore recovers the former renderer exactly.

The amortized proposal mirrors both sites with proper second-order GMRFs,

$$
q_\phi(r\mid x)=\mathcal N(r;m_{\phi,r}(x),Q_{\phi,r}^{-1}),\qquad
q_\phi(u\mid x)=\mathcal N(u;m_{\phi,u}(x),Q_{\phi,u}^{-1}),
$$

using a globally decoded canonical texture mean and a coarse convolutional
velocity mean. Their scalar element and bond precisions are learned.
