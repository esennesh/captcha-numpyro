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
