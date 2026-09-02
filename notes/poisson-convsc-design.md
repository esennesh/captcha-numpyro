# Poisson convolutional sparse coding for the captcha model — design

**Branch:** `feature/poisson_convsc`
**Status:** all six build steps implemented (§9). Inference history: §11 (RLOO cold-start), §12
(matched filter — localization solved), §13 (double control variate), §14 (DSGD), §15 (**the
likelihood was the bug**; default switched from `mixture` to `blend`), §16 (three-way
comparison: DoubleCV and DSGD both work, RLOO does not), §17 (ELBO arbitration: the
objective's tail, not the estimator, is what is left to fix), §18 (Student-t tail: ELBO and
reconstruction now agree). §19 is an open question about when sparsity actually helps; §20-21 work out how to couple the
Student-t degrees of freedom to the spike count (§20's sign was wrong; §21 corrects it); §22
reruns the estimator comparison, where DSGD now wins.
**Replaces:** `PoissonMarkedPlacements` and the 37-component `SpatialMixtureSameFamily`
likelihood in `marionette_captcha_model`.

The idea comes from `poisson_hesc.py`: an integer activation field at *image* resolution,
stamped through a transposed convolution. Ported here, the spike count modulates **only the
opacity** of a stamp, and the likelihood's per-pixel uncertainty varies with the ink painted
at that pixel.

---

## 0. Ground truth about the target

Measured, not assumed.

| | |
|---|---|
| images | 5000 RGB, **natively 80x80** — `CaptchaDataset`'s resize is a no-op, so nothing is blurred by it |
| caption length | **1 character in 100% of the dataset** (`Counter({1: 2000})` over a 2000-image sample) |
| dictionary | 36 glyphs, `(36, 38, 26, 4)` RGBA, alpha is a real mask (mean 0.238), values in [0,1] |
| glyph RGB | white masks, so RGB carries no information today — all the shape is in alpha |
| ink extent | 30 x 25 within the 38 x 26 dictionary frame, matching the min-sum notes' generator measurements |
| current anchor grid | `(80-38)+1 = 43` by `(80-26)+1 = 55` = 2365 VALID anchors |

Three further measurements drive §5, and they were surprising enough to be worth stating up
front:

| | |
|---|---|
| background colour | **exactly `(1, 1, 1)` in every image** — per-image median has zero spread across the dataset |
| background exactness | **95.33% of all pixels are bit-identical to that white.** Robust residual scale (MAD) is *exactly 0* |
| ink pixels | 299 of 6400 per image |
| ink composition | **95.1% of non-background pixels are intermediate values** (median luminance 0.502), i.e. anti-aliased partial coverage, not saturated ink |

So: the background carries *zero* residual variance, and essentially all of the model's
fitting error will live on partial-coverage pixels.

---

## 1. The latent field: integer counts at image resolution

```
a[k, y, x] ~ Poisson(rate)          k = 1..36,  (y, x) over the full 80x80 grid
```

230,400 sites per image. `(y, x)` is where the glyph **centre** lands, not the top-left of its
box.

Implementation: transposed convolution at stride 1 to full support `(H+kh-1, W+kw-1)`, then a
crop at offset `((kh-1)//2, (kw-1)//2)`. **Convention:** a unit spike at `(y, x)` lays the
glyph's `(kh, kw)` frame down with its top-left corner at `(y - (kh-1)//2, x - (kw-1)//2)`,
unflipped. For an odd kernel dimension the frame centre is exactly `y`; for an even one — 38 x
26 here — it falls half a pixel past `y`, because a box with an even side has no centre pixel.
That is unavoidable rather than a choice, and nothing depends on it: *equivariance* is exact
either way, and sub-pixel offsets (§6) absorb the half pixel when they land.

Verified by comparing the rendered patch against the dictionary entry itself, which tests
position and orientation together: `max|patch - glyph| = 2.4e-4` (float32/TF32 convolution
precision) against `1.0` for the 180°-rotated glyph, confirming `transpose_kernel=True` does
*not* leave the stamp flipped. Equivariance measures exactly `0.000e+00` under integer shifts.

An earlier draft of this note claimed the spike centres the glyph exactly; that came from an
integer division in the check, not from the renderer.

Why image resolution and centre-anchored, rather than the current VALID anchor grid:

- **The coordinate means something.** `a[k, y, x] = 1` says "character k is at pixel (y, x)".
  On the VALID grid the coordinate is the top-left corner of a 38x26 box, on a 43x55 lattice
  that aligns with nothing in the image.
- **Every position is reachable**, including glyphs clipped by the frame. The VALID grid
  cannot place a character whose box leaves the image at all.
- **The prior is exactly translation-invariant** — a homogeneous marked Poisson process — so
  translating the count field translates the render, up to the boundary. This is the
  equivariance `poisson_hesc.py` checks at the end. It also makes the *proposal* a plain
  image-to-image CNN (§7) and makes deformation offsets live on the image grid (§6). Those two
  payoffs are the real reason for this choice.

### The rate is a parameter, not a latent

```python
log_rate = numpyro.param("log_rate", jnp.log(expected_count / (K * H * W)))
a = numpyro.sample("a", dist.Poisson(jnp.exp(log_rate)).expand((H, W, K)).to_event(3))
```

`expected_count` comes from the Hydra model config and is used **only to initialise**
`log_rate`, which then learns freely. There is no `z_rate` and no Gamma. We are not being fully
Bayesian over firing rates at this stage: with a single dictionary layer there is no second
level of convolutional features for a rate prior to be informative *about*, so a prior on the
rate would only add an unidentified global scalar and a KL term that fights the likelihood for
control of overall sparsity. When a second layer lands — where the layer-1 rate field is
genuinely predicted top-down, as `ahat0 = Phi1 a1` in `poisson_hesc.py` — the rate becomes a
quantity worth being uncertain about, and this is the line that changes.

Shape of `log_rate`: scalar by default; `(K,)` is the one other shape that is safe, and it
learns per-glyph frequency. A per-*site* rate `(H, W, K)` would destroy translation invariance
and with it §6 and §7 — don't.

**Both Dirichlets go away.** `z_where` (2365-way) and `z_mark` (36-way per anchor) exist only
to allocate a total firing mass across anchors and glyphs. With an explicit integer count per
(glyph, location) there is nothing left to allocate. Consequences:

- `notes/minsum-session-2026-07-29.md` Finding 2 (concentration < 1 ⇒ density unbounded on the
  simplex boundary ⇒ no interior mode) — gone, no simplex.
- Finding 3 (2364-deep stick-breaking producing off-support draws and `-inf` weights) — gone.
- The grid-dependence footgun documented at length in `configs/model/poisson_captcha.yaml`
  ("this value is grid-dependent... recompute it rather than carrying this number over") —
  gone. `rate` is per site, so changing the grid changes the site count and the expected total
  is preserved by the initialiser.
- The KL-collapse failure recorded for `z_mark`'s Concrete density — gone. No relaxation, no
  temperature, and the Poisson log-density is bounded above on its support.

At this stage the model's entire latent surface was `a` and `color`. As of the
2026-09-01 foreground-field integration, the active model adds the continuous
`color_texture` and `warp_velocity` sites described in §§3 and 6. The discrete
structural surface remains the single count field `a`.

---

## 2. Counts drive opacity, and nothing else

Define the **optical depth** at pixel `p`:

```
tau(p) = sum_{k,y,x} a[k,y,x] * alpha_k(p - (y,x))
```

which is exactly a transposed convolution of the count field with the dictionary's alpha
channel. Opacity is the Beer–Lambert / Poisson-void term:

```
A(p) = 1 - exp(-tau(p))          # use -jnp.expm1(-tau) for the small-tau regime
```

The count enters here and nowhere else. Two spikes at a site double the optical depth: more
opaque, saturating at 1, never a different hue. That is the requested semantics, and it is also
the only place where "counts add" is physically meaningful — **alpha composes by addition in
log-transmittance; colour does not compose by addition at all.** Modulating colour by a count
would be a category error; modulating opacity is what a count *is*.

Two further properties fall out for free:

- `tau >= 0` always, since `a >= 0` (integer counts) and `alpha in [0,1]`. This is the
  non-negativity constraint `poisson_hesc.py` had to install by hand (`build_phi1` uses
  non-negative impulse maps precisely so `ahat0` is a valid Poisson rate). Here the alpha
  channel is non-negative by definition, so no constraint is needed on the dictionary and no
  cancellation between glyphs is possible.
- This is the *same* expression already in `generate_marionette_captcha` (`model.py:346`,
  `transmittance = exp(-sum coverage)`), which currently justifies it as the Poisson void
  probability of a process whose counts were marginalized away. Now the counts are explicit and
  the expression is derived rather than approximated.

---

## 3. Colour: premultiplied, folded into the same convolution

Build a 4-channel kernel from the dictionary once:

```
kernel[..., 0:3] = alpha_k * rgb_k     # premultiplied colour
kernel[..., 3]   = alpha_k             # optical depth
```

One `conv_transpose` with `C_out = 4` yields `(tau * c_bar, tau)` in a single pass. Then

```
c_bar(p) = premultiplied(p) / max(tau(p), eps)
```

`c_bar` is the depth-weighted mean ink colour at the pixel. There is no depth ordering in a
Poisson process — the points are exchangeable — so a weighted average is the correct
exchangeable answer, not an ordered `over`.

With today's white dictionary `c_bar == 1` identically and this collapses to a single global
ink colour. **Decision: keep the 4-channel form anyway**, so coloured dictionary entries drop
in later. It costs one extra output channel.

The image-level baseline remains `color ~ Beta(1,1)^3`. It now anchors a shared
canonical logit texture

```
color_texture ~ SecondOrderGaussianMrf(0, Q_texture)
color_field(u) = sigmoid(logit(color) + color_texture(u))
color_modulation(u) = color_field(u) / color
```

of shape `(kh, kw, 3)`. The same baseline-relative modulation multiplies every
dictionary kernel before stamping, after which the old compositor multiplies by
`color`. There are not 36 class-specific fields and not one field per active
occurrence. At `color_texture = 0`, the modulation is exactly one, recovering
the former global-colour renderer, `fg = c_bar * color`, without a special case.

### The ink field

Everything above §3 produces exactly one object:

```
ink : (..., H, W, 4)      # [0:3] premultiplied colour, [3] optical depth
```

**Design rule.** A placement module returns an ink field. It never sees the likelihood, and the
likelihood never sees glyph identity, counts, or geometry. §4 and §6 are both drop-in ladders
hanging off this one interface.

---

## 4. Likelihood: two-component spatial mixture (default) or single-layer blend

Both options consume the ink field and nothing else, and both have the same mean. Choose in
config.

**(a) Blend — one layer, plain Gaussian.** The literal reading of "a single image layer":

```
mean(p) = A(p) * fg(p) + (1 - A(p)) * bg(p)
obs ~ Normal(mean, sigma(A)).to_event(3)                    # batch (B,), event (H, W, C)
```

**(b) Mixture — background and foreground as two components.** Reuses the existing
`SpatialMixtureSameFamily`, cut down from 37 components to 2:

```
weights = stack([1 - A, A], -1)                             # (B, H, W, 2)
means   = stack([bg, fg], -2)                               # (B, H, W, 2, C)
obs ~ SpatialMixtureSameFamily(Categorical(probs=weights),
                               Normal(means, scales).to_event(1),
                               reinterpreted_batch_ndims=2)
```

The weights need no normalization: `1 - A` and `A` already sum to 1 by construction, because
`1 - A = exp(-tau)` *is* the void probability. The current code has to renormalize
(`model.py:392`, `coverage / coverage.sum(-1, keepdims=True)`); here that line disappears.

### `ambient_depth`: a floor the implementation forced, and then earned

Implementing the mixture surfaced a hard failure the design had not anticipated.
`MixtureSameFamily.log_prob` stabilises with `m = max_k log p_k(x)` taken over **all**
components, including zero-weight ones. At an inkless pixel — `A` exactly 0, which is most of
the canvas — whose data nonetheless looks like ink, the zero-weight foreground component attains
that max, and the weighted background term underflows to zero: `log 0 = -inf`. Measured on real
images before the fix: `log_prob = [-inf, -inf, -inf, -inf]`. That is the same `-inf`-weights
failure `notes/minsum-session-2026-07-29.md` Finding 3 records as fatal.

The fix is a small optical-depth floor added before computing opacity — exactly the `EPS` added
to `ahat0` in `poisson_hesc.py`:

```
A = 1 - exp(-(tau + ambient_depth))            # ambient_depth = 1e-4
```

It earns its place twice over. Numerically it keeps the foreground weight strictly positive, so
the max is never attained at a zero-weight component. Modelling-wise it **caps the cost of an
unexplained ink pixel** at roughly `-log(ambient_depth)` nats instead of letting the confident
background component charge a quadratic `(x-1)^2 / 2 sigma_bg^2`. Measured: a no-ink model
scores a real image at `-47089` nats relative to blank white, about 152 nats per missed ink
pixel, against roughly 13850 uncapped — a 91x reduction. The gradient in the opacity is then
`~1/A`, largest exactly where ink is missing. A strong, *bounded* signal to place a glyph
rather than an enormous one.

A second, related fix: where `tau = 0` the mean ink colour is `0/0`. Falling back to 0 makes the
foreground hypothesis black — an assertion about the dictionary rather than the image. The
fallback is 1, so the foreground component reads "if this pixel were ink, it would be this
image's ink colour".

Both are properties of the *mixture* path. The blend never produced `-inf` (its per-image
log-densities came out around `-5e5` to `-2e6`), but it pays the uncapped quadratic, which is
one more reason the mixture is the default.

**Why (b) is the default.** Two arguments, both about gradients.

*Responsibility.* Differentiating a marginalized mixture gives

```
d log p / d A  =  (N_fg - N_bg) / p  =  r_fg / A  -  r_bg / (1 - A)
```

where `r_fg`, `r_bg` are the posterior responsibilities. This is a *ratio*: it does not shrink
as the foreground/background contrast shrinks, and it points hard toward whichever hypothesis
actually explains the pixel. The blend model instead gives

```
d log p / d A  =  (x - mean) * (fg - bg) / sigma^2
```

which is proportional to the contrast and vanishes when `fg ~ bg` — precisely the regime a
half-placed glyph sits in.

*The optimum under uncertainty.* For a pixel the model is unsure about, the blend's best answer
is an intermediate mean: literally half-ink, a grey pixel that matches neither hypothesis. The
mixture's best answer is a 50/50 weight over two hypotheses that both stay sharp. This is the
direct mechanism behind the outcome recorded in the min-sum notes §3 — "it localizes and
colors, it does not identify", reconstructions as "faint blobs". A blended mean is *rewarded*
for being a faint blob; a mixture is not.

`likelihood.mean` is exactly option (a)'s mean, so the existing `plot_mean` / `residual`
plumbing is unchanged.

**Correction (measured in step 4).** An earlier draft claimed that cutting from 37 components
to 2 makes the per-pixel assignment "nearly deterministic", fixing the dithering complained
about at `model.py:398-403`. It does not, by itself. The assignment concentrates only as
`A -> 1`, and a *single* spike reaches `A = 0.632` — so 37% of a glyph's interior pixels are
still drawn from the background component, and prior-predictive `obs` samples are visibly
speckled. What removes the dithering is saturation (`n >= 3`), not the component count. This
affects inspection only: the likelihood is exact either way, and `mean` is the site to plot,
which is why the model exposes it.

Cost: `(32, 80, 80, 2, 3)` = 4.7 MiB, against `(32, 37, 80, 80, 3)` = 108 MiB for the current
per-glyph mixture.

---

## 5. Ink-dependent variance, in whichever direction wins

The requirement is that per-pixel variance correlate with ink. The *sign* of that correlation
is an open empirical question, and the design should not hard-code it:

- **Slack where ink is** (`sigma^2` increasing in `A`) tolerates the sub-pixel placement and
  anti-aliasing error that concentrates on glyphs, so the optimizer is not punished for a
  nearly-right stamp.
- **Steepness where ink is** (`sigma^2` decreasing in `A`) makes
  `d log p / d mean = (x - mean) / sigma^2` largest exactly where the character is, so gradient
  signal concentrates on the pixels that carry information rather than spreading over blank
  paper.

So the ink-to-variance map is a pluggable component with signature `A -> sigma^2`:

| name | form | direction |
|---|---|---|
| `affine` | `sigma_bg^2 + k_ink * A` | slack where ink |
| `edge` | `sigma_bg^2 + k_edge * A * (1 - A)` | slack at partial coverage only |
| `inverse` | `sigma_bg^2 + k_ink / (A + eps)` | steepness where ink |
| `endpoints` | `(1 - A) * sigma_bg^2 + A * sigma_fg^2` | **either**, per learned endpoints |

`endpoints` is the recommended default and subsumes the sign question: two positive scales, and
whether ink gets slack or steepness is decided by which endpoint ends up larger, not by a
config choice made in advance. It is also bounded at both ends, unlike `inverse`, which is
singular at `A = 0` and assigns its *largest* variance to blank paper — the opposite of what
the "steeper gradients on ink" argument wants. If that direction is what you want to test,
`sigma_bg^2 + k_ink * (1 - A)` is the bounded version of the same monotone direction and should
be tried first.

All coefficients are **global positive scalars** — a handful of degrees of freedom, not 6400.

### `sigma_bg` must be fixed, and the data says why

The earlier draft asked whether to learn `sigma_bg` or measure it. Measured (§0): the
background is exactly `(1,1,1)` in every image and **95.33% of pixels are bit-identical to it**,
so the robust residual scale is *exactly zero*.

That makes a freely-learned `sigma_bg` a **degenerate optimum**: on 95% of the canvas the model
can predict the data exactly, so driving `sigma_bg -> 0` sends the Gaussian log-density to
`+inf`. The likelihood is unbounded above and the optimizer will find it. This is not a
hypothetical failure mode — it is the guaranteed one.

**Decision: `sigma_bg` is a fixed config constant, not a learnable parameter.** Learn only
`sigma_fg` (or the `k_*` coefficient of whichever schedule is in use).

Choosing its value is an optimization decision, not a fitting one, because the data's answer is
literally 0:

- 8-bit quantization floor: uniform on a `1/255` step has std `1/(255*sqrt(12))` = **0.0011**.
  The true information-theoretic floor.
- At `sigma_bg = 1/255 = 0.0039`, a background pixel the model wrongly inks by 0.1 costs
  `(0.1/0.0039)^2/2` ~ **329 nats**, times thousands of pixels. Correct in principle, brutal in
  practice: misplaced ink dominates every other gradient.
- Suggested start: **`sigma_bg = 0.01`**, roughly 2.5x quantization, as a config knob. Loose
  enough not to detonate on early random placements, tight enough that blank canvas is not a
  free lunch. Anneal downward later if it helps.

### The data also picks the direction

95.1% of non-background pixels are *intermediate* values — the character is nearly all
anti-aliased ramp, with a median luminance of 0.502. So virtually all irreducible mismatch sits
on partial-coverage pixels, and the data-fit answer to §5's sign question is unambiguously
**more variance where the ink is**, i.e. the `affine`/`edge` direction. That conflicts with the
"steeper gradients on ink" optimization intuition, which is a real consideration but a
different one. `endpoints` with `sigma_bg` fixed and `sigma_fg` learned lets the run settle it:
if `sigma_fg` climbs well above `sigma_bg`, the data argument won.

### The mixture derives all of this for free

For the two-component mixture of §4(b), the marginal per-pixel variance is

```
Var(p) = (1 - A) * sigma_bg^2  +  A * sigma_fg^2  +  A * (1 - A) * (fg - bg)^2
```

— the `endpoints` form *plus* the `edge` form, with the edge coefficient pinned to the
foreground/background contrast rather than left free. Under option (b) the ink-dependent
variance is not posited at all: it is what a two-component mixture with per-component scales
*is*, and the only free parameter is `sigma_fg`. Under option (a) the schedule has to be
written down by hand. That is a third argument for the mixture, on top of the two in §4.

---

## 6. The seam for deformable convolutions

The ink field of §3 is the seam. Three rungs, each a drop-in replacement:

The implemented main-line model now takes a whole-image variant of rung 2:

```
ink0 = render_dense(counts, color_field)
warp_velocity ~ SecondOrderGaussianMrf(0, Q_warp)
velocity = affine_free(resize(warp_velocity))
ink = pullback(ink0, exp(velocity))
```

It warps the complete four-channel foreground after stamping, so glyph centres
may move together with their strokes while the paper remains fixed. This keeps
the latent shape independent of the number of active glyphs and retains the
optimized transposed convolution. Per-occurrence fields remain a tested
reference, not the active generative path.

1. **`render_dense(counts)`** — the transposed convolution above. Rigid stamps.
2. **`render_warped_dense(counts, warp)`** — sample a per-image warp latent (affine 6-dof, or a
   low-rank thin-plate-spline control grid), resample all 36 glyphs once with
   `jax.scipy.ndimage.map_coordinates`, then run the *same* convolution with the warped kernel.
   One gather over `36 x 38 x 26`; **zero change downstream**. This already captures the
   generator's per-character rotation and shear, which is most of the deformation in this
   dataset.
3. **`render_sparse(sites, counts, offsets)`** — true deformable convolution. Gather the `M`
   sites with `a > 0`, warp each stamp by its own offset field `delta(k,y,x)` in
   `R^{kh x kw x 2}`, `scatter_add` into the canvas. `O(M * kh * kw)`.

Rung 3 is affordable **only because the activations are integer and sparse**: `E[M] =
expected_count = 1`. A continuous relaxation has no support to gather. This is the concrete
payoff of counts over a Gamma or Concrete field, and it is a reason for the whole design rather
than a consequence of it.

Two structural details that keep rung 3 reachable:

- **Sub-pixel placement is the low-order case and is free.** A fractional offset is a bilinear
  splat of the count onto the 4 neighbouring integer sites — still a convolution, still rung 1.
  Build the splat in from the start (`render_dense(splat(a, frac_offset))`) so continuous
  position exists before deformation does.
- **The offset field has the same shape as the activation field**, because both live at image
  resolution. A deformable variant is a second head on the same CNN that proposes `a` (§7), with
  no resampling between grids. On the 43x55 VALID grid the offsets would live on a lattice
  unrelated to the image and would need their own interpolation.

---

## 7. Inference: RLOO through `ELBOTracer`, with a convolutional proposal

`dist.Poisson` has `has_rsample = False` and `has_enumerate_support = False` (verified). That is
exactly the case `ELBOTracer` is built for, and it needs no changes to accept a Poisson site:

- `setup` records `reparameterized = site["fn"].has_rsample` (`tracer.py:224`), so `a` is
  classified non-reparameterized automatically.
- Its `log_q` is `stop_gradient`-ed out of the main weight (`tracer.py:189-190`) and re-entered
  as the score-function surrogate `log_q * stop_gradient(advantage)` (`tracer.py:203-206`).
- `advantage = downstream_cost - downstream_cost.mean(axis=0)` (`tracer.py:202`) combined with
  `loss_fn` dividing by `K - 1` rather than `K` (`tracer.py:214`) is **exactly** the
  leave-one-out baseline: `c_k - (sum_{j != k} c_j)/(K-1) = K/(K-1) * (c_k - mean)`.
- `downstream_costs` is restricted to graphically downstream terms via `_model_deps` /
  `_guide_deps` (from numpyro's `get_nonreparam_deps`) and reduced with
  `MultiFrameTensor.sum_to`.

Two consequences worth stating up front.

**Rao-Blackwellization buys latent separation here, not per-site credit.** With `a` as one site
carrying `.to_event(3)`, `sum_to` yields one scalar advantage per (particle, batch element).
There is no per-site credit assignment — but there is also nothing to gain from it, because the
likelihood is a single image-wide `obs` and every site is graphically upstream of every pixel.
What the downstream-cost restriction does buy is keeping `color`'s cost out of `a`'s advantage
and vice versa.

**The variance profile improves as the posterior sharpens.** A 230k-dimensional discrete latent
against a scalar reward sounds hopeless, but nearly every site has `lambda_hat -> 0`, samples
`0` deterministically, and contributes no variance. The effective dimension is the number of
sites with appreciable rate — order `expected_count`. So this estimator gets *better* as the
proposal concentrates. That is the opposite profile from a Gaussian/CLT approximation to `tau`,
which is accurate only while many sites contribute and degrades exactly as the model starts
working. Hence RLOO as the primary path; a Gaussian local reparameterization
(`E[tau] = lambda_hat * alpha`, `Var[tau] = lambda_hat * alpha^2`, both single convolutions) is
worth keeping as a warm-start or a gradient cross-check, not as the objective.

**Update after step 6 (§11): the warm start is not optional.** The reasoning above holds for the
sharp regime but says nothing about reaching it, and a cold RLOO run does not. Read §11 with
this section.

Escalation if RLOO variance is too high: `DoubleCVTracer` (`tracer.py:418`) is already drafted
against this interface, and its docstring anticipates precisely this latent — "assumes each
latent's log_prob has a smooth continuous extension (**Poisson via gammaln**) and a defined
`.mean`". It is marked DRAFT and not yet run.

### The proposal: a fully-convolutional CNN over `a`

```
images (B, 80, 80, 3)  ->  CNN, all SAME  ->  log_rate_hat (B, 80, 80, K)
q(a) = Poisson(softplus(log_rate_hat))
```

No anchor geometry, no alignment argument. Compare the effort `captcha_encoder.py`'s module
docstring spends on it today: a single VALID `(kh, kw)` conv chosen so the head lands on
"exactly the anchor grid shape `(H', W') = ((H - kh)//stride + 1, ...)` — the same geometry as
`BayesianMarioNettePlacements` on the generative side, so each per-anchor posterior parameter
sees exactly one glyph's worth of input pixels and aligns spatially with the corresponding
prior cell". At image resolution that entire alignment problem does not exist, and the proposal
is an ordinary image-to-image network.

Three requirements on the architecture:

- **Receptive field at least `38 x 26`.** Deciding whether glyph `k` is centred at pixel `p`
  requires seeing all of `p`'s 38x26 neighbourhood. A stack of dilated 3x3 convs or a small
  U-Net with skips; a shallow local stack will not work, and its failure mode (rates that
  respond to strokes rather than whole characters) is subtle enough to be worth checking
  deliberately.
- **Stay fully convolutional.** Then `q` is translation-equivariant like the prior: translating
  the image translates the proposed rate field. That symmetry is free structure, and it is what
  makes the rest of §1's argument pay off.
- **Initialise the output bias at the prior rate**, so training starts at `q = p` and the KL
  term starts near zero rather than fighting a randomly-initialised rate field.

Additional heads for `color` (global pool -> Beta concentrations) follow the existing multi-head
pattern in `captcha_encoder.py`, and §6's offset field would be a fourth head.

Better and more biologically plausible inference is deferred; a CNN amortizer is the baseline
that makes everything else measurable.

---

## 8. Costs measured

Removing the K axis from every intermediate — which one composited layer permits — is what makes
a full-resolution activation grid affordable. At B=32:

| render | output | memory | forward | backward |
|---|---|---|---|---|
| single grouped conv (this design) | `(32, 80, 80, 4)` | **3.1 MiB** | 21.8 ms | 2.3 ms |
| per-feature vmap (repo, at image res) | `(32, 36, 117, 105, 4)` | 215.9 MiB | 109 ms | — |

70x less memory. The forward being ~10x its own backward suggests XLA picks a poor algorithm for
`transpose_kernel=True`; if it matters, write the stamp as a flipped-kernel
`conv_general_dilated` with full padding instead.

---

## 9. Build order

1. **Done.** `_stamp(counts, kernel)` in `src/model/model.py` — folds leading dims, renders full
   support, crops. Verified for placement, orientation and exact equivariance (§1).
2. **Done.** `PoissonConvPlacements` — `log_rate` param, `a` Poisson over `(H, W, K)`,
   `ink_field()` returning `(..., H, W, 4)`. No Dirichlets, no `z_rate`, no `stride`, no
   `where_concentration`. The rate initialiser is calibrated: `expected_count = 1` and `3` give
   empirical mean totals of 0.96 and 2.97 glyphs over 2000 draws.
3. **Done.** `poisson_convsc_model` — `generate_poisson_convsc` compositing, `_ink_scale`
   variance schedules, both §4 likelihoods behind the `likelihood` switch, `mean` / `residual`
   deterministic sites preserved.
4. **Done.** `notebooks/poisson_convsc.ipynb` — executed end to end, 9 code cells, zero errors,
   three figures with stored outputs. Findings below.
5. **Done.** `configs/model/poisson_convsc.yaml`, `configs/experiment/poisson_convsc.yaml`.
6. **Implemented.** `src/inference/poisson_conv_encoder.py` +
   `configs/guide/poisson_convsc_encoder.yaml`, now the experiment's guide in place of the
   `auto_mean_field` mirror. `PoissonConvBackbone` is a stride-1 SAME stack of 3x3 convolutions
   at dilations `(1, 2, 4, 8, 16)`; `PoissonRateHead` is a 1x1 convolution to `K` log-rates;
   `color` reuses the existing `MarioNetteColorFinder`. Verified at initialisation:

   - **Receptive field 63x63**, against the 38x26 glyph frame the design requires. Dilation
     rather than downsampling, so the proposal grid stays the image grid and `q` inherits the
     prior's translation equivariance.
   - **`q(a)` starts exactly at the prior.** The head's kernel is zero-initialised and its bias
     set to `log(expected_count / (H W K))`, so the log-rate field is constant at `-12.3476`
     (std `9.5e-07`, i.e. float32 noise) and `log q - log p = +0.0000` for `a`. The KL of 1.29
     that shows up in telemetry is entirely `color`, whose head is randomly initialised.
   - The log-rate is **clipped, not softplus'd**. The score-function gradient depends on
     `d log(lambda_hat) / d logit`, which under `exp` is exactly 1 at every rate including the
     `4.3e-6` the field starts at; the clip at `exp(5)` only bounds the tail.

   **It does not learn.** See §11.

---

## 11. RLOO does not get the proposal off the ground

600 steps, batch 8, 8 particles, ~0.36 s/step. Everything stays finite; nothing localizes.

| | start | after 600 steps |
|---|---|---|
| max `lambda_hat` | `4.34e-6` (the prior) | `7.04e-6` — **1.6x**, i.e. still flat |
| expected spikes/image | 1.000 | 1.049 |
| ESS (8 particles) | 0.125 | **0.125** — pinned at the `1/K` floor the whole run |
| reconstruction MSE | 0.0164 | 0.0153 |
| blank-white MSE | | **0.0122** |

The model is *worse than a blank white canvas* after 600 steps. The loss does fall (−260k to
−476k), but that is the likelihood's nuisance parameters — `sigma_ink`, `color` — not structure.

**What the proposal actually learned is the marginal.** The rate fields for four test images with
characters in four different places are essentially identical: a bright central rectangle with
dark margins about 19 px wide — exactly the region where a centre-anchored glyph would be clipped
by the frame. That is a correct fact about *where glyphs can be*, learned from the prior geometry,
and it is completely independent of the image in front of it.

**Diagnosis: needle in a haystack.** A draw from `q` puts about one spike uniformly over 230,400
sites. The chance a particle lands on the true glyph at the true position, within the few pixels
that would raise the likelihood, is order `1e-3`. With 8 particles, essentially no particle is
ever good, so the leave-one-out baseline ranks eight equally-wrong samples. The advantage carries
information about global nuisances (which is why `color` and `sigma_ink` do move) and none about
*where the character is*. ESS pinned at the floor for 600 straight steps is the same fact seen
from the importance weights.

**This refines §7 rather than refuting it.** §7 argued that RLOO's variance profile improves as the
posterior sharpens. That still looks right — but it is a statement about the *sharp* regime, and
it says nothing about how to reach it. Sharpening needs a sample that lands on the character, and
such a sample needs an already-concentrated proposal; RLOO on its own has no way into that loop.

So the estimator is not the problem, and nothing about it needs replacing: `ELBOTracer` handles
the Poisson site correctly and without modification, and RLOO remains the right thing to finish
with. What is missing is a way to *start*. The Gaussian local reparameterization §7 demoted is
accurate precisely in the diffuse regime — which turns out to be the complementarity that
matters, in the opposite order from the one §7 assumed.

### Two ways out, not mutually exclusive

1. **Matched-filter initialisation of the rate head. Implemented — see §12.**
2. **A pathwise gradient to escape the diffuse regime.** See below.

Worth noting that neither is a change to the *model*. Steps 1–5 stand as verified; this is
entirely about how to search 230,400 discrete sites.

### Why a pathwise gradient helps, and how to get one for a Poisson

The two estimators differ in what they are permitted to look at. The score-function estimator,
`E_q[f(a) grad_phi log q_phi(a)]`, treats `f` as a black box: draw `a`, evaluate, raise the
log-probability of whatever scored well. It never differentiates `f`, which is exactly why it
handles discrete latents — and exactly why it stalls here, since "which of these eight junk
samples was best?" carries no information. A pathwise estimator writes `a = g_phi(eps)` and
computes `E[grad_a f * grad_phi g_phi]`, differentiating *through* the decoder. It does not need
a good sample; it needs a good gradient at the current sample, which is a different and much
weaker requirement.

Poisson is not reparameterizable, but it does not have to be. The likelihood depends on `a` only
through `tau = sum a * alpha`, which is **linear** in `a`. So reparameterize `tau` instead. Under
mean-field `q(a) = prod Poisson(lambda_hat_i)`, `tau(p)` is a weighted sum of independent
Poissons, and since `Var(Poisson) = mean` and `Var(cX) = c^2 Var(X)`:

```
E[tau]   = lambda_hat  conv  alpha
Var[tau] = lambda_hat  conv  alpha^2
```

Two convolutions, both with kernels already in hand. Sampling `tau = mean + sqrt(var) * eps`
makes `d tau / d lambda_hat` analytic, so the pixel residual back-propagates through the
convolution to **all 230,400 rates at once**. This is the local reparameterization trick (Kingma,
Salimans & Welling 2015): sample the pre-activation whose distribution is available in closed
form rather than the latent whose is not.

It is a warm start rather than a replacement, for two reasons. The Gaussian step is a CLT claim
— excellent while many sites contribute comparably to a pixel, wrong once one site dominates, at
which point `tau` is essentially a single scaled Poisson supported on multiples of `alpha`. It
therefore degrades exactly as the posterior sharpens, the mirror image of RLOO. And it is
*biased*: a surrogate for optimisation, not the bound to report.

`DoubleCVTracer` (`tracer.py:418`) is the more principled version and is already drafted against
this interface. It does not reparameterize at all: it keeps REINFORCE and subtracts a control
variate built from the first-order Taylor expansion of `f` about `E_q[a]`, `f~(a) = f(abar) +
g . (a - abar)` with `g = grad_a f(abar)`. The term `grad_phi E_q[f~] = g . grad_phi E_q[a]` *is*
a pathwise-flavoured gradient — it uses the decoder derivative — while REINFORCE only has to
cover the small remainder `f - f~`. Low-variance local geometry, and unbiased. Its docstring's
requirement of "a smooth continuous extension (Poisson via gammaln) and a defined `.mean`" is
precisely the demand that `f` be differentiable in `a`.

### What the end-to-end smoke test established

Composing `+experiment=poisson_convsc` through `GraphicalModelLearner` and running 30
`train_step` calls: loss and every parameter stay finite, and

- `a` is classified `has_rsample=False`, so `ELBOTracer` routes it through the score-function
  surrogate with the leave-one-out baseline **with no changes to the tracer** — as predicted in
  §7. `model_deps` comes out `{'a': ['a'], 'obs': ['a']}`, i.e. the image cost is attributed
  downstream of the counts and `color`'s is not.
- `d(log p)/d(log_rate)` is exactly `sum(a) - sum(lambda)` (measured `5 - 4 = 1` on a
  4-image batch at `expected_count = 1`), confirming the rate parameter is wired.
- `placements_p$params` is empty — the dictionary is fixed, as decided.

### What the prior-predictive notebook established

**The opacity-only invariant holds exactly.** The composite's hue does shift with the count —
that is alpha compositing over a white background, not the count reaching colour. The invariant
that encodes "opacity only" is that the pixel stays on the line from `bg` to `fg`, i.e.
`(mean - bg)/(fg - bg)` is one scalar equal to `A`. Measured per-channel spread across
`n = 1..10`: at most `1.8e-7`, float32 noise.

**One spike reaches `A = 0.632`, and that is the mechanism, not a limit.** A glyph's alpha peaks
at 1.0, so `n = 1` gives `tau_max = 1` and `A = 1 - e^-1`. Driving opacity the rest of the way
is precisely what the count is for. I briefly proposed a learnable gain `tau = g * (alpha * a)`
to saturate a single stamp; that was wrong and was reverted — it would saturate at `n = 1` and
leave the count with nothing to modulate, collapsing the field to a presence indicator and
taking the sparse-scatter route to deformable convolutions (§6, `E[M] = expected_count`) with
it.

The data confirms the count carries real signal. Against per-image darkest-pixel luminance
(median 0.404, p5 0.230, min 0.114), and taking pure black ink as the best case:

| | opacity ceiling | composite floor | images with an unreachable pixel |
|---|---|---|---|
| `n = 1` | 0.632 | 0.368 | **40.8%** |
| `n = 2` | 0.865 | 0.135 | 1.0% |
| `n = 3` | 0.950 | 0.050 | 0.0% |

**Renders are lighter than the data, and it is the colour prior, not opacity.** Median ink-pixel
luminance is 0.709 for prior renders against 0.499 for real captchas, while coverage is in the
right range (415 vs 297 pixels per image; a single glyph's alpha support is ~235 and two-glyph
draws overshoot). `Beta(1,1)^3` has mean luminance 0.5 and real captcha ink is darker than
mid-grey.

**The factorized prior cannot co-locate spikes.** Reaching `n >= 2` at one site requires two
draws at the same `(k, y, x)`, which at a rate of `4.3e-6` essentially never happens. So
prior-predictive draws are always single-stamp and faint; the *posterior* has to do the
co-location, at a KL cost of roughly 25–40 nats against a likelihood of order `1e4`. Cheap, but
it says where the pressure must come from — and suggests the `expected_count` initialiser may
want to be 2 rather than 1, since a one-character image wants one or two spikes, not one.

Two things to watch, neither a defect at step 0:

- **KL is ~0 and stays there.** A Poisson field this sparse contributes a KL of order `1e-6`
  against a likelihood of order `1e4`, so the prior exerts essentially no pressure. Sparsity
  will come from `log_rate` learning, not from KL pressure. Worth re-checking once the CNN
  proposal replaces the mean-field mirror and `q` actually concentrates.
- **ESS is 0.333 with 3 particles** — the floor, meaning one particle dominates completely.
  Expected with an untrained mean-field guide over 230k discrete sites, and §7 predicts it
  improves as the proposal sharpens. If it does not, that is the signal to escalate to
  `DoubleCVTracer`.

---

## 10. Decisions and remaining questions

Settled:

- **`expected_count`** — fixed in the Hydra model config, alterable later. It only initialises
  `log_rate` (§1).
- **Dictionary** — fixed, not learned, for now. A non-negativity reparameterization exists and
  can be dropped in when the dictionary becomes learnable (§2 explains why `tau >= 0` requires
  one).
- **Per-glyph colour** — keep the 4-channel premultiplied path so coloured dictionary entries
  drop in later (§3).
- **`sigma_bg`** — fixed config constant, *not* learnable. The background is bit-exact in 95% of
  pixels, so a learnable `sigma_bg` has an unbounded optimum at 0. Start at 0.01 (§5).

Open, and worth a call before or during implementation:

- **Likelihood default.** I recommend committing to the two-component mixture (§4b) and keeping
  the blend as a config switch for ablation. Three independent arguments favour it, and the
  third — that it derives the ink-dependent variance instead of positing it — was not in the
  original framing.
- **Variance schedule.** `endpoints` with `sigma_bg` fixed and `sigma_fg` learned is the default
  I would start with; it collapses to a single learnable scalar under the mixture. Run `affine`
  vs `inverse` as an explicit A/B if the direction question matters empirically.
- **`log_rate` shape.** Scalar, or `(K,)` for per-glyph frequency? Scalar unless you want the
  model to learn character frequency, which is uniform in this dataset.

Unrelated but worth knowing: `configs/model/categorical_captcha.yaml` and
`configs/experiment/categorical_captcha.yaml` are untracked in the working tree and target
`src.model.model.categorical_captcha_model` / `CategoricalMarkedPlacements`, which **do not
exist** in `model.py` (409 lines, neither symbol present). The uncommitted implementation
described in `notes/minsum-session-2026-07-29.md` §6 is gone; those two configs are dangling.

---

## 12. Matched-filter initialisation: localization solved, magnitude not

`PoissonRateHead` now parameterizes the rate field as a total times a shape,

```
log lambda_hat = log_total_q + log_softmax(match_gain * z + head(features))
```

with `z` the standardized matched-filter score field from `_matched_filter`, the **adjoint of the
generative stamp**. Being the adjoint is what makes the geometry line up for free: `Phi^T x` is
the matched filter for *this* decoder, so a peak at `(k, y, x)` is a statement about the latent
site of the same name. Verified numerically: `<Phi a, r> = <a, Phi^T r>` to a relative error of
`3.7e-4`, i.e. TF32 convolution precision.

The `log_softmax` keeps the field calibrated — the rates sum to `exp(log_total_q)`, so a flat
score field reproduces the prior rate exactly and a peaked one redistributes the same mass. Both
`log_total_q` and `match_gain` are learnable scalars; `head` is zero-initialised, so at step 0 the
proposal *is* the matched filter.

### The detector is essentially exact on this dataset

Over 300 images, an argmax over all 230,400 sites of the score field:

| | |
|---|---|
| correct glyph identity | **100.0%** |
| position within 5 px of the ink bbox centre | **100.0%** (median \|dy\| 0.5 px, \|dx\| 0.0 px) |
| true glyph's rank among 36 | median 0 (top-1 100%) |
| peak height above the field mean | 11.7 sd (min 10.2 over 300 images) |

That is unsurprising and worth saying plainly: these captchas are clean, undistorted renderings of
the very glyphs in the dictionary, so the correlation is close to a sufficient statistic. The
learned `head` is a *residual* on top, and it is the part that will matter once deformation
(§6) makes the rigid correlation wrong. On this data the hand-designed feature is doing the work,
not the network.

### What changed, and what did not

Same probe as §11 — 600 steps, batch 8, 8 particles:

| | §11 (cold) | §12 (matched filter) |
|---|---|---|
| max `lambda_hat` at init | `4.34e-6` | **0.647** |
| reconstruction MSE at init | 0.0164 | **0.0083** |
| reconstruction MSE after 600 steps | 0.0153 | 0.0079 |
| blank-white baseline | 0.0122 | 0.0123 |
| ESS (8 particles) | 0.125 throughout | 0.125 throughout |
| KL | ~0.1 | ~12 |

The model now **beats a blank canvas before a single gradient step** (0.0083 against 0.0123),
where the cold run never beat it at all. The rate field is a single sharp point that tracks the
character across images, and where a spike is drawn the render is the right glyph, in the right
place, in the right colour. The KL of ~12 nats is the honest price of a concentrated posterior
and is negligible against a likelihood of order `1e4`, as §11 predicted.

**What RLOO still is not doing is fixing the magnitude.** With `log_total_q = 1` the peak site
holds `lambda_hat = 0.669`, so a Poisson draw yields nothing **51%** of the time and half the
renders are blank. Decomposing the MSE on a 16-image batch:

| | MSE |
|---|---|
| blank canvas | 0.0125 |
| one spike placed deterministically at the matched-filter peak | **0.0045** |
| sampling from `q` as configured | 0.0084 |

So the achievable target is 0.0045, the proposal delivers 0.0084, and essentially all of the gap
is empty draws. Closing it needs `log_total_q` to climb to roughly 1.3 (peak `lambda_hat ~ 2.5`,
`P(no spike) ~ 0.08`). Over 600 steps it moved from 0.000 to **-0.003**.

That is a single scalar, with a smooth and monotone effect on the expected likelihood, that the
score-function estimator cannot find — which is about as clean a case for the pathwise gradient
(§11) as one could ask for. Localization was the cold-start problem and the matched filter solved
it; magnitude is a separate problem and it is an estimator problem.

*(Caveat on the MSE decomposition: `color` is drawn from its prior in every row rather than
optimized, so these numbers compare placements under a randomly-coloured ink, not the best
achievable reconstruction. The ordering is what matters, not the absolute values.)*

---

## 13. DoubleCVTracer: a real gradient, pointed the wrong way

`DoubleCVTracer` was a draft that had never been run. Four things had to be fixed before it
would execute at all:

1. **`posterior_mean` did not exist.** Referenced in the docstring and in `_guide_means`, defined
   nowhere. Now a `Messenger` that substitutes each unobserved site's `.mean` during
   `process_message`, short-circuiting the draw as `substitute` does so no randomness is consumed
   and the values stay differentiable in the guide parameters — which is what makes `CV1` a
   gradient rather than a constant.
2. **Trace entries were indexed as tuples** (`site[1]`, `site[2]`, `site[3]`) while `trace_entry`
   returns a dict; the class was written against an older representation. The inner product also
   assumed a trailing feature axis, but the count field is `(K, B, H, W, 36)`, so it has to run
   over every axis past particle and batch.
3. **The control variate was applied to every site**, double-counting the reparameterized
   `color`, whose sampled value already carries a pathwise gradient. Added a `setup` recording
   `has_rsample` per site and restricted both `CV1` and the REINFORCE score term to
   non-reparameterized latents.
4. **The expansion point is off-support by design, and numpyro masks it.** `E_q[a] = lambda_hat`
   is not an integer; `validate_sample` maps out-of-support values to `-inf`, whose gradient is
   identically `0`, so the control variate silently became a no-op and the loss went `NaN`:

   ```
   Poisson(1e-5).log_prob(0.5) as-is         : -inf      grad wrt value : -0.0
   Poisson(1e-5).log_prob(0.5), validate off : -5.6357   grad wrt value : -11.549
   ```

   Wrapping that one evaluation in `numpyro.validation_enabled(False)` recovers the analytic
   continuation the class docstring assumes it gets. Worth knowing the assumption is not free.

### Controlled head-to-head

Identical fixed probe batch, identical fixed training stream, identical initialisation, 400
steps, 8 particles. `E[MSE]` averaged over 10 draws, so empty-draw luck is averaged out.

| | `log_total_q` | `max_rate` | `E[MSE]` | s |
|---|---|---|---|---|
| `ELBOTracer` (RLOO) | -0.0027 | 0.8100 | 0.00754 | 177 |
| `DoubleCVTracer` | **-0.0276** | 0.7789 | 0.00794 | 159 |
| blank canvas | | | 0.01338 | |
| one spike at the matched-filter peak | | | 0.00451 | |

DoubleCV moves `log_total_q` about 10x further than RLOO and, unlike RLOO, moves it
*monotonically* — every checkpoint decreases, where RLOO wanders inside its own noise. So the
estimator does what it was supposed to do: it turns noise into signal.

**But the signal points at the prior.** `log_total_q`, `log_match_gain_q` and `max_rate` all fall
steadily; `P(fire)` drifts 0.557 -> 0.541 when the reconstruction target needs it to *rise*.

### Why: the linearisation is dominated by the sites that do not matter

`CV1` is `sum_i g_i * d lambda_i / d phi` with `g = grad_a f` evaluated at `a = E_q[a]`. Measured
on a 4-image batch:

| | at the 8 peak sites | at the 918,671 background sites (`lambda < 1e-6`) |
|---|---|---|
| `grad_a log p(x, a)` | +1007.7 | +635.8 |
| `grad_a (-log q(a))` | +1.2 | +19.3 |
| **total, summed over sites** | **+8.07e3** | **+6.02e8** |

The background aggregate exceeds the peak aggregate by a factor of about **75,000**. The control
variate is therefore an accurate description of a direction almost none of the posterior mass
lies in: a first-order expansion at the mean treats all 918,671 near-zero sites as equally
movable, when the actual sample is `a = 0` at every one of them with probability
`1 - lambda ~ 1`. For a *sparse* Poisson field the mean is simply a bad expansion point, and the
continuous extension of the log-pmf is steepest exactly where the rates are smallest.

This is a property of expanding about `E_q[z]`, not a bug in the implementation, and it is
specific to sparse count fields — the method's own examples are low-dimensional discrete models
where the mean is a perfectly reasonable centre.

---

## 14. Next: Diagonalisation SGD

`esennesh/numpyro` has a `feature/diagonalization_sgd` branch implementing Wagner, Khajwal & Ong,
*"Diagonalisation SGD: Fast & Convergent SGD for Non-Differentiable Models via Reparameterisation
and Smoothing"* (AISTATS 2024): `numpyro/contrib/diag_sgd.py` (1068 lines), 917 lines of tests,
and — directly to the point — an `examples/dsgd_pvae.py` **Poisson** VAE.

It replaces a discrete site with a smooth inverse-CDF reparameterisation at temperature `eta`,
annealed on the Theorem 5.6 schedule (`eta_schedule(K, ell, eta_final)`, with `ell` from
`count_layers`), so the smoothed objective converges to the true one as `eta -> 0`. The
"diagonalisation" is doing the SGD steps and the annealing together rather than solving each
smoothed problem to convergence.

Why it addresses §13's failure specifically. DSGD does **not** linearise anywhere. For unbounded
families it uses `adaptive_relaxed_count`, whose relaxed sample is a convex combination of the
integer outcomes weighted by soft one-hot `w_k = sigma_eta(u - F(k-1)) - sigma_eta(u - F(k))`.
A site with `lambda_hat ~ 2e-9` returns `~0` and contributes `~0` — gradient flows through sites
in proportion to their actual probability of firing, not through an expansion that treats all
918,671 near-zero sites as equally movable. Its docstring notes the gradient is "a bounded
weighted sum of `dF(k)` (no `1/density` factor), so it stays low-variance across `eta` and keeps
signal as `eta -> 0`", which is exactly the property the Gaussian/CLT sketch in §11 lacked.

Three details that make the integration cheap:

- **`SmoothedCount.has_rsample = True`** (`diag_sgd.py:626`). `ELBOTracer.setup` classifies sites
  on `site["fn"].has_rsample`, so `a` would flip to the *reparameterized* path automatically —
  no tracer changes, and the RLOO machinery simply stops being used for that site.
- **No static truncation.** The horizon is discovered at runtime by a `lax.while_loop` with a
  `custom_vjp`, so the `1e-9 .. 1` spread of our rate field is fine, and `eta` may be traced —
  a schedule entry can be indexed inside a jitted step without recompiling.
- **The density term is the analytic continuation of the log-pmf** (`k! -> Gamma(z+1)`), i.e.
  the same continuation §13 had to unlock by hand with `validation_enabled(False)` — here it is
  the intended design rather than a workaround.

Blocked on the branch being rebased onto `master`/`develop`; `pyproject.toml:38` currently pins
the fork to `feature/qem`.

---

## 15. The likelihood was wrong, not the inference

DSGD (`feature/diagonalization_sgd`, now rebased onto `develop`) integrated with essentially no
friction. `pyproject.toml` repinned; `mixtures.py`, `elbo.py`, `util.py` and `discrete.py` are
byte-identical between that branch and `feature/qem`, so nothing we depend on moved.
`DSGDMessenger` peels the `Independent`/`Expanded` wrappers off the `.to_event(3)` site, and
because `SmoothedCount.has_rsample = True` the `a` site is reclassified as **reparameterized
automatically** — `{'color': True, 'a': False}` becomes `{'color': True, 'a': True}` with no
change to `tracer.py`. The relaxed count stays genuinely sparse (221 sites above `1e-3` out of
3.7M), so it does not suffer §13's smearing.

At `eta = 0.5` the gradient was `d(loss)/d(log_total_q) = -459.7` — raise the rate, the direction
we wanted and the opposite of DoubleCV's.

**But over an annealed run it reversed.** `log_total_q` climbed while `eta` was large (peaking at
`+0.0069` around step 80, `eta ~ 0.25`) and then fell monotonically as `eta -> 0.05`, finishing at
`-0.0414`. Since DSGD converges to the true objective as `eta -> 0`, that says the *smoothed*
objective wants more firing and the *exact* one wants less.

Three independent estimators now agreed. That is not a coincidence, so the premise was checked
instead.

### The premise was wrong

Sweeping `log_total_q` with `color` pinned to each image's true ink colour:

| `log_total_q` | P(fire) | `E[log p(x|a)]` | ELBO | E[MSE] |
|---|---|---|---|---|
| -1.00 | 0.259 | 427425.9 | **427386.8** | 0.00938 |
| 0.00 | 0.557 | 407866.2 | 407769.7 | 0.00576 |
| +1.00 | 0.890 | 396546.5 | 396303.1 | **0.00311** |
| +1.50 | 0.974 | 396501.5 | 396092.1 | 0.00331 |
| +2.50 | 1.000 | 399769.4 | 398541.6 | 0.00712 |

**The ELBO was maximized exactly where reconstruction was worst.** Every estimator was reporting
this faithfully; the objective itself was the problem.

### Why: a 2-component mixture cannot represent a partially covered pixel

For a pixel at coverage `A`, the data sits at `A*fg + (1-A)*bg` — *between* the two components,
where a 2-component mixture has almost no mass:

| coverage | `log p` mixture | `log p` blend |
|---|---|---|
| 0.00 | 11.06 | 11.06 |
| 0.25 | **-269.72** | 8.72 |
| 0.50 | **-116.27** | 7.85 |
| 0.63 | -60.07 | 7.53 |
| 1.00 | 6.90 | 6.90 |

The mixture is only sane at `A = 0` or `A = 1`. And §0 measured, before any of this was built,
that **95.1% of this dataset's ink pixels are intermediate anti-aliased values** with median
luminance 0.502. Placing a glyph creates hundreds of such pixels, each costing 60-270 nats, so
the ELBO's conclusion — do not place ink — was correct given the likelihood it was handed.

Under `blend` the ELBO is maximized at `log_total_q = +1.50` (P(fire) 0.974) and tracks E[MSE]
instead of opposing it. **The default is now `blend`** — which is what was originally asked for,
before §4 argued its way to the mixture.

### What §4 got wrong, precisely

All three arguments for the mixture were about *gradient quality*: the responsibility ratio
`r_fg/A - r_bg/(1-A)` surviving low contrast, the optimum under uncertainty being two sharp
hypotheses rather than a grey blur, and the ink-dependent variance falling out rather than being
posited. Every one of those still holds. None of them is about whether the likelihood can
**represent the data**, and that question was already answered in §0 by a measurement quoted
repeatedly without being connected to the choice. A likelihood that assigns `exp(-116)` to the
median observed ink pixel cannot be rescued by having good gradients.

The general lesson worth keeping: *three estimators agreeing against your expectation is evidence
about the objective, not about the estimators.*

### Where this leaves DSGD

Untested against a correct objective. It integrated cleanly, produced finite gradients, kept the
relaxed field sparse, and tracked the smoothed objective faithfully — every mechanical property
checked out. Whether it beats RLOO now that the likelihood tracks reconstruction is an open
question and the obvious next run. Note also that `blend`'s penalty is the uncapped quadratic
(`ELBO ~ -6.1e6` at `log_total_q = -1`), where the mixture had `ambient_depth` bounding it at
`-log(ambient)`; if optimisation destabilises, that asymmetry is the first thing to look at.

### Addendum: DSGD on the corrected objective

Same annealed run, `likelihood: blend`:

| | mixture (§14) | blend |
|---|---|---|
| `log_total_q` trajectory | up to +0.0069, then **reversed** to -0.0414 | **monotone up at all 20 checkpoints**, to +0.0470 |
| final P(fire) | 0.497 | 0.551 |
| final E[MSE] | 0.00841 | 0.00786 |

The sign defect is gone. The magnitude is simply the optimizer: `clipped_adamw` runs at
`lr = 1e-4`, and Adam's per-coordinate step is ~`lr` irrespective of gradient scale, so 400 steps
can move a scalar at most `400 * 1e-4 = 0.040`. It moved 0.047. The parameter is travelling as
fast as it can — reaching the sweep optimum of `+1.50` needs on the order of 15,000 steps at this
learning rate, i.e. a real training run rather than a probe.

Open, in priority order:

1. **A full-length run** (~15k steps) on `blend`, to confirm the sweep's optimum is actually
   reached and that E[MSE] approaches the 0.00451 single-spike figure.
2. **RLOO vs DSGD on the corrected objective.** Every comparison in §11-§14 was run against a
   likelihood that rewarded not placing ink, so none of it says which estimator is better now.
   `log_total_q` is a clean one-scalar probe for a rerun of §13's controlled harness.
3. **A per-parameter learning rate**, or simply a larger `lr` for the two guide scalars
   (`log_total_q`, `log_match_gain_q`). They are the slowest-moving quantities in the model and
   they gate everything downstream.
4. **`sigma_bg` as a stability knob** — `blend` pays the uncapped quadratic the mixture's
   `ambient_depth` used to bound.

---

## 16. Three-way comparison on the corrected objective

Same controlled harness as §13 — fixed probe batch, fixed training stream, identical init —
with `likelihood: blend` and `lr` raised from `1e-4` to `1e-3` for all arms, since at `1e-4`
Adam's unit step caps a scalar's travel at `steps * lr` and the comparison would measure sign
only. 1000 steps, 8 particles.

| | `log_total_q` | `log_gain` | P(fire) | **E[MSE]** | s/step |
|---|---|---|---|---|---|
| RLOO (`ELBOTracer`) | 0.0219 | 0.6441 | 0.539 | 0.00791 | 0.37 |
| `DoubleCVTracer` | **1.0246** | 1.0492 | 0.929 | 0.00365 | 0.39 |
| DSGD (+`ELBOTracer`) | 0.9315 | 0.9376 | 0.921 | **0.00263** | 0.63 |
| *sweep optimum / reference* | *+1.50* | | *0.974* | *0.00451* | |
| *blank canvas* | | | | *0.01338* | |

**RLOO is not viable for this model.** Over 1000 steps at 10x the learning rate it still wanders
— `log_total_q` crosses zero five times and ends at 0.0219, `P(fire)` moves 0.557 -> 0.539. The
score-function estimator never gets traction on a 230,400-site field against a single image-wide
reward, exactly as §11 diagnosed; the corrected objective does not rescue it.

**§13's verdict on `DoubleCVTracer` was an artifact of the broken likelihood.** Its `CV1`
gradient at background sites was measured under the mixture as `grad_a log p = +635.8` — the old
likelihood rewarded ink almost everywhere, so the 918,671-site aggregate swamped the 8 real ones
*in the wrong direction*. Under `blend`, painting ink on white paper costs `(x-mu)^2/2sigma^2`,
that term changes sign, and the estimator climbs monotonically at every checkpoint. The
implementation was never the problem. It was faithfully linearising a broken objective.

**Both working estimators beat the one-spike reference**, which turns out not to have been a
ceiling: 0.00451 was measured with `color` drawn from its *prior*, and a trained model with a
fitted ink colour does better.

**Neither has converged, and the scalar comparison is saturated.** `1000 * 1e-3 = 1.0` is Adam's
travel limit, and DoubleCV finished at 1.0246 while DSGD reached 0.9315 after spending ~200 steps
going the wrong way first. Both are pinned at the ceiling, so their *rates* on `log_total_q`
cannot be distinguished here — only the sign and the endpoint quality can.

**The signal that does separate them is reconstruction:** DSGD 0.00263 against DoubleCV 0.00365,
a 28% improvement, at 1.6x the cost per step. In equal wall-clock DoubleCV gets ~1.6x more steps,
so which wins per unit time is genuinely open.

### Recommendation

- **`DoubleCVTracer` as the default.** Nearly as good, 1.6x cheaper per step, already in-repo,
  and no extra dependency on the DSGD branch. The four fixes from §13 are what made it usable.
- **DSGD when reconstruction quality is the priority**, and as the check that the objective is
  being optimised rather than an estimator artifact — its `eta -> 0` limit is what exposed §15.
- **Drop RLOO for this model.** Keep it only as a baseline.
- Next: a longer run (`log_total_q` needs to reach ~1.5) and a per-parameter learning rate so the
  two guide scalars are not the bottleneck for the whole model.

### Addendum: initialising `log_total_q` at the sweep optimum

`log_total_q` is initialised from the *guide's* `expected_count`, which is a different quantity
from the model's: the model's 1.0 is a statement about the data ("one character per image") and
seeds the prior `log_rate`; the guide's seeds the posterior's starting point. They are allowed to
differ, and that difference *is* the KL. Setting the guide's to 4.5 starts `log_total_q` at
`log(4.5) = 1.50` rather than 0, so the optimizer no longer has to walk six orders of magnitude
with one scalar at `lr` per step.

Same harness, 1000 steps, `lr = 1e-3`:

| | `log_total_q` | max `lambda_hat` | P(fire) | **E[MSE]** | s/step |
|---|---|---|---|---|---|
| `DoubleCVTracer` | 1.5041 -> **1.7447** | 5.72 | 0.997 | 0.00387 | 0.42 |
| DSGD (+`ELBOTracer`) | 1.5041 -> **1.1450** | 3.14 | 0.957 | **0.00231** | 0.69 |
| *blank canvas* | | | | *0.01338* | |

**The saturation is gone.** Travel was 0.24 and 0.36 against an Adam ceiling of 1.0, so for the
first time both arms are converging rather than still climbing — DoubleCV plateaus from step 800,
DSGD dips to 1.01 by step 600 and settles back to ~1.15.

**They converge to different optima.** DoubleCV drives the peak site to 5.7 expected spikes
(`A = 0.997`, fully opaque); DSGD settles at 3.1 (`A = 0.957`). Over-inking is the wrong direction
for data whose ink is 95% intermediate rather than saturated, and the 40% MSE gap is consistent
with the bias §13 identified — linearising at `E_q[a]` over-weights the mass of near-zero sites.

The sweep's `+1.50` was therefore a good *initialisation* but not the optimum: DSGD's converged
answer is nearer 1.15, and the sweep was run at untrained parameters.

Net effect: DSGD 0.00263 -> **0.00231**, the best result so far and 5.8x better than a blank
canvas. DoubleCV 0.00365 -> 0.00387, i.e. marginally worse — starting near the optimum let it run
*past* it.

**Caveat on the arbiter.** All of the above ranks by E[MSE], which is a proxy. The fair test is
the ELBO at each endpoint, and these runs did not checkpoint parameters, so whether DoubleCV
genuinely attains a *higher ELBO* at 1.74 (in which case it is right and MSE is the misleading
quantity) or is biased past the optimum is not yet settled. That is a short re-run with
checkpointing, and it should happen before the default is treated as final.

---

## 17. ELBO arbitration: both estimators were fine, the objective has a tail

Ranking by E[MSE] (§16) was a proxy. The clean test is a sweep of `log_total_q` *within* a fixed
parameter set — freezing everything else each arm learned, so the CNN weights, colour head and
`log_sigma_ink` cannot confound it — asking where the ELBO actually peaks. Run inside both arms'
converged parameters, 48 particles, standard errors reported.

| within DSGD's parameter set | ELBO | +/- se | E[MSE] |
|---|---|---|---|
| 1.00 | -65,159 | +/-116,601 | 0.00252 |
| 1.15 *(its own)* | 112,249 | +/-97,151 | 0.00252 |
| 1.50 | 462,697 | +/-32,201 | 0.00279 |
| **1.74** | **518,105** | **+/-464** | 0.00330 |
| 2.00 | 513,073 | +/-402 | 0.00400 |
| 2.30 | 507,926 | +/-381 | 0.00475 |

**Read the standard errors first.** They span three orders of magnitude — `+/-181,280` at
`log_total_q = 0.8` against `+/-363` at 2.30. The ELBO at low rates is dominated by rare
catastrophic draws: when no spike fires the render is blank, and `blend`'s *uncapped quadratic*
charges on the order of `-6e6` nats. Any conclusion drawn from the low-rate region of this or the
§15 sweep is unreliable; only the high-rate end, where the variance collapses, is trustworthy.

Where the estimates are reliable, **the ELBO peaks at 1.74**, and the decline to 2.00 is 5,033 —
about 11 standard errors, so the peak is real. `DoubleCVTracer` converged to **1.758**. It was
optimising the stated objective correctly, and §16's MSE-based lean against it was wrong.

DSGD settles at 1.145 in both runs, systematically *below* the ELBO optimum. A self-consistent
explanation: smoothing replaces integer counts with relaxed ones, so a "miss" still paints partial
ink rather than leaving the canvas blank. That suppresses exactly the catastrophic tail events
driving the ELBO's preference for a high rate, and at `eta = 0.05` the residual softening remains.
DSGD is optimising a slightly different — and arguably better-behaved — objective.

### The real finding

**The ELBO and reconstruction genuinely disagree.** E[MSE] is minimised at 1.00-1.30 in both
parameter sets; the ELBO peaks at 1.74. The cause is `sigma_bg = 0.01`: a missed glyph costs
~`6e6` nats, so the objective buys insurance against blank renders at the price of over-inking
(peak `lambda_hat` 5.7, `A = 0.997`, fully opaque) on data whose ink is 95% *intermediate*
coverage.

That is a property of the likelihood, not of any estimator. §15 replaced a likelihood that could
not represent partial coverage with one that can; this is the milder, remaining version of the
same class of problem — the blend represents partial coverage correctly but weights its tail so
heavily that miss-avoidance dominates render accuracy.

### Decision and follow-ups

- **`DoubleCVTracer` stays the default.** Cheaper per step and the more faithful optimiser of the
  objective as written.
- **Tame the tail before tuning anything else.** Options, in order of appeal: raise `sigma_bg`
  (it is a fixed config constant precisely so it can be tuned — 0.01 was chosen for optimisation
  comfort, not fitted); reintroduce a cap analogous to the mixture's `ambient_depth`; or make the
  observation model heavy-tailed (Student-t) so a blank render is expensive but not catastrophic.
- **Re-run §16 afterwards.** If the ELBO optimum moves to where MSE is minimised, the estimator
  comparison becomes meaningful again and DSGD's advantage may reappear.
- Report ELBO estimates with standard errors from now on; on this objective a mean over 48
  particles can be worthless.

---

## 18. A heavy tail fixes the ELBO/reconstruction disagreement

`observation_df` selects the observation tail: `null` gives a Normal, a finite value gives a
Student-t with that many degrees of freedom, rescaled by `sqrt((df-2)/df)` so `sigma_bg` and
`sigma_ink` keep their meaning as standard deviations and *only* the tail changes.

Cost of a residual at `sigma_bg = 0.01`, per channel:

| residual | Normal | Student-t df=3 | df=5 |
|---|---|---|---|
| 0.01 | -3.2 | -2.8 | -3.0 |
| 0.10 | 46.3 | 5.1 | 6.7 |
| 0.50 | **1246.3** | **11.5** | 16.3 |
| 0.90 | 4046.3 | 13.8 | 19.8 |

Quadratic and unbounded against logarithmic. Small residuals are barely affected, which is the
point: the tail changes, the fit does not.

Sweeping `log_total_q` with `color` pinned to the true ink colour, 32 particles:

| tail | ELBO peak | MSE min | verdict | se range |
|---|---|---|---|---|
| Normal | 2.00 | 1.30 | **disagree** | 5,340 .. 277,327 |
| Student-t df=5 | 1.30 | 1.30 | **agree** | 636 .. 2,245 |
| Student-t df=3 | 1.30 | 1.30 | **agree** | 550 .. 1,534 |

Two wins, and the second may matter more than the first.

**The objective now optimises what we want.** ELBO and reconstruction agree exactly at 1.30
under both df values, where the Normal put the ELBO optimum at 2.00 — over-inking to `A ~ 0.997`
purely to insure against blank renders.

**The estimator variance collapses by ~180x** at the worst point (277,327 -> 1,534). That is not
a side effect: a heavy-tailed likelihood means no single draw can dominate the average, so the
ELBO *and every gradient estimator built on it* become dramatically better conditioned. §17's
observation that a 48-particle ELBO mean could be worthless was a symptom of the Normal tail, not
an inherent property of the problem.

`observation_df: 3.0` is now the default; `df = 5` gives the same optimum with slightly higher
variance. Fixed rather than learnable, on the same reasoning as `sigma_bg`: a learnable df has a
standing incentive toward heavier tails simply to forgive its own errors. ELBO *values* are not
comparable across tails — only the location of the peak and the standard errors are.

**Worth re-running now:** §16's estimator comparison was conducted against the Normal tail, so
DoubleCV's advantage there was partly an advantage at optimising miss-avoidance. With the
objective realigned and the variance collapsed, DSGD's reconstruction edge may reappear as a
genuine ELBO edge.

---

## 19. Open question: when does sparsity actually help?

Worth thinking about properly, because this design assumes an answer it has not tested. Three
distinct senses got conflated throughout, and the evidence in this note separates them.

**(a) Sparsity as truth.** The captcha generator really does place one glyph. Here sparsity is a
*correctness* property, not an accuracy one — and it bought nothing measurable for reconstruction.
Localization was solved by the matched filter `Phi^T x` (§12), a correlation that would work just
as well against a dense overcomplete code; reconstruction quality came from the right dictionary
and, eventually, the right likelihood (§15, §18). Nothing in E[MSE] 0.0023-0.0039 against a blank
baseline of 0.0134 is attributable to the *prior* being sparse.

**(b) Sparsity as prior / regulariser.** Measured to be **inert**. §9 recorded a KL of order
`1e-6` against a likelihood of order `1e4`. The sparse Poisson prior at rate `4.3e-6` exerts no
pressure whatsoever; the sparsity of the fitted model comes from `log_total_q` and `log_rate` —
learned parameters — not from the prior. That is not a tuning failure but an arithmetic one: one
19,200-dimensional observation against a 230,400-dimensional latent field means the likelihood
outweighs the prior by orders of magnitude. **Prediction:** a sparse prior only regularises when
the KL is commensurate with the likelihood, i.e. when data is scarce relative to latent dimension
or when the latent is *shared* across many observations. The obvious place to look is the
dictionary — shared across all 5000 images — rather than the per-image count field.

**(c) Sparsity as exploitable structure.** This is the one place it genuinely paid, and the
payoff is computational rather than statistical. §6 rung 3 — true deformable convolution, each
stamp warped by its own offset field — is affordable *only* because `E[M] = expected_count`, so
one gathers `M` stamps instead of 230,400. The same applies to any per-instance attribute
(per-stamp colour, depth ordering) and to anything symbolic downstream ("a Q at (20,15)"). A
dense or relaxed code has no support to gather. If sparsity earns its place in this project on
current evidence, it earns it here.

**And where sparsity actively hurt: inference.** §11's needle-in-a-haystack (a uniform spike
lands usefully with probability ~`1e-3`, so RLOO never got traction) and §13's linearisation
dominated 75,000:1 by 918,671 near-zero sites are both *caused* by sparsity. Any estimator that
treats sites symmetrically drowns in the empty ones. Both were fixed by not using the prior to
search at all — the matched filter is classical sparse-coding practice (matching pursuit, ISTA)
and it exploits the *dictionary*, not the sparsity.

### The regime map, as a hypothesis

| regime | helps reconstruction? | helps inference? |
|---|---|---|
| true process sparse, dictionary known | no — correctness, not accuracy | **hurts**, unless the method exploits the support |
| true process sparse, dictionary *learned* | plausibly — sparsity is what identifies the dictionary | hurts more |
| overcomplete dictionary, dense truth | yes — compression and identifiability | hurts |
| hierarchical, layer-1 sparse feeding layer-2 | **the untested case, and the reason this design exists** | ? |

### The test that would settle it

The hierarchical row is the claim `poisson_hesc.py` actually makes: `ahat0 = Phi1 a1` is a
*prediction* of layer-1 rates from layer-2 counts. If layer 1 were dense, that prediction would
be nearly uninformative — a high-entropy field of which the top-down term explains little. The
argument for sparsity is that it is what makes the top-down prediction carry information, i.e.
what turns `KL(q(a0) || p(a0 | a1))` into a real training signal for layer 2 rather than the
inert `1e-6` measured in (b).

Concretely: **build the two-layer model and measure whether the layer-2 KL is commensurate with
the likelihood.** If it is, sparsity is earning its keep statistically and the whole design is
vindicated. If the same inertness recurs one level up, then sparsity in this project is purely a
computational convenience for §6 — still worth having, but a much weaker claim than the one the
architecture implies.

A cheaper preliminary: replace the Poisson field with a dense continuous code of the same
dimension and compare reconstruction. If E[MSE] is unchanged, (a) is confirmed and sparsity is
buying nothing for reconstruction in the single-layer setting.

---

## 20. Opacity-coupled degrees of freedom: a negative result

`observation_df` now accepts `None` (Normal), a scalar (flat Student-t), or a `(nu_bg, nu_ink)`
pair interpolated with opacity exactly as `_ink_scale` interpolates the scale, plus `learn_df`
to make the endpoints learnable. Interpolating the *excess over 2* keeps `nu > 2` — and the
variance finite — by construction.

The motivating argument: `nu` is a statement about uncertainty in the *scale* (Student-t is a
scale mixture of normals, `x | w ~ N(mu, sigma^2/w)`, `w ~ Gamma(nu/2, nu/2)`; equivalently the
conjugate Normal-Inverse-Gamma posterior predictive has df equal to the pseudo-count behind the
variance estimate). Our two scales are known very unequally — `sigma_bg` is effectively measured,
`sigma_ink` has never been estimated — so heavy tails ought to belong on ink and light tails on
background.

Sweeping `log_total_q` with `color` pinned to the true ink colour, 32 particles:

| configuration | ELBO peak | se range |
|---|---|---|
| Normal | 2.00 | 6,372 .. 221,349 |
| **flat df=3** | **1.30** | **715 .. 1,268** |
| flat df=5 | 1.30 | 831 .. 1,763 |
| `nu(A)` bg=30, ink=3 | 1.74 | 1,199 .. 7,596 |
| `nu(A)` bg=10, ink=3 | 1.50 | 842 .. 3,117 |
| `nu(A)` **reversed** bg=3, ink=30 | 0.80 | 1,405 .. 1,710 |

**The argument was wrong, and the reversed control says exactly how.** Coupling in the motivated
direction moves the ELBO peak *away* from the reconstruction optimum (1.30 -> 1.74) and inflates
the variance 6x, monotonically in `nu_bg`. Reversing it — heavy tails on background — pushes the
peak down to 0.80. So the tail carrying the effect is the **background** one.

In hindsight the outliers are not where the argument assumed. The catastrophic residuals are not
ink pixels whose scale is poorly known; they are pixels where **the model placed no ink and the
data has ink**. Those have `A ~ 0`, so `nu(A)` labels them background and hands them the *light*
tail — the forgiving tail lands precisely where the outliers are not. Indexing `nu` on the
model's own prediction cannot work for this failure mode, because the prediction is what is
wrong. A flat heavy tail covers them regardless of current belief.

**`observation_df: 3.0` (flat) stays the default** — best of the seven, with the tightest
standard errors. The pair form and `learn_df` remain available.

Two readings to correct:

- The "MSE minimum" moved between 1.00 and 1.30 across sweeps (0.00316 vs 0.00311, ~1.6%),
  which is seed noise. The MSE curve is flat over that interval and the flat-df ELBO peak at
  1.30 sits inside it; only the Normal's 2.00 is a real disagreement.
- An earlier note here speculated that a learnable `nu` would run to its lower boundary, since
  heavier tails forgive errors. That is wrong, and the measured tail table refutes it: at a
  residual of 0.01 the Normal costs -3.2 nats against df=3's -2.8, so the **Normal is better at
  small residuals**. With 95% of pixels at near-zero residual, mode sharpness pulls `nu` up while
  the tail pulls it down, giving a genuine interior optimum. `nu` is better identified here than
  it usually is, and `learn_df: true` on the flat scalar is worth running.

---

## 21. Coupling df to the spike count: the direction was backwards in §20

§20 concluded that opacity-coupled degrees of freedom made things worse and that a flat `df = 3`
was best. Both halves need correcting. The arm labelled "REVERSED" there — fat tails where *few*
spikes touch a pixel, near-Normal where many do — was run as a falsification control for a
hypothesis of mine, and its result was read as confirming that the mechanism ran through the
background tail. It does. What went unnoticed is that the same row also scored *better* than the
incumbent on the criterion that matters, and that it peaked at the grid's lower boundary, so its
optimum had not actually been located.

Re-run with the grid extended to 0.0 and both couplings — `nu(A)` (saturating) and
`nu(tau) = nu_0 + kappa tau` (unbounded, since `tau` is the alpha-weighted spike count):

| configuration | ELBO peak | MSE min | agreement | se range |
|---|---|---|---|---|
| flat df=3 (incumbent) | 1.30 | 1.00 | 0.30 | **715 .. 1,470** |
| `nu(A)` 3 -> 30 | 0.80 | 1.00 | 0.20 | 1,357 .. 1,710 |
| **`nu(A)` 3 -> 10** | **1.00** | 1.00 | **exact** | 998 .. 1,461 |
| `nu(tau)` 3 + 1.0 tau | 1.30 | 1.00 | 0.30 | 839 .. 1,475 |
| `nu(tau)` 3 + 3.0 tau | **1.00** | 1.00 | **exact** | 1,021 .. 1,456 |
| **`nu(tau)` 2.5 + 2.0 tau** | **1.00** | 1.00 | **exact** | **910 .. 1,340** |

**No arm peaked at 0.0**, so a fat background tail does not abolish the pressure to place ink —
the over-correction this was expected to risk does not occur. Three arms land the ELBO peak
*exactly* on the reconstruction optimum, which no previous configuration has done.

**Strength matters more than the coupling variable.** `tau` and `A` are monotonically related, so
the choice between them mostly rescales `kappa`; what separates the rows is how aggressive the
schedule is. `nu -> 30` overshoots to 0.80; moderate settings (`nu -> 10`, or `kappa ~ 2-3`) hit
1.00 exactly.

**Why this direction and not §20's.** The catastrophic residuals are at pixels the model left
blank and the data inked. Those have `tau ~ 0`, so a df that *rises* with `tau` gives them the
fattest tail and forgives them; §20's schedule gave them the lightest tail, which is why it
reverted toward the Normal's miss-avoidance behaviour. The scale-uncertainty argument that
motivated §20 was about where `sigma` is poorly known; the operative question is where the model
is *wrong*, and those are different pixels.

`observation_df: [2.5, 4.0]` with `df_couples_to: depth` is now the default. Two caveats kept in
view: flat `df = 3` still has the tightest standard errors overall (715 against 910), so the
coupled version buys alignment at roughly 30% more variance; and the incumbent's "0.30 off" is
partly illusory, since E[MSE] at 1.00 and 1.30 differs by 1.6% — seed noise — so its peak already
sat inside the flat region. This is a real improvement, and a modest one.

---

## 22. Estimator comparison on the corrected objective, and `learn_df`

Both prior comparisons (§16, §17) ran against likelihoods that have since changed twice — §18
gave the observations a Student-t tail, §21 coupled its degrees of freedom to optical depth. Rerun
on the current objective: `blend` + `nu(tau) = 2.5 + 2.0 tau`, whose ELBO peak sits at
`log_total_q = 1.00`.

The guide still initialises `log_total_q` at 1.50 (`expected_count: 4.5`, chosen when the optimum
was thought to be 1.50), so every arm now has to travel **down**. That is a sharper test than
§16, where all three had to climb and a merely upward-biased estimator would have looked good.

| | `log_total_q` (optimum 1.00) | max `lambda_hat` | E[MSE] | s/step |
|---|---|---|---|---|
| RLOO (`ELBOTracer`) | 1.5507 — moved *away* | 3.28 | 0.00555 | 0.39 |
| `DoubleCVTracer` | 1.2456 | 3.47 | 0.00392 | 0.40 |
| **DSGD (+`ELBOTracer`)** | **1.0986** | 3.00 | **0.00299** | 0.69 |
| blank canvas | | | 0.01338 | |

**This inverts §17.** DSGD is now closest to the ELBO optimum *and* best on reconstruction — the
two criteria agree, which is what §18 and §21 were for. RLOO drifted the wrong way and its E[MSE]
degraded over the last 400 steps (0.00420 at step 300 to 0.00555 at 1000).

**The inversion is explained by §17's own hypothesis.** There, DSGD undershot the ELBO optimum and
the proposed reason was that smoothing replaces integer counts with relaxed ones, so a "miss"
still paints partial ink and the catastrophic tail events driving the ELBO upward are suppressed.
If that is the mechanism, then a likelihood that suppresses those same events *explicitly* should
remove the discrepancy — and it does. **DSGD's bias under the Normal tail was an implicit
robustification.** With an explicitly robust likelihood the bias disappears and DSGD becomes the
most accurate of the three. Two independent routes to the same correction agreeing is decent
evidence the mechanism is real.

### `learn_df` does not work

Fitting `(nu_0, kappa)` rather than picking them off §21's grid:

```
nu_0    2.500 -> 2.164     (floor is 2.0)
kappa   2.000 -> 1.205
E[MSE]  0.00398   against   0.00395 for fixed (2.5, 2.0)
```

Both drift monotonically downward with no sign of settling — the per-100-step deltas
(-0.050, -0.044, ... -0.020) are decelerating but still nonzero at step 1000 — and buy nothing.

§20's addendum argued this would be *identified*, on the grounds that the Normal beats the t at
small residuals (-3.2 against -2.8 nats at `r = 0.01`) so mode sharpness on the 95% of near-zero
residual pixels would pull `nu` up. That argument was about a **flat** `nu` and does not transfer:
`nu_0` governs only the low-`tau` pixels, which are exactly where the model wants forgiveness for
its own misses. The forgive-your-own-errors pull wins. `learn_df: false` stays, on the same
footing as `sigma_bg`.

### Open

DSGD is not a config-level default: it is a program transformation (`dsgd(model)`, `dsgd(guide)`)
plus an `eta` schedule driven across training, which the probes here manage by hand in 20 stages,
rebuilding the learner's jitted step each time. Making it the shipped default means plumbing
`eta` annealing into `GraphicalModelLearner` or the trainer, and costs 1.7x per step. It is now
best on both criteria, so it is worth doing; `DoubleCVTracer` remains the correct default until
then.

---

## 23. The dark fringe: two causes, and the smaller one was mine

The trained reconstructions showed a dark outline around every glyph where the data
has a clean anti-aliased edge. Decomposed at four overlapping spikes with a
realistic ink colour, the edge error of 0.266 splits as:

| cause | contribution |
|---|---|
| anti-aliasing applied twice, as opacity *and* as colour | 0.067 |
| `A = 1 - exp(-n alpha)` exceeding the true coverage `alpha` | **0.199** |

**The first is a bug, now fixed.** `_rgba_shape_transform` derives `alpha` as
`file_alpha * max(RGB)` while leaving RGB untouched, so on a glyph's support
`rgb.max(-1)` equals `alpha` exactly. Premultiplying by raw RGB and dividing back
out as `tint = premult / depth` recovers that ramp *as the foreground colour*:
`tint = 0.14` at edges against 0.996 at the core. `_ink_kernel` now premultiplies by
unit hue, which puts the ramp in alpha alone. Worth 28% of E[MSE] at 800 steps
(0.00407 -> 0.00292) and it lets the model raise `log_total_q` from 1.07 to 1.31,
since there is no longer a dark fringe to pay for.

**The second is structural**, and it is the larger share. `tau = n alpha` scales the
entire ramp, so the count cannot saturate the core without over-opacifying the edge:

| n | true `alpha` at edge | `A` | edge composite | correct |
|---|---|---|---|---|
| 1 | 0.155 | 0.140 | 0.888 | 0.876 |
| 4 | 0.140 | 0.390 | 0.688 | 0.888 |

At `n = 1` the edge is nearly exact and the core reaches only 0.63; at `n = 4` the
core saturates and the edge darkens. The converged model runs at total rate 4.2, so
it sits in the second regime. This also retires an idea from §12 from a new angle: a
learnable gain would not have helped, because gain and count are the same knob.

**The fix for it, not yet implemented.** Beer-Lambert is the wrong law for a *single*
glyph's anti-aliasing: `1 - exp(-alpha)` maps a fully covered pixel to 0.632 rather
than 1, which is exactly why `n > 1` is needed. Anti-aliasing is *coverage*, which
composes as `1 - prod(1 - alpha_i) = 1 - exp(sum log(1 - alpha_i))`. So convolving
**`-log(1 - alpha)`** in the depth channel instead of `alpha` makes
`1 - exp(-tau)` exact alpha compositing: one stamp gives `A = alpha` precisely, `n`
stamps give `1 - (1 - alpha)^n`, and `alpha = 1` gives `A = 1`. One convolution
still, non-negativity still free (`-log(1-alpha) >= 0`), and §2's "counts add in
log-transmittance" becomes literally true rather than a first-order approximation.

It changes what `tau` means, so `ambient_depth`, `nu(tau) = 2.5 + 2.0 tau` and the
converged `log_total_q` all need re-deriving, and it needs a retrain to evaluate.

**Done in §24, which corrects this paragraph on two counts.** `ambient_depth` did
*not* need re-deriving — it is a floor at `tau = 0`, and zero spikes is zero depth
under either convention. And `sigma_ink_init`, which is not listed here, needed it
most: 0.04 -> 0.01. The retrain is also where the prediction two paragraphs below
finally resolved, in the direction predicted.

### Method note

The first diagnosis of this attributed the whole fringe to the tint bug. That
measurement used `color = 0`, where `tint * 0 = 0` and the tint term cannot
influence the result at all -- the effect being diagnosed was invisible in the
numbers used to diagnose it. Choosing a probe value that happens to annihilate the
term under test is easy to do and hard to notice; the decomposition above uses
`color = 0.2` instead.

A prediction that has *not* been confirmed: `sigma_ink` was expected to fall after
the fix, as independent evidence that it had been absorbing this systematic edge
error as noise. At 800 steps it is identical across arms (0.0936 vs 0.0937), both
still climbing from 0.04. Untested, not supported.

---

## 24. Re-deriving tau, and the colour bistability it exposed

§23 closed by saying `ambient_depth`, `nu(tau)` and the converged `log_total_q` all
needed re-deriving after c1b95bf, and that it needed a retrain to evaluate. The retrain
(`2026-08-04_00-05-57`) came back with **perfect glyph identification and placement and
a uniform grey ink colour** — reconstruction chroma 0.02 against 0.39–0.69 in the data.
This section re-derives the three quantities and, on the way, explains the grey.

Two of the three needed changing. One did not, and one was never wrong in the first
place.

### `observation_df` was never mis-set; `nu` still moved 3.6x

`_observation_df` interpolates the **excess over 2** for *both* entries, so a `depth`
pair reads `(nu_0, kappa + 2)`. `observation_df: [2.5, 4.0]` therefore is
`nu = 2.5 + 2.0 tau` — exactly §21's measured default. There was no drift, and the
config comment agreed with the value all along. (Read it as `2.5 + 4.0 tau`, as I first
did, and every number downstream comes out wrong.)

What *did* change is `tau`. Depth per fully-covered stamp went `1.0 -> 6.908`, so at one
spike on one glyph:

| | nu at a glyph core |
|---|---|
| pre-c1b95bf, `2.5 + 2.0 * 0.999` | 4.5 |
| post-c1b95bf, `2.5 + 2.0 * 6.908` | 16.3 |

3.6x more Normal than the schedule was tuned for, with no config change. Worse,
`tau = -log(1 - A)` is now unbounded, so a depth-coupled schedule's strength depends on
the `1 - 1e-3` alpha clip inside `_ink_kernel` — an implementation detail nobody would
think to re-derive against.

**Decision: `df_couples_to: opacity`, `observation_df: [3.0, 10.0]`.** §21 measured
`nu(A) 3 -> 10` as co-equal best with `2.5 + 2.0 tau` (both landed the ELBO peak exactly
on the reconstruction optimum) and observed that `tau` and `A` are monotonically related
so the choice mostly rescales strength. That tie is now broken by invariance: `A` is
bounded and its schedule is scale-free in `tau`, so it survives the next change to the
depth convention. The equivalent depth setting, if wanted back, is `[2.5, 2.29]`.

### `ambient_depth` did not need re-deriving

The blanket claim in §23 was wrong about this one. `ambient_depth` is a floor at
`tau = 0`, and zero spikes is zero depth under either convention, so `A_min` stays
`1 - exp(-1e-4) = 1e-4` and the cost cap on an unexplained ink pixel stays 9.21 nats.
Unchanged in meaning *and* value. Worth stating explicitly, because "everything in tau
units is suspect" is the kind of heuristic that generates busywork.

### The rate: 2.72 -> 2.0, and how to read `log_total_q`

Method is §21's, with an oracle placement so opacity is the only thing moving: true
glyph identity from the filename, matched-filter argmax location, closed-form ink
colour, `sigma_ink` profiled out at every point since it is learnable. 96 images.
(Rendering places the kernel directly at the oracle sites rather than running
`conv_transpose` over 230,400 sites; the two agree bit-for-bit.)

| spikes/glyph | 1 | 2 | 3 | 4 | 6 | blank |
|---|---|---|---|---|---|---|
| E[MSE] | **0.000370** | 0.000880 | 0.001567 | 0.002141 | 0.002952 | 0.026426 |
| mean `A` on true ink | 0.792 | 0.863 | 0.899 | 0.921 | 0.946 | — |

One spike wins by 2.4x, which is the whole point of c1b95bf: `A = alpha` exactly, at
core and fringe alike. `expected_count: 2.72` is a leftover from the renderer where a
single stamp reached only `A = 0.63`.

**`log_total_q` is spikes per IMAGE, not per glyph.** It is the total rate summed over
all `H*W*K` sites, and it reduces to "spikes per glyph x glyphs" only because the
learned allocation is very nearly a delta. Measured on both checkpoints:

| | `log_total_q` | mass within 1px of a correct site | sites holding 90% | spikes/glyph |
|---|---|---|---|---|
| 2026-08-03 | 1.434 | 101.7% | 1 | 2.13 |
| 2026-08-04 | 2.393 | 97.2% | 2 | **5.32** |

So the 08-04 run was at 5.3 spikes per glyph — fringe pixels at `A = 0.938` against a
true 0.408 — not the 11 a naive reading of `exp(2.393)` gives. Sandbox captchas carry
**two** glyphs, not one; the model config's `expected_count: 1.0` and its comment
("exactly one character") were both wrong, independently of any of this.

**Target: 2.0 on both sides** (`log_total_q = 0.693`) — **superseded by §25, which
corrects this to 4.0.** The sweep above holds the count *fixed*; the ELBO takes an
expectation over `q(a)`, and a Poisson at `lambda = 1` renders nothing 37% of the time.

### `sigma_ink_init`: 0.04 -> 0.01

With a correct colour at one spike per glyph, the profiled optimum is **0.006** and the
residual sd at inked pixels is **0.0102**. So 0.04 was ~5x high. It lands next to
`sigma_bg = 0.01` because exact compositing leaves no systematic edge error for it to
absorb — which *is* the confirmation §23 predicted and could not find at 800 steps.

The 08-04 run's converged 0.200 is not evidence against this. Pinning the colour to the
grey that run proposed moves the profiled optimum to **0.187**. `sigma_ink` inflation was
a symptom of the colour collapse, not a measurement of ink noise.

### What the sweep did NOT find, and the colour

The working hypothesis going in was that a mis-scaled `nu(tau)` rewarded over-inking:
`nu` rises with `tau`, 95% of pixels sit at near-zero residual where higher `nu` scores
better, so spikes buy density. **The sweep refutes it.** All eight tail schedules put the
likelihood optimum at one spike per glyph, by thousands of nats, *including* with the
colour pinned to grey:

```
                        n=1       n=2       n=3       n=4       n=6
depth (2.5, 4.0)       +0.0   -5063.4   -5811.9   -6154.7   -6488.1
opacity (3.0, 10.0)    +0.0   -4798.5   -5563.6   -5918.4   -6260.9
flat 3.0               +0.0   -4664.6   -5406.3   -5739.4   -6053.8
```

So 5.3 spikes/glyph is **not the likelihood's preference** and the tau recalibration does
not explain it. That is a smaller fix than §23 implied, and it leaves the over-inking
attributable to the guide side — estimator or KL — not the objective.

The colour, though, is now fully diagnosed, and it is not an amortization *ceiling*:

| mean-pooled backbone features -> true ink colour | held-out `R^2` |
|---|---|
| untrained, random init | 0.86 / 0.67 / 0.41 |
| 2026-08-03 | **0.993 / 0.994 / 0.993** |
| 2026-08-04 | **-0.32 / -0.36 / -0.36** |

The colour is present at initialization. The 08-03 run *sharpened* it and read the ink
colour to 0.027 mean absolute error. The 08-04 run *destroyed* it, ending **below random
init** — a trained invariance, not a missing capacity. Same architecture, same data,
opposite outcomes: a bistability.

Two properties make one of those basins absorbing. `_valid_num_groups(32, 32) == 32`, so
the backbone's first `GroupNorm` is instance norm: a first-layer map is `a * mask + b`
with the colour amplitude in `a`, and normalizing per channel over space returns
`sign(a) * (mask - mean) / std(mask)`, independent of `|a|` and `b`. Cross-channel
amplitude ratios *are* hue, and only their signs survive. And `PoissonRateHead.scores`
uses deliberately colour-invariant evidence, so the shared backbone's one strong gradient
wants colour invariance — Adam's second moments put the gradient RMS at 9.1 on the rate
head against 8.5e-3 on the colour head's input layer, and `clip_by_global_norm` preserves
the ratio. Once the features go, the head can only emit the dataset marginal, and the
grey it emitted (0.522, 0.531, 0.517) is exactly that.

**Fix: `InkColorFinder`**, which takes the colour off that competition entirely. `A_hat =
max_c (1 - x_c)`, weight `A_hat ** 8`, average the image under it — the coverage-weighted
mean, which reports the ink colour undiluted at pixels where `A = 1`. Exact to **0.002**
with no fitting, `R^2 = 0.9998`, and it recovers grey and black ink correctly, which a
per-pixel inversion `1 - (1 - x_c)/A_hat` would drive to full saturation. The learned
part is a zero-initialised logit residual plus a per-image concentration, so step 0 *is*
the closed form. Same division of labour `PoissonRateHead` already uses for placement:
the statistic comes from the pixels, the network corrects it.

### Also fixed: `DoubleCVTracer` deleted the entropy gradient

`elbo = (logp_total - sg(logq_total)) + (cv2 - sg(cv2))` detached `log q` for *every*
site. That is right for a score-function site — `E_q[grad_phi log q]` is identically zero,
so dropping it is unbiased and sheds variance, and CV2 already carries the whole
integrand. It is wrong for a reparameterized one, where `log q` also depends on `phi`
through the sampled *value*, and that pathwise piece is the entropy gradient.
`ELBOTracer.log_weights` had always made the distinction site by site; `DoubleCVTracer`
did not.

Checked against a 200k-sample reference on a Beta + Poisson toy:

| | `d(ELBO)/d log_a` | `d log_b` |
|---|---|---|
| reference | -0.3140 | +0.5083 |
| old (log q fully detached) | **+0.1730** | +0.2742 |
| fixed | -0.3324 | +0.5626 |

A **sign error** on the concentration: the old estimator pushed `q` sharper when the ELBO
wanted it broader. That explains the *confidence* of the colour collapse (`c1 + c0 = 586`,
pinned at the prior mean) but not its location, and note it cannot be the trigger for the
regression — the tracer was identical in both runs, and the 08-03 colour was fine.

### Method notes

Two probes in this work were wrong in the way §23's method note warns about, both by
using `1 - max_c x_c` where coverage is `max_c (1 - x_c)`. The first reported that 27% of
the dataset carries no ink (it is ~0%; all 5000 images have ink). The second built
synthetic recolourings whose "cores" were only half covered, which annihilated the very
assumption under test and made a correct estimator look 0.43 off. `max_p A_hat` is *not*
a test of `A = 1` either — it maxes out at `1 - min_c c_c`, so its dataset median of
0.808 says the inks are not pure primaries, not that coverage is partial.

The general lesson is the same one §23 recorded, and it keeps recurring: when a probe
disagrees with a derivation, suspect the probe's construction before the code.

---

## 25. A Poisson cannot say "exactly one stamp", and what that costs

Prompted by an observation on the retrained notebook: in 5 of 30 validation images the
two glyphs appear to have *slightly different colours*, even though `color` is a single
global latent per image and the data's two glyphs are bit-identical in colour (measured:
per-glyph colour difference is exactly 0.000 in all 30).

**It is not a colour difference.** Under the blend, `1 - x = A (1 - c)`, so two glyphs of
the same colour at different opacity give *parallel* absorption vectors. Measured on the
five affected figures:

| fig | cos angle between the glyphs' absorption | hue err g1 | hue err g2 | A(g1) | A(g2) | ratio |
|---|---|---|---|---|---|---|
| 00 | 1.00000 | 1.54° | 1.52° | 0.691 | 0.921 | 1.334 |
| 01 | 1.00000 | 0.08° | 0.00° | 0.890 | 0.667 | 0.749 |
| 11 | 0.99999 | 1.24° | 1.35° | 0.688 | 0.914 | 1.327 |
| 17 | 0.99999 | 1.62° | 1.34° | 0.676 | 0.899 | 1.328 |
| 28 | 1.00000 | 0.66° | 0.71° | 0.681 | 0.907 | 1.333 |

Identical hue to five decimal places, and each glyph within 1.6° of the *true* ink colour.
The whole effect is opacity, and the ratio is **4/3** (or its reciprocal) in every case.
Four particles, `predictions["obs"]["ev"].mean(axis=0)`: one glyph received a spike in all
four draws and the other in three. Desaturating toward white by 25% reads as a paler
colour, so the perception is correct and its cause is not.

### The structural part

`q(a)` is a product of Poissons, and

```
max_lambda P(n = 1) = 1/e = 0.368,   at lambda = 1
```

A Poisson **cannot** concentrate on one. So per-glyph opacity variance is irreducible in
this parameterization, not an artifact of undertraining. Worse, it interacts with c1b95bf:

```
E_n[A] = 1 - E[exp(-n d)] = 1 - exp(-lambda (1 - e^-d)) = 1 - exp(-lambda alpha)
```

which is **exactly the pre-c1b95bf saturation curve** with `lambda` in place of `n`. The
exact-compositing property survives a fixed `n = 1` and is given back in expectation: the
`lambda` solving `E[A] = alpha` is 6.91 at a core and 1.28 at a fringe, so no single rate
works. §23's structural failure returns, one level up.

### Consequence for the rate: 2.0 -> 4.0

§24 derived `expected_count: 2.0` from a fixed-count sweep. That is the wrong functional.
Marginalising the count over the oracle placement, 64 images, recalibrated tail:

| lambda/glyph | 0.5 | 1.0 | 1.5 | **2.0** | 2.5 | 3.0 | 4.0 | 5.3 | 7.0 |
|---|---|---|---|---|---|---|---|---|---|
| `E_q[log p]` | 62771 | 66970 | 68630 | **68989** | 68719 | 68179 | 66944 | 65611 | 64462 |
| MSE, per sample | .01570 | .00977 | .00631 | .00433 | .00323 | .00266 | **.00233** | .00252 | .00294 |
| MSE of `E[image]` | .00953 | .00357 | .00148 | .00089 | **.00087** | .00106 | .00158 | .00221 | .00282 |
| P(miss a glyph) | .845 | .600 | .396 | .252 | .157 | .097 | .036 | .010 | .002 |

`E_q[log p]` peaks at `lambda = 2` per glyph, so **4.0 total**. The second spike buys
reliability of *presence*, not opacity: at `lambda = 1`, 60% of samples miss at least one
glyph, and a missed glyph is expensive even with the fat background tail. It is paid for
by over-inking whenever the draw exceeds one — the two failure modes are traded against
each other and cannot both be avoided.

This also substantially closes the gap §24 left open. The 2026-08-04 run's 5.3
spikes/glyph is not 5x the optimum, it is ~2.5x, and note that 4.0 total is essentially
the 2026-08-03 run's converged 4.2. What looked like a mystery was mostly my having
derived the target from the wrong functional.

### What would actually fix it

The proposal need not be Poisson — `q` is free, only the model's prior is Poisson. A
Bernoulli or Binomial proposal over `{0, 1}` (or a small truncated count) would put ~1.0
on `n = 1`, removing the opacity variance, keeping `A = alpha` exact per sample, and
paying only a bounded KL against the Poisson prior. `ELBOTracer` already routes
non-reparameterized sites through its score-function surrogate, so nothing downstream
changes. Untested, and it is the obvious next experiment.

Two smaller notes. The `1 - e^-1` presence problem is also an argument for the sub-pixel
offsets of §6: part of what extra spikes are currently buying is coverage of placement
uncertainty. And for reading reconstructions, `ev.mean(axis=0)` blends across posterior
count draws, so a genuinely uncertain glyph renders pale rather than sometimes-absent —
plotting a single particle, or the per-particle spread, shows what the posterior actually
believes.
