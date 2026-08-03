# Min-sum / Find-Weigh-Learn on the captcha model — session record

**Dates:** 2026-07-29 to 2026-07-30
**Branch:** `feature/minsum` (rebased onto `origin/develop` @ `17254d8` mid-session)
**Goal as stated:** run the numpyro fork's min-sum message-passing inference on the captcha
model, in a notebook similar to `notebooks/captcha1-marionette.ipynb`.

This is a reconstructed record of the working session, not a byte-exact transcript. Numbers,
file/line references and commands are as measured. Nothing here is committed except where
noted.

---

## 1. What the "shiny new inference method" actually is

`pyproject.toml` pins `numpyro` to `esennesh/numpyro` @ branch
`claude/inference-learning-procedure-6b62d7`, whose relevant module is
**`numpyro.contrib.fwl`** — *Find, Weigh, Learn*:

1. **Find** (`fwl/find.py`) — locate `K` modes of the unnormalized joint. Enumerable
   discrete latents are eliminated exactly by **max-product** message passing (via funsor);
   continuous latents go to an `optimistix` minimiser, either as one joint solve
   (`elimination="joint"`) or clique-by-clique over the junction tree with each
   factor-to-separator message an inner solve (`elimination="nested"` — **this is the
   min-sum pass**, see the `_nested_solve` docstring).
2. **Weigh** (`fwl/weigh.py`) — wrap each mode in a Gaussian whose covariance is the damped
   local empirical Fisher, `(F + lambda I)^-1`, mixed uniformly.
3. **Learn** (`fwl/learn.py`) — expose `log_weights`, `elbo`, `iwae` as differentiable
   functions of the parameters, so `log Z(theta)` can be optimized directly.

`optimistix` and `funsor` were both **missing from the environment**; added to
`pyproject.toml` (that change is committed, `1bd03c3`, and survived the rebase).

Cost model worth remembering: nested elimination costs `(solver steps) ** tree height`, and
`FWLOptions.max_nesting_depth` defaults to 4 for that reason.

---

## 2. False start: the wrong tree

I first read `src/model/model.py` on the stale local `develop` (`8616370`) and reported the
model as relaxed-only (`ConcreteLogits` / `RelaxedBernoulli`). Eli said he remembered
Gamma/Dirichlet variables. He was right and I was reading an out-of-date tree:

- `git fetch --all` showed `origin/develop` had moved 20+ commits ahead to `17254d8`, adding
  `PoissonMarkedPlacements`, `configs/experiment/poisson_captcha.yaml`,
  `notebooks/poisson_captcha.ipynb`, `configs/guide/conjugate_mean_field.yaml`.
- Branches I had cited from stale remote-tracking refs (`bugfix/laplace_over_poisson`,
  `feature/marionette`, `bugfix/damned_clanker`, `bugfix/remove_kornia`) are **deleted on
  GitHub**. `laplace_over_poisson`'s `GammaSlabPrior` is superseded by
  `PoissonMarkedPlacements`.
- Also on `origin/develop`: `f9845e8 pyproject.toml: pin numpyro fork to feature/qem branch`
  — i.e. upstream points at a *different* fork branch than the `claude/inference-learning-
  procedure-6b62d7` we used here. **Open question:** which fork branch should the min-sum
  work target?

Lesson: `git fetch` before reading branch state, and don't trust remote-tracking refs.

---

## 3. Deliverable: `notebooks/captcha1-minsum.ipynb`

Written and executed end to end (10 code cells, zero errors, real stored outputs) against
`+experiment=poisson_captcha`. Uncommitted. It runs `find_weigh_learn` on the
Gamma/Dirichlet model, shows data-vs-reconstruction figures, log-Z bounds, and an IWAE
learning loop, and records three findings inline.

### Finding 1 — min-sum has nothing to eliminate on this model

```
continuous latents: ['color', 'z_rate', 'z_where', 'z_mark'] (packed dim 340572)  # batch 4
discrete latents:   []
1 cliques, height 0, root 0
  clique 0: ['color','z_mark','z_rate','z_where'] | separator [] | factors ['color','obs',...]
```

No discrete latents (everything is Gamma/Dirichlet/Beta), so the max-product half never
runs. And the junction tree is a **single clique of height 0**, because the whole image is
one `obs` sample site whose factor scope is every latent; moralizing that connects
everything. Verified empirically: `elimination="nested"` returns **bit-identical**
`log_joint` to `"joint"`.

This is structural, not a parameterization quirk: min-sum needs the *likelihood* to
factorize, and a monolithic image likelihood cannot.

### Finding 2 — the configured MAP is unbounded

`configs/model/poisson_captcha.yaml` sets `where_concentration=0.1`,
`mark_concentration=0.5`. A Dirichlet with concentration < 1 has a density diverging as any
component approaches 0, so `log gamma_theta` is unbounded above on the simplex boundary and
**no interior mode exists**. Consequences measured:

- `init_to_sample` (the FWL default) is unusable — prior draws are already numerically on the
  boundary, so the unconstrained init contains `-inf`. Used `init_to_uniform` throughout.
- Modes escape to the boundary; a coordinate underflows to exactly 0 in float32 and falls off
  the support, giving `-inf` in `log_joint` and an `inf` in that mode's empirical Fisher.
- `proposal_scale = sqrt(1/(F + lambda))` then returns **exactly 0** (`weigh.py:119`), i.e. an
  invalid `Normal`. **`find_weigh_learn` does not raise** — the zero is baked into
  `_fwl_scale`, and `ValueError: Normal distribution got invalid scale parameter` only fires
  on the first use of `guide`/`log_weights`/`elbo`/`iwae`. *Worth guarding in the library.*
- Tightening the solver makes it worse: loose (`rtol=1e-4`, 128 steps) left 2/4 modes usable;
  tight (`rtol=1e-6`, 1024 steps) left **0/4**.

The class docstring anticipates this mechanism for *learning* the concentrations ("adapted
allocation samples pin to the simplex boundary ... KL -> -inf"); mode-finding hits it with no
learning at all.

Notebook workaround (uses only public data, no library patch): components whose Weigh output
is invalid are replaced by a copy of the best surviving mode, since `make_estimators`
substitutes `_fwl_loc`/`_fwl_scale` from the params dict. The count of replaced modes is
itself the diagnostic.

### Finding 3 — `-inf` weights break IWAE learning

Some proposal draws land off-support, so `log_weights` contains `-inf` (measured 3 of 8
finite). `elbo` is a mean, so it becomes `-inf`; `iwae` is a log-sum-exp and survives — but
the **gradient does not**: an `inf` potential energy differentiates to `NaN`, and `NaN * 0`
from the vanishing log-sum-exp weight poisons everything. The notebook's loop stops itself
after 3 usable steps.

Tested and rejected as fixes: `damping` 1 -> 1000 (narrower proposals still straddle the
boundary and the bound degrades by an order of magnitude), more particles, interior
concentrations (`where=mark=1.0`, where `(alpha-1)*log 0` is a `NaN` outright).
`jax_enable_x64` is the principled fix but needs `ShapeDictionary.shapes` cast to float64
(`lax.conv_general_dilated` rejects mixed dtypes); an x64 attempt was abandoned at 12 minutes
and ~93 GB of device memory without completing one Find. Untested, not ruled out.

### What inference actually achieved

Per-image **hue is recovered correctly** (B brown-red, P purple, R green, C cyan), `z_where`
mass concentrates near the character, MSE 0.0103 vs 0.0154 for blank white. But `max mark
mass` is 0.450, so no anchor commits to an identity and reconstructions are faint blobs:
**it localizes and colors, it does not identify.**

---

## 4. Incident: JupyterLab was killed

This box (GB10) has **121 GB unified between CPU and GPU**, and JAX preallocates ~75% of
"device" memory per process. My scratch runs — the x64 attempt in particular, at ~93 GB —
starved Eli's kernel and killed JupyterLab. Nothing was lost on disk.

Mitigation used for everything afterwards, and the right default for any run here:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 python ...
```

Offered but not done: putting those in `.claude/settings.json` env so they apply
automatically.

---

## 5. Two clarifications Eli asked for

**"What do you mean 2365-way simplex?"** — My phrasing conflated two different sites. There
are *two* Dirichlets in `PoissonMarkedPlacements.sample_latents`:

- `z_where` — **one** Dirichlet over `H*W = 2365` anchors (allocates the total firing mass
  `z_rate`). This is the 2365-way one; `analyze` reports `z_where (1, 2365) event (2365,)`.
- `z_mark` — a 43x55 grid of **independent 36-dimensional** simplices. This is what Eli
  described, and he was right about it.

Both carry the boundary problem, at different severity: 2364 sequential stick-breaking
breaks for `z_where` vs 35 for each mark.

**"What are tiles and margin as observations? Don't we just get an image?"** — Yes, one
80x80x3 image; tiles are not new data. It's the same pixels re-expressed so the likelihood is
a *product of per-region factors* rather than one factor over everything, because funsor
requires the factor reading an enumerated variable to sit inside that variable's plate. With
stride = kernel `(38, 26)` the tiles partition the image:

```
rows: [0,38) [38,76)          leftover 76..79  (4 rows)
cols: [0,26) [26,52) [52,78)  leftover 78,79   (2 cols)
```

`tiles` is a gather of that image into `(6, B, 38, 26, 3)` — every pixel exactly once — passed
as `obs=` inside the anchor plate. The "margin" is the 472 uncovered pixels (7.4% of 6400),
which no anchor's support can reach, so only the background explains them; their factor
depends on `color` alone and can live outside the plate. The total log-density is unchanged
in value (a product over a partition equals one factor over the whole). What changes is the
per-pixel mixture — currently weights come from coverage summed over every anchor reaching
that pixel; tiled, it is background plus that anchor's glyph. Equal only when supports are
disjoint, which is why tiling forces stride >= kernel and gives up the generator's 5 px of
permitted overlap.

---

## 6. Requested changes, implemented (uncommitted)

Eli asked for two things: (a) reduce the grid stride to approximate only the overlap the real
generative process allows, and (b) use a real `dist.Categorical` per anchor.

### Stride, derived from `~/jupyterlab/sandbox/packages/captcha-generator`

`render.py:187-209`: `MAX_OVERLAP_FRACTION = 0.25` of the **average ink width**, capping the
*penetration depth* between ink boxes. Measured over the 36 sandbox dictionary glyphs:

```
frame (38, 26) == (kh, kw)
ink width  mean 20.92  (min 12, max 26)
ink height mean 30.22  (min 30, max 38)
budget = int(0.25 * 20.92) = 5 px
minimum separating offset: 16 px across, 25 px down
```

Chosen stride **(21, 18)**: respects the budget, reaches every legal offset exactly, and the
transposed convolution lands back on exactly 80x80 with no padding. Grid **3x4 = 12 anchors**
(vs 43x55 = 2365 at stride 1).

### Code added

- `src/model/model.py`
  - `_render_dictionary_placements` now accepts a `(stride_h, stride_w)` pair (the frame is
    not square).
  - **`CategoricalMarkedPlacements`** — coarse grid, `z_rate` Gamma, `z_where` Dirichlet over
    12 anchors (`where_concentration: 1.0`, interior per Finding 2), `z_mark ~
    dist.Categorical` per anchor inside `plate("anchor", A, dim=-2)`; batch plate keeps
    `dim=-1`, leaving `dim=-3` for the enumeration dim. `_stamp` folds leading dims for the
    conv and trims/pads to the image (a transposed conv emits
    `n*stride + max(kernel-stride, 0)`, which overshoots when stride > kernel and falls short
    when the grid can't reach the last legal offset).
  - **`categorical_captcha_model`** — same likelihood as `marionette_captcha_model`, kept
    separate because the anchor plate changes the shape convention (anchors before batch).
- `configs/model/categorical_captcha.yaml`, `configs/experiment/categorical_captcha.yaml`

### Measured effect

| | Poisson/Dirichlet | Categorical + generator stride |
|---|---|---|
| anchors | 43x55 = 2365 | 3x4 = **12** |
| continuous dim (batch 1) | 85,143 | **15** |
| `z_mark` | Dirichlet simplex (continuous) | **discrete, enumerable** |
| allocation simplex | 2365-way | 12-way |
| `analyze` verdict | `discrete latents: []` | `discrete latents: ['z_mark']` |

Both of Eli's hypotheses held: smaller grid, and marks that are genuinely discrete so
max-product is selected. Finding 3's 2364-deep stick-breaking is gone; Finding 2's
unboundedness should be gone with interior concentrations (unconfirmed — see below).

---

## 7. The blocker: FWL cannot run on the Categorical model

`find_modes` (`fwl/find.py:488-540`) branches on `structure.has_discrete`, and once any site
is discrete the only reachable branch calls `_discrete_argmax` -> funsor
`_sample_posterior` on **every sweep**. There is no flag to opt a discrete site out
(`analyze` classifies purely on `support.is_discrete and has_enumerate_support`,
`structure.py:204-208`), and `continuous_objective="marginal"` also routes through funsor.

Result, for **both** the overlapping (21,18) and disjoint (38,27) strides:

```
max-product raised: KeyError: '_pyro_dim_2'
  in funsor/tensor.py:615 tensor_to_data, via numpyro/contrib/funsor/discrete.py:76
```

Mechanism, confirmed by tracing under `enum(config_enumerate(...))`:

```
max_plate_nesting 2 -> first_available_dim -3
z_mark  value (36, 1, 1)      batch (12, 1)      # enum -3, anchor plate -2, batch -1
obs     value (1, 80, 80, 3)  batch (36, 1)      # enum dim has slid to -2
```

Folding the anchors into one rendered image consumes the anchor plate dim, so by the time
`obs` is emitted the enum dim sits where funsor has the `anchor` plate registered, and
nothing downstream can name or reduce it. The invariant: **a factor reading an enumerated
site must be emitted while that site's plate dim is still intact.**

The three responsible lines in the new code: `model.py:381-382` (mark inside the anchor
plate — *required*, since `Categorical.to_event(2).enumerate_support()` raises
`NotImplementedError`, so a discrete site's batch dims must be plate dims),
`model.py:392` (`jnp.moveaxis(weighted, -3, -1)`, where the plate dim becomes image
geometry), and `model.py:599` (one image-wide `obs`).

Escape hatches checked:

- `max_sweeps=0` skips the loop body, so Find *returns* — but does nothing: `sweeps [0 0]`,
  the "modes" are the random init draws, and no continuous solve runs either (it is inside
  the same loop).
- Even then **Learn fails independently**: `make_estimators` sets
  `use_enum = structure.has_discrete` (`learn.py:67`), and the contraction leaves the size-36
  enum dim unreduced — `ValueError: Expected the joint log density is a scalar, but got (36,)`.
- Taking the marks *out* of the plate is not a fix: their batch dims become unregistered free
  dims, which is the `_pyro_dim_2` naming in the first place.

Not caused by the coarse grid or the stride — stride 1 with 2365 anchors would fail the same
way, just slower. The Dirichlet formulation hid this incompatibility by never being discrete.

---

## 8. Open decision (where we stopped)

Both options are small changes on top of what now exists; nothing is committed, so dropping
the two config files and the two new functions in `model.py` is clean too.

1. **Per-anchor tile factors** (my recommendation) — one `obs` factor per anchor inside the
   anchor plate over that anchor's own 38x26 tile, at stride = kernel (6 anchors), plus a
   background-only factor for the 472 margin pixels. Makes enumeration valid and exact (36
   configs per anchor, independent), and finally gives the junction tree one clique per tile
   — a bushy height-1 star with `color`/`z_rate`/`z_where` at the root, which is the shape
   nested min-sum can actually afford. Costs the generator's 5 px of permitted glyph overlap.
2. **Slot + mark, no anchor plate** — `z_slot ~ Categorical(12)` and `z_mark ~
   Categorical(36)` as scalar sites in the batch plate, keeping one image-wide `obs`. Nothing
   to factorize; max-product enumerates 12 x 36 = 432 configurations exactly and cheaply.
   Hard-codes one glyph per image, so it does not generalize to multi-character captchas.

Still unanswered from earlier: which fork branch to pin (`claude/inference-learning-
procedure-6b62d7` as used here, vs `feature/qem` per `origin/develop`), and whether to fix
the two library-side issues found (Weigh's zero scale escaping silently; the funsor
`KeyError` instead of a diagnostic).

---

## 9. Reproducing

```bash
# always cap memory on this box -- see section 4
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

# the executed notebook
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3000 notebooks/captcha1-minsum.ipynb

# structure of either model
python - <<'EOF'
import hydra, jax, jax.numpy as jnp, rootutils
rootutils.setup_root(".", indicator=".project-root", pythonpath=True)
from numpyro.contrib.fwl import analyze, build_clique_tree
with hydra.initialize_config_dir(version_base="1.3", config_dir="configs", job_name="x"):
    cfg = hydra.compose(config_name="eval.yaml",
                        overrides=["+experiment=categorical_captcha", "data.batch_size=1"])
dm = hydra.utils.instantiate(cfg.data); model = hydra.utils.instantiate(cfg.model)
for b in dm.test_dataloader():
    images = jnp.asarray(b[0])[:1]; break
s = analyze(model, jax.random.key(0), (images,))
print(s.summary()); print(build_clique_tree(s).summary())
EOF
```

Scratch scripts from the session (stride derivation, max-product probes, damping and x64
sweeps, the notebook builder) live in the session scratchpad under
`/tmp/claude-1000/-home-eli-jupyterlab-captcha-numpyro/<session-id>/scratchpad/` and are
disposable; everything load-bearing is described above.
