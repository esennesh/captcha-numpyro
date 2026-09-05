"""Checks for the two-field, Gamma--Normal observation model.

Section 3 of ``notebooks/partitioned_gmrf.ipynb`` is the reference: a coarse
paper GMRF and a rendered glyph layer combined with ``over``, under a
partitioned GMRF whose per-pixel precision is a latent Gamma.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from flax import nnx
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.util import log_density

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dictionary import ShapeDictionary
from src.distributions.layers import POTENTIAL_EDGE_ALPHA
from src.model.model import (
    PaperGmrfBackground,
    _ink_layer,
    TexturedDiffeomorphicPoissonConvPlacements,
    composite_poisson_convsc,
    poisson_convsc_model,
)


def make_dictionary() -> ShapeDictionary:
    glyph = jnp.zeros((5, 5, 4)).at[1:4, 2, :].set(1.0)
    return ShapeDictionary(shapes=glyph[jnp.newaxis], targets={"glyph": 0})


def make_model(*, background=True, expected_count=1.0):
    placements = TexturedDiffeomorphicPoissonConvPlacements(
        make_dictionary(),
        cg_iters=20,
        expected_count=expected_count,
        img_h=9,
        img_w=9,
        rngs=nnx.Rngs(0),
        warp_coarse_height=2,
        warp_coarse_width=2,
        warp_scale=1.0,
    )
    paper = (
        PaperGmrfBackground(
            img_h=9, img_w=9, cg_iters=20, coarse_height=3, coarse_width=3
        )
        if background
        else None
    )
    return functools.partial(
        poisson_convsc_model,
        placements=placements,
        backgrounder=paper,
        likelihood="gmrf",
    )


def latent_values(model, image, seed_value=0):
    model_trace = trace(seed(model, jax.random.PRNGKey(seed_value))).get_trace(
        image
    )
    return {
        name: site["value"]
        for name, site in model_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }


def test_gmrf_likelihood_exposes_paper_and_precision_latents():
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)
    model = make_model()
    model_trace = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
        image, plot_mean=True
    )
    names = {
        name
        for name, site in model_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    assert names == {
        "a",
        "color",
        "color_texture",
        "local_precision",
        "paper_field",
        "paper_logit",
        "warp_velocity",
    }
    assert model_trace["local_precision"]["value"].shape == (1, 9, 9)
    assert model_trace["mean"]["value"].shape == (1, 9, 9, 3)
    for name, site in model_trace.items():
        if site["type"] != "sample":
            continue
        assert np.isfinite(site["fn"].log_prob(site["value"])).all(), name


def test_degrees_of_freedom_label_ink_against_paper():
    """nu = 1 on bare paper and 2 under ink, so only paper gets the fat tail."""
    image = jnp.ones((1, 9, 9, 3))
    model = make_model()
    values = latent_values(model, image)
    values["a"] = jnp.zeros_like(values["a"]).at[0, 4, 4, 0].set(3)
    model_trace = trace(
        substitute(seed(model, jax.random.PRNGKey(1)), values)
    ).get_trace(image)
    concentration = model_trace["local_precision"]["fn"].base_dist.concentration
    assert set(np.unique(np.asarray(concentration))) == {0.5, 1.0}
    assert np.asarray(concentration).max() == 1.0


def test_composite_is_the_alpha_blend_of_ink_over_paper():
    image = jnp.ones((1, 9, 9, 3))
    model = make_model()
    values = latent_values(model, image)
    placements = model.keywords["placements"]
    paper = model.keywords["backgrounder"]

    def composite():
        return composite_poisson_convsc(placements, paper)

    layer = substitute(seed(composite, jax.random.PRNGKey(2)), values)()

    # Paper covers the sheet, so the composite is opaque everywhere and its
    # image is exactly what `over_background` hands the likelihood.
    np.testing.assert_allclose(np.asarray(layer.coverage), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(layer.over_background()),
        np.asarray(layer.image),
        atol=1e-6,
    )
    # The count is the ink labelling, not the paper's.
    assert set(np.unique(np.asarray(layer.count))) <= {0, 1}


def test_ink_count_follows_the_edge_alpha_threshold():
    """The region labelling is the ink layer's, not the opaque composite's."""
    image = jnp.ones((1, 9, 9, 3))
    model = make_model()
    values = latent_values(model, image)
    placements = model.keywords["placements"]
    values["a"] = jnp.zeros(values["a"].shape).at[0, 4, 4, 0].set(5.0)
    values["warp_velocity"] = jnp.zeros_like(values["warp_velocity"])

    def composite():
        return composite_poisson_convsc(placements, None)

    def ink_only():
        return _ink_layer(
            placements.warp_ink(
                placements.ink_field(
                    values["a"],
                    placements.color_modulation(values["color"]),
                )
            ),
            values["color"],
        )

    key = jax.random.PRNGKey(3)
    layer = substitute(seed(composite, key), values)()
    ink = substitute(seed(ink_only, key), values)()

    # Paper covers the sheet, so the composite is opaque everywhere.
    np.testing.assert_allclose(np.asarray(layer.coverage), 1.0, atol=1e-6)
    # But `over` keeps the top layer's count where the top layer is present,
    # and that is what cuts the GMRF bonds and sets the Gamma degrees of
    # freedom.
    np.testing.assert_array_equal(
        np.asarray(layer.count), np.asarray(ink.count)
    )
    counted = np.asarray(ink.count) > 0
    covered = np.asarray(ink.coverage) >= POTENTIAL_EDGE_ALPHA
    np.testing.assert_array_equal(counted, covered)
    assert 0 < counted.sum() < counted.size


def test_gmrf_log_density_is_finite_and_differentiable():
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)
    model = make_model()
    values = latent_values(model, image)

    density, _ = log_density(model, (image,), {}, values)
    assert np.isfinite(float(density))

    # A non-integer count is off the Poisson's support, so the *joint* is
    # -inf there by construction; the fitter maximizes the DSGD relaxation
    # instead. What has to survive is the gradient, and in particular the
    # gradient on the counts, which reach the image only through the optical
    # depth and so vanish once that depth saturates.
    relaxed = dict(values, a=values["a"].astype(jnp.float32) + 0.5)
    gradients = jax.grad(
        lambda v: log_density(model, (image,), {}, v)[0]
    )(relaxed)
    for name, gradient in gradients.items():
        assert np.isfinite(np.asarray(gradient)).all(), name
    assert np.abs(np.asarray(gradients["a"])).max() > 0.0


def test_paper_field_removes_the_incentive_to_tile_the_canvas():
    """The failure this whole change targets.

    A sheet whose colour is not white costs a fixed-white model dearly at
    every pixel, and covering it with ink of the right colour is the cheapest
    escape the count field has. At full scale on ``data/examples/0000_LJ`` the
    model's own log joint scored 49 tiling stamps at -21,091 against -69,192
    for the correct two glyphs. Giving the paper a latent colour takes that
    incentive away.
    """
    tint = jnp.asarray([0.90, 0.93, 0.88])
    image = jnp.broadcast_to(tint, (1, 12, 12, 3))
    logit = jnp.log(tint) - jnp.log1p(-tint)
    # An opaque block, so a handful of stamps really can cover the sheet.
    dictionary = ShapeDictionary(
        shapes=jnp.ones((1, 4, 4, 4)), targets={"block": 0}
    )

    def build(background):
        placements = TexturedDiffeomorphicPoissonConvPlacements(
            dictionary,
            cg_iters=20,
            expected_count=1.0,
            img_h=12,
            img_w=12,
            rngs=nnx.Rngs(0),
            warp_coarse_height=2,
            warp_coarse_width=2,
            warp_scale=0.0,
        )
        paper = (
            PaperGmrfBackground(
                img_h=12, img_w=12, cg_iters=20, coarse_height=3,
                coarse_width=3
            )
            if background
            else None
        )
        return functools.partial(
            poisson_convsc_model,
            placements=placements,
            backgrounder=paper,
            likelihood="gmrf",
        )

    def score(model, counts, extra):
        values = {
            "a": counts,
            "color": tint,
            "color_texture": jnp.zeros((4, 4, 3)),
            "warp_velocity": jnp.zeros((2, 2, 2)),
            "local_precision": jnp.ones((1, 12, 12)),
            **extra,
        }
        return float(log_density(model, (image,), {}, values)[0])

    blank = jnp.zeros((1, 12, 12, 1), dtype=jnp.int32)
    tiled = blank
    for y in range(1, 12, 4):
        for x in range(1, 12, 4):
            tiled = tiled.at[0, y, x, 0].set(1)

    white = build(False)
    white_margin = score(white, blank, {}) - score(white, tiled, {})
    assert white_margin < 0.0, "the failure this fix targets"

    paper = {"paper_logit": logit, "paper_field": jnp.zeros((3, 3, 3))}
    fielded = build(True)
    paper_margin = score(fielded, blank, paper) - score(fielded, tiled, paper)
    assert paper_margin > white_margin + 50.0
