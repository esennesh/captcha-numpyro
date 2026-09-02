"""Integration checks for textured, diffeomorphic Poisson rendering."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from flax import nnx
from numpyro.handlers import condition, seed, trace
from numpyro.infer.util import get_importance_trace

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dictionary import ShapeDictionary
from src.inference.poisson_conv_encoder import (
    ForegroundFieldGuide,
    InkColorFinder,
    PoissonConvBackbone,
    PoissonRateHead,
    poisson_convsc_guide,
)
from src.model.model import (
    PoissonConvPlacements,
    TexturedDiffeomorphicPoissonConvPlacements,
    poisson_convsc_model,
)


def make_dictionary() -> ShapeDictionary:
    """Return two small RGBA glyphs with different alpha supports."""
    first = jnp.zeros((5, 4, 4)).at[1:4, 1:3, :3].set(1.0)
    first = first.at[1:4, 1:3, 3].set(0.8)
    second = jnp.zeros((5, 4, 4)).at[::2, :, :3].set(1.0)
    second = second.at[::2, :, 3].set(0.6)
    return ShapeDictionary(
        shapes=jnp.stack((first, second)),
        targets={"first": 0, "second": 1},
    )


def make_placements(
    *, warp_scale: float = 0.0
) -> TexturedDiffeomorphicPoissonConvPlacements:
    return TexturedDiffeomorphicPoissonConvPlacements(
        make_dictionary(),
        expected_count=1.0,
        img_h=12,
        img_w=11,
        cg_iters=50,
        rngs=nnx.Rngs(0),
        warp_coarse_height=3,
        warp_coarse_width=3,
        warp_scale=warp_scale,
    )


def model_mean(placements, values):
    model = condition(poisson_convsc_model, data=values)
    model_trace = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
        None, placements, plot_mean=True
    )
    return model_trace["mean"]["value"]


def test_guide_matches_all_model_field_sites_with_finite_densities():
    dictionary = make_dictionary()
    model = functools.partial(
        poisson_convsc_model,
        placements=make_placements(warp_scale=0.2),
    )
    guide = functools.partial(
        poisson_convsc_guide,
        backbone=PoissonConvBackbone((4,), (1,), rngs=nnx.Rngs(1)),
        color_finder=InkColorFinder(4, hidden_dim=8, rngs=nnx.Rngs(2)),
        fields=ForegroundFieldGuide(
            dictionary,
            4,
            cg_iters=50,
            rngs=nnx.Rngs(3),
            warp_coarse_height=3,
            warp_coarse_width=3,
        ),
        placements=PoissonRateHead(
            dictionary,
            4,
            1.0,
            12,
            11,
            rngs=nnx.Rngs(4),
        ),
    )
    images = jnp.full((1, 12, 11, 3), 0.9)
    model_trace, guide_trace = get_importance_trace(
        seed(model, jax.random.PRNGKey(5)),
        seed(guide, jax.random.PRNGKey(6)),
        (images,),
        {},
        {},
    )
    latent_names = {"a", "color", "color_texture", "warp_velocity"}
    assert latent_names <= set(model_trace)
    assert latent_names <= set(guide_trace)
    for site in (*model_trace.values(), *guide_trace.values()):
        if site["type"] == "sample":
            assert np.isfinite(site["log_prob"]).all()


def test_model_exposes_normalized_texture_and_velocity_latents():
    placements = make_placements()
    model_trace = trace(
        seed(poisson_convsc_model, jax.random.PRNGKey(7))
    ).get_trace(None, placements)
    assert model_trace["color_texture"]["fn"].event_shape == (5, 4, 3)
    assert model_trace["warp_velocity"]["fn"].event_shape == (3, 3, 2)
    for name in ("color_texture", "warp_velocity"):
        site = model_trace[name]
        assert np.isfinite(site["fn"].log_prob(site["value"])).all()


def test_texture_and_velocity_each_change_the_rendered_mean():
    color = jnp.asarray([[0.2, 0.4, 0.6]])
    counts = jnp.zeros((1, 12, 11, 2)).at[0, 6, 5, 0].set(1.0)
    placements = make_placements(warp_scale=0.5)
    zero_texture = jnp.zeros((1, 5, 4, 3))
    zero_velocity = jnp.zeros((1, 3, 3, 2))
    values = {
        "a": counts,
        "color": color,
        "color_texture": zero_texture,
        "warp_velocity": zero_velocity,
    }
    baseline = model_mean(placements, values)
    textured = model_mean(
        placements,
        {
            **values,
            "color_texture": zero_texture.at[:, :, :2, 0].set(1.0),
        },
    )
    warped = model_mean(
        placements,
        {
            **values,
            "warp_velocity": zero_velocity.at[:, 1, 1].set(
                jnp.asarray([1.0, -1.0])
            ),
        },
    )
    assert np.max(np.abs(textured - baseline)) > 1e-3
    assert np.max(np.abs(warped - baseline)) > 1e-3


def test_zero_texture_and_velocity_recover_the_original_renderer_exactly():
    color = jnp.asarray([[0.2, 0.4, 0.6]])
    counts = jnp.zeros((1, 12, 11, 2)).at[0, 6, 5, 0].set(1.0)
    base = PoissonConvPlacements(
        make_dictionary(),
        expected_count=1.0,
        img_h=12,
        img_w=11,
        rngs=nnx.Rngs(0),
    )
    ink = base.ink_field(counts)
    depth = ink[..., 3:]
    tint = jnp.where(
        depth > 1e-6,
        ink[..., :3] / jnp.clip(depth, 1e-6, None),
        1.0,
    )
    foreground = tint * color[..., jnp.newaxis, jnp.newaxis, :]
    opacity = -jnp.expm1(-(depth + 1e-4))
    expected = opacity * foreground + (1.0 - opacity)
    base_mean = model_mean(base, {"a": counts, "color": color})
    actual = model_mean(
        make_placements(),
        {
            "a": counts,
            "color": color,
            "color_texture": jnp.zeros((1, 5, 4, 3)),
            "warp_velocity": jnp.zeros((1, 3, 3, 2)),
        },
    )
    assert np.array_equal(base_mean, expected)
    assert np.array_equal(actual, expected)
