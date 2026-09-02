"""Tests for observation-restricted online CAPTCHA inference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from flax import nnx
from numpyro.handlers import condition, seed, trace

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dictionary import ShapeDictionary
from src.inference.online import (
    CandidateTexturedDiffeomorphicPoissonConvPlacements,
    candidate_indices,
    restrict_poisson_model,
)
from src.model.model import (
    TexturedDiffeomorphicPoissonConvPlacements,
    poisson_convsc_model,
)


def make_dictionary() -> ShapeDictionary:
    first = jnp.zeros((5, 5, 4)).at[1:4, 2, :].set(1.0)
    second = jnp.zeros((5, 5, 4)).at[2, 1:4, :].set(1.0)
    return ShapeDictionary(
        shapes=jnp.stack((first, second)),
        targets={"first": 0, "second": 1},
    )


def make_placements():
    return TexturedDiffeomorphicPoissonConvPlacements(
        make_dictionary(),
        cg_iters=30,
        expected_count=2.0,
        img_h=12,
        img_w=11,
        rngs=nnx.Rngs(0),
        warp_coarse_height=3,
        warp_coarse_width=3,
        warp_scale=0.0,
    )


def test_candidate_counts_are_scattered_and_omitted_mass_is_retained():
    sites = jnp.asarray(((4, 5, 0), (7, 8, 1)))
    placements = (
        CandidateTexturedDiffeomorphicPoissonConvPlacements.from_placements(
            sites, make_placements()
        )
    )
    values = jnp.asarray([[2.0, 1.0]])
    placement_trace = trace(
        seed(condition(placements.sample_counts, {"candidate_counts": values}), 0)
    ).get_trace()
    counts = placement_trace["a"]["value"]
    expected_rate = placements.expected_count / placements.num_sites
    expected_omitted = -(placements.num_sites - len(sites)) * expected_rate

    assert counts[0, 4, 5, 0] == 2.0
    assert counts[0, 7, 8, 1] == 1.0
    assert counts.sum() == 3.0
    np.testing.assert_allclose(
        placement_trace["omitted_count_mass"]["value"], expected_omitted
    )


def test_candidate_selection_retains_class_alternatives():
    image = jnp.ones((12, 11, 3)).at[4:9, 5, :].set(0.0)
    sites = candidate_indices(
        image,
        make_dictionary(),
        classes_per_location=2,
        min_distance=4.0,
        num_candidates=4,
    )
    first_location = np.asarray(sites[:2, :2])
    first_classes = np.asarray(sites[:2, 2])

    assert sites.shape == (4, 3)
    assert np.array_equal(first_location[0], first_location[1])
    assert set(first_classes) == {0, 1}


def test_restricted_model_uses_candidate_latent_and_renders_full_image():
    model = functools.partial(poisson_convsc_model, placements=make_placements())
    image = jnp.ones((1, 12, 11, 3)).at[:, 4:9, 5, :].set(0.0)
    restricted, sites = restrict_poisson_model(
        model,
        image,
        classes_per_location=2,
        min_distance=4.0,
        num_candidates=4,
    )
    model_trace = trace(seed(restricted, jax.random.key(1))).get_trace(
        image, plot_mean=True
    )

    assert model_trace["candidate_counts"]["value"].shape == (1, 4)
    assert model_trace["mean"]["value"].shape == image.shape
    assert np.array_equal(
        model_trace["candidate_counts"]["value"],
        model_trace["a"]["value"][:, sites[:, 0], sites[:, 1], sites[:, 2]],
    )
    assert all(
        np.isfinite(site["fn"].log_prob(site["value"])).all()
        for site in model_trace.values()
        if site["type"] == "sample"
    )
