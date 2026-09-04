"""Tests for MAP-proposal online CAPTCHA inference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from flax import nnx

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dictionary import ShapeDictionary
from src.inference.map_proposal_online import (
    MAPProposalCaptchaInference,
    _initial_guide_values,
)
from src.model.model import (
    TexturedDiffeomorphicPoissonConvPlacements,
    poisson_convsc_model,
)


def make_model():
    glyph = jnp.zeros((5, 5, 4)).at[1:4, 2, :].set(1.0)
    dictionary = ShapeDictionary(
        shapes=glyph[jnp.newaxis], targets={"glyph": 0}
    )
    placements = TexturedDiffeomorphicPoissonConvPlacements(
        dictionary,
        cg_iters=20,
        expected_count=1.0,
        img_h=9,
        img_w=9,
        rngs=nnx.Rngs(0),
        warp_coarse_height=2,
        warp_coarse_width=2,
        warp_scale=0.0,
    )
    return functools.partial(poisson_convsc_model, placements=placements)


def test_initial_values_form_a_coherent_foreground_explanation():
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)
    sites = jnp.asarray(
        ((2, 3, 0), (2, 3, 1), (6, 7, 0), (6, 7, 1))
    )
    values = _initial_guide_values(make_model(), sites, image, count_mass=2.0)

    np.testing.assert_allclose(values["candidate_counts"].sum(), 2.0)
    assert values["candidate_counts"][0, 0] > values["candidate_counts"][0, 1]
    assert values["candidate_counts"][0, 2] > values["candidate_counts"][0, 3]
    np.testing.assert_allclose(values["color"], ((0.1, 0.1, 0.1),))
    np.testing.assert_allclose(values["color_texture"], 0.0)
    np.testing.assert_allclose(values["warp_velocity"], 0.0)


def test_map_proposal_runs_on_candidate_restricted_captcha():
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)
    inference = MAPProposalCaptchaInference(
        make_model(),
        classes_per_location=1,
        dsgd_kwargs={"max_count": 8, "width": 16},
        min_distance=2.0,
        num_candidates=1,
        num_dispersion_particles=2,
        num_importance_samples=3,
        map_max_steps=2,
        proposal_max_steps=2,
    )
    result = inference(jax.random.key(3), image)

    assert result.candidate_sites.shape == (1, 3)
    assert result.log_weights.shape == (3,)
    assert result.map_losses.shape == (2,)
    assert result.map_num_steps == 2
    assert result.reconstructions.shape == (3, 1, 9, 9, 3)
    assert result.dispersion_losses.shape == (2,)
    assert result.dispersion_num_steps == 2
    assert result.weighted_reconstruction.shape == (1, 9, 9, 3)
    assert set(result.samples) == {
        "candidate_counts",
        "color",
        "color_texture",
        "warp_velocity",
    }
    assert np.isfinite(result.effective_sample_size)
    assert np.isfinite(result.log_weights).all()
    assert np.isfinite(result.reconstructions).all()
    assert 1.0 <= result.effective_sample_size <= 3.0
    np.testing.assert_allclose(result.normalized_weights.sum(), 1.0, rtol=1e-6)
