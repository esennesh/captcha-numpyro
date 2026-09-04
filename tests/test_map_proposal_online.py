"""Tests for MAP-proposal online CAPTCHA inference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from flax import nnx

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dictionary import ShapeDictionary
from src.inference.map_proposal_online import MAPProposalCaptchaInference
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


def test_map_proposal_runs_on_full_captcha_model():
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)
    inference = MAPProposalCaptchaInference(
        make_model(),
        dsgd_kwargs={"max_count": 8, "width": 16},
        map_max_steps=2,
        num_dispersion_particles=2,
        num_importance_samples=3,
        proposal_max_steps=2,
    )
    result = inference(jax.random.key(3), image)

    assert result.dispersion_losses.shape == (2,)
    assert result.dispersion_num_steps == 2
    assert result.log_weights.shape == (3,)
    assert result.map_losses.shape == (2,)
    assert result.map_num_steps == 2
    assert result.reconstructions.shape == (3, 1, 9, 9, 3)
    assert result.samples["a"].shape == (3, 1, 9, 9, 1)
    assert result.weighted_counts.shape == (1, 9, 9, 1)
    assert result.weighted_reconstruction.shape == (1, 9, 9, 3)
    assert set(result.samples) == {
        "a",
        "color",
        "color_texture",
        "warp_velocity",
    }
    assert np.isfinite(result.effective_sample_size)
    assert np.isfinite(result.log_weights).all()
    assert np.isfinite(result.reconstructions).all()
    assert np.isfinite(result.weighted_counts).all()
    assert 1.0 <= result.effective_sample_size <= 3.0
    np.testing.assert_allclose(result.normalized_weights.sum(), 1.0, rtol=1e-6)
