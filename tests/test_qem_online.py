"""Tests for QEM-based online CAPTCHA inference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from flax import nnx
from numpyro.distributions import Beta
from numpyro.distributions.exp_family import from_mean_params, mean_params

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dictionary import ShapeDictionary
from src.inference.qem_online import QEMCaptchaInference
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


def test_beta_mean_parameter_round_trip():
    target = Beta(jnp.asarray((2.0, 4.0)), jnp.asarray((5.0, 3.0)))
    matched = from_mean_params(Beta(jnp.ones(2), jnp.ones(2)), mean_params(target))

    np.testing.assert_allclose(
        matched.concentration1, target.concentration1, rtol=1e-5
    )
    np.testing.assert_allclose(
        matched.concentration0, target.concentration0, rtol=1e-5
    )


def test_qem_runs_on_candidate_restricted_captcha():
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)
    inference = QEMCaptchaInference(
        make_model(),
        classes_per_location=1,
        min_distance=2.0,
        num_candidates=1,
        num_posterior_samples=2,
        num_samples=2,
        num_steps=1,
    )
    result = inference(jax.random.key(3), image)

    assert result.candidate_sites.shape == (1, 3)
    assert result.qem_result.log_marginals.shape == (1,)
    assert result.reconstructions.shape == (2, 1, 9, 9, 3)
    assert set(result.samples) == {
        "candidate_counts",
        "color",
        "color_texture",
        "warp_velocity",
    }
    assert np.isfinite(result.qem_result.log_marginals).all()
    assert np.isfinite(result.reconstructions).all()
