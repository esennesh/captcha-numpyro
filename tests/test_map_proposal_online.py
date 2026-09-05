"""Tests for MAP-proposal online CAPTCHA inference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from flax import nnx

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from numpyro.infer.initialization import init_to_uniform

from src.data.dictionary import ShapeDictionary
from src.inference.map_proposal_online import (
    MAPProposalCaptchaInference,
    init_to_count_mean,
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


def test_count_mean_init_starts_at_the_prior_rate():
    """``init_to_uniform`` starts the relaxed count field at exp(U(-2, 2)).

    That is a mean of 1.81 at *every* coordinate. Here that is 81 coordinates
    and merely wrong; at 80 x 80 x 36 it is 417,081 glyph stamps, which drives
    the optical depth to 7.1e4 per pixel, saturates the opacity, and leaves
    ``dA/dtau = exp(-tau)`` at exactly zero in float32 -- no likelihood
    gradient at all on any count coordinate.
    """
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)

    totals = {}
    for name, init_loc_fn in (("uniform", init_to_uniform),
                              ("count-mean", init_to_count_mean)):
        inference = MAPProposalCaptchaInference(
            make_model(),
            dsgd_kwargs={"max_count": 8, "width": 4},
            init_loc_fn=init_loc_fn,
            map_max_steps=1,
            num_dispersion_particles=2,
            num_importance_samples=2,
            proposal_max_steps=1,
            termination_check_interval=1,
        )
        result = inference(jax.random.key(5), image)
        totals[name] = float(np.sum(result.map_values["a"]))

    # expected_count = 1.0 over 9 * 9 * 1 sites.
    np.testing.assert_allclose(totals["count-mean"], 1.0, rtol=0.2)
    assert totals["uniform"] > 20.0 * totals["count-mean"]


def test_dispersion_losses_stay_finite_at_a_sparse_count_rate():
    """A narrow DSGD window is required, not merely preferred.

    ``anchored_relaxed_count`` builds its right tail as
    ``cumprod(1 / pmf_ratio_down)``, which for a Poisson is
    ``prod_k (k / lambda)``. At the rates a sparse count field visits that
    overflows float32; ``jnp.clip(cdf, 0, 1)`` hides the infinity in the
    forward pass while the backward pass returns NaN, so every proposal
    update is silently discarded by ``eval_and_stable_update``.
    """
    image = jnp.ones((1, 9, 9, 3)).at[:, 3:6, 4, :].set(0.1)
    inference = MAPProposalCaptchaInference(
        make_model(),
        dsgd_kwargs={"max_count": 8, "width": 4},
        map_max_steps=2,
        num_dispersion_particles=2,
        num_importance_samples=2,
        proposal_max_steps=2,
        termination_check_interval=2,
    )
    result = inference(jax.random.key(7), image)
    assert np.isfinite(result.map_losses).all()
    assert np.isfinite(result.dispersion_losses).all()
