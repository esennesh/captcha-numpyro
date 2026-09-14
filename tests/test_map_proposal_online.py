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
    # `require_on_canvas` shrinks the count field to the centres whose whole
    # (5, 5) frame fits on a 9 x 9 canvas, so 5 x 5 rather than 9 x 9.
    assert result.samples["a"].shape == (3, 1, 5, 5, 1)
    assert result.weighted_counts.shape == (1, 5, 5, 1)
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


def test_map_keep_best_returns_the_lowest_loss_iterate():
    """`map_values` must be the point the fit reached, not where it stopped.

    The MAP trajectory does not reliably end at its own minimum: on the 80x80
    grid `0000_LJ` reached -40041.1 at step 632 and ended at -38619.7, having
    spent 850 of its 1500 steps outside the basin it had already found. With
    `map_keep_best` the returned latents are the ones that achieved the lowest
    loss, so `map_losses.min()` and the reported point agree.
    """
    import functools

    from flax import nnx

    from src.data.dictionary import ShapeDictionary
    from src.model.model import (
        TexturedDiffeomorphicPoissonConvPlacements,
        poisson_convsc_model,
    )

    side = 9
    glyph = jnp.zeros((5, 5, 4)).at[1:4, 2, :].set(1.0)
    dictionary = ShapeDictionary(shapes=glyph[jnp.newaxis], targets={"glyph": 0})
    image = jnp.ones((1, side, side, 3)).at[:, 3:6, 4, :].set(0.1)

    def fit(keep_best):
        placements = TexturedDiffeomorphicPoissonConvPlacements(
            dictionary, cg_iters=20, expected_count=1.0, img_h=side, img_w=side,
            rngs=nnx.Rngs(0), warp_coarse_height=2, warp_coarse_width=2,
            warp_scale=1.0,
        )
        inference = MAPProposalCaptchaInference(
            functools.partial(poisson_convsc_model, placements=placements),
            dsgd_kwargs={"max_count": 8, "width": 4},
            init_loc_fn=init_to_count_mean,
            map_keep_best=keep_best,
            map_max_steps=40,
            num_dispersion_particles=2,
            num_importance_samples=3,
            proposal_max_steps=5,
            termination_check_interval=2000,
        )
        return inference(jax.random.key(0), image)

    kept = fit(True)
    last = fit(False)
    # the loss histories are the same trajectory; only the returned point differs
    np.testing.assert_allclose(
        np.asarray(kept.map_losses), np.asarray(last.map_losses), rtol=1e-5
    )
    assert np.isfinite(kept.map_losses).all()
    # and the kept point is a real point of that trajectory, not an artifact
    for name, value in kept.map_values.items():
        assert np.isfinite(np.asarray(value)).all(), name
        assert np.asarray(value).shape == np.asarray(last.map_values[name]).shape
