"""Checks for linear-space relaxed counts under a nonnegativity projection."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import rootutils
from flax import nnx

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from numpyro import handlers
from numpyro.contrib.map_proposal import AutoMAPProposal
from numpyro.distributions.transforms import IdentityTransform
from numpyro.optim import Adam

from src.data.dictionary import ShapeDictionary
from src.inference.linear_count_proposal import (
    LinearCountMAPProposal,
    count_block_adam,
    pose_step_size,
)
from src.inference.map_proposal_online import (
    MAPProposalCaptchaInference,
    init_to_count_mean,
)
from src.model.model import (
    TexturedDiffeomorphicPoissonConvPlacements,
    poisson_convsc_model,
)

SIDE = 9


def make_model(expected_count=1.0):
    glyph = jnp.zeros((5, 5, 4)).at[1:4, 2, :].set(1.0)
    dictionary = ShapeDictionary(
        shapes=glyph[jnp.newaxis], targets={"glyph": 0}
    )
    placements = TexturedDiffeomorphicPoissonConvPlacements(
        dictionary,
        cg_iters=20,
        expected_count=expected_count,
        img_h=SIDE,
        img_w=SIDE,
        rngs=nnx.Rngs(0),
        warp_coarse_height=2,
        warp_coarse_width=2,
        warp_scale=1.0,
    )
    return functools.partial(poisson_convsc_model, placements=placements)


def make_image():
    return jnp.ones((1, SIDE, SIDE, 3)).at[:, 3:6, 4, :].set(0.1)


def fit(guide_class, optimizer, steps=30):
    inference = MAPProposalCaptchaInference(
        make_model(),
        dsgd_kwargs={"max_count": 8, "width": 4},
        guide_class=guide_class,
        init_loc_fn=init_to_count_mean,
        map_max_steps=steps,
        map_optimizer=optimizer,
        num_dispersion_particles=2,
        num_importance_samples=3,
        proposal_max_steps=5,
        termination_check_interval=steps,
    )
    return inference(jax.random.key(3), make_image())


def test_count_sites_lose_their_logarithm_without_moving_the_init():
    """Swapping the transform must not move the starting point."""
    image = make_image()
    model = make_model()

    logarithmic = AutoMAPProposal(model, dsgd_kwargs={"max_count": 8, "width": 4},
                                  init_loc_fn=init_to_count_mean)
    linear = LinearCountMAPProposal(model, dsgd_kwargs={"max_count": 8, "width": 4},
                                    init_loc_fn=init_to_count_mean)
    # `fit` runs prototype discovery under these handlers.
    for guide in (logarithmic, linear):
        with handlers.block(), handlers.seed(rng_seed=jax.random.PRNGKey(0)):
            guide._setup_prototype(image)

    assert linear.linear_count_sites == ("a",)
    assert isinstance(linear._transforms["a"], IdentityTransform)
    assert not isinstance(logarithmic._transforms["a"], IdentityTransform)

    # The constrained starting count is the same under both.
    np.testing.assert_allclose(
        np.asarray(linear._transforms["a"](linear._init_locs["a"])),
        np.asarray(logarithmic._transforms["a"](logarithmic._init_locs["a"])),
        rtol=1e-6,
    )
    # Every other site is untouched.
    for name in logarithmic._transforms:
        if name == "a":
            continue
        assert type(linear._transforms[name]) is type(logarithmic._transforms[name])


def test_projection_keeps_counts_nonnegative_and_makes_exact_zeros():
    result = fit(
        LinearCountMAPProposal,
        count_block_adam(count_step_size=1e-2, step_size=1e-2),
        steps=60,
    )
    counts = np.asarray(result.map_values["a"])
    assert counts.min() >= 0.0
    # Projected gradient descent against the Poisson prior pins unsupported
    # sites exactly at zero; the logarithmic parameterization cannot.
    assert (counts == 0.0).sum() > 0
    assert np.isfinite(result.map_losses).all()
    assert np.isfinite(result.dispersion_losses).all()


def test_logarithmic_parameterization_never_reaches_zero():
    result = fit(AutoMAPProposal, Adam(step_size=1e-2), steps=60)
    counts = np.asarray(result.map_values["a"])
    assert counts.min() > 0.0
    assert (counts == 0.0).sum() == 0


def test_count_block_adam_applies_a_separate_step_size():
    optimizer = count_block_adam(
        count_step_size=0.2, step_size=0.01, count_sites=("a",)
    )
    parameters = {"a": jnp.ones((3,)), "color": jnp.ones((2,))}
    state = optimizer.init(parameters)
    gradients = {"a": jnp.ones((3,)), "color": jnp.ones((2,))}
    updates, _ = optax.with_extra_args_support(optimizer).update(
        gradients, state, parameters
    )
    np.testing.assert_allclose(np.asarray(updates["a"]), -0.2, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(updates["color"]), -0.01, rtol=1e-5)


def test_projection_clips_an_update_that_would_go_negative():
    optimizer = count_block_adam(count_step_size=0.5, step_size=0.01)
    parameters = {"a": jnp.asarray([0.1, 2.0])}
    state = optimizer.init(parameters)
    # A positive gradient drives the parameter down by the step size; the first
    # coordinate would land at -0.4 and must be clipped to exactly zero.
    updates, _ = optax.with_extra_args_support(optimizer).update(
        {"a": jnp.ones((2,))}, state, parameters
    )
    moved = optax.apply_updates(parameters, updates)
    assert float(moved["a"][0]) == 0.0
    assert float(moved["a"][1]) == pytest.approx(1.5, abs=1e-5)


def test_step_sizes_must_be_positive():
    with pytest.raises(ValueError):
        count_block_adam(count_step_size=0.0)
    with pytest.raises(ValueError):
        count_block_adam(step_size=-1.0)


def test_named_step_sizes_override_the_default():
    """`pose` needs its own scale: one unconstrained unit is 90 degrees."""
    optimizer = count_block_adam(
        count_step_size=1e-3, step_size=0.05, step_sizes={"pose": 0.02}
    )
    parameters = {"a": jnp.ones((2,)), "pose": jnp.zeros((2,)),
                  "color": jnp.ones((2,))}
    state = optimizer.init(parameters)
    updates, _ = optax.with_extra_args_support(optimizer).update(
        {name: jnp.ones_like(value) for name, value in parameters.items()},
        state, parameters,
    )
    np.testing.assert_allclose(float(updates["a"][0]), -1e-3, rtol=1e-4)
    np.testing.assert_allclose(float(updates["pose"][0]), -0.02, rtol=1e-4)
    np.testing.assert_allclose(float(updates["color"][0]), -0.05, rtol=1e-4)


def test_pose_step_size_converts_degrees_through_the_circular_jacobian():
    """biject_to(circular) has derivative pi/2 at zero, so a unit is 90 deg."""
    import math

    np.testing.assert_allclose(
        pose_step_size(1.8), math.radians(1.8) / (math.pi / 2.0), rtol=1e-9
    )
    np.testing.assert_allclose(pose_step_size(90.0), 1.0, rtol=1e-9)
    # the config's plain 0.02 has to agree with the helper it replaced
    np.testing.assert_allclose(pose_step_size(1.8), 0.02, rtol=2e-3)
    with pytest.raises(ValueError, match="degrees"):
        pose_step_size(0.0)


def test_named_step_sizes_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        count_block_adam(step_sizes={"pose": 0.0})
