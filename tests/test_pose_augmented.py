"""Checks for the discrete per-occurrence pose."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rootutils
from flax import nnx

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpyro
from numpyro.handlers import seed, trace

from src.data.dictionary import (
    ShapeDictionary,
    pose_augmented_dictionary,
    von_mises_pose_weights,
)
from src.model.model import PoseAugmentedPlacements, poisson_convsc_model

ANGLES = (-20, -10, 0, 10, 20)


def make_dictionary():
    first = jnp.zeros((9, 7, 4)).at[2:7, 3, :].set(1.0)
    second = jnp.zeros((9, 7, 4)).at[4, 1:6, :].set(1.0)
    return ShapeDictionary(shapes=jnp.stack((first, second)),
                           targets={0: "bar_42", 1: "dash_42"})


def make_placements(**kwargs):
    defaults = dict(
        expected_count=2.0, img_h=20, img_w=20, cg_iters=20,
        rngs=nnx.Rngs(0), pose_angles=ANGLES, pose_sigma_degrees=10.0,
        warp_coarse_height=3, warp_coarse_width=3, warp_modes="affine",
        warp_scale=1.5,
    )
    defaults.update(kwargs)
    return PoseAugmentedPlacements(make_dictionary(), **defaults)


def test_augmentation_is_angle_major_and_keeps_the_zero_block_intact():
    source = make_dictionary()
    augmented, angles = pose_augmented_dictionary(source, ANGLES)
    glyphs = len(source)
    assert angles == tuple(float(a) for a in ANGLES)
    assert len(augmented) == glyphs * len(ANGLES)
    zero = angles.index(0.0)
    np.testing.assert_array_equal(
        np.asarray(augmented.shapes[zero * glyphs:(zero + 1) * glyphs]),
        np.asarray(source.shapes),
    )
    assert augmented.targets[zero * glyphs] == "bar@+0"
    assert augmented.targets[0] == "bar@-20"


def test_rotation_actually_rotates():
    """A vertical bar at +-90 degrees has to become a horizontal one."""
    augmented, angles = pose_augmented_dictionary(make_dictionary(), (0, 90))
    upright = np.asarray(augmented.shapes[0, ..., 3])
    turned = np.asarray(augmented.shapes[2, ..., 3])
    # the bar is taller than wide upright, and wider than tall once turned
    def extent(alpha):
        rows, columns = np.nonzero(alpha > 0.3)
        return np.ptp(rows), np.ptp(columns)
    assert extent(upright)[0] > extent(upright)[1]
    assert extent(turned)[1] > extent(turned)[0]


@pytest.mark.parametrize("sigma", (5.0, 10.0, 20.0))
def test_von_mises_weights_are_normalized_and_peaked_at_zero(sigma):
    weights = np.asarray(von_mises_pose_weights(ANGLES, sigma))
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-6)
    assert weights.argmax() == ANGLES.index(0)
    # symmetric about zero, and monotone away from it
    np.testing.assert_allclose(weights, weights[::-1], rtol=1e-6)
    assert weights[0] < weights[1] < weights[2]


def test_narrower_sigma_concentrates_the_prior():
    tight = np.asarray(von_mises_pose_weights(ANGLES, 5.0))
    loose = np.asarray(von_mises_pose_weights(ANGLES, 20.0))
    assert tight.max() > loose.max()
    assert tight[0] < loose[0]


def test_von_mises_rejects_a_nonpositive_width():
    with pytest.raises(ValueError, match="sigma_degrees"):
        von_mises_pose_weights(ANGLES, 0.0)


def test_the_rate_carries_the_pose_prior_without_changing_the_total():
    """expected_count must survive augmentation, and pose must be its marginal."""
    placements = make_placements()
    assert placements.num_poses == len(ANGLES)
    assert placements.num_glyphs == 2
    assert placements.num_features == 2 * len(ANGLES)

    with seed(rng_seed=0):
        counts_trace = trace(placements.sample_counts).get_trace()
    rate = np.asarray(counts_trace["a"]["fn"].base_dist.rate)
    np.testing.assert_allclose(rate.sum(), 2.0, rtol=1e-5)

    per_pose = rate[0, 0, 0].reshape(placements.num_poses,
                                     placements.num_glyphs).sum(axis=1)
    np.testing.assert_allclose(
        per_pose / per_pose.sum(),
        np.asarray(von_mises_pose_weights(ANGLES, 10.0)),
        rtol=1e-5,
    )


def test_the_model_runs_and_stays_differentiable():
    placements = make_placements()
    model = functools.partial(poisson_convsc_model, placements=placements,
                              likelihood="gmrf")
    image = jnp.full((1, 20, 20, 3), 0.95).at[:, 8:13, 10, :].set(0.2)
    model_trace = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
        image, plot_mean=True
    )
    assert model_trace["a"]["value"].shape == (
        1, *placements.count_shape, placements.num_features
    )
    assert model_trace["mean"]["value"].shape == (1, 20, 20, 3)
    for name, site in model_trace.items():
        if site["type"] == "sample":
            assert np.isfinite(site["fn"].log_prob(site["value"])).all(), name

    from numpyro.infer.util import log_density
    values = {name: site["value"] for name, site in model_trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    relaxed = dict(values, a=values["a"].astype(jnp.float32) + 0.25)
    gradients = jax.grad(
        lambda v: log_density(model, (image,), {}, v)[0]
    )(relaxed)
    for name, gradient in gradients.items():
        assert np.isfinite(np.asarray(gradient)).all(), name
    assert np.abs(np.asarray(gradients["a"])).max() > 0.0
