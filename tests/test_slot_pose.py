"""Checks for one continuous rotation per spatial slot."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rootutils
from flax import nnx

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from numpyro.distributions.transforms import biject_to
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.util import log_density

from src.data.dictionary import ShapeDictionary
from src.model.diffeomorphism import rotate_patches
from src.model.model import SlotPosePlacements, poisson_convsc_model

SIDE = 24


def make_dictionary():
    """A vertical bar and a horizontal one, so rotation is visible."""
    bar = jnp.zeros((9, 7, 4)).at[1:8, 3, :].set(1.0)
    dash = jnp.zeros((9, 7, 4)).at[4, 1:6, :].set(1.0)
    return ShapeDictionary(shapes=jnp.stack((bar, dash)),
                           targets={0: "bar_42", 1: "dash_42"})


def make_placements(**kwargs):
    defaults = dict(
        expected_count=2.0, img_h=SIDE, img_w=SIDE, cg_iters=20,
        rngs=nnx.Rngs(0), pose_sigma_degrees=10.0, warp_coarse_height=3,
        warp_coarse_width=3, warp_modes="affine", warp_scale=1.5,
    )
    defaults.update(kwargs)
    return SlotPosePlacements(make_dictionary(), **defaults)


def test_rotation_commutes_with_the_sum_over_glyphs():
    """The identity the whole design rests on: R is linear in the patch."""
    placements = make_placements()
    kernel = placements.ink_kernel
    patches = jnp.moveaxis(kernel, -1, 0)
    counts = jnp.asarray([1.0, 0.4])
    angle = jnp.radians(jnp.asarray(23.0))
    per_glyph = sum(
        counts[k] * rotate_patches(patches[k], angle) for k in range(2)
    )
    summed_first = rotate_patches(
        jnp.einsum("k,khwc->hwc", counts, patches), angle
    )
    np.testing.assert_allclose(
        np.asarray(per_glyph), np.asarray(summed_first), atol=1e-5
    )


def test_pose_is_an_ordinary_continuous_site():
    """No relaxation: a Circular support is one `biject_to` handles."""
    placements = make_placements()
    with seed(rng_seed=0):
        pose_trace = trace(placements.sample_pose).get_trace()
    site = pose_trace["pose"]
    assert site["value"].shape == placements.count_shape
    assert not site["fn"].support.is_discrete
    transform = biject_to(site["fn"].support)
    unconstrained = transform.inv(site["value"])
    assert np.isfinite(np.asarray(unconstrained)).all()
    np.testing.assert_allclose(
        np.asarray(transform(unconstrained)), np.asarray(site["value"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        placements.pose_concentration,
        1.0 / math.radians(10.0) ** 2, rtol=1e-6,
    )


def test_one_slot_one_angle_turns_whatever_it_stamps():
    """The bar and the dash at one slot share that slot's rotation."""
    placements = make_placements()
    rows, columns = placements.count_shape
    slot = (rows // 2, columns // 2)
    kernel_height, kernel_width = placements.ink_kernel.shape[:2]

    def depth(glyph, degrees):
        counts = jnp.zeros((rows, columns, 2)).at[slot[0], slot[1], glyph].set(1.0)
        pose = jnp.zeros((rows, columns)).at[slot].set(jnp.radians(degrees))
        ink = placements.pose_ink_field(counts, pose)
        return np.asarray(ink[..., 3])

    def extent(field):
        y, x = np.nonzero(field > 0.3)
        return np.ptp(y), np.ptp(x)

    # a vertical bar turned 90 degrees becomes horizontal
    upright = extent(depth(0, 0.0))
    turned = extent(depth(0, 90.0))
    assert upright[0] > upright[1]
    assert turned[1] > turned[0]
    # the same slot rotates the other glyph too
    dash_upright = extent(depth(1, 0.0))
    dash_turned = extent(depth(1, 90.0))
    assert dash_upright[1] > dash_upright[0]
    assert dash_turned[0] > dash_turned[1]


def test_every_slot_stamps_wholly_inside_the_canvas():
    placements = make_placements()
    rows, columns = placements.count_shape
    for corner in ((0, 0), (0, columns - 1), (rows - 1, 0), (rows - 1, columns - 1)):
        counts = jnp.zeros((rows, columns, 2)).at[corner[0], corner[1], 0].set(1.0)
        ink = placements.pose_ink_field(counts, jnp.zeros((rows, columns)))
        assert ink.shape == (SIDE, SIDE, 4)
        assert np.isfinite(np.asarray(ink)).all()
        assert float(jnp.sum(ink[..., 3])) > 0.0


def test_the_model_runs_and_pose_carries_a_gradient():
    placements = make_placements()
    model = functools.partial(poisson_convsc_model, placements=placements,
                              likelihood="gmrf")
    image = jnp.full((1, SIDE, SIDE, 3), 0.95).at[:, 8:15, 12, :].set(0.2)
    model_trace = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
        image, plot_mean=True
    )
    assert model_trace["a"]["value"].shape == (1, *placements.count_shape, 2)
    assert model_trace["pose"]["value"].shape == (1, *placements.count_shape)
    assert model_trace["mean"]["value"].shape == (1, SIDE, SIDE, 3)
    for name, site in model_trace.items():
        if site["type"] == "sample":
            assert np.isfinite(site["fn"].log_prob(site["value"])).all(), name

    values = {name: site["value"] for name, site in model_trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    relaxed = dict(values, a=values["a"].astype(jnp.float32) + 0.3)
    gradients = jax.grad(
        lambda v: log_density(model, (image,), {}, v)[0]
    )(relaxed)
    for name, gradient in gradients.items():
        assert np.isfinite(np.asarray(gradient)).all(), name
    assert np.abs(np.asarray(gradients["pose"])).max() > 0.0
    assert np.abs(np.asarray(gradients["a"])).max() > 0.0


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError, match="require_on_canvas"):
        make_placements(require_on_canvas=False)
    with pytest.raises(ValueError, match="pose_sigma_degrees"):
        make_placements(pose_sigma_degrees=0.0)
