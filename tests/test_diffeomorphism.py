"""Checks for stationary-velocity diffeomorphic image warps."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.diffeomorphism import (
    bilinear_sample,
    compose_displacements,
    coordinate_grid,
    diffeomorphic_warp,
    jacobian_determinant,
    resize_velocity,
    scaling_and_squaring,
)


def smooth_velocity(height=32, width=28):
    grid = coordinate_grid((height, width), dtype=jnp.float64)
    x = grid[..., 1]
    x_center = 0.5 * (width - 1)
    x_taper = jnp.sin(jnp.pi * x / (width - 1))
    y = grid[..., 0]
    y_center = 0.5 * (height - 1)
    y_taper = jnp.sin(jnp.pi * y / (height - 1))
    taper = x_taper * y_taper
    return 0.06 * taper[..., None] * jnp.stack(
        (x - x_center, -(y - y_center)), axis=-1
    )


def test_bilinear_sample_is_exact_for_a_plane():
    grid = coordinate_grid((6, 7), dtype=jnp.float64)
    plane = 2.0 * grid[..., 0] + 3.0 * grid[..., 1]
    points = jnp.asarray([[1.25, 2.5], [3.75, 4.125]])
    actual = bilinear_sample(plane, points)
    expected = 2.0 * points[:, 0] + 3.0 * points[:, 1]
    assert np.allclose(actual, expected, atol=1e-12)


def test_composition_adds_constant_displacements():
    inner = jnp.broadcast_to(jnp.asarray([-0.5, 0.25]), (8, 9, 2))
    outer = jnp.broadcast_to(jnp.asarray([1.0, 2.0]), (8, 9, 2))
    actual = compose_displacements(outer, inner)
    assert np.allclose(actual, inner + outer, atol=1e-12)


def test_diffeomorphic_warp_uses_the_inverse_map():
    grid = coordinate_grid((8, 9), dtype=jnp.float64)
    source = grid[..., 1]
    velocity = jnp.broadcast_to(jnp.asarray([0.0, 1.0]), (8, 9, 2))
    warped = diffeomorphic_warp(source, velocity)
    assert np.allclose(warped[:, 1:], source[:, :-1], atol=1e-12)


def test_gradients_are_finite():
    grid = coordinate_grid((12, 10), dtype=jnp.float64)
    source = jnp.sin(0.3 * grid[..., 0]) * jnp.cos(0.2 * grid[..., 1])
    velocity = smooth_velocity(12, 10)

    def loss(argument):
        return jnp.square(diffeomorphic_warp(source, argument)).sum()

    gradient = jax.grad(loss)(velocity)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0.0


def test_identity_velocity_preserves_an_image():
    source = jnp.arange(9 * 7 * 3, dtype=jnp.float64).reshape(9, 7, 3)
    velocity = jnp.zeros((9, 7, 2), dtype=jnp.float64)
    assert np.array_equal(diffeomorphic_warp(source, velocity), source)


def test_jacobian_determinant_matches_an_affine_map():
    grid = coordinate_grid((9, 8), dtype=jnp.float64)
    x_scale = 0.08
    y_scale = -0.05
    displacement = jnp.stack(
        (y_scale * grid[..., 0], x_scale * grid[..., 1]), axis=-1
    )
    expected = (1.0 + y_scale) * (1.0 + x_scale)
    assert np.allclose(
        jacobian_determinant(displacement), expected, atol=1e-12
    )


def test_resize_velocity_preserves_constant_pixel_units():
    velocity = jnp.broadcast_to(jnp.asarray([1.5, -2.0]), (4, 3, 2))
    resized = resize_velocity(velocity, (13, 11))
    assert resized.shape == (13, 11, 2)
    assert np.allclose(resized, jnp.asarray([1.5, -2.0]), atol=1e-6)


def test_scaling_and_squaring_has_a_numerical_inverse():
    velocity = smooth_velocity()
    forward = scaling_and_squaring(velocity, squaring_steps=8)
    inverse = scaling_and_squaring(-velocity, squaring_steps=8)
    residual = compose_displacements(forward, inverse)
    assert np.max(np.abs(residual[2:-2, 2:-2])) < 0.012
    assert np.min(jacobian_determinant(forward)) > 0.85


def test_scaling_and_squaring_preserves_constant_velocity():
    velocity = jnp.broadcast_to(jnp.asarray([1.25, -0.75]), (8, 9, 2))
    displacement = scaling_and_squaring(velocity, squaring_steps=7)
    assert np.allclose(displacement, velocity, atol=1e-12)
