"""Checks for stationary-velocity diffeomorphic image warps."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.diffeomorphism import (
    affine_basis,
    affine_free_velocity,
    bilinear_sample,
    boundary_taper,
    compose_displacements,
    coordinate_grid,
    diffeomorphic_warp,
    jacobian_determinant,
    resize_velocity,
    scaling_and_squaring,
    sparse_diffeomorphic_stamp,
)
from src.model.model import _stamp


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


def test_affine_free_velocity_annihilates_all_six_modes():
    basis = affine_basis((9, 8), dtype=jnp.float64)
    coefficients = jnp.asarray(
        [[1.2, -0.7], [0.4, 0.9], [-0.3, 0.2]], dtype=jnp.float64
    )
    velocity = basis @ coefficients
    projected = affine_free_velocity(velocity, lambda value: value, (9, 8))
    assert np.allclose(projected, 0.0, atol=1e-12)


def test_affine_free_velocity_is_differentiable():
    grid = coordinate_grid((13, 11), dtype=jnp.float64)
    source = jnp.sin(0.17 * grid[..., 0]) * jnp.cos(0.13 * grid[..., 1])
    velocity = smooth_velocity(13, 11)
    window = boundary_taper(velocity.shape[:2], dtype=velocity.dtype)

    def loss(argument):
        projected = affine_free_velocity(
            argument,
            lambda value: value,
            velocity.shape[:2],
            window=window,
        )
        return jnp.square(diffeomorphic_warp(source, projected)).sum()

    gradient = jax.grad(loss)(velocity)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0.0


def test_affine_free_velocity_preserves_a_fixed_boundary():
    output_shape = (13, 11)
    velocity = smooth_velocity(7, 6)
    window = boundary_taper(output_shape, dtype=velocity.dtype)
    projected = affine_free_velocity(
        velocity,
        lambda value: value,
        output_shape,
        window=window,
    )
    basis = affine_basis(output_shape, dtype=velocity.dtype)
    moments = jnp.einsum("hwk,hwc->kc", basis, projected)
    boundary = jnp.concatenate(
        (projected[0], projected[-1], projected[:, 0], projected[:, -1])
    )
    assert np.allclose(boundary, 0.0, atol=1e-12)
    assert np.allclose(moments, 0.0, atol=1e-11)


def test_affine_free_velocity_respects_the_supplied_covariance():
    basis = affine_basis((7, 6), dtype=jnp.float64)
    velocity = smooth_velocity(7, 6)
    variance = jnp.linspace(0.5, 2.0, 7 * 6).reshape(7, 6)

    def covariance_solve(value):
        return variance[..., None] * value

    actual = affine_free_velocity(velocity, covariance_solve, (7, 6))
    covariance_basis = variance[..., None] * basis
    conditional_gram = jnp.einsum(
        "hwk,hwl->kl", basis, covariance_basis
    )
    coordinates = jnp.einsum("hwk,hwc->kc", basis, velocity)
    coefficients = jnp.linalg.solve(conditional_gram, coordinates)
    expected = velocity - jnp.einsum(
        "hwk,kc->hwc", covariance_basis, coefficients
    )
    assert np.allclose(actual, expected, atol=1e-12)


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


def test_sparse_stamp_is_differentiable():
    amplitudes = jnp.asarray([0.8, 1.1])
    centers = jnp.asarray([[5, 5], [8, 9]], dtype=jnp.int32)
    glyph_indices = jnp.asarray([0, 1], dtype=jnp.int32)
    ink_kernel = jnp.linspace(0.0, 1.0, 5 * 4 * 3 * 2).reshape(5, 4, 3, 2)
    velocities = jnp.stack((smooth_velocity(5, 4), -smooth_velocity(5, 4)))

    def loss(amplitude_argument, velocity_argument):
        canvas = sparse_diffeomorphic_stamp(
            amplitude_argument,
            centers,
            glyph_indices,
            ink_kernel,
            velocity_argument,
            canvas_shape=(13, 15),
        )
        return jnp.square(canvas).sum()

    amplitude_gradient, velocity_gradient = jax.grad(loss, argnums=(0, 1))(
        amplitudes, velocities
    )
    assert np.isfinite(amplitude_gradient).all()
    assert np.isfinite(velocity_gradient).all()
    assert np.linalg.norm(amplitude_gradient) > 0.0
    assert np.linalg.norm(velocity_gradient) > 0.0


def test_sparse_stamp_matches_the_convolutional_renderer_at_zero_velocity():
    amplitudes = jnp.asarray([1.0, 0.4, 1.2, 0.6])
    canvas_shape = (10, 12)
    centers = jnp.asarray(
        [[0, 0], [5, 6], [5, 6], [9, 11]], dtype=jnp.int32
    )
    glyph_indices = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    ink_kernel = jnp.linspace(0.01, 0.8, 4 * 6 * 3 * 2).reshape(4, 6, 3, 2)
    velocities = jnp.zeros((4, 4, 6, 2))
    counts = jnp.zeros((*canvas_shape, 2)).at[
        centers[:, 0], centers[:, 1], glyph_indices
    ].add(amplitudes)

    actual = sparse_diffeomorphic_stamp(
        amplitudes,
        centers,
        glyph_indices,
        ink_kernel,
        velocities,
        canvas_shape=canvas_shape,
    )
    expected = _stamp(counts[None], ink_kernel)[0]
    assert np.allclose(actual, expected, atol=1e-12)


def test_sparse_stamp_superposes_independent_occurrences():
    amplitudes = jnp.asarray([0.7, 1.3])
    canvas_shape = (14, 16)
    centers = jnp.asarray([[5, 5], [9, 11]], dtype=jnp.int32)
    glyph_indices = jnp.asarray([0, 1], dtype=jnp.int32)
    ink_kernel = jnp.linspace(0.0, 0.9, 5 * 4 * 2 * 2).reshape(5, 4, 2, 2)
    velocities = jnp.stack((smooth_velocity(5, 4), -smooth_velocity(5, 4)))
    together = sparse_diffeomorphic_stamp(
        amplitudes,
        centers,
        glyph_indices,
        ink_kernel,
        velocities,
        canvas_shape=canvas_shape,
    )
    separate = sum(
        sparse_diffeomorphic_stamp(
            amplitudes[index:index + 1],
            centers[index:index + 1],
            glyph_indices[index:index + 1],
            ink_kernel,
            velocities[index:index + 1],
            canvas_shape=canvas_shape,
        )
        for index in range(2)
    )
    assert np.allclose(together, separate, atol=1e-12)
