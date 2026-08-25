"""Checks on :class:`LocallyScaledGaussianMrf` against independent forms."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.distributions.layers import AlphaFormat, Layer
from src.distributions.spatial import (LocallyScaledGaussianMrf,
                                       PartitionedGaussianMrf)


def dense_precision(element_precision, horizontal_precision,
                    vertical_precision):
    """Build the one-channel base precision independently."""
    element_precision = np.asarray(element_precision)
    horizontal_precision = np.asarray(horizontal_precision)
    vertical_precision = np.asarray(vertical_precision)
    height, width = element_precision.shape
    precision = np.diag(element_precision.ravel().astype(float))

    def index(y, x):
        return y * width + x

    for y in range(height - 1):
        for x in range(width):
            bond = float(vertical_precision[y, x])
            first, second = index(y, x), index(y + 1, x)
            precision[first, first] += bond
            precision[first, second] -= bond
            precision[second, first] -= bond
            precision[second, second] += bond
    for y in range(height):
        for x in range(width - 1):
            bond = float(horizontal_precision[y, x])
            first, second = index(y, x), index(y, x + 1)
            precision[first, first] += bond
            precision[first, second] -= bond
            precision[second, first] -= bond
            precision[second, second] += bond
    return precision


def explicit_stencil_log_prob(base, local_precision, value):
    """Section 6's site-and-edge implementation, independent of the wrapper."""
    element_precision, (vertical_precision, horizontal_precision) = (
        base.precision_parameters
    )
    residual = value - base.loc
    scaled_residual = jnp.sqrt(local_precision)[..., None] * residual
    horizontal_difference = (
        scaled_residual[:, 1:, :] - scaled_residual[:, :-1, :]
    )
    vertical_difference = (
        scaled_residual[1:, :, :] - scaled_residual[:-1, :, :]
    )
    quadratic = (
        jnp.sum(element_precision[..., None] * scaled_residual**2)
        + jnp.sum(
            horizontal_precision[..., None] * horizontal_difference**2
        )
        + jnp.sum(vertical_precision[..., None] * vertical_difference**2)
    )
    log_determinant = (
        base.logdet_precision()
        + value.shape[-1] * jnp.log(local_precision).sum()
    )
    return 0.5 * (
        log_determinant - value.size * jnp.log(2.0 * jnp.pi) - quadratic
    )


def make_distributions(height=5, width=4, channels=3):
    """Return the base and locally scaled fields plus their local precision."""
    count = jnp.zeros((height, width), dtype=jnp.int32)
    count = count.at[1:4, 1:3].set(1)
    coverage = jnp.where(count > 0, 0.75, 0.0)
    image = jnp.linspace(
        0.15, 0.85, height * width * channels
    ).reshape(height, width, channels)
    layer = Layer(count=count, coverage=coverage, image=image,
                  format=AlphaFormat.STRAIGHT)
    element_precision = jnp.linspace(
        25.0, 90.0, height * width
    ).reshape(height, width)
    local_precision = jnp.linspace(
        0.25, 2.5, height * width
    ).reshape(height, width)
    bond_precision = jnp.asarray(35.0)
    base = PartitionedGaussianMrf(
        layer, element_precision, bond_precision, cg_iters=800
    )
    scaled = LocallyScaledGaussianMrf(
        layer, element_precision, bond_precision, local_precision,
        cg_iters=800
    )
    return base, local_precision, scaled


def recommended_log_prob(base, local_precision, value):
    """The change-of-variables wrapper recommended in the notebook review."""
    residual = value - base.loc
    scaled_value = (
        base.loc + jnp.sqrt(local_precision)[..., None] * residual
    )
    log_jacobian = (
        0.5 * value.shape[-1] * jnp.log(local_precision).sum()
    )
    return base.log_prob(scaled_value) + log_jacobian


def test_batch_shapes_include_local_precision():
    base, local_precision, _ = make_distributions()
    batched_precision = jnp.broadcast_to(local_precision, (4,) + local_precision.shape)
    scaled = LocallyScaledGaussianMrf(
        base._layer,
        base.element_precision,
        base.bond_precision,
        batched_precision,
    )
    assert scaled.batch_shape == (4,)
    assert scaled.log_prob(jnp.broadcast_to(base.mean, (4,) + base.event_shape)).shape == (4,)
    assert scaled.sample(jax.random.PRNGKey(0), (2,)).shape == (
        2, 4, *base.event_shape
    )


def test_density_matches_dense_multivariate_normal():
    base, local_precision, scaled = make_distributions(channels=2)
    element_precision, (vertical_precision, horizontal_precision) = (
        base.precision_parameters
    )
    base_precision = dense_precision(
        element_precision, horizontal_precision, vertical_precision
    )
    diagonal = np.diag(np.sqrt(np.asarray(local_precision).ravel()))
    precision = diagonal @ base_precision @ diagonal
    value = base.mean + 0.04 * jax.random.normal(
        jax.random.PRNGKey(1), base.event_shape
    )
    residual = np.asarray(value - base.mean)
    quadratic = sum(
        residual[..., channel].ravel()
        @ precision
        @ residual[..., channel].ravel()
        for channel in range(base.event_shape[-1])
    )
    dimension = precision.shape[0] * base.event_shape[-1]
    expected = 0.5 * (
        base.event_shape[-1] * np.linalg.slogdet(precision)[1]
        - dimension * np.log(2.0 * np.pi)
        - quadratic
    )
    actual = float(scaled.log_prob(value))
    assert abs(actual - expected) < 1e-9 * max(1.0, abs(expected))


def test_density_matches_notebook_stencil():
    base, local_precision, scaled = make_distributions()
    value = base.mean + 0.03 * jax.random.normal(
        jax.random.PRNGKey(2), base.event_shape
    )
    expected = explicit_stencil_log_prob(base, local_precision, value)
    assert jnp.allclose(scaled.log_prob(value), expected, rtol=1e-12)


def test_density_matches_recommended_wrapper():
    base, local_precision, scaled = make_distributions()
    value = base.mean + 0.03 * jax.random.normal(
        jax.random.PRNGKey(3), base.event_shape
    )
    expected = recommended_log_prob(base, local_precision, value)
    assert jnp.allclose(scaled.log_prob(value), expected, rtol=1e-12)


def test_gradients_are_finite():
    base, local_precision, _ = make_distributions()
    value = base.mean + 0.03 * jax.random.normal(
        jax.random.PRNGKey(7), base.event_shape
    )

    def objective(precision):
        scaled = LocallyScaledGaussianMrf(
            base._layer,
            base.element_precision,
            base.bond_precision,
            precision,
        )
        return scaled.log_prob(value)

    gradient = jax.grad(objective)(local_precision)
    assert jnp.all(jnp.isfinite(gradient))


def test_local_precision_one_recovers_base():
    base, _, _ = make_distributions()
    scaled = LocallyScaledGaussianMrf(
        base._layer,
        base.element_precision,
        base.bond_precision,
        jnp.ones(base.event_shape[:-1]),
    )
    value = base.mean + 0.05 * jax.random.normal(
        jax.random.PRNGKey(4), base.event_shape
    )
    assert jnp.allclose(scaled.log_prob(value), base.log_prob(value), rtol=1e-12)


def test_sample_matches_recommended_transform():
    base, local_precision, scaled = make_distributions()
    key = jax.random.PRNGKey(5)
    base_sample = base.sample(key, sample_shape=(3,))
    expected = (
        base.mean
        + (base_sample - base.mean)
        / jnp.sqrt(local_precision)[..., None]
    )
    actual = scaled.sample(key, sample_shape=(3,))
    assert jnp.allclose(actual, expected, rtol=1e-12)


def test_sampler_second_moment():
    base, local_precision, scaled = make_distributions(channels=1)
    element_precision, (vertical_precision, horizontal_precision) = (
        base.precision_parameters
    )
    precision = dense_precision(
        element_precision, horizontal_precision, vertical_precision
    )
    draws = scaled.sample(jax.random.PRNGKey(6), sample_shape=(2000,))
    residual = np.asarray(draws - scaled.mean)
    transformed = (
        np.sqrt(np.asarray(local_precision))[None, ..., None] * residual
    )[..., 0].reshape(2000, -1)
    quadratic = np.einsum(
        "si,ij,sj->s", transformed, precision, transformed
    ).mean()
    assert abs(quadratic / precision.shape[0] - 1.0) < 0.06


def test_support_is_full_image_real_event():
    base, _, scaled = make_distributions()
    assert scaled.support.event_dim == 3
    assert jnp.all(scaled.support(base.mean))
