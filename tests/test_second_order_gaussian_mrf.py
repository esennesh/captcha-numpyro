"""Checks on the normalized curvature prior in ``SecondOrderGaussianMrf``."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.distributions.spatial import SecondOrderGaussianMrf


def dense_operator(element, horizontal, vertical):
    """Build the one-channel factor ``A`` independently of the class."""
    element = np.asarray(element)
    horizontal = np.asarray(horizontal)
    vertical = np.asarray(vertical)
    height, width = element.shape
    operator = np.diag(element.ravel().astype(float))

    def index(y, x):
        return y * width + x

    for y in range(height - 1):
        for x in range(width):
            bond = float(vertical[y, x])
            first, second = index(y, x), index(y + 1, x)
            operator[first, first] += bond
            operator[first, second] -= bond
            operator[second, first] -= bond
            operator[second, second] += bond
    for y in range(height):
        for x in range(width - 1):
            bond = float(horizontal[y, x])
            first, second = index(y, x), index(y, x + 1)
            operator[first, first] += bond
            operator[first, second] -= bond
            operator[second, first] -= bond
            operator[second, second] += bond
    return operator


def make_field(height=5, width=4, channels=2, batch_shape=()):
    lead = tuple(batch_shape)
    element = jnp.linspace(1.5, 3.0, height * width).reshape(height, width)
    loc = jnp.linspace(
        -0.3, 0.4, height * width * channels
    ).reshape(height, width, channels)
    loc = jnp.broadcast_to(loc, lead + loc.shape)
    return SecondOrderGaussianMrf(
        loc,
        element,
        jnp.asarray(0.7),
        cg_iters=500,
    )


def test_batch_shapes():
    field = make_field(batch_shape=(3,))
    assert field.batch_shape == (3,)
    assert field.event_shape == (5, 4, 2)
    assert field.log_prob(field.mean).shape == (3,)
    assert field.sample(jax.random.PRNGKey(0)).shape == (3, 5, 4, 2)
    assert field.sample(jax.random.PRNGKey(1), (2,)).shape == (2, 3, 5, 4, 2)


def test_coverage_constructor_keeps_only_interior_bonds():
    coverage = jnp.asarray(
        [
            [0, 255, 255, 0],
            [0, 255, 255, 0],
            [0, 0, 255, 0],
        ],
        dtype=jnp.uint8,
    )
    field = SecondOrderGaussianMrf.from_coverage(
        coverage,
        jnp.asarray(2.0),
        jnp.asarray(3.0),
        channels=2,
    )
    element, (vertical, horizontal) = field.operator_parameters
    support = np.asarray(coverage) >= 128
    expected_horizontal = support[:, :-1] & support[:, 1:]
    expected_vertical = support[:-1, :] & support[1:, :]

    assert field.event_shape == (3, 4, 2)
    assert np.allclose(field.mean, 0.0)
    assert np.allclose(element, 2.0)
    assert np.array_equal(np.asarray(horizontal) > 0, expected_horizontal)
    assert np.array_equal(np.asarray(vertical) > 0, expected_vertical)
    assert np.allclose(np.asarray(horizontal)[expected_horizontal], 3.0)
    assert np.allclose(np.asarray(vertical)[expected_vertical], 3.0)


def test_coverage_constructor_preserves_a_supplied_mean():
    coverage = jnp.asarray([[0.0, 1.0], [1.0, 1.0]])
    loc = jnp.arange(12.0).reshape(2, 2, 3)
    field = SecondOrderGaussianMrf.from_coverage(
        coverage,
        jnp.asarray(1.5),
        jnp.asarray(0.5),
        loc=loc,
    )
    assert field.event_shape == (2, 2, 3)
    assert jnp.array_equal(field.mean, loc)


def test_log_prob_matches_a_dense_multivariate_normal():
    field = make_field(height=4, width=3, channels=2)
    element, (vertical, horizontal) = field.operator_parameters
    operator = dense_operator(element, horizontal, vertical)
    precision = operator.T @ operator
    value = field.mean + 0.08 * jax.random.normal(
        jax.random.PRNGKey(2), field.event_shape
    )
    residual = np.asarray(value - field.mean)
    quadratic = sum(
        residual[..., channel].ravel()
        @ precision
        @ residual[..., channel].ravel()
        for channel in range(field.event_shape[-1])
    )
    dimension = value.size
    expected = 0.5 * (
        field.event_shape[-1] * np.linalg.slogdet(precision)[1]
        - dimension * np.log(2.0 * np.pi)
        - quadratic
    )
    actual = float(field.log_prob(value))
    assert abs(actual - expected) < 1e-10 * max(1.0, abs(expected))


def test_logdet_scan_matches_dense():
    field = make_field(height=6, width=5, channels=3)
    scan = float(field.logdet_precision("scan"))
    dense = float(field.logdet_precision("dense"))
    assert abs(scan - dense) < 1e-10 * max(1.0, abs(dense))


def test_operator_and_precision_matrices_match_dense_construction():
    field = make_field(height=5, width=4, channels=1)
    element, (vertical, horizontal) = field.operator_parameters
    operator = dense_operator(element, horizontal, vertical)
    assert np.allclose(field.operator_matrix, operator, atol=1e-11)
    assert np.allclose(field.precision_matrix, operator.T @ operator, atol=1e-11)


def test_sampler_second_moment():
    field = make_field(height=4, width=3, channels=1)
    draws = np.asarray(
        field.sample(jax.random.PRNGKey(3), sample_shape=(3000,)) - field.mean
    )[..., 0].reshape(3000, -1)
    precision = np.asarray(field.precision_matrix)
    quadratic = np.einsum(
        "si,ij,sj->s", draws, precision, draws
    ).mean()
    dimension = precision.shape[0]
    assert abs(quadratic / dimension - 1.0) < 0.05


def test_support_is_a_full_real_image_event():
    field = make_field()
    assert field.support.event_dim == 3
    assert jnp.all(field.support(field.mean))
