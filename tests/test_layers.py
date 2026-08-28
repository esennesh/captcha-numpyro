"""Round-trip checks on :class:`src.distributions.layers.Layer` alpha formats.

Every tolerance assumes float64, enabled below.
"""
import itertools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.distributions.layers import (AlphaFormat, Layer,
                                      alpha_to_closeness,
                                      closeness_to_alpha)

CONVERT = {
    AlphaFormat.STRAIGHT: Layer.to_straight,
    AlphaFormat.PREMULTIPLIED: Layer.to_premultiplied,
    AlphaFormat.CLOSENESS_PREMULTIPLIED: Layer.to_closeness_premultiplied,
}


def straight_layer(shape=(6, 5), channels=3, seed=0, alpha_low=0.02,
                   alpha_high=0.95):
    rng = np.random.default_rng(seed)
    return Layer(count=jnp.zeros(shape, jnp.int32),
                 coverage=jnp.asarray(rng.uniform(alpha_low, alpha_high, shape)),
                 image=jnp.asarray(rng.uniform(0.05, 0.95, shape + (channels,))),
                 format=AlphaFormat.STRAIGHT)


@pytest.mark.parametrize("first,second",
                         list(itertools.permutations(AlphaFormat, 2)))
def test_round_trip_through_two_formats(first, second):
    base = straight_layer()
    there = CONVERT[second](CONVERT[first](base))
    assert there.format is second
    back = there.to_straight()
    assert jnp.allclose(back.image, base.image, atol=1e-12)
    assert jnp.allclose(back.coverage, base.coverage, atol=1e-12)


@pytest.mark.parametrize("fmt", list(AlphaFormat))
def test_round_trip_is_idempotent(fmt):
    layer = CONVERT[fmt](straight_layer())
    again = CONVERT[fmt](layer)
    assert jnp.allclose(again.image, layer.image, atol=1e-12)
    assert jnp.allclose(again.coverage, layer.coverage, atol=1e-12)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_round_trip_batches(batch_shape):
    base = straight_layer(shape=tuple(batch_shape) + (6, 5))
    back = base.to_closeness_premultiplied().to_premultiplied().to_straight()
    assert back.image.shape == base.image.shape
    assert jnp.allclose(back.image, base.image, atol=1e-12)


def test_closeness_holds_the_full_colour_not_just_the_hue():
    """Two colours of the same hue but different value must stay distinct."""
    dim = jnp.array([[[0.1, 0.12, 0.275]]])
    bright = dim * 2.0
    alpha = jnp.array([[0.6]])
    out = []
    for colour in (dim, bright):
        layer = Layer(count=jnp.zeros((1, 1), jnp.int32), coverage=alpha,
                      image=colour, format=AlphaFormat.STRAIGHT)
        out.append(layer.to_closeness_premultiplied().to_straight().image)
    assert jnp.allclose(out[0], dim, atol=1e-12)
    assert jnp.allclose(out[1], bright, atol=1e-12)
    assert not jnp.allclose(out[0], out[1])


def test_zero_coverage_cannot_return_a_colour():
    """Documented limit: premultiplying by zero is not invertible."""
    layer = Layer(count=jnp.zeros((1, 1), jnp.int32),
                  coverage=jnp.zeros((1, 1)),
                  image=jnp.array([[[0.2, 0.4, 0.6]]]),
                  format=AlphaFormat.STRAIGHT)
    for fmt in (AlphaFormat.PREMULTIPLIED, AlphaFormat.CLOSENESS_PREMULTIPLIED):
        back = CONVERT[fmt](layer).to_straight()
        assert jnp.allclose(back.image, 0.0)
        assert jnp.allclose(back.coverage, 0.0)


def test_full_coverage_is_clipped_not_infinite():
    """Documented limit: the closeness of alpha = 1 is infinite, so it clips."""
    layer = Layer(count=jnp.zeros((1, 1), jnp.int32),
                  coverage=jnp.ones((1, 1)),
                  image=jnp.array([[[0.2, 0.4, 0.6]]]),
                  format=AlphaFormat.STRAIGHT)
    closeness = layer.to_closeness_premultiplied()
    assert jnp.all(jnp.isfinite(closeness.coverage))
    assert jnp.allclose(closeness.coverage, -np.log(1e-3), atol=1e-9)
    back = closeness.to_straight()
    assert jnp.allclose(back.coverage, 1.0 - 1e-3, atol=1e-9)
    assert jnp.allclose(back.image, layer.image, atol=1e-9)


def test_alpha_closeness_inverse():
    alpha = jnp.linspace(0.0, 0.9, 25)
    assert jnp.allclose(closeness_to_alpha(alpha_to_closeness(alpha)), alpha,
                        atol=1e-12)


def test_over_background_is_alpha_compositing():
    base = straight_layer()
    want = (base.coverage[..., None] * base.image
            + (1.0 - base.coverage[..., None]) * 1.0)
    for fmt in AlphaFormat:
        got = CONVERT[fmt](base).over_background(1.0)
        assert jnp.allclose(got, want, atol=1e-12), fmt
        assert float(got.max()) <= 1.0 + 1e-12, fmt


def test_multiplication_composites_coverage_and_averages_colour():
    """``__mul__`` adds closeness, which is exact alpha compositing."""
    left = straight_layer(seed=1)
    right = straight_layer(seed=2)
    product = (left * right).to_straight()

    a_l, a_r = left.coverage, right.coverage
    want_alpha = 1.0 - (1.0 - jnp.clip(a_l, 0., 1 - 1e-3)) * (
        1.0 - jnp.clip(a_r, 0., 1 - 1e-3))
    assert jnp.allclose(product.coverage, want_alpha, atol=1e-12)

    t_l = alpha_to_closeness(a_l)[..., None]
    t_r = alpha_to_closeness(a_r)[..., None]
    want_colour = (left.image * t_l + right.image * t_r) / (t_l + t_r)
    assert jnp.allclose(product.image, want_colour, atol=1e-12)
