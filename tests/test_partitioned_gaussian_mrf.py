"""Checks on :class:`src.distributions.spatial.PartitionedGaussianMrf`.

Runs under pytest if it is installed, and as a plain script otherwise::

    uv run python tests/test_partitioned_gaussian_mrf.py

Every tolerance here assumes float64, which is enabled below. The scan-based
log-determinant is an EXACT factorization rather than an approximation, so the
comparisons against dense linear algebra are held to machine precision.
"""
import itertools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.distributions.layers import AlphaFormat, Layer
from src.distributions.spatial import PartitionedGaussianMrf


# -- fixtures, built by hand so the tests do not need the glyph dictionary --

def make_layer(h=9, w=7, channels=3, batch_shape=(), seed=0):
    """A layer with three regions, so some lattice edges get deleted."""
    rng = np.random.default_rng(seed)
    lead = tuple(batch_shape)
    count = np.zeros(lead + (h, w), dtype=np.int32)
    count[..., 2:6, 1:4] = 1
    count[..., 3:5, 3:6] = 2
    coverage = np.where(count > 0, 0.8, 0.0)
    image = rng.uniform(0.1, 0.9, lead + (h, w, channels))
    return Layer(count=jnp.asarray(count), coverage=jnp.asarray(coverage),
                 image=jnp.asarray(image), format=AlphaFormat.STRAIGHT)


def make_field(h=9, w=7, channels=3, batch_shape=(), seed=0, bond=None):
    layer = make_layer(h, w, channels, batch_shape, seed)
    rng = np.random.default_rng(seed + 1)
    lead = tuple(batch_shape)
    tau = jnp.asarray(rng.uniform(40.0, 400.0, lead + (h, w)))
    if bond is None:
        bond = rng.uniform(50.0, 800.0, lead)
    return PartitionedGaussianMrf(layer, tau, jnp.asarray(bond))


def dense_precision(tau, kv, kh):
    """Build Q by hand, independently of the class, for one unbatched field."""
    tau = np.asarray(tau); kv = np.asarray(kv); kh = np.asarray(kh)
    h, w = tau.shape
    n = h * w
    idx = lambda y, x: y * w + x
    Q = np.diag(tau.ravel().astype(float))
    for y in range(h - 1):
        for x in range(w):
            k = float(kv[y, x]); i, j = idx(y, x), idx(y + 1, x)
            Q[i, i] += k; Q[j, j] += k; Q[i, j] -= k; Q[j, i] -= k
    for y in range(h):
        for x in range(w - 1):
            k = float(kh[y, x]); i, j = idx(y, x), idx(y, x + 1)
            Q[i, i] += k; Q[j, j] += k; Q[i, j] -= k; Q[j, i] -= k
    return Q


# -- 1. shapes: leading dimensions batch, trailing three are the event ------

def test_unbatched_shapes():
    d = make_field(h=9, w=7, channels=3)
    assert d.batch_shape == ()
    assert d.event_shape == (9, 7, 3)
    x = d.sample(jax.random.PRNGKey(0))
    assert x.shape == (9, 7, 3)
    assert d.log_prob(x).shape == ()
    assert d.sample(jax.random.PRNGKey(0), (4,)).shape == (4, 9, 7, 3)
    assert d.mean.shape == (9, 7, 3)


def test_batched_shapes():
    for spec in ("layer", "tau", "bond"):
        if spec == "layer":
            d = make_field(batch_shape=(5,))
        elif spec == "tau":
            layer = make_layer()
            tau = jnp.broadcast_to(jnp.linspace(50.0, 300.0, 5)[:, None, None],
                                   (5, 9, 7))
            d = PartitionedGaussianMrf(layer, tau, jnp.asarray(200.0))
        else:
            layer = make_layer()
            tau = jnp.full((9, 7), 120.0)
            d = PartitionedGaussianMrf(layer, tau,
                                       jnp.linspace(60.0, 600.0, 5))
        assert d.batch_shape == (5,), spec
        assert d.event_shape == (9, 7, 3), spec
        x = d.sample(jax.random.PRNGKey(0))
        assert x.shape == (5, 9, 7, 3), spec
        assert d.log_prob(x).shape == (5,), spec
        assert d.sample(jax.random.PRNGKey(0), (2,)).shape == (2, 5, 9, 7, 3), spec


def test_two_batch_dimensions():
    d = make_field(batch_shape=(2, 3))
    assert d.batch_shape == (2, 3)
    assert d.event_shape == (9, 7, 3)
    x = d.sample(jax.random.PRNGKey(0))
    assert x.shape == (2, 3, 9, 7, 3)
    assert d.log_prob(x).shape == (2, 3)


# -- 2. the channel axis is the rightmost event dimension -------------------

def test_channels_are_the_rightmost_event_axis():
    for channels in (1, 3, 4):
        d = make_field(channels=channels)
        assert d.event_shape == (9, 7, channels)
        assert d.event_shape[-1] == channels
        x = d.sample(jax.random.PRNGKey(0))
        assert x.shape[-1] == channels


def test_log_prob_scales_with_channels():
    """Independent channels: the density of C copies is C times one channel's."""
    layer1 = make_layer(channels=1)
    tau = jnp.full((9, 7), 150.0)
    one = PartitionedGaussianMrf(layer1, tau, jnp.asarray(300.0))

    layer3 = Layer(count=layer1.count, coverage=layer1.coverage,
                   image=jnp.repeat(layer1.image, 3, axis=-1),
                   format=AlphaFormat.STRAIGHT)
    three = PartitionedGaussianMrf(layer3, tau, jnp.asarray(300.0))

    r = jax.random.normal(jax.random.PRNGKey(2), (9, 7, 1)) * 0.05
    lp1 = one.log_prob(one.mean + r)
    lp3 = three.log_prob(three.mean + jnp.repeat(r, 3, axis=-1))
    assert jnp.allclose(lp3, 3.0 * lp1, rtol=1e-12), (float(lp1), float(lp3))


# -- 3. the log-determinant: scan against dense ----------------------------

def test_logdet_scan_matches_dense_by_hand():
    """The headline check, against a Q assembled independently in this file."""
    for h, w in ((5, 4), (9, 7), (12, 12), (16, 11)):
        d = make_field(h=h, w=w)
        tau, (kv, kh) = d.precision_parameters
        # logdet_precision covers the whole event, so scale the one-channel
        # determinant by the channel count.
        want = (d.event_shape[-1]
                * np.linalg.slogdet(dense_precision(tau, kv, kh))[1])
        got = float(d.logdet_precision("scan"))
        assert abs(got - want) < 1e-8 * max(1.0, abs(want)), (h, w, got, want)


def test_logdet_scan_matches_the_dense_method():
    d = make_field(h=10, w=8)
    scan = float(d.logdet_precision("scan"))
    dense = float(d.logdet_precision("dense"))
    assert abs(scan - dense) < 1e-8 * max(1.0, abs(dense)), (scan, dense)


def test_precision_matrix_matches_the_hand_built_one():
    d = make_field(h=8, w=6)
    tau, (kv, kh) = d.precision_parameters
    assert np.allclose(np.asarray(d.precision_matrix),
                       dense_precision(tau, kv, kh), atol=1e-9)


def test_logdet_batched_elementwise():
    d = make_field(h=9, w=7, batch_shape=(4,))
    tau, (kv, kh) = d.precision_parameters
    scan = np.asarray(d.logdet_precision("scan"))
    assert scan.shape == (4,)
    for b in range(4):
        want = (d.event_shape[-1]
                * np.linalg.slogdet(dense_precision(tau[b], kv[b], kh[b]))[1])
        assert abs(scan[b] - want) < 1e-8 * max(1.0, abs(want)), b
    assert np.allclose(scan, np.asarray(d.logdet_precision("dense")), atol=1e-8)


def test_logdet_rejects_an_unknown_method():
    d = make_field()
    try:
        d.logdet_precision("magic")
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError")


def test_logdet_does_not_retrace_when_the_render_changes():
    """The scan's shapes do not depend on the count field, so one trace only."""
    traces = {"n": 0}

    @jax.jit
    def logdet(tau, kv, kh):
        from src.distributions.spatial import _logdet_scan
        traces["n"] += 1
        return _logdet_scan(tau, kv, kh)

    base = make_field(h=9, w=7)
    other = make_field(h=9, w=7, seed=7)
    for d in (base, other, base):
        tau, (kv, kh) = d.precision_parameters
        logdet(tau, kv, kh)
    assert traces["n"] == 1, traces["n"]


# -- 4. the rest of the density, so the log det is not checked alone --------

def test_log_prob_matches_a_dense_multivariate_normal():
    d = make_field(h=8, w=6, channels=2)
    tau, (kv, kh) = d.precision_parameters
    Q = dense_precision(tau, kv, kh)
    x = d.mean + jax.random.normal(jax.random.PRNGKey(5), d.event_shape) * 0.03
    r = np.asarray(x - d.mean)

    n, channels = Q.shape[0], d.event_shape[-1]
    quad = sum(r[..., c].ravel() @ Q @ r[..., c].ravel() for c in range(channels))
    want = 0.5 * (channels * np.linalg.slogdet(Q)[1]
                  - n * channels * np.log(2 * np.pi) - quad)
    got = float(d.log_prob(x))
    assert abs(got - want) < 1e-7 * max(1.0, abs(want)), (got, want)


def test_precision_is_block_diagonal_across_regions():
    d = make_field(h=9, w=7)
    tau, (kv, kh) = d.precision_parameters
    Q = dense_precision(tau, kv, kh)
    labels = np.asarray(d._layer.count).ravel()
    parts = 0.0
    for a in np.unique(labels):
        ia = np.flatnonzero(labels == a)
        parts += np.linalg.slogdet(Q[np.ix_(ia, ia)])[1]
        for b in np.unique(labels):
            if b > a:
                ib = np.flatnonzero(labels == b)
                assert np.abs(Q[np.ix_(ia, ib)]).max() == 0.0
    total = float(d.logdet_precision("scan")) / d.event_shape[-1]
    assert abs(parts - total) < 1e-8 * abs(parts), (parts, total)


def test_sampler_second_moment():
    """E[x' Q x] = n for a draw from N(loc, Q^-1), per channel."""
    d = make_field(h=7, w=6, channels=1)
    tau, (kv, kh) = d.precision_parameters
    Q = dense_precision(tau, kv, kh)
    draws = np.asarray(d.sample(jax.random.PRNGKey(11), (2000,)) - d.mean)
    flat = draws[..., 0].reshape(2000, -1)
    quad = np.einsum("si,ij,sj->s", flat, Q, flat).mean()
    n = Q.shape[0]
    assert abs(quad / n - 1.0) < 0.06, (quad, n)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    raise SystemExit(1 if failed else 0)
