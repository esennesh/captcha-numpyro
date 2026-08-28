"""Spatial specialization of :class:`numpyro.distributions.MixtureSameFamily`.

``MixtureSameFamily`` mixes over the last *batch* dimension and treats each
remaining batch element as an independent draw. For a spatial tensor -- an image,
a volume, a video -- we instead want an *independent per-pixel* finite mixture
whose spatial dimensions are folded into the *event*, so that a whole tensor
receives a single log-probability equal to the sum of the per-pixel mixture
log-probabilities.

Crucially, the per-pixel assignment is drawn over the component distribution's
*event* as a unit. For an RGB image that means the color channel lives in the
component event, so a single categorical draw selects the whole RGB vector at a
pixel. This keeps colors correlated within a pixel even when the mixing weights
are soft/split -- unlike putting the channel in the mixture batch, which would
draw an independent component per channel and produce saturated per-channel
speckle wherever the weights are split.

This is the mixture analogue of writing ``dist.Normal(...).to_event(2)`` for an
``(H, W, C)`` image likelihood: the assignment categorical fires independently at
every pixel, but the resulting density is over the whole tensor.

Concretely, for a ``K``-component mixture over an ``(H, W, C)`` image with batch
size ``B`` you provide

* a mixing ``Categorical`` with ``logits``/``probs`` of shape ``(B, H, W, K)``
  (the mixture axis is the last one, as usual), and
* a component distribution whose ``batch_shape`` is ``(B, H, W, K)`` and whose
  ``event_shape`` carries the per-pixel channel structure, e.g.
  ``dist.Normal(loc, scale).to_event(1)`` with ``loc`` of shape
  ``(B, H, W, K, C)`` -- ``batch_shape == (B, H, W, K)``, ``event_shape == (C,)``;

and ``reinterpreted_batch_ndims=2`` folds ``(H, W)`` into the event, leaving
``batch_shape == (B,)`` and ``event_shape == (H, W, C)``.

Delegating to ``MixtureSameFamily`` inherits its numerically stable, linear-space
``log_prob`` (finite gradients at exactly-zero mixing weights -- empty pixels --
without any epsilon flooring). Vectorized (event-wrapped) components require the
``esennesh/numpyro`` fork, whose ``MixtureSameFamily`` unwraps ``Independent``
supports before checking for a parameter-free base.
"""

from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int

from numpyro.distributions import constraints
from numpyro.distributions.discrete import CategoricalLogits, CategoricalProbs
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.mixtures import MixtureSameFamily
from numpyro.distributions.util import (lazy_property, promote_shapes,
                                        sum_rightmost, validate_sample)
from numpyro.util import is_prng_key

from . import layers
import src.utils as utils

__all__ = ["CompleteGaussianMrf", "LocallyScaledGaussianMrf",
           "PartitionedGaussianMrf", "SecondOrderGaussianMrf",
           "SpatialMixtureSameFamily"]


def _apply_precision(v, tau, kv, kh):
    """Apply the GMRF precision to ``v``, batching over leading dimensions.

    ``v`` is ``(*batch, H, W, C)``, ``tau`` is ``(*batch, H, W)``, ``kv`` is
    ``(*batch, H-1, W)`` and ``kh`` is ``(*batch, H, W-1)``. Every index below
    counts from the RIGHT, so any number of leading batch axes ride along.
    """
    tau, kv, kh = tau[..., None], kv[..., None], kh[..., None]
    fv = kv * (v[..., 1:, :, :] - v[..., :-1, :, :])
    fh = kh * (v[..., :, 1:, :] - v[..., :, :-1, :])
    zv = jnp.zeros_like(fv[..., :1, :, :])
    zh = jnp.zeros_like(fh[..., :, :1, :])
    lap = (jnp.concatenate((zv, fv), axis=-3)
           - jnp.concatenate((fv, zv), axis=-3)
           + jnp.concatenate((zh, fh), axis=-2)
           - jnp.concatenate((fh, zh), axis=-2))
    return tau * v + lap

def _quadratic_form(v, tau, kv, kh):
    """``v' Q v``, reduced over the event dimensions only, so the batch stays."""
    return jnp.sum(v * _apply_precision(v, tau, kv, kh), axis=(-3, -2, -1))

def _logdet_scan(tau, kv, kh):
    """``log det Q`` for ONE unbatched field, by block Cholesky as a scan.

    Ordered row by row, ``Q`` is block tridiagonal with ``H`` diagonal blocks of
    size ``W x W`` and sub-diagonal blocks that are themselves diagonal, because
    pixel ``(y, x)`` couples only to ``(y+1, x)``. Block Cholesky is then a
    recursion whose every step has the same shape, so it is a ``lax.scan``:

        C_1 = chol(A_11);  C_i = chol(A_ii - B_i B_i'),  B_i = A_i,i-1 C_i-1^-T

    This is EXACT, not an approximation, and its cost is ``H * W^3`` flops. The
    sparsity pattern never enters, because a deleted edge is a zero weight and
    the pattern is the lattice whatever the render does.
    """
    diag = tau
    diag = diag.at[:-1, :].add(kv).at[1:, :].add(kv)
    diag = diag.at[:, :-1].add(kh).at[:, 1:].add(kh)
    A = jax.vmap(jnp.diag)(diag)
    A = A + jax.vmap(lambda o: jnp.diag(o, 1) + jnp.diag(o, -1))(-kh)
    A_sub = jax.vmap(jnp.diag)(-kv)

    C0 = jnp.linalg.cholesky(A[0])

    def step(C_prev, inputs):
        A_ii, A_sub_i = inputs
        B = jax.scipy.linalg.solve_triangular(C_prev, A_sub_i.T, lower=True).T
        C = jnp.linalg.cholesky(A_ii - B @ B.T)
        return C, jnp.sum(jnp.log(jnp.diag(C)))

    _, logs = jax.lax.scan(step, C0, (A[1:], A_sub))
    return 2.0 * (jnp.sum(jnp.log(jnp.diag(C0))) + jnp.sum(logs))

def _batched(fn, n):
    """Wrap ``fn`` in ``n`` nested vmaps, one per batch dimension."""
    for _ in range(n):
        fn = jax.vmap(fn)
    return fn

class PartitionedGaussianMrf(Distribution):
    """A Gaussian Markov random field whose graph comes from a region labelling.

        x ~ N(loc, Q^-1),   Q = diag(element_precision) + bond_precision * L

    ``L`` is the Laplacian of the four-neighbour lattice, restricted to the
    edges that survive: an edge whose two pixels carry different region counts
    is deleted. ``Q`` is then block diagonal, one block per region, and the
    density factorizes into one independent Gaussian per region rather than
    mixing over them.

    Shapes. One *event* is a whole ``(H, W, C)`` tensor, and every leading
    dimension batches. So ``batch_shape`` is broadcast from the leading axes of
    the layer's image, of ``element_precision`` and of ``bond_precision``, and
    ``event_shape`` is always the trailing three axes of the image.

    :param loc: a :class:`~src.distributions.layers.Layer`. Its composited image
        is the mean and its count field supplies the graph.
    :param element_precision: ``(*batch, H, W)``, the diagonal of ``Q``.
    :param bond_precision: ``(*batch,)``, one edge weight per batch element.
    :param cg_iters: conjugate-gradient steps used by :meth:`sample`.
    """

    arg_constraints = {
        "bond_precision": constraints.greater_than(0.),
        "element_precision": constraints.greater_than(0.),
        "loc": constraints.real,
    }
    pytree_data_fields = ("_layer", "bond_precision", "element_precision", "loc")
    pytree_aux_fields = ("cg_iters",)
    reparametrized_params = ["bond_precision", "element_precision", "loc"]
    support = constraints.independent(constraints.real, 3)

    def __init__(self, loc: layers.Layer,
                 element_precision: Float[Array, "*batch H W"],
                 bond_precision: Float[Array, "*batch"], *,
                 cg_iters: int = 300, validate_args: Optional[bool] = None):
        self._layer = loc
        image = loc.over_background()
        if jnp.ndim(image) < 3:
            raise ValueError("loc of PartitionedGaussianMrf needs at least "
                             f"(H, W, C) dimensions, got {jnp.shape(image)}")

        element_precision = jnp.asarray(element_precision)
        bond_precision = jnp.asarray(bond_precision)

        # The event is the trailing (H, W, C); everything to the left batches.
        event_shape = jnp.shape(image)[-3:]
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(image)[:-3],
            jnp.shape(element_precision)[:-2],
            jnp.shape(bond_precision),
        )
        self.loc = jnp.broadcast_to(image, batch_shape + event_shape)
        self.element_precision = jnp.broadcast_to(
            element_precision, batch_shape + event_shape[:2]
        )
        self.bond_precision = jnp.broadcast_to(bond_precision, batch_shape)
        self.cg_iters = cg_iters

        super().__init__(batch_shape=batch_shape, event_shape=event_shape,
                         validate_args=validate_args)

    # -- density ---------------------------------------------------------

    def logdet_precision(self, method: str = "scan"):
        """``log det Q`` over the WHOLE event, shaped like ``batch_shape``.

        The channels share one precision and are independent given it, so the
        event's log-determinant is the channel count times a single channel's.
        :attr:`precision_matrix` is the one-channel ``(H*W, H*W)`` block.

        ``"scan"`` is exact and has fixed shapes, so it never retraces when the
        render changes. ``"dense"`` is the same number by a dense factorization
        and exists to check the first.
        """
        if method == "dense":
            per_channel = jnp.linalg.slogdet(self.precision_matrix)[1]
        elif method == "scan":
            tau, (kv, kh) = self.precision_parameters
            per_channel = _batched(_logdet_scan, len(self.batch_shape))(tau, kv,
                                                                        kh)
        else:
            raise NotImplementedError(
                f"Unknown log-determinant method {method!r}; expected 'scan' or "
                "'dense'."
            )
        return self.event_shape[-1] * per_channel

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        residual = value - self.loc
        tau, (kv, kh) = self.precision_parameters
        event_size = int(np.prod(self.event_shape))
        return 0.5 * (self.logdet_precision() -
                      event_size * jnp.log(2 * jnp.pi) -
                      _quadratic_form(residual, tau, kv, kh))

    @lazy_property
    def precision_parameters(self):
        """``(tau, (kv, kh))``, all broadcast to the full batch shape."""
        mask_v, mask_h = self._layer.edge_masks
        weight = self.bond_precision[..., None, None]
        h, w = self.event_shape[:2]
        kv = jnp.broadcast_to(mask_v * weight, self.batch_shape + (h - 1, w))
        kh = jnp.broadcast_to(mask_h * weight, self.batch_shape + (h, w - 1))
        return self.element_precision, (kv, kh)

    @lazy_property
    def precision_matrix(self):
        """``Q`` as a dense ``(*batch, H*W, H*W)``. For checking only."""
        h, w = self.event_shape[:2]
        tau, (kv, kh) = self.precision_parameters
        basis = jnp.eye(h * w).reshape(h * w, h, w, 1)

        def one(t, a, b):
            return jax.vmap(
                lambda e: _apply_precision(e, t, a, b)[..., 0].ravel()
            )(basis)

        return _batched(one, len(self.batch_shape))(tau, kv, kh)

    # -- sampling --------------------------------------------------------

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        """Draw ``x ~ N(loc, Q^-1)`` with no factorization anywhere.

        ``eta ~ N(0, Q)`` is built from ``Q``'s own factor sum -- one standard
        normal per pixel and one per surviving edge -- and then ``Q x = eta`` is
        solved by conjugate gradients. A draw is linear in those normals, which
        is what makes ``loc``, ``element_precision`` and ``bond_precision``
        reparametrized.
        """
        assert is_prng_key(key)
        tau, (kv, kh) = self.precision_parameters
        h, w = self.event_shape[:2]
        lead = tuple(sample_shape) + self.batch_shape
        total = max(1, int(np.prod(lead)))
        cg_iters = self.cg_iters

        def flat(x, shape):
            return jnp.broadcast_to(x, lead + shape).reshape((total,) + shape)

        def one(subkey, loc, t, a, b):
            k0, kv_key, kh_key = jax.random.split(subkey, 3)
            channels = loc.shape[-1]
            sv = jnp.sqrt(a)[..., None] * jax.random.normal(
                kv_key, a.shape + (channels,))
            sh = jnp.sqrt(b)[..., None] * jax.random.normal(
                kh_key, b.shape + (channels,))
            eta = jnp.sqrt(t)[..., None] * jax.random.normal(
                k0, t.shape + (channels,))
            zv = jnp.zeros_like(sv[:1])
            zh = jnp.zeros_like(sh[:, :1])
            eta = (eta + jnp.concatenate((sv, zv), axis=0)
                       - jnp.concatenate((zv, sv), axis=0)
                       + jnp.concatenate((sh, zh), axis=1)
                       - jnp.concatenate((zh, sh), axis=1))
            solved = utils.cg_solve(
                lambda v: _apply_precision(v, t, a, b), eta, iters=cg_iters
            )
            return loc + solved

        draws = jax.vmap(one)(jax.random.split(key, total),
                             flat(self.loc, self.event_shape),
                             flat(tau, (h, w)),
                             flat(kv, (h - 1, w)),
                             flat(kh, (h, w - 1)))
        return draws.reshape(lead + self.event_shape)

    # -- moments ---------------------------------------------------------

    def marginal_std(self, key, draws=48):
        # Empirical, from actual draws. Honest and slow.
        xs = self.sample(key, (draws,))
        return xs.std(axis=0)

    @property
    def mean(self) -> ArrayLike:
        return jnp.broadcast_to(self.loc, self.shape())

def CompleteGaussianMrf(loc: Float[Array, "*batch H W C"],
                        element_precision: Float[Array, "*batch H W"],
                        bond_precision: Float[Array, "*batch"], *,
                        cg_iters: int=300, validate_args: Optional[bool]=None):
    return PartitionedGaussianMrf(layers.Layer.straight(loc), element_precision,
                                  bond_precision, cg_iters=cg_iters,
                                  validate_args=validate_args)

class LocallyScaledGaussianMrf(Distribution):
    """A diagonal local-precision transform of a partitioned Gaussian MRF.

    For ``D = diag(sqrt(local_precision))`` and base spatial precision ``Q``,
    the conditional precision is ``D Q D`` independently in every channel.
    Density evaluation and sampling are change-of-variable transforms of
    :class:`PartitionedGaussianMrf`, so neither operation materializes ``Q``.
    """

    arg_constraints = {
        "local_precision": constraints.greater_than(0.)
    }
    pytree_data_fields = ("base", "local_precision")
    support = constraints.independent(constraints.real, 3)

    def __init__(self, loc, element_precision, bond_precision, local_precision,
                 *, cg_iters=300, validate_args=None):
        base = PartitionedGaussianMrf(
            loc, element_precision, bond_precision, cg_iters=cg_iters,
            validate_args=validate_args
        )
        local_precision = jnp.asarray(local_precision)
        batch_shape = jax.lax.broadcast_shapes(
            base.batch_shape, local_precision.shape[:-2]
        )
        self.base = base.expand(batch_shape)
        self.local_precision = jnp.broadcast_to(
            local_precision, batch_shape + base.event_shape[:-1]
        )

        super().__init__(batch_shape=batch_shape,
                         event_shape=base.event_shape,
                         validate_args=validate_args)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        loc = self.mean

        residual = value - loc
        scaled = loc + jnp.sqrt(self.local_precision)[..., None] * residual

        log_jacobian = (
            0.5 * self.event_shape[-1]
            * jnp.log(self.local_precision).sum(axis=(-2, -1))
        )

        return self.base.log_prob(scaled) + log_jacobian

    @property
    def mean(self) -> ArrayLike:
        return self.base.mean

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        residual = self.base.sample(key, sample_shape=sample_shape) - self.mean
        return self.mean + residual / jnp.sqrt(self.local_precision)[..., None]

class SecondOrderGaussianMrf(Distribution):
    r"""A proper second-difference GMRF, factorized through a sparse operator.

    Let ``A = diag(element_precision) + bond_precision * L``, where ``L`` is
    the four-neighbour graph Laplacian after applying any supplied edge masks.
    This distribution is

    .. math::

        x \sim \mathcal N(\mu, Q^{-1}), \qquad Q = A^\mathsf{T}A.

    Consequently, its quadratic term is ``||A (x - loc)||^2``. Because ``L``
    is a second-difference operator, this penalizes curvature; the resulting
    precision contains an ``L^2`` term and has a graph-distance-two stencil.

    The factorization is also generative and normalized. Sampling draws white
    noise ``epsilon`` and solves ``A (x - loc) = epsilon``; the log density uses
    ``log det(Q) = 2 log det(A)``. Neither operation materializes ``Q``.

    Shapes follow :class:`PartitionedGaussianMrf`: leading dimensions batch,
    and the trailing ``(H, W, C)`` dimensions form one event. Channels are
    conditionally independent and share the same spatial operator.

    :param loc: Mean field, shaped ``(*batch, H, W, C)``.
    :param element_precision: Positive diagonal of ``A``, broadcastable to
        ``(*batch, H, W)``.
    :param bond_precision: Positive lattice-edge weight, broadcastable to
        ``batch_shape``.
    :param edge_masks: Optional ``(vertical, horizontal)`` masks with trailing
        shapes ``(H-1, W)`` and ``(H, W-1)``. The default retains every edge.
    :param cg_iters: Conjugate-gradient steps used by :meth:`sample`.
    """

    arg_constraints = {
        "bond_precision": constraints.greater_than(0.),
        "element_precision": constraints.greater_than(0.),
        "loc": constraints.real,
    }
    pytree_data_fields = ("_mask_h", "_mask_v", "bond_precision",
                          "element_precision", "loc")
    pytree_aux_fields = ("cg_iters",)
    reparametrized_params = [
        "bond_precision", "element_precision", "loc"
    ]
    support = constraints.independent(constraints.real, 3)

    def __init__(
        self,
        loc: Float[Array, "*batch H W C"],
        element_precision: Float[Array, "*batch H W"],
        bond_precision: Float[Array, "*batch"],
        *,
        cg_iters: int = 300,
        edge_masks: tuple[ArrayLike, ArrayLike] | None = None,
        validate_args: bool | None = None,
    ):
        bond_precision = jnp.asarray(bond_precision)
        element_precision = jnp.asarray(element_precision)
        loc = jnp.asarray(loc)
        if jnp.ndim(loc) < 3:
            raise ValueError(
                "loc of SecondOrderGaussianMrf needs at least (H, W, C) "
                f"dimensions, got {jnp.shape(loc)}"
            )

        event_shape = jnp.shape(loc)[-3:]
        height, width = event_shape[:2]
        if edge_masks is None:
            mask_h = jnp.ones((height, width - 1), dtype=loc.dtype)
            mask_v = jnp.ones((height - 1, width), dtype=loc.dtype)
        else:
            mask_v, mask_h = (jnp.asarray(mask) for mask in edge_masks)
        if jnp.ndim(mask_h) < 2 or jnp.shape(mask_h)[-2:] != (
            height, width - 1
        ):
            raise ValueError(
                "horizontal edge mask needs trailing shape "
                f"{(height, width - 1)}, got {jnp.shape(mask_h)}"
            )
        if jnp.ndim(mask_v) < 2 or jnp.shape(mask_v)[-2:] != (
            height - 1, width
        ):
            raise ValueError(
                "vertical edge mask needs trailing shape "
                f"{(height - 1, width)}, got {jnp.shape(mask_v)}"
            )

        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(bond_precision),
            jnp.shape(element_precision)[:-2],
            jnp.shape(loc)[:-3],
            jnp.shape(mask_h)[:-2],
            jnp.shape(mask_v)[:-2],
        )
        self._mask_h = jnp.broadcast_to(
            mask_h, batch_shape + (height, width - 1)
        )
        self._mask_v = jnp.broadcast_to(
            mask_v, batch_shape + (height - 1, width)
        )
        self.bond_precision = jnp.broadcast_to(bond_precision, batch_shape)
        self.cg_iters = cg_iters
        self.element_precision = jnp.broadcast_to(
            element_precision, batch_shape + (height, width)
        )
        self.loc = jnp.broadcast_to(loc, batch_shape + event_shape)

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @classmethod
    def from_coverage(
        cls,
        coverage: ArrayLike,
        element_precision: ArrayLike,
        bond_precision: ArrayLike,
        *,
        cg_iters: int = 300,
        channels: int = 1,
        loc: ArrayLike | None = None,
        threshold: float = layers.POTENTIAL_EDGE_ALPHA,
        validate_args: bool | None = None,
    ) -> "SecondOrderGaussianMrf":
        """Build a support-restricted field from a raw alpha bitmap.

        ``coverage`` has shape ``(*batch, H, W)``. Integer bitmaps, including
        ordinary uint8 alpha channels, are normalized by their dtype maximum;
        floating-point inputs are interpreted on ``[0, 1]``. A lattice bond is
        retained only when both endpoint pixels satisfy
        ``coverage >= threshold``. Uncovered pixels remain proper independent
        Gaussian variables through the positive element term, but have no
        smoothing bonds and cannot transmit texture across the glyph boundary.

        When ``loc`` is omitted, the mean is zero with ``channels`` channels.
        Supplying ``loc`` overrides that default and determines the event's
        channel count.
        """
        coverage = jnp.asarray(coverage)
        if jnp.ndim(coverage) < 2:
            raise ValueError(
                "coverage needs at least (H, W) dimensions, "
                f"got {jnp.shape(coverage)}"
            )
        if jnp.issubdtype(coverage.dtype, jnp.integer):
            coverage = coverage.astype(jnp.float32) / jnp.iinfo(
                coverage.dtype
            ).max
        if loc is None:
            if channels < 1:
                raise ValueError(f"channels needs to be positive, got {channels}")
            loc = jnp.zeros(
                jnp.shape(coverage) + (channels,),
                dtype=jnp.result_type(coverage.dtype, jnp.float32),
            )
        support = coverage >= threshold
        mask_h = support[..., :, :-1] & support[..., :, 1:]
        mask_v = support[..., :-1, :] & support[..., 1:, :]
        return cls(
            loc,
            element_precision,
            bond_precision,
            cg_iters=cg_iters,
            edge_masks=(mask_v, mask_h),
            validate_args=validate_args,
        )

    def logdet_precision(self, method: str = "scan") -> Array:
        """Return ``log det(A.T A)`` over every channel in the event."""
        if method == "dense":
            per_channel = 2.0 * jnp.linalg.slogdet(self.operator_matrix)[1]
        elif method == "scan":
            element, (vertical, horizontal) = self.operator_parameters
            per_channel = 2.0 * _batched(
                _logdet_scan, len(self.batch_shape)
            )(element, vertical, horizontal)
        else:
            raise NotImplementedError(
                f"Unknown log-determinant method {method!r}; expected 'scan' "
                "or 'dense'."
            )
        return self.event_shape[-1] * per_channel

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        element, (vertical, horizontal) = self.operator_parameters
        residual = value - self.loc
        transformed = _apply_precision(
            residual, element, vertical, horizontal
        )
        quadratic = jnp.sum(transformed**2, axis=(-3, -2, -1))
        event_size = int(np.prod(self.event_shape))
        return 0.5 * (
            self.logdet_precision()
            - event_size * jnp.log(2.0 * jnp.pi)
            - quadratic
        )

    @property
    def mean(self) -> ArrayLike:
        return jnp.broadcast_to(self.loc, self.shape())

    @lazy_property
    def operator_matrix(self) -> Array:
        """Dense one-channel ``A`` matrix, for diagnostics and tests only."""
        element, (vertical, horizontal) = self.operator_parameters
        height, width = self.event_shape[:2]
        basis = jnp.eye(height * width).reshape(
            height * width, height, width, 1
        )

        def one(diagonal, horizontal_bonds, vertical_bonds):
            return jax.vmap(
                lambda vector: _apply_precision(
                    vector,
                    diagonal,
                    vertical_bonds,
                    horizontal_bonds,
                )[..., 0].ravel()
            )(basis)

        return _batched(one, len(self.batch_shape))(
            element, horizontal, vertical
        )

    @lazy_property
    def operator_parameters(self) -> tuple[Array, tuple[Array, Array]]:
        """Return ``(element, (vertical, horizontal))`` parameters of ``A``."""
        weight = self.bond_precision[..., None, None]
        horizontal = self._mask_h * weight
        vertical = self._mask_v * weight
        return self.element_precision, (vertical, horizontal)

    @lazy_property
    def precision_matrix(self) -> Array:
        """Dense one-channel ``Q = A.T A``, for diagnostics and tests only."""
        operator_transpose = jnp.swapaxes(self.operator_matrix, -1, -2)
        return operator_transpose @ self.operator_matrix

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        """Draw white noise and solve ``A (x - loc) = epsilon``."""
        assert is_prng_key(key)
        element, (vertical, horizontal) = self.operator_parameters
        height, width = self.event_shape[:2]
        lead = tuple(sample_shape) + self.batch_shape
        total = max(1, int(np.prod(lead)))

        def flat(value, shape):
            return jnp.broadcast_to(value, lead + shape).reshape(
                (total,) + shape
            )

        def one(subkey, loc, diagonal, horizontal_bonds, vertical_bonds):
            noise = jax.random.normal(subkey, loc.shape)
            residual = utils.cg_solve(
                lambda value: _apply_precision(
                    value,
                    diagonal,
                    vertical_bonds,
                    horizontal_bonds,
                ),
                noise,
                iters=self.cg_iters,
            )
            return loc + residual

        draws = jax.vmap(one)(
            jax.random.split(key, total),
            flat(self.loc, self.event_shape),
            flat(element, (height, width)),
            flat(horizontal, (height, width - 1)),
            flat(vertical, (height - 1, width)),
        )
        return draws.reshape(lead + self.event_shape)

class SpatialMixtureSameFamily(Distribution):
    """A per-pixel finite mixture over a spatial tensor.

    Wraps a :class:`~numpyro.distributions.MixtureSameFamily` (which performs an
    independent mixture at every element of its ``batch_shape``) and reinterprets
    the trailing ``reinterpreted_batch_ndims`` batch dimensions -- the spatial
    axes -- as event dimensions. ``log_prob`` therefore sums the per-pixel mixture
    log-probabilities over those axes. Each per-pixel assignment selects an entire
    component *event* (e.g. the RGB vector), so channels stay correlated within a
    pixel.

    :param mixing_distribution: A :class:`~numpyro.distributions.Categorical`
        giving the per-pixel component weights. Its last dimension is the mixture
        axis and its size is ``mixture_size``.
    :param component_distribution: A single vectorized
        :class:`~numpyro.distributions.Distribution` whose last batch dimension
        equals ``mixture_size``. The leading batch dimensions carry the spatial
        structure; any ``event_shape`` (e.g. ``(C,)``) is the per-pixel event
        selected as a unit by the assignment.
    :param reinterpreted_batch_ndims: The number of trailing (spatial) batch
        dimensions of the underlying per-pixel mixture to fold into the event.
        Defaults to *all* of them (mirroring ``to_event()`` with no argument).
    """

    pytree_data_fields = ("_mixture",)
    pytree_aux_fields = ("reinterpreted_batch_ndims",)

    def __init__(
        self,
        mixing_distribution: Union[CategoricalProbs, CategoricalLogits],
        component_distribution: Distribution,
        *,
        reinterpreted_batch_ndims: Optional[int] = None,
        validate_args: Optional[bool] = None,
    ):
        # The inner mixture does the per-pixel work; its ``batch_shape`` holds the
        # leading batch dims followed by the spatial dims we fold in. It validates
        # its own arguments (mixing family, mixture-size match, component support)
        # regardless of ``validate_args``; sample validation is handled once, by
        # this outer distribution, against the reinterpreted support.
        mixture = MixtureSameFamily(
            mixing_distribution, component_distribution, validate_args=False
        )
        self._mixture = mixture

        n_batch = len(mixture.batch_shape)
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = n_batch
        if not 0 <= reinterpreted_batch_ndims <= n_batch:
            raise ValueError(
                "reinterpreted_batch_ndims must be in [0, "
                f"{n_batch}] (the number of batch dimensions of the underlying "
                f"per-pixel mixture), but got {reinterpreted_batch_ndims}."
            )
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        split = n_batch - reinterpreted_batch_ndims
        batch_shape = mixture.batch_shape[:split]
        event_shape = mixture.batch_shape[split:] + mixture.event_shape
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    # -- structure --------------------------------------------------------

    @property
    def mixture(self) -> MixtureSameFamily:
        """The underlying per-pixel :class:`MixtureSameFamily`."""
        return self._mixture

    @property
    def mixing_distribution(self) -> Union[CategoricalProbs, CategoricalLogits]:
        return self._mixture.mixing_distribution

    @property
    def component_distribution(self) -> Distribution:
        return self._mixture.component_distribution

    @property
    def mixture_size(self) -> int:
        return self._mixture.mixture_size

    @constraints.dependent_property
    def support(self) -> constraints.Constraint:
        return constraints.independent(
            self._mixture.support, self.reinterpreted_batch_ndims
        )

    @property
    def is_discrete(self) -> bool:
        return self._mixture.is_discrete

    @property
    def has_rsample(self) -> bool:
        # Inherited from the mixture, which samples the (discrete) assignment.
        return False

    # -- sampling / density ----------------------------------------------

    def sample(self, key: jax.Array, sample_shape: tuple = ()):
        # The inner mixture already returns samples of shape
        # ``sample_shape + batch_shape + event_shape``; folding batch dims into
        # the event does not change the sampled array.
        return self._mixture.sample(key, sample_shape=sample_shape)

    def sample_with_intermediates(self, key: jax.Array, sample_shape: tuple = ()):
        """Sample, additionally returning the per-pixel component indices.

        The returned indices have the underlying mixture's shape
        ``sample_shape + batch_shape + spatial_shape`` (one assignment per pixel),
        *not* this distribution's ``batch_shape``.
        """
        return self._mixture.sample_with_intermediates(
            key, sample_shape=sample_shape
        )

    @validate_sample
    def log_prob(self, value):
        # Per-pixel mixture log-probs, then sum over the folded-in spatial dims.
        elementwise = self._mixture.log_prob(value)
        return sum_rightmost(elementwise, self.reinterpreted_batch_ndims)

    # -- moments ----------------------------------------------------------

    @property
    def mean(self):
        # The mixture mean is elementwise, so its shape already matches
        # ``batch_shape + event_shape``; no reduction is needed.
        return self._mixture.mean

    @property
    def variance(self):
        return self._mixture.variance
