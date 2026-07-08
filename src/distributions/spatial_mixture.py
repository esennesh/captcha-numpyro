"""Spatial specialization of :class:`numpyro.distributions.MixtureSameFamily`.

``MixtureSameFamily`` mixes over the last *batch* dimension and treats each
element as an independent draw. For a spatial tensor -- an image, a volume, a
video -- we instead want an *independent per-element* (per-pixel/voxel) finite
mixture whose spatial and channel dimensions are folded into the *event*, so
that a whole tensor receives a single log-probability equal to the sum of the
per-element mixture log-probabilities.

This is the mixture analogue of writing ``dist.Normal(...).to_event(3)`` for a
``(C, H, W)`` image likelihood: the assignment categorical fires independently
at every location, but the resulting density is over the whole tensor.

Concretely, for a ``K``-component mixture over a ``(C, H, W)`` image with batch
size ``B`` you provide

* a mixing ``Categorical`` with ``logits``/``probs`` of shape ``(B, C, H, W, K)``
  (the mixture axis is the last one, as usual), and
* a component distribution whose ``batch_shape`` is ``(B, C, H, W, K)`` (e.g.
  ``dist.Normal(loc, scale)`` with per-element, per-component parameters);

and ``reinterpreted_batch_ndims=3`` folds ``(C, H, W)`` into the event, leaving
``batch_shape == (B,)`` and ``event_shape == (C, H, W)``.
"""

from typing import Optional, Union

import jax

from numpyro.distributions import constraints
from numpyro.distributions.discrete import CategoricalLogits, CategoricalProbs
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.mixtures import MixtureSameFamily
from numpyro.distributions.util import sum_rightmost, validate_sample

__all__ = ["SpatialMixtureSameFamily"]


class SpatialMixtureSameFamily(Distribution):
    """A per-element finite mixture over a spatial tensor.

    Wraps a :class:`~numpyro.distributions.MixtureSameFamily` (which performs an
    independent mixture at every element of its ``batch_shape``) and reinterprets
    the trailing ``reinterpreted_batch_ndims`` batch dimensions -- the spatial and
    channel axes -- as event dimensions. ``log_prob`` therefore sums the
    per-element mixture log-probabilities over those axes.

    :param mixing_distribution: A :class:`~numpyro.distributions.Categorical`
        giving the per-element component weights. Its last dimension is the
        mixture axis and its size is ``mixture_size``.
    :param component_distribution: A single vectorized
        :class:`~numpyro.distributions.Distribution` whose last batch dimension
        equals ``mixture_size``. The leading batch dimensions carry the spatial
        and channel structure.
    :param reinterpreted_batch_ndims: The number of trailing (spatial/channel)
        batch dimensions of the underlying per-element mixture to fold into the
        event. Defaults to *all* of them, i.e. every remaining batch dimension
        becomes part of the event (mirroring ``to_event()`` with no argument).
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
        # The inner mixture does the per-element work; its ``batch_shape`` holds
        # the leading batch dims followed by the spatial/channel dims we fold in.
        # It validates its own arguments (mixing family, mixture-size match)
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
                f"per-element mixture), but got {reinterpreted_batch_ndims}."
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
        """The underlying per-element :class:`MixtureSameFamily`."""
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
        # Inherited from the mixture, which marginalizes the (discrete)
        # assignment and so is not reparameterizable.
        return False

    # -- sampling / density ----------------------------------------------

    def sample(self, key: jax.Array, sample_shape: tuple = ()):
        # The inner mixture already returns samples of shape
        # ``sample_shape + batch_shape + event_shape``; folding batch dims into
        # the event does not change the sampled array.
        return self._mixture.sample(key, sample_shape=sample_shape)

    def sample_with_intermediates(self, key: jax.Array, sample_shape: tuple = ()):
        """Sample, additionally returning the per-element component indices.

        The returned indices have the underlying mixture's shape
        ``sample_shape + batch_shape + spatial_shape`` (one assignment per
        element), *not* this distribution's ``batch_shape``.
        """
        return self._mixture.sample_with_intermediates(
            key, sample_shape=sample_shape
        )

    @validate_sample
    def log_prob(self, value):
        # Per-element mixture log-probs, then sum over the folded-in dims.
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
