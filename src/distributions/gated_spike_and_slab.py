"""GatedSpikeAndSlab: a conditional spike-and-slab gated by an integer-valued
variable.

When the gate equals 0, the sample is drawn from the spike distribution; when
the gate is positive, the sample is drawn from the slab distribution. The
gate value is supplied as data (typically the realised sample of a separate,
outer integer-valued distribution like ``Poisson`` or ``Bernoulli``), so this
class represents the *conditional* :math:`p(\\text{value} \\mid \\text{gate})`.

The full marginal :math:`p(\\text{value}) = \\sum_g p(\\text{gate}{=}g) \\,
p(\\text{value} \\mid \\text{gate}{=}g)` is a proper mixture and could in
principle be expressed as ``numpyro.distributions.MixtureGeneral``. Exposing
the gate as a separate sample site (and using this class as the conditional
mark) is usually preferable in a Bayesian generative model: it keeps an
explicit posterior over the gate and avoids spurious log-prob contributions
at gated-off sites that an unconditional mark would produce.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key, validate_sample


class GatedSpikeAndSlab(Distribution):
    """Conditional mixture: spike if ``gate == 0``, slab if ``gate > 0``.

    Args:
        gate: array of nonnegative integer gate values. Broadcast against
            ``spike.batch_shape`` and ``slab.batch_shape`` to determine this
            distribution's batch_shape.
        spike: distribution fired when ``gate == 0`` (typically a ``Delta``).
        slab: distribution fired when ``gate > 0``. Must share ``event_shape``
            with ``spike``.

    Notes:
        - The class structurally mirrors ``MixtureGeneral`` (cf.
          ``numpyro.distributions.mixtures``): both implement ``sample`` and
          ``log_prob`` by combining per-component samples / log-probs with a
          mixing variable. Here the mixing variable is *observed* (the gate),
          which collapses the mixture to a one-hot selector and lets
          ``log_prob`` skip the log-sum-exp.
        - ``spike`` and ``slab`` are stored as pytree data fields, so JAX
          transforms recurse into them; ``gate`` is a leaf array.
    """

    arg_constraints = {}
    pytree_data_fields = ("gate", "spike", "slab")

    def __init__(
        self,
        gate,
        spike: Distribution,
        slab: Distribution,
        *,
        validate_args: Optional[bool] = None,
    ):
        if spike.event_shape != slab.event_shape:
            raise ValueError(
                "spike and slab must share event_shape, "
                f"got {spike.event_shape} vs {slab.event_shape}"
            )

        self.gate = jnp.asarray(gate)
        self.spike = spike
        self.slab = slab

        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.gate),
            spike.batch_shape,
            slab.batch_shape,
        )
        event_shape = spike.event_shape

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        # The full support is the union of spike and slab supports; without a
        # standard union-constraint we defer to the slab's support, which is
        # typically the richer of the two (the spike is usually a Delta).
        return self.slab.support

    def sample(self, key: jax.Array, sample_shape: tuple = ()):
        assert is_prng_key(key)
        key_spike, key_slab = jax.random.split(key)

        # Expand each component to this distribution's batch_shape so the per-
        # cell draws are independent across the broadcast positions.
        spike_sample = self.spike.expand(self.batch_shape).sample(
            key_spike, sample_shape,
        )
        slab_sample = self.slab.expand(self.batch_shape).sample(
            key_slab, sample_shape,
        )

        # Broadcast gate across sample_shape + batch_shape, then add singleton
        # axes for each event dimension so the where-select is event-wise.
        gate = jnp.broadcast_to(self.gate, sample_shape + self.batch_shape)
        gate_e = gate.reshape(gate.shape + (1,) * len(self.event_shape))

        return jnp.where(gate_e > 0, slab_sample, spike_sample)

    @validate_sample
    def log_prob(self, value):
        spike_lp = self.spike.expand(self.batch_shape).log_prob(value)
        slab_lp = self.slab.expand(self.batch_shape).log_prob(value)
        gate = jnp.broadcast_to(self.gate, jnp.shape(slab_lp))
        return jnp.where(gate > 0, slab_lp, spike_lp)

    @property
    def mean(self):
        gate = jnp.broadcast_to(self.gate, self.batch_shape)
        gate_e = gate.reshape(gate.shape + (1,) * len(self.event_shape))
        return jnp.where(
            gate_e > 0,
            self.slab.expand(self.batch_shape).mean,
            self.spike.expand(self.batch_shape).mean,
        )
