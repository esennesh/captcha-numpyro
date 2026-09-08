r"""Optimize relaxed counts in linear space, under a nonnegativity projection.

``AutoMAPProposal`` reparameterizes every site through
``biject_to(site["fn"].support)``. A DSGD-relaxed count site has support
``(0, inf)``, so that bijection is a logarithm and the MAP phase walks
``log a`` rather than ``a``. Adam moves each coordinate by about one step size
per step, so from the prior rate ``4/230400`` a single stamp needs

.. math::

    \frac{\log 1 - \log(1.736\times10^{-5})}{\rho} = \frac{10.96}{\rho}

steps before it renders anything at all: 219 at ``rho = 0.05``. Meanwhile the
paper logit reaches its correct value in about 34, the ink colour sooner, and
the texture sooner still. Whichever block explains the ink first keeps it, and
in log space the count field is guaranteed to arrive last. The first pod run of
``notebooks/online_map_proposal`` shows the consequence: the paper field drew
both glyphs, and the count field was left stamping the confetti.

Sections 3 and 4 of ``notebooks/partitioned_gmrf`` do not have this problem.
They hold the counts as a plain nonnegative parameter, start it at zero, and
step it at ``1e-3`` in *linear* space, so a coefficient reaches a useful
amplitude in about 40 steps and the counts arrive first. This module ports that
geometry to the guide:

* :class:`LinearCountMAPProposal` replaces the count sites' exponential
  reparameterization with the identity, mapping the initial locations through
  the old transform so the starting point is unchanged.
* :func:`count_block_adam` gives the count block its own step size and appends
  ``optax.keep_params_nonnegative``, which turns the update into a projected
  gradient step. ``numpyro.optim.optax_to_numpyro`` forwards ``params`` to
  ``update``, which is what that projection needs.

A count pinned at zero is a genuine part of the answer rather than a numerical
accident: the relaxed Poisson log PMF at ``a = 0`` is ``-rate``, and its
derivative is ``log(rate) - digamma(1) = -10.4``, so the prior holds an
unsupported site exactly at zero and the likelihood has to outbid it to lift
one off. That is the same mechanism the notebook's support threshold exploits,
without the hard pruning step.
"""

from typing import Sequence

import optax
from numpyro.contrib.map_proposal import AutoMAPProposal
from numpyro.distributions.transforms import IdentityTransform

__all__ = ["LinearCountMAPProposal", "count_block_adam"]


class LinearCountMAPProposal(AutoMAPProposal):
    """An :class:`AutoMAPProposal` whose count sites are not log-transformed.

    The MAP phase then needs an optimizer that keeps those coordinates
    nonnegative; :func:`count_block_adam` builds one. Pass it as
    ``map_optimizer``. Nothing else changes: ``_initial_proposal_params`` and
    ``_get_proposal`` read the count value through the same
    ``self._transforms`` entry, so an identity there hands them the linear
    count they already expect.
    """

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        for name in self._count_relaxations:
            # `_init_locs` comes back unconstrained, so push it through the
            # logarithm's inverse before discarding the transform. The starting
            # point is therefore exactly the one `init_loc_fn` asked for.
            self._init_locs[name] = self._transforms[name](self._init_locs[name])
            self._transforms[name] = IdentityTransform()

    @property
    def linear_count_sites(self) -> tuple[str, ...]:
        """Names of the sites this guide optimizes in linear space."""
        return tuple(self._count_relaxations)


def count_block_adam(
    count_step_size: float = 1e-3,
    step_size: float = 1e-2,
    count_sites: Sequence[str] = ("a",),
) -> optax.GradientTransformation:
    """Adam with a separate, projected step size for the count block.

    ``count_sites`` names the sites :class:`LinearCountMAPProposal` keeps in
    linear space. They are matched by site name because the optimizer is built
    before the guide discovers its prototype; the default matches the count
    site of :func:`src.model.model.poisson_convsc_model`.

    The defaults mirror sections 3 and 4 of ``notebooks/partitioned_gmrf``,
    which reach a useful count amplitude in about 40 steps rather than the 219
    the logarithmic parameterization needs.
    """
    if count_step_size <= 0 or step_size <= 0:
        raise ValueError("step sizes need to be positive")
    names = frozenset(count_sites)

    def labels(parameters):
        return {
            name: ("counts" if name in names else "other") for name in parameters
        }

    return optax.multi_transform(
        {
            # `keep_params_nonnegative` clips the update so that `p + u >= 0`,
            # and optax requires it last in the chain.
            "counts": optax.chain(
                optax.adam(count_step_size), optax.keep_params_nonnegative()
            ),
            "other": optax.adam(step_size),
        },
        labels,
    )
