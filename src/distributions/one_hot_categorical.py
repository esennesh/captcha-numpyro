"""OneHotCategorical distribution ported from Pyro (PyTorch) to numpyro (JAX).

Pyro reference:
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/torch.py
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.nn import one_hot, softmax

import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key, validate_sample


class OneHotCategorical(Distribution):
    """One-hot categorical distribution over ``K`` classes.

    Samples are K-dimensional one-hot vectors, so ``event_shape == (K,)``. The
    interface mirrors Pyro's ``OneHotCategorical`` from
    ``pyro.distributions.torch``; internally it wraps a numpyro ``Categorical``
    and delegates sampling, log-prob, and entropy to it.
    """

    has_enumerate_support = True
    pytree_data_fields = ("_categorical",)

    def __init__(
        self,
        probs=None,
        logits=None,
        *,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Exactly one of `probs` or `logits` must be provided."
            )
        self._categorical = dist.Categorical(
            probs=probs, logits=logits, validate_args=validate_args,
        )
        K = (
            self._categorical.probs.shape[-1]
            if probs is not None
            else jnp.shape(logits)[-1]
        )
        super().__init__(
            batch_shape=self._categorical.batch_shape,
            event_shape=(K,),
            validate_args=validate_args,
        )

    @property
    def probs(self):
        raise NotImplementedError

    @property
    def logits(self):
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return self.event_shape[-1]

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        # A one-hot vector lives on the simplex (specifically at a vertex).
        # numpyro has no ``constraints.one_hot``; ``simplex`` is the closest
        # standard constraint and validates everything strict one-hot samples
        # satisfy.
        return constraints.simplex

    def sample(self, key: jax.Array, sample_shape: tuple = ()):
        assert is_prng_key(key)
        indices = self._categorical.sample(key, sample_shape)
        return one_hot(
            indices,
            num_classes=self.num_classes,
            dtype=jnp.result_type(self.probs),
        )

    @validate_sample
    def log_prob(self, value):
        # Convert one-hot back to an integer index and delegate to Categorical,
        # matching Pyro's implementation. For relaxed / continuous values use
        # ``RelaxedOneHotCategorical`` instead.
        indices = jnp.argmax(value, axis=-1)
        return self._categorical.log_prob(indices)

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def enumerate_support(self, expand: bool = True):
        K = self.num_classes
        values = jnp.eye(K).reshape((K,) + (1,) * len(self.batch_shape) + (K,))
        if expand:
            values = jnp.broadcast_to(values, (K,) + self.batch_shape + (K,))
        return values

    def entropy(self):
        return self._categorical.entropy()

class OneHotCategoricalLogits(Distribution):
    """One-hot categorical distribution over ``K`` classes.

    Samples are K-dimensional one-hot vectors, so ``event_shape == (K,)``. The
    interface mirrors Pyro's ``OneHotCategorical`` from
    ``pyro.distributions.torch``; internally it wraps a numpyro ``Categorical``
    and delegates sampling, log-prob, and entropy to it.
    """

    arg_constraints = {
        "logits": constraints.real_vector,
    }
    has_enumerate_support = True
    pytree_data_fields = ("_categorical",)

    def __init__(self, logits, *, validate_args: Optional[bool] = None):
        self._categorical = dist.Categorical(logits=logits,
                                             validate_args=validate_args)
        K = jnp.shape(logits)[-1]
        super().__init__(
            batch_shape=self._categorical.batch_shape,
            event_shape=(K,),
            validate_args=validate_args,
        )

    @property
    def probs(self):
        return jax.nn.softmax(self.logits, axis=-1)

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def num_classes(self) -> int:
        return self.event_shape[-1]

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        # A one-hot vector lives on the simplex (specifically at a vertex).
        # numpyro has no ``constraints.one_hot``; ``simplex`` is the closest
        # standard constraint and validates everything strict one-hot samples
        # satisfy.
        return constraints.simplex

    def sample(self, key: jax.Array, sample_shape: tuple = ()):
        assert is_prng_key(key)
        indices = self._categorical.sample(key, sample_shape)
        return one_hot(
            indices,
            num_classes=self.num_classes,
            dtype=jnp.result_type(self.probs),
        )

    @validate_sample
    def log_prob(self, value):
        # Convert one-hot back to an integer index and delegate to Categorical,
        # matching Pyro's implementation. For relaxed / continuous values use
        # ``RelaxedOneHotCategorical`` instead.
        indices = jnp.argmax(value, axis=-1)
        return self._categorical.log_prob(indices)

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def enumerate_support(self, expand: bool = True):
        K = self.num_classes
        values = jnp.eye(K).reshape((K,) + (1,) * len(self.batch_shape) + (K,))
        if expand:
            values = jnp.broadcast_to(values, (K,) + self.batch_shape + (K,))
        return values

    def entropy(self):
        return self._categorical.entropy()

class OneHotCategoricalProbs(Distribution):
    """One-hot categorical distribution over ``K`` classes.

    Samples are K-dimensional one-hot vectors, so ``event_shape == (K,)``. The
    interface mirrors Pyro's ``OneHotCategorical`` from
    ``pyro.distributions.torch``; internally it wraps a numpyro ``Categorical``
    and delegates sampling, log-prob, and entropy to it.
    """

    arg_constraints = {
        "probs":  constraints.simplex,
    }
    has_enumerate_support = True
    pytree_data_fields = ("_categorical",)

    def __init__(self, probs, *, validate_args: Optional[bool] = None):
        self._categorical = dist.Categorical(
            probs=probs, validate_args=validate_args,
        )
        K = self._categorical.probs.shape[-1]
        super().__init__(
            batch_shape=self._categorical.batch_shape,
            event_shape=(K,),
            validate_args=validate_args,
        )

    @property
    def probs(self):
        return self._categorical.probs

    @property
    def logits(self):
        return jnp.log(self.probs)

    @property
    def num_classes(self) -> int:
        return self.event_shape[-1]

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        # A one-hot vector lives on the simplex (specifically at a vertex).
        # numpyro has no ``constraints.one_hot``; ``simplex`` is the closest
        # standard constraint and validates everything strict one-hot samples
        # satisfy.
        return constraints.simplex

    def sample(self, key: jax.Array, sample_shape: tuple = ()):
        assert is_prng_key(key)
        indices = self._categorical.sample(key, sample_shape)
        return one_hot(
            indices,
            num_classes=self.num_classes,
            dtype=jnp.result_type(self.probs),
        )

    @validate_sample
    def log_prob(self, value):
        # Convert one-hot back to an integer index and delegate to Categorical,
        # matching Pyro's implementation. For relaxed / continuous values use
        # ``RelaxedOneHotCategorical`` instead.
        indices = jnp.argmax(value, axis=-1)
        return self._categorical.log_prob(indices)

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def enumerate_support(self, expand: bool = True):
        K = self.num_classes
        values = jnp.eye(K).reshape((K,) + (1,) * len(self.batch_shape) + (K,))
        if expand:
            values = jnp.broadcast_to(values, (K,) + self.batch_shape + (K,))
        return values

    def entropy(self):
        return self._categorical.entropy()
