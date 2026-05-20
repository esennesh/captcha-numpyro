import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jaxlie._base import MatrixLieGroup
from jaxlie.utils import register_lie_group
from jaxtyping import Array, Float
import flax.nnx as nnx
from typing import Optional, Tuple


@register_lie_group(
    matrix_dim=2,
    parameters_dim=1,
    tangent_dim=1,
    space_dim=2,
)
@jdc.pytree_dataclass
class RPlus(MatrixLieGroup):
    """Positive reals under multiplication, acting on R^2 by isotropic scaling.

    Internal parameterization is `(log_scale,)`. Tangent parameterization is
    `(log_scale_rate,)`, which coincides with the internal parameterization since
    log-coordinates are the natural flat coordinates on this group.
    """

    log_scale: jax.Array
    """Log of the scale factor. Shape: `(*, 1)`."""

    @classmethod
    def identity(cls, batch_axes: jdc.Static[Tuple[int, ...]] = ()) -> "RPlus":
        return cls(log_scale=jnp.zeros((*batch_axes, 1)))

    @classmethod
    def from_matrix(cls, matrix: jax.Array) -> "RPlus":
        assert matrix.shape[-2:] == (2, 2)
        return cls(log_scale=jnp.log(matrix[..., 0:1, 0]))

    def as_matrix(self) -> jax.Array:
        s = jnp.exp(self.log_scale[..., 0])
        return jnp.einsum("...,ij->...ij", s, jnp.eye(2))

    def parameters(self) -> jax.Array:
        return self.log_scale

    def apply(self, target: jax.Array) -> jax.Array:
        assert target.shape[-1:] == (2,)
        return jnp.exp(self.log_scale) * target

    def multiply(self, other: "RPlus") -> "RPlus":
        return RPlus(log_scale=self.log_scale + other.log_scale)

    def inverse(self) -> "RPlus":
        return RPlus(log_scale=-self.log_scale)

    def normalize(self) -> "RPlus":
        return self

    @classmethod
    def exp(cls, tangent: jax.Array) -> "RPlus":
        assert tangent.shape[-1] == 1
        return cls(log_scale=tangent)

    def log(self) -> jax.Array:
        return self.log_scale

    def adjoint(self) -> jax.Array:
        return jnp.ones((*self.get_batch_axes(), 1, 1))

    def jlog(self) -> jax.Array:
        # R+ is abelian; the left Jacobian of the log map is the 1×1 identity.
        return jnp.ones((*self.get_batch_axes(), 1, 1))

    @classmethod
    def sample_uniform(
        cls,
        key: jax.Array,
        batch_axes: jdc.Static[Tuple[int, ...]] = (),
    ) -> "RPlus":
        # R+ has non-compact Haar measure, so we adopt the convention of sampling
        # log_scale ~ Uniform(-π, π), matching the angular range of SO(2) when
        # this group is used as the scale factor in the direct product SO2RPlus.
        log_scale = jax.random.uniform(
            key=key,
            shape=(*batch_axes, 1),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        return cls(log_scale=log_scale)


@register_lie_group(
    matrix_dim=2,
    parameters_dim=3,
    tangent_dim=2,
    space_dim=2,
)
@jdc.pytree_dataclass
class SO2RPlus(MatrixLieGroup):
    """Direct product of SO(2) and R+, acting on R^2 by rotation and isotropic scaling.

    The two factors commute (isotropic scaling is central in GL(2)), so this is a
    direct product, not a proper semi-direct product. The group law is simply
    (R1, s1)(R2, s2) = (R1 R2, s1 s2) with no coupling term.

    Internal parameterization is `(cos, sin, log_scale)`. Tangent parameterization
    is `(omega, log_scale_rate)`.
    """

    unit_complex: jax.Array
    """Rotation as a unit complex number `(cos, sin)`. Shape: `(*, 2)`."""

    log_scale: jax.Array
    """Log of the scale factor. Shape: `(*, 1)`."""

    @classmethod
    def identity(cls, batch_axes: jdc.Static[Tuple[int, ...]] = ()) -> "SO2RPlus":
        return cls(
            unit_complex=jnp.stack(
                [jnp.ones(batch_axes), jnp.zeros(batch_axes)], axis=-1
            ),
            log_scale=jnp.zeros((*batch_axes, 1)),
        )

    @classmethod
    def from_matrix(cls, matrix: jax.Array) -> "SO2RPlus":
        assert matrix.shape[-2:] == (2, 2)
        scale = jnp.linalg.norm(matrix[..., :, 0], axis=-1, keepdims=True)
        unit_complex = matrix[..., :, 0] / scale
        return cls(unit_complex=unit_complex, log_scale=jnp.log(scale))

    def as_matrix(self) -> jax.Array:
        c = self.unit_complex[..., 0]
        s = self.unit_complex[..., 1]
        scale = jnp.exp(self.log_scale[..., 0])
        return jnp.einsum(
            "...,...ij->...ij",
            scale,
            jnp.stack(
                [
                    jnp.stack([ c, -s], axis=-1),
                    jnp.stack([ s,  c], axis=-1),
                ],
                axis=-2,
            ),
        )

    def parameters(self) -> jax.Array:
        return jnp.concatenate([self.unit_complex, self.log_scale], axis=-1)

    def apply(self, target: jax.Array) -> jax.Array:
        assert target.shape[-1:] == (2,)
        return jnp.einsum("...ij,...j->...i", self.as_matrix(), target)

    def multiply(self, other: "SO2RPlus") -> "SO2RPlus":
        return SO2RPlus(
            unit_complex=jnp.einsum(
                "...ij,...j->...i",
                self.rotation().as_matrix(),
                other.unit_complex,
            ),
            log_scale=self.log_scale + other.log_scale,
        )

    def inverse(self) -> "SO2RPlus":
        return SO2RPlus(
            unit_complex=self.unit_complex * jnp.array([1.0, -1.0]),
            log_scale=-self.log_scale,
        )

    def normalize(self) -> "SO2RPlus":
        return SO2RPlus(
            unit_complex=self.unit_complex
            / jnp.linalg.norm(self.unit_complex, axis=-1, keepdims=True),
            log_scale=self.log_scale,
        )

    @classmethod
    def exp(cls, tangent: jax.Array) -> "SO2RPlus":
        assert tangent.shape[-1] == 2
        omega = tangent[..., 0:1]
        sigma = tangent[..., 1:2]
        return cls(
            unit_complex=jnp.concatenate(
                [jnp.cos(omega), jnp.sin(omega)], axis=-1
            ),
            log_scale=sigma,
        )

    def log(self) -> jax.Array:
        theta = jnp.arctan2(
            self.unit_complex[..., 1:2], self.unit_complex[..., 0:1]
        )
        return jnp.concatenate([theta, self.log_scale], axis=-1)

    def adjoint(self) -> jax.Array:
        return jnp.broadcast_to(
            jnp.eye(2), (*self.get_batch_axes(), 2, 2)
        )

    def jlog(self) -> jax.Array:
        # SO2RPlus is a direct product of two abelian factors; the left
        # Jacobian of the log map is the 2×2 identity.
        return jnp.broadcast_to(
            jnp.eye(2), (*self.get_batch_axes(), 2, 2)
        )

    @classmethod
    def sample_uniform(
        cls,
        key: jax.Array,
        batch_axes: jdc.Static[Tuple[int, ...]] = (),
    ) -> "SO2RPlus":
        key_rot, key_scale = jax.random.split(key)
        rotation = jaxlie.SO2.sample_uniform(key_rot, batch_axes=batch_axes)
        scale = RPlus.sample_uniform(key_scale, batch_axes=batch_axes)
        return cls.from_rotation_and_scale(rotation, scale)

    def rotation(self) -> jaxlie.SO2:
        return jaxlie.SO2(unit_complex=self.unit_complex)

    def scaling(self) -> RPlus:
        return RPlus(log_scale=self.log_scale)

    @classmethod
    def from_rotation_and_scale(cls, rotation: jaxlie.SO2,
                                scale: RPlus) -> "SO2RPlus":
        return cls(unit_complex=rotation.unit_complex,
                   log_scale=scale.log_scale)
