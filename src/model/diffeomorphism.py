"""Differentiable pixel-grid approximations to diffeomorphic flows.

Coordinate-valued arrays use ``(y, x)`` order and pixel units.  A displacement
``d`` represents the map ``phi(u) = u + d(u)``.  Stationary velocities are
exponentiated with scaling and squaring; images are warped by sampling through
the inverse displacement.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    "bilinear_sample",
    "compose_displacements",
    "coordinate_grid",
    "diffeomorphic_warp",
    "jacobian_determinant",
    "resize_velocity",
    "scaling_and_squaring",
    "warp_image",
]


def bilinear_sample(
    field: Array,
    coordinates: Array,
) -> Array:
    """Sample a scalar or channel-valued field at pixel coordinates.

    Coordinates outside the field use constant boundary extension.  The final
    coordinate component is ordered ``(y, x)``.
    """
    if field.ndim not in (2, 3):
        raise ValueError(
            "field needs shape (H, W) or (H, W, C), "
            f"got {field.shape}"
        )
    if coordinates.shape[-1] != 2:
        raise ValueError(
            "coordinates need a final (y, x) axis, "
            f"got {coordinates.shape}"
        )

    height, width = field.shape[:2]
    x = jnp.clip(coordinates[..., 1], 0.0, width - 1.0)
    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, width - 1)
    x_weight = x - x0
    y = jnp.clip(coordinates[..., 0], 0.0, height - 1.0)
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, height - 1)
    y_weight = y - y0

    if field.ndim == 3:
        x_weight = x_weight[..., None]
        y_weight = y_weight[..., None]

    bottom = (
        field[y1, x0] * (1.0 - x_weight)
        + field[y1, x1] * x_weight
    )
    top = (
        field[y0, x0] * (1.0 - x_weight)
        + field[y0, x1] * x_weight
    )
    return top * (1.0 - y_weight) + bottom * y_weight


def compose_displacements(
    outer: Float[Array, "H W 2"],
    inner: Float[Array, "H W 2"],
) -> Float[Array, "H W 2"]:
    r"""Return the displacement of ``outer_map o inner_map``.

    If ``phi(u) = u + outer(u)`` and ``psi(u) = u + inner(u)``, then

    .. math::

        (\phi\circ\psi)(u)-u
        = inner(u) + outer(u + inner(u)).
    """
    if outer.shape != inner.shape or outer.shape[-1] != 2:
        raise ValueError(
            "outer and inner need the same (H, W, 2) shape, got "
            f"{outer.shape} and {inner.shape}"
        )
    grid = coordinate_grid(outer.shape[:2], dtype=outer.dtype)
    return inner + bilinear_sample(outer, grid + inner)


def coordinate_grid(
    shape: tuple[int, int], *, dtype=jnp.float32
) -> Float[Array, "H W 2"]:
    """Return an identity pixel grid in ``(y, x)`` coordinate order."""
    height, width = shape
    x = jnp.arange(width, dtype=dtype)
    y = jnp.arange(height, dtype=dtype)
    y_grid, x_grid = jnp.meshgrid(y, x, indexing="ij")
    return jnp.stack((y_grid, x_grid), axis=-1)


def diffeomorphic_warp(
    source: Array,
    velocity: Float[Array, "H W 2"],
    *,
    squaring_steps: int = 7,
) -> Array:
    r"""Warp ``source`` by ``phi = exp(velocity)`` using backward sampling.

    The inverse of a stationary flow is ``exp(-velocity)``, so output location
    ``p`` samples the source at ``phi^{-1}(p)``.
    """
    inverse = scaling_and_squaring(
        -velocity, squaring_steps=squaring_steps
    )
    return warp_image(source, inverse)


def jacobian_determinant(
    displacement: Float[Array, "H W 2"],
) -> Float[Array, "H W"]:
    r"""Approximate ``det D phi`` for ``phi(u) = u + displacement(u)``."""
    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        raise ValueError(
            "displacement needs shape (H, W, 2), "
            f"got {displacement.shape}"
        )
    mapping = coordinate_grid(
        displacement.shape[:2], dtype=displacement.dtype
    ) + displacement
    phi_x_dy, phi_x_dx = jnp.gradient(mapping[..., 1])
    phi_y_dy, phi_y_dx = jnp.gradient(mapping[..., 0])
    return phi_y_dy * phi_x_dx - phi_y_dx * phi_x_dy


def resize_velocity(
    velocity: Float[Array, "h w 2"],
    shape: tuple[int, int],
    *,
    method: str = "cubic",
) -> Float[Array, "H W 2"]:
    """Resize a velocity whose values are already in output-pixel units."""
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError(
            "velocity needs shape (H, W, 2), "
            f"got {velocity.shape}"
        )
    return jax.image.resize(velocity, (*shape, 2), method=method)


def scaling_and_squaring(
    velocity: Float[Array, "H W 2"],
    *,
    squaring_steps: int = 7,
) -> Float[Array, "H W 2"]:
    r"""Approximate the displacement of ``exp(velocity)``.

    Start from the small map ``psi_0(u) = u + velocity(u) / 2**J`` and square
    it ``J`` times: ``psi_(j+1) = psi_j o psi_j``.
    """
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError(
            "velocity needs shape (H, W, 2), "
            f"got {velocity.shape}"
        )
    if squaring_steps < 0:
        raise ValueError(
            f"squaring_steps needs to be nonnegative, got {squaring_steps}"
        )
    displacement = velocity / float(2**squaring_steps)
    for _ in range(squaring_steps):
        displacement = compose_displacements(displacement, displacement)
    return displacement


def warp_image(
    source: Array,
    inverse_displacement: Float[Array, "H W 2"],
) -> Array:
    """Backward-sample ``source`` through an inverse displacement field."""
    if source.shape[:2] != inverse_displacement.shape[:2]:
        raise ValueError(
            "source and inverse_displacement need matching spatial shapes, "
            f"got {source.shape[:2]} and {inverse_displacement.shape[:2]}"
        )
    grid = coordinate_grid(source.shape[:2], dtype=inverse_displacement.dtype)
    return bilinear_sample(source, grid + inverse_displacement)
