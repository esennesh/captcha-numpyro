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
    "affine_basis",
    "affine_free_velocity",
    "bilinear_sample",
    "boundary_taper",
    "compose_displacements",
    "conditioned_velocity",
    "coordinate_grid",
    "diffeomorphic_warp",
    "jacobian_determinant",
    "resize_velocity",
    "scaling_and_squaring",
    "sparse_diffeomorphic_stamp",
    "translation_basis",
    "translation_free_velocity",
    "warp_image",
]


def affine_basis(
    shape: tuple[int, int], *, dtype=jnp.float32
) -> Float[Array, "H W 3"]:
    """Return the normalized coordinate basis ``[1, y, x]``."""
    grid = coordinate_grid(shape, dtype=dtype)
    height, width = shape
    x = 2.0 * grid[..., 1] / max(width - 1, 1) - 1.0
    y = 2.0 * grid[..., 0] / max(height - 1, 1) - 1.0
    return jnp.stack((jnp.ones_like(y), y, x), axis=-1)


def translation_basis(
    shape: tuple[int, int], *, dtype=jnp.float32
) -> Float[Array, "H W 1"]:
    """Return the constant basis ``[1]``, whose moment is a translation."""
    return affine_basis(shape, dtype=dtype)[..., :1]


def conditioned_velocity(
    velocity: Float[Array, "h w 2"],
    covariance_solve,
    shape: tuple[int, int],
    basis: Float[Array, "H W k"],
    *,
    method: str = "cubic",
    window: Float[Array, "H W"] | None = None,
) -> Float[Array, "H W 2"]:
    r"""Condition a coarse Gaussian velocity to have no ``basis`` image modes.

    Let the raw coarse field be ``u ~ N(0, Q^-1)``, let ``R`` resize it to
    ``shape``, let ``W`` be the optional image-space window, and let ``B`` be
    ``basis``.  The conditioned field satisfies

    .. math::

        B^\mathsf{T}WRu_\perp=0.

    Defining the coarse constraint matrix ``C = R.T @ W @ B``, a draw from the
    Gaussian conditional on this constraint is

    .. math::

        u_\perp = u - Q^{-1}C
        (C^\mathsf{T}Q^{-1}C)^{-1}C^\mathsf{T}u.

    ``covariance_solve`` must apply ``Q^-1`` on the coarse lattice; passing
    :meth:`SecondOrderGaussianMrf.solve_precision` retains the GMRF covariance
    geometry without materializing a dense precision.  The only dense solve is
    ``k`` by ``k``.  Both velocity channels use the same conditional map.

    Passing a window that vanishes at the image boundary additionally gives a
    fixed boundary.  Resizing, conditioning, and windowing are linear, so the
    result is a Gaussian field.  It is singular in ambient image coordinates,
    as any Gaussian supported on an exactly constrained subspace must be.

    :param basis: image-space modes to remove, ``(H, W, k)``.  See
        :func:`translation_basis` and :func:`affine_basis`.
    """
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError(
            "velocity needs shape (h, w, 2), "
            f"got {velocity.shape}"
        )
    if window is not None and window.shape != shape:
        raise ValueError(
            "window and output need matching spatial shapes, "
            f"got {window.shape} and {shape}"
        )
    if basis.ndim != 3 or basis.shape[:2] != shape:
        raise ValueError(
            f"basis needs shape {(*shape, -1)}, got {basis.shape}"
        )

    modes = basis.shape[-1]
    weights = (
        jnp.ones(shape, dtype=velocity.dtype)
        if window is None
        else window.astype(velocity.dtype)
    )
    weighted_basis = weights[..., None] * basis.astype(velocity.dtype)

    coarse_shape = velocity.shape[:2]

    def resize_basis(field):
        return jax.image.resize(field, (*shape, modes), method=method)

    _, resize_transpose = jax.vjp(
        resize_basis,
        jnp.zeros((*coarse_shape, modes), dtype=velocity.dtype),
    )
    constraints = resize_transpose(weighted_basis)[0]
    covariance_constraints = covariance_solve(constraints)
    conditional_gram = jnp.einsum(
        "hwk,hwl->kl", constraints, covariance_constraints
    )
    basis_coordinates = jnp.einsum("hwk,hwc->kc", constraints, velocity)
    coefficients = jnp.linalg.solve(conditional_gram, basis_coordinates)
    conditioned = velocity - jnp.einsum(
        "hwk,kc->hwc", covariance_constraints, coefficients
    )
    resized = resize_velocity(conditioned, shape, method=method)
    return weights[..., None] * resized


def affine_free_velocity(
    velocity: Float[Array, "h w 2"],
    covariance_solve,
    shape: tuple[int, int],
    *,
    method: str = "cubic",
    window: Float[Array, "H W"] | None = None,
) -> Float[Array, "H W 2"]:
    """Remove the six image affine modes, ``[1, y, x]`` in each channel.

    The three scalar modes per channel are the two translations and the four
    coefficients of a linear ``2 x 2`` map, so this deletes translation,
    rotation, scale, and shear together.

    **This is rarely the conditioning you want with the convolutional
    renderer.** Translation is genuinely redundant with the count field's site
    index, but rotation, scale, and shear have no other latent to express them,
    so removing them removes them from the model. Worse, a *local* deformation
    generally carries a nonzero global affine moment, so the projection forces
    a compensating counter-deformation elsewhere in the image: least-squares
    fitting one glyph-sized 25 degree rotation costs 2294 nats here against 99
    nats under :func:`translation_free_velocity`. Prefer that function unless
    another latent supplies the pose.
    """
    return conditioned_velocity(
        velocity,
        covariance_solve,
        shape,
        affine_basis(shape, dtype=velocity.dtype),
        method=method,
        window=window,
    )


def translation_free_velocity(
    velocity: Float[Array, "h w 2"],
    covariance_solve,
    shape: tuple[int, int],
    *,
    method: str = "cubic",
    window: Float[Array, "H W"] | None = None,
) -> Float[Array, "H W 2"]:
    """Remove the two image translation modes and nothing else.

    Translation is the one affine mode that is redundant with the count field,
    which already places a glyph at any site. Rotation, scale, and shear stay
    available, at a prior cost that rises with how far the deformation departs
    from the identity rather than with where it sits in the image.
    """
    return conditioned_velocity(
        velocity,
        covariance_solve,
        shape,
        translation_basis(shape, dtype=velocity.dtype),
        method=method,
        window=window,
    )


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


def boundary_taper(
    shape: tuple[int, int], *, dtype=jnp.float32
) -> Float[Array, "H W"]:
    """Return a sine window that fixes a velocity to zero on the boundary."""
    grid = coordinate_grid(shape, dtype=dtype)
    height, width = shape
    x_taper = jnp.sin(jnp.pi * grid[..., 1] / max(width - 1, 1))
    y_taper = jnp.sin(jnp.pi * grid[..., 0] / max(height - 1, 1))
    taper = x_taper * y_taper
    taper = taper.at[0, :].set(0.0).at[-1, :].set(0.0)
    return taper.at[:, 0].set(0.0).at[:, -1].set(0.0)


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


def sparse_diffeomorphic_stamp(
    amplitudes: Array,
    centers: Array,
    glyph_indices: Array,
    ink_kernel: Float[Array, "Kh Kw C K"],
    velocities: Float[Array, "N Kh Kw 2"],
    *,
    canvas_shape: tuple[int, int],
    squaring_steps: int = 7,
) -> Float[Array, "H W C"]:
    r"""Warp and scatter-add a sparse list of glyph occurrences.

    Occurrence ``i`` contributes

    .. math::

        a_i K_{k_i}\!\left(\exp(-v_i)(p-s_i)\right)

    to the canvas.  ``centers`` are integer ``(y, x)`` image sites using the
    same even-kernel convention as :func:`src.model.model._stamp`.  The ink
    channels should be closeness-premultiplied so their sum has exactly the
    optical-depth semantics of the convolutional renderer.
    """
    occurrence_count = amplitudes.shape[0]
    if amplitudes.shape != (occurrence_count,):
        raise ValueError(f"amplitudes need shape (N,), got {amplitudes.shape}")
    if centers.shape != (occurrence_count, 2):
        raise ValueError(f"centers need shape (N, 2), got {centers.shape}")
    if glyph_indices.shape != (occurrence_count,):
        raise ValueError(
            f"glyph_indices need shape (N,), got {glyph_indices.shape}"
        )
    if ink_kernel.ndim != 4:
        raise ValueError(
            "ink_kernel needs shape (Kh, Kw, C, K), "
            f"got {ink_kernel.shape}"
        )
    if not jnp.issubdtype(centers.dtype, jnp.integer):
        raise ValueError(f"centers need an integer dtype, got {centers.dtype}")

    kernel_height, kernel_width, channels = ink_kernel.shape[:3]
    expected_velocity_shape = (
        occurrence_count, kernel_height, kernel_width, 2
    )
    if velocities.shape != expected_velocity_shape:
        raise ValueError(
            f"velocities need shape {expected_velocity_shape}, "
            f"got {velocities.shape}"
        )

    selected_kernels = jnp.moveaxis(
        ink_kernel[..., glyph_indices], -1, 0
    )
    warped_kernels = jax.vmap(
        lambda kernel, velocity: diffeomorphic_warp(
            kernel, velocity, squaring_steps=squaring_steps
        )
    )(selected_kernels, velocities)
    weighted_kernels = (
        amplitudes.astype(ink_kernel.dtype)[:, None, None, None]
        * warped_kernels
    )

    canvas_height, canvas_width = canvas_shape
    x = (
        centers[:, 1, None, None]
        + jnp.arange(kernel_width)[None, None, :]
        - (kernel_width - 1) // 2
    )
    x = jnp.broadcast_to(
        x, (occurrence_count, kernel_height, kernel_width)
    )
    y = (
        centers[:, 0, None, None]
        + jnp.arange(kernel_height)[None, :, None]
        - (kernel_height - 1) // 2
    )
    y = jnp.broadcast_to(
        y, (occurrence_count, kernel_height, kernel_width)
    )
    valid = (x >= 0) & (x < canvas_width) & (y >= 0) & (y < canvas_height)
    x = jnp.clip(x, 0, canvas_width - 1)
    y = jnp.clip(y, 0, canvas_height - 1)

    canvas = jnp.zeros(
        (canvas_height, canvas_width, channels), dtype=ink_kernel.dtype
    )
    return canvas.at[y, x].add(weighted_kernels * valid[..., None])


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
