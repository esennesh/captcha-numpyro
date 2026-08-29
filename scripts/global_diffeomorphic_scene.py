"""Warp a convolved foreground with one affine-free whole-image flow."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from mpl_toolkits.axes_grid1 import make_axes_locatable

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dictionary import ShapeDictionary
from src.distributions.spatial import SecondOrderGaussianMrf
from src.model.diffeomorphism import (
    affine_basis,
    affine_free_velocity,
    boundary_taper,
    coordinate_grid,
    diffeomorphic_warp,
    jacobian_determinant,
    scaling_and_squaring,
)
from src.model.model import _ink_kernel, _stamp

BOND_PRECISION = 0.4
CANVAS_SHAPE = (80, 100)
CENTERS = jnp.asarray([[40, 28], [40, 72]], dtype=jnp.int32)
COARSE_SHAPE = (10, 13)
COLORS = jnp.asarray([[0.10, 0.35, 0.28], [0.38, 0.16, 0.52]])
ELEMENT_PRECISION = 0.1
GLYPH_NAMES = ("J_42", "8_42")


def ink_over_background(ink, background):
    """Composite closeness-premultiplied foreground ink over fixed paper."""
    depth = ink[..., 3]
    opacity = -jnp.expm1(-depth)
    tint = jnp.where(
        depth[..., None] > 1e-6,
        ink[..., :3] / jnp.clip(depth[..., None], 1e-6, None),
        1.0,
    )
    return opacity[..., None] * tint + (1.0 - opacity[..., None]) * background


def main(arguments):
    """Sample one global velocity and warp only a stamped foreground."""
    shape_dictionary = ShapeDictionary.load(arguments.dictionary)
    glyph_names = list(shape_dictionary.targets.values())
    glyph_indices = jnp.asarray(
        [glyph_names.index(name) for name in GLYPH_NAMES], dtype=jnp.int32
    )
    ink_kernel = _ink_kernel(shape_dictionary.shapes)
    for occurrence, glyph_index in enumerate(glyph_indices.tolist()):
        depth = ink_kernel[..., 3:4, glyph_index]
        ink_kernel = ink_kernel.at[..., :3, glyph_index].set(
            depth * COLORS[occurrence]
        )

    counts = jnp.zeros((*CANVAS_SHAPE, ink_kernel.shape[-1])).at[
        CENTERS[:, 0], CENTERS[:, 1], glyph_indices
    ].set(1.0)
    foreground = _stamp(counts[None], ink_kernel)[0]

    prior = SecondOrderGaussianMrf(
        jnp.zeros((*COARSE_SHAPE, 2)),
        ELEMENT_PRECISION * jnp.ones(COARSE_SHAPE),
        jnp.asarray(BOND_PRECISION),
        cg_iters=500,
    )
    coarse_velocity = prior.sample(jax.random.PRNGKey(arguments.seed))
    window = boundary_taper(CANVAS_SHAPE, dtype=coarse_velocity.dtype)
    velocity = arguments.velocity_scale * affine_free_velocity(
        coarse_velocity,
        prior.solve_precision,
        CANVAS_SHAPE,
        window=window,
    )

    forward = scaling_and_squaring(
        velocity, squaring_steps=arguments.squaring_steps
    )
    determinant = jacobian_determinant(forward)
    warped_foreground = diffeomorphic_warp(
        foreground, velocity, squaring_steps=arguments.squaring_steps
    )

    basis = affine_basis(CANVAS_SHAPE, dtype=velocity.dtype)
    affine_moments = (
        jnp.einsum("hwk,hwc->kc", basis, velocity)
        / (CANVAS_SHAPE[0] * CANVAS_SHAPE[1])
    )
    boundary_velocity = jnp.concatenate(
        (velocity[0], velocity[-1], velocity[:, 0], velocity[:, -1])
    )

    grid = coordinate_grid(CANVAS_SHAPE, dtype=velocity.dtype)
    paper = jnp.stack(
        (
            0.96 + 0.012 * jnp.sin(0.11 * grid[..., 0]),
            0.97 + 0.010 * jnp.cos(0.09 * grid[..., 1]),
            0.985 + 0.008 * jnp.sin(0.07 * (grid[..., 0] + grid[..., 1])),
        ),
        axis=-1,
    )
    reference_image = ink_over_background(foreground, paper)
    warped_image = ink_over_background(warped_foreground, paper)
    difference = jnp.linalg.norm(warped_image - reference_image, axis=-1)

    figure, axes = plt.subplots(1, 6, figsize=(22, 4), layout="constrained")
    axes[0].imshow(reference_image, vmin=0.0, vmax=1.0)
    axes[0].set_title("stamp over fixed paper")
    plot_velocity(axes[1], velocity)
    axes[1].set_title("affine-free velocity")
    plot_deformed_grid(axes[2], forward)
    axes[2].set_title(r"global $\phi=\exp(v)$")
    axes[3].imshow(warped_image, vmin=0.0, vmax=1.0)
    axes[3].set_title("warped ink, fixed paper")
    difference_image = axes[4].imshow(difference, cmap="magma", vmin=0.0)
    axes[4].set_title("foreground warp effect")
    difference_divider = make_axes_locatable(axes[4])
    difference_colorbar_axis = difference_divider.append_axes(
        "right", size="5%", pad=0.05
    )
    figure.colorbar(difference_image, cax=difference_colorbar_axis)
    determinant_image = axes[5].imshow(
        determinant, cmap="coolwarm", vmin=0.0, vmax=2.5
    )
    axes[5].set_title(r"$\det D\phi$")
    determinant_divider = make_axes_locatable(axes[5])
    determinant_colorbar_axis = determinant_divider.append_axes(
        "right", size="5%", pad=0.05
    )
    figure.colorbar(determinant_image, cax=determinant_colorbar_axis)
    for axis in axes[[0, 3, 4, 5]]:
        axis.set_xticks([])
        axis.set_yticks([])

    affine_error = float(jnp.max(jnp.abs(affine_moments)))
    boundary_error = float(jnp.max(jnp.abs(boundary_velocity)))
    maximum_velocity = float(jnp.linalg.norm(velocity, axis=-1).max())
    minimum_determinant = float(determinant.min())
    figure.suptitle(
        "One foreground flow; "
        f"min det $D\\phi$={minimum_determinant:.3f}, "
        f"affine moment={affine_error:.1e}"
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"affine moment error: {affine_error:.8e}")
    print(f"boundary velocity error: {boundary_error:.8e}")
    print(f"maximum velocity: {maximum_velocity:.6f} pixels")
    print(f"minimum Jacobian determinant: {minimum_determinant:.6f}")
    print(f"output: {arguments.output.resolve()}")


def plot_deformed_grid(axis, displacement, spacing=8):
    """Plot the forward image of the whole-image coordinate lattice."""
    mapping = coordinate_grid(
        displacement.shape[:2], dtype=displacement.dtype
    ) + displacement
    height, width = displacement.shape[:2]
    for y in range(0, height, spacing):
        axis.plot(mapping[y, :, 1], mapping[y, :, 0], color="tab:blue", lw=0.7)
    for x in range(0, width, spacing):
        axis.plot(mapping[:, x, 1], mapping[:, x, 0], color="tab:blue", lw=0.7)
    axis.set_aspect("equal")
    axis.set_xlim(0, width - 1)
    axis.set_ylim(height - 1, 0)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_velocity(axis, velocity, spacing=8):
    """Plot the global velocity on the image coordinate grid."""
    grid = coordinate_grid(velocity.shape[:2], dtype=velocity.dtype)
    axis.quiver(
        np.asarray(grid[::spacing, ::spacing, 1]),
        np.asarray(grid[::spacing, ::spacing, 0]),
        np.asarray(velocity[::spacing, ::spacing, 1]),
        np.asarray(velocity[::spacing, ::spacing, 0]),
        angles="xy",
        color="tab:red",
        scale=1.0,
        scale_units="xy",
        width=0.007,
    )
    axis.set_aspect("equal")
    axis.set_xlim(0, velocity.shape[1] - 1)
    axis.set_ylim(velocity.shape[0] - 1, 0)
    axis.set_xticks([])
    axis.set_yticks([])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dictionary", default="data/captcha_sandbox/dictionary"
    )
    parser.add_argument(
        "--output", default=Path("logs/global_diffeomorphic_scene.png"), type=Path
    )
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--squaring-steps", default=7, type=int)
    parser.add_argument("--velocity-scale", default=3.0, type=float)
    main(parser.parse_args())
