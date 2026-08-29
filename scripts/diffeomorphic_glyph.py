"""Visualize a second-order-GMRF glyph deformation by scaling and squaring."""

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
    compose_displacements,
    coordinate_grid,
    diffeomorphic_warp,
    jacobian_determinant,
    resize_velocity,
    scaling_and_squaring,
)

BOND_PRECISION = 0.4
COARSE_SHAPE = (7, 5)
ELEMENT_PRECISION = 0.1


def boundary_taper(shape, dtype):
    """Return a sine window that fixes the velocity to zero on the boundary."""
    grid = coordinate_grid(shape, dtype=dtype)
    height, width = shape
    x_taper = jnp.sin(jnp.pi * grid[..., 1] / (width - 1))
    y_taper = jnp.sin(jnp.pi * grid[..., 0] / (height - 1))
    return x_taper * y_taper


def main(arguments):
    """Draw, exponentiate, and display one random canonical velocity field."""
    shape_dictionary = ShapeDictionary.load(arguments.dictionary)
    glyph_names = list(shape_dictionary.targets.values())
    glyph_index = glyph_names.index(arguments.glyph)
    alpha = shape_dictionary.shapes[glyph_index, ..., 3]
    height, width = alpha.shape

    prior = SecondOrderGaussianMrf(
        jnp.zeros((*COARSE_SHAPE, 2)),
        ELEMENT_PRECISION * jnp.ones(COARSE_SHAPE),
        jnp.asarray(BOND_PRECISION),
        cg_iters=500,
    )
    coarse_velocity = prior.sample(jax.random.PRNGKey(arguments.seed))
    velocity = resize_velocity(coarse_velocity, alpha.shape)
    velocity = (
        arguments.velocity_scale
        * velocity
        * boundary_taper(alpha.shape, velocity.dtype)[..., None]
    )
    speed = jnp.linalg.norm(velocity, axis=-1)

    forward = scaling_and_squaring(
        velocity, squaring_steps=arguments.squaring_steps
    )
    determinant = jacobian_determinant(forward)
    inverse = scaling_and_squaring(
        -velocity, squaring_steps=arguments.squaring_steps
    )
    inverse_residual = compose_displacements(forward, inverse)
    warped_alpha = diffeomorphic_warp(
        alpha, velocity, squaring_steps=arguments.squaring_steps
    )

    figure, axes = plt.subplots(1, 5, figsize=(18, 4), layout="constrained")
    axes[0].imshow(1.0 - alpha, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"canonical {arguments.glyph}")

    axes[1].imshow(1.0 - alpha, cmap="gray", vmin=0.0, vmax=1.0)
    plot_velocity(axes[1], velocity)
    axes[1].set_title("sampled velocity $v$")

    plot_deformed_grid(axes[2], forward)
    axes[2].set_title(r"flow map $\phi=\exp(v)$")

    axes[3].imshow(1.0 - warped_alpha, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title(r"$g\circ\phi^{-1}$")

    determinant_image = axes[4].imshow(
        determinant, cmap="coolwarm", vmin=0.5, vmax=1.5
    )
    axes[4].set_title(r"$\det D\phi$")
    divider = make_axes_locatable(axes[4])
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(determinant_image, cax=colorbar_axis)

    for axis in axes[[0, 1, 3, 4]]:
        axis.set_xticks([])
        axis.set_yticks([])

    interior_residual = inverse_residual[2:-2, 2:-2]
    inverse_error = float(jnp.linalg.norm(interior_residual, axis=-1).max())
    minimum_determinant = float(determinant.min())
    figure.suptitle(
        "Second-order GMRF velocity; "
        f"min det $D\\phi$={minimum_determinant:.3f}, "
        f"inverse error={inverse_error:.3f} px"
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"glyph: {arguments.glyph} ({height} x {width})")
    print(f"maximum velocity: {float(speed.max()):.6f} pixels")
    print(f"inverse consistency error: {inverse_error:.6f} pixels")
    print(f"minimum Jacobian determinant: {minimum_determinant:.6f}")
    print(f"output: {arguments.output.resolve()}")


def plot_deformed_grid(axis, displacement, spacing=4):
    """Plot the forward image of the canonical coordinate lattice."""
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


def plot_velocity(axis, velocity, spacing=4):
    """Plot a velocity field over its canonical pixel grid."""
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
        width=0.008,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dictionary",
        default="data/captcha_sandbox/dictionary",
    )
    parser.add_argument("--glyph", default="M_42")
    parser.add_argument(
        "--output", default=Path("logs/diffeomorphic_glyph.png"), type=Path
    )
    parser.add_argument("--seed", default=4, type=int)
    parser.add_argument("--squaring-steps", default=7, type=int)
    parser.add_argument("--velocity-scale", default=1.5, type=float)
    main(parser.parse_args())
