"""Render two independently warped glyph occurrences by sparse scatter-add."""

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
    boundary_taper,
    coordinate_grid,
    jacobian_determinant,
    resize_velocity,
    scaling_and_squaring,
    sparse_diffeomorphic_stamp,
)
from src.model.model import _ink_kernel, _stamp

BOND_PRECISION = 0.4
CANVAS_SHAPE = (80, 100)
CENTERS = jnp.asarray([[40, 28], [40, 72]], dtype=jnp.int32)
COARSE_SHAPE = (7, 5)
COLORS = jnp.asarray([[0.10, 0.35, 0.28], [0.38, 0.16, 0.52]])
ELEMENT_PRECISION = 0.1
GLYPH_NAMES = ("J_42", "8_42")


def ink_over_white(ink):
    """Convert closeness-premultiplied ink to an RGB image on white paper."""
    depth = ink[..., 3]
    opacity = -jnp.expm1(-depth)
    tint = jnp.where(
        depth[..., None] > 1e-6,
        ink[..., :3] / jnp.clip(depth[..., None], 1e-6, None),
        1.0,
    )
    return opacity[..., None] * tint + (1.0 - opacity[..., None])


def main(arguments):
    """Sample two velocities and compare sparse warping with ordinary stamps."""
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

    kernel_shape = ink_kernel.shape[:2]
    prior = SecondOrderGaussianMrf(
        jnp.zeros((*COARSE_SHAPE, 2)),
        ELEMENT_PRECISION * jnp.ones(COARSE_SHAPE),
        jnp.asarray(BOND_PRECISION),
        cg_iters=500,
    )
    coarse_velocities = prior.sample(
        jax.random.PRNGKey(arguments.seed), sample_shape=(len(GLYPH_NAMES),)
    )
    velocities = jax.vmap(
        lambda velocity: resize_velocity(velocity, kernel_shape)
    )(coarse_velocities)
    velocities = (
        arguments.velocity_scale
        * velocities
        * boundary_taper(kernel_shape, dtype=velocities.dtype)[None, ..., None]
    )

    amplitudes = jnp.ones((len(GLYPH_NAMES),))
    warped_ink = sparse_diffeomorphic_stamp(
        amplitudes,
        CENTERS,
        glyph_indices,
        ink_kernel,
        velocities,
        canvas_shape=CANVAS_SHAPE,
        squaring_steps=arguments.squaring_steps,
    )
    zero_velocity_ink = sparse_diffeomorphic_stamp(
        amplitudes,
        CENTERS,
        glyph_indices,
        ink_kernel,
        jnp.zeros_like(velocities),
        canvas_shape=CANVAS_SHAPE,
        squaring_steps=arguments.squaring_steps,
    )

    counts = jnp.zeros((*CANVAS_SHAPE, ink_kernel.shape[-1])).at[
        CENTERS[:, 0], CENTERS[:, 1], glyph_indices
    ].add(amplitudes)
    convolutional_ink = _stamp(counts[None], ink_kernel)[0]
    zero_velocity_error = float(
        jnp.max(jnp.abs(zero_velocity_ink - convolutional_ink))
    )

    forwards = jax.vmap(
        lambda velocity: scaling_and_squaring(
            velocity, squaring_steps=arguments.squaring_steps
        )
    )(velocities)
    determinants = jax.vmap(jacobian_determinant)(forwards)
    reference_image = ink_over_white(convolutional_ink)
    warped_image = ink_over_white(warped_ink)
    difference = jnp.linalg.norm(warped_image - reference_image, axis=-1)

    figure, axes = plt.subplots(1, 7, figsize=(24, 4), layout="constrained")
    axes[0].imshow(reference_image, vmin=0.0, vmax=1.0)
    axes[0].set_title("ordinary `_stamp`")
    for occurrence, name in enumerate(GLYPH_NAMES):
        grid_axis = axes[1 + 2 * occurrence]
        determinant_axis = axes[2 + 2 * occurrence]
        plot_deformed_grid(grid_axis, forwards[occurrence])
        grid_axis.set_title(rf"{name}: $\exp(v_{occurrence})$")
        determinant_image = determinant_axis.imshow(
            determinants[occurrence], cmap="coolwarm", vmin=0.5, vmax=1.5
        )
        determinant_axis.set_title(rf"$\det D\phi_{occurrence}$")
        divider = make_axes_locatable(determinant_axis)
        colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
        figure.colorbar(determinant_image, cax=colorbar_axis)

    axes[5].imshow(warped_image, vmin=0.0, vmax=1.0)
    axes[5].set_title("sparse warped scene")
    difference_image = axes[6].imshow(difference, cmap="magma", vmin=0.0)
    axes[6].set_title("warp effect")
    divider = make_axes_locatable(axes[6])
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(difference_image, cax=colorbar_axis)
    for axis in axes[[0, 2, 4, 5, 6]]:
        axis.set_xticks([])
        axis.set_yticks([])

    minimum_determinants = np.asarray(determinants.min(axis=(1, 2)))
    figure.suptitle(
        "Independent per-occurrence flows; "
        f"zero-flow error={zero_velocity_error:.1e}"
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"minimum Jacobian determinants: {minimum_determinants.round(6)}")
    print(f"output: {arguments.output.resolve()}")
    print(f"zero-velocity `_stamp` error: {zero_velocity_error:.8e}")


def plot_deformed_grid(axis, displacement, spacing=4):
    """Plot the forward image of one occurrence's canonical lattice."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dictionary", default="data/captcha_sandbox/dictionary"
    )
    parser.add_argument(
        "--output", default=Path("logs/sparse_diffeomorphic_scene.png"), type=Path
    )
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--squaring-steps", default=7, type=int)
    parser.add_argument("--velocity-scale", default=1.2, type=float)
    main(parser.parse_args())
