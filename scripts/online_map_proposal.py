"""Fit a MAP-centered proposal and importance-sample one CAPTCHA image."""

import argparse
from pathlib import Path

import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from PIL import Image

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)


def load_image(path: Path, shape: tuple[int, int]) -> jax.Array:
    """Load one RGB observation as ``(1, H, W, 3)``."""
    height, width = shape
    image = Image.open(path).convert("RGB").resize((width, height))
    return jax.numpy.asarray(np.asarray(image), dtype=jax.numpy.float32)[None] / 255.0


def main(arguments: argparse.Namespace) -> None:
    config_directory = str(PROJECT_ROOT.joinpath("configs"))
    with hydra.initialize_config_dir(
        config_dir=config_directory, version_base="1.3"
    ):
        model_config = hydra.compose(config_name="model/poisson_convsc_gmrf")
        online_config = hydra.compose(config_name="online/map_proposal")
    model = hydra.utils.instantiate(model_config.model)
    online = hydra.utils.instantiate(online_config.online)(model)
    # Only override what was asked for. The config's step budgets are derived
    # from how far the count field has to travel in log space; overriding them
    # with a smaller number gives a compile smoke test, not a fit.
    for name in ("map_max_steps", "num_dispersion_particles",
                 "num_importance_samples", "proposal_max_steps"):
        value = getattr(arguments, name)
        if value is not None:
            setattr(online, name, value)

    images = load_image(
        arguments.image,
        (model.keywords["placements"].height, model.keywords["placements"].width),
    )
    result = online(jax.random.key(arguments.seed), images)
    reconstruction = np.asarray(result.weighted_reconstruction)[0]

    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(np.asarray(images[0]))
    axes[0].set_title("Observation")
    axes[1].imshow(np.clip(reconstruction, 0.0, 1.0))
    axes[1].set_title("Importance-weighted render")
    for axis in axes:
        axis.set_axis_off()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150)

    print(
        "dispersion fit: "
        f"{int(result.dispersion_num_steps)} steps, "
        f"converged={bool(result.dispersion_converged)}"
    )
    print(f"effective sample size: {float(result.effective_sample_size):.2f}")
    print(f"expected total count: {float(result.weighted_counts.sum()):.3f}")
    print(f"dispersion losses finite: {bool(np.isfinite(result.dispersion_losses).all())}")
    print(f"largest normalized weight: {float(result.normalized_weights.max()):.4f}")
    print(
        f"MAP fit: {int(result.map_num_steps)} steps, "
        f"converged={bool(result.map_converged)}"
    )
    print(f"saved {arguments.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--map-max-steps", default=None, type=int)
    parser.add_argument("--num-dispersion-particles", default=None, type=int)
    parser.add_argument("--num-importance-samples", default=None, type=int)
    parser.add_argument(
        "--output", default=Path("map-proposal-online.png"), type=Path
    )
    parser.add_argument("--proposal-max-steps", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
