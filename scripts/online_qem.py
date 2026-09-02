"""Fit the QEM online posterior to one CAPTCHA image."""

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
        model_config = hydra.compose(config_name="model/poisson_convsc")
        online_config = hydra.compose(config_name="online/qem")
    model = hydra.utils.instantiate(model_config.model)
    online = hydra.utils.instantiate(online_config.online)(model)
    online.num_posterior_samples = arguments.num_posterior_samples
    online.num_samples = arguments.num_samples
    online.num_steps = arguments.num_steps

    images = load_image(
        arguments.image,
        (model.keywords["placements"].height, model.keywords["placements"].width),
    )
    result = online(jax.random.key(arguments.seed), images)
    reconstruction = np.asarray(result.reconstructions).mean(axis=0)[0]

    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(np.asarray(images[0]))
    axes[0].set_title("Observation")
    axes[1].imshow(np.clip(reconstruction, 0.0, 1.0))
    axes[1].set_title("QEM posterior mean render")
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(arguments.output, dpi=150)

    final_log_marginal = float(result.qem_result.log_marginals[-1])
    print(f"candidate sites:\n{np.asarray(result.candidate_sites)}")
    print(f"final relative log P_MP(x): {final_log_marginal:.3f}")
    print(f"saved {arguments.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--num-posterior-samples", default=8, type=int)
    parser.add_argument("--num-samples", default=4, type=int)
    parser.add_argument("--num-steps", default=20, type=int)
    parser.add_argument("--output", default=Path("qem-online.png"), type=Path)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
