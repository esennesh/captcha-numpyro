r"""Candidate-restricted online inference for the Poisson CAPTCHA model.

The full count field has ``H * W * K`` integer coordinates. Online methods that
optimize or importance-sample that tensor as one latent site are not
computationally useful, so this module first constructs a deterministic set
``L`` of spatial locations from the observation. Every one of the ``K`` glyph
classes remains available at each retained location, giving
``S = L x {0, ..., K - 1}``. Counts outside ``S`` are conditioned to zero.

If ``S`` is the shortlisted set and the homogeneous count rate is ``lambda``,
the restricted count factor is

.. math::

    p_\theta(a_S, a_{\neg S}=0)
    = \prod_{i\in S}\operatorname{Poisson}(a_i;\lambda)
      \exp\{-\lambda(|\mathcal I|-|S|)\}.

The final exponential is retained with :func:`numpyro.factor`, so the
restricted model has the full model's joint density on its support.  Candidate
selection is an inference approximation, not part of the generative model.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from flax import nnx
from jaxtyping import Array

from src.data.dictionary import ShapeDictionary
from src.model.model import (
    TexturedDiffeomorphicPoissonConvPlacements,
    _dictionary_alpha_rgb,
)


def _candidate_scores(images: Array, shape_dict: ShapeDictionary) -> Array:
    """Return normalized darkness-template scores with shape ``(H, W, K)``."""
    images = jnp.asarray(images)
    if images.ndim == 3:
        images = images[jnp.newaxis]
    if images.ndim != 4 or images.shape[0] != 1:
        raise ValueError(
            "online candidate selection needs one (H, W, C) image, got "
            f"{images.shape}"
        )

    alpha, _ = _dictionary_alpha_rgb(shape_dict.shapes)
    darkness = 1.0 - images.min(axis=-1, keepdims=True)
    kernel = jnp.transpose(alpha, (1, 2, 3, 0))
    scores = jax.lax.conv_general_dilated(
        darkness,
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[0]
    normalizer = jnp.maximum(alpha.sum(axis=(1, 2, 3)), 1e-6)
    return scores / normalizer


def candidate_indices(
    images: Array,
    shape_dict: ShapeDictionary,
    *,
    min_distance: float = 6.0,
    num_locations: int = 4,
) -> Array:
    """Select locations by matched filtering and retain every glyph there.

    Non-maximum suppression acts only on spatial locations. Every dictionary
    identity remains a candidate at each selected location, so the candidate
    restriction cannot discard an identity while retaining its location.
    """
    if min_distance < 0:
        raise ValueError("min_distance needs to be nonnegative")
    if num_locations < 1:
        raise ValueError("num_locations needs to be positive")

    scores = np.asarray(_candidate_scores(images, shape_dict))
    height, width, _ = scores.shape
    location_order = np.argsort(scores.max(axis=-1).ravel())[::-1]
    locations: list[tuple[int, int]] = []
    for flat_index in location_order:
        y, x = np.unravel_index(flat_index, (height, width))
        separated = all(
            (y - other_y) ** 2 + (x - other_x) ** 2 >= min_distance**2
            for other_y, other_x in locations
        )
        if separated:
            locations.append((int(y), int(x)))
        if len(locations) == num_locations:
            break

    candidates: list[tuple[int, int, int]] = []
    for y, x in locations:
        class_order = np.argsort(scores[y, x])[::-1]
        candidates.extend((y, x, int(glyph)) for glyph in class_order)
    return jnp.asarray(candidates, dtype=jnp.int32)


class CandidateTexturedDiffeomorphicPoissonConvPlacements(
    TexturedDiffeomorphicPoissonConvPlacements
):
    """Render a small candidate count vector through the full field renderer."""

    def __init__(self, candidate_sites: Array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        candidate_sites = jnp.asarray(candidate_sites, dtype=jnp.int32)
        if candidate_sites.ndim != 2 or candidate_sites.shape[-1] != 3:
            raise ValueError(
                "candidate_sites needs shape (M, 3), got "
                f"{candidate_sites.shape}"
            )
        if candidate_sites.shape[0] < 1:
            raise ValueError("candidate_sites cannot be empty")
        self.candidate_sites = candidate_sites

    @classmethod
    def from_placements(
        cls,
        candidate_sites: Array,
        placements: TexturedDiffeomorphicPoissonConvPlacements,
    ) -> "CandidateTexturedDiffeomorphicPoissonConvPlacements":
        """Copy renderer and prior settings from a configured placement module."""
        return cls(
            candidate_sites,
            placements.shape_dict,
            cg_iters=placements.cg_iters,
            expected_count=placements.expected_count,
            img_h=placements.height,
            img_w=placements.width,
            rngs=nnx.Rngs(0),
            texture_bond_precision=placements.texture_bond_precision,
            texture_element_precision=placements.texture_element_precision,
            warp_bond_precision=placements.warp_bond_precision,
            warp_coarse_height=placements.warp_coarse_height,
            warp_coarse_width=placements.warp_coarse_width,
            warp_element_precision=placements.warp_element_precision,
            warp_scale=placements.warp_scale,
            warp_squaring_steps=placements.warp_squaring_steps,
        )

    def sample_counts(self) -> Array:
        """Sample shortlisted counts and scatter them onto the dense render grid."""
        log_rate = numpyro.param(
            "log_rate", jnp.log(self.expected_count / self.num_sites)
        )
        rate = jnp.exp(log_rate)
        candidate_rate = jnp.broadcast_to(
            rate, (1, self.candidate_sites.shape[0])
        )
        candidate_counts = numpyro.sample(
            "candidate_counts", dist.Poisson(candidate_rate).to_event(1)
        )
        omitted_count = self.num_sites - self.candidate_sites.shape[0]
        numpyro.factor(
            "omitted_count_mass",
            -omitted_count * rate,
        )

        y, x, glyph = jnp.moveaxis(self.candidate_sites, -1, 0)
        counts = jnp.zeros(
            candidate_counts.shape[:-1]
            + (self.height, self.width, self.num_features),
            dtype=candidate_counts.dtype,
        )
        counts = counts.at[..., y, x, glyph].add(candidate_counts)
        numpyro.deterministic("a", counts)
        return counts


def restrict_poisson_model(
    model: functools.partial,
    images: Array,
    *,
    min_distance: float = 6.0,
    num_locations: int = 4,
) -> tuple[functools.partial, Array]:
    """Return the observation-specific candidate model and its site indices."""
    if not isinstance(model, functools.partial):
        raise TypeError("restrict_poisson_model expects a configured partial model")
    placements = model.keywords.get("placements")
    if not isinstance(placements, TexturedDiffeomorphicPoissonConvPlacements):
        raise TypeError(
            "the configured model needs textured diffeomorphic placements"
        )
    sites = candidate_indices(
        images,
        placements.shape_dict,
        min_distance=min_distance,
        num_locations=num_locations,
    )
    restricted = CandidateTexturedDiffeomorphicPoissonConvPlacements.from_placements(
        sites, placements
    )
    keywords = {**model.keywords, "placements": restricted}
    return functools.partial(model.func, *model.args, **keywords), sites
