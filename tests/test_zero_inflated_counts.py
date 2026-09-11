"""Checks for the zero-inflated count field: occurrence priced apart from opacity."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rootutils
from flax import nnx

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpyro.distributions as dist
from numpyro import handlers
from numpyro.contrib.diag_sgd import SmoothedCount

from src.data.dictionary import ShapeDictionary
from src.inference.linear_count_proposal import (
    LinearCountMAPProposal,
    count_block_adam,
)
from src.inference.map_proposal_online import (
    MAPProposalCaptchaInference,
    init_to_count_mean,
)
from src.model.model import (
    TexturedDiffeomorphicPoissonConvPlacements,
    poisson_convsc_model,
)

SIDE = 12
ETA = 0.1  # the discrete_temperature configs/online/map_proposal.yaml uses


def make_placements(side=SIDE, **kwargs):
    glyph = jnp.zeros((5, 5, 4)).at[1:4, 2, :].set(1.0)
    dictionary = ShapeDictionary(shapes=glyph[jnp.newaxis], targets={"glyph": 0})
    return TexturedDiffeomorphicPoissonConvPlacements(
        dictionary, cg_iters=20, expected_count=1.0, img_h=side, img_w=side,
        rngs=nnx.Rngs(0), warp_coarse_height=2, warp_coarse_width=2,
        warp_scale=1.0, **kwargs,
    )


def make_image(side=SIDE):
    return jnp.ones((1, side, side, 3)).at[:, 4:8, 6, :].set(0.1)


def test_the_two_priors_are_calibrated_to_the_same_expected_total():
    """Only then is a comparison between them about shape, not about scale."""
    plain = make_placements()
    inflated = make_placements(count_zero_inflated=True)
    image = make_image()

    totals = {}
    for name, placements in (("plain", plain), ("inflated", inflated)):
        model = functools.partial(poisson_convsc_model, placements=placements,
                                  likelihood="gmrf")
        model_trace = handlers.trace(
            handlers.seed(model, jax.random.PRNGKey(0))
        ).get_trace(image, plot_mean=True)
        totals[name] = float(jnp.sum(model_trace["a"]["fn"].mean))
    np.testing.assert_allclose(totals["plain"], 1.0, rtol=1e-4)
    np.testing.assert_allclose(totals["inflated"], 1.0, rtol=1e-3)


def test_initial_gate_logits_matches_the_closed_form():
    placements = make_placements(count_zero_inflated=True)
    firing = 1.0 / (placements.num_sites * placements.count_base_rate)
    np.testing.assert_allclose(
        placements.initial_gate_logits,
        math.log1p(-firing) - math.log(firing),
        rtol=1e-9,
    )
    # a higher base rate means each occurrence carries more, so fewer fire
    denser = make_placements(count_zero_inflated=True, count_base_rate=4.0)
    assert denser.initial_gate_logits > placements.initial_gate_logits


def test_a_firing_probability_outside_the_unit_interval_is_rejected():
    """`expected_count` above what the sites can carry has no gate."""
    placements = make_placements(count_zero_inflated=True, count_base_rate=1e-4)
    with pytest.raises(ValueError, match="firing probability"):
        placements.initial_gate_logits
    with pytest.raises(ValueError, match="count_base_rate"):
        make_placements(count_base_rate=0.0)


def test_zero_inflation_reverses_the_prior_s_preference_for_a_smear():
    """The load-bearing claim, at the temperature the configs actually use.

    The plain Poisson's relaxed price is near-linear in the amplitude, so it is
    indifferent to how a total is spread and in fact pays a little to spread
    it. Zero inflation puts a price on *existing*, which a smear pays once per
    coefficient.
    """
    sites, total, smear = 230400, 2.0, 1682
    plain = SmoothedCount(
        dist.Poisson(jnp.asarray(1.0 / sites)), eta=ETA, width=4, max_count=8
    )
    inflated = SmoothedCount(
        dist.ZeroInflatedPoisson(gate=jnp.asarray(1.0 - 1.0 / sites),
                                 rate=jnp.asarray(1.0)),
        eta=ETA, width=4, max_count=8,
    )

    def advantage(distribution):
        """How much better a smear scores than two committed glyphs."""
        density = lambda a: float(distribution.log_prob(jnp.asarray(a)))
        spread = smear * density(total / smear) + (sites - smear) * density(0.0)
        committed = 2 * density(1.0) + (sites - 2) * density(0.0)
        return spread - committed

    # the plain prior PAYS for the diffuse solution
    assert advantage(plain) > 0.0
    # zero inflation reverses that, and by a wide margin
    assert advantage(inflated) < -50.0


def test_the_barrier_stays_under_what_the_likelihood_can_supply():
    """A glyph has to be able to climb off zero.

    The likelihood supplies +51 to +174 nats per unit at a matched stamp
    (section 28 B2), so a barrier steeper than that pins every coefficient at
    zero however good the evidence.
    """
    inflated = SmoothedCount(
        dist.ZeroInflatedPoisson(gate=jnp.asarray(1.0 - 1.0 / 230400.0),
                                 rate=jnp.asarray(1.0)),
        eta=ETA, width=4, max_count=8,
    )
    barrier = float(
        -jax.grad(lambda a: inflated.log_prob(a))(jnp.asarray(1e-3))
    )
    assert 0.0 < barrier < 174.0


def test_the_model_traces_and_carries_finite_gradients():
    from numpyro.infer.util import log_density

    placements = make_placements(count_zero_inflated=True)
    model = functools.partial(poisson_convsc_model, placements=placements,
                              likelihood="gmrf")
    image = make_image()
    model_trace = handlers.trace(
        handlers.seed(model, jax.random.PRNGKey(0))
    ).get_trace(image, plot_mean=True)
    assert model_trace["a"]["value"].shape == (1, *placements.count_shape, 1)
    for name, site in model_trace.items():
        if site["type"] == "sample":
            assert np.isfinite(site["fn"].log_prob(site["value"])).all(), name

    values = {name: site["value"] for name, site in model_trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    relaxed = dict(values, a=values["a"].astype(jnp.float32) + 0.3)
    gradients = jax.grad(
        lambda v: log_density(model, (image,), {}, v)[0]
    )(relaxed)
    for name, gradient in gradients.items():
        assert np.isfinite(np.asarray(gradient)).all(), name
    assert np.abs(np.asarray(gradients["a"])).max() > 0.0


def test_a_fit_runs_end_to_end_and_still_reaches_exact_zeros():
    side = 9
    placements = make_placements(side=side, count_zero_inflated=True)
    inference = MAPProposalCaptchaInference(
        functools.partial(poisson_convsc_model, placements=placements),
        dsgd_kwargs={"max_count": 8, "width": 4},
        guide_class=LinearCountMAPProposal,
        init_loc_fn=init_to_count_mean,
        map_max_steps=60,
        map_optimizer=count_block_adam(count_step_size=1e-2, step_size=1e-2),
        num_dispersion_particles=2,
        num_importance_samples=3,
        proposal_max_steps=5,
        termination_check_interval=60,
    )
    result = inference(jax.random.key(3), make_image(side))
    counts = np.asarray(result.map_values["a"])
    assert counts.min() >= 0.0
    assert (counts == 0.0).sum() > 0
    assert np.isfinite(result.map_losses).all()
    assert np.isfinite(result.dispersion_losses).all()


def test_the_plain_path_is_untouched():
    """`count_zero_inflated=False` must still build a plain Poisson site."""
    placements = make_placements()
    model = functools.partial(poisson_convsc_model, placements=placements,
                              likelihood="gmrf")
    model_trace = handlers.trace(
        handlers.seed(model, jax.random.PRNGKey(0))
    ).get_trace(make_image(), plot_mean=True)
    assert isinstance(model_trace["a"]["fn"].base_dist, dist.Poisson)
    assert "log_rate" in {
        name for name, site in model_trace.items() if site["type"] == "param"
    }
