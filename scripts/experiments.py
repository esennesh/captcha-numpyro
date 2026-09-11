"""Experiment drivers for the online MAP-proposal path.

Each subcommand runs one of the ladders in §26 of
``notes/poisson-convsc-design.md`` and writes a JSON record plus a summary
table. Every fit is scored the same way, so rows are comparable across
subcommands.

The score that has mattered most is **glyph evidence**: the fitted log joint
minus the log joint of the same model with the count field pinned at zero and
every other latent free. On ``0000_LJ`` the second pod run's whole glyph
placement was worth +58 nats by that measure, where the correct two glyphs are
worth +804. A run that recovers glyphs should show hundreds of nats.

Identity is scored automatically from the filename: ``0000_LJ.png`` contains an
``L`` and a ``J``, so ``identity_recall`` is how many of those two characters
appear among the top count coefficients.

Subcommands
-----------
``examples``
    Ladder A. Fit each example, optionally with the warp disabled. A1 is
    ``--group upright --warp-scale 0``: five images that are upright with an
    exact matched filter, where the count gradient at a full stamp is positive
    and no warp is needed. If recovery fails there, the joint-identification
    story is wrong and the rest of the ladder is misdirected.
``degrees-of-freedom``
    Ladders C1 and C3. Sweep the Gamma degrees-of-freedom floor and the
    statistic it follows. C1 is the ``--floors`` sweep; C3 is
    ``--sources count coverage``.
``warp-marginal``
    Ladder B2. Pin the warp at each of a few candidate global rotations and fit
    the counts under each, instead of asking one optimizer to find pose and
    placement together.
``prior-recovery``
    Ladder D1. Draw a latent from the model's own prior, render it, and fit
    that. Recovery of a *known* latent, which we have never checked.
"""

import argparse
import json
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import rootutils
import yaml
from numpyro import handlers
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)

from src.model.diffeomorphism import (  # noqa: E402
    boundary_taper,
    fit_coarse_velocity,
    rotation_velocity,
    translation_free_velocity,
)

# Rotation measured by correlating the dictionary against each image over
# rotations and scales; see notes/diffeomorphic-glyphs.md. The upright group
# gains nothing from rotating the template, so it needs no warp.
UPRIGHT = ("0000_N6", "0001_OO", "0002_J8", "0003_LN", "0004_V6")
ROTATED = ("0000_LJ", "0001_JA", "0002_QS", "0003_7Z", "0004_7D")


def load_image(path, shape):
    height, width = shape
    image = Image.open(path).convert("RGB").resize((width, height))
    return jnp.asarray(np.asarray(image, dtype=np.float32) / 255.0)[None]


MODEL_CONFIG = "model/poisson_convsc_gmrf"

# Filled from `--set key=value` before any subcommand runs. `compose` folds
# these in underneath the per-subcommand overrides, so a dedicated flag such as
# `--warp-scale` still wins over a `--set placements.warp_scale`.
BASE_OVERRIDES: dict = {}


def parse_setting(text):
    """``placements.require_on_canvas=false`` -> the key and a typed value.

    Values go through the YAML scalar parser, which is the one hydra itself
    uses, so ``false``, ``False``, ``1.5``, ``8`` and ``coverage`` all arrive as
    the type the config expects rather than as strings.
    """
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(
            f"--set needs key=value, got {text!r}"
        )
    return key, yaml.safe_load(value)


def compose(model_overrides=None, online_name="online/map_proposal_linear",
            model_name=None):
    with hydra.initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"
    ):
        model_config = hydra.compose(config_name=model_name or MODEL_CONFIG)
        online_config = hydra.compose(config_name=online_name)
    # Overrides may introduce keys the config leaves to the model function's
    # own defaults, so drop struct mode rather than requiring every knob to be
    # declared before it can be swept.
    OmegaConf.set_struct(model_config, False)
    model_config.model.placements.shape_dict.path = str(
        PROJECT_ROOT / "data/captcha_sandbox/dictionary"
    )
    for key, value in {**BASE_OVERRIDES, **(model_overrides or {})}.items():
        node, _, leaf = key.rpartition(".")
        target = model_config.model
        for part in node.split(".") if node else []:
            target = target[part]
        target[leaf] = value
    model = hydra.utils.instantiate(model_config.model)
    inference = hydra.utils.instantiate(online_config.online)(model)
    return model, inference


def glyph_names(model):
    targets = model.keywords["placements"].shape_dict.targets
    return [str(targets[index]).split("_")[0] for index in sorted(targets)]


def truth_from_stem(stem):
    """``0000_LJ`` -> ``('L', 'J')``. The filename carries the answer."""
    _, _, characters = stem.partition("_")
    return tuple(characters)


def visible_fraction(model, y, x):
    """Fraction of a glyph's frame that lands on the canvas, centred at (y, x).

    ``_stamp`` crops, so a centre within half a kernel of an edge renders only a
    sliver of the glyph -- and slivers of different letters are nearly
    interchangeable, which makes them a cheap generic ink primitive rather than
    a glyph. Ladder A3 found 21 of 30 top peaks below 75% visible, many at
    exactly 50%, on the run with the *highest* glyph evidence and the worst
    identity. So this is reported alongside the evidence: it is the detector for
    evidence bought by painting with half-glyphs.
    """
    placements = model.keywords["placements"]
    height, width = placements.ink_kernel.shape[:2]
    half_y, half_x = (height - 1) // 2, (width - 1) // 2
    rows = min(y + height - half_y, placements.height) - max(y - half_y, 0)
    columns = min(x + width - half_x, placements.width) - max(x - half_x, 0)
    return max(rows, 0) * max(columns, 0) / (height * width)


def summarize(result, model, image, truth=None, blank_log_joint=None, top=6):
    names = glyph_names(model)
    counts = np.asarray(result.map_values["a"])
    order = np.argsort(counts.ravel())[::-1][:top]
    # The count field is not the image grid under `require_on_canvas`, so shift
    # peaks into image coordinates before reporting them.
    top_offset, left_offset = model.keywords["placements"].count_origin
    peaks = []
    for rank in order:
        _, y, x, k = np.unravel_index(rank, counts.shape)
        y, x = int(y) + top_offset, int(x) + left_offset
        peaks.append(
            {"glyph": names[k], "y": y, "x": x,
             "count": round(float(counts.ravel()[rank]), 4),
             "visible": round(visible_fraction(model, y, x), 3)}
        )
    observed = np.asarray(image[0])
    reconstruction = np.asarray(result.weighted_reconstruction)[0]
    losses = np.asarray(result.map_losses)
    record = {
        "log_joint": round(-float(result.map_losses[-1]), 1),
        # The optimizer returns its LAST iterate, not its best, and Adam at a
        # fixed step size does not converge to a point -- it wanders. On the
        # C3-off grid the trajectory has an interior optimum and walks away
        # from it: `0000_LJ` reached 39009.8 by step 640 and ended at 38309.6
        # at 1500. Early stopping used to cut the run near the optimum by
        # accident, so removing it exposed the overshoot rather than causing
        # it. Record the shape of the curve so a budget or a step-size
        # schedule can be chosen by inspection.
        "log_joint_best": round(-float(np.min(losses)), 1),
        "map_loss_argmin": int(np.argmin(losses)),
        "map_loss_first": round(float(losses[0]), 1),
        "map_loss_curve": [
            round(float(v), 1) for v in losses[:: max(1, len(losses) // 30)]
        ],
        "map_steps": int(result.map_num_steps),
        "map_converged": bool(result.map_converged),
        "losses_finite": bool(np.isfinite(result.map_losses).all()
                              and np.isfinite(result.dispersion_losses).all()),
        "effective_sample_size": round(float(result.effective_sample_size), 3),
        "count_total": round(float(counts.sum()), 4),
        "count_max": round(float(counts.max()), 4),
        "coefficients": int((counts > 1e-4).sum()),
        "exact_zeros": int((counts == 0.0).sum()),
        "rmse": round(float(np.sqrt(((observed - reconstruction) ** 2).mean())), 4),
        "peak_visible_area": round(
            float(np.mean([peak["visible"] for peak in peaks])), 3
        ),
        "peaks": peaks,
    }
    if blank_log_joint is not None:
        record["blank_log_joint"] = round(blank_log_joint, 1)
        record["glyph_evidence"] = round(record["log_joint"] - blank_log_joint, 1)
    if truth is not None:
        record["truth"] = "".join(truth)
        # Where does the truth's own class actually rank? Top-k membership
        # conflates "the model has no idea" with "the model knows but ranks
        # fragments of other classes above it", and ladder A2 turned out to be
        # the second: with the warp on, three to five distinct classes appear in
        # the top six and the true class is present but outranked and displaced.
        ranking = np.argsort(counts.ravel())[::-1]
        classes = np.asarray(names)[
            np.unravel_index(ranking, counts.shape)[3]
        ]
        record["truth_rank"] = [
            int(np.flatnonzero(classes == character)[0])
            if (classes == character).any() else None
            for character in dict.fromkeys(truth)
        ]
        # Multiset match, so a repeated character can score twice: `0001_OO`
        # holds two `O`s, and two `O` peaks is a full recovery rather than a
        # half one. Each peak may satisfy at most one truth character.
        available = [peak["glyph"] for peak in peaks]
        matched = 0
        for character in truth:
            if character in available:
                available.remove(character)
                matched += 1
        record["identity_recall"] = matched
        record["identity_total"] = len(truth)
    if "paper_logit" in result.samples:
        particle = int(np.argmax(np.asarray(result.normalized_weights)))
        logit = np.asarray(result.samples["paper_logit"][particle, 0])
        record["paper"] = [round(float(v), 3) for v in 1.0 / (1.0 + np.exp(-logit))]
        # `paper` is the global logit alone. The rendered background is that
        # logit plus the coarse `paper_field`, which has a marginal standard
        # deviation of 2 in logit space and can carry a tint of its own, so the
        # scalar on its own cannot say whether the background came out right.
        # Record what the model actually paints.
        background = model.keywords.get("backgrounder")
        if background is not None and "paper_field" in result.samples:
            field = np.asarray(
                jax.nn.sigmoid(
                    logit
                    + np.asarray(
                        background.resize_field(
                            result.samples["paper_field"][particle, 0]
                        )
                    )
                )
            )
            record["paper_rendered"] = [
                round(float(v), 3) for v in field.reshape(-1, field.shape[-1]).mean(0)
            ]
        # The paper in these examples is flat, so its median, 90th percentile
        # and modal colour agree to three decimals on all ten images; the
        # median is therefore the true paper and not merely a robust summary.
        record["paper_truth"] = [
            round(float(v), 3) for v in np.median(observed.reshape(-1, 3), axis=0)
        ]
        record["paper_error"] = [
            round(float(a - b), 3)
            for a, b in zip(
                record.get("paper_rendered", record["paper"]),
                record["paper_truth"],
            )
        ]
    return record


def map_log_joint(inference, target, image, condition=None, seed=0):
    """Run only the MAP phase and return the log joint it reaches.

    This builds the guide directly rather than going through
    ``MAPProposalCaptchaInference.__call__``, because conditioning the count
    site makes it observed and the importance stage then has no ``samples["a"]``
    to weight. A reference log joint needs no particles anyway, and skipping
    them roughly halves the cost.
    """
    if condition is not None:
        target = handlers.condition(target, condition)
    guide = inference.guide_class(
        target,
        discrete_temperature=inference.discrete_temperature,
        dsgd_kwargs=inference.dsgd_kwargs,
        init_dispersion=inference.init_dispersion,
        init_loc_fn=inference.init_loc_fn,
        map_max_steps=inference.map_max_steps,
        map_optimizer=inference.map_optimizer,
        map_tolerance=inference.map_tolerance,
        num_dispersion_particles=inference.num_dispersion_particles,
        proposal_max_steps=1,
        proposal_optimizer=inference.proposal_optimizer,
        proposal_tolerance=inference.proposal_tolerance,
        termination_check_interval=inference.termination_check_interval,
        termination_patience=inference.termination_patience,
    )
    fitted = guide.fit(jax.random.key(seed), image)
    return -float(fitted.map_result.losses[-1])


def blank_reference(inference, model, image, seed=0):
    """Log joint with the count field pinned at zero and everything else free.

    Conditions ``inference.model`` rather than ``model``, so any conditioning
    the caller has already applied -- a pinned warp, in ladder B2 -- is shared
    by the reference and the fit it is compared against.
    """
    placements = model.keywords["placements"]
    zeros = jnp.zeros((1, *placements.count_shape, placements.num_features))
    return map_log_joint(inference, inference.model, image, {"a": zeros}, seed)


def run_fit(inference, model, image, truth=None, seed=0, reference=True):
    blank = blank_reference(inference, model, image, seed) if reference else None
    start = time.time()
    result = inference(jax.random.key(seed), image)
    record = summarize(result, model, image, truth=truth, blank_log_joint=blank)
    record["seconds"] = round(time.time() - start, 1)
    return record


def print_rows(rows, keys):
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows))
              for key in keys}
    print("  " + "  ".join(key.ljust(widths[key]) for key in keys), flush=True)
    for row in rows:
        print("  " + "  ".join(
            str(row.get(key, "")).ljust(widths[key]) for key in keys), flush=True)


def write(records, output):
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    print("\nwrote %s" % output, flush=True)


# --------------------------------------------------------------------------
# Ladder A: the example sweep
# --------------------------------------------------------------------------


def experiment_examples(arguments):
    stems = {"upright": UPRIGHT, "rotated": ROTATED,
             "all": UPRIGHT + ROTATED}[arguments.group]
    overrides = {}
    if arguments.warp_scale is not None:
        overrides["placements.warp_scale"] = arguments.warp_scale
    records = []
    for stem in stems:
        model, inference = compose(overrides)
        placements = model.keywords["placements"]
        image = load_image(
            PROJECT_ROOT / f"data/examples/{stem}.png",
            (placements.height, placements.width),
        )
        print("\n=== %s (warp_scale %s) ===" % (stem, arguments.warp_scale), flush=True)
        record = run_fit(inference, model, image, truth=truth_from_stem(stem),
                         seed=arguments.seed, reference=not arguments.no_reference)
        record.update(image=stem, warp_scale=arguments.warp_scale)
        records.append(record)
        print("  %s" % json.dumps({k: record[k] for k in (
            "log_joint", "glyph_evidence", "identity_recall", "count_total",
            "rmse") if k in record}), flush=True)
        print("  peaks: %s" % "  ".join(
            "%s@(%d,%d)=%.3f" % (p["glyph"], p["y"], p["x"], p["count"])
            for p in record["peaks"]), flush=True)
    print("\n=== summary ===")
    # `map_converged` is False by construction now that early stopping is off,
    # so its column is spent on the overshoot instead: `log_joint_best` against
    # `log_joint` says how much the last iterate gave back, and
    # `map_loss_argmin` says where the trajectory actually turned.
    print_rows(records, ["image", "glyph_evidence", "identity_recall",
                         "identity_total", "truth", "truth_rank",
                         "peak_visible_area", "count_total", "coefficients",
                         "rmse", "log_joint", "log_joint_best",
                         "map_loss_argmin", "map_steps"])
    write(records, arguments.output)


# --------------------------------------------------------------------------
# Ladders C1 and C3: the Gamma degrees of freedom
# --------------------------------------------------------------------------


def experiment_degrees_of_freedom(arguments):
    records = []
    for source in arguments.sources:
        for floor in arguments.floors:
            overrides = {"gamma_df_floor": float(floor),
                         "gamma_df_source": source}
            if arguments.warp_scale is not None:
                overrides["placements.warp_scale"] = arguments.warp_scale
            model, inference = compose(overrides)
            placements = model.keywords["placements"]
            image = load_image(
                PROJECT_ROOT / f"data/examples/{arguments.image}.png",
                (placements.height, placements.width),
            )
            print("\n=== nu = %.1f + %s ===" % (floor, source), flush=True)
            record = run_fit(inference, model, image,
                             truth=truth_from_stem(arguments.image),
                             seed=arguments.seed,
                             reference=not arguments.no_reference)
            record.update(gamma_df_floor=float(floor), gamma_df_source=source,
                          image=arguments.image)
            records.append(record)
            print("  glyph evidence %s   identity_recall %s   rmse %s"
                  % (record.get("glyph_evidence"), record.get("identity_recall"),
                     record["rmse"]), flush=True)
    print("\n=== summary ===")
    print_rows(records, ["gamma_df_source", "gamma_df_floor", "glyph_evidence",
                         "identity_recall", "count_total", "count_max", "rmse",
                         "log_joint"])
    write(records, arguments.output)


# --------------------------------------------------------------------------
# Ladder B2: the warp-marginal sweep
# --------------------------------------------------------------------------


def experiment_warp_marginal(arguments):
    overrides = {}
    model, inference = compose(overrides)
    placements = model.keywords["placements"]
    shape = (placements.height, placements.width)
    image = load_image(
        PROJECT_ROOT / f"data/examples/{arguments.image}.png", shape
    )
    prior = placements.velocity_prior()
    window = boundary_taper(shape)
    coarse = (placements.warp_coarse_height, placements.warp_coarse_width)
    records = []
    centres = [tuple(float(v) for v in pair.split(","))
               for pair in arguments.centres] or [None]
    for degrees in arguments.degrees:
        if degrees == 0:
            velocity = jnp.zeros((*coarse, 2))
        else:
            # A rotation about the image centre is expensive -- about 1,000
            # nats at 20 degrees -- because the boundary is pinned and the
            # field has to turn the whole sheet. Local bumps at candidate glyph
            # centres cost far less: the two-glyph field that fits 0000_LJ's
            # actual deformation costs 42 nats and buys +3,223. `--sigma` with
            # `--centres` selects that form.
            target = sum(
                rotation_velocity(shape, float(degrees), centre=centre,
                                  sigma=arguments.sigma)
                for centre in centres
            )
            velocity = fit_coarse_velocity(
                target, coarse, translation_free_velocity,
                prior.solve_precision, scale=placements.warp_scale,
                window=window,
            )
        cost = -float(prior.log_prob(velocity)
                      - prior.log_prob(jnp.zeros((*coarse, 2))))
        print("\n=== warp pinned at %+d degrees (prior cost %.1f nats) ==="
              % (degrees, cost), flush=True)
        model, inference = compose(overrides)
        inference.model = handlers.condition(
            model, {"warp_velocity": velocity[jnp.newaxis]}
        )
        record = run_fit(inference, model, image,
                         truth=truth_from_stem(arguments.image),
                         seed=arguments.seed,
                         reference=not arguments.no_reference)
        record.update(degrees=int(degrees), warp_prior_cost=round(cost, 1),
                      image=arguments.image)
        records.append(record)
        print("  glyph evidence %s   identity_recall %s   peaks: %s"
              % (record.get("glyph_evidence"), record.get("identity_recall"),
                 "  ".join("%s@(%d,%d)=%.3f"
                           % (p["glyph"], p["y"], p["x"], p["count"])
                           for p in record["peaks"][:3])), flush=True)
    print("\n=== summary ===")
    print_rows(records, ["degrees", "warp_prior_cost", "glyph_evidence",
                         "identity_recall", "count_total", "rmse", "log_joint"])
    best = max(records, key=lambda row: row["log_joint"])
    print("\nbest pose: %+d degrees, log joint %s, identity_recall %s"
          % (best["degrees"], best["log_joint"], best.get("identity_recall")))
    write(records, arguments.output)


# --------------------------------------------------------------------------
# Ladder D1: recovery of the model's own prior draw
# --------------------------------------------------------------------------


def experiment_prior_recovery(arguments):
    records = []
    for draw in range(arguments.draws):
        overrides = {}
        if arguments.warp_scale is not None:
            overrides["placements.warp_scale"] = arguments.warp_scale
        model, inference = compose(overrides)
        placements = model.keywords["placements"]
        names = glyph_names(model)
        prior_trace = handlers.trace(
            handlers.seed(model, jax.random.PRNGKey(1000 + draw))
        ).get_trace(None)
        image = jnp.clip(prior_trace["obs"]["value"], 0.0, 1.0)
        true_counts = np.asarray(prior_trace["a"]["value"])
        origin_y, origin_x = placements.count_origin
        placed = [
            {"glyph": names[k], "y": int(y) + origin_y,
             "x": int(x) + origin_x,
             "count": int(true_counts[0, y, x, k])}
            for y, x, k in zip(*np.nonzero(true_counts[0])[:3])
        ] if true_counts.sum() else []
        print("\n=== prior draw %d: %d spikes at %s ==="
              % (draw, int(true_counts.sum()),
                 "  ".join("%s@(%d,%d)x%d" % (p["glyph"], p["y"], p["x"], p["count"])
                           for p in placed) or "nothing"), flush=True)
        if not placed and not arguments.keep_empty:
            print("  skipped: the prior drew no glyphs", flush=True)
            continue
        record = run_fit(inference, model, image,
                         truth=tuple(p["glyph"] for p in placed),
                         seed=arguments.seed,
                         reference=not arguments.no_reference)
        record.update(draw=draw, true_total=int(true_counts.sum()),
                      true_placed=placed)
        # positional accuracy: distance from each true spike to the nearest peak
        if placed:
            distances = []
            for true in placed:
                distances.append(min(
                    abs(true["y"] - peak["y"]) + abs(true["x"] - peak["x"])
                    for peak in record["peaks"]))
            record["nearest_peak_l1"] = distances
        records.append(record)
        print("  glyph evidence %s   identity_recall %s / %d   nearest peak L1 %s"
              % (record.get("glyph_evidence"), record.get("identity_recall"),
                 len(set(record["truth"])), record.get("nearest_peak_l1")),
              flush=True)
    print("\n=== summary ===")
    if records:
        print_rows(records, ["draw", "true_total", "glyph_evidence",
                             "identity_recall", "nearest_peak_l1", "count_total",
                             "rmse", "log_joint"])
    write(records, arguments.output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--model-config", default=MODEL_CONFIG,
        help="hydra config name for the model, e.g. model/poisson_convsc_pose "
             "for the discrete-pose variant",
    )
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument(
        "--set", dest="settings", default=[], action="append", type=parse_setting,
        metavar="KEY=VALUE",
        help="override a model config key, e.g. "
             "--set placements.require_on_canvas=false. Keys are relative to "
             "the `model` node, dotted for nesting, and every subcommand "
             "honours them.",
    )
    parser.add_argument(
        "--no-reference", action="store_true",
        help="skip the pinned-at-zero count fit that glyph_evidence needs, "
             "halving the runtime",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    examples = subparsers.add_parser("examples", help="ladder A")
    examples.add_argument("--group", default="upright",
                          choices=("upright", "rotated", "all"))
    examples.add_argument("--warp-scale", default=None, type=float)
    examples.set_defaults(function=experiment_examples)

    degrees = subparsers.add_parser("degrees-of-freedom", help="ladders C1, C3")
    degrees.add_argument("--image", default="0000_LJ")
    degrees.add_argument("--floors", default=[1.0, 2.0, 4.0, 8.0], nargs="+",
                         type=float)
    degrees.add_argument("--sources", default=["count"], nargs="+",
                         choices=("count", "coverage"))
    degrees.add_argument("--warp-scale", default=None, type=float)
    degrees.set_defaults(function=experiment_degrees_of_freedom)

    warp = subparsers.add_parser("warp-marginal", help="ladder B2")
    warp.add_argument("--image", default="0000_LJ")
    warp.add_argument("--degrees", default=[-30, -20, -10, 0, 10, 20, 30],
                      nargs="+", type=int)
    warp.add_argument(
        "--sigma", default=None, type=float,
        help="fade each rotation out over this radius in pixels, making it a "
             "local rotation about --centres instead of a global one. About 13 "
             "is the glyph half-height. A global rotation costs ~1000 nats at "
             "20 degrees; a two-glyph local field costs ~42.",
    )
    warp.add_argument(
        "--centres", default=[], nargs="+", metavar="Y,X",
        help="rotation centres for --sigma, e.g. 42,25 42,57 -- the layout the "
             "sandbox captchas use. Ignored without --sigma.",
    )
    warp.set_defaults(function=experiment_warp_marginal)

    recovery = subparsers.add_parser("prior-recovery", help="ladder D1")
    recovery.add_argument("--draws", default=4, type=int)
    recovery.add_argument("--warp-scale", default=None, type=float)
    recovery.add_argument("--keep-empty", action="store_true",
                          help="fit draws in which the prior placed no glyphs")
    recovery.set_defaults(function=experiment_prior_recovery)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    globals()["MODEL_CONFIG"] = arguments.model_config
    BASE_OVERRIDES.update(dict(arguments.settings or []))
    if BASE_OVERRIDES:
        print("model overrides: %s" % BASE_OVERRIDES, flush=True)
    arguments.function(arguments)
