"""Fit one subject's recorded choices and RTs with the compiled PEC GPU sampler.

CMA-ES fits an observation-conditioned particle likelihood with retained control state.
The legacy trial-marginal histogram objective requires --likelihood marginal.
The recovery entry point uses this same pipeline with synthetic observations.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import psyneulink as pnl
import torch

from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.globals.utilities import set_global_seed
from dawa_batched_simulation import SOURCE, build_model, fit_surface, node
from dawa_adaptive_fit import AdaptiveConfig, fit_adaptive
from psyneulink.core.components.functions.nonstateful import adaptivefit
from psyneulink.core.components.functions.nonstateful.adaptivefit import StagedConfig
from dawa_ndt_profile import NDTProfile


# Coordinates follow fit_surface(): response threshold, NDT, shared bias,
# control gain, LC mode for PrevCongruency 0 and 1, LC scaling, and LC base gain.
# Only LC mode expands into two coordinates for a single-subject fit.
TRUTH = (0.40, 0.22, -0.40, 12.0, 0.65, 0.80, 1.5, 5.5)
STARTS = (
    (0.475, 0.20, -0.25, 12.5, 0.50, 0.50, 2.5, 6.5),
    (0.35, 0.18, -0.35, 9.0, 0.35, 0.65, 2.0, 6.0),
)
# Grids describe the seven model parameters before conditional expansion.
GRID_STEPS = (0.001, 0.0001, 0.001, 0.01, 0.001, 0.001, 0.001)
LAUNCH = dict(
    block_size=32,
    num_warps=1,
    trial_schedule="independent",
    normal_rng="philox4x_fast_v1",
)
RT_RANGE = (0.0, 3.0)
RT_BIN_WIDTH = 0.03


def histogram_settings(frame, likelihood):
    """Cover every conditioned observation without changing the RT resolution."""
    bins = 100
    if likelihood == "conditioned":
        # Masked trials still condition the next control-state distribution. Their
        # RTs therefore need support even though they do not contribute a log score.
        bins = max(bins, int(np.ceil(frame.response_time.max() / RT_BIN_WIDTH)))
    return bins, (RT_RANGE[0], bins * RT_BIN_WIDTH)


def make_fit_pec(model, inputs, outputs, frame, args):
    """Use one PEC objective for optimization and independent-seed rescoring."""
    observed = frame[
        ["decision", "response_time", "subject_nr", "PrevCongruency"]
    ].copy()
    for name in ("decision", "subject_nr", "PrevCongruency"):
        observed[name] = pd.Categorical(
            observed[name], categories=[0.0, 1.0] if name == "decision" else None
        )
    surface = fit_surface(model)
    grids = {
        key: np.linspace(lo, hi, round((hi - lo) / step) + 1)
        for (key, (lo, hi, _)), step in zip(surface.items(), GRID_STEPS, strict=True)
    }
    depends = {
        key: "subject_nr"
        for key in surface
        if key[0] in ("termination_threshold", "gain", "slope")
    }
    depends[("intercept", node(model, "RT_GATE"))] = "subject_nr"
    depends[("mode", node(model, "LC"))] = "PrevCongruency"
    bins, rt_range = histogram_settings(frame, args.likelihood)
    pec = pnl.ParameterEstimationComposition(
        model=model,
        parameters=grids,
        depends_on=depends,
        outcome_variables=list(outputs),
        data=observed,
        likelihood_include_mask=frame.likelihood_include_mask.to_numpy(dtype=bool),
        optimization_function=pnl.PECOptimizationFunction(
            # Construct the PEC here; _run replaces this method with its CMA-ES study.
            method="differential_evolution",
            max_iterations=args.evaluations,
            batched_backend="triton",
            batched_max_steps=args.max_steps,
            batched_seed=args.simulation_seed,
            batched_strict_truncation=True,
            batched_bins=bins,
            batched_bin_range=[rt_range],
            batched_pseudocount=args.pseudocount,
            batched_smoothing_sigma=0.5,
            batched_categorical_cardinalities=[2],
            conditioned_likelihood=args.likelihood == "conditioned",
            batched_fused_likelihood=True,
            batched_specialize_fixed_parameters=True,
            batched_parameter_batch_size=args.population,
            batched_triton_launch_options=LAUNCH,
            fit_truncation="penalize",
        ),
        num_estimates=args.estimates,
        initial_seed=args.simulation_seed,
        same_seed_for_all_parameter_combinations=True,
    )
    return pec


def load_subject(path, subject, *, trials=None, recovery=False):
    """Preserve the selected design's order and masked trials; validate before fitting."""
    frame = pd.read_csv(path)
    design_columns = [
        "subject_nr",
        "PrevCongruency",
        "T1",
        "T2",
        "S1",
        "S2",
        "S3",
        "S4",
        "likelihood_include_mask",
    ]
    outcomes = [] if recovery else ["decision", "response_time"]
    missing = set(design_columns + outcomes) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
    frame = frame[(frame.subject_nr == subject) & frame.PrevCongruency.notna()].copy()
    if trials is not None:
        frame = frame.iloc[:trials].copy()
    if frame.empty:
        raise ValueError(f"No retained trials for subject_nr={subject}")
    # Keep optional design identifiers for provenance, but discard empirical
    # outcomes in recovery mode so they cannot enter generation or scoring.
    identifiers = [
        name
        for name in ("trial_within_subject", "row_id", "Congruency")
        if name in frame
    ]
    frame = frame[design_columns + identifiers + outcomes].reset_index(drop=True)
    for name in design_columns[1:] + outcomes:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(
                f"Column {name} must contain finite values on every retained trial"
            )
    if not frame.likelihood_include_mask.isin([0, 1]).all():
        raise ValueError("likelihood_include_mask must contain only 0/1 or booleans")
    frame["likelihood_include_mask"] = frame.likelihood_include_mask.astype(bool)
    if set(frame.PrevCongruency.unique()) != {0, 1}:
        raise ValueError(
            "The selected design must contain PrevCongruency levels 0 and 1"
        )
    scored = frame[frame.likelihood_include_mask]
    if set(scored.PrevCongruency.unique()) != {0, 1}:
        raise ValueError(
            "At least one scored trial is required for each PrevCongruency level"
        )
    if not recovery:
        if not frame.decision.isin([0, 1]).all():
            raise ValueError("decision must be 0 or 1 on every retained trial")
        if (frame.response_time <= 0).any():
            raise ValueError(
                "response_time must be positive and in seconds on every retained trial"
            )
        if not scored.response_time.between(*RT_RANGE).all():
            raise ValueError(
                "Scored response_time values must be in seconds within the 0–3 s histogram range; "
                "review the units and scoring mask"
            )
    return frame


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_validation(rows):
    """Summarize complete-filter scores and within-seed candidate differences.

    This averages log scores, never individual trial factors. The uncertainty
    describes repeated simulation on the same observations, not new subjects.
    """
    seeds = [row["seed"] for row in rows]
    if not rows or len(set(seeds)) != len(seeds):
        raise ValueError(
            "Validation requires distinct seeds and at least one complete evaluation"
        )
    keys = set(rows[0]) - {"seed"}
    if any(set(row) - {"seed"} != keys for row in rows):
        raise ValueError("Validation candidates must match across seeds")
    summary = {}
    for key in sorted(keys):
        values = np.asarray([row[key] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Validation scores must be finite")
        sd = float(values.std(ddof=1)) if len(values) > 1 else None
        summary[key] = {
            "mean": float(values.mean()),
            "sd": sd,
            "mc_standard_error": None if sd is None else sd / np.sqrt(len(values)),
        }
    return {
        "replicates": len(rows),
        "statistics": summary,
        "note": "Complete-run log scores; differences pair the same seed. Monte Carlo uncertainty only.",
    }


def synthetic_parameters(model, frame, coordinates):
    """Match the original fitting order, with modes for previous conditions 0/1."""
    if set(frame.PrevCongruency.unique()) != {0.0, 1.0}:
        raise ValueError(
            "This recovery design requires previous-congruency levels 0 and 1."
        )
    values = iter(coordinates)
    parameters = {}
    for parameter, mechanism in fit_surface(model):
        if parameter == "mode":
            modes = (next(values), next(values))
            value = BatchedTrialParameter(
                np.where(frame.PrevCongruency.to_numpy() == 0, *modes)
            )
        else:
            value = next(values)
        parameters[f"{mechanism.name}.{parameter}"] = value
    if next(values, None) is not None:
        raise ValueError("Unconsumed generating coordinates.")
    return parameters


def summarize_samples(values, frame):
    """Summarize joint choice/RT samples while preserving sequence generation."""
    result = {}
    for previous in (0, 1):
        mask = (
            frame.likelihood_include_mask.to_numpy(dtype=bool)
            & (frame.PrevCongruency == previous).to_numpy()
        )
        sample = values[mask].reshape(-1, 2)
        result[str(previous)] = {
            "count": len(sample),
            "choice_one_probability": float(sample[:, 0].mean()),
            "mean_rt": float(sample[:, 1].mean()),
            "rt_quantiles": np.quantile(sample[:, 1], [0.1, 0.5, 0.9]).tolist(),
        }
    return result


def recovery_observations(
    latent, *, observation_model, seed, estimates, pseudocount, device="cuda"
):
    """Apply the declared measurement law after generating a complete latent history.

    Observation noise changes the recorded choice/RT, never the simulated state
    entering the next trial. Overflow is an explicit error rather than silently
    truncating, clipping, or rejecting a generated history.
    """
    from dawa_conditioned_reference import noisy_observations, observation_edges

    if observation_model == "latent":
        return np.array(latent, copy=True), {"kind": "latent_choices_rt"}
    if observation_model != "conditioned":
        raise ValueError("Unknown synthetic observation model")
    edges = observation_edges(100, RT_RANGE, device=device)
    ratio = pseudocount / estimates
    observed = noisy_observations(
        np.asarray(latent),
        np.random.default_rng(seed),
        edges=edges,
        bins=100,
        rt_range=RT_RANGE,
        sigma=0.5,
        alpha_per_estimate=ratio,
    )
    return observed, {
        "kind": "binned_smoothed_uniform_contamination",
        "seed": seed,
        "bins": 100,
        "rt_range": list(RT_RANGE),
        "edges": edges.tolist(),
        "smoothing_sigma": 0.5,
        "alpha_per_estimate": ratio,
        "contamination_fraction": 200 * ratio / (1 + 200 * ratio),
        "representation": "bin_centers",
        "latent_history_unchanged": True,
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("dawa_conditioned_reference.py").read_bytes()
        ).hexdigest(),
    }


def main(argv=None, *, recovery=False):
    description = (
        "Generate a synthetic full-sequence subject and recover its parameters on the GPU."
        if recovery
        else __doc__
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data", type=Path, default=SOURCE.parent / "flanker_data_part1.csv"
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="Actual subject_nr value in the CSV (default: 1)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Prefix for smoke tests; default is the complete subject",
    )
    parser.add_argument(
        "--estimates",
        type=int,
        default=100000,
        help="Simulated trajectories per proposal (default: 100000)",
    )
    parser.add_argument(
        "--likelihood",
        choices=("conditioned", "marginal"),
        default="conditioned",
        help="Condition retained state on observed choices/RTs (default); marginal selects the legacy objective",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help="Pseudocount per joint cell at the fitting budget; uniform observation contamination in conditioned mode (default: 1)",
    )
    parser.add_argument(
        "--validation-estimates",
        type=int,
        default=1000000,
        help="Fresh-seed rescoring budget (default: 1000000). Pseudocount scales with this budget to preserve prior weight.",
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=5000,
        help="Total parameter proposals, not generations (default: 5000)",
    )
    parser.add_argument(
        "--fit-strategy",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="Fixed particle count (default), or adaptive budgets using the policy appropriate to the likelihood",
    )
    parser.add_argument(
        "--profile-ndt",
        action="store_true",
        help="Profile nondecision time with exact compiled counts (marginal adaptive fitting only)",
    )
    parser.add_argument(
        "--batch-sampling-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run independent adaptive NDT sampling blocks together (disable for timing comparisons)",
    )
    parser.add_argument(
        "--adaptive-min-estimates",
        type=int,
        help="Exploration particles for conditioned fits (default: 10000); minimum marginal race budget (default: 5000)",
    )
    parser.add_argument(
        "--adaptive-rank-tolerance",
        type=float,
        default=1.0,
        help="Marginal only: weighted rank-regret tolerance (not a confidence bound)",
    )
    parser.add_argument(
        "--adaptive-initial-blocks",
        type=int,
        default=4,
        help="Marginal only: initial blocks per race",
    )
    parser.add_argument(
        "--adaptive-search-evaluations",
        type=int,
        default=2000,
        help="Marginal only: maximum coarse-search proposals before refinement",
    )
    parser.add_argument("--adaptive-check-every", type=int, default=250)
    parser.add_argument("--adaptive-min-evaluations", type=int, default=1000)
    parser.add_argument("--adaptive-patience", type=int, default=2)
    parser.add_argument("--adaptive-progress-tolerance", type=float, default=0.25)
    parser.add_argument("--adaptive-refine-evaluations", type=int, default=600)
    parser.add_argument(
        "--adaptive-selection-repeats",
        "--adaptive-selection-blocks",
        type=int,
        default=3,
        help="Fresh reference-budget repetitions: average complete log scores (conditioned), pool densities (marginal)",
    )
    parser.add_argument("--adaptive-selection-candidates", type=int, default=8)
    parser.add_argument(
        "--adaptive-checkpoint-candidates",
        type=int,
        default=4,
        help="Conditioned only: recent candidates per reference-budget checkpoint",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=10,
        help="CMA-ES population and candidate batch size (default: 10)",
    )
    parser.add_argument(
        "--start",
        type=int,
        choices=(0, 1),
        default=0,
        help="Which of the two predefined starting points to use",
    )
    parser.add_argument("--optimizer-seed", type=int, default=101)
    parser.add_argument(
        "--optimizer-storage",
        choices=("journal", "memory"),
        default="memory",
        help="Memory (default) avoids per-proposal journal writes; evaluation logs are always saved. Journal does not enable automatic resume.",
    )
    parser.add_argument("--simulation-seed", type=int, default=29)
    if recovery:
        parser.add_argument("--data-seed", type=int, default=20260925)
        parser.add_argument("--observation-seed", type=int, default=20260926)
        parser.add_argument(
            "--observation-model",
            choices=("auto", "latent", "conditioned"),
            default="auto",
            help="Synthetic measurement law: auto matches the chosen likelihood; latent reproduces older studies",
        )
    else:
        parser.add_argument("--predictive-seed", type=int, default=21260925)
    parser.add_argument("--model-seed", type=int, default=29)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[91001, 91002, 91003, 91004, 91005],
        help="Distinct seeds reserved for final rescoring (default: 91001 91002 91003 91004 91005)",
    )
    parser.add_argument("--predictive-estimates", type=int, default=4096)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4000,
        help="Strict execution cap per trial (default: 4000, as in the conditioned pilot)",
    )
    parser.add_argument(
        "--source-revision", help="Commit of an isolated source snapshot without .git"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing directories are never overwritten",
    )
    args = parser.parse_args(argv)
    if recovery:
        if args.observation_model == "auto":
            args.observation_model = (
                "conditioned" if args.likelihood == "conditioned" else "latent"
            )
        if args.observation_model == "conditioned" and args.likelihood != "conditioned":
            parser.error(
                "The conditioned observation model requires --likelihood conditioned"
            )
    if args.likelihood == "conditioned" and args.profile_ndt:
        parser.error(
            "Observation-conditioned fitting does not support --profile-ndt: "
            "shifting NDT changes particle history. Fit NDT jointly with the other parameters."
        )
    if args.profile_ndt and args.fit_strategy != "adaptive":
        parser.error("--profile-ndt currently requires --fit-strategy adaptive")
    if args.adaptive_min_estimates is None:
        args.adaptive_min_estimates = (
            10000 if args.likelihood == "conditioned" else 5000
        )
    # The CLI exposes one adaptive strategy. Record its resolved implementation
    # in the manifest: conditioned filters use stages; marginal estimates can pool blocks.
    args.fit_policy = "fixed"
    strategy_config = None
    if args.fit_strategy == "adaptive":
        if args.likelihood == "conditioned":
            args.fit_policy = "staged"
            strategy_config = StagedConfig(
                search_estimates=args.adaptive_min_estimates,
                reference_estimates=args.estimates,
                refine_evaluations=args.adaptive_refine_evaluations,
                check_every=args.adaptive_check_every,
                min_evaluations=args.adaptive_min_evaluations,
                patience=args.adaptive_patience,
                progress_tolerance=args.adaptive_progress_tolerance,
                checkpoint_candidates=args.adaptive_checkpoint_candidates,
                selection_candidates=args.adaptive_selection_candidates,
                selection_repeats=args.adaptive_selection_repeats,
            )
        else:
            args.fit_policy = "block_racing"
            strategy_config = AdaptiveConfig(
                min_estimates=args.adaptive_min_estimates,
                max_estimates=args.estimates,
                rank_tolerance=args.adaptive_rank_tolerance,
                check_every=args.adaptive_check_every,
                min_evaluations=args.adaptive_min_evaluations,
                patience=args.adaptive_patience,
                progress_tolerance=args.adaptive_progress_tolerance,
                refine_evaluations=args.adaptive_refine_evaluations,
                initial_blocks=args.adaptive_initial_blocks,
                max_search_evaluations=args.adaptive_search_evaluations,
                selection_blocks=args.adaptive_selection_repeats,
                selection_candidates=args.adaptive_selection_candidates,
            )
        try:
            strategy_config.validate(args.evaluations, args.population)
        except ValueError as error:
            parser.error(str(error))
    if (
        min(
            args.estimates,
            args.evaluations,
            args.population,
            args.predictive_estimates,
            args.max_steps,
        )
        < 1
    ):
        parser.error("Counts must be positive")
    if args.validation_estimates is not None and args.validation_estimates < 1:
        parser.error("Validation estimates must be positive")
    if not np.isfinite(args.pseudocount) or args.pseudocount < 0:
        parser.error("Pseudocount must be finite and nonnegative")
    if args.trials is not None and args.trials < 2:
        parser.error("A prefix must contain at least two trials")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        parser.error(
            "Validation seeds must be distinct; repeated seeds are not independent repetitions"
        )
    independent_seeds = {args.simulation_seed}
    if recovery:
        independent_seeds.add(args.data_seed)
        if args.observation_model == "conditioned":
            if args.observation_seed in independent_seeds | set(args.validation_seeds):
                parser.error(
                    "Observation, generation, fitting, and validation seeds must be independent"
                )
            independent_seeds.add(args.observation_seed)
    if (recovery and args.data_seed == args.simulation_seed) or set(
        args.validation_seeds
    ) & independent_seeds:
        parser.error("Generation, fitting, and validation seeds must be independent")
    predictive_seed = args.data_seed + 1000000 if recovery else args.predictive_seed
    if predictive_seed in independent_seeds | set(args.validation_seeds):
        parser.error("Predictive, fitting, and validation seeds must be independent")
    try:
        frame = load_subject(
            args.data, args.subject, trials=args.trials, recovery=recovery
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and CUDA-enabled PyTorch are required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "status": "preparing",
        "mode": "recovery" if recovery else "empirical_fit",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    save_json(args.output / "manifest.json", manifest)
    try:
        _run(
            args, frame, recovery, independent_seeds, strategy_config, manifest, started
        )
    except BaseException as error:
        manifest.update(
            failed_phase=manifest["status"],
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        save_json(args.output / "manifest.json", manifest)
        raise


def _run(args, frame, recovery, independent_seeds, strategy_config, manifest, started):
    set_global_seed(args.model_seed)
    pnl.set_num_threads(8)
    torch.set_num_threads(8)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    noise = dict(c_noise=0.1, s_noise=0.1, d_noise=0.1, r_noise=0.1)
    model, inputs, outputs = build_model(trials=len(frame), **noise)
    inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
    inputs[node(model, "Stimulus Input")] = frame[["S1", "S2", "S3", "S4"]].to_numpy()
    if recovery:
        truth_parameters = synthetic_parameters(model, frame, TRUTH)
        generator = BatchedCompositionCompiler.compile(
            model, backend="triton", outputs=outputs, max_steps=args.max_steps
        )
        generated = generator.run(
            inputs,
            [truth_parameters],
            1,
            seed=args.data_seed,
            strict_truncation=True,
            triton_launch_options=LAUNCH,
        ).values[0, 0]
        if not np.all(np.isin(generated[..., 0], [0.0, 1.0])):
            raise AssertionError("Invalid generated choices")
        latent_frame = frame.copy()
        latent_frame[["decision", "response_time"]] = generated[:, 0, :]
        latent_frame.to_csv(args.output / "latent_subject.csv", index=False)
        observed, observation_model = recovery_observations(
            generated[:, 0, :],
            observation_model=args.observation_model,
            seed=args.observation_seed,
            estimates=args.estimates,
            pseudocount=args.pseudocount,
        )
        frame[["decision", "response_time"]] = observed
    observations_path = args.output / (
        "synthetic_subject.csv" if recovery else "observed_subject.csv"
    )
    frame.to_csv(observations_path, index=False)
    pec = make_fit_pec(model, inputs, outputs, frame, args)
    function = pec.controller.function
    names = function.fit_param_names
    if len(names) != len(STARTS[args.start]):
        raise AssertionError(f"Expected eight fit coordinates, got {names}")
    bounds = function.fit_param_bounds
    plan = function._compile_batched_plan()
    indices = function._batched_outcome_indices(plan)
    if recovery:
        replay = plan.run(
            inputs,
            [function._batched_parameter_set(TRUTH)],
            1,
            seed=args.data_seed,
            strict_truncation=True,
            triton_launch_options=LAUNCH,
        ).values[0, 0][..., indices]
        np.testing.assert_array_equal(replay, generated)
    steps = {
        name: value
        for name, value in plan.fixed_parameters.items()
        if name.endswith(".time_step_size")
    }
    if len(steps) != 5 or any(
        value != (0.02 if name.startswith("LC.") else 0.01)
        for name, value in steps.items()
    ):
        raise AssertionError(f"Unexpected model clocks: {steps}")
    initial = dict(zip(names, STARTS[args.start], strict=True))
    for name, value in initial.items():
        lo, hi, step = bounds[name]
        if not lo <= value <= hi or not np.isclose(
            (value - lo) / step, round((value - lo) / step)
        ):
            raise AssertionError(
                f"Initial value is outside parameter grid: {name}={value}"
            )
    ndt_profile = (
        NDTProfile(plan, names, bounds, pec._data_numpy) if args.profile_ndt else None
    )
    optimizer_names = names if ndt_profile is None else ndt_profile.dynamic_names
    optimizer_initial = {name: initial[name] for name in optimizer_names}
    optimizer_bounds = {name: bounds[name] for name in optimizer_names}
    storage = (
        JournalStorage(JournalFileBackend(str(args.output / "optimizer.journal")))
        if args.optimizer_storage == "journal"
        else None
    )
    study = optuna.create_study(
        study_name="dawa_recovery" if recovery else "dawa_fit",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(
            x0=optimizer_initial,
            sigma0=0.2,
            lr_adapt=True,
            popsize=args.population,
            seed=args.optimizer_seed,
        ),
    )
    study.enqueue_trial(optimizer_initial)
    function.method = study
    manifest.update(
        {
            "status": "fitting",
            "mode": "recovery" if recovery else "empirical_fit",
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "git_commit": args.source_revision
            or subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[4],
                text=True,
            ).strip(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "entry_point": Path(sys.argv[0]).name,
            "source_model_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "design_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "observations_sha256": hashlib.sha256(
                observations_path.read_bytes()
            ).hexdigest(),
            "trials": len(frame),
            "scored_trials": int(frame.likelihood_include_mask.sum()),
            "initial": initial,
            "bounds": bounds,
            "noise": noise,
            "time_steps": steps,
            "lc_internal_steps_per_pass": 10,
            "estimator": {
                "kind": (
                    "observation_conditioned_particle_histogram"
                    if args.likelihood == "conditioned"
                    else "trial_marginal_histogram"
                ),
                "bins": function.batched_bins,
                "rt_range": list(function.batched_bin_range[0]),
                "smoothing_sigma": 0.5,
                "pseudocount": args.pseudocount,
                "observation_conditioned_history": args.likelihood == "conditioned",
                "masked_observations_condition_history": args.likelihood
                == "conditioned",
                **(
                    {
                        "resampling": "systematic",
                        "observation_model": "binned_smoothed_uniform_contamination",
                        "smoothing_normalization": "source_bin",
                        "contamination_fraction": (
                            2
                            * function.batched_bins
                            * args.pseudocount
                            / (
                                args.estimates
                                + 2 * function.batched_bins * args.pseudocount
                            )
                        ),
                    }
                    if args.likelihood == "conditioned"
                    else {}
                ),
            },
            "launch_options": LAUNCH,
            "gpu": torch.cuda.get_device_name(),
            "hostname": platform.node(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "data_summary": summarize_samples(
                frame[["decision", "response_time"]].to_numpy(), frame
            ),
            "setup_seconds": time.perf_counter() - started,
            "note": (
                "One subject; observed-history particle filtering under a binned, smoothed observation model. "
                "Masked observations condition history but do not add a log score. Budget completion is not convergence."
                if args.likelihood == "conditioned"
                else "Legacy trial-marginal fitting with unconditional simulated histories. Budget completion is not convergence."
            ),
            "fit_strategy": args.fit_strategy,
            "fit_policy": args.fit_policy,
        }
    )
    if args.fit_strategy == "adaptive":
        manifest["adaptive_config"] = asdict(strategy_config)
        policy_path = (
            Path(adaptivefit.__file__)
            if args.fit_policy == "staged"
            else Path(__file__).with_name("dawa_adaptive_fit.py")
        )
        manifest["adaptive_driver_sha256"] = hashlib.sha256(
            policy_path.read_bytes()
        ).hexdigest()
    if ndt_profile is not None:
        manifest["ndt_profile"] = ndt_profile.describe()
    if recovery:
        manifest.update(
            truth=dict(zip(names, TRUTH, strict=True)),
            generator_matches_pec_exactly=True,
            synthetic_sha256=manifest["observations_sha256"],
            synthetic_observation_model=observation_model,
            latent_observations_sha256=hashlib.sha256(
                (args.output / "latent_subject.csv").read_bytes()
            ).hexdigest(),
        )
    save_json(args.output / "manifest.json", manifest)
    optimization_started = time.perf_counter()
    records = []
    invalid = []
    sampling_work = (
        {}
        if args.fit_policy == "staged"
        else {"calls": 0, "kernel_launches": 0, "candidate_trajectories": 0}
    )

    def record_batch(candidates, scores, elapsed, metadata=None):
        if ndt_profile is not None:
            values = ndt_profile.values[np.asarray(metadata["profile_indices"])]
            candidates = ndt_profile.expand(candidates, values)
        with (args.output / "evaluations.jsonl").open("a") as stream:
            for candidate, score in zip(candidates, scores, strict=True):
                record = {
                    "evaluation": len(records) + 1,
                    "parameters": list(candidate),
                    "log_likelihood": float(score),
                    "elapsed_fit_seconds": time.perf_counter() - optimization_started,
                    "batch_seconds": elapsed,
                    "batch_size": len(candidates),
                    **(metadata or {}),
                }
                records.append(record)
                stream.write(json.dumps(record) + "\n")
        # Adaptive records mix particle budgets and seeds. This progress value
        # is diagnostic; the policy's reference checks and final selection choose the fit.
        best = max(records, key=lambda row: row["log_likelihood"])
        best_key = (
            "best"
            if args.fit_strategy == "fixed"
            else "best_search_record_not_final_selection"
        )
        progress = {
            "completed_evaluations": len(records),
            best_key: best,
            "fit_seconds": time.perf_counter() - optimization_started,
            "invalid_candidates": len(function.fit_diagnostics["invalid_proposals"])
            if function.fit_diagnostics is not None
            else len(invalid),
        }
        save_json(args.output / "progress.json", progress)
        print(json.dumps({"progress": progress}), flush=True)

    def check_densities(result, candidates, estimates, seed):
        expected_shape = (
            (len(candidates), 1, len(frame))
            if ndt_profile is None
            else (len(candidates), 1, len(ndt_profile.values), len(frame))
        )
        if result.shape != expected_shape:
            raise FloatingPointError("Unexpected adaptive density output shape")
        for index, candidate in enumerate(candidates):
            if np.isnan(result[index]).all():
                invalid.append(
                    {
                        "parameters": list(candidate),
                        "reason": "A simulation history exceeded max_steps",
                        "estimates": estimates,
                        "seed": seed,
                    }
                )
            elif not np.isfinite(result[index]).all():
                raise FloatingPointError("Unexpected nonfinite adaptive density")
        return result[:, 0]

    def sample_density_blocks(candidates, sizes, seeds):
        # Group equal sizes without changing seed order, the RNG addressing, or
        # the block weights used by the adaptive uncertainty calculation.
        parameter_sets = [
            function._batched_parameter_set(row)
            for row in ndt_profile.expand(candidates)
        ]
        results = [None] * len(sizes)
        for size in dict.fromkeys(sizes):
            positions = [i for i, n in enumerate(sizes) if n == size]
            counts = plan.discrete_output_count_blocks(
                inputs,
                parameter_sets,
                size,
                data=pec._data_numpy,
                categorical_dims=pec.data_categorical_dims,
                outcome_indices=indices,
                support=ndt_profile.support,
                seeds=[seeds[i] for i in positions],
                invalid_candidates="nan",
                triton_launch_options=LAUNCH,
            )
            sampling_work["calls"] += len(positions)
            sampling_work["kernel_launches"] += 1
            sampling_work["candidate_trajectories"] += (
                len(candidates) * size * len(positions)
            )
            sampling_work["max_count_buffer_bytes"] = max(
                sampling_work.get("max_count_buffer_bytes", 0),
                sum(c.counts.numel() * c.counts.element_size() for c in counts),
            )
            for i, count in zip(positions, counts, strict=True):
                results[i] = ndt_profile.scorer.densities(
                    count, pseudocount=args.pseudocount * size / args.estimates
                )
            del count, counts  # Release all views before allocating another group.
        return [
            check_densities(result, candidates, size, seed)
            for result, size, seed in zip(results, sizes, seeds, strict=True)
        ]

    def sample_densities(candidates, estimates, seed):
        if args.likelihood != "marginal":
            raise RuntimeError(
                "Independent adaptive density blocks require the explicit marginal objective"
            )
        if ndt_profile is not None:
            return sample_density_blocks(candidates, [estimates], [seed])[0]
        sampling_work["calls"] += 1
        sampling_work["kernel_launches"] += 1
        sampling_work["candidate_trajectories"] += len(candidates) * estimates
        result = plan.histogram_likelihood(
            inputs,
            [function._batched_parameter_set(row) for row in candidates],
            estimates,
            data=pec._data_numpy,
            categorical_dims=pec.data_categorical_dims,
            outcome_indices=indices,
            bins=100,
            bin_range=[RT_RANGE],
            smoothing_sigma=0.5,
            pseudocount=args.pseudocount * estimates / args.estimates,
            categorical_cardinalities=[2],
            seed=seed,
            invalid_candidates="nan",
            triton_launch_options=LAUNCH,
        )
        return check_densities(result, candidates, estimates, seed)

    strategy_report = None
    reserved_seeds = (set(args.validation_seeds) | independent_seeds) - {
        args.simulation_seed
    }
    reserved_seeds.add(args.data_seed + 1000000 if recovery else args.predictive_seed)
    if args.fit_strategy == "adaptive" and args.fit_policy == "block_racing":
        # The marginal block-racing policy remains a research driver for now.
        fit, strategy_report, refinement = fit_adaptive(
            study,
            optimizer_bounds,
            optimizer_initial,
            sample_densities,
            frame.likelihood_include_mask.to_numpy(dtype=bool),
            strategy_config,
            evaluations=args.evaluations,
            population=args.population,
            simulation_seed=args.simulation_seed,
            optimizer_seed=args.optimizer_seed,
            reserved_seeds=reserved_seeds,
            log_batch=record_batch,
            profile_parameter=None
            if ndt_profile is None
            else (ndt_profile.name, ndt_profile.values),
            sample_blocks=sample_density_blocks
            if ndt_profile is not None and args.batch_sampling_blocks
            else None,
        )
        strategy_report["policy"] = args.fit_policy
    else:
        function.fit_callback = record_batch
        if args.fit_strategy == "adaptive":
            function.fit_strategy = "adaptive"
            function.adaptive_options = {
                key: value
                for key, value in asdict(strategy_config).items()
                if key != "reference_estimates"
            }
            function.adaptive_options.update(
                optimizer_seed=args.optimizer_seed,
                reserved_seeds=sorted(reserved_seeds),
            )
        pec.run(inputs=inputs)
        fit = {
            "fitted_params": pec.optimized_parameter_values,
            "optimal_value": pec.optimal_value,
        }
        invalid[:] = function.fit_diagnostics["invalid_proposals"]
        sampling_work = function.fit_diagnostics["sampling_work"]
        if args.fit_strategy == "adaptive":
            strategy_report = {
                key: value
                for key, value in function.fit_diagnostics.items()
                if key not in ("invalid_proposals", "sampling_work")
            }
            refinement = function.refinement_study
    if strategy_report is not None:
        refinement.trials_dataframe(
            attrs=("number", "value", "params", "state")
        ).to_csv(args.output / "optimizer_refinement_trials.csv", index=False)
    fit_seconds = time.perf_counter() - optimization_started
    expected_evaluations = (
        args.evaluations
        if strategy_report is None
        else strategy_report["search_evaluations"]
        + strategy_report["refinement_evaluations"]
    )
    if len(records) != expected_evaluations:
        raise AssertionError("Optimizer did not evaluate the requested budget")
    fitted = np.asarray([fit["fitted_params"][name] for name in names])
    if float(fit["optimal_value"]) <= -1.0e9:
        raise RuntimeError("No valid fit found")
    np.testing.assert_allclose(
        [study.trials[0].params[name] for name in optimizer_names],
        list(optimizer_initial.values()),
        rtol=0,
        atol=1e-12,
    )
    study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "state",
            "datetime_start",
            "datetime_complete",
        )
    ).to_csv(args.output / "optimizer_trials.csv", index=False)
    if strategy_report is not None:
        save_json(
            args.output / f"{args.fit_strategy}.json",
            {**strategy_report, "sampling_work": sampling_work},
        )
    save_json(
        args.output / "fit_checkpoint.json",
        {
            "status": "search_complete",
            "mode": manifest["mode"],
            "subject": args.subject,
            "fit_strategy": args.fit_strategy,
            "fit_policy": args.fit_policy,
            "fitted": dict(zip(names, fitted.tolist(), strict=True)),
            "initial": initial,
            "best_training_log_likelihood": float(fit["optimal_value"]),
            "evaluations": len(records),
            "fit_seconds": fit_seconds,
            "invalid_proposals": invalid,
            "estimator": manifest["estimator"],
            "note": "Search result saved before independent validation. This is not an optimizer resume checkpoint.",
        },
    )
    comparison = {"initial": STARTS[args.start], "fitted": fitted}
    if recovery:
        comparison = {"truth": TRUTH, **comparison}
    validation_estimates = args.validation_estimates or args.estimates
    # Keep alpha/N fixed so higher precision does not change the observation model.
    validation_pseudocount = args.pseudocount * validation_estimates / args.estimates
    manifest.update(status="validating", fit_seconds=fit_seconds)
    save_json(args.output / "manifest.json", manifest)
    validation = []
    validation_record = {
        "status": "validating",
        "estimates": validation_estimates,
        "pseudocount": validation_pseudocount,
        "requested_seeds": args.validation_seeds,
        "independent_seed_rescoring": validation,
    }
    save_json(args.output / "validation.json", validation_record)
    for seed in args.validation_seeds:
        scores = pec.log_likelihood_batch(
            list(comparison.values()),
            inputs=inputs,
            num_estimates=validation_estimates,
            seed=seed,
        )
        score = dict(zip(comparison, map(float, scores), strict=True))
        validation.append(
            {
                "seed": seed,
                **score,
                "fitted_minus_initial": score["fitted"] - score["initial"],
            }
        )
        if recovery:
            validation[-1]["fitted_minus_truth"] = score["fitted"] - score["truth"]
        save_json(args.output / "validation.json", validation_record)
    validation_summary = summarize_validation(validation)
    validation_record.update(status="complete", summary=validation_summary)
    save_json(args.output / "validation.json", validation_record)
    manifest.update(status="predicting")
    save_json(args.output / "manifest.json", manifest)
    # These are fresh, unconditioned model trajectories for predictive summaries.
    # Their RTs precede the observation kernel used to score the recorded data.
    predictions = {}
    predictive_seed = args.data_seed + 1000000 if recovery else args.predictive_seed
    predictive_parameters = (
        {"truth": TRUTH, "fitted": fitted} if recovery else {"fitted": fitted}
    )
    for label, values in predictive_parameters.items():
        samples = plan.run(
            inputs,
            [function._batched_parameter_set(values)],
            args.predictive_estimates,
            seed=predictive_seed,
            strict_truncation=True,
            triton_launch_options=LAUNCH,
        ).values[0, 0][..., indices]
        predictions[label] = summarize_samples(samples, frame)
    report = {
        "status": "complete",
        "mode": manifest["mode"],
        "subject": args.subject,
        "initial": initial,
        "estimator": manifest["estimator"],
        "fitted": dict(zip(names, fitted.tolist(), strict=True)),
        "best_training_log_likelihood": float(fit["optimal_value"]),
        "initial_training_log_likelihood": (
            records[0]["log_likelihood"]
            if strategy_report is None
            else strategy_report["initial_reference_score"]
        ),
        "evaluations": len(records),
        "requested_evaluations": args.evaluations,
        "fit_seconds": fit_seconds,
        "fit_strategy": args.fit_strategy,
        "fit_policy": args.fit_policy,
        "total_seconds": time.perf_counter() - started,
        "invalid_proposals": invalid,
        "validation_estimates": validation_estimates,
        "validation_pseudocount": validation_pseudocount,
        "validation_summary": validation_summary,
        "independent_seed_rescoring": validation,
        "data_summary": manifest["data_summary"],
        "predictive_summaries": predictions,
        "predictive_seed": predictive_seed,
        "predictive_summary_kind": "latent_choices_rt_before_observation_kernel",
        "optimizer_stop_reason": "Requested evaluation budget completed; convergence not asserted",
    }
    if strategy_report is not None:
        report[args.fit_strategy] = {**strategy_report, "sampling_work": sampling_work}
        report["optimizer_stop_reason"] = (
            strategy_report["search_stop_reason"]
            + "; "
            + strategy_report["refinement_stop_reason"]
        )
    if ndt_profile is not None:
        report["ndt_profile"] = ndt_profile.describe()
        report["initial_training_score_note"] = (
            "Initial dynamics with optimized NDT; independent-seed 'initial' scores retain the original NDT."
        )
    if recovery:
        widths = np.array([bounds[name][1] - bounds[name][0] for name in names])
        report.update(
            truth=manifest["truth"],
            synthetic_observation_model=observation_model,
            errors=dict(zip(names, (fitted - TRUTH).tolist(), strict=True)),
            errors_as_fraction_of_bounds=dict(
                zip(names, ((fitted - TRUTH) / widths).tolist(), strict=True)
            ),
        )
    save_json(args.output / ("recovery.json" if recovery else "fit.json"), report)
    manifest.update(status="complete", fit_seconds=fit_seconds)
    save_json(args.output / "manifest.json", manifest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
