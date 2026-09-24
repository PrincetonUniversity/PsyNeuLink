"""Synthetic recovery and local identifiability for fresh-trial DAWA conditions.

Data come from the independent continuous SDE sampler used by the convergence
study. Trials are independent resets, not an empirical sequential-history fit.
Choice/RT bins and the right-censored category form a multinomial likelihood.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from dawa_continuous_study import PARAMETERS, CONDITIONS
from dawa_likelihood import PARAMETER_NAMES
from dawa_likelihood.model import PARAMETER_BOUNDS
from dawa_likelihood.fit import minimize_bounded_bfgs
from dawa_likelihood.continuous_model import continuous_path
from dawa_likelihood.continuous_native import continuous_path_native
from dawa_likelihood.continuous_solver import ContinuousConfig, ContinuousResponseSolver


def bin_probabilities(distribution, edges, ndt):
    """Vectorized conservative linear-density integration, followed by censoring."""
    mass = distribution.choice_mass
    t = torch.arange(len(mass), dtype=mass.dtype, device=mass.device) * distribution.time_step
    relative = ((edges[:, None] - ndt - t[None]) / distribution.time_step).clamp(0., 1.)
    left, right = relative[:-1], relative[1:]
    slope = torch.cat(((mass[1] - mass[0])[None], .5 * (mass[2:] - mass[:-2]), (mass[-1] - mass[-2])[None]))
    slope = torch.maximum(torch.minimum(slope, 2. * mass), -2. * mass)
    average = mass[None] + slope[None] * (.5 * (left + right) - .5)[..., None]
    probabilities = ((right - left)[..., None] * average).sum(dim=1)
    return torch.cat((probabilities.reshape(-1), (1. - probabilities.sum())[None]))


def probabilities(parameters, conditions, edges, config):
    path_function = continuous_path_native if config.ode_backend == "generated" else continuous_path
    # A fixed horizon removes an unnecessary candidate-dependent allocation;
    # RT-bin edges still depend differentiably on the candidate's NDT.
    steps = int(np.ceil((float(edges[-1]) - PARAMETER_BOUNDS[1][0]) / config.time_step)) + 1
    solver = ContinuousResponseSolver(config)
    rows = []
    for ci in conditions:
        _, task, stimulus = CONDITIONS[ci]
        path = path_function(parameters, task, stimulus, steps=steps, time_step=config.time_step,
                             ode_step=config.ode_step, clock_ratio=config.lc_clock_ratio)
        distribution = solver.solve(path, parameters)
        if float(distribution.lower_loss.detach()) > 1.e-7:
            raise FloatingPointError("Lower truncation loss exceeds recovery tolerance.")
        rows.append(bin_probabilities(distribution, edges, parameters[1]))
    return torch.stack(rows)


def observation_counts(samples, ndt, edges):
    bins = len(edges) - 1
    counts = np.zeros(2 * bins + 1)
    rt = samples[:, 1] + ndt
    completed = (samples[:, 0] >= 0) & (rt < edges[-1])
    if np.any(rt[completed] < edges[0]):
        raise ValueError("Observation bins do not cover all completed responses.")
    k = np.searchsorted(edges, rt[completed], side="right") - 1
    np.add.at(counts, 2 * k + samples[completed, 0].astype(int), 1.)
    counts[-1] = np.count_nonzero(~completed)
    return counts


def information_check(parameters, conditions, edges, config, estimates, free=None):
    bounds = np.asarray(PARAMETER_BOUNDS)
    scale = bounds[:, 1] - bounds[:, 0]
    with torch.no_grad():
        center = probabilities(parameters, conditions, edges, config).numpy().ravel()
        columns = []
        free = list(range(7)) if free is None else free
        for i in free:
            h = 1.e-4 * scale[i]
            plus, minus = parameters.clone(), parameters.clone()
            plus[i] += h
            minus[i] -= h
            derivative = (probabilities(plus, conditions, edges, config) - probabilities(minus, conditions, edges, config)) / (2.e-4)
            columns.append(derivative.numpy().ravel())
            print(f"information sensitivity: {PARAMETER_NAMES[i]}", flush=True)
    keep = center > 1.e-10
    weighted = np.stack(columns, axis=1)[keep] * np.sqrt(estimates / center[keep, None])
    _, singular, vectors = np.linalg.svd(weighted, full_matrices=False)
    return {"coordinate_system": "parameter bounds scaled to [0,1]", "samples_per_condition": estimates,
            "singular_values_of_weighted_sensitivity": singular.tolist(),
            "condition_number": float(singular[0] / singular[-1]),
            "information_per_trial": (weighted.T @ weighted / (estimates * len(conditions))).tolist(),
            "weakest_parameter_direction": dict(zip([PARAMETER_NAMES[i] for i in free], vectors[-1].tolist())),
            "relative_rank_at_1e_minus_6": int(np.count_nonzero(singular > singular[0] * 1.e-6)),
            "note": "Local expected information for these binned fresh-trial conditions, not a global identifiability proof."}


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    record = min(report["fits"], key=lambda f: f["accepted_losses"][-1])
    bounds = np.asarray(PARAMETER_BOUNDS)
    low, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    true = (np.array([report["truth"][name] for name in PARAMETER_NAMES]) - low) / width
    fitted = (np.array([record["fitted"][name] for name in PARAMETER_NAMES]) - low) / width
    free = [PARAMETER_NAMES.index(name) for name in report["free_parameters"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].scatter(np.arange(7), true, color="black", label="Generating values", zorder=3)
    axes[0].scatter(free, fitted[free], marker="x", s=65, color="tab:blue", label="Estimated values", zorder=4)
    axes[0].set(xticks=np.arange(7), xticklabels=[name.replace("_", " ") for name in PARAMETER_NAMES],
                ylabel="Position within original fitting range", ylim=(-.03, 1.03), title="Parameter recovery")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].legend(fontsize=8)
    refined = record["refined_check"]
    n = len(report["conditions"])
    truth_cdf = np.asarray(refined["truth_probabilities"])[:, :-1].reshape(n, -1, 2).sum(2).cumsum(1)
    fit_cdf = np.asarray(refined["fit_probabilities"])[:, :-1].reshape(n, -1, 2).sum(2).cumsum(1)
    counts = np.asarray(report["heldout_counts"])
    empirical = counts[:, :-1].reshape(n, -1, 2).sum(2).cumsum(1) / counts.sum(1, keepdims=True)
    for i, name in enumerate(report["conditions"]):
        color = f"C{i}"
        axes[1].plot(report["rt_edges"][1:], truth_cdf[i], color=color, label=f"{name.replace('_', ' ')}: generating")
        axes[1].plot(report["rt_edges"][1:], fit_cdf[i], color=color, ls="--", label="Fitted")
        axes[1].scatter(report["rt_edges"][1::2], empirical[i, ::2], color=color, s=8, alpha=.5, label="Held out")
    axes[1].set(xlabel="Response time (s)", ylabel="Cumulative response probability", xlim=(.2, 1.5),
                title=f"Predictive CDFs at {refined['points']}×{refined['points']}")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for extension in (".png", ".pdf"):
        fig.savefig(output.with_suffix(extension), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("/tmp/dawa_continuous_study"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/dawa_continuous_recovery.json"))
    parser.add_argument("--truth", type=int, choices=(0, 1), default=1)
    parser.add_argument("--conditions", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--estimates", type=int, default=2000)
    parser.add_argument("--heldout", type=int, default=10000)
    parser.add_argument("--points", type=int, default=65)
    parser.add_argument("--verify-points", type=int, default=129)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--starts", type=int, choices=(1, 2), default=1)
    parser.add_argument("--initial", type=float, nargs=7, action="append",
                        help="One or more explicit starting parameter vectors (overrides --starts)")
    parser.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE",
                        help="Hold a parameter at a specified value; useful for profile likelihoods")
    parser.add_argument("--resume", action="store_true", help="Resume accepted optimizer states in --output")
    parser.add_argument("--gradient-tolerance", type=float, default=1.e-6)
    parser.add_argument("--precondition-points", type=int, default=0,
                        help="Initialize curvature from coarse-grid expected information; objective grid is unchanged")
    parser.add_argument("--time-step", type=float, default=.001)
    parser.add_argument("--retain-rates", action="store_true", help="Use more RAM to avoid recomputing coefficient blocks")
    parser.add_argument("--free", nargs="+", choices=PARAMETER_NAMES, default=PARAMETER_NAMES)
    parser.add_argument("--information", action="store_true")
    args = parser.parse_args()
    if min(args.estimates, args.heldout, args.iterations) < 1 or any(c not in range(4) for c in args.conditions):
        parser.error("Invalid sample counts, iterations, or condition indices")
    if len(set(args.conditions)) != len(args.conditions) or len(set(args.free)) != len(args.free):
        parser.error("Conditions and free parameters must be unique")
    torch.set_num_threads(1)
    truth = torch.tensor(PARAMETERS[args.truth], dtype=torch.float64)
    edges = torch.tensor(np.r_[0., np.arange(.225, 1.501, .025)], dtype=torch.float64)
    train, test = [], []
    for ci in args.conditions:
        samples = np.load(args.study / f"continuous_{args.truth}_{ci}.npz")["samples"]
        if len(samples) < args.estimates + args.heldout:
            parser.error("Not enough independent samples in study cache")
        train.append(observation_counts(samples[:args.estimates], float(truth[1]), edges.numpy()))
        test.append(observation_counts(samples[args.estimates:args.estimates + args.heldout], float(truth[1]), edges.numpy()))
    train, test = torch.tensor(np.asarray(train), dtype=torch.float64), torch.tensor(np.asarray(test), dtype=torch.float64)
    cfg = ContinuousConfig(points=args.points, time_step=args.time_step, ode_backend="generated", flux_backend="native",
                           recompute_rates=not args.retain_rates)
    source_metadata = None
    if (args.study / "results.json").exists():
        metadata = json.loads((args.study / "results.json").read_text())
        if metadata["parameter_sets"][args.truth] != dict(zip(PARAMETER_NAMES, truth.tolist())) \
                or metadata["lc_clock_ratio"] != cfg.lc_clock_ratio:
            parser.error("Cached simulation parameters or LC clock ratio do not match this recovery")
        for ci in args.conditions:
            name, task, stimulus = CONDITIONS[ci]
            if metadata["conditions"][ci] != {"name": name, "task": list(task), "stimulus": list(stimulus)}:
                parser.error("Cached simulation conditions do not match this recovery")
        source_metadata = {"configuration": metadata["configuration"], "parameter_set": metadata["parameter_sets"][args.truth],
                           "lc_clock_ratio": metadata["lc_clock_ratio"], "conditions": [metadata["conditions"][ci] for ci in args.conditions]}
    fixed = {}
    for item in args.fix:
        name, value = item.split("=", 1)
        if name not in PARAMETER_NAMES:
            parser.error(f"Unknown fixed parameter {name}")
        index = PARAMETER_NAMES.index(name)
        value = float(value)
        if not PARAMETER_BOUNDS[index][0] <= value <= PARAMETER_BOUNDS[index][1]:
            parser.error(f"Fixed parameter {name} is outside its bounds")
        fixed[name] = value
    free = [PARAMETER_NAMES.index(name) for name in args.free if name not in fixed]
    if not free:
        parser.error("At least one free parameter is required")
    base = truth.clone()
    for name, value in fixed.items():
        base[PARAMETER_NAMES.index(name)] = value
    bounds = np.asarray(PARAMETER_BOUNDS)
    scales, lower = bounds[:, 1] - bounds[:, 0], bounds[:, 0]
    report = {"design": "independent fresh trials; two choice-specific RT histograms plus right censoring",
              "truth": dict(zip(PARAMETER_NAMES, truth.tolist())), "conditions": [CONDITIONS[i][0] for i in args.conditions],
              "training_samples_per_condition": args.estimates, "heldout_samples_per_condition": args.heldout,
              "rt_edges": edges.tolist(), "free_parameters": [PARAMETER_NAMES[i] for i in free], "points": args.points,
              "fixed_parameters": fixed, "time_step": cfg.time_step,
              "gradient_tolerance": args.gradient_tolerance,
              "training_counts": train.tolist(), "heldout_counts": test.tolist(),
              "ode_backend": cfg.ode_backend, "flux_backend": cfg.flux_backend, "fits": []}
    report["optimizer"] = "BFGS in unit coordinates; box-constrained quadratic steps with feasible Armijo backtracking"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous = json.loads(args.output.read_text())
        for key in ("truth", "conditions", "training_counts", "heldout_counts", "rt_edges", "free_parameters",
                    "fixed_parameters", "points", "time_step", "ode_backend", "flux_backend"):
            if previous.get(key) != report.get(key):
                parser.error(f"Resume configuration changed: {key}")
        report = previous
    report["recompute_rates"] = cfg.recompute_rates
    report["solver_config"] = asdict(cfg)
    report["simulation_source"] = source_metadata
    report["device"], report["dtype"] = "cpu", "float64"
    report["gradient_tolerance"] = args.gradient_tolerance
    report["optimizer"] = "BFGS in unit coordinates; box-constrained quadratic steps with feasible Armijo backtracking"

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    def log_score(prob, counts):
        selected = counts > 0
        if not bool(torch.isfinite(prob).all()) or bool((prob[selected] <= 0).any()):
            raise FloatingPointError("Observation probabilities are zero or nonfinite.")
        return (counts[selected] * prob[selected].log()).sum() / counts.sum()

    with torch.no_grad():
        ref = probabilities(truth, args.conditions, edges, cfg)
        report["truth_scores"] = {"training_per_trial": float(log_score(ref, train)), "heldout_per_trial": float(log_score(ref, test))}
    save()
    starts = args.initial or ((.30, .20, -.45, 10., .75, 1.3, 5.2), (.29, .18, -.44, 13., .4, 2.5, 5.))[:args.starts]
    if args.resume:
        starts = report.get("requested_initials", [[r["initial"][name] for name in PARAMETER_NAMES] for r in report["fits"]])
    report["requested_initials"] = [list(start) for start in starts]
    save()
    for start_index, start in enumerate(starts):
        initial = base.numpy().copy()
        initial[free] = np.asarray(start)[free]
        if args.resume and start_index < len(report["fits"]):
            record = report["fits"][start_index]
            if record.get("success") and record.get("projected_gradient_norm", np.inf) < args.gradient_tolerance \
                    and (not args.verify_points or record.get("refined_check", {}).get("points") == args.verify_points):
                continue
            evaluations, rejected = record["evaluations"], record["rejected"]
        else:
            evaluations, rejected = [], []
            record = {"initial": dict(zip(PARAMETER_NAMES, initial.tolist())), "evaluations": evaluations, "rejected": rejected}
            report["fits"].append(record)
        probability_cache = {}

        def evaluate(unit):
            begin = time.perf_counter()
            p = base.detach().clone()
            p[free] = torch.tensor(lower[free] + scales[free] * unit)
            p.requires_grad_()
            try:
                prob = probabilities(p, args.conditions, edges, cfg)
                score = log_score(prob, train)
                gradient = torch.autograd.grad(score, p)[0].numpy()
                if not np.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite score gradient")
            except FloatingPointError as error:
                rejected.append({"parameters": p.detach().tolist(), "reason": str(error)})
                save()
                # Reject infeasible proposals without flooring probabilities.
                return np.inf, np.zeros(len(free))
            probability_cache.update(parameters=p.detach().clone(), probabilities=prob.detach().clone())
            # A candidate can assign zero mass to an unobserved training bin
            # that is occupied in the holdout. This is a -inf predictive score,
            # not a reason to reject a finite training objective or crash a fit.
            try:
                heldout = float(log_score(prob.detach(), test))
            except FloatingPointError:
                heldout = None
            evaluation = {"parameters": p.detach().tolist(), "training_per_trial": float(score.detach()),
                          "heldout_per_trial": heldout, "heldout_zero_probability": heldout is None,
                          "seconds": time.perf_counter() - begin}
            evaluations.append(evaluation)
            print(f"fit {start_index + 1} evaluation {len(evaluations)}: train={evaluation['training_per_trial']:.9f}, "
                  f"heldout={heldout}, {evaluation['seconds']:.2f}s", flush=True)
            save()
            return -evaluation["training_per_trial"], -gradient[free] * scales[free]

        def accepted(state):
            record["optimizer_state"] = state
            if record.get("preconditioner"):
                record["preconditioner"]["applied"] = True
            print(f"accepted {len(state['accepted_losses']) - 1}: projected gradient {state['projected_gradient_norm']:.3g}", flush=True)
            save()

        inverse = None
        if args.precondition_points and not record.get("preconditioner"):
            center = base.clone()
            center[free] = torch.tensor(lower[free] + scales[free] * np.asarray(record["optimizer_state"]["x"])) \
                if record.get("optimizer_state") else torch.tensor(initial[free])
            info = information_check(center, args.conditions, edges, replace(cfg, points=args.precondition_points), 1., free)
            values, vectors = np.linalg.eigh(info["information_per_trial"])
            floor = max(1.e-3, values[-1] * 1.e-6)
            inverse = (vectors / np.maximum(values, floor)) @ vectors.T
            record["preconditioner"] = {"points": args.precondition_points, "center": center.tolist(),
                                         "eigenvalue_floor": floor, "information": info,
                                         "inverse_hessian": inverse.tolist(), "applied": False}
            save()
        elif record.get("preconditioner", {}).get("applied") is False:
            inverse = np.asarray(record["preconditioner"]["inverse_hessian"])
        result = minimize_bounded_bfgs(evaluate, (initial[free] - lower[free]) / scales[free], iterations=args.iterations,
                                       tolerance=args.gradient_tolerance, state=record.get("optimizer_state"), callback=accepted,
                                       initial_inverse=inverse)
        fitted = base.clone()
        fitted[free] = torch.tensor(lower[free] + scales[free] * result["x"])
        record.update({"success": result["success"], "message": result["message"], "iterations": result["nit"],
                       "projected_gradient_norm": result["projected_gradient_norm"], "optimizer_state": result["state"],
                       "accepted_losses": result["accepted_losses"], "rejected_proposals": result["rejected_proposals"],
                       "fitted": dict(zip(PARAMETER_NAMES, fitted.tolist())),
                       "errors_in_parameter_range_units": dict(zip(PARAMETER_NAMES, ((fitted.numpy() - truth.numpy()) / scales).tolist()))})
        with torch.no_grad():
            coarse_fit = probability_cache["probabilities"] if torch.equal(fitted, probability_cache.get("parameters", fitted + 1.)) \
                else probabilities(fitted, args.conditions, edges, cfg)
        record["coarse_check"] = {"points": cfg.points, "truth_probabilities": ref.tolist(), "fit_probabilities": coarse_fit.tolist()}
        save()
        if args.verify_points:
            fine = replace(cfg, points=args.verify_points)
            with torch.no_grad():
                fine_truth = probabilities(truth, args.conditions, edges, fine)
                fine_fit = probabilities(fitted, args.conditions, edges, fine)
            record["refined_check"] = {"points": args.verify_points,
                                       "solver_config": asdict(fine),
                                       "truth_probabilities": fine_truth.tolist(), "fit_probabilities": fine_fit.tolist(),
                                       "truth_heldout_per_trial": float(log_score(fine_truth, test)),
                                       "fit_heldout_per_trial": float(log_score(fine_fit, test)),
                                       "maximum_joint_cdf_difference_fit_truth": float(
                                           (fine_fit[:, :-1].reshape(len(args.conditions), -1, 2).cumsum(1)
                                            - fine_truth[:, :-1].reshape(len(args.conditions), -1, 2).cumsum(1)).abs().max())}
        save()
    if args.information:
        report["local_information"] = information_check(truth, args.conditions, edges, cfg, args.estimates)
        save()
    if args.verify_points:
        plot_report(report, args.output)
    print(f"Saved recovery audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
