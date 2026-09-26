"""Local fitting-acceleration experiments; does not change the production fitter.

Use saved CMA-ES populations to audit smaller Monte Carlo budgets against fresh
100,000-trajectory references. Also profile nondecision time using cached exact
decision-time counts. Every simulation retains the complete subject history.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched import likelihood
from psyneulink.core.globals.utilities import set_global_seed
from dawa_batched_simulation import build_model, fit_surface, node
from dawa_pec_fit import LAUNCH, RT_RANGE, TRUTH, load_subject, save_json, synthetic_parameters


REFERENCE_ESTIMATES = 100000


def select_candidates(history):
    records = [json.loads(line) for line in history.read_text().splitlines()]
    groups = {}
    for name, start in (("early", 2), ("middle", 1002), ("late", 4991)):
        rows = [r for r in records if start <= r["evaluation"] < start + 10]
        if len(rows) != 10 or any(r["log_likelihood"] < -1e9 for r in rows):
            raise ValueError(f"Expected ten valid proposals in population {name}")
        groups[name] = [{"evaluation": r["evaluation"], "parameters": r["parameters"]} for r in rows]
    return groups


def rescale_prior(densities, estimates, pseudocount, width):
    """Recover weighted counts from alpha=1 scores; keep the same smoothing."""
    counts = np.maximum(0., np.asarray(densities, dtype=np.float64) * ((estimates + 200) * width) - 1.)
    return (counts + pseudocount) / ((estimates + 200 * pseudocount) * width)


def total_scores(densities, mask):
    return np.log(np.maximum(densities, likelihood.ZERO_PROB))[..., mask].sum(-1)


def rank_metrics(scores, reference):
    """Reference itself is finite precision; report uncertainty separately."""
    reference_mean = reference.mean(0)
    reference_se = reference.std(0, ddof=1) / np.sqrt(len(reference))
    i, j = np.triu_indices(scores.shape[1], 1)
    ref_delta = reference_mean[i] - reference_mean[j]
    delta_se = (reference[:, i] - reference[:, j]).std(0, ddof=1) / np.sqrt(len(reference))
    resolved = np.abs(ref_delta) > 2 * delta_se
    delta = scores[:, i] - scores[:, j]
    errors = np.sign(delta) != np.sign(ref_delta)
    top = np.argsort(reference_mean)[-5:]
    independent_var = scores.var(0, ddof=1)[i] + scores.var(0, ddof=1)[j]
    paired_var = delta.var(0, ddof=1)
    return {
        "mean_score_difference_from_reference": float((scores.mean(0) - reference_mean).mean()),
        "mean_seed_score_sd": float(scores.std(0, ddof=1).mean()),
        "pair_disagreement_fraction": float(errors.mean()),
        "reference_resolved_pairs": int(resolved.sum()),
        "resolved_pair_disagreement_fraction": float(errors[:, resolved].mean()) if resolved.any() else None,
        "mean_top_five_overlap": float(np.mean([len(set(np.argsort(row)[-5:]) & set(top)) / 5 for row in scores])),
        "winner_reference_regret": [float(reference_mean.max() - reference_mean[np.argmax(row)]) for row in scores],
        "reference_mean_score_se": float(reference_se.mean()),
        "crn_pair_variance_ratio": float(independent_var.sum() / paired_var.sum()) if paired_var.sum() else None,
    }


class Study:
    def __init__(self, frame, max_steps):
        self.frame = frame
        self.observed = frame[["decision", "response_time"]].to_numpy()
        self.mask = frame.likelihood_include_mask.to_numpy(dtype=bool)
        set_global_seed(29)
        model, inputs, outputs = build_model(trials=len(frame), c_noise=.1, s_noise=.1, d_noise=.1, r_noise=.1)
        inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
        inputs[node(model, "Stimulus Input")] = frame[["S1", "S2", "S3", "S4"]].to_numpy()
        plan = BatchedCompositionCompiler.compile(model, backend="triton", outputs=outputs, max_steps=max_steps)
        fitted = {f"{mechanism.name}.{parameter}" for parameter, mechanism in fit_surface(model)}
        self.plan = plan.specialize_parameters({p.name: p.default for p in plan.ir.params if p.name not in fitted})
        self.model, self.inputs = model, inputs
        observed = torch.tensor(self.observed, device="cuda", dtype=torch.float32)
        self.edges = likelihood._bin_edges(observed[:, 1:], observed[:, 1:], 100, [RT_RANGE], torch)[0]
        self.width = float((self.edges[1] - self.edges[0]).item())

    def parameters(self, candidates):
        return [synthetic_parameters(self.model, self.frame, row) for row in candidates]

    def score(self, candidates, estimates, seed, *, alpha=1.):
        # Observe the existing scorer's density output, without modifying its
        # kernel or retaining the large per-trajectory simulation output.
        captured = []
        original = likelihood._sum_histogram_log_likelihood

        def capture(values, include_mask):
            captured.append(values.copy())
            return original(values, include_mask)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with patch.object(likelihood, "_sum_histogram_log_likelihood", capture):
            scores = self.plan.log_likelihood(
                self.inputs, self.parameters(candidates), estimates, data=self.observed,
                categorical_dims=[0], bins=100, bin_range=[RT_RANGE], smoothing_sigma=.5,
                pseudocount=alpha, categorical_cardinalities=[2], include_mask=self.mask,
                seed=seed, strict_truncation=True, triton_launch_options=LAUNCH,
            )
        torch.cuda.synchronize()
        return np.asarray(scores).reshape(-1), captured[0], time.perf_counter() - start

    def sample(self, candidate, estimates, seed):
        return self.plan.run(self.inputs, self.parameters([candidate]), estimates, seed=seed,
                             strict_truncation=True, triton_launch_options=LAUNCH,
                             keep_device_values=True).values[0, 0]


def decision_time_counts(samples):
    """Lossless compression: actual FP32 decision times, not rounded step counts."""
    times, index = torch.unique(samples[..., 1], sorted=True, return_inverse=True)
    choices = samples[..., 0].long()
    if not torch.all((choices == 0) | (choices == 1)):
        raise AssertionError("Expected binary choices")
    trial = torch.arange(len(samples), device=samples.device)[:, None]
    address = (trial * 2 + choices) * len(times) + index
    counts = torch.bincount(address.flatten(), minlength=len(samples) * 2 * len(times))
    return times, counts.reshape(len(samples), 2, len(times))


def score_ndt_grid(times, counts, shifts, observed, mask, edges, estimates, *, pseudocount=1., chunk=64):
    """Apply FP32 addition and the production histogram rule to cached counts.

    Work scales with unique decision times rather than number of trajectories.
    Rebin after shifting: shifting a coarse RT histogram would lose information.
    """
    data = torch.as_tensor(observed, device=times.device, dtype=torch.float32)
    observed_bin = torch.bucketize(data[:, 1].contiguous(), edges[1:-1])
    offsets = torch.arange(-2, 3, device=times.device)
    kernel = torch.exp(-.5 * (offsets.float() / .5) ** 2)
    norm = (((observed_bin[:, None] + offsets >= 0) & (observed_bin[:, None] + offsets < 100)) * kernel).sum(-1)
    selected = counts[torch.arange(len(data), device=times.device), data[:, 0].long()].float()
    mask = torch.tensor(mask, device=times.device)
    results = []
    for start in range(0, len(shifts), chunk):
        shift = torch.as_tensor(shifts[start:start + chunk], device=times.device, dtype=torch.float32)
        rt = times[None, :] + shift[:, None]
        bins = torch.bucketize(rt, edges[1:-1])
        delta = bins[:, None, :] - observed_bin[None, :, None]
        valid = ((delta.abs() <= 2) & (rt[:, None, :] >= edges[0]) & (rt[:, None, :] <= edges[-1])
                 & (data[None, :, 1, None] >= edges[0]) & (data[None, :, 1, None] <= edges[-1]))
        weight = torch.exp(-.5 * (delta.float() / .5) ** 2) * valid / norm[None, :, None]
        weighted = (weight * selected[None]).sum(-1)
        density = (weighted + pseudocount) / ((estimates + 200 * pseudocount) * (edges[1] - edges[0]))
        results.append(torch.log(torch.clamp(density, min=likelihood.ZERO_PROB))[:, mask].sum(-1))
    return torch.cat(results).cpu().numpy()


def ndt_study(study, candidate, estimates, seed):
    base = list(candidate)
    base[1] = 0.
    # Warm separately; include materialization and compression in measured cost.
    study.sample(base, estimates, seed)
    study.score([candidate], estimates, seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    samples = study.sample(base, estimates, seed)
    times, counts = decision_time_counts(samples)
    torch.cuda.synchronize()
    generation = time.perf_counter() - start
    # Verify pathwise equality over the full subject and all estimates. This
    # checks that shifting the readout cannot change stopping or carried state.
    check_shifts = [.1, .1234, .2, .2199, .3]
    checks = []
    for shift in check_shifts:
        proposal = list(candidate)
        proposal[1] = shift
        actual = study.sample(proposal, estimates, seed)
        torch.testing.assert_close(actual[..., 0], samples[..., 0], rtol=0, atol=0)
        torch.testing.assert_close(actual[..., 1], samples[..., 1] + np.float32(shift), rtol=0, atol=0)
        expected, _, seconds = study.score([proposal], estimates, seed)
        cached = score_ndt_grid(times, counts, [shift], study.observed, study.mask, study.edges, estimates)
        np.testing.assert_allclose(cached, expected, rtol=0, atol=.002)
        checks.append({"ndt": shift, "fused_score": float(expected[0]), "cached_score": float(cached[0]),
                       "fused_seconds": seconds})
        del actual
    shifts = np.linspace(.1, .3, 2001)
    score_ndt_grid(times, counts, shifts[:64], study.observed, study.mask, study.edges, estimates)
    start = time.perf_counter()
    scores = score_ndt_grid(times, counts, shifts, study.observed, study.mask, study.edges, estimates)
    seconds = time.perf_counter() - start
    start = time.perf_counter()
    # Equal bin-membership maps imply exactly equal scores. Include finite-range
    # exclusion in the signature; simply clipping to the first/last bin is wrong.
    rt = times[None, :] + torch.tensor(shifts, device=times.device, dtype=torch.float32)[:, None]
    signature = torch.bucketize(rt, study.edges[1:-1])
    signature[(rt < study.edges[0]) | (rt > study.edges[-1])] = -1
    _, representatives, inverse = np.unique(signature.cpu().numpy(), axis=0, return_index=True, return_inverse=True)
    reduced = score_ndt_grid(times, counts, shifts[representatives], study.observed, study.mask, study.edges, estimates)[inverse]
    reduced_seconds = time.perf_counter() - start
    np.testing.assert_allclose(reduced, scores, rtol=0, atol=.0001)
    best = int(np.argmax(scores))
    return {"estimates": estimates, "seed": seed, "dynamic_parameters": list(candidate),
            "pathwise_checks_passed": True, "checks": checks,
            "sample_and_compress_seconds": generation, "grid_seconds": seconds,
            "grid_points": len(shifts), "unique_decision_times": len(times),
            "distinct_bin_maps": len(representatives), "deduplicated_grid_seconds": reduced_seconds,
            "distinct_grid_scores": len(np.unique(scores)), "best_ndt": float(shifts[best]),
            "best_score": float(scores[best]), "cache_bytes": counts.numel() * counts.element_size() + times.numel() * times.element_size(),
            "materialized_sample_bytes": samples.numel() * samples.element_size()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Observed or saved synthetic subject CSV")
    parser.add_argument("--generate-seed", type=int, help="Replace recorded responses with one TRUTH trajectory at this seed")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history", type=Path, help="Full 5,000-proposal evaluations.jsonl")
    source.add_argument("--candidates", type=Path, help="Previously exported candidate groups JSON")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000, 10000, 25000, 100000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4201, 4202, 4203])
    parser.add_argument("--reference-seeds", type=int, nargs="+", default=[9101, 9102, 9103])
    parser.add_argument("--ndt-estimates", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) < 3 or len(args.reference_seeds) < 3 or set(args.seeds) & set(args.reference_seeds):
        parser.error("Use at least three disjoint study and reference seeds")
    if min(args.budgets + [args.ndt_estimates, args.max_steps]) < 1:
        parser.error("Counts must be positive")
    frame = load_subject(args.data, args.subject, trials=args.trials, recovery=args.generate_seed is not None)
    if args.generate_seed is not None:
        # The model is compiled independently of outcomes; these placeholders
        # are replaced before any scoring and never enter simulation dynamics.
        frame["decision"], frame["response_time"] = 0., .1
    groups = select_candidates(args.history) if args.history else json.loads(args.candidates.read_text())
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "candidates.json", groups)
    pnl.set_num_threads(8)
    torch.set_num_threads(8)
    study = Study(frame, args.max_steps)
    if args.generate_seed is not None:
        generated = study.sample(TRUTH, 1, args.generate_seed).cpu().numpy()[:, 0]
        frame["decision"], frame["response_time"] = generated[:, 0], generated[:, 1]
        study.observed = frame[["decision", "response_time"]].to_numpy()
    frame.to_csv(args.output / "observations.csv", index=False)
    report = {"status": "running", "gpu": torch.cuda.get_device_name(), "hostname": platform.node(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "generate_seed": args.generate_seed,
              "observations_sha256": hashlib.sha256((args.output / "observations.csv").read_bytes()).hexdigest(),
              "torch": torch.__version__, "trials": len(frame), "scored_trials": int(study.mask.sum()),
              "lca_dt": .01, "noise_all_lcas": .1, "launch": LAUNCH, "max_steps": args.max_steps,
              "seeds": args.seeds, "reference_seeds": args.reference_seeds, "reference_estimates": REFERENCE_ESTIMATES,
              "estimator": {"bins": 100, "range": RT_RANGE, "smoothing_sigma": .5, "reference_pseudocount": 1.},
              "groups": {}}
    started = time.perf_counter()
    for name, rows in groups.items():
        candidates = [row["parameters"] for row in rows]
        result = {"references": [], "budgets": []}
        report["groups"][name] = result
        study.score(candidates, REFERENCE_ESTIMATES, args.reference_seeds[0])
        for seed in args.reference_seeds:
            scores, _, seconds = study.score(candidates, REFERENCE_ESTIMATES, seed)
            result["references"].append({"seed": seed, "scores": scores.tolist(), "seconds": seconds})
        reference = np.array([row["scores"] for row in result["references"]])
        for budget in args.budgets:
            study.score(candidates, budget, args.seeds[0])
            case = {"estimates": budget, "runs": []}
            pooled = []
            for seed in args.seeds:
                scores, densities, seconds = study.score(candidates, budget, seed)
                scaled = rescale_prior(densities, budget, budget / REFERENCE_ESTIMATES, study.width)
                pooled.append(scaled)
                case["runs"].append({"seed": seed, "seconds": seconds, "fixed_prior_scores": scores.tolist(),
                                     "scaled_prior_scores": total_scores(scaled, study.mask).tolist()})
            for kind in ("fixed", "scaled"):
                case[f"{kind}_prior_metrics"] = rank_metrics(np.array([r[f"{kind}_prior_scores"] for r in case["runs"]]), reference)
            # Equal-sized independent trajectory blocks can pool densities when
            # alpha/N is fixed. Average densities BEFORE taking logs.
            case["pooled_scaled_scores"] = total_scores(np.mean(pooled, axis=0), study.mask).tolist()
            case["pooled_estimates"] = budget * len(args.seeds)
            case["seconds_per_candidate"] = float(np.median([r["seconds"] for r in case["runs"]]) / len(candidates))
            case["speedup_vs_100k"] = float(np.median([r["seconds"] for r in result["references"]]) /
                                                  np.median([r["seconds"] for r in case["runs"]]))
            result["budgets"].append(case)
            save_json(args.output / "report.json", report)
            print(json.dumps({"group": name, "estimates": budget, "seconds_per_candidate": case["seconds_per_candidate"],
                              "speedup": case["speedup_vs_100k"], "scaled_metrics": case["scaled_prior_metrics"]}), flush=True)
        # Check density-based prior conversion against a fresh production call.
        check_budget = args.budgets[0]
        _, densities, _ = study.score(candidates, check_budget, args.seeds[0])
        _, actual, _ = study.score(candidates, check_budget, args.seeds[0], alpha=check_budget / REFERENCE_ESTIMATES)
        expected = rescale_prior(densities, check_budget, check_budget / REFERENCE_ESTIMATES, study.width)
        # Compare per-trial densities. The production FP32 sum of hundreds of
        # large negative terms can differ appreciably from a FP64 reduction.
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-7)
    report["ndt"] = ndt_study(study, TRUTH, args.ndt_estimates, 7201)
    report.update(status="complete", total_seconds=time.perf_counter() - started)
    save_json(args.output / "report.json", report)
    print(json.dumps({"ndt": report["ndt"], "total_seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
