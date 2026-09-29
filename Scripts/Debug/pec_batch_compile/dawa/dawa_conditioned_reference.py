"""Check the particle likelihood against independent short-sequence importance sampling.

The reference simulates complete, unresampled histories and independently
evaluates the declared observation kernel in NumPy. It estimates an expectation
of products, not a product of marginal expectations. This is a Monte Carlo
reference with reported uncertainty and effective sample size, not exact truth.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np


def observation_edges(bins=100, rt_range=(0., 3.), *, device="cpu"):
    """Export the declared coordinates; scoring and normalization stay independent.

    FP32 Torch linspace can differ by an ULP from NumPy linspace followed by a
    cast. Export the actual device coordinates so boundary observations define
    exactly the same event in the two implementations.
    """
    import torch

    lo, hi = rt_range
    return torch.linspace(lo, hi + (hi - lo) * 1e-6, bins + 1,
                          dtype=torch.float32, device=device).cpu().numpy()


def numpy_observation_density(samples, observation, *, edges, bins=100, rt_range=(0., 3.),
                               sigma=.5, alpha_per_estimate=1e-5):
    """Independent normalized choice/RT observation kernel, with no Torch calls.

    For a simulated response bin j and observed bin k, Gaussian mass is
    exp(-(j-k)^2/(2*sigma^2)), truncated at ceil(3*sigma) and normalized over
    possible observed bins around j. The constant alpha/N stays fixed across
    budgets. Its uniform-contamination probability is K*c/(1+K*c), K=2*bins.
    """
    samples = np.asarray(samples)
    observed = np.asarray(observation, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    if not edges[0] <= observed[1] <= edges[-1]:
        raise ValueError("The observation must lie in the finite RT domain.")
    source_bin = np.searchsorted(edges[1:-1], samples[..., 1], side="left")
    observed_bin = np.searchsorted(edges[1:-1], observed[1], side="left")
    delta = observed_bin - source_bin
    if sigma == 0.:
        response_mass = (delta == 0).astype(float)
    else:
        radius = max(1, int(np.ceil(3. * sigma)))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-.5 * (offsets / sigma) ** 2)
        source_locations = np.arange(bins)[:, None] + offsets
        normalizers = (((source_locations >= 0) & (source_locations < bins)) * kernel).sum(axis=1)
        response_mass = np.where(np.abs(delta) <= radius,
                                 np.exp(-.5 * (delta / sigma) ** 2) / normalizers[source_bin], 0.)
    matches = np.abs(samples[..., 0] - observed[0]) <= 1e-6
    inside = (samples[..., 1] >= edges[0]) & (samples[..., 1] <= edges[-1])
    mass = matches * inside * response_mass
    width = float(np.float32(edges[1] - edges[0]))
    return (mass + alpha_per_estimate) / ((1. + 2 * bins * alpha_per_estimate) * width)


def noisy_observations(latent, rng, *, edges, bins=100, rt_range=(0., 3.), sigma=.5, alpha_per_estimate=1e-5):
    """Generate bin-center observations under that same discrete measurement law."""
    edges = np.asarray(edges, dtype=np.float32)
    contamination = 2 * bins * alpha_per_estimate / (1. + 2 * bins * alpha_per_estimate)
    rows = []
    for choice, rt in latent:
        if not edges[0] <= rt <= edges[-1]:
            raise ValueError("Synthetic source RT is outside the observation domain; increase --rt-upper.")
        if rng.random() < contamination:
            choice, target = rng.integers(0, 2), rng.integers(0, bins)
        else:
            source = np.searchsorted(edges[1:-1], rt, side="left")
            radius = max(1, int(np.ceil(3. * sigma))) if sigma else 0
            targets = np.arange(max(0, source - radius), min(bins, source + radius + 1))
            masses = np.exp(-.5 * ((targets - source) / sigma) ** 2) if sigma else np.ones(1)
            target = rng.choice(targets, p=masses / masses.sum())
        rows.append((float(choice), float((float(edges[target]) + float(edges[target + 1])) / 2.)))
    return np.asarray(rows)


def weighted_moments(values, weights):
    total = weights.sum()
    normalized = weights / total
    mean = np.sum(normalized[:, None] * values, axis=0)
    centered = values - mean
    covariance = np.einsum("n,ni,nj->ij", normalized, centered, centered)
    return dict(mean=mean.tolist(), covariance=covariance.tolist())


def full_history_estimate(values, observed, **kernel):
    """Prefix joint likelihoods, ratio factors, ESS, and control posterior moments."""
    count = values.shape[1]
    products = np.ones(count)
    previous = products.copy()
    prefixes = []
    for trial, observation in enumerate(observed):
        products *= numpy_observation_density(values[trial, :, :2], observation, **kernel)
        likelihood = products.mean()
        factor = likelihood / previous.mean()
        prefixes.append(dict(
            trials=trial + 1, joint_likelihood=float(likelihood), log_likelihood=float(np.log(likelihood)),
            joint_standard_error=float(products.std(ddof=1) / np.sqrt(count)),
            conditional_density=float(factor),
            conditional_standard_error=float((products - factor * previous).std(ddof=1)
                                             / (np.sqrt(count) * previous.mean())),
            effective_sample_size=float(products.sum() ** 2 / np.sum(products ** 2)),
            control_posterior=weighted_moments(values[trial, :, 2:4], products),
        ))
        previous = products.copy()
    return prefixes


def aggregate_comparison(reference_runs, filter_runs):
    comparisons = []
    for index in range(len(reference_runs[0]["prefixes"])):
        reference = np.array([run["prefixes"][index]["joint_likelihood"] for run in reference_runs])
        filtered = np.array([run["prefixes"][index]["joint_likelihood"] for run in filter_runs])
        ref_se = reference.std(ddof=1) / np.sqrt(len(reference))
        pf_se = filtered.std(ddof=1) / np.sqrt(len(filtered))
        difference = filtered.mean() - reference.mean()
        ref_control = np.array([run["prefixes"][index]["control_posterior"]["mean"] for run in reference_runs])
        pf_control = np.array([run["prefixes"][index]["control_posterior"]["mean"] for run in filter_runs])
        # Self-normalized moments have finite-sample bias; report replicate
        # variability and avoid labelling this comparison an exact oracle.
        comparisons.append(dict(
            trials=index + 1,
            reference_joint_mean=float(reference.mean()), reference_joint_mean_se=float(ref_se),
            filter_joint_mean=float(filtered.mean()), filter_joint_mean_se=float(pf_se),
            difference=float(difference), combined_standard_error=float(np.hypot(ref_se, pf_se)),
            standardized_difference=float(difference / np.hypot(ref_se, pf_se)),
            reference_log_of_mean=float(np.log(reference.mean())), filter_log_of_mean=float(np.log(filtered.mean())),
            reference_mean_log=float(np.log(reference).mean()), filter_mean_log=float(np.log(filtered).mean()),
            reference_control_mean=ref_control.mean(axis=0).tolist(), filter_control_mean=pf_control.mean(axis=0).tolist(),
            reference_control_mean_se=(ref_control.std(axis=0, ddof=1) / np.sqrt(len(reference))).tolist(),
            filter_control_mean_se=(pf_control.std(axis=0, ddof=1) / np.sqrt(len(filtered))).tolist(),
        ))
    return comparisons


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, choices=(2, 3), default=3)
    parser.add_argument("--reference-estimates", type=int, default=200000)
    parser.add_argument("--filter-estimates", type=int, default=100000)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--model-seed", type=int, default=29)
    parser.add_argument("--data-seed", type=int, default=20260929)
    parser.add_argument("--observation-seed", type=int, default=4381)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--rt-upper", type=float, default=3.)
    parser.add_argument("--sigma", type=float, default=.5)
    parser.add_argument("--pseudocount-at-100000", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report path.")
    if min(args.reference_estimates, args.filter_estimates, args.replicates) < 2:
        parser.error("At least two estimates and independent replicates are required.")
    if args.bins < 1 or args.rt_upper <= 0 or args.sigma < 0 or args.pseudocount_at_100000 < 0:
        parser.error("Invalid observation-kernel configuration.")

    import torch
    from psyneulink.core.batched import BatchedCompositionCompiler
    from psyneulink.core.batched.backend.triton import conditioned
    from psyneulink.core.globals.utilities import set_global_seed
    from dawa_batched_simulation import SOURCE, build_model, fit_surface, node

    torch.set_num_threads(4)
    set_global_seed(args.model_seed)
    # The source sequence helper expects an even count; use its ordered prefix.
    model, inputs, outputs = build_model(trials=4, c_noise=.1, s_noise=.1, d_noise=.1, r_noise=.1)
    inputs = {key: np.asarray(value)[:args.trials] for key, value in inputs.items()}
    outputs = (*outputs, node(model, "Control Units\n[Color, Location]").output_port)
    plan = BatchedCompositionCompiler.compile(model, backend="triton", outputs=outputs, max_steps=2000)
    parameters = {f"{mechanism.name}.{parameter}": values[2]
                  for (parameter, mechanism), values in fit_surface(model).items()}
    plan = plan.specialize_parameters({parameter.name: parameter.default for parameter in plan.ir.params
                                       if parameter.name not in parameters})
    launch = dict(block_size=32, num_warps=1, trial_schedule="independent", normal_rng="philox4x_fast_v1")
    simulation = dict(strict_truncation=True, triton_launch_options=launch)
    latent = plan.run(inputs, [parameters], 1, seed=args.data_seed, **simulation).values[0, 0, :, 0, :2]
    kernel = dict(edges=observation_edges(args.bins, (0., args.rt_upper), device="cuda").tolist(),
                  bins=args.bins, rt_range=(0., args.rt_upper), sigma=args.sigma,
                  alpha_per_estimate=args.pseudocount_at_100000 / 100000.)
    observed = noisy_observations(latent, np.random.default_rng(args.observation_seed), **kernel)
    report = dict(status="running", gpu=torch.cuda.get_device_name(), hostname=platform.node(),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arguments={**vars(args), "output": str(args.output)}, parameters=parameters,
                  inputs={key.name: value.tolist() for key, value in inputs.items()},
                  latent_choices_rt=latent.tolist(), observations=observed.tolist(), observation_kernel=kernel,
                  control_observable="Control Units RESULT activation [Color, Location] at trial end",
                  interpretation="Independent whole-history importance sampling is a finite Monte Carlo reference, not exact truth. "
                  "Compare probability-scale estimates; mean log likelihoods have Jensen bias. Terminal posterior moments are self-normalized. "
                  "The exact FP32 device bin coordinates are exported as shared observation definitions; "
                  "bin lookup, Gaussian normalization, contamination, and reference likelihood arithmetic are independently implemented in NumPy float64.",
                  reference_runs=[], filter_runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_report(args.output, report)
    for replicate in range(args.replicates):
        seed = 1001 + replicate
        start = time.perf_counter()
        values = plan.run(inputs, [parameters], args.reference_estimates, seed=seed, **simulation).values[0, 0]
        prefixes = full_history_estimate(values, observed, **kernel)
        report["reference_runs"].append(dict(seed=seed, seconds=time.perf_counter() - start, prefixes=prefixes))
        save_report(args.output, report)
        print(json.dumps(dict(kind="reference", replicate=replicate, joint=prefixes[-1]["joint_likelihood"],
                              ess=prefixes[-1]["effective_sample_size"])), flush=True)
    original_factory = conditioned.prepare_conditioned_runner
    for replicate in range(args.replicates):
        captured = []

        def capture_factory(*positional, **keywords):
            runner = original_factory(*positional, **keywords)

            def run(trial_index, states):
                result = runner(trial_index, states)
                captured.append(result.values.detach().cpu().numpy()[0, 0, 0].copy())
                return result

            return run

        seed = 5001 + replicate
        start = time.perf_counter()
        with patch.object(conditioned, "prepare_conditioned_runner", capture_factory):
            score, diagnostics = plan.conditioned_log_likelihood(
                inputs, [parameters], args.filter_estimates, data=observed, outcome_indices=[0, 1],
                categorical_dims=[0], categorical_cardinalities=[2], bins=args.bins,
                bin_range=[kernel["rt_range"]], smoothing_sigma=args.sigma,
                pseudocount=kernel["alpha_per_estimate"] * args.filter_estimates,
                seed=seed, return_diagnostics=True, execution="prepared", **simulation,
            )
        densities = np.asarray(diagnostics["per_trial_densities"])[0, 0]
        prefixes = []
        for trial, (values, observation) in enumerate(zip(captured, observed)):
            weights = numpy_observation_density(values[:, :2], observation, **kernel)
            np.testing.assert_allclose(weights.mean(), densities[trial], rtol=2e-6, atol=1e-7,
                                       err_msg="Independent NumPy observation kernel disagrees with GPU")
            prefixes.append(dict(trials=trial + 1, joint_likelihood=float(np.prod(densities[:trial + 1], dtype=float)),
                                 log_likelihood=float(np.log(densities[:trial + 1]).sum()),
                                 conditional_density=float(densities[trial]),
                                 effective_sample_size=float(np.asarray(diagnostics["effective_sample_size"])[0, 0, trial]),
                                 control_posterior=weighted_moments(values[:, 2:4], weights)))
        report["filter_runs"].append(dict(seed=seed, seconds=time.perf_counter() - start,
                                          log_likelihood=score, prefixes=prefixes))
        save_report(args.output, report)
        print(json.dumps(dict(kind="filter", replicate=replicate, joint=prefixes[-1]["joint_likelihood"],
                              ess=prefixes[-1]["effective_sample_size"])), flush=True)
    report["comparison"] = aggregate_comparison(report["reference_runs"], report["filter_runs"])
    report["status"] = "complete"
    save_report(args.output, report)
    print(json.dumps(report["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
