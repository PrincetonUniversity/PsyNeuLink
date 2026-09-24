"""Convergence of the scheduled DAWA source toward its continuous extension.

This study uses the actual PNL graph and ordinary batch compiler for scheduled
simulations. Both integration steps shrink together, preserving ten internal LC
updates and an LC/LCA clock ratio of 20. Fresh conditions are separate subjects,
so no unobserved stochastic history is substituted into the comparison.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from dawa_likelihood import DEFAULT_PARAMETERS, PARAMETER_NAMES
from dawa_likelihood.continuous_model import continuous_path
from dawa_likelihood.continuous_monte_carlo import simulate_continuous
from dawa_likelihood.continuous_solver import ContinuousConfig, ContinuousResponseSolver
from dawa_likelihood.model import response_path, initial_history


PARAMETERS = (DEFAULT_PARAMETERS, (.28, .17, -.43, 12., .6, 1.5, 5.4))
STEPS = (.01, .005, .0025, .00125, .000625, .0003125)
SIMULATION_STEPS = STEPS + (.00015625, .000078125)
CONDITIONS = (("color_congruent", (1., 0.), (0., 1., 0., 1.)),
              ("color_incongruent", (1., 0.), (1., 0., 0., 1.)),
              ("location_congruent", (0., 1.), (0., 1., 0., 1.)),
              ("location_incongruent", (0., 1.), (1., 0., 0., 1.)))


def empirical_cdf(samples, times):
    """Joint choice/time CDF, preserving censoring in the denominator."""
    times = np.asarray(times)
    return np.stack([np.searchsorted(np.sort(samples[samples[:, 0] == c, 1]), times + 1.e-7, side="right")
                     / len(samples) for c in (0, 1)], axis=1)


def summarize_samples(samples):
    completed = samples[:, 0] >= 0
    return {"choice_probability": [float(np.mean(samples[:, 0] == c)) for c in (0, 1)],
            "censored_fraction": float(np.mean(~completed)),
            "mean_decision_time_among_completed": float(samples[completed, 1].mean()) if completed.any() else None}


def source_graph(dt, *, deterministic=False):
    from dawa_batched_simulation import DEFAULTS, source_module, node
    c = source_module().make_lca_model(**{**DEFAULTS, "time_step_size": dt, "r_noise": 0. if deterministic else .1})
    node(c, "LC").integrator_function.parameters.time_step_size.set(2. * dt)
    inputs = {node(c, "Task Input"): np.asarray([x[1] for x in CONDITIONS]),
              node(c, "Stimulus Input"): np.asarray([x[2] for x in CONDITIONS])}
    inputs.update({node(c, name): np.zeros((len(CONDITIONS), 1)) for name in ("Bias Mechanism", "w1 Mechanism", "w2 Mechanism")})
    outputs = tuple(node(c, name).output_port for name in ("DECISION_GATE", "RT_GATE"))
    candidates = []
    for p in PARAMETERS:
        pairs = (("Response Units\n[Left, Right]", "termination_threshold", p[0]), ("RT_GATE", "intercept", p[1]),
                 ("Bias Mechanism", "intercept", p[2]), ("Control Units\n[Color, Location]", "gain", p[3]),
                 ("LC", "mode", p[4]), ("LC", "slope", p[5]), ("LC", "intercept", p[6]))
        candidates.append({f"{node(c, name).name}.{parameter}": value for name, parameter, value in pairs})
    return c, inputs, outputs, candidates


def trajectory_study():
    """Two forced-duration trials exercise carried control and all trial resets."""
    from dawa_likelihood.validation import native_replay_error
    parity = []
    for dt in (STEPS[0], STEPS[2], STEPS[-1]):
        error = native_replay_error(PARAMETERS[1], trials=2, steps=4, time_step=dt)
        if error > 2.e-12:
            raise AssertionError(f"Refined replay/source parity failed at dt={dt}: {error}")
        parity.append({"time_step": dt, "maximum_native_error": error})
    records, arrays = [], {}
    duration, finest = .4, STEPS[-1]
    common_times = np.arange(1, 81) * .01
    arrays["trajectory_times"] = common_times
    for index, values in enumerate(PARAMETERS):
        p = torch.tensor(values, dtype=torch.float64)
        reference_states, reference_gain, reference_drive = [], [], []
        control = None
        with torch.no_grad():
            for _, task, stimulus in (CONDITIONS[0], CONDITIONS[3]):
                ref = continuous_path(p, task, stimulus, steps=round(duration / finest), time_step=finest,
                                      ode_step=.000125, control=control)
                stride = round(.01 / finest)
                reference_states.append(ref.states[stride::stride].numpy())
                reference_gain.append(ref.gain[stride::stride].numpy())
                reference_drive.append(ref.inputs[stride::stride].numpy())
                control = ref.states[-1, :2]
            target, gains, drive = np.concatenate(reference_states), np.concatenate(reference_gain), np.concatenate(reference_drive)
            arrays[f"trajectory_{index}_continuous"] = np.c_[target, gains, drive]
            for dt in STEPS:
                history, states, actual_gain, actual_drive = None, [], [], []
                for _, task, stimulus in (CONDITIONS[0], CONDITIONS[3]):
                    path = response_path(p, task, stimulus, round(duration / dt), history, time_step=dt)
                    stride = round(.01 / dt)
                    selected = np.arange(stride - 1, len(path.gain), stride)
                    c = torch.stack([path.history[k].control_pre for k in selected])
                    held = torch.stack([path.history[k].sampled_sd_gain for k in selected])
                    s = torch.logit(path.stimulus[selected]) / held[:, None] - p[2]
                    d = torch.logit(path.decision[selected]) / held[:, None] - p[2]
                    states.append(torch.cat((c, s, d, path.lc[selected]), dim=1).numpy())
                    actual_gain.append(path.gain[selected].numpy())
                    actual_drive.append(path.inputs[selected].numpy())
                    history = path.history[-1]
                actual, g, u = np.concatenate(states), np.concatenate(actual_gain), np.concatenate(actual_drive)
                record = {"parameter_set": index, "time_step": dt, "lc_internal_step": 2. * dt,
                          "maximum_state_error": float(np.abs(actual - target).max()),
                          "state_rms_error": float(np.sqrt(np.mean((actual - target)**2))),
                          "maximum_gain_error": float(np.abs(g - gains).max()),
                          "maximum_drive_error": float(np.abs(u - drive).max())}
                records.append(record)
                arrays[f"trajectory_{index}_{dt:g}"] = np.c_[actual, g, u]
                print(f"trajectory p={index} dt={dt:g}: max state error={record['maximum_state_error']:.6g}", flush=True)
    for index in range(len(PARAMETERS)):
        errors = [r["state_rms_error"] for r in records if r["parameter_set"] == index]
        if errors[-1] >= errors[0] / 10.:
            raise AssertionError(f"Trajectory refinement did not converge: {errors}")
    return {"native_parity": parity, "trial_duration": duration, "trial_conditions": [0, 3],
            "sampling_interval": .01, "records": records}, arrays


def startup_study():
    """Separate LC internal integration and initial allocation from outer dt."""
    p = torch.tensor(PARAMETERS[0], dtype=torch.float64)
    cases = ((.01, 10, 1.), (.01, 100, 1.), (.01, 10, 5.), (.0025, 10, 1.), (.0003125, 10, 1.))
    records, arrays = [], {}
    with torch.no_grad():
        ref = continuous_path(p, [1, 0], [0, 1, 0, 1], steps=1600, time_step=.00025, ode_step=.000125)
        arrays["startup_continuous"] = np.c_[np.arange(1601) * .00025, ref.states[:, 8].numpy(), ref.gain.numpy()]
        for index, (dt, substeps, gain) in enumerate(cases):
            history = replace(initial_history(p), held_gain=p.new_tensor(gain))
            path = response_path(p, [1, 0], [0, 1, 0, 1], round(.4 / dt), history, time_step=dt, lc_substeps=substeps)
            arrays[f"startup_{index}"] = np.c_[np.arange(1, len(path.gain) + 1) * dt, path.lc[:, 0].numpy(), path.gain.numpy()]
            records.append({"outer_step": dt, "lc_substeps": substeps, "lc_internal_step": 20. * dt / substeps,
                            "initial_held_gain": gain, "peak_lc_v": float(path.lc[:, 0].max()), "peak_gain": float(path.gain.max())})
    return {"records": records, "continuous_peak_lc_v": float(ref.states[:, 8].max()),
            "continuous_peak_gain": float(ref.gain.max()),
            "note": "Controlled replay ablations; the source schedule and default initial gain remain unchanged."}, arrays


@dataclass(frozen=True)
class StudyConfig:
    estimates: int = 100000
    horizon: float = 2.
    seed: int = 431
    continuous_step: float = .00025


def continuous_samples(path, p, config, *, seed):
    # Independent chunks keep every Triton RNG counter within uint32 range.
    chunk = min(config.estimates, (2**32 - 1) // (4 * (len(path.gain) - 1)))
    if chunk < 1:
        raise ValueError("Simulation horizon is too long for the RNG counter.")
    parts = []
    for start in range(0, config.estimates, chunk):
        parts.append(simulate_continuous(path, p, estimates=min(chunk, config.estimates - start),
                                         seed=seed + 100003 * (start // chunk)))
    return np.concatenate(parts)


def simulation_study(config, directory, save):
    from psyneulink.core.batched import BatchedCompositionCompiler
    times = np.arange(1, round(config.horizon / .001) + 1) * .001
    arrays = {"cdf_times": times}
    records, references = [], []
    for dt in SIMULATION_STEPS:
        cache = directory / f"scheduled_{dt:g}.npz"
        begin = time.perf_counter()
        if cache.exists():
            values = np.load(cache)["values"]
            metadata = json.loads(cache.with_suffix(".json").read_text())
        else:
            c, inputs, outputs, candidates = source_graph(dt)
            plan = BatchedCompositionCompiler.compile(c, backend="triton", outputs=outputs, max_steps=round(config.horizon / dt))
            warmup = plan.run(inputs, candidates, config.estimates, subject_slices=[slice(i, i + 1) for i in range(len(CONDITIONS))],
                              seed=config.seed, strict_truncation=False)
            compile_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            result = plan.run(inputs, candidates, config.estimates,
                              subject_slices=[slice(i, i + 1) for i in range(len(CONDITIONS))], seed=config.seed,
                              strict_truncation=False)
            metadata = {"warm_simulation_seconds": time.perf_counter() - begin, "setup_and_warmup_seconds": compile_seconds,
                        "truncation": result.metadata["truncation"], "warmup_shape": list(warmup.values.shape)}
            values = result.values[:, :, 0]
            np.savez_compressed(cache, values=values)
            cache.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        for pi, p in enumerate(PARAMETERS):
            for ci, (name, _, _) in enumerate(CONDITIONS):
                samples = values[pi, ci].astype(float).copy()
                completed = samples[:, 1] > p[1]
                samples[~completed, 0] = -1
                samples[:, 1] = np.where(completed, samples[:, 1] - p[1], config.horizon)
                arrays[f"scheduled_{pi}_{ci}_{dt:g}"] = empirical_cdf(samples, times)
                records.append({"parameter_set": pi, "condition": name, "condition_index": ci, "time_step": dt,
                                "lc_internal_step": 2. * dt, **summarize_samples(samples)})
        print(f"source batch dt={dt:g}: {metadata['warm_simulation_seconds']:.3f}s", flush=True)
        save({"scheduled": records}, arrays)
    with torch.no_grad():
        for pi, values in enumerate(PARAMETERS):
            p = torch.tensor(values, dtype=torch.float64)
            for ci, (name, task, stimulus) in enumerate(CONDITIONS):
                cache = directory / f"continuous_{pi}_{ci}.npz"
                if cache.exists():
                    samples = np.load(cache)["samples"]
                else:
                    path = continuous_path(p, task, stimulus, steps=round(config.horizon / config.continuous_step),
                                           time_step=config.continuous_step, ode_step=.000125)
                    samples = continuous_samples(path, p, config, seed=config.seed + 1009 * (1 + 4 * pi + ci))
                    np.savez_compressed(cache, samples=samples)
                cdf = empirical_cdf(samples, times)
                arrays[f"continuous_{pi}_{ci}"] = cdf
                references.append({"parameter_set": pi, "condition": name, "condition_index": ci,
                                   "time_step": config.continuous_step, **summarize_samples(samples)})
                for record in records:
                    if record["parameter_set"] == pi and record["condition_index"] == ci:
                        actual = arrays[f"scheduled_{pi}_{ci}_{record['time_step']:g}"]
                        record["joint_cdf_max_error"] = float(np.abs(actual - cdf).max())
                        record["joint_cdf_integrated_absolute_error"] = float(np.abs(actual - cdf).sum() * .001)
                print(f"continuous reference p={pi} {name}: completed", flush=True)
                save({"scheduled": records, "continuous": references}, arrays)
    for pi in range(len(PARAMETERS)):
        for ci in range(len(CONDITIONS)):
            local = [r for r in records if r["parameter_set"] == pi and r["condition_index"] == ci]
            if local[-1]["joint_cdf_max_error"] > max(.02, local[0]["joint_cdf_max_error"] * .6):
                raise AssertionError(f"Stochastic convergence needs investigation: {local}")
    return {"scheduled": records, "continuous": references,
            "sampling_note": "Independent source/reference streams; differences below roughly 0.006 at 100k draws can be sampling noise."}, arrays


def density_study(config, arrays):
    """Selected cases audit the PDE reference separately from source refinement."""
    records = []
    with torch.no_grad():
        for pi, ci in ((0, 0), (1, 1)):
            p = torch.tensor(PARAMETERS[pi], dtype=torch.float64)
            _, task, stimulus = CONDITIONS[ci]
            path = continuous_path(p, task, stimulus, steps=round(config.horizon / .001), time_step=.001)
            for points in (129, 257):
                begin = time.perf_counter()
                result = ContinuousResponseSolver(ContinuousConfig(points=points)).solve(path, p)
                cdf = result.choice_mass.cumsum(0).numpy()
                arrays[f"density_{pi}_{ci}_{points}"] = cdf
                record = {"parameter_set": pi, "condition_index": ci, "points": points, "time_step": .001,
                          "seconds": time.perf_counter() - begin, "mass_error": float(result.mass_error),
                          "lower_loss": float(result.lower_loss), "survival": float(result.survival),
                          "minimum_mass": float(result.minimum_mass), "substeps": result.substeps,
                          "joint_cdf_max_error": float(np.abs(cdf - arrays[f"continuous_{pi}_{ci}"]).max())}
                records.append(record)
                print(f"density p={pi} c={ci} points={points}: CDF error={record['joint_cdf_max_error']:.6g}", flush=True)
    return records


def plot_results(report, arrays, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    if "startup" in report:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for col, (coordinate, label) in enumerate(((1, "LC fast state v"), (2, "Gain"))):
            ref = arrays["startup_continuous"]
            axes[col].plot(ref[:, 0], ref[:, coordinate], color="black", lw=2, label="Continuous")
            for i, r in enumerate(report["startup"]["records"]):
                row = arrays[f"startup_{i}"]
                label_text = f"Outer {r['outer_step']*1000:g} ms; LC {r['lc_substeps']} steps; initial gain {r['initial_held_gain']:g}"
                axes[col].plot(row[:, 0], row[:, coordinate], label=label_text)
            axes[col].set(xlabel="Time since trial onset (s)", ylabel=label, xlim=(0., .25))
        axes[0].legend(fontsize=7)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"startup_ablation.{extension}", dpi=160)
        plt.close(fig)
    if "trajectory" in report:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        times = arrays["trajectory_times"]
        for pi in range(2):
            reference = arrays[f"trajectory_{pi}_continuous"]
            for col, (coordinate, label) in enumerate(((8, "LC fast state v"), (10, "Published gain"))):
                ax = axes[pi, col]
                ax.plot(times, reference[:, coordinate], color="black", lw=2, label="Continuous")
                for dt in (STEPS[0], STEPS[2], STEPS[-1]):
                    ax.plot(times, arrays[f"trajectory_{pi}_{dt:g}"][:, coordinate], label=f"Scheduled {1000*dt:g} ms")
                ax.axvline(.4, color="gray", ls=":")
                ax.set(xlabel="Time (s)", ylabel=label, title=f"Parameter set {pi}; reset/task switch at 0.4 s")
            local = [r for r in report["trajectory"]["records"] if r["parameter_set"] == pi]
            axes[pi, 2].loglog([r["time_step"] * 1000 for r in local], [r["state_rms_error"] for r in local], "o-")
            axes[pi, 2].set(xlabel="LCA step (ms)", ylabel="RMS error of 10 internal states", title="Fixed-time trajectory convergence")
        axes[0, 0].legend(fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"trajectory_convergence.{extension}", dpi=160)
        plt.close(fig)
    if "simulation" in report and "continuous" in report["simulation"]:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
        times = arrays["cdf_times"]
        for pi in range(2):
            for ci, (name, _, _) in enumerate(CONDITIONS):
                ax = axes[pi, ci]
                ax.plot(times, arrays[f"continuous_{pi}_{ci}"].sum(1), color="black", lw=2, label="Continuous SDE")
                for dt in (SIMULATION_STEPS[0], SIMULATION_STEPS[2], SIMULATION_STEPS[-1]):
                    ax.plot(times, arrays[f"scheduled_{pi}_{ci}_{dt:g}"].sum(1), label=f"Scheduled {dt*1000:g} ms")
                if f"density_{pi}_{ci}_257" in arrays:
                    ax.plot(times, arrays[f"density_{pi}_{ci}_257"].sum(1), ls="--", color="tab:red", label="257×257 PDE")
                ax.set(xlabel="Decision time (s)", ylabel="Cumulative response probability", title=f"Set {pi}: {name.replace('_', ' ')}")
        axes[0, 0].legend(fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"response_cdf_convergence.{extension}", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for pi, ax in enumerate(axes):
            for ci, (name, _, _) in enumerate(CONDITIONS):
                rows = [r for r in report["simulation"]["scheduled"] if r["parameter_set"] == pi and r["condition_index"] == ci]
                ax.loglog([r["time_step"] * 1000 for r in rows], [r["joint_cdf_max_error"] for r in rows], "o-", label=name.replace('_', ' '))
            ax.axhspan(1.e-4, .006 * np.sqrt(100000 / report["configuration"]["estimates"]), color="gray", alpha=.12)
            ax.set(xlabel="LCA step (ms)", ylabel="Maximum joint choice/time CDF difference", title=f"Parameter set {pi}", ylim=(.001, .1))
        axes[0].legend(fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"distribution_error_convergence.{extension}", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dawa_continuous_study"))
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--horizon", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--stages", nargs="+", choices=("trajectory", "simulation", "density", "startup", "plots"),
                        default=("trajectory", "simulation", "density", "startup", "plots"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = StudyConfig(estimates=args.estimates, horizon=args.horizon, seed=args.seed)
    if config.estimates < 1 or not np.isfinite(config.horizon) or config.horizon <= 0:
        parser.error("Positive estimates and horizon are required")
    if any(abs(config.horizon / dt - round(config.horizon / dt)) > 1.e-8 for dt in (*SIMULATION_STEPS, .001)):
        parser.error("Horizon must be a multiple of every study time step")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path, arrays_path = args.output / "results.json", args.output / "curves.npz"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["configuration"] != asdict(config):
            parser.error("Existing output directory has a different configuration; use another directory")
        expected_parameters = [dict(zip(PARAMETER_NAMES, p)) for p in PARAMETERS]
        expected_conditions = [{"name": n, "task": list(t), "stimulus": list(s)} for n, t, s in CONDITIONS]
        if report["parameter_sets"] != expected_parameters or report["conditions"] != expected_conditions or report["lc_clock_ratio"] != 20.:
            parser.error("Existing output directory has different model parameters or conditions; use another directory")
        arrays = dict(np.load(arrays_path)) if arrays_path.exists() else {}
    else:
        report = {"configuration": asdict(config), "lc_clock_ratio": 20., "lc_updates_per_pass": 10,
                  "parameter_sets": [dict(zip(PARAMETER_NAMES, p)) for p in PARAMETERS],
                  "conditions": [{"name": n, "task": t, "stimulus": s} for n, t, s in CONDITIONS],
                  "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
        arrays = {}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(arrays_path, **arrays)

    if "trajectory" in args.stages:
        report["trajectory"], new_arrays = trajectory_study()
        arrays.update(new_arrays)
        save()
    if "simulation" in args.stages:
        def progress(record, new_arrays):
            report["simulation"] = record
            arrays.update(new_arrays)
            save()
        report["simulation"], new_arrays = simulation_study(config, args.output, progress)
        arrays.update(new_arrays)
        save()
    if "density" in args.stages:
        if "continuous" not in report.get("simulation", {}):
            parser.error("Density checks require a completed simulation stage")
        report["density"] = density_study(config, arrays)
        save()
    if "startup" in args.stages:
        report["startup"], new_arrays = startup_study()
        arrays.update(new_arrays)
        save()
    if "plots" in args.stages:
        plot_results(report, arrays, args.output)
    print(f"Saved study to {args.output}", flush=True)


if __name__ == "__main__":
    main()
