"""Independent native replay, Monte Carlo, refinement, and gradient checks."""

from dataclasses import replace
import time

import numpy as np
import torch

from .model import DEFAULT_PARAMETERS, PARAMETER_NAMES, DT, response_path
from .solver import ResponseSolver, SolverConfig
from .likelihood import sequence_likelihood


def native_replay_error(values=DEFAULT_PARAMETERS, *, trials=3, steps=6, time_step=DT, lc_clock_ratio=20.):
    """Compare every pass against native Python, including trial resets."""
    import psyneulink as pnl
    from dawa_batched_simulation import DEFAULTS, source_module, node

    values = np.asarray(values, dtype=float)
    params = {**DEFAULTS, "sdr_bias": values[2], "c_gain": values[3], "lc_mode": values[4],
              "lc_scaling": values[5], "lc_base_gain": values[6], "r_noise": 0., "r_threshold": .99,
              "time_step_size": time_step}
    source = source_module()
    composition = source.make_lca_model(**params)
    node(composition, "LC").integrator_function.parameters.time_step_size.set(lc_clock_ratio * time_step / 10.)
    tasks, stimuli = source.conflict_task_sequence(trials, seed=3)
    inputs = {node(composition, "Task Input"): tasks, node(composition, "Stimulus Input"): stimuli}
    inputs.update({node(composition, name): np.zeros((trials, 1))
                   for name in ("Bias Mechanism", "w1 Mechanism", "w2 Mechanism")})
    names = ("Control Units\n[Color, Location]", "Stimulus Units\n[Red, Blue, Left, Right]",
             "Decision Units\n[Left, Right]", "Response Units\n[Left, Right]")
    rows = []

    def collect():
        row = [np.asarray(node(composition, name).parameters.value.get(composition)).ravel() for name in names]
        row.append(np.asarray(node(composition, "LC").integrator_function.parameters.value.get(composition)).ravel()[:2])
        rows.append(np.concatenate(row))

    composition.run(inputs, termination_processing={pnl.TimeScale.TRIAL: pnl.AfterNPasses(steps)}, call_after_pass=collect)
    p = torch.tensor(values, dtype=torch.float64)
    history, expected = None, []
    with torch.no_grad():
        for task, stimulus in zip(tasks, stimuli):
            path = response_path(p, task, stimulus, steps, history, time_step=time_step, lc_clock_ratio=lc_clock_ratio)
            pre, act = p.new_zeros(2), path.initial_activity
            for k in range(steps):
                pre = pre + time_step * (path.inputs[k] - 8. * act.flip(0) - 8. * pre)
                act = torch.sigmoid(path.gain[k] * (pre + p[2]))
                expected.append(torch.cat((path.control[k], path.stimulus[k], path.decision[k], act, path.lc[k])).numpy())
            history = path.history[-1]
    return float(np.max(np.abs(np.asarray(rows) - np.asarray(expected))))


def finite_difference_check(*, device="cpu", points=65, max_steps=105, trials=2):
    """Check every fitted coordinate, including RT observation and history."""
    p = torch.tensor((.31, .195, -.43, 11., .75, 1.3, 5.2), dtype=torch.float64, device=device, requires_grad=True)
    tasks = [[1., 0.]] * trials
    stimuli = [[0., 1., 0., 1.], [1., 0., 0., 1.]][:trials]
    choices, rts, history = [1, 0][:trials], [.98, 1.02][:trials], [78, 82][:trials]
    config = SolverConfig(points=points)

    def evaluate(vector, local_config=config):
        return sequence_likelihood(vector, tasks, stimuli, choices, rts, history, max_steps=max_steps,
                                   config=local_config).log_likelihood

    start = time.perf_counter()
    value = evaluate(p)
    gradient = torch.autograd.grad(value, p)[0]
    if device == "cuda":
        torch.cuda.synchronize()
    gradient_seconds = time.perf_counter() - start
    finite = []
    with torch.no_grad():
        for i in range(7):
            h = 1.e-5 * max(1., abs(float(p[i])))
            plus, minus = p.detach().clone(), p.detach().clone()
            plus[i] += h
            minus[i] -= h
            finite.append(float((evaluate(plus) - evaluate(minus)) / (2. * h)))
    finite = np.asarray(finite)
    actual = gradient.detach().cpu().numpy()
    relative = np.abs(actual - finite) / np.maximum(1., np.abs(finite))
    if not np.all(np.isfinite(actual)) or np.max(relative) > 2.e-4:
        raise AssertionError(f"Autograd/finite difference mismatch: {actual}, {finite}")
    # Checkpointing must not change either the forward objective or its gradient.
    # The uncheckpointed reference deliberately uses a small grid: retaining
    # every Gaussian transition for two trials otherwise exhausts small GPUs.
    check_config = replace(config, points=min(points, 33))
    checked = p.detach().clone().requires_grad_()
    checked_gradient = torch.autograd.grad(evaluate(checked, check_config), checked)[0]
    plain = p.detach().clone().requires_grad_()
    uncheckpointed = evaluate(plain, replace(check_config, checkpoint_steps=0))
    plain_gradient = torch.autograd.grad(uncheckpointed, plain)[0]
    torch.testing.assert_close(plain_gradient, checked_gradient, atol=1.e-10, rtol=1.e-10)
    return {"parameters": dict(zip(PARAMETER_NAMES, p.detach().cpu().tolist())),
            "log_likelihood": float(value.detach()), "autograd": actual.tolist(), "finite_difference": finite.tolist(),
            "max_scaled_error": float(relative.max()), "value_and_gradient_seconds": gradient_seconds,
            "trials": trials, "points": points, "checkpoint_comparison_points": check_config.points,
            "checkpoint_gradient_max_error": float((plain_gradient - checked_gradient).abs().max())}


def simulator_samples(*, estimates=100000, seed=29):
    """Fresh first trial: no unobserved-history conditioning approximation."""
    from dawa_batched_simulation import build_model, node
    from psyneulink.core.batched import BatchedCompositionCompiler

    composition, inputs, outputs = build_model(trials=2)
    inputs = {n: np.asarray(v)[:1] for n, v in inputs.items()}
    inputs[node(composition, "Stimulus Input")] = np.asarray([[0., 1., 0., 1.]])
    plan = BatchedCompositionCompiler.compile(composition, backend="triton", outputs=outputs, max_steps=180)
    return plan.run(inputs, [{}], estimates, seed=seed, strict_truncation=True).values[0, 0, 0]


def gradient_grid_check(*, device="cuda"):
    """Refine spatial and simultaneous-crossing quadrature for the gradient."""
    records = []
    for points, winner in ((97, 24), (129, 24), (129, 48)):
        p = torch.tensor((.31, .195, -.43, 11., .75, 1.3, 5.2), dtype=torch.float64, device=device, requires_grad=True)
        start = time.perf_counter()
        result = sequence_likelihood(p, [[1., 0.]] * 2, [[0., 1., 0., 1.], [1., 0., 0., 1.]],
                                     [1, 0], [.98, 1.02], [78, 82], max_steps=105,
                                     config=SolverConfig(points=points, winner_points=winner))
        gradient = torch.autograd.grad(result.log_likelihood, p)[0].cpu().numpy()
        records.append({"points": points, "winner_points": winner, "value": float(result.log_likelihood.detach()),
                        "gradient": gradient.tolist(), "seconds": time.perf_counter() - start})
    fine = np.asarray(records[-1]["gradient"])
    scaled_error = float((np.abs(np.asarray(records[0]["gradient"]) - fine) / np.maximum(1., np.abs(fine))).max())
    if scaled_error > 2.e-5:
        raise AssertionError(f"Gradient quadrature refinement has not converged: {scaled_error}")
    return {"evaluations": records, "max_scaled_gradient_difference": scaled_error}


def refinement_check(*, device="cuda", points=(65, 97, 129), estimates=100000):
    samples = simulator_samples(estimates=estimates)
    step = np.rint((samples[:, 1] - .2) / DT).astype(int)
    if step.max() > 160:
        raise AssertionError("Validation horizon must be extended.")
    empirical = np.zeros((160, 2))
    np.add.at(empirical, (step - 1, samples[:, 0].astype(int)), 1. / estimates)
    p = torch.tensor(DEFAULT_PARAMETERS, dtype=torch.float64, device=device)
    path = response_path(p, [1., 0.], [0., 1., 0., 1.], 160)
    records, previous = [], None
    for size in points:
        start = time.perf_counter()
        with torch.no_grad():
            result = ResponseSolver(SolverConfig(points=size)).solve(path, p[0], p[2])
        if device == "cuda":
            torch.cuda.synchronize()
        pmf = result.choice_step.cpu().numpy()
        cdf_error = float(np.abs(np.cumsum(pmf - empirical, axis=0)).max())
        record = {"points": size, "forward_seconds": time.perf_counter() - start,
                  "choice_probability": pmf.sum(axis=0).tolist(),
                  "mean_rt": float((pmf.sum(axis=1) * (DT * np.arange(1, 161) + .2)).sum()),
                  "survival": float(result.survival), "lower_loss": float(result.lower_loss),
                  "mass_error": float(result.mass_error), "quadrature_defect": float(result.quadrature_defect),
                  "monte_carlo_joint_cdf_max_error": cdf_error}
        if previous is not None:
            record["previous_grid_joint_cdf_max_difference"] = float(np.abs(np.cumsum(pmf - previous, axis=0)).max())
        records.append(record)
        previous = pmf
    # Six standard-error upper envelope; deterministic refinement is checked
    # separately so sampling variability cannot conceal a coarse-grid error.
    if records[-1]["monte_carlo_joint_cdf_max_error"] > 3. / np.sqrt(estimates) + 2.e-4:
        raise AssertionError(f"Simulation CDF mismatch: {records[-1]}")
    if len(records) > 1 and records[-1]["previous_grid_joint_cdf_max_difference"] > 2.e-4:
        raise AssertionError("The two finest grids have not converged.")
    if records[-1]["mass_error"] > 1.e-10 or records[-1]["lower_loss"] > 1.e-8:
        raise AssertionError("Probability conservation/domain truncation failed.")
    return {"estimates": estimates, "seed": 29, "empirical_choice": empirical.sum(axis=0).tolist(),
            "empirical_mean_rt": float(samples[:, 1].mean()), "grids": records}
