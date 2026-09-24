"""Oracles, refinement, Monte Carlo, and gradient audits for continuous DAWA."""

from dataclasses import replace
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import expit, ndtr
import torch

from .model import DEFAULT_PARAMETERS, PARAMETER_NAMES
from .continuous_model import continuous_path
from .continuous_solver import ContinuousConfig, ContinuousResponseSolver
from .continuous_likelihood import continuous_sequence_likelihood


def ode_reference_check(parameters=None, *, task=(1., 0.), stimulus=(0., 1., 0., 1.),
                        time_step=.002, horizon=.8, backend="torch"):
    """Independent adaptive DOP853 integration of the ten deterministic states."""
    p = np.asarray(DEFAULT_PARAMETERS if parameters is None else parameters)
    path_function = continuous_path
    if backend == "generated":
        from .continuous_native import continuous_path_native
        path_function = continuous_path_native
    elif backend != "torch":
        raise ValueError("Unknown ODE backend.")
    stimulus = np.asarray(stimulus)

    def rhs(t, q):
        c, s, d, v, w = q[:2], q[2:6], q[6:8], q[8], q[9]
        g = p[6] + p[5] * w
        ca, sa, da = expit(p[3] * c), expit(g * (s + p[2])), expit(g * (d + p[2]))
        dc = np.asarray(task) - 7. * c - 3. * ca[::-1]
        raw = np.array([stimulus[0] - stimulus[1], stimulus[1] - stimulus[0],
                        1.2 * (stimulus[2] - stimulus[3]), 1.2 * (stimulus[3] - stimulus[2])])
        ds = raw + 4. * np.repeat(ca, 2) - 8. * s - 8. * sa[[1, 0, 3, 2]]
        contrast = sa[0] - sa[1] + 1.2 * (sa[2] - sa[3])
        dd = 4. * ca.sum() + np.array([contrast, -contrast]) - 8. * d - 8. * da[::-1]
        dv = 20. * (-v**3 + 1.5 * v*v - .5 * v - w + .3 * da.sum()) / .05
        dw = 20. * (p[4] * v - w + (1. - p[4]) * .5) / 5.
        return np.r_[dc, ds, dd, dv, dw]

    steps = round(horizon / time_step)
    times = np.arange(steps + 1) * time_step
    reference = solve_ivp(rhs, (0., times[-1]), np.zeros(10), method="DOP853", t_eval=times,
                          rtol=2.e-12, atol=2.e-13)
    if not reference.success:
        raise AssertionError(reference.message)
    errors = []
    with torch.no_grad():
        for ode_step in (.001, .0005, .00025):
            path = path_function(torch.tensor(p), task, stimulus, steps=steps,
                                 time_step=time_step, ode_step=ode_step)
            errors.append({"ode_step": ode_step, "maximum_state_error": float(np.abs(path.states.numpy() - reference.y.T).max())})
    if errors[-1]["maximum_state_error"] > 1.e-6 or errors[-1]["maximum_state_error"] >= errors[0]["maximum_state_error"]:
        raise AssertionError(f"ODE refinement failed: {errors}")
    return errors


def independent_race_cdf(times, *, boundary=.15, drift=(.35, .25), noise=.2):
    """Two independent Brownian first passages: integral f_i(t) S_j(t)."""
    t = np.asarray(times)[1:]
    cdf, density = [], []
    for v in drift:
        cdf.append(ndtr((v * t - boundary) / (noise * np.sqrt(t)))
                   + np.exp(2. * v * boundary / noise**2) * ndtr(-(v * t + boundary) / (noise * np.sqrt(t))))
        density.append(boundary / (noise * np.sqrt(2. * np.pi * t**3))
                       * np.exp(-(boundary - v * t)**2 / (2. * noise**2 * t)))
    return np.stack([cumulative_trapezoid(np.r_[0., density[0] * (1. - cdf[1])], times, initial=0),
                     cumulative_trapezoid(np.r_[0., density[1] * (1. - cdf[0])], times, initial=0)], axis=1)


def analytic_race_check(*, device="cpu", points=(33, 65, 129)):
    t = torch.arange(601, dtype=torch.float64, device=device) * .001
    reference = independent_race_cdf(t.cpu().numpy())[1:]
    inputs = t.new_tensor([.4, .3]).expand(len(t), 2)
    records = []
    with torch.no_grad():
        for size in points:
            cfg = ContinuousConfig(points=size, time_step=.001, leak=0., competition=0., noise=.2, lower_bound=-.6)
            start = time.perf_counter()
            d = ContinuousResponseSolver(cfg).solve_coefficients(inputs, torch.ones_like(t), t.new_zeros(()), .15 + .05 * t, t * 0 + .05)
            actual = d.choice_mass.cumsum(0).cpu().numpy()
            records.append({"points": size, "maximum_joint_cdf_error": float(np.abs(actual - reference).max()),
                            "mass_error": float(d.mass_error), "lower_loss": float(d.lower_loss),
                            "minimum_mass": float(d.minimum_mass), "substeps": d.substeps,
                            "seconds": time.perf_counter() - start})
    if records[-1]["maximum_joint_cdf_error"] > 6.e-4 or max(r["mass_error"] for r in records) > 1.e-10:
        raise AssertionError(f"Analytic moving-boundary race validation failed: {records}")
    return records


def gradient_check(*, device="cpu"):
    p = torch.tensor((.18, .193, -.43, 11., .75, 1.3, 5.2), dtype=torch.float64, device=device, requires_grad=True)
    cfg = ContinuousConfig(points=25, time_step=.002)

    def objective(vector, config=cfg):
        return continuous_sequence_likelihood(vector, [[1, 0]] * 2, [[0, 1, 0, 1], [1, 0, 0, 1]],
                                              [1, 0], [.3437, .4117], config=config).log_likelihood

    start = time.perf_counter()
    value = objective(p)
    grad = torch.autograd.grad(value, p)[0].cpu().numpy()
    elapsed = time.perf_counter() - start
    finite = []
    with torch.no_grad():
        for i in range(7):
            h = 1.e-6 * max(1., float(p[i].abs()))
            plus, minus = p.detach().clone(), p.detach().clone()
            plus[i] += h
            minus[i] -= h
            finite.append(float((objective(plus) - objective(minus)) / (2. * h)))
    error = np.abs(grad - finite) / np.maximum(1., np.abs(finite))
    if not np.all(np.isfinite(grad)) or error.max() > 3.e-4:
        raise AssertionError(f"Continuous gradient check failed: {grad}, {finite}, {error}")
    raw = p.detach().clone().requires_grad_()
    plain = objective(raw, replace(cfg, checkpoint_steps=0))
    raw_grad = torch.autograd.grad(plain, raw)[0].cpu().numpy()
    np.testing.assert_allclose(raw_grad, grad, atol=2.e-10, rtol=2.e-10)
    return {"parameters": dict(zip(PARAMETER_NAMES, p.detach().cpu().tolist())), "log_likelihood": float(value.detach()),
            "autograd": grad.tolist(), "finite_difference": finite, "max_scaled_error": float(error.max()),
            "checkpoint_gradient_max_error": float(np.abs(raw_grad - grad).max()), "value_gradient_seconds": elapsed}


def gradient_refinement_check(*, device="cpu"):
    records = []
    for points, dt in ((33, .001), (65, .001), (129, .001), (129, .0005)):
        p = torch.tensor((.18, .193, -.43, 11., .75, 1.3, 5.2), dtype=torch.float64, device=device, requires_grad=True)
        start = time.perf_counter()
        result = continuous_sequence_likelihood(p, [[1, 0]] * 2, [[0, 1, 0, 1], [1, 0, 0, 1]], [1, 0],
                                                [.3437, .4117], config=ContinuousConfig(points=points, time_step=dt))
        grad = torch.autograd.grad(result.log_likelihood, p)[0].cpu().numpy()
        records.append({"points": points, "time_step": dt, "log_likelihood": float(result.log_likelihood.detach()),
                        "gradient": grad.tolist(), "value_gradient_seconds": time.perf_counter() - start})
        print(f"gradient refinement: points={points}, dt={dt}, completed", flush=True)
    reference = np.asarray(records[2]["gradient"])
    scale = np.maximum(1., np.abs(reference))
    errors = [float(np.max(np.abs(np.asarray(r["gradient"]) - reference) / scale)) for r in records]
    if errors[1] >= errors[0] or errors[-1] > 3.e-4 or not np.isfinite(errors).all():
        raise AssertionError(f"Gradient refinement failed: {records}")
    return {"evaluations": records, "scaled_errors_to_129_1ms": errors}


def device_gradient_check():
    records = []
    for device in ("cpu", "cuda"):
        p = torch.tensor((.18, .193, -.43, 11., .75, 1.3, 5.2), dtype=torch.float64, device=device, requires_grad=True)
        result = continuous_sequence_likelihood(p, [[1, 0]], [[0, 1, 0, 1]], [1], [.3437],
                                                config=ContinuousConfig(points=33, time_step=.002))
        grad = torch.autograd.grad(result.log_likelihood, p)[0].cpu().numpy()
        records.append({"device": device, "log_likelihood": float(result.log_likelihood.detach()), "gradient": grad.tolist()})
    np.testing.assert_allclose(records[0]["gradient"], records[1]["gradient"], atol=1.e-10, rtol=1.e-10)
    np.testing.assert_allclose(records[0]["log_likelihood"], records[1]["log_likelihood"], atol=1.e-12, rtol=1.e-12)
    if abs(records[0]["gradient"][1]) < 1.e-6:
        raise AssertionError("An RT interval inside one numerical cell must retain its NDT derivative.")
    return {"gpu": torch.cuda.get_device_name(), "evaluations": records}


def pde_refinement_check(*, device="cpu"):
    p = torch.tensor(DEFAULT_PARAMETERS, dtype=torch.float64, device=device)
    records, distributions = [], {}
    with torch.no_grad():
        for points, dt in ((33, .001), (65, .001), (129, .001), (257, .001), (65, .002), (65, .0005)):
            start = time.perf_counter()
            path = continuous_path(p, [1, 0], [0, 1, 0, 1], steps=round(1.2 / dt), time_step=dt)
            d = ContinuousResponseSolver(ContinuousConfig(points=points, time_step=dt)).solve(path, p)
            cdf = d.choice_mass.cumsum(0).cpu().numpy()
            distributions[points, dt] = cdf
            records.append({"points": points, "time_step": dt, "choice_probability": cdf[-1].tolist(),
                            "survival": float(d.survival), "mass_error": float(d.mass_error),
                            "lower_loss": float(d.lower_loss), "minimum_mass": float(d.minimum_mass),
                            "substeps": d.substeps, "maximum_cfl": d.maximum_cfl,
                            "forward_seconds": time.perf_counter() - start})
            print(f"refinement: points={points}, dt={dt}, completed", flush=True)
    spatial = [float(np.max(np.abs(distributions[n, .001] - distributions[257, .001]))) for n in (33, 65, 129)]
    temporal = [float(np.max(np.abs(distributions[65, .002] - distributions[65, .0005][3::4]))),
                float(np.max(np.abs(distributions[65, .001] - distributions[65, .0005][1::2])))]
    if not spatial[2] < spatial[1] < spatial[0] or temporal[1] >= temporal[0] or max(r["mass_error"] for r in records) > 1.e-10:
        raise AssertionError(f"PDE refinement failed: {spatial}, {temporal}")
    return {"evaluations": records, "spatial_cdf_errors_to_257": spatial, "temporal_cdf_errors_to_0_5ms": temporal}


def monte_carlo_check(*, device="cuda", estimates=100000):
    from .continuous_monte_carlo import simulate_continuous
    p = torch.tensor(DEFAULT_PARAMETERS, dtype=torch.float64, device=device)
    horizon = 1.2
    with torch.no_grad():
        path = continuous_path(p, [1, 0], [0, 1, 0, 1], steps=1200, time_step=.001)
        d = ContinuousResponseSolver(ContinuousConfig(points=257)).solve(path, p)
        reference = d.choice_mass.cumsum(0).cpu().numpy()
        times = np.arange(1, 1201) * .001
        records = []
        for dt in (.0005, .00025):
            path = continuous_path(p, [1, 0], [0, 1, 0, 1], steps=round(horizon / dt), time_step=dt)
            start = time.perf_counter()
            samples = simulate_continuous(path, p, estimates=estimates)
            empirical = np.stack([np.searchsorted(np.sort(samples[samples[:, 0] == c, 1]), times, side="right")
                                  / estimates for c in (0, 1)], axis=1)
            records.append({"time_step": dt, "estimates": estimates, "seed": 43,
                            "pde_points": 257, "pde_time_step": .001, "gpu": torch.cuda.get_device_name(),
                            "joint_cdf_max_error": float(np.abs(empirical - reference).max()),
                            "choice_probability": empirical[-1].tolist(), "survival": float(np.mean(samples[:, 0] < 0)),
                            "sample_seconds": time.perf_counter() - start})
            print(f"Monte Carlo: dt={dt}, completed", flush=True)
    if records[-1]["joint_cdf_max_error"] > 3. / np.sqrt(estimates) + .003:
        raise AssertionError(f"Continuous SDE validation failed: {records}")
    return records


def run_validation(report, save, *, device, estimates):
    for key, title, action in (
        ("ode", "Checking coupled ODE against adaptive integration", ode_reference_check),
        ("analytic_race", "Checking analytic moving-boundary Brownian race", lambda: analytic_race_check(device=device)),
        ("pde_refinement", "Refining the DAWA state and time grids", lambda: pde_refinement_check(device=device)),
        ("gradient", "Checking all seven gradients and history reconstruction", lambda: gradient_check(device=device)),
        ("gradient_refinement", "Refining likelihood gradients", lambda: gradient_refinement_check(device=device)),
        ("device_gradient", "Comparing CPU and GPU gradients", device_gradient_check),
        ("monte_carlo", "Checking independent continuous SDE samples", lambda: monte_carlo_check(device=device, estimates=estimates)),
    ):
        print(title + "...", flush=True)
        report[key] = action()
        save()
