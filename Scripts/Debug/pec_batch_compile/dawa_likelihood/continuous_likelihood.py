"""Sequential continuous-time choice/RT likelihood, using CSI-style RT bins."""

from dataclasses import dataclass
import math

import torch

from .continuous_model import continuous_path, advance_control
from .continuous_solver import ContinuousConfig, ContinuousResponseSolver


@dataclass(frozen=True)
class ContinuousLikelihoodResult:
    log_likelihood: torch.Tensor
    probability: torch.Tensor
    decision_times: torch.Tensor
    control_after_trial: torch.Tensor
    distributions: tuple


def continuous_sequence_likelihood(parameters, tasks, stimuli, choices, response_times, *,
                                   config=None, include=None, resolution=.001):
    """Score RT rounding intervals without added measurement noise.

    Each candidate supplies decision duration RT-ndt. Continuous control state
    advances for that duration, including excluded trials. As in CSI, history
    uses recorded RT bin centers rather than integrating all rounding-time
    uncertainty. Parameters can be shared [7] or tied through [trial,7] tensors.
    """
    cfg = config or ContinuousConfig()
    path_function, control_function = continuous_path, advance_control
    if cfg.ode_backend == "generated":
        from .continuous_native import continuous_path_native, advance_control_native
        path_function, control_function = continuous_path_native, advance_control_native
    n = len(tasks)
    if n < 1 or any(len(v) != n for v in (stimuli, choices, response_times)) or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Expected matching nonempty observations and positive RT resolution.")
    if parameters.shape == (7,):
        parameters = parameters.expand(n, 7)
    if parameters.shape != (n, 7):
        raise ValueError("Expected parameters[7] or parameters[trial,7].")
    mask = torch.ones(n, dtype=torch.bool) if include is None else torch.as_tensor(include, dtype=torch.bool)
    if mask.shape != (n,) or not bool(mask.any()):
        raise ValueError("include must select at least one trial.")
    if any(float(choices[k]) not in (0., 1.) for k in range(n) if bool(mask[k])):
        raise ValueError("Choices must be 0 or 1 on included trials.")
    rt = torch.as_tensor(response_times, dtype=parameters.dtype, device=parameters.device)
    durations = rt - parameters[:, 1]
    if not bool(torch.isfinite(durations).all()) or bool((durations <= resolution / 2.).any()):
        raise FloatingPointError("Observed RT intervals must follow the candidate nondecision time.")
    control, probabilities, distributions, histories = parameters.new_zeros(2), [], [], []
    solver = ContinuousResponseSolver(cfg)
    for k in range(n):
        p, duration = parameters[k], durations[k]
        if bool(mask[k]):
            low, high = duration - resolution / 2., duration + resolution / 2.
            # One extra cell supplies the right neighbor for the conservative
            # linear reconstruction of density within the final scored cell.
            steps = math.ceil(float(high.detach()) / cfg.time_step) + 1
            path = path_function(p, tasks[k], stimuli[k], steps=steps, time_step=cfg.time_step,
                                   ode_step=cfg.ode_step, clock_ratio=cfg.lc_clock_ratio, control=control)
            distribution = solver.solve(path, p)
            probabilities.append(distribution.interval_probability(choices[k], low, high))
        else:
            distribution = None
            probabilities.append(p.new_tensor(float("nan")))
        control = control_function(control, p, tasks[k], duration)
        histories.append(control)
        distributions.append(distribution)
    probability = torch.stack(probabilities)
    return ContinuousLikelihoodResult(probability[mask.to(parameters.device)].log().sum(), probability, durations,
                                      torch.stack(histories), tuple(distributions))
