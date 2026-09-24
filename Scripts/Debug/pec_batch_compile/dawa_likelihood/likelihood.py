"""Choice/RT scoring conditional on explicitly supplied stopping-step history."""

from dataclasses import dataclass

import torch

from .model import DT, response_path
from .solver import ResponseSolver, normal_interval


@dataclass(frozen=True)
class LikelihoodResult:
    log_likelihood: torch.Tensor
    probability: torch.Tensor
    distributions: tuple


def rt_bin_probability(distribution, choice, response_time, non_decision_time, *, resolution=.001, measurement_sd=.01):
    """An explicit Gaussian RT observation model, integrated over rounding bins.

    The latent model still stops only on 10 ms steps. Measurement noise affects
    the reported RT, never the latent network or its trial duration.
    """
    if measurement_sd <= 0 or resolution <= 0:
        raise ValueError("RT measurement SD and resolution must be positive.")
    if float(choice) not in (0., 1.):
        raise ValueError("Choices must be 0 or 1.")
    steps = torch.arange(1, len(distribution.choice_step) + 1, dtype=non_decision_time.dtype, device=non_decision_time.device)
    residual = response_time - (steps * DT + non_decision_time)
    bins = normal_interval((residual - resolution / 2.) / measurement_sd, (residual + resolution / 2.) / measurement_sd)
    return (distribution.choice_step[:, int(choice)] * bins).sum()


def sequence_likelihood(parameters, tasks, stimuli, choices, response_times, history_steps, *,
                        max_steps=160, config=None, include=None, resolution=.001, measurement_sd=.01,
                        observation="rt"):
    """Score trials conditional on known/fixed previous stopping steps.

    Parameters can be [7] or [trial, 7], allowing tied or condition-specific
    fitted parameters with ordinary autograd. Supplied history steps are fixed
    data, not rounded/detached parameter-dependent expressions. Inferring these
    steps from empirical RTs gives a plug-in history approximation, NOT the
    fully marginalized likelihood of a noisy-RT trial sequence.
    """
    if observation not in {"rt", "step"}:
        raise ValueError("observation must be 'rt' or 'step'.")
    n = len(tasks)
    if n < 1 or any(len(x) != n for x in (stimuli, choices, response_times, history_steps)):
        raise ValueError("All observation and history arrays must have equal positive length.")
    if parameters.shape == (7,):
        parameters = parameters.expand(n, 7)
    if parameters.shape != (n, 7):
        raise ValueError("Expected parameters[7] or parameters[trial, 7].")
    mask = torch.ones(n, dtype=torch.bool, device=parameters.device) if include is None else torch.as_tensor(include, dtype=torch.bool, device=parameters.device)
    if mask.shape != (n,) or not bool(mask.any()):
        raise ValueError("include must select at least one trial and have one Boolean per trial.")
    steps = torch.as_tensor(history_steps)
    if bool((steps != steps.round()).any()) or bool((steps < 1).any()) or bool((steps > max_steps).any()):
        raise ValueError("History steps must be integers in [1, max_steps].")
    solver = ResponseSolver(config)
    history, probabilities, distributions = None, [], []
    for k in range(n):
        p = parameters[k]
        length = max_steps if bool(mask[k]) and observation == "rt" else int(steps[k])
        path = response_path(p, tasks[k], stimuli[k], length, history)
        if bool(mask[k]):
            distribution = solver.solve(path, p[0], p[2])
            if observation == "step":
                if float(choices[k]) not in (0., 1.):
                    raise ValueError("Choices must be 0 or 1.")
                probabilities.append(distribution.choice_step[int(steps[k]) - 1, int(choices[k])])
            else:
                probabilities.append(rt_bin_probability(distribution, choices[k], response_times[k], p[1],
                                                         resolution=resolution, measurement_sd=measurement_sd))
        else:
            # Excluded rows advance history, but need no response-density solve.
            distribution = None
            probabilities.append(p.new_tensor(float("nan")))
        distributions.append(distribution)
        history = path.history[int(steps[k]) - 1]
    probability = torch.stack(probabilities)
    # Do not floor zero probabilities or renormalize the finite time horizon.
    return LikelihoodResult(probability[mask].log().sum(), probability, tuple(distributions))


def step_sequence_likelihood(parameters, tasks, stimuli, choices, decision_steps, *, config=None, include=None):
    """Direct joint likelihood of choices and observed integer stopping steps.

    This mode needs no RT measurement model and no plug-in history. Spatial
    quadrature is its only numerical approximation (plus the reported lower
    domain loss). Nondecision time is not identifiable from stopping steps.
    """
    return sequence_likelihood(parameters, tasks, stimuli, choices, [0.] * len(tasks), decision_steps,
                               max_steps=int(max(decision_steps)), config=config, include=include, observation="step")
