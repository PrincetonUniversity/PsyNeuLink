"""Bounded gradient optimizers for prototype likelihood audits."""

import numpy as np


def _box_quadratic_step(inverse, gradient, x, limit):
    """Solve the small positive-definite BFGS subproblem by a primal active set."""
    hessian = np.linalg.solve(inverse, np.eye(len(x)))
    hessian = .5 * (hessian + hessian.T)
    low, high = np.maximum(-x, -limit), np.minimum(1. - x, limit)
    step = np.zeros_like(x)
    # -1 lower bound, +1 upper bound, 0 free.
    active = np.zeros(len(x), dtype=int)
    active[(low == 0) & (gradient > 0)] = -1
    active[(high == 0) & (gradient < 0)] = 1
    for _ in range(20 * len(x) + 20):
        residual = gradient + hessian @ step
        free = active == 0
        direction = np.zeros_like(x)
        if free.any():
            direction[free] = np.linalg.solve(hessian[np.ix_(free, free)], -residual[free])
        if np.max(np.abs(direction)) < 1.e-12:
            violation = np.where(active == -1, -residual, np.where(active == 1, residual, -np.inf))
            if violation.max() <= 1.e-10:
                return step
            active[np.argmax(violation)] = 0
            continue
        fractions = np.full(len(x), np.inf)
        positive, negative = direction > 1.e-15, direction < -1.e-15
        fractions[positive] = (high[positive] - step[positive]) / direction[positive]
        fractions[negative] = (low[negative] - step[negative]) / direction[negative]
        hit = int(np.argmin(fractions))
        alpha = min(1., max(0., fractions[hit]))
        step = np.clip(step + alpha * direction, low, high)
        if alpha < 1.:
            active[hit] = 1 if direction[hit] > 0 else -1
    # A projected steepest-descent direction remains a valid fallback.
    return np.clip(-gradient, low, high)


def minimize_bounded_bfgs(evaluate, initial, *, iterations=25, step_limit=.1, max_backtracks=24,
                          tolerance=1.e-6, state=None, callback=None, initial_inverse=None):
    """Minimize using bounded quadratic BFGS steps and Armijo backtracking.

    The callable returns (loss, gradient), or +inf for an infeasible proposal.
    A rejected proposal cannot be reported as relative-function convergence.
    """
    x = np.asarray(initial, dtype=float).copy()
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1) or iterations < 1 or step_limit <= 0 or tolerance <= 0:
        raise ValueError("Valid unit coordinates, iteration count, and step limit are required.")
    if state is not None:
        x = np.asarray(state["x"], dtype=float)
        if x.shape != np.asarray(initial).shape or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError("Invalid restart coordinates.")
    value, gradient = evaluate(x)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("The initial objective must be finite.")
    inverse = np.eye(len(x)) if state is None else np.asarray(state["inverse_hessian"], dtype=float)
    if initial_inverse is not None:
        inverse = np.asarray(initial_inverse, dtype=float)
    if inverse.shape != (len(x), len(x)) or not np.isfinite(inverse).all():
        raise ValueError("Invalid restart inverse Hessian.")
    np.linalg.cholesky(inverse)
    trace = [float(value)] if state is None else list(state["accepted_losses"])
    rejected = 0 if state is None else state["rejected_proposals"]

    def snapshot():
        return {"x": x.tolist(), "inverse_hessian": inverse.tolist(), "loss": float(value),
                "gradient": gradient.tolist(), "accepted_losses": trace.copy(), "rejected_proposals": rejected,
                "projected_gradient_norm": float(np.max(np.abs(np.clip(x - gradient, 0., 1.) - x)))}
    success, message = False, "Iteration limit reached"
    for _ in range(iterations):
        if np.max(np.abs(np.clip(x - gradient, 0., 1.) - x)) < tolerance:
            success, message = True, "Projected gradient tolerance reached"
            break
        direction = _box_quadratic_step(inverse, gradient, x, step_limit)
        if gradient @ direction >= 0:
            inverse = np.eye(len(x))
            direction = np.clip(-gradient, np.maximum(-x, -step_limit), np.minimum(1. - x, step_limit))
        accepted = False
        for backtrack in range(max_backtracks):
            candidate = np.clip(x + 2.**(-backtrack) * direction, 0., 1.)
            displacement = candidate - x
            predicted = float(gradient @ displacement)
            if predicted >= 0:
                continue
            next_value, next_gradient = evaluate(candidate)
            if np.isfinite(next_value) and np.isfinite(next_gradient).all() and next_value <= value + 1.e-4 * predicted:
                accepted = True
                break
            rejected += 1
        if not accepted:
            message = "No improving feasible step found"
            break
        delta_gradient = next_gradient - gradient
        curvature = float(displacement @ delta_gradient)
        if curvature > 1.e-10 * max(1.e-20, np.linalg.norm(displacement) * np.linalg.norm(delta_gradient)):
            transform = np.eye(len(x)) - np.outer(displacement, delta_gradient) / curvature
            inverse = transform @ inverse @ transform.T + np.outer(displacement, displacement) / curvature
        x, value, gradient = candidate, next_value, next_gradient
        trace.append(float(value))
        if callback is not None:
            callback(snapshot())
    if np.max(np.abs(np.clip(x - gradient, 0., 1.) - x)) < tolerance:
        success, message = True, "Projected gradient tolerance reached"
    return {"x": x, "success": success, "message": message, "nit": len(trace) - 1,
            "accepted_losses": trace, "rejected_proposals": rejected, "state": snapshot(),
            "projected_gradient_norm": snapshot()["projected_gradient_norm"]}


def fit_projected_gradient(evaluate, initial, bounds, *, iterations=10, step_size=.1, max_backtracks=12):
    """Ascend in unit parameter coordinates with an Armijo backtracking search.

    Invalid/underflowed likelihood proposals are rejected, never assigned a
    probability floor. This intentionally small optimizer is a gradient smoke
    test; it does not claim multi-start fitting or parameter recovery.
    """
    bounds = np.asarray(bounds, dtype=float)
    initial = np.asarray(initial, dtype=float)
    scale = bounds[:, 1] - bounds[:, 0]
    unit = (initial - bounds[:, 0]) / scale
    if np.any(scale <= 0) or np.any(unit < 0) or np.any(unit > 1):
        raise ValueError("Initial parameters must lie inside nonempty bounds.")
    value, physical_gradient = evaluate(initial)
    trace, rejected = [float(value)], 0
    success, message = False, "Iteration limit reached"
    for _ in range(iterations):
        gradient = physical_gradient * scale
        projected = np.clip(unit + gradient, 0., 1.) - unit
        if np.max(np.abs(projected)) < 1.e-6:
            success, message = True, "Projected gradient tolerance reached"
            break
        direction = gradient / max(1., np.max(np.abs(gradient)))
        step, accepted = step_size, False
        for _ in range(max_backtracks):
            candidate = np.clip(unit + step * direction, 0., 1.)
            predicted = float(gradient @ (candidate - unit))
            try:
                next_value, next_gradient = evaluate(bounds[:, 0] + scale * candidate)
            except FloatingPointError:
                next_value, next_gradient = -np.inf, None
            if np.isfinite(next_value) and next_value >= value + 1.e-4 * predicted:
                unit, value, physical_gradient = candidate, next_value, next_gradient
                trace.append(float(value))
                accepted = True
                break
            rejected += 1
            step *= .5
        if not accepted:
            message = "No improving feasible step found"
            break
    return {"success": success, "message": message, "iterations": len(trace) - 1,
            "log_likelihood": float(value), "parameters": (bounds[:, 0] + scale * unit).tolist(),
            "accepted_log_likelihoods": trace, "rejected_proposals": rejected,
            "optimizer": "projected gradient ascent with Armijo backtracking in unit coordinates"}
