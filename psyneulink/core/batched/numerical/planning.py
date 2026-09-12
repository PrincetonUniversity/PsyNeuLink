"""Select numerical kernels by mathematical requirements, not model identity.

This is a backend contract, NOT evidence that a source Composition admits the
requested reduction. A future continuous lowering must establish that separately.
In particular, supplying a deterministic mean of a random coefficient is not
conditioning on that coefficient's unobserved history.
"""

from dataclasses import asdict, dataclass, field
import math
from numbers import Real

import torch

from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.numerical.first_passage import MovingBoundaryDDMSolver


@dataclass(frozen=True)
class FirstPassageProblem:
    """Mathematical requirements supplied by a caller or a future checked lowering.

    Stochastic dimension is the dimension of the Markov state whose density must
    be evolved, NOT the number of RNG sources or deterministic upstream states.
    No particular number or topology of deterministic upstream states is assumed.
    """

    process: str
    stochastic_dimensions: int
    drift_dependence: str
    diffusion: str
    boundary: str
    initial_state: str
    coefficient_source: str
    observation: str


@dataclass(frozen=True)
class FirstPassageMesh:
    time_step: float = 0.001
    spatial_points: int = 65
    boundary_floor: float = 1.e-5
    backward_euler_steps: int = 2

    def __post_init__(self):
        for name in ("time_step", "boundary_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive real number.")
        if type(self.spatial_points) is not int or self.spatial_points < 5 or self.spatial_points % 2 != 1:
            raise ValueError("spatial_points must be an odd integer of at least five.")
        if type(self.backward_euler_steps) is not int or self.backward_euler_steps < 0:
            raise ValueError("backward_euler_steps must be a nonnegative integer.")


_SCALAR_REQUIREMENTS = FirstPassageProblem(
    process="continuous_time",
    stochastic_dimensions=1,
    drift_dependence="time_only",
    diffusion="constant_scalar",
    boundary="symmetric_linear",
    initial_state="point_center",
    coefficient_source="conditioned_deterministic",
    observation="choice_rt_interval",
)
_GRADIENT_INPUTS = ("drift", "threshold", "collapse_rate", "interval_low", "interval_high")


@dataclass(frozen=True)
class FirstPassagePlan:
    """Reusable batched interval-probability solve, without a subject/model schema.

    ``drift[b, k]`` is sampled at (k+1/2)*dt. All other inputs have shape [batch].
    The boundary is +/- (threshold + collapse_rate*t); choice 0/1 means lower/
    upper. Intervals use decision time, not RT including nondecision time.

    Only result.probability has a supported first derivative on cpp_cpu. Other
    outputs are diagnostics. No probability flooring, density/bin-width division,
    latent-history integration, RT conversion, or log aggregation is performed.
    """

    problem: FirstPassageProblem
    mesh: FirstPassageMesh
    noise: float
    backend: str
    _solver: MovingBoundaryDDMSolver = field(repr=False, compare=False)

    def explain(self):
        return dict(
            evaluator="scalar_symmetric_linear_first_passage",
            method="numerical", backend=self.backend, problem=asdict(self.problem),
            mesh=asdict(self.mesh), noise=self.noise,
            gradient="first_order_discrete_adjoint" if self.backend == "cpp_cpu" else "torch_autodiff",
            gradient_inputs=_GRADIENT_INPUTS,
            assumptions=(
                "Caller supplies correctly conditioned deterministic coefficients; no graph-reduction proof is attached.",
                "Centered scalar diffusion, fixed positive noise, symmetric linear bounds.",
            ),
            approximations=(
                "Fixed normalized grid, Chang-Cooper flux, backward-Euler startup then Crank-Nicolson.",
                "RT interval mass from time-cell flux overlap; gradients are piecewise at mesh/interval knots.",
                "Boundary-floor violations at active midpoints return zero probability and invalid_boundary=True.",
                "No formal error bound; mass error and minimum density must be monitored and the mesh refined.",
            ),
        )

    def solve_observation_batch(self, *, drift, threshold, collapse_rate, interval_low, interval_high, choice):
        values = dict(drift=drift, threshold=threshold, collapse_rate=collapse_rate,
                      interval_low=interval_low, interval_high=interval_high, choice=choice)
        if not isinstance(drift, torch.Tensor) or drift.ndim != 2:
            raise ValueError("drift must be a tensor of shape [batch, time].")
        if drift.dtype not in (torch.float32, torch.float64) or drift.device.type != "cpu":
            raise ValueError("The first-passage plan requires CPU float32 or float64 tensors.")
        for name, value in values.items():
            expected = drift.shape if name == "drift" else (drift.shape[0],)
            if not isinstance(value, torch.Tensor) or value.shape != expected:
                raise ValueError(f"{name} must have shape {tuple(expected)}.")
            if value.dtype != drift.dtype or value.device != drift.device:
                raise ValueError("All first-passage inputs must share dtype and device.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite values.")
        if choice.requires_grad or not bool(((choice == 0) | (choice == 1)).all()):
            raise ValueError("choice must be nondifferentiable and contain only 0 or 1.")
        if bool((interval_high <= interval_low).any()):
            raise ValueError("RT intervals must have positive width.")
        # Refuse silent truncation when the caller's coefficient path is too short.
        horizon = drift.shape[1] * self.mesh.time_step
        tolerance = 8 * torch.finfo(drift.dtype).eps * max(1., horizon)
        if bool((interval_high > horizon + tolerance).any()):
            raise ValueError("The drift time mesh must cover every observation interval.")
        return self._solver.solve_observation_batch(**values)


def compile_first_passage(problem, *, noise, mesh=None, backend="cpp_cpu", required_gradients=()):
    """Select the current scalar kernel by requirements, without CSI recognition.

    This deliberately does not register a Composition-level numerical provider:
    the continuous equations and history reduction still need checked lowering.
    Unsupported dimensions, latent coefficients, initial states, or gradient
    axes raise diagnostics rather than falling back to a different process.
    """
    if type(problem) is not FirstPassageProblem:
        raise TypeError("problem must be a FirstPassageProblem.")
    mismatches = [name for name, expected in asdict(_SCALAR_REQUIREMENTS).items()
                  if getattr(problem, name) != expected or type(getattr(problem, name)) is not type(expected)]
    if mismatches:
        raise LikelihoodPlanningError("numerical.unsupported_problem", "No first-passage provider for: " + ", ".join(mismatches))
    if backend not in ("cpp_cpu", "torch_cpu"):
        raise LikelihoodPlanningError("numerical.backend_unsupported", "First-passage backends are cpp_cpu and torch_cpu; no device fallback is performed.")
    if isinstance(required_gradients, str) or any(name not in _GRADIENT_INPUTS for name in required_gradients):
        raise LikelihoodPlanningError("numerical.gradient_unsupported", "Only drift, threshold, collapse_rate, interval_low, and interval_high derivatives are implemented.")
    if isinstance(noise, bool) or not isinstance(noise, Real) or not math.isfinite(noise) or noise <= 0:
        raise ValueError("noise must be a fixed finite positive real number, not a differentiable tensor.")
    mesh = FirstPassageMesh() if mesh is None else mesh
    if type(mesh) is not FirstPassageMesh:
        raise TypeError("mesh must be a FirstPassageMesh.")
    solver = MovingBoundaryDDMSolver(
        time_step=mesh.time_step, spatial_points=mesh.spatial_points, noise=noise,
        boundary_floor=mesh.boundary_floor, rannacher_steps=mesh.backward_euler_steps,
        native_forward=backend == "cpp_cpu", custom_adjoint=backend == "cpp_cpu",
        # The debug oracle must support autograd.grad/gradcheck. The prototype's
        # reentrant checkpoint path only supports ordinary .backward(). The CPU
        # whole-solve adjoint manages its own tape and does not use this option.
        checkpoint_steps=0,
    )
    return FirstPassagePlan(problem, mesh, float(noise), backend, solver)
