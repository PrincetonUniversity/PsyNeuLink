# Reusable numerical likelihood backend

The subsequent [continuous-equation milestone](CONTINUOUS_EQUATION_COMPILATION.md)
adds an explicit equation IR, deterministic dependency slicing, and generated
C++ phase dynamics/adjoints. It does not yet provide Composition-level admission.

This milestone extracts the existing CSI scalar first-passage solver into
`psyneulink/core/batched/numerical`. It preserves its C++ loops and adjoint; it
does not add a whole-model CSI recognizer or infer a continuous process from
the source simulator. CSI now imports the shared solver through compatibility
exports. Its LCA equations, deterministic history scan, parameter schema,
duration bucketing, and optimizer remain research-local.

## Implemented boundary

`compile_first_passage` selects a numerical kernel by mathematical requirements.
It is a **low-level backend API**, not the Composition-level `compile_likelihood`
entry point. A caller currently supplies those requirements. They are not a
checked witness that a PsyNeuLink graph reduces to the specified diffusion.
Composition-level `method="numerical"` remains rejected until that lowering
exists. Source, ideal-real, and continuous process targets remain distinct.

```python
import torch
from psyneulink.core.batched.numerical import (
    FirstPassageProblem, FirstPassageMesh, compile_first_passage,
)

problem = FirstPassageProblem(
    process="continuous_time",
    stochastic_dimensions=1,
    drift_dependence="time_only",
    diffusion="constant_scalar",
    boundary="symmetric_linear",
    initial_state="point_center",
    coefficient_source="conditioned_deterministic",
    observation="choice_rt_interval",
)
plan = compile_first_passage(
    problem, noise=0.1,
    mesh=FirstPassageMesh(time_step=0.001, spatial_points=65),
    backend="cpp_cpu",
    required_gradients=("drift", "threshold", "collapse_rate"),
)
t = (torch.arange(500, dtype=torch.float64) + 0.5) * 0.001
amplitude = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
result = plan.solve_observation_batch(
    drift=(amplitude * torch.sin(4 * t))[None, :],
    threshold=torch.tensor([0.12], dtype=torch.float64),
    collapse_rate=torch.tensor([-0.02], dtype=torch.float64),
    interval_low=torch.tensor([0.3973], dtype=torch.float64),
    interval_high=torch.tensor([0.4027], dtype=torch.float64),
    choice=torch.tensor([1.0], dtype=torch.float64),
)
gradient, = torch.autograd.grad(result.probability.log().sum(), amplitude)
print(plan.explain())
```

The example is deliberately not an LCA or CSI model. Any correctly conditioned
deterministic upstream computation can supply the drift tensor and receive its
chain-rule gradient. It can have many deterministic state dimensions. There is
no Python callback inside the native PDE time loop.

## Numerical and gradient contract

- Drift is `[batch, time]`, evaluated at time-step midpoints. Other tensor inputs
  are `[batch]`; all use CPU float32 or float64 with a common dtype. Choice is
  0 for the lower boundary and 1 for the upper boundary.
- Physical boundaries are `+/- (threshold + collapse_rate*t)`. Positive and
  negative slopes are supported while the active boundary remains admissible.
- RT intervals are in **decision time**, after any observation transformation.
  The returned probability is joint choice/interval mass, not density and not
  RT conditional on choice. No bin-width division or likelihood floor is added.
- The caller must supply a sufficiently long drift path. Insufficient coverage,
  nonfinite inputs, malformed intervals, and nonbinary choices raise errors.
- The normalized spatial grid and Chang-Cooper operator are unchanged. Startup
  uses two full backward-Euler steps by default, then Crank-Nicolson. This is
  Rannacher-style damping, not the classical half-step construction. Boundary
  flux uses the existing average of input/output density, including startup.
- An active midpoint boundary at/below `boundary_floor` returns zero probability
  with `invalid_boundary=True`. This is not a fully collapsed-boundary solver.
  Mass-error and minimum-density diagnostics are retained. There is no formal
  positivity, discretization-error, or floating-point certificate.
- The C++ whole-solve adjoint differentiates only the selected interval
  probability, with respect to drift, threshold, collapse rate, and both interval
  endpoints. Other returned fields are nondifferentiable diagnostics. No Hessian,
  noise, initial-distribution, or mesh derivative is promised. Requests for
  unsupported derivative axes fail rather than returning fabricated zeros.
  Native reverse calls with `create_graph=True` explicitly reject higher-order
  differentiation rather than allowing a partially detached Hessian.
- Derivatives are of this discretized objective; interval/mesh alignments and
  invalid-boundary transitions are piecewise-smooth, not globally smooth.
- `torch_cpu` is an explicit ordinary-autograd correctness oracle without
  reentrant checkpointing. `cpp_cpu` never silently chooses another device.
  A compiler and Ninja are needed to build the CPU extension on first use.

The compiled ABI also checks dimensions, dtypes, devices, and adjoint history
shapes before entering parallel loops. Extension builds are lazy and cached
outside the source tree. The C++ source is included in package distributions.

## Performance preservation

The extracted C++ mathematical loops are unchanged. We keep:

- one native call per batch for the forward solve and one for the adjoint;
- OpenMP trial parallelism, per-lane scratch, and linear-time tridiagonal solves;
- the existing density-tape layout and no tape for value-only calls;
- caller-owned duration bucketing, avoiding a new model-level wrapper per step.

Ordinary score-only calls with nondifferentiable inputs now select the fused
path even without an explicit `torch.no_grad()` context. The new checked backend
adds batch-level input validation, not per-cell Python work.

Run the repeatable fitting-sized PDE benchmark (not a full fit):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_first_passage_backend.py
```

`--route csi_compat` exercises the research package's compatibility import.
Defaults are 480 trial lanes, 1,000 time cells, 1 ms, 65 spatial points, float64,
four threads, two warmups and seven measured runs. Compilation and digest
calculation are excluded from timed calls. Probability and all five gradient
tensor digests allow exact comparisons on the same toolchain. Different builds
or architectures need numerical tolerances instead of digest equality.

Extraction check on the local i7-9700K (four threads, configuration above):

| Route | Score | Score + all five coefficient-gradient tensors |
| --- | ---: | ---: |
| Before extraction, research-local solver | 141.68 ms | 488.62 ms |
| Extracted backend, including input validation | 140.60 ms | 484.92 ms |

Probabilities and every gradient tensor were bitwise identical on this build.
The timing differences are under 1% and within run-to-run variation: this is
evidence of preserved performance, not a speedup claim. These are PDE-stage
measurements with prescribed coefficients, not full CSI subject-fit runtimes.

## Path to general Composition compilation

1. **Done here: separate reusable numerical kernels from model equations.**
   Unsupported stochastic dimension, coefficient dependence, boundary family,
   initial law, observation operator, and gradient requirements fail explicitly.
2. **Next: continuous primitive equations and explicit clocks.** Declare or
   lower continuous RHS expressions, diffusion terms, resets, event surfaces,
   and readouts. A source Euler implementation does not automatically establish
   the intended continuous model or resolve multiple scheduler clocks.
3. **Checked deterministic/stochastic decomposition.** Prove from the lowered
   dependencies when coefficients can be supplied from deterministic upstream
   evolution conditioned on observations. Trial history must include all phases
   and observed stopping durations. Binned prior RTs can leave uncertainty in
   persistent state: a point-history approximation must remain explicitly
   labeled, not promoted to exact interval conditioning.
4. **Generated compiled dynamics and VJPs.** Emit model equations into reusable
   C++ integration/history kernels. Preserve the current whole-scan and
   whole-trajectory call boundaries, fusion, OpenMP, and memory layouts. Do not
   replace those loops with Python callbacks for every node/time step.
5. **Wire admitted reductions into `compile_likelihood`.** Select the backend
   from a checked mathematical witness and expose observation transformations,
   parameter bindings, diagnostics, and gradient axes through the common facade.
6. **Add different backends when the mathematics requires them.** More general
   boundaries and initial laws, multidimensional diffusion, and latent-history
   filtering are distinct extensions with their own admission/validation rules.
   Other observation laws need not be first passage or even decision models.

For Dawa's fitting variant, deterministic upstream dynamics may still be
precomputable, but the two noisy response accumulators can require a **2D**
probability state. This 1D backend must reject that request. The eventual
continuous interpretation must also resolve its nested LCA/LC clocks, stopping
rule, and persistent-state conditioning. The reusable pieces are the equation
representation, dependency analysis, observation contracts, and backend planning;
not a claim that today's tridiagonal solver already handles that model.

## Validation / reading order

1. `numerical/planning.py`: mathematical admission and explainable limitations.
2. `numerical/first_passage.py`: grid, Torch reference, and whole-solve autograd.
3. `numerical/native.py`: lazy build and coarse native ABI.
4. `numerical/first_passage_cpu.cpp`: unchanged PDE/adjoint loops, checked entry.
5. `tests/composition/pec/test_batched_numerical_first_passage.py`: non-CSI
   coefficient paths, all-input finite differences, analytic interval-mass
   reference, invalid/empty cases, tape allocation, and rejection diagnostics.
6. Existing `test_csi_direct_likelihood.py`: full CSI value/gradient regression.

Validation for this extraction: the combined backend/CSI/Gaussian/Wiener/common
planning run passed 156 checks with 7 skips (including existing GPU checks).
After adding the explicit higher-derivative guard, the final standalone backend
suite passed 51 checks. Ruff and whitespace checks passed. An offline isolated
wheel build succeeded; archive inspection confirmed all five numerical-backend
files, including the C++ source, with no research-local Python imports.
