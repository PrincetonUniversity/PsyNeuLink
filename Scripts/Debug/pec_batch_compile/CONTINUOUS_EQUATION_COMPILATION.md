# Explicit continuous equations and generated native dynamics

This milestone adds a model-independent equation representation, conservative
stochastic-dependency analysis, deterministic-subsystem extraction, and C++
generation for deterministic continuous phases. It builds on the extracted
scalar first-passage backend without changing existing CSI fitting defaults.

**This is not yet automatic lowering of a PsyNeuLink Composition.** Equations
are explicit mathematical declarations. The compiler does not infer a continuous
limit from an Euler body, source scheduler passes, or the name of a model. No
reset, termination event, cross-trial likelihood factorization, or observed RT
history rule is inferred here. Composition-level `method="numerical"` remains
rejected until those contracts and their source bindings are implemented.

## Components

| Component | Responsibility |
| --- | --- |
| `continuous_ir.py` | Immutable scalar expression DAGs; vector drift, sparse Itô diffusion, readouts, latent initial/input declarations, physical clock unit |
| `analyze_continuous_dynamics` | Conservative closure of randomness through drift and diffusion dependencies, including feedback |
| `extract_deterministic_subsystem` | Closed deterministic equation slice and explicit state/input/parameter/readout index maps |
| `numerical/dynamics_codegen.py` | C++ expressions with common-subexpression sharing and generated reverse derivatives; Torch equation oracle |
| `numerical/rk4_cpu.h` | Reusable OpenMP RK4 phase loop and discrete integration-method adjoint |
| `numerical/dynamics.py` | Lazy cached compilation, runtime tensor contracts, autograd boundary, diagnostics |

The expression language supports constants, named state/input/parameter
references, physical time, arithmetic, exp, tanh, and sigmoid. Names are never
interpolated as C++ identifiers or executable code. Unknown operations, unbound
names, malformed dimensions, and nonfinite constants are rejected. There is no
arbitrary-Python callback inside generated integration loops.

The mathematics is independent of a particular observation type. A continuous
phase can return algebraic readouts, just a final state, or both. It does not
require a decision, reaction time, or first-passage likelihood.

## Example: nonlinear deterministic dynamics

```python
import torch
from psyneulink.core.batched.continuous_ir import (
    ContinuousDynamics, state, parameter, input_value, physical_time,
)
from psyneulink.core.batched.numerical import compile_continuous_phase

x, y = state("x"), state("y")
k, drive, t = parameter("decay"), input_value("drive"), physical_time()
dynamics = ContinuousDynamics(
    states=("x", "y"), inputs=("drive",), parameters=("decay",),
    drift=(-k*x + y.sigmoid() + drive, x.tanh() - y),
    readouts=(("signal", (x-y).sigmoid() + 0.1*t),),
)
plan = compile_continuous_phase(dynamics)

decay = torch.tensor([[1.2]], dtype=torch.float64, requires_grad=True)
result = plan.integrate(
    state=torch.tensor([[0.1, -0.2]], dtype=torch.float64),
    inputs=torch.tensor([[0.3]], dtype=torch.float64),
    parameters=decay,
    duration=torch.tensor([0.2], dtype=torch.float64),
    start_time=torch.tensor([0.4], dtype=torch.float64),
    steps=torch.tensor([200], dtype=torch.int64),
)
gradient, = torch.autograd.grad(result.readouts.sum(), decay)
print(result.final_state, gradient, plan.explain())
```

Each lane is a supplied initial state and conditioned input/parameter row.
Inputs and parameters remain constant within the phase, while explicit time
and every state coordinate co-evolve. Nonautonomous expressions can use the
physical-time variable. Additional external-input changes require separate
phases or a future time-dependent-input representation.

The example's cell width is 1 ms. RHS evaluations use physical time starting at
0.4 seconds, rather than resetting the clock to zero. There are two RK4
half-steps per cell; readouts are evaluated on the actual intermediate state
at the cell midpoint. This preserves the original direct solver's coefficient
sampling convention while admitting arbitrary supported vector equations.

## Stochastic dependency analysis

A nonzero declared diffusion term seeds randomness in its state coordinate.
Named Wiener drivers can be shared across coordinates. Randomness then
propagates through all drift/diffusion dependencies; latent initial states and
latent inputs seed the same closure. This is not just counting RNG streams.

For example, consider deterministic states h0/h1 that drive noisy response
states r0/r1. Without response-to-control feedback, the analysis retains h0/h1
as a deterministic subsystem. If r0 feeds h0 and h0/h1 are coupled, randomness
propagates into both control states and the deterministic reduction disappears.
Tests exercise both cases without using any model recognizer.

`extract_deterministic_subsystem` returns a new equation system plus index maps
for slicing the original state, input, parameter and readout arrays. Unused
latent inputs are removed, not silently relabeled as conditioned observations.
It returns None when no deterministic state remains. Algebraic cancellation is
not assumed: `0*x` still depends on x. This deliberately conservative analysis
does not discover a minimal stochastic state or prove that a full-dimensional
density exists (shared noise can create lower-dimensional support).

The phase-local slice is **not** proof of exact observed-history conditioning.
A stopping time can determine the duration for which deterministic state
evolves; uncertain/binned prior RTs can therefore leave a distribution over
persistent states. A future history lowering must retain that uncertainty or
explicitly label any point-history approximation.

## Generated gradients and timing

Equation VJPs are generated mechanically by reverse accumulation over the DAG.
The RK4 adjoint is implemented once, independently of model equations. It
recomputes the RK4 stages from stored cell-start states during the reverse pass.
It differentiates through both the midpoint readout and the final state.

Supported first derivatives cover initial state, inputs, parameters, duration,
and phase start time. Integration counts are integer, nondifferentiable mesh
choices. For lane j, the cell width is duration[j]/steps[j]; gradients include
the resulting changes to RK4 stage sizes **and** explicit evaluation times.
Tests check these derivatives with ordinary Torch autograd and finite
differences, including chaining two phases with parameter-dependent timing.

Zero-count phases must have zero duration: they preserve the input state and
return zero-padded readouts, with identity state gradients and zero clock/input/
parameter gradients. This is an explicit discrete mesh branch, not a statement
of smoothness across changes in step count. Lanes with fewer cells have zero
readout padding. Empty batches are supported. A solver is not an optimizer and
does not choose a mesh or assess its convergence automatically.

The native adjoint is first-order only. Higher-order requests fail explicitly
rather than allowing a partial Hessian. CPU float64, finite inputs, positive
or zero durations, and valid buffer shapes are checked. Nonfinite results or
gradients raise numerical errors. The current code-generation budget is 256
state coordinates and 4096 distinct expression nodes.

## Connection to the direct PDE

Generated readout tensors retain their gradient connection to the underlying
equation parameters. Tests pass a non-CSI generated coefficient path directly
to the shared C++ PDE and finite-difference the resulting log probability.

The caller must align the two clocks/meshes: for the current first-passage
backend, active phase lanes must provide midpoint coefficients at the PDE's
fixed dt, i.e. duration/steps must match that dt. Merely passing a tensor with
the right shape is not a proof of matching physical time. Boundary equations,
observation transformations, and history conditioning are still supplied
separately; no full subject-level likelihood lowering is claimed here.

## Performance check

The research-only CSI equation fixture in
`benchmark_continuous_dynamics.py` is compared against the existing handwritten
C++ drift kernel. It is not registered in the compiler. The benchmark uses 480
trials, 1,000 cells at 1 ms, four CPU threads, float64, two warmups and seven
interleaved measured runs. Compilation is excluded.

| Drift stage | Handwritten | Generated |
| --- | ---: | ---: |
| Forward | 25.09 ms | 28.08 ms |
| Forward + gradients | 80.83 ms | 81.51 ms |

Readout paths and final states were identical in this run. All compared
initial-state, input and gain derivatives differed by less than 5e-13. The
generated forward path is about 12% slower in this stage; gradient execution
was within about 1%. These are drift-stage measurements, not whole-subject
scores or fit times. The generated path also implements duration/start-time
derivatives that the old fixed-step drift API does not expose.

This is promising evidence that general equation generation need not discard
the C++ performance work, but not a reason to retire the old CSI path yet.
The new code preserves one native call per phase, OpenMP across lanes, inline
equations, reusable adjoints and no history tape for value-only calls. Module
source and the integrator template are frozen into the plan before lazy build;
the cache key includes both. Build artifacts stay outside the repository.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_continuous_dynamics.py
```

## Remaining steps

1. Bind explicit continuous primitive equations to frozen PNL implementations,
   parameters, ports and state initialization/reset rules. Validate clock
   interpretations rather than guessing conversions from scheduler passes.
2. Lower projections, nonlinear readouts and controls into the common equation
   representation, checking publication/phase semantics and dependency slices.
3. Add explicit phases, resets, stopping/readout surfaces and observed-history
   reconstruction. Keep subject scans compiled instead of calling this phase
   API once per node per time step from Python.
4. Select numerical probability backends from the admitted stochastic subsystem
   and observation contracts, then wire the result into `compile_likelihood`.

Dawa's model is a motivating case, not yet compiled or verified here. The
representation accommodates larger coupled deterministic systems and identifies
multiple stochastic response coordinates. A suitable multidimensional
probability backend, the nested model's explicit clock/reset semantics, and
its observation/history reduction are still required. General compositions
with other data types need their own observation and probability providers;
this milestone does not force them into a first-passage framework.

## Validation

The combined continuous-dynamics, shared PDE, full CSI, Gaussian, Wiener and
common likelihood-planning suite passed 175 checks with 7 skips, including the
existing GPU checks. Coverage includes closed-form scalar/four-state dynamics,
all-input finite differences, explicit-time and duration gradients, multi-phase
chains, generated-drift-to-PDE gradients, zero/empty lanes, stochastic-feedback
rejection, deterministic index maps, frozen templates and handwritten CSI
parity. Ruff and whitespace checks passed. An offline isolated wheel build
succeeded and contains the equation IR, generator/runtime, RK4 header and PDE
source; neither generated backend depends on research-local model imports.
