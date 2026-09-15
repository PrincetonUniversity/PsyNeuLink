# Explicit continuous equations and generated C++ dynamics

The symbolic implementation now uses SymPy directly. The custom `Equation`
language, elementary reverse-derivative rules, and experimental adapter were
removed after checkpoint `e2721cc518`. The C++ RK4/PDE kernels are unchanged.

This milestone adds a model-independent equation representation, conservative
stochastic-dependency analysis, deterministic-subsystem extraction, and C++
generation for deterministic continuous phases. It builds on the extracted
scalar first-passage backend without changing existing CSI fitting defaults.

**This is not yet automatic continuous-time likelihood lowering of a PsyNeuLink
Composition.** Linear/Logistic/dense-projection value graphs and checked readout
regions inside stateful graphs can now supply equations from the frozen source IR;
continuous dynamics remain explicit
mathematical declarations. The compiler does not infer a continuous
limit from an Euler body, source scheduler passes, or the name of a model. No
reset, termination event, cross-trial likelihood factorization, or observed RT
history rule is inferred here. Composition-level `method="numerical"` remains
rejected until those contracts and their source bindings are implemented.

## Components

| Component | Responsibility |
| --- | --- |
| `LikelihoodEffectContract.symbolic_value`, built-in primitive specs | Trusted scalar SymPy laws attached to existing compiled implementations |
| `symbolic_values.py` | Frozen static graph value propagation using existing port, projection, parameter and publication identities |
| `continuous_ir.py` | Phase metadata over SymPy expressions: drift, Itô diffusion, readouts, latent initial/input declarations, physical clock unit |
| `analyze_continuous_dynamics` | Conservative closure of randomness through drift and diffusion dependencies, including feedback |
| `extract_deterministic_subsystem` | Closed deterministic equation slice and explicit state/input/parameter/readout index maps |
| `numerical/dynamics_codegen.py` | SymPy differentiation, CSE and C++/Torch printing; retained denominator-domain guards |
| `numerical/rk4_cpu.h` | Reusable OpenMP RK4 phase loop and discrete integration-method adjoint |
| `numerical/dynamics.py` | Lazy cached compilation, runtime tensor contracts, autograd boundary, diagnostics |

Equations use real SymPy symbols, finite numeric constants, arithmetic, integer
powers, exp, tanh, and the stable `Sigmoid` primitive. State/input/parameter/time
bindings must be distinct symbols (same-named `Dummy` symbols can represent
distinct identities). Names are never interpolated as C++ identifiers or
executable code. Unknown operations, unbound symbols, unsupported power domains,
malformed dimensions, and nonfinite constants are rejected. There is no
arbitrary-Python callback inside generated integration loops.

The mathematics is independent of a particular observation type. A continuous
phase can return algebraic readouts, just a final state, or both. It does not
require a decision, reaction time, or first-passage likelihood.

## Deriving stateless values from PNL

```python
from psyneulink.core.batched import BatchedCompositionCompiler

values = BatchedCompositionCompiler.derive_symbolic_values(
    composition, outputs=[response.output_port],
)
print(values.expressions, values.explain())
```

Alternatively, call `simulation_plan.derive_symbolic_values()` to use an already
frozen simulation snapshot. Neither path requires executing a simulation; the
first does not require an available simulation device. Both require supported
source IR. Subsequent mutations to live PNL nodes or global registrations do not
change a previously frozen plan's equations.

Each registered scalar law is a `sympy.Lambda` with arguments `x` and every
registered parameter argument exactly once. Registration checks those bindings
and the supported expression domain. Linear and Logistic carry these declarations
beside their existing batched implementations; dense projections use their
existing contract and frozen matrices. No CSI recognizer or separate operation
registry is involved. These are trusted mathematical declarations, not proofs
that a Python/Triton body implements them or promises of bitwise source arithmetic.

The lowering visits the existing graph in execution order, verifies that each
projection source publishes in an earlier consideration set, combines incoming
values, and substitutes each scalar law. Static-schedule/stateless admission is
shared with the existing Gaussian analyzer. Noise, retained state, controls,
clipping, stopping events and unsupported output transforms are rejected rather
than silently dropped. Expressions retain source dependencies and denominator
domains under `sympy.evaluate(False)`.

`SymbolicValues` holds existing `BatchedInputSpec`, `BatchedParamSpec` and
`BatchedOutputSpec` records, their symbolic bindings, and flattened expressions.
Inputs and outputs flatten in port/coordinate order; parameters retain canonical
kernel order. A caller can bind these inputs to continuous state expressions
using `xreplace` under `sympy.evaluate(False)`, then supply the resulting readouts
to `ContinuousDynamics`. Their generated C++ and gradients use the same backend
as explicitly authored equations. No symbolic evaluation happens inside the
integration loop.

The default `scope="graph"` still requires a **whole admitted stateless graph**.
The research CSI-style fixture builds its response network in PNL and binds its
control inputs to an explicit LCA phase. Its LCA ODE, response coding, timing and
history remain caller-supplied.

### Readout regions inside stateful graphs

```python
values = BatchedCompositionCompiler.derive_symbolic_values(
    composition, outputs=[response.output_port], scope="readout",
)
# Also available on an already frozen simulation plan:
# values = simulation_plan.derive_symbolic_values(scope="readout")
print(values.boundary_ports, values.boundary_symbols, values.explain())
```

This walks backward from one requested execution owner through supported value
nodes. It cuts at stateful/noisy producers, modified values such as clipping,
unregistered symbolic laws, and publications whose scheduler timing cannot be
safely substituted. These cuts are **explicit symbolic arguments**, identified
by existing `BatchedPortSpec` records. A cut does not declare the value observed,
deterministic, independent across trials, or equal to the producer's initial
state. The original simulation graph and its state/reset/scheduler declarations
are not rewritten.

Within a checked single-pass schedule, earlier value publications may be
substituted. For multi-pass graphs, substitution currently admits Always
producers and matching AtPass/AtTrialStart/WhenFinished gates. Finished-flag
owners must publish before both gated nodes, so their flags cannot change
between those reads. Other conditional producers remain held-value arguments.
Same-set/delayed/recurrent edges and mixed held/recomputed uses of one producer
are rejected. This is conservative, not a maximal region finder or general
scheduler equivalence proof.

The result describes `readout = F(external inputs, boundary publications,
parameters)` **at that readout's execution**, not necessarily at trial end.
Boundary symbols are separate from external-input symbols. Parameters retain
source IDs and canonical order, but readout scope drops unrelated owners to
avoid unused runtime columns and gradient work. `publication_schedule` reuses
the relevant frozen scheduler records; `boundary_dependencies` reuses the
existing execution-axis analysis, including stochastic stopping-time effects.
For example, the noise-free CSI LCA can retain estimate-dependent trial-end
state because the DDM determines how many times it executes.

Readout derivatives hold supplied boundary publications fixed. A full model
derivative additionally needs the chain rule through their dynamics and timing;
latent boundary values need an appropriate probability treatment. Omitting
boundary bindings fails the ordinary unbound-symbol check. A caller may explicitly
bind deterministic boundary trajectories to continuous state expressions and
use the existing C++ generator, but this API does not infer those trajectories.

Tests attach a nonlinear readout branch to the actual stateful CSI research
Composition and compare generated values against PNL at each readout execution.
The existing opaque CSI drift UDF itself still needs a generic multivariate
symbolic-law route or a supported primitive graph; this milestone does not
recognize its name or derive arbitrary Python bodies. The benchmark's
`--readout-region` variant uses a PNL graph containing a persistent LCA and
extracts its response readout. Its continuous LCA boundary binding is explicit;
it does not equate the source Euler trajectory with RK4.

## Example: nonlinear deterministic dynamics

```python
import torch
import sympy as sp
from psyneulink.core.batched.continuous_ir import ContinuousDynamics, Sigmoid
from psyneulink.core.batched.numerical import compile_continuous_phase

x, y, k, drive, t = sp.symbols("x y decay drive t", real=True)
with sp.evaluate(False):
    dynamics = ContinuousDynamics(
        states=(x, y), inputs=(drive,), parameters=(k,), time=t,
        drift=(-k*x + Sigmoid(y) + drive, sp.tanh(x) - y),
        readouts=(("signal", Sigmoid(x-y) + 0.1*t),),
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
not assumed on retained expressions: `0*x` constructed under
`sympy.evaluate(False)` still depends on x. This deliberately conservative analysis
does not discover a minimal stochastic state or prove that a full-dimensional
density exists (shared noise can create lower-dimensional support).

The phase-local slice is **not** proof of exact observed-history conditioning.
A stopping time can determine the duration for which deterministic state
evolves; uncertain/binned prior RTs can therefore leave a distribution over
persistent states. A future history lowering must retain that uncertainty or
explicitly label any point-history approximation.

## Retained expressions and domains

Construct equations inside `with sympy.evaluate(False):` and retain those
expressions as the declaration. Dependency analysis uses their `free_symbols`;
code generation optimizes a separate copy. An ordinary expression such as
`x/x` may already have become `1` before the constructor receives it. Neither
this compiler nor SymPy can reconstruct information erased by the caller.
Such a supplied constant declares a different domain; it is not evidence that
a source PNL division or stochastic dependency can be removed.

For retained negative integer powers (division), code generation records each
original denominator before CSE/differentiation and checks it for zero or
nonfinite values at every RHS/readout evaluation. A canceled `x/x` still fails
at x=0, including intermediate RK4 stages. The Torch oracle checks the same
domains. Fractional/variable powers and other unsupported functions are rejected
rather than assigned guessed domain rules. `Sigmoid` remains a stable numerical
primitive instead of being expanded into exponentials.

These guards do not promise bitwise preservation of all floating-point
intermediates under mathematical simplification. The backend targets declared
continuous equations; source floating-point endpoint arithmetic is unchanged.
General symbolic complexity and arbitrary model reduction remain outside the
admitted backend's guarantees. Reuse a compiled plan across parameter proposals.

## Generated gradients and timing

Equation VJPs are generated by SymPy differentiation of weighted outputs,
followed by common-subexpression elimination and C++ printing.
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

### Readout-region extraction from a stateful graph

With `--graph-readout --readout-region --repeats 21`, the same local CPU,
480-lane/1,000-cell/1-ms/four-thread/float64 workload gave:

| Drift stage | Handwritten CSI | Explicit SymPy | Stateless graph | Stateful graph's readout region |
| --- | ---: | ---: | ---: | ---: |
| Forward | 25.131 ms | 28.735 ms | 28.896 ms | 28.827 ms |
| Forward + initial-state/input/parameter gradients | 78.689 ms | 80.060 ms | 79.804 ms | 79.938 ms |

This rerun includes the explicitly bound LCA boundary described above. Region
extraction adds no meaningful execution cost relative to authored equations in
this run; the generated forward stage still trails the handwritten kernel by
about 15%. Maximum drift difference against handwritten C++ was 1.67e-16,
final states matched exactly, and compared gradients differed by at most
5.12e-13. There were two warmups and no concurrent test/build workload during
timing. Graph analysis, symbolic binding and compilation are excluded. These
remain drift-stage timings, not whole-subject likelihoods or fits.

### Graph-derived versus explicitly authored readouts

`benchmark_continuous_dynamics.py --graph-readout` compares the handwritten CSI
C++ drift stage, explicitly authored SymPy equations, and the same equations
with the response readout derived from a PNL graph. On the local i7-9700K, with
480 lanes, 1,000 cells at 1 ms, four threads, float64, two warmups and 21
interleaved measurements:

| Drift stage | Handwritten CSI | Explicit SymPy | Graph-derived readout |
| --- | ---: | ---: | ---: |
| Forward | 25.190 ms | 28.830 ms | 28.847 ms |
| Forward + initial-state/input/parameter gradients | 78.887 ms | 80.420 ms | 79.840 ms |

Graph derivation added no meaningful execution cost relative to the explicit
equations in this run. The generated forward stage remains about 14.5% slower
than the handwritten kernel. Against that kernel, graph-derived drift differed
by at most 1.67e-16, final states matched exactly, and compared gradients differed
by at most 5.12e-13. Graph construction, equation derivation and compilation are
excluded. This is a CPU drift-stage check, not a whole-subject fit, GPU likelihood
benchmark, or evidence that the entire CSI model is automatically compiled.

### SymPy replacement against the checkpoint

After the migration, the same CSI fixture was compared directly with checkpoint
`e2721cc518`: 480 lanes, 1,000 cells at 1 ms, four CPU threads, float64, two
warmups, and 21 interleaved measurements. Checkpoint modules were loaded from
`git show` into isolated in-memory module names, without modifying either tree.
No concurrent test/build workload ran during timing. The C++ integration header
is unchanged; source generation and module loading are excluded from runtime.

| Drift stage | Checkpoint custom algebra | SymPy replacement |
| --- | ---: | ---: |
| Forward | 28.603 ms | 29.888 ms |
| Forward + all five input-gradient groups | 80.843 ms | 80.577 ms |
| Source generation, one invocation | 2.68 ms | 146.18 ms |

Forward was about 4.5% slower in this run; gradient execution was effectively
unchanged. Maximum drift difference was 1.67e-16, final states matched exactly,
and the maximum difference across initial-state, input, parameter, duration and
start-time gradients was 1.14e-13. These are phase-stage timings, not full fits.

The four production continuous-backend files decreased from 688 to 583 physical
lines (105 fewer, about 15%). The 228-line disposable SymPy adapter/benchmark was
removed rather than retained as a second backend; the existing handwritten-CSI
benchmark remains. Domain tests were expanded. The numerical kernels, source
GPU compiler, and production CSI fitting defaults were not changed.

### Earlier extraction benchmark (before symbolic replacement)

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
2. Bind the checked readout regions above to continuous state publications and
   phase-dependent controls. Linear/Logistic readouts and dense projections are
   supported, including regions cut from stateful graphs; arbitrary controls
   and opaque multivariate UDF algebra are not.
3. Derive numerical phases, resets and observed-history reconstruction from
   existing graph, scheduler and observation evidence. Add only missing
   continuous interpretations, not a second user-authored experiment schema.
   Keep subject scans compiled instead of per-step Python calls.
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

After readout-region extraction, the same nine-file regression suite passed
**234 checks with 14 skips**. All six dedicated GPU regressions passed separately
on the RTX 2080 Ti. The new checks include stochastic and held boundaries,
preserved zero-weight dependencies, parameter-ID-preserving subsets, matched and
mismatched pass gates, native boundary/parameter finite differences, and a
nonlinear readout branch attached to the real CSI Composition. PNL comparisons
capture each readout execution, not a potentially different trial-end boundary.
The stateful-graph benchmark also checks the chain rule through the explicitly
bound continuous LCA against the handwritten drift kernel. Ruff and whitespace
checks passed. Production fit defaults and C++ numerical kernels are unchanged.

Previous whole-graph milestone validation:

The nine-file graph-value/continuous/PDE/CSI/Gaussian/planning/Wiener/registry/
analysis regression run passed **225 checks with 14 skips** in interpreter mode.
The six GPU-specific checks skipped there all passed in a separate required-GPU
run on the local RTX 2080 Ti; the other eight skips were unchanged style checks.
Ruff and whitespace checks passed.

The graph-derived value tests compare vector/reconvergent networks against PNL
Python execution, the batched CPU interpreter and GPU execution. Every bound
input/parameter derivative is checked by finite differences and a Torch oracle;
another check differentiates a graph-derived readout through the shared PDE.
Tests also cover frozen registrations/parameters, device-independent derivation,
unsupported state/noise/event rejection, and CSI-style readout/gradient parity.
The execution comparisons isolate CPU/GPU processes to avoid Triton's
import-time interpreter selection leaking between tests. Broader interpreter
regressions must start with `TRITON_INTERPRET=1`; dedicated GPU regressions start
with that variable unset. An offline isolated wheel build includes the new
graph-value module and updated built-in declarations.

After the SymPy migration, the six-file continuous/PDE/CSI/Gaussian/Wiener/
planning regression run passed **180 tests with 7 skips**, including the GPU
endpoint check. New checks cover retained zero-times-random dependencies,
canceled division domains in drift/readouts/intermediate RK4 stages, safe
symbol/Dummy names, deterministic generated source, sigmoid saturation and
constant readouts. All prior finite-difference and Torch gradient checks remain.
Ruff/whitespace checks passed. An offline isolated wheel build succeeded and
its metadata declares `sympy>=1.14.0,<1.15`; the migrated sources, RK4 header and
PDE kernel were verified in the wheel. A non-isolated build initially failed
because the local environment lacks `versioneer`; the isolated build used the
project's declared build dependencies without changing that environment.

Historical pre-migration validation:

The combined continuous-dynamics, shared PDE, full CSI, Gaussian, Wiener and
common likelihood-planning suite passed 175 checks with 7 skips, including the
existing GPU checks. Coverage includes closed-form scalar/four-state dynamics,
all-input finite differences, explicit-time and duration gradients, multi-phase
chains, generated-drift-to-PDE gradients, zero/empty lanes, stochastic-feedback
rejection, deterministic index maps, frozen templates and handwritten CSI
parity. Ruff and whitespace checks passed. An offline isolated wheel build
succeeded and contains the equation IR, generator/runtime, RK4 header and PDE
source; neither generated backend depends on research-local model imports.
