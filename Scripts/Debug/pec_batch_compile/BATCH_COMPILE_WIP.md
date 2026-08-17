# PEC Batched Compile WIP Handoff

> **Historical development log.** This document records experiments and
> intermediate conclusions, so older sections below intentionally describe
> behavior that no longer exists. The authoritative current semantic contract
> and fail-closed support boundary are maintained in
> `docs/source/BatchedCompilation.rst` and executable tests. The current
> checkpoint executes an exactly authenticated deterministic CSI subset through
> typed control operations and a lane-local co-evolving `KernelIR` region.
> Delayed ITI, stochastic CSI, generalized cue/control transforms, and other
> co-evolving topologies still fail closed. Older statements below that CSI is
> wholly rejected describe the preceding exact-LCA checkpoint.

This branch is `feat/pec_batch_compile`, currently based on `origin/devel`.
The work is experimental and opt-in. Existing PEC fitting, `Composition.run`,
`ParameterEstimationComposition.log_likelihood`, LLVM, and PTX behavior should
remain unchanged unless a caller explicitly uses the new batched compiler API.

## High-Level Goal

The goal is to improve GPU utilization for PEC-style workloads that evaluate a
large number of independent stochastic simulations, especially DDM-like models
and graph models that combine LCA, DDM, gates, and transfer functions.

The main design choice is to compile over the actual PEC parallelism:

```text
parameter_set x subject x trial x estimate
```

or, for stateful graph models:

```text
parameter_set x subject x estimate
```

with trials looped inside the lane so lane-local state persists across trials.

This differs from the existing LLVM/PTX path, which tries to compile the general
PsyNeuLink execution machinery. That generality is useful, but it also carries
scheduler, port, state, and `is_finished` overhead that is often unnecessary for
static stochastic PEC batches.

## Current Public Surface

The public experimental API is in `psyneulink.core.batched`.

```python
from psyneulink.core.batched import BatchedCompositionCompiler

report = BatchedCompositionCompiler.diagnose(comp, backend="triton_cpu")
plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=256)
result = plan.run(inputs, parameter_sets, num_estimates, seed=11)

# Simulate + score experimental data with the on-device histogram likelihood:
ll = plan.log_likelihood(inputs, parameter_sets, num_estimates,
                         data=exp_data, categorical_dims=[True, False], bins=100)
```

The serial PEC data-fitting objective can be routed through this path (opt-in):

```python
pnl.PECOptimizationFunction(method=..., batched_backend="triton")
```

PEC entry points are not unified behind one simulation engine yet; objective
mode, distributed workers, and the public log-likelihood path still require the
post-CSI interface cleanup. Use the direct compiler API above when strict
no-fallback behavior is required.

Researcher-facing op registration (see "Code Map / specs.py"):

- `@batched_op(ComponentClass)` — a class-level op (elementwise function or
  mechanism), auto-bound from the body signature.
- `@batched_node_op("<node name>")` — an **instance-level** op for one node
  (keyed by name), whose `tl` body takes the node's whole combined input vector
  and may reduce it. This is how a `UserDefinedFunction` node (all UDFs share one
  class) gets a kernel — e.g. the CSI drift rate.  See
  `register_batched_instance_op` / `unregister_batched_instance_op`.

Supported backends (both run the *same* generated Triton kernels):

- `triton_cpu`: runs the kernels through Triton's interpreter on CPU (no CUDA).
  This is the default and the CPU test/debug path. It is slow (pure-Python /
  numpy interpretation), so keep cases small.
- `triton`: compiles and runs the kernels on a CUDA GPU.

There is no separate numpy CPU executor (`ir_debug` was removed): the CPU and
GPU paths execute the identical kernel source, including identical `tl.randn`
Philox draws.

The old prototype backend names `reference` and `ir_debug` are intentionally
rejected.

**Mode constraint:** Triton bakes interpret-vs-compiled when its library
`@triton.jit` functions (e.g. `tl.randn`) are first imported, so `triton_cpu`
(interpret) and `triton` (compiled GPU) cannot be used in the **same process**.
`triton_cpu` sets `TRITON_INTERPRET=1` before importing triton; a clear error is
raised if a process tries to mix the two. Validate CPU and GPU in separate
processes.

### Reference / testing model

PsyNeuLink itself is the correctness oracle, not a hand-written twin:

- The shared semantic cases construct independent Python and batched models,
  bind exact components and ports, and compare every output lane. Discrete
  values and execution counts are exact; deterministic fp32 values use declared
  per-case tolerances. Stochastic cases compare distribution summaries.
- **Prefer a deterministic comparison where one is possible.** It catches
  systematic drift that can be hidden by a plausible-looking distribution.
  Stochastic tests remain necessary for stream layout, Philox draws, and lane
  independence.
- Triton-interpreter and compiled-GPU pytest cases run in separate processes.
  The historical `pec_grid_correctness_check.py` script is useful for research
  exploration but is not the executable support gate. The deterministic CSI
  acceptance case runs in both modes; unsupported CSI variants fail closed.

`ParameterEstimationComposition.can_compile_batched(..., backend="triton")` is
diagnostic-only. It calls `BatchedCompositionCompiler.diagnose()` on the model
composition and does not route PEC fitting through Triton.

`BatchedSimulationResult.values` has shape:

```text
[parameter_set, subject, trial, estimate, outcome]
```

## Current Commit Stack

The **"Milestone Roadmap → Done"** section below is a historical record, not
the authoritative current status. Foundational commits (early architecture):

- `99b2313fc0` - initial opt-in Triton batched simulator for PEC models.
- `667963121b` - generated graph compiler for DDM and stateful graph cases.
- `0b8547955a` - KernelIR refactor and move Triton backend into
  `backend/triton`.
- `640c690a55` - component-owned Triton hooks with `@pnl_triton_op`.
- `4fd3be6718` - remove stability-flexibility-specific compiler/runtime paths.
- `653a57cbf9` - declarative batched op specs: class-keyed spec registry + the
  `@batched_op` auto-binding decorator; retire the monolithic DDM kernel.
- `960d3aba6d` - CPU interpret execution + PsyNeuLink oracle (see below).

Milestones toward the CSI surrogate (details in "Done" below): step 4 AtPass(0)
scheduling; step 6 instance-level UDF ops (`batched_node_op`); step 7 fires-once
integrating transfers; step 8 collapsing DDM threshold + control routing (CSI
compiles); step 5 **co-evolution loop** (CSI decision **and** RT match PNL
Python); a refactor factoring the LCA/DDM recurrence into one shared
`@triton.jit` helper each; and a `CSISurrogate` asv benchmark + a triton-vs-LLVM
comparison (`csi_triton_vs_llvm.py`; ~100x steady state, ~1.3x cold one-shot).

After `4fd3be6718`, stability-flexibility is no longer a special model family
in the batched compiler. It is just an example of a supported static stateful
graph.

After `653a57cbf9`, components are registered through
`psyneulink.core.batched.specs` instead of installing private methods on
component classes, and single-DDM models execute through the generated
`ddm_graph` kernel (the hand-written monolithic DDM kernel is gone; the
generated kernel was benchmarked checksum-identical and faster).

After `960d3aba6d`, there is one implementation per op (the Triton kernel body),
run compiled on GPU (`triton`) and interpreted on CPU (`triton_cpu`). The numpy
CPU executor (`ir_debug`) and the per-op numpy CPU bodies (`cpu_body` /
`cpu_execute`) were removed; PsyNeuLink itself is the correctness oracle.

## Code Map

### Root Batched Package

- `psyneulink/core/batched/compiler.py`
  - Public compiler facade.
  - Defines `BatchedCompositionCompiler`, `BatchedSimulationPlan`, and
    `BatchedCompileError`.
  - Dispatches plan execution to `backend.triton.run_triton`, choosing
    `device="cpu"` (interpret) for `triton_cpu` or `device="cuda"` (compiled)
    for `triton`.

- `psyneulink/core/batched/diagnostics.py`
  - Capability report and diagnostic dataclasses.
  - Unsupported components, functions, projections, and scheduler conditions
    should be reported here rather than failing deep in execution.

- `psyneulink/core/batched/ir.py`
  - Semantic graph-level dataclasses:
    `BatchedGraphIR`, node specs, projection specs, input/output specs,
    parameter specs, state specs, scheduler specs, and result container.
  - This IR describes PsyNeuLink-ish graph semantics and should not contain
    Triton syntax.

- `psyneulink/core/batched/graph.py`
  - Extracts a supported subset of `Composition` into `BatchedGraphIR`.
  - Classifies model/fusion/schedule kinds.
  - Builds node-qualified parameters and component bindings.
  - Rejects unsupported pieces with `BatchedDiagnostic`.

- `psyneulink/core/batched/specs.py`
  - Declarative batched op specs and the researcher-facing registration API.
  - `@batched_op(ComponentClass)` introspects the decorated kernel body's
    signature and auto-binds argument names against PNL `Parameters` metadata.
    Reserved names: `x`, `seed`/`rng_base`, `max_steps`, `lane_mask` (which
    lanes of the block are in range — for bounded loops that exit early).
    Irregular bindings
    use `bind={...}` with `specs.param(...)`.  The decorated body is the single
    kernel body (written against `triton.language`, `tl.*`), captured as Triton
    source.
  - Spec kinds: `ElementwiseFunctionSpec` (one kernel body), `MechanismOpSpec`
    (declared params/state/RNG/outputs with a Triton kernel body, plus a
    `triton_emit` escape hatch for irregular emission), `PassthroughMechanismSpec`,
    `DenseProjectionSpec`.  There are no CPU bodies — the same Triton kernel
    runs on CPU via interpret mode.
  - Registry is keyed by **exact** component class; subclasses are not
    inherited (PNL subclasses change semantics, e.g.
    LCA < RecurrentTransfer < Transfer).
  - **Instance-level ops** (`_INSTANCE_SPECS`, keyed by node **name**): resolved
    ahead of the class spec in `mechanism_spec_for` (also matches the unsuffixed
    name, so a registered op survives PNL's `-N` duplicate-name suffix on
    in-process rebuilds). `@batched_node_op("<name>")` builds one from a
    whole-input-vector `tl` body.
  - **Co-evolution step interface** on `MechanismOpSpec` (for the fused loop):
    `step_emit` (emit ONE integration step), `trial_states` (per-trial reset
    state, e.g. DDM `value`/`steps`/`finished`), and `finished_output` +
    `readout_emit` for a *terminator* op (produces its outputs after the loop).
  - IRs reference specs only by string `spec_key`, so `BatchedGraphIR` and
    `KernelIR` stay serializable and backend-neutral.

- `psyneulink/core/batched/prep.py`
  - Backend-neutral input/parameter normalization (`normalize_parameter_sets`,
    `prepare_inputs`, `lca_max_steps`, ...) shared by the CPU and GPU runtimes.

- `psyneulink/core/batched/components/`
  - Built-in batched op definitions, one module per component:
    `linear`, `logistic`, `passthrough` (Transfer/Processing),
    `mapping_projection`, `ddm` (the declarative reference example),
    `lca` (the escape-hatch example).
  - Registered on first lowering via `specs.ensure_builtin_specs()`.
  - These modules double as reference examples for researchers registering
    their own components.
  - `ddm` and `lca` each expose **two** forms sharing one recurrence: a
    run-to-completion body (internal loop) and a co-evolution `step_emit`, both
    calling a single shared `@triton.jit` helper (`_pnl_triton_ddm_update` /
    `_pnl_triton_lca_width2_recurrence`) so the math lives in one place. `ddm`
    also carries the collapsing-threshold recognition (`threshold_override_collapse`)
    used by `graph.py` to absorb the threshold-driving transfer.

- `psyneulink/core/batched/kernel_ir.py`
  - Backend-neutral execution-level IR.
  - Converts `BatchedGraphIR` into structured execution ops:
    `LoadInput`, `CallProjection`, `CombineSum`, `CombineProduct`,
    `CallFunction`, `CallMechanism`, `StoreOutput`, `StoreFlag` (diagnostics),
    and stateful `InitializeState` / `ForTrials`.
  - Dispatch is driven by `spec_kind`/`spec_key`/`rng_streams`/`op_outputs`
    node attrs written at lowering time; kernel_ir does not import the
    registry.
  - `coevolving_graph` reuses the stateful lane layout + `ForTrials`; the fused
    per-step loop itself is emitted by `emit/ops.py`, not a new IR op.
  - The intent is that a future MLIR backend starts here, not from generated
    Triton source.

- `psyneulink/core/batched/bindings.py`
  - Sidecar live-object map from graph specs back to the actual PsyNeuLink
    components.
  - Keeps live Python objects out of `BatchedGraphIR` and `KernelIR`.

- `psyneulink/core/batched/registry.py`
  - Ties lowering, backend availability, and capability reporting together.
  - Checks optional `torch`/`triton` availability for the Triton backend.

### Triton Backend

All Triton-specific code should stay under:

```text
psyneulink/core/batched/backend/triton/
```

- `runtime.py`
  - Imports Torch/Triton lazily.  `device="cpu"` enables interpret mode (sets
    `TRITON_INTERPRET=1` before importing triton) and runs on CPU tensors;
    `device="cuda"` requires a GPU.  Guards against mixing interpret and
    compiled modes in one process.
  - Prepares Torch buffers (via `prep.py`).
  - Dispatches to stateless graph, DDM graph, stateful graph, or **co-evolving
    graph** kernels (the last shares the stateful runner, one kernel name apart).
  - The monolithic DDM and stability-flexibility kernel paths have been
    removed; everything goes through the generated graph kernels.

- `cache.py`
  - Writes generated source to an importable module, tagged by interpret vs
    compiled mode so both can coexist in `sys.modules`.  `interpret_scope`
    holds `knobs.runtime.interpret` across import and launch.

- `emit/` (package; `graph_emit.py` is now a thin re-export shim)
  - Emits inspectable Triton Python source from `KernelIR`.
  - `emitter.py`: `TritonGraphEmitter` (composed from the mixins below) — shared
    state, `emit()` orchestration, signature/module rendering, params/state setup.
  - `lanes.py` (`LaneEmitMixin`): lane decode, RNG-base layout, raw input loads.
    Owns `RNG_STREAM_STRIDE` and the fixed-stride stream allocation (see "Stream
    layout") — the reason offsets do not depend on the step caps.
  - `ops.py` (`OpEmitMixin`): per-`KernelOp` emission + value table. New op kinds
    (e.g. truncation `StoreFlag`) are added here.  Also emits the **co-evolution
    fused loop** (`_emit_coevolving_trial_loop`): hoist loop-invariant ops out
    once (Triton loop-body vars don't escape the loop), step the stepper ops
    inside masked by the terminator's `finished`, then run the terminator
    `readout_emit` + post-loop ops after.
  - Spec-driven op emission (declarative elementwise/mechanism calls plus
    `triton_emit` escape hatches) resolved through `spec_key`.

- `api.py`
  - Defines `@pnl_triton_op`, `TritonOpTemplate`, `TritonOpCall`,
    `TritonEmitContext`, and `TritonOpError`.
  - `@pnl_triton_op` captures inspectable Python source and emits it as a
    `@triton.jit` helper without importing Triton at PsyNeuLink import time.
  - Helpers may reference `tl` (`triton.language`); other globals/closures are
    rejected — **except** helper templates passed via `helpers=`, which the body
    may call by name (the emitter emits those device functions ahead of the
    caller, via `TritonOpTemplate.dependencies`). This lets one shared recurrence
    helper back both a run-to-completion loop and a single-step path.
  - `backend/triton/__init__.py` imports `runtime` lazily so importing the
    spec/registration API does not create an import cycle.

- `cache.py`
  - Writes generated source to a real Python module path and imports it. This
    is necessary because Triton JIT requires inspectable source.

- `source_builder.py`
  - Small indentation/source helper for Triton emission.

## Current Supported Subset

Supported mechanisms:

- `TransferMechanism`, `ProcessingMechanism` (stateless)
- `TransferMechanism` with `integrator_mode=True` **when fires-once** (resets
  each trial via `AtTrialStart`, fires once via `AtPass`) — lowered statelessly
  as `function(a*input + b)` (Adaptive/Simple single affine step); a multi-step
  integrating transfer is still rejected
- `DDM` (fixed **or** collapsing boundary; a control mechanism OVERRIDE'ing the
  threshold from a SimpleIntegrator transfer is recognized and that transfer is
  absorbed into the DDM)
- `LCAMechanism`, an exact deterministic width-2 subset with either a static
  `TimeScale.TRIAL` execution-count threshold or the narrowly validated scalar
  cue -> `OVERRIDE` threshold paths; see "LCA Caveats"
- `ControlMechanism` — recognized for LCA-termination and DDM-threshold OVERRIDE
  routing. Exact scheduler-visible LCA control lowers to registered controller
  compute plus `ApplyModulation`; absorbed threshold control remains part of the
  authenticated DDM boundary.

Supported functions:

- `Linear`, `Logistic`, `DriftDiffusionIntegrator` (DDM)
- arbitrary elementwise/reduction bodies via `@batched_op` (class) or
  `@batched_node_op` (a specific node instance, e.g. a `UserDefinedFunction`)

Supported projections: dense, feed-forward `MappingProjection` using
`MatrixTransform(operation=DOT_PRODUCT, normalize=False)` with no weight or
exponent and a finite real fp32-representable matrix. Feedback and other
projection semantics fail closed.

Supported input combines: single input, `SUM`, `PRODUCT`.

Supported executable schedules / fusions:

- `static_graph` schedule, including `AtPass(0)` origins.
- finite stateless precomputed traces and fixed-count stateful
  `Always`/`WhenFinished` producer/follower traces;
- `dynamic_lane_local` for the exact scalar-controlled LCA/follower chain and
  the exact deterministic CSI co-evolving topology;
- Fusions: `stateless_graph`, `ddm_graph`, `stateful_graph` (sequential
  stateful execution with a static threshold or the validated scalar identity
  cue -> `OVERRIDE` threshold path), and `coevolving_graph` for the authenticated
  deterministic CSI graph.

Generic `Always`/`WhenFinished` co-evolution, multiple dynamic controllers,
and `AtPass(n>0)` onset are recognized where possible but are not executable.

Unsupported scheduler conditions should return diagnostics. Do not silently
fall back to Python or LLVM inside this stack.

## Capability Gaps — Real Research Models

The stability-flexibility model in the test suite is a *toy*. The model a
researcher actually runs is the **CSI surrogate** stability-flexibility model in
`csi_model_surrogate.py` (fit at PEC scale by `csi_fit_parameter_recovery.py`:
128 trials x 10k estimates, CMA-ES over 5 parameters). It is the north-star
target for "supporting researchers' models," and the roadmap in "Good Next
Steps" is scoped to close the gap to it.

`BatchedCompositionCompiler.diagnose()` now accepts the exact deterministic
surrogate after its drift UDF is registered when `iti=0`, `csi_repeat=0`,
`csi_switch=1`, `ddm_noise=0`, and `lca_noise=0`. The public compiler lowers it
to explicit effective-parameter/control operations and one authenticated
`lane_local_coevolving` pass region; Triton interpreter and compiled CUDA both
match fresh Python oracle cases. This is the first narrow CSI checkpoint, not
full research-model support: nonzero ITI, fitted/generalized CSI cue transforms,
and stochastic DDM/LCA execution still reject. Older measurements below
describe the retired heuristic emitter and are not evidence for the current
typed path.

1. **Custom / UDF reduction op** (`driftRate`, a nested-logistic reducing a
   7-vector to a scalar) — RESOLVED (step 6). `batched_node_op("<node name>")`
   registers an instance-level op whose `tl` body takes the node's whole input
   vector. (A cross-element reduction, so it rides the mechanism/`triton_emit`
   path, not the scalar→scalar elementwise path.)
2. **Stateful integrating `TransferMechanism`** (`taskInput`) — RESOLVED (step 7).
   A fires-once (`AtPass`), reset-each-trial (`AtTrialStart`) integrator advances
   one affine step from its initializer and lowers statelessly as
   `function(a*input + b)`.
3. **`AtPass` multi-pass timing** — PARTIAL. `AtPass(0)` is executable, including
   the deterministic CSI origins. `AtPass(n>0)` (the ITI onset) is still
   deferred.
4. **Collapsing DDM threshold** (`threshold_collapse`) — RESOLVED (step 8). The
   DDM kernel boundary is now `threshold(step) = threshold + collapse*step`.
5. **Control routing** (`csiOverride`/`thresholdOverride` OVERRIDE a target
   param) — RESOLVED FOR THE EXACT SLICE. `csiOverride` -> LCA
   `termination_threshold` is represented by a held effective value and
   `ApplyModulation`; `thresholdOverride` -> DDM `threshold` (monitoring the
   `thresholdMechanism` SimpleIntegrator) is authenticated and absorbed into
   the DDM boundary. The latter retains its held override across trials.

### Typed CSI performance checkpoint (2026-08-17)

The current typed deterministic path was measured on an RTX 2080 Ti with a
fresh Triton cache. The CSI workload was 4 parameter values x 512 estimates x
128 trials = 262,144 simulations, with `iti=0`, `csi_repeat=0`,
`csi_switch=1`, alternating cue counts 1/3, `ddm_noise=0`, and `lca_noise=0`.
Seven warm sweeps were reduced by their median:

| | compile estimate | warm sweep | cold total | warm simulations/s |
| --- | ---: | ---: | ---: | ---: |
| typed Triton GPU | 3.453 s | 19.2 ms | 3.472 s | 13.64 M |
| LLVM PEC | 3.417 s | 1.426 s | 4.843 s | 0.184 M |

This is **74.2x steady-state** and **1.4x cold one-shot**. The output sums after
excluding trial 0 match exactly (`410972.2`). LLVM retains its known first-trial
initialization artifact, so its all-trial checksum is not an accuracy oracle;
the elementwise Python/interpreter/GPU acceptance tests remain authoritative.

For context, detached historical commit `7ad3d0637d` was rerun on the same GPU
and software environment with its original broader configuration
(`csi_switch=0`, `ddm_noise=.1`). It measured 16.5 ms Triton, 1.539 s LLVM, and
**93.5x steady-state**. Relative to that run, the typed GPU path is about 16%
slower in absolute warm time and its GPU/LLVM speedup is about 21% lower. This
is directionally useful, but not an exact regression comparison because the
old stochastic/zero-cue model is outside the current executable boundary.

For a strict equal-output comparison, both implementations were also run with
the same constant cue count 1, zero LCA/DDM noise, zero threshold collapse, and
otherwise identical 262,144-simulation workload. Both GPU checksums were
exactly `453365.75`. Against one common current LLVM median of 1.550 s, the
retired GPU kernel took 14.9 ms (**104.3x**) and the typed GPU kernel took
20.2 ms (**76.7x**). At equal work and output, the typed path is therefore about
36% slower in warm GPU time and the speedup ratio is about 26% lower. The cost
buys explicit scheduler/control/state semantics and fail-closed authentication;
profiling is still needed to apportion it among the emitted loop and state
operations.

A common-size GPU-only sweep used 8 parameter rows x 1,024 estimates x 128
trials = 1,048,576 simulations for each model:

| model | median warm time | simulations/s |
| --- | ---: | ---: |
| DDM | 8.6 ms | 122.3 M |
| transfer -> DDM | 8.9 ms | 118.5 M |
| deterministic LCA, 64 steps | 8.0 ms | 130.9 M |
| toy stability/flexibility | 58.7 ms | 17.9 M |
| typed deterministic CSI | 27.8 ms | 37.7 M |

These rates compare complete simulations, not equal amounts of mechanism work;
they are useful for capacity planning, not per-operation efficiency claims.

### Historical CSI experiment (not current support)

The following measurements describe the retired heuristic emitter and are kept
only as performance history. `make_stab_flex` previously compiled and ran on
`triton_cpu`/`triton` for
`iti=0` **and `iti>0`**, with **both decision outcomes and response times
matching** PNL Python mode (RT within ~1 DDM step). That path required the
**co-evolution loop** (see below): the CSI LCA is `Always`-scheduled and its
activation feeds the drift, so the LCA and DDM step together each timestep. For
`iti>0` the fused loop gated the `AtPass(iti)` task-input onset per step. These
terminal-output checks did not establish general scheduler/control fidelity,
so that emitter was retired. The current deterministic `iti=0` checkpoint uses
a different, typed `KernelIR` region; the historical `iti>0` result below is
not current support.

**Benchmark (vs LLVM PEC).** `csi_triton_vs_llvm.py` compares the co-evolving
CSI on the `triton` GPU path against PNL's PEC `grid_evaluate` LLVM baseline on
the same workload (sweeping `non_decision_time` — the DDM `threshold` is already
controlled, so PEC cannot also modulate it; that conflict raises
`len(mod_afferents) <= 1` in LLVM). At PEC scale (4 params x 512 estimates x 128
trials = 262k sims, RTX 2080 Ti), **quote the steady-state figure**:

| | cold compile (one-time) | steady state, per parameter eval |
| --- | --- | --- |
| LLVM | ~2.9 s | ~322 ms |
| triton | ~3.2 s (cold kernel cache) | ~3.6 ms |

- **Steady state, ~100x.** Four warm parameter evals: LLVM ~1470 ms vs triton
  ~14.4 ms. This is the number that matters for fitting, where one compile is
  amortised over hundreds of objective evaluations.
- **Cold one-shot, ~1.3x.** A single 4-point sweep including compilation: LLVM
  ~3.9 s vs triton ~3.0 s. Compilation costs are near-identical, so a one-off
  job is close to a wash. This one depends on Triton's on-disk kernel cache:
  with it already warm the compile drops to ~1.6 s and the ratio reads ~2.3x, so
  use `TRITON_CACHE_DIR` pointed at an empty directory for a true cold number.

`csi_triton_vs_llvm.py` reports all of this directly — compile, warm sweep and
cold total per side, then both speedups. (It previously printed a single ~288x,
timing triton warm against LLVM cold; that asymmetry alone was the whole
difference between 288x and ~100x.)

**Accuracy.** The script's "checksum" is `np.sum` over every decision and
response time — 262k lanes collapsed to one scalar, mixing two quantities, so a
decision rate that is too high can cancel response times that are too short.
Componentwise the two paths agree far better than that number conveys: decision
rate to 0.2%, RT quantiles exactly, and — once trial 0 is excluded — RT mean and
standard deviation to ~0.1% (e.g. 0.4449/0.0952 vs 0.4444/0.0951). Trial 0 is
the entire residual, and only under noise: **noise-free, PNL Python and LLVM
agree exactly, and the batched path matches both to within one DDM step** (see
`test_csi_surrogate_noise_free_matches_pnl_elementwise`). Note also that the
LLVM baseline is not reproducible run to run despite a fixed `initial_seed`,
while the batched path is bit-identical — which is why the deterministic test,
not the checksum, is the real accuracy check.

> Earlier revisions of this file reported ~62x here with a large unexplained
> checksum gap. Both were artifacts of the co-evolving parameter-set bug (see
> "Done"): with `--param-evals 4`, triton computed only parameter set 0 while
> LLVM evaluated all four, so it was timed on a quarter of the work and its
> checksum summed three buffers of uninitialized memory. Fixing it *raised* the
> speedup — the GPU was nowhere near saturated at 262k lanes — and collapsed the
> checksum gap. Treat any pre-fix co-evolving multi-parameter number as void.

asv retains a `CSISurrogate` benchmark identifier for historical continuity,
but the builder still skips it because its original research configuration is
broader than the newly admitted deterministic slice. Its recorded results
predate the parameter-set fix and are void (`PARAM_SETS = 8`); there is no
current CSI baseline.

## Fusion and Lane Layout

Current fusion kinds:

- `stateless_graph`
  - Transfer/Processing-only static graph.
  - Lane layout: `(parameter_set, subject, trial, estimate)`.

- `ddm_graph`
  - Static graph with one non-persistent mechanism op (e.g. a DDM), possibly
    behind stateless nodes.  A single-DDM composition is just the one-node
    case (`model_kind="ddm"` is retained as a naming policy: unqualified
    public parameter names).
  - Lane layout: `(parameter_set, subject, trial, estimate)`.

- `stateful_graph`
  - Static graph with lane-local state where the stateful ops run **sequentially**
    to completion (e.g. an LCA with a static threshold, or with the narrow
    scalar identity cue -> `OVERRIDE` threshold path, settles before a DDM).
  - Lane layout: `(parameter_set, subject, estimate)`.
  - Trials run inside the Triton lane so LCA state persists across trials.

- `coevolving_graph`
  - Current typed optimization for the exactly authenticated deterministic CSI
    graph. Its `lane_local_coevolving` `ForPasses` region carries explicit LCA
    and DDM finished identities, effective controlled count, trial-local DDM
    state, persistent LCA state, and a persistent held threshold override.
  - The retired emitter inferred `Always`/`WhenFinished` behavior from op order.
    That history is documented below, but it is not used by the current path.
    Other coupled stateful graphs still require their own complete semantic
    admission and typed region contract.

Fusion kind is a dispatch/optimization detail. It should not encode model
architecture semantics.

## Stability-Flexibility Status

The **toy** stab-flex model is treated as a generic graph. Its LCA threshold
uses the narrowly validated scalar identity cue -> `OVERRIDE` chain:

```text
model_kind="graph"
fusion_kind="stateful_graph"   # threshold-count LCA settles, then the DDM decides
schedule_kind="static_graph"
```

The **CSI surrogate** (the realistic model, with an `Always`-scheduled LCA
co-evolving with the DDM) now executes for the deterministic identity-cue,
zero-ITI slice. Admission authenticates the complete typed graph and the same
scheduler, control, and state semantics required of any composition with that
topology; there is no public CSI model kind or display-name dispatch. Research
variants with delayed ITI, fitted affine cue transforms, or noise still fail
closed.

There should be no `STABILITY_FLEXIBILITY_MODEL`, no
`stability_flexibility_roles` metadata, no forced `cue`/`correct` aliases, and
no monolithic SF Triton kernel in the normal runtime path.

Parameter sets should use generic node-qualified names, for example:

```python
{
    "DDM.threshold": 0.05,
    "DDM.noise": 0.0,
}
```

Old SF-only names such as `threshold`, `ddm_noise`, `lca_noise`,
`automaticity`, or `scale` should not be reintroduced as special compiler
aliases. If users need friendlier names later, that should be a general
parameter naming/alias policy, not a model-specific shortcut.

## LCA Caveats

The current batched LCA hook is exact for a narrow deterministic width-2
`LCAMechanism` subset. It is not an approximation: accepted configurations
must match Python PsyNeuLink, and all other configurations fail closed.

The real PsyNeuLink `LCAMechanism` is implemented through:

```text
LCAMechanism
  -> RecurrentTransferMechanism
      -> TransferMechanism
          -> LeakyCompetingIntegrator + Logistic
```

The batched hook currently handles:

- width 2 only;
- persistent lane-local `pre` and `act` state;
- recurrent coupling via scalar `self_excitation` and `competition`;
- finite Logistic parameters within the fp32 range (`gain`, `bias`, `x_0`,
  `scale`, and `offset`) that are scalar or exactly uniform broadcasts,
  including lane-specific fitted values;
- finite recurrent parameters within the fp32 range that are scalar or exactly
  uniform broadcasts, with a strictly positive `time_step_size`;
- activation initialization as `Logistic(0)` through the registered decorated
  Logistic implementation;
- zero integrator initializer and offset, deterministic zero noise, no clip,
  canonical recurrent matrix, and `Never` reset;
- a nonnegative `TimeScale.TRIAL` execution-count threshold, either static or
  supplied through a narrowly validated scalar identity cue -> `OVERRIDE`
  control chain;
- host discretization of static thresholds using `ceil`, with a minimum of one
  execution; both the effective static count and runtime cue values are bounded
  by `2**24`, and runtime cues must already be exact nonnegative integers before
  fp32 conversion;
- a positive-integer `max_executions_before_finished` cap for each LCA node,
  with persistent state between trials; and
- an absorbed identity cue whose lowered Linear parameters are
  validated-default-only in runtime parameter rows. Only an absent explicit
  condition or `AtPass(0)` at the default `ENVIRONMENT_STATE_UPDATE` time scale
  on that cue is accepted; other explicit cue/controller/target conditions or
  time scales fail closed.

It does not yet cover:

- arbitrary width;
- arbitrary recurrent matrices;
- separate recurrent input ports;
- generic `combination_function`;
- learning;
- full output-port variants;
- convergence or other termination measures;
- stochastic noise, clipping, nonzero/custom initialization, or trial reset;
- non-identity or otherwise general termination-threshold control;
- generic co-evolving `Always`/`WhenFinished` scheduling and control outside
  the exact deterministic CSI topology.

The old handwritten recurrence test and heuristic co-evolution checks are not
semantic support evidence. Fresh Python and batched compositions now compare
representative cases across the accepted subset on both Triton interpreter and
compiled GPU, including the first step, multiple trials, threshold counts,
Logistic parameters, and distinct parameter lanes.

## RNG and Common Random Numbers

The Triton backend uses `tl.randn` in stochastic component helpers. The exact
supported LCA subset is deterministic and declares no RNG stream; DDM remains
the active stochastic mechanism. RNG offsets are derived from lane indices and
stream layout.

### Stream layout

Every RNG stream owns a fixed `RNG_STREAM_STRIDE` (2**32) of Philox counter
space (`emit/lanes.py`). Stream identity goes in the high 32 bits of the offset
and the step index in the low 32, which `randint4x` splits into two counter
words:

```text
random_base = lane_index * (n_streams * RNG_STREAM_STRIDE)   # int64
stream_off  = random_base + slot * RNG_STREAM_STRIDE
draw(step)  = tl.randn(SEED, stream_off + step)
```

**Offsets do not depend on any step cap.** That is the point: step caps are
safety or loop bounds, and raising one must not change results.
Streams used to be packed by cap (`n_lca*LCA_MAX_STEPS + n_ddm*MAX_STEPS`), which
had two consequences, both now gone:

- Changing a cap changed every downstream offset, so simulations silently
  produced different draws — even when nothing truncated. Bumping a cap for
  safety looked like a bug and broke reproducibility against recorded numbers.
- A stream could **run off its own end**. The retired stochastic co-evolving LCA
  stepped up to `MAX_STEPS` times but was strided by `LCA_MAX_STEPS`, so with
  `MAX_STEPS > LCA_MAX_STEPS` and `lca_noise > 0` its two units drew the same
  sequence time-shifted, and then collided with the DDM's stream. Observable as
  a decision rate that drifted with the cap (0.781 / 0.773 / 0.769 at
  `MAX_STEPS` 128 / 256 / 512); it is now 0.82294 at all three.

Two constraints worth knowing before touching this:

- The 64-bit base must be built **inside** the kernel from constants and lane
  arithmetic. Passing a precomputed 64-bit base as a runtime kernel argument
  silently drops the high word on the GPU (the offset stays 32-bit) and
  collapses every stream onto the same draws — it fails quietly, not loudly.
- The lane index is int32; widen it (`.to(tl.int64)`) *before* scaling by the
  stride.

`runtime._check_step_caps` guards the invariant that a cap never exceeds the
per-stream space. At a 2**32 stride that is a sanity bound, not a real limit.

Common random numbers:

- `common_random_numbers=True`
  - Excludes `parameter_set` from the base RNG index.
  - Intended to make stochastic draws comparable across parameter sets.

- `common_random_numbers=False`
  - Includes `parameter_set` in the RNG base.

The current backend is deterministic for a fixed backend, seed, shape, and
common-random-number setting. It is not expected to be bitwise-identical to
PsyNeuLink LLVM/PTX RNG streams.

## Precision and Termination

The Triton runtime currently uses `torch.float32` buffers and Triton `tl.float32`
state. Parity tolerances should account for this. Every published outcome tensor
is checked before outcome-buffer host conversion or likelihood scoring; NaN or
infinite values raise `BatchedNumericalError`.

DDM and LCA integration use bounded loops:

- `MAX_STEPS` for DDM.
- `LCA_MAX_STEPS` for LCA.

The existing LLVM/PsyNeuLink path uses mechanism/scheduler termination
machinery. The Triton path uses caps and must be given caps large enough for
the parameter/input regime.

### Bounded-loop early exit (DDM; historical co-evolution measurements)

The run-to-completion DDM body (`components/ddm.py`) stops as soon as every
in-range lane of the **block** has finished, instead of always running to the
cap. The retired fused co-evolution loop used the same optimization:

```python
step = 0
while (step < MAX_STEPS) & (tl.max(tl.where(mask & (finished == 0.0), 1, 0)) > 0):
    ...
    step += 1
```

This is a pure optimization: a finished lane is already frozen by the op
(`_pnl_triton_ddm_update` gates on `finished`, and freezing on the terminator's
`finished` is the `step_emit` contract), so the skipped iterations were no-ops.
RNG streams are indexed by absolute `step`, so skipping does not shift draws.
Results are unchanged — checksums are identical before/after at every cap.

Two things the test must get right, and both are easy to get wrong:

- **Exclude out-of-range lanes.** Lanes past `total_lanes` load default
  parameters (drift 0, noise 0), never finish, and would pin every partial
  block open to the cap. Hence the `mask &`. A body that wants this needs the
  reserved `lane_mask` argument (see `specs.py`).
- **The exit is per block, not per lane.** A block runs until its *slowest*
  lane finishes, so the win tracks `max` over the block, not the mean.

Consequence: **`MAX_STEPS` is now free to set generously** — in both senses.
Raising it costs almost nothing (this section), and it does not change your
draws (see "Stream layout"; before that change it did). So prefer a cap that
avoids truncation over one tuned for speed. Measured on an RTX 2080 Ti:

| workload | cap | before | after |
| --- | --- | --- | --- |
| CSI co-evolving (4x512x128, 262k sims) | 512 | 58.8 ms | 13.9 ms |
| CSI co-evolving | 1024 | 106.7 ms | 14.1 ms |
| `ddm_graph` (8x32x8192, 2.1M lanes) | 256 | 17.9 ms | 12.2 ms |
| `ddm_graph` | 1024 | 44.9 ms | 10.0 ms |

Runtime is now flat in the cap rather than linear. End-to-end, the CSI GPU
parameter recovery (128 trials x 4000 estimates x 150 iterations) went 9.4 s ->
5.3 s with identical recovered parameters and log-likelihood. The `triton_cpu`
interpret test suite also roughly halved (569 s -> 321 s).

**Truncation visibility (roadmap step 2 — DONE for DDM).** A bounded op may
declare trailing diagnostic returns via `MechanismOpSpec.diagnostics` (the DDM
body returns a `truncated` flag — still inside the boundary after `max_steps`).
The lowering emits a `StoreFlag` KernelOp per diagnostic into a separate per-lane
`diag` buffer (only present when a kernel has diagnostics, so the stateless
golden is unchanged); the runtime aggregates the truncated fraction per node into
`result.metadata["truncation"]`, **warns** by default and **raises**
`BatchedTruncationError` under `run(..., strict_truncation=True)`. The channel is
node-generic. The retired co-evolving DDM emitter also used it; a future
KernelIR-based coupled terminator may reuse the same channel once that execution
path is semantically supported.

The supported LCA sizes `LCA_MAX_STEPS` from metadata and either the largest
host-discretized static threshold or the validated integer runtime cues. Each
node's `max_executions_before_finished` is also applied, and every trial still
executes at least once. General controlled thresholds and co-evolving execution
are rejected rather than assigned an inferred cap.

## Contrast With LLVM/PTX

LLVM/PTX path:

- integrated with the existing PsyNeuLink compiled execution stack;
- aims to preserve broader `Composition.run` semantics;
- has component, port, projection, scheduler, state, and `is_finished`
  machinery;
- supports PEC through the current controller/grid-evaluate/log-likelihood
  path;
- can target CPU LLVM and GPU PTX;
- tends to carry significant overhead for static stochastic PEC batches because
  it preserves general graph execution machinery.

Triton batched path:

- deliberately narrower;
- opt-in and experimental;
- optimized around many independent stochastic lanes;
- erases static scheduler overhead;
- uses structure-of-arrays input/parameter/state/output buffers;
- generates kernels over batches of lanes rather than compiling the full
  PsyNeuLink object model;
- computes the PEC objective on-device via a **histogram** likelihood
  (`plan.log_likelihood`); a KDE option is not yet implemented;
- does not aim to support arbitrary compositions in this branch.

Benchmark scripts should compare Triton against the current PEC
`grid_evaluate` style LLVM/PTX baselines, not against repeated Python
`Composition.run()` calls.

## Validation Commands

Focused checks used during this work:

```bash
.venv/bin/python -m compileall -q psyneulink/core/batched tests/composition/pec Scripts/Debug/pec_batch_compile
.venv/bin/python -m pytest tests/composition/pec/test_batched_compile.py tests/composition/pec/test_batched_reference.py -q -n 0
```

Numeric tests run the real kernels on CPU through Triton interpret mode and so
require `torch`+`triton` importable (no CUDA); they skip otherwise. They cannot
share a process with compiled GPU runs.

Correctness scripts:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python Scripts/Debug/pec_batch_compile/pec_grid_correctness_check.py --strict
PYTHONUNBUFFERED=1 .venv/bin/python Scripts/Debug/pec_batch_compile/ddm_batch_compile_smoke.py
PYTHONUNBUFFERED=1 .venv/bin/python Scripts/Debug/pec_batch_compile/stability_flexibility_batch_compile_smoke.py
```

**GPU parameter recovery (steps 9 + 10 end-to-end).** `csi_batched_parameter_recovery.py`
generates CSI experimental data at known "true" parameters, then fits them back
through the batched path (`PECOptimizationFunction(batched_backend="triton")`,
optuna CMA-ES, on-device histogram likelihood) and reports recovered vs. true:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python Scripts/Debug/pec_batch_compile/csi_batched_parameter_recovery.py \
    --trials 128 --estimates 8000 --max-iterations 400 --seed 1
```

Recovers `gain` / `csi_switch` / `non_decision_time` (the params that map cleanly
onto the batched IR; the DDM `threshold` is left fixed — it is already driven by
the collapse control, the same `mod_afferents<=1` conflict that blocks fitting it
under LLVM PEC). A 128-trial x 8000-estimate x 400-iteration fit (~410M sims) runs
in ~22 s on an RTX 2080 Ti; `non_decision_time` recovers tightly, `gain`/`csi_switch`
within the expected identifiability of a single 128-trial data set. Use
`--backend triton_cpu` only for a tiny interpret-mode smoke test.

**Identifiability (not a bug).** With the experimental-data-anchored bins, recovery
is stable across `num_estimates`. `non_decision_time` is strongly identified — its
log-likelihood varies by hundreds of units across its range, so it recovers tightly
and sharpens with more estimates. `gain` (and to a lesser degree `csi_switch`) are
**weakly identified** in a single 128-trial data set: sweeping `gain` over 6-20
moves the log-likelihood by only ~8 units and the surface is nearly flat/bumpy for
`gain >= 10`, so the MLE sits ~2-3 above the true value and more estimates cannot
create a gradient that the data does not contain. Recovering `gain`/`csi_switch`
well needs more trials or a design that makes the LCA dynamics matter, not more
estimates.

## Benchmarking Methodology

Every trap below produced a number that looked entirely credible — tight spread,
plausible magnitude, reproducible on re-run — and was wrong by 2.5x to 30x. Check
all four before quoting anything.

**1. Time warm against warm.** Both backends compile, and the costs are
comparable: ~3.2 s for triton against a cold kernel cache, ~2.9 s for LLVM on its
first evaluation. Warming one side and not the other is worth ~2.5x on the CSI
comparison alone. Decide which question is being asked and be explicit:

- *fitting* (hundreds of objective evals, compile amortised) → steady state;
- *one-shot* (a single sweep) → include compilation for both.

`gpu_batch_compile_benchmark.py` gets this right — its `median_ms` excludes the
first call and `first_ms` reports it separately. `csi_triton_vs_llvm.py` does
not (see "Benchmark (vs LLVM PEC)").

Triton compilation has three regimes, so say which one a number reflects: cold
kernel cache ~3.2 s, warm cache in a fresh process ~1.7 s, same process ~14 ms.
`TRITON_CACHE_DIR` pointed at a scratch directory gives a genuine cold measure
without disturbing the real cache.

**2. One process per case.** Cases measured in the same process contaminate each
other through accumulated allocations, kernel-cache state and clock/thermal
history. A model benchmarked after two others reported **33 ms where the same
case alone reported 14 ms** — and both readings were internally tight (32.9-33.7
vs 13.5-14.1), so the spread gave no warning at all. This is the default now
(`--no-isolate` restores the old behaviour); `first_ms` is not comparable across
isolated rows because each pays a fresh compile.

**3. Size the case past the launch-bound region.** Below roughly 8k lanes the GPU
is idle and the clock is measuring fixed overhead, not the kernel. CSI throughput
against lane count (RTX 2080 Ti, BLOCK=128):

| lanes | sims | time | throughput |
| --- | --- | --- | --- |
| 512 | 65,536 | 14.9 ms | 4.4M/s |
| 2,048 | 262,144 | 14.1 ms | 18.6M/s |
| 8,192 | 1,048,576 | 13.8 ms | 75.7M/s |
| 32,768 | 4,194,304 | 35.2 ms | 119.0M/s |
| 131,072 | 16,777,216 | 126.3 ms | 132.8M/s |

16x the work for the same wall clock, from 512 to 8,192 lanes. A 512-lane case is
4 thread blocks on 68 SMs. Speedups measured there are meaningless in *both*
directions — LLVM has large per-call overhead at small sizes too.

**4. Do not compare `sims/s` across models.** A "sim" is a trial, and trials
differ enormously: a toy stab-flex trial runs 1200 LCA steps then a DDM, a CSI
trial co-evolves ~44. That is ~27x, so cross-model throughput says nothing.
Speedup *ratios* remain comparable.

Applying all of this — isolated, warm, matched at 65,536 lanes:

| model | LLVM | triton | speedup | checksum agreement |
| --- | --- | --- | --- | --- |
| `ddm` | 218.7 ms | 1.94 ms | 113x | 0.30% |
| `ddm_graph` | 257.8 ms | 2.35 ms | 110x | 0.30% |
| `stability_flexibility` | 780.7 ms | 13.59 ms | 57x | 0.13% |

An earlier revision reported 9.1x for stability-flexibility, which prompted this
section: a 512-lane case measured in a contaminated process. It is not the
outlier it appeared to be.

**Machine note.** These were taken under WSL2, which shares the GPU with the
Windows host — 5-24% background load and SM clocks idling at 330-435 MHz of 2100.
Absolute figures are therefore conservative, and the GPU should be checked
(`nvidia-smi`) before a measurement run.

Benchmark script:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python Scripts/Debug/pec_batch_compile/gpu_batch_compile_benchmark.py --backend triton --backend llvm --skip-ddm --skip-ddm-graph --sf-case 8x32x8192 --repeats 1 --warmups 0
```

Adjust case sizes based on available GPU memory and runtime. The benchmark
script supports:

```bash
--backend triton_cpu
--backend triton
--backend llvm
--backend ptx
--ddm-case PARAMSxTRIALSxESTIMATES
--ddm-graph-case PARAMSxTRIALSxESTIMATES
--sf-case PARAMSxTRIALSxESTIMATES
--no-isolate            # measure every case in one process (see "2." above)
```

(`triton` and `triton_cpu` cannot be combined in one invocation.)

For the **co-evolving CSI surrogate** specifically (not covered by
`gpu_batch_compile_benchmark.py`), use:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python Scripts/Debug/pec_batch_compile/csi_triton_vs_llvm.py \
    --trials 128 --estimates 512 --param-evals 4
```

It sweeps `non_decision_time` (the DDM `threshold` is already controlled, so PEC
cannot also modulate it → LLVM `mod_afferents<=1`). It reports compile, warm
sweep and cold total per side: ~100x steady state, ~1.3x cold one-shot. See
"Benchmark (vs LLVM PEC)" and "Benchmarking Methodology".

The one-off scripts above give ad-hoc triton-vs-LLVM numbers. To **track**
batched-compiler performance across commits, use the asv suite in `benchmarks/`
(see `benchmarks/README.md`). It reuses the existing venv (GPU box) and
forward-tracks — after each commit:

```bash
.venv/bin/asv run --set-commit-hash $(git rev-parse HEAD)   # record current commit
.venv/bin/asv publish && .venv/bin/asv preview              # dashboard
```

`.asv/results/` is committed (per-commit timings + checksums); `.asv/env|html/`
are git-ignored. The `--set-commit-hash` flag is required for `existing`
environments to persist results.

## Milestone Roadmap

### Done

- **Declarative batched op specs** (`653a57cbf9`). `psyneulink.core.batched.specs`
  provides the class-keyed registry and the `@batched_op` auto-binding decorator.
  `Linear`, `Logistic`, dense `MappingProjection`, and DDM are declarative; LCA
  uses the `triton_emit` escape hatch. Researchers register one module per
  component without touching PNL core classes:

  ```python
  from psyneulink.core.batched import batched_op  # body uses tl (triton.language)

  @batched_op(SoftReLU)
  def soft_relu(x, gain, bias):
      return tl.log(1.0 + tl.exp(gain * (x - bias))) / gain
  ```

- **CPU interpret execution + PsyNeuLink oracle** (`960d3aba6d`). `triton_cpu`
  runs the same kernels on CPU via Triton interpret mode; the `ir_debug` numpy
  executor and per-op CPU twins are gone (one body per op); PsyNeuLink Python
  mode is the test oracle (`tests/composition/pec/test_batched_reference.py`).

- **`integrator_mode` guard + `graph_emit.py` split** (roadmap steps 1 & 3).
  Stateful `integrator_mode=True` transfers are now rejected (no longer silently
  lowered as stateless `Linear`). `graph_emit.py` is now the `emit/` package
  (`emitter`/`lanes`/`ops` mixins); a byte-identical golden-source snapshot test
  (`tests/composition/pec/test_batched_kernel_source.py`,
  `golden_kernels/*.py`) guards emission.

- **Truncation visibility** (roadmap step 2, DDM). Bounded ops declare trailing
  `diagnostics` returns; a `StoreFlag` KernelOp routes them to a per-lane `diag`
  buffer; the runtime reports the truncated fraction per node in
  `result.metadata["truncation"]`, warns by default and raises under
  `strict_truncation=True`. Tested on `triton_cpu`
  (`tests/composition/pec/test_batched_truncation.py`). See "Truncation
  visibility" above for the LCA scoping note. DDM + stateful goldens were
  regenerated for the new `diag` arg / store.

- **`AtPass(0)` scheduling recognition** (roadmap step 4, first increment).
  `graph.py:_condition_schedule_kind` now recognizes `AtPass`: `AtPass(0)` ("fire
  only on pass 0") lowers as `static_graph` — exactly the batched origin default
  (each node computes once per trial) — so the CSI surrogate's four `AtPass(0)`
  origins plus `csiOverride` are accepted instead of rejected. `AtPass(n>0)` (the
  ITI-delayed `taskInput` onset) is recognized but deferred as `precomputed_trace`
  rather than silently mis-timed as static. `schedule_kind` is informational only
  (fusion is driven by node types), so no emit/runtime/golden changes were needed.
  Tested in `test_batched_compile.py` (accept `AtPass(0)` / defer `AtPass(n>0)`)
  and `test_batched_reference.py` (deterministic stab-flex with explicit
  `AtPass(0)` origins still matches PNL Python). With `iti=0`, the CSI model's 5
  `AtPass` scheduler rejections are gone; only the node-level gaps (steps 5-8)
  remain.

- **UDF / instance-level ops** (roadmap step 6). The op registry is now resolvable
  per **node instance**, not just per component class: `specs._INSTANCE_SPECS`
  (keyed by node name) is consulted first in `mechanism_spec_for`, so a node whose
  class already has a class-level spec (e.g. a `ProcessingMechanism` wrapping a
  `UserDefinedFunction`) can be given its own kernel. The new `batched_node_op(
  "<node name>")` decorator captures a `tl` body that receives the node's **whole
  combined input vector** (one positional arg per input component) and may reduce
  it to the node's output — unlike the class-level `batched_op`, which is
  element-wise scalar→scalar. Instance ops reuse the existing `MechanismOpSpec` +
  `triton_emit` machinery (auto-generated from the body), so `kernel_ir`/`emit`
  are unchanged (they dispatch on `spec_kind`/`spec_key`, never node class) and
  the goldens are untouched. This clears the CSI `Drift Rate Value` (nested-logistic
  7→1 reduction) rejection; the two `integrator_mode` transfers still block a full
  CSI compile (step 7). Tested in `test_batched_compile.py` (numeric 2→1 reducer
  on `triton_cpu`; instance-scoped, reversible registration) and
  `test_batched_reference.py` (real CSI drift-rate node leaves the rejected set).
  Binding extra `tl` args to node Parameters/RNG is a future extension
  (input-components-only for now).

- **Fires-once integrating transfers** (roadmap step 7). An integrator_mode
  `TransferMechanism` that resets each trial (`reset_stateful_function_when=
  AtTrialStart`) and fires exactly once per trial (an `AtPass` schedule) advances
  its integrator a single step from its initializer — which is affine in the
  input. `graph.py:_integrating_transfer_affine` returns `(a, b)` for the
  AdaptiveIntegrator (`a=rate, b=(1-rate)*init`) and SimpleIntegrator (`a=rate,
  b=init+offset`); the node then lowers through the **stateless elementwise
  path** with `attrs["integrator_pre"]=(a,b)`, and `emit/ops.py:_emit_function_call`
  prepends `function(a*input + b)`. No lane state, no trial-loop coupling. The
  gate is sound: an integrator that is *not* fires-once (no `AtPass`, e.g. the CSI
  `Threshold Mechanism`, which steps with the DDM) stays rejected. Clears the CSI
  `Task Input`; the per-DDM-step `Threshold Mechanism` is step 8. Existing goldens
  unchanged (no current golden uses an integrator transfer). Tested in
  `test_batched_compile.py` (fires-once accepted / no-schedule rejected) and
  `test_batched_reference.py` (Adaptive rate=0.5 + Logistic matches PNL Python).

- **Collapsing DDM threshold + control routing** (roadmap step 8). The DDM kernel
  boundary is now time-varying: `threshold(step) = threshold + threshold_collapse
  * step` (`threshold_collapse=0` for an ordinary DDM, so its behavior — and the
  non-CSI numerics — are unchanged; the DDM goldens were regenerated for the new
  `threshold_collapse` arg). `components/ddm.py:threshold_override_collapse`
  recognizes a `ControlMechanism` that OVERRIDEs the DDM `threshold` from a
  SimpleIntegrator transfer (`Identity`/`Linear` control fn) and reads its
  per-step `offset` as the collapse rate, bound into the kernel via
  `param(get=ddm_threshold_collapse)`. The driving `Threshold Mechanism` is
  **absorbed**: `graph.py:_absorbed_nodes` drops it (and its `WhenFinished`
  schedule entry) from the lowered graph, since its effect lives entirely in the
  DDM boundary. Instance-op matching is now suffix-insensitive
  (`mechanism_spec_for` strips a `-\d+` rebuild suffix) so a registered op keeps
  matching across in-process model rebuilds. **Completes CSI compilation**
  (`iti=0`): `test_batched_reference.py` asserts the surrogate compiles + runs
  with decision outcomes matching PNL, and that a collapsing threshold shortens
  RT vs a fixed one.

- **Co-evolution loop** (roadmap step 5; **CSI RT parity**). Coupled stateful
  mechanisms now step together in a single fused per-step loop instead of running
  sequentially. New op interface on `MechanismOpSpec`: `step_emit` (one
  integration step), `trial_states` (per-trial reset state, e.g. DDM
  `value`/`steps`/`finished`), `finished_output` + `readout_emit` for a
  *terminator* op. `components/lca.py` and `components/ddm.py` provide step bodies
  (the DDM step carries the collapsing boundary; the LCA step freezes when the
  terminator has finished, so its persisted state carried to the next trial
  matches when the trial actually ended). `graph.py:_is_coevolving` routes a graph
  to `COEVOLVING_GRAPH_FUSION` when an `Always`-scheduled persistent stepper feeds
  a stateful terminator (CSI); cue-terminated LCAs (toy stab-flex) stay
  `stateful_graph`. `emit/ops.py` emits the loop: loop-invariant ops are hoisted
  out once (Triton loop-body vars don't escape), stepper ops run inside, and the
  terminator's `readout_emit` + truncation diag run after. Diagnosis showed the
  batched LCA *recurrence is correct*; the gap was purely that the sequential path
  gave the CSI LCA 0 steps (cue=0) while PNL's `Always` LCA co-evolves with the
  DDM ~1:1. Result: `make_stab_flex(iti=0)` matches PNL on **decision AND RT**
  (within ~1 DDM step), validated on interpret and GPU (`triton`); the toy
  stab-flex stays `stateful_graph` and its golden is byte-identical. New
  `golden_kernels/coevolving_graph.py`; `test_batched_reference.py` asserts RT
  parity + the co-evolving fusion kind.

- **`AtPass(n>0)` ITI onset in the co-evolution loop** (rest of roadmap step 4).
  A delayed within-trial onset (`Task Input` at `AtPass(iti)`) is now executable
  in a co-evolving graph: `graph.py` accepts `AtPass(n>0)` when `coevolving`
  (still deferred otherwise), records `onset_step` on the node and
  `coevolve_warmup=max(onset)` in graph metadata. In the fused loop, the onset
  node's output is gated `where(step >= onset, val, 0)` (so the `Always`-LCA
  integrates 0 and *decays* during the ITI), and the DDM terminator is frozen
  until `step >= warmup` with its collapse step counting from there
  (`_pnl_triton_ddm_step` gained a `start` arg). Onset nodes become loop-variant
  (partition), so they + downstream run inside the loop. `make_stab_flex(iti>0)`
  now matches PNL decision + RT (RT rises with ITI then saturates, as in PNL);
  `csi>0` also matches (its RT contribution is the `cueStimulusInterval ->
  responseGate` term, and the LCA-convergence effect of the extra delay is within
  the ~1-step tolerance). Tested in `test_batched_reference.py` (iti=10 RT parity)
  and `test_batched_compile.py` (AtPass(n>0) accepted when co-evolving, deferred
  otherwise). Coevolving golden regenerated for the DDM step `start` arg.

- **Shared recurrence helpers** (code-quality). The LCA and DDM each had their
  math duplicated (a run-to-completion body and a step body); each now lives in
  one shared `@triton.jit` helper (`_pnl_triton_lca_width2_recurrence`,
  `_pnl_triton_ddm_update`) that both paths call. Enabled by `pnl_triton_op(
  helpers=...)` + `TritonOpTemplate.dependencies` + recursive `register_template`
  (a helper may call other helper templates; they're emitted ahead of the
  caller). Numerically identical (the run-to-completion DDM now threads a
  `finished` flag — same result for a fixed/collapsing boundary, and strictly
  more correct for a growing one); goldens regenerated; verified on GPU.

- **Demand-bounded stateful LCA + per-process benchmark isolation.** The
  `stateful_graph` LCA looped to `lca_max_steps` — the largest cue anywhere in
  the data — with `active = step < lca_steps` masking, so every trial paid the
  worst cue in the dataset. It now computes the block's stopping point once
  before the loop. Note the asymmetry with the DDM: an LCA's settling length is
  cue-driven and known up front, so a *precomputed bound* is right, whereas a
  per-step block reduction (the DDM's form) measured ~24% **slower** than the
  original masking when cues are uniform — the reduction costs more than it
  saves. Varying cues (mean 493 against a cap of 1200): 16.4 -> 11.0 ms; uniform
  cues within noise; checksums identical. This is free in the current benchmark,
  whose cue is a constant 1200 (`generate_trial_sequence` has eight CSI branches
  all hardcoded to `[1200]`, which reads like it was meant to vary by condition
  and was later collapsed) — varying CSIs are what exposes it.
  `gpu_batch_compile_benchmark.py` now runs each (model, case, backend) in its
  own subprocess by default; see "Benchmarking Methodology" for why.

- **Noise-free elementwise accuracy test.** `zero_all_noise(comp)` in
  `tests/composition/pec/test_batched_reference.py` walks a composition and zeros
  every `noise` Parameter (they live on the mechanism, its function, *and* its
  integrator function), making any model deterministic so execution paths can be
  compared elementwise instead of statistically. The CSI surrogate is checked
  across four configurations x Python/LLVM: decisions match **exactly**, and RTs
  differ by **either zero or exactly one DDM step** — a structural assertion, not
  a tolerance, so drift or scaling breaks it immediately. Two limits by
  construction: it exercises none of the RNG machinery, and RT is quantised to
  steps (~50/trial), so errors below ~2% hide inside the one-step allowance (a
  10% drift error fails the test, a 1% one does not).

- **Co-evolving multi-parameter-set fix** (correctness). `_emit_lane_decode`
  special-cased only `STATEFUL_GRAPH_FUSION`, so co-evolving kernels got the 4-D
  (trial-inclusive) lane decode while their runtime sized the launch with the 3-D
  lane-persistent layout. Every lane collapsed onto `param_idx 0`: with N
  parameter sets, only set 0 was computed and sets 1..N-1 came back as
  uninitialized buffer. It hid well — set 0 is correct, the result has the right
  shape, `trial_idx` is reassigned to 0 right after the decode so trials still
  worked, and nothing asserts across parameter sets on this path.
  **Any pre-fix co-evolving number with >1 parameter set is void**, including the
  triton-vs-LLVM CSI speedup and the recorded asv `CSISurrogate` timings and
  checksums (`PARAM_SETS = 8`). `ddm_graph` and `stateful_graph` were verified
  unaffected, as is PEC fit routing (its objective passes one parameter set per
  call). Add a regression test before trusting this path again — the gap that let
  this through is that no test varies parameter sets on a co-evolving model.

- **Cap-independent RNG stream layout** (correctness). Streams are allocated a
  fixed `RNG_STREAM_STRIDE` of Philox counter space instead of being packed by
  `MAX_STEPS`/`LCA_MAX_STEPS` (details under "Stream layout"). Results no longer
  move when a step cap changes — the CSI surrogate now checksums identically at
  caps 256/512/1024, where it previously differed at each — which is what makes
  the early exit's "set the cap generously" advice actually safe. Also fixes the
  co-evolving LCA stream-overlap bug that corrupted a noisy co-evolving LCA.
  Adds `runtime._check_step_caps`. This changes the draws, so stochastic results
  and recorded checksums shift (statistics are unchanged: decision rate, RT
  mean/std and percentiles, and cross-lane independence all match).

- **Bounded-loop early exit** (performance). Both bounded loops stop once every
  in-range lane of the block has finished instead of always running to the cap,
  so runtime tracks decision times rather than `MAX_STEPS` (details, caveats,
  and measurements under "Bounded-loop early exit"). Semantics are unchanged —
  finished lanes were already frozen, so the skipped iterations were no-ops, and
  checksums are identical before/after at every cap. Adds the reserved
  `lane_mask` body argument so a kernel body can exclude out-of-range lanes from
  the exit test. Goldens regenerated for all three affected kernels.

- **Benchmarks** (regression + comparison). `benchmarks/batched.py` gains a
  `CSISurrogate` asv case (the `coevolving_graph` path). `csi_triton_vs_llvm.py`
  compares the co-evolving CSI (triton GPU) vs PNL PEC `grid_evaluate` (LLVM) on
  the same workload — **~100x** steady state at PEC scale (262k sims), triton
  throughput rising with lane count. Finding: the CSI model **cannot fit `threshold` in LLVM PEC**
  (already controlled → `mod_afferents<=1`), so the comparison sweeps
  `non_decision_time`; the batched path also runs a fit config LLVM can't.

- **GPU histogram likelihood** (roadmap step 9). `batched/likelihood.py` provides
  `histogram_likelihood` / `histogram_log_likelihood`: the histogram analogue of
  `fitfunctions.simulation_likelihood` (KDE), computed in Torch so it runs on the
  same device the outcomes were produced on. It is fully vectorized over
  `[*lanes, trial, estimate, outcome]` — for each experimental trial, the density
  is `(# sims matching the observed category AND continuous bin) / (num_sims *
  bin_volume)` (the histogram version of the KDE's per-category-pdf-times-share
  scaling). `BatchedSimulationPlan.log_likelihood(inputs, parameter_sets,
  num_estimates, data, categorical_dims, ...)` simulates and scores in one call,
  returning one total log-likelihood per parameter set; on the `triton` (GPU)
  backend the outcome buffer stays on-device (`run_triton(...,
  keep_device_values=True)`; the per-kernel helpers now return the device tensor
  and `run_triton` does the host copy). Histogram (not KDE) is the intentional
  first cut — the accuracy knob is `bins`; a KDE option can be added later.
  **Bin range** (when not given explicitly) is anchored to the **experimental**
  data, not the simulated data: the density is only evaluated at the experimental
  points, so the bins must cover those, and anchoring to the fixed data keeps the
  bins identical across every parameter set and every `num_estimates` — otherwise
  the bins drift with the simulated spread and inject noise into the MLE objective
  (this showed up as erratic recovery of weakly-identified parameters as
  `num_estimates` changed).
  Tested in `tests/composition/pec/test_batched_likelihood.py` (vs an independent
  numpy histogram; shared bins across lanes; peak at the matching distribution;
  `plan.log_likelihood` recovers a DDM threshold on `triton_cpu`).

- **PEC fit routing** (roadmap step 10). `PECOptimizationFunction` gained an opt-in
  `batched_backend=("triton"|"triton_cpu")` (plus `batched_bins`,
  `batched_max_steps`, `batched_bin_range`, `batched_seed`). When set in
  data-fitting mode, `_make_objective_func` returns a batched objective that
  compiles the model once (cached), feeds the **raw stimulus** as batched inputs,
  supplies the fitting parameters as **parameter sets** (the PEC's dummy
  control mechanisms are absorbed by the compiler as OVERRIDE routing, so
  `fit_param_names` — already `"{mech.name}.{param}"` — are exactly the batched
  parameter keys, no mapping needed), and returns the on-device histogram total
  log-likelihood — the *same quantity* as the default objective, so the optimizer
  / direction handling is unchanged. It never silently falls back: an unsupported
  model raises `OptimizationFunctionError`, and trial-conditional (`depends_on`)
  parameters are rejected. Robustness details: batched inputs are recovered from a
  **node-keyed** stash (`ParameterEstimationComposition._pec_input_values_by_node`,
  saved in `set_pec_inputs_cache` before the inputs are concatenated), and matched
  to the plan's input specs by node name — so models with **absorbed input nodes**
  (the CSI Threshold Mechanism) or extra control-mech inputs line up correctly. The
  data columns are matched to plan outputs by **name** (node + port, tolerating the
  `-N` rebuild suffix), since `outcome_variable_indices` index the *composition's*
  wider output, not the plan's. Tested in
  `tests/composition/pec/test_batched_pec_fit.py` (the objective peaks at the
  data-generating DDM threshold on `triton_cpu`; default path untouched when
  `batched_backend=None`; unsupported model raises). End-to-end **GPU parameter
  recovery** for the CSI surrogate runs via
  `csi_batched_parameter_recovery.py` (see below).

### Roadmap status (original execution order)

The deterministic CSI slice now compiles and runs. The remaining north-star is
the broader **CSI research configuration** and its full PEC fitting workflow.

1. **Reject `integrator_mode` transfers** — DONE (see Done above). Rejected in
   `graph.py:_node_support_diagnostic` with a clear diagnostic instead of being
   silently lowered as stateless `Linear`.

2. **Truncation visibility** — DONE for DDM (see Done above). Bounded ops surface
   a per-lane truncation flag via the `diagnostics`/`StoreFlag`/`diag` channel;
   the runtime warns or (with `strict_truncation`) raises when `max_steps` is too
   low. The channel is node-generic; the co-evolving DDM terminator reuses it.

3. **Split `graph_emit.py`** into a `backend/triton/emit/` package — DONE (see
   Done above). New `KernelOp` emitters are added in `emit/ops.py`.

4. **Tiered scheduling** (pay only when needed) — PARTIAL. Static execution,
   finite precomputed traces, fixed-count stateful producer/follower traces,
   scalar-controlled lane-local counts, and the exact deterministic CSI region
   are executable. `AtPass(n>0)`, additional finished dependencies, and generic
   dynamic consideration-set evaluation remain fail-closed.

5. **Co-evolution loop for coupled stateful mechanisms** — DONE FOR THE EXACT
   DETERMINISTIC CSI SLICE. `lane_local_coevolving` carries explicit LCA/DDM
   finished identities, scheduler ordering, effective controlled count,
   private DDM trial state, persistent LCA state, and persistent threshold
   control. General coupled graphs, delayed ITI, stochastic CSI, width-N LCA,
   and arbitrary recurrent matrices remain separate open work.

6. **UDF / instance-level ops** (gap 1) — DONE (see Done above). Researchers
   register an op for a specific node via `batched_node_op("<node name>")`; the
   `tl` body receives the node's whole combined input vector and may reduce it.
   Clears the CSI `Drift Rate Value` rejection.

7. **Fires-once integrating transfers** (gap 2) — DONE (see Done above). An
   integrator_mode transfer that resets each trial (`AtTrialStart`) and fires
   once per trial (`AtPass`) advances its integrator a single affine step from
   its initializer, so it lowers *statelessly* as `function(a*input + b)` — no
   lane state needed. Clears the CSI `Task Input`. *Remaining for a multi-step
   integrating transfer* (one that accumulates within a trial, e.g. the CSI
   `Threshold Mechanism`, which steps with the DDM): folded into step 8.

8. **Time-varying DDM boundary + control routing** (gaps 4-5) — DONE (see Done
   above). The DDM kernel boundary is `threshold(step)=threshold+collapse*step`;
   a control mechanism OVERRIDE'ing the DDM `threshold` from a SimpleIntegrator
   transfer is recognized and that transfer is absorbed into the DDM. Completes
   CSI compilation (`iti=0`).

9. **GPU histogram likelihood** — DONE (see Done above). `batched/likelihood.py`
   scores experimental data with a Torch **histogram** density estimate (not KDE)
   that runs on the same device the outcomes were produced on;
   `plan.log_likelihood(...)` keeps GPU outcomes on-device (no host round-trip).

10. **PEC fit routing** — DONE (see Done above). `PECOptimizationFunction(
    batched_backend=...)` routes the data-fitting objective through the batched
    plan + histogram likelihood, guarded by the batched compiler; unsupported
    models raise (never silently fall back). *Remaining:* a KDE option (histogram
    only for now) and generalizing input reconstruction beyond
    node-qualified/absorbed-control models.

### Design invariants (do not regress)

- **KernelIR backend neutrality.** No `tl.*`, source fragments, or Triton-only
  expressions in `KernelIR`; a future MLIR backend should lower from it.
- **No silent fallback.** Unsupported components/conditions return diagnostics;
  never silently fall back to Python or LLVM inside this stack.
- **Honest benchmarks.** Compare against PEC `grid_evaluate` LLVM/PTX baselines,
  not repeated `Composition.run()` loops.

## Known Sharp Edges

- Emission now lives in the `emit/` package (`emitter`/`lanes`/`ops` mixins);
  the byte-identical `golden_kernels` snapshots must be regenerated
  (`PNL_UPDATE_KERNEL_GOLDENS=1`) on any *intentional* emission change.
- `triton_cpu` (interpret mode) is for testing/debugging, not performance; it
  runs the real kernel but pure-Python/numpy interpretation is slow, so keep
  CPU cases small.
- `triton_cpu` (interpret) and `triton` (compiled GPU) cannot be used in the
  same process; triton bakes the mode when `tl.randn` et al. are first imported.
- Triton execution imports Torch/Triton lazily; the GPU path requires CUDA.
- Numeric tests need `torch`+`triton` importable and skip otherwise.
- The generated kernels use float32.
- Parameter aliases are intentionally generic. Avoid adding model-specific
  aliases to make one example script more convenient.
- Stability-flexibility examples depend on helper builders in
  `tests/composition/pec/test_stab_flex_pec_fit.py`; the **CSI surrogate** lives
  in `Scripts/Debug/pec_batch_compile/csi_model_surrogate.py` and needs its
  drift-rate op registered (`batched_node_op("Drift Rate Value")`) before it
  compiles.
- Bounded loops now **exit early per lane block** (see "Bounded-loop early
  exit"), so `MAX_STEPS` no longer scales runtime — but the exit is per *block*,
  so a block costs its slowest lane. A single pathological lane still holds its
  block (127 others) open to the cap.
- RNG streams are addressed by a fixed stride, not by the step caps (see "Stream
  layout"). Do not reintroduce a cap into an RNG offset: it makes results depend
  on a bound that is supposed to be a don't-care, and it can let a stream run
  into its neighbour.
- Benchmark numbers on this stack have been wrong by 2.5x to 30x while looking
  perfectly credible. Read "Benchmarking Methodology" before quoting one, and
  distrust any figure that does not say whether it is warm or cold, how many
  lanes it ran, and whether the case had the process to itself.
- `csi_triton_vs_llvm.py` reports timings honestly (compile / warm / cold), but
  its **checksum** still sums decisions and response times into one scalar, so
  errors in one can cancel the other. It is a smoke test. The componentwise
  comparison is in "Benchmark (vs LLVM PEC)", and the real accuracy check is the
  noise-free elementwise test.
- Stochastic agreement with LLVM is capped by LLVM itself: its PEC grid results
  are not reproducible run to run despite a fixed `initial_seed`, while the
  batched path is bit-identical. Use the noise-free elementwise test for
  accuracy, not checksums.

## Current Mental Model

This branch should be treated as a prototype for a high-throughput batched
stochastic simulator, not a replacement for the existing PsyNeuLink compiler.

The right division of labor is:

- LLVM/PTX: broad compiled PsyNeuLink semantics.
- Triton batched compiler: narrow, diagnosed, high-throughput static/stochastic
  PEC simulation subset.
- Future MLIR backend: possible, but it should start from `KernelIR`, not from
  Triton source generation.
