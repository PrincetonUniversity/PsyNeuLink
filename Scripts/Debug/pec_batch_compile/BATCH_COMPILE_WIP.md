# PEC Batched Compile WIP Handoff

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
```

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

- Unit tests (`tests/composition/pec/test_batched_reference.py`) compare the
  batched backend against real PNL **Python-mode** execution — deterministic
  (noise=0) cases exactly, stochastic cases by summary statistics.
- PEC-scale / GPU-compiled validation lives in
  `pec_grid_correctness_check.py`, which uses PNL **LLVM** grid evaluation as
  the reference (separate process from interpret-mode tests).

`ParameterEstimationComposition.can_compile_batched(..., backend="triton")` is
diagnostic-only. It calls `BatchedCompositionCompiler.diagnose()` on the model
composition and does not route PEC fitting through Triton.

`BatchedSimulationResult.values` has shape:

```text
[parameter_set, subject, trial, estimate, outcome]
```

## Current Commit Stack

The important commits on this branch are:

- `99b2313fc0` - initial opt-in Triton batched simulator for PEC models.
- `667963121b` - generated graph compiler for DDM and stateful graph cases.
- `0b8547955a` - KernelIR refactor and move Triton backend into
  `backend/triton`.
- `640c690a55` - component-owned Triton hooks with `@pnl_triton_op`.
- `4fd3be6718` - remove stability-flexibility-specific compiler/runtime paths.
- `653a57cbf9` - declarative batched op specs: replace the `_gen_triton_*`
  monkeypatched hooks with a class-keyed spec registry and the
  `@batched_op` auto-binding decorator; retire the monolithic DDM kernel.
- `960d3aba6d` - CPU interpret execution + PsyNeuLink oracle (see below).

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
    Reserved names: `x`, `seed`/`rng_base`, `max_steps`.  Irregular bindings
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

- `psyneulink/core/batched/kernel_ir.py`
  - Backend-neutral execution-level IR.
  - Converts `BatchedGraphIR` into structured execution ops:
    `LoadInput`, `CallProjection`, `CombineSum`, `CombineProduct`,
    `CallFunction`, `CallMechanism`, `StoreOutput`, and stateful `ForTrials`.
  - Dispatch is driven by `spec_kind`/`spec_key`/`rng_streams`/`op_outputs`
    node attrs written at lowering time; kernel_ir does not import the
    registry.
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
  - Dispatches to stateless graph, DDM graph, or stateful graph kernels.
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
  - `ops.py` (`OpEmitMixin`): per-`KernelOp` emission + value table. New op kinds
    (e.g. truncation `StoreFlag`, scheduling variants) are added here.
  - Spec-driven op emission (declarative elementwise/mechanism calls plus
    `triton_emit` escape hatches) resolved through `spec_key`.

- `api.py`
  - Defines `@pnl_triton_op`, `TritonOpTemplate`, `TritonOpCall`,
    `TritonEmitContext`, and `TritonOpError`.
  - `@pnl_triton_op` captures inspectable Python source and emits it as a
    `@triton.jit` helper without importing Triton at PsyNeuLink import time.
  - Helpers may reference `tl` (`triton.language`); other globals/closures are
    rejected.
  - `backend/triton/__init__.py` imports `runtime` lazily so importing the
    spec/registration API does not create an import cycle.

- `cache.py`
  - Writes generated source to a real Python module path and imports it. This
    is necessary because Triton JIT requires inspectable source.

- `source_builder.py`
  - Small indentation/source helper for Triton emission.

## Current Supported Subset

Supported mechanisms:

- `TransferMechanism`
- `ProcessingMechanism`
- `DDM`
- `LCAMechanism`, width 2 only

Supported functions:

- `Linear`
- `Logistic`
- `DriftDiffusionIntegrator` for DDM

Supported projections:

- dense `MappingProjection`

Supported input combines:

- single input
- `SUM`
- `PRODUCT`

Supported executable schedule kind:

- `static_graph`

Recognized but not executable yet:

- `precomputed_trace`
- `dynamic_lane_local`

Unsupported scheduler conditions should return diagnostics. Do not silently
fall back to Python or LLVM inside this stack.

## Capability Gaps — Real Research Models

The stability-flexibility model in the test suite is a *toy*. The model a
researcher actually runs is the **CSI surrogate** stability-flexibility model in
`csi_model_surrogate.py` (fit at PEC scale by `csi_fit_parameter_recovery.py`:
128 trials x 10k estimates, CMA-ES over 5 parameters). It is the north-star
target for "supporting researchers' models," and the roadmap in "Good Next
Steps" is scoped to close the gap to it.

`BatchedCompositionCompiler.diagnose()` now **accepts** the CSI surrogate (with
`iti=0`) once its `driftRate` op is registered: `make_stab_flex(iti=0)` is
`is_supported`, compiles, and runs (decision outcomes match PNL Python;
see "CSI status" below). The original gaps and how each was closed:

1. **Custom / UDF reduction op** (`driftRate`, a nested-logistic reducing a
   7-vector to a scalar) — RESOLVED (step 6). `batched_node_op("<node name>")`
   registers an instance-level op whose `tl` body takes the node's whole input
   vector. (A cross-element reduction, so it rides the mechanism/`triton_emit`
   path, not the scalar→scalar elementwise path.)
2. **Stateful integrating `TransferMechanism`** (`taskInput`) — RESOLVED (step 7).
   A fires-once (`AtPass`), reset-each-trial (`AtTrialStart`) integrator advances
   one affine step from its initializer and lowers statelessly as
   `function(a*input + b)`.
3. **`AtPass` multi-pass timing** — RESOLVED (step 4, `iti=0`). `AtPass(0)` lowers
   as a static graph; `AtPass(n>0)` (the ITI onset) is still deferred.
4. **Collapsing DDM threshold** (`threshold_collapse`) — RESOLVED (step 8). The
   DDM kernel boundary is now `threshold(step) = threshold + collapse*step`.
5. **Control routing** (`csiOverride`/`thresholdOverride` OVERRIDE a target
   param) — RESOLVED. `csiOverride` -> LCA `termination_threshold` was already
   handled by the LCA cue extraction; `thresholdOverride` -> DDM `threshold`
   (monitoring the `thresholdMechanism` SimpleIntegrator) is recognized in step 8
   and the `thresholdMechanism` is **absorbed** into the DDM boundary (not lowered
   as its own op).

### CSI status

`make_stab_flex(iti=0)` compiles and runs end-to-end on `triton_cpu`/`triton`,
with **both decision outcomes and response times matching** PNL Python mode
(RT within ~1 DDM step). This required the **co-evolution loop** (see below): the
CSI LCA is `Always`-scheduled and its activation feeds the drift, so the LCA and
DDM step together each timestep rather than running sequentially. Remaining for
the full PEC fit: `iti>0` (`AtPass(n>0)`, rest of step 4), GPU likelihood/KDE
(step 9), and PEC fit routing (step 10).

**Benchmark (vs LLVM PEC).** `csi_triton_vs_llvm.py` compares the co-evolving
CSI on the `triton` GPU path against PNL's PEC `grid_evaluate` LLVM baseline on
the same workload (sweeping `non_decision_time` — the DDM `threshold` is already
controlled, so PEC cannot also modulate it; that conflict raises
`len(mod_afferents) <= 1` in LLVM). At PEC scale (4 params x 512 estimates x 128
trials = 262k sims, RTX 2080 Ti): **triton ~57 ms (4.6M sims/s) vs LLVM ~3.5 s
(75k sims/s) -> ~62x**, and triton throughput rises with lane count. Note the
checksum vs LLVM diverges mainly because PNL's **own LLVM mode disagrees with its
Python mode** on the fresh-LCA first trial (RT 1.23 vs 0.53); the batched path
matches PNL **Python** (the test-suite reference) trial-for-trial. asv tracks the
co-evolving path via the `CSISurrogate` benchmark in `benchmarks/batched.py`.

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
    to completion (e.g. a cue-terminated LCA settles, then the DDM decides — the
    toy stab-flex model).
  - Lane layout: `(parameter_set, subject, estimate)`.
  - Trials run inside the Triton lane so LCA state persists across trials.

- `coevolving_graph`
  - Coupled stateful ops that **step together** in a single fused per-step loop
    (the CSI surrogate: an `Always`-scheduled LCA whose activation feeds the
    drift, co-evolving with the DDM until the DDM crosses its boundary).
  - Lane layout: `(parameter_set, subject, estimate)` (same as `stateful_graph`).
  - Each step, in topological order: loop-invariant ops are hoisted out once;
    stepper ops (LCA `step_emit`, DDM `step_emit`) advance one step; lanes whose
    terminator has `finished` freeze; the terminator's outputs are produced by a
    `readout_emit` after the loop. See "Co-evolution loop" below.

Fusion kind is a dispatch/optimization detail. It should not encode model
architecture semantics.

## Stability-Flexibility Status

Stability-flexibility is now treated as:

```text
model_kind="graph"
fusion_kind="stateful_graph"
schedule_kind="static_graph"
```

There should be no `STABILITY_FLEXIBILITY_MODEL`, no
`stability_flexibility_roles` metadata, no forced `cue`/`correct` aliases, and
no monolithic SF Triton kernel in the normal runtime path.

Parameter sets should use generic node-qualified names, for example:

```python
{
    "DDM.threshold": 0.05,
    "DDM.noise": 0.0,
    "Task Activations [Act1, Act2].noise": 0.0,
}
```

Old SF-only names such as `threshold`, `ddm_noise`, `lca_noise`,
`automaticity`, or `scale` should not be reintroduced as special compiler
aliases. If users need friendlier names later, that should be a general
parameter naming/alias policy, not a model-specific shortcut.

## LCA Caveats

The current batched LCA hook is a narrow approximation for width-2 LCA-like
stateful graphs. It is not a full implementation of PsyNeuLink
`LCAMechanism`.

The real PsyNeuLink `LCAMechanism` is implemented through:

```text
LCAMechanism
  -> RecurrentTransferMechanism
      -> TransferMechanism
          -> LeakyCompetingIntegrator + Logistic
```

The batched hook currently handles:

- width 2 only;
- persistent lane-local `pre` state;
- persistent lane-local `act` state;
- recurrent coupling via scalar `self_excitation` and `competition`;
- Logistic activation;
- Gaussian noise through Triton `tl.randn`;
- a cue/termination input lowered to an LCA step count.

It does not yet cover:

- arbitrary width;
- arbitrary recurrent matrices;
- separate recurrent input ports;
- generic `combination_function`;
- learning;
- full output-port variants;
- full `execute_until_finished` / `WhenFinished` behavior;
- full controller semantics.

Be careful not to describe the batched LCA hook as the authoritative
PsyNeuLink LCA semantics. It is currently a performance-oriented lowering for a
small supported subset.

## RNG and Common Random Numbers

The Triton backend uses `tl.randn` in component helpers. RNG offsets are derived
from lane indices and stream layout.

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
state. Parity tolerances should account for this.

DDM and LCA integration use bounded loops:

- `MAX_STEPS` for DDM.
- `LCA_MAX_STEPS` for LCA.

The existing LLVM/PsyNeuLink path uses mechanism/scheduler termination
machinery. The Triton path uses caps and must be given caps large enough for
the parameter/input regime.

**Truncation visibility (roadmap step 2 — DONE for DDM).** A bounded op may
declare trailing diagnostic returns via `MechanismOpSpec.diagnostics` (the DDM
body returns a `truncated` flag — still inside the boundary after `max_steps`).
The lowering emits a `StoreFlag` KernelOp per diagnostic into a separate per-lane
`diag` buffer (only present when a kernel has diagnostics, so the stateless
golden is unchanged); the runtime aggregates the truncated fraction per node into
`result.metadata["truncation"]`, **warns** by default and **raises**
`BatchedTruncationError` under `run(..., strict_truncation=True)`. The channel is
node-generic: when threshold-terminated LCA lands (step 5) its body returns the
same `truncated` flag and reuses this path unchanged.

Note the *cue-driven* LCA (stab-flex / CSI) cannot truncate today: `LCA_MAX_STEPS`
is sized from the cue data (`prep.lca_max_steps` = `max(metadata, ceil(cue))`), so
the cap is always ≥ demand. Data-dependent LCA truncation only appears with the
general threshold-terminated LCA, which is not yet implemented.

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
- does not yet compute the full PEC likelihood/KDE on GPU;
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
```

(`triton` and `triton_cpu` cannot be combined in one invocation.)

The one-off script above gives ad-hoc triton-vs-LLVM numbers. To **track**
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

### Planned (in execution order; scoped to close the Capability Gaps above)

The end goal is the **CSI surrogate model**: steps 1-8 make it *compilable*;
steps 9-10 make the full PEC fit run on the batched path.

1. **Reject `integrator_mode` transfers** — DONE (see Done above). Rejected in
   `graph.py:_node_support_diagnostic` with a clear diagnostic instead of being
   silently lowered as stateless `Linear`.

2. **Truncation visibility** — DONE for DDM (see Done above). Bounded ops surface
   a per-lane truncation flag via the `diagnostics`/`StoreFlag`/`diag` channel;
   the runtime warns or (with `strict_truncation`) raises when `max_steps` is too
   low. The channel is node-generic; threshold-terminated LCA reuses it at step 5.

3. **Split `graph_emit.py`** into a `backend/triton/emit/` package — DONE (see
   Done above). New `KernelOp` emitters are added in `emit/ops.py`.

4. **Tiered scheduling** (pay only when needed) — PARTIALLY DONE. Recognition of
   the `AtPass(0)` origins / `Always` LCA / `WhenFinished(LCA)` idiom (gap 3) is
   done (see Done above): the CSI/ITI idiom with `iti=0` now lowers to the
   existing cue-driven stateful graph. **Remaining:** precomputed per-trial traces
   for `EveryNCalls`-style conditions, and the `AtPass(n>0)` ITI-delayed onset
   (currently deferred as `precomputed_trace`). Prefer static erasure / precomputed
   traces over dynamic lane-local scheduler state (which would recreate the
   LLVM/PTX overhead problem).

5. **Co-evolution loop for coupled stateful mechanisms** — DONE (see Done above).
   This was the actual blocker to CSI RT parity (not LCA *width*): the diagnosis
   showed the batched LCA recurrence is correct, but the CSI LCA is
   `Always`-scheduled and co-evolves with the DDM (~1:1), while the sequential
   `stateful_graph` ran the LCA to a cue-driven step count (0 for CSI). The fused
   co-evolution loop closes that. *Still open under "generalize LCA":* width-N /
   arbitrary recurrent matrix, and dropping the init-`act=0` approximation for an
   isolated LCA — neither is needed for CSI.

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

9. **GPU likelihood/KDE.** Torch port of `fitfunctions.simulation_likelihood` so
   PEC objective aggregation runs near the GPU execution path (the current
   simulator only returns outcomes).

10. **PEC fit routing.** Route the PEC objective through `BatchedSimulationPlan`,
    guarded by `can_compile_batched`; never silently fall back.

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
  `tests/composition/pec/test_stab_flex_pec_fit.py`.

## Current Mental Model

This branch should be treated as a prototype for a high-throughput batched
stochastic simulator, not a replacement for the existing PsyNeuLink compiler.

The right division of labor is:

- LLVM/PTX: broad compiled PsyNeuLink semantics.
- Triton batched compiler: narrow, diagnosed, high-throughput static/stochastic
  PEC simulation subset.
- Future MLIR backend: possible, but it should start from `KernelIR`, not from
  Triton source generation.
