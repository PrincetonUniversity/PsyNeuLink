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

report = BatchedCompositionCompiler.diagnose(comp, backend="ir_debug")
plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=256)
result = plan.run(inputs, parameter_sets, num_estimates, seed=11)
```

Supported backends:

- `ir_debug`: CPU debug executor. This is the default.
- `triton`: CUDA/Triton executor.

The old prototype backend name `reference` is intentionally rejected. It is not
an alias for `ir_debug`.

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

After `4fd3be6718`, stability-flexibility is no longer a special model family
in the batched compiler. It is just an example of a supported static stateful
graph.

## Code Map

### Root Batched Package

- `psyneulink/core/batched/compiler.py`
  - Public compiler facade.
  - Defines `BatchedCompositionCompiler`, `BatchedSimulationPlan`, and
    `BatchedCompileError`.
  - Dispatches plan execution to `ir_debug` or `backend.triton.run_triton`.

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

- `psyneulink/core/batched/kernel_ir.py`
  - Backend-neutral execution-level IR.
  - Converts `BatchedGraphIR` into structured execution ops:
    `LoadInput`, `CallProjection`, `CombineSum`, `CombineProduct`,
    `CallFunction`, `CallMechanism`, `StoreOutput`, and stateful `ForTrials`.
  - The intent is that a future MLIR backend starts here, not from generated
    Triton source.

- `psyneulink/core/batched/ir_debug.py`
  - CPU debug executor for the same lowered semantics.
  - Also owns common input and parameter normalization helpers used by Triton.
  - This is not a performance backend; it is for debugging and deterministic
    parity against generated kernels.

- `psyneulink/core/batched/bindings.py`
  - Sidecar live-object map from graph specs back to the actual PsyNeuLink
    components.
  - Keeps live Python objects out of `BatchedGraphIR` and `KernelIR`, while
    letting the Triton backend call component-owned hooks.

- `psyneulink/core/batched/registry.py`
  - Ties lowering, backend availability, and capability reporting together.
  - Checks optional `torch`/`triton` availability for the Triton backend.

### Triton Backend

All Triton-specific code should stay under:

```text
psyneulink/core/batched/backend/triton/
```

- `runtime.py`
  - Imports Torch/Triton lazily.
  - Checks CUDA availability.
  - Prepares Torch buffers.
  - Dispatches to DDM, stateless graph, DDM graph, or stateful graph kernels.
  - The normal stability-flexibility monolithic kernel path has been removed.

- `graph_emit.py`
  - Emits inspectable Triton Python source from `KernelIR`.
  - Owns lane decode, parameter loads, state initialization, trial loops, RNG
    base layout, output stores, and calls into component hooks.
  - This file is still more complex than ideal and is a major future refactor
    target.

- `component_hooks.py`
  - Installs private `_gen_triton_*` hooks on supported component classes.
  - Current hooks cover:
    - `Linear`
    - `Logistic`
    - dense `MappingProjection`
    - `DDM`
    - width-2 `LCAMechanism`
  - The helpers are decorated with `@pnl_triton_op`.

- `api.py`
  - Defines `@pnl_triton_op`, `TritonOpTemplate`, `TritonOpCall`,
    `TritonEmitContext`, and `TritonOpError`.
  - `@pnl_triton_op` captures inspectable Python source and emits it as a
    `@triton.jit` helper without importing Triton at PsyNeuLink import time.
  - Helpers may reference `tl`; other globals/closures are rejected.

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

## Fusion and Lane Layout

Current fusion kinds:

- `ddm`
  - Single DDM composition.
  - Uses the monolithic DDM fast path.

- `stateless_graph`
  - Transfer/Processing-only static graph.
  - Lane layout: `(parameter_set, subject, trial, estimate)`.

- `ddm_graph`
  - Static graph with stateless nodes feeding one DDM.
  - Lane layout: `(parameter_set, subject, trial, estimate)`.

- `stateful_graph`
  - Static graph with lane-local state, currently mainly LCA plus DDM/gates.
  - Lane layout: `(parameter_set, subject, estimate)`.
  - Trials run inside the Triton lane so LCA state persists across trials.

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
the parameter/input regime. A future pass should report truncation or
non-termination explicitly instead of silently returning capped trajectories.

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
.venv/bin/python -m compileall -q psyneulink/core/batched tests/composition/pec/test_batched_compile.py Scripts/Debug/pec_batch_compile
.venv/bin/python -m pytest tests/composition/pec/test_batched_compile.py -q -n 0
```

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
--backend ir_debug
--backend triton
--backend llvm
--backend ptx
--ddm-case PARAMSxTRIALSxESTIMATES
--ddm-graph-case PARAMSxTRIALSxESTIMATES
--sf-case PARAMSxTRIALSxESTIMATES
```

## Good Next Steps

1. Add declarative Triton component specs.

   `@pnl_triton_op` solved source extraction, but `_gen_triton_*` still repeats
   binding boilerplate. Add internal specs such as:

   ```text
   TritonElementwiseFunctionSpec
   TritonDenseProjectionSpec
   TritonMechanismCallSpec
   TritonStatefulMechanismSpec
   ```

   Convert `Linear`, `Logistic`, and dense `MappingProjection` first. Convert
   DDM next. Leave LCA custom until the state/RNG/control spec is proven.

2. Make truncation visible.

   DDM/LCA helpers should be able to return or store a termination/truncation
   flag. Tests and diagnostics should fail or warn when `MAX_STEPS` is too low.

3. Generalize LCA carefully.

   Move from width-2 scalar recurrence to matrix/vector recurrence. Keep the
   current width-2 path as a specialized lowering only after a generic semantic
   representation exists.

4. Implement precomputed scheduler traces.

   Some scheduler cases such as `EveryNCalls` can likely be represented as a
   precomputed per-trial/per-node execution trace without dynamic scheduler
   state inside the GPU kernel.

5. Avoid dynamic scheduler machinery unless needed.

   Full lane-local scheduler state may recreate the overhead problems of the
   LLVM/PTX path. Prefer static erasure or precomputed traces for PEC-style
   workloads.

6. Move more PEC objective work onto GPU.

   The current batched simulator returns simulated outcomes. Full PEC speedups
   will likely require moving likelihood/KDE or summary aggregation closer to
   the GPU execution path.

7. Preserve KernelIR backend neutrality.

   Do not put `tl.*`, source fragments, or Triton-only expressions into
   `KernelIR`. If an MLIR backend is added later, it should lower from
   `KernelIR`.

8. Keep benchmark baselines honest.

   Use PEC `grid_evaluate` LLVM/PTX baselines for comparison, not repeated
   `run()` loops.

## Known Sharp Edges

- `graph_emit.py` is still hard to maintain. The source builder is better than
  the initial monolithic generator, but op emission should be split further
  once declarative component specs exist.
- `ir_debug` is useful for parity, not performance.
- Triton execution requires CUDA and imports Torch/Triton lazily.
- Current tests may skip Triton on non-CUDA machines.
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
