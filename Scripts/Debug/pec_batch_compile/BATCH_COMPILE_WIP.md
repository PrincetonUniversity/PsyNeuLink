# Batched Compiler Engineering Roadmap

This is a developer roadmap for the experimental PEC batched compiler. The
authoritative public contract, support table, failure behavior, and API live in
`docs/source/BatchedCompilation.rst`. Executable tests are the semantic
authority. Historical measurements belong in `benchmarks/README.md`; this file
must not accumulate benchmark transcripts or completed milestone diaries.

## Current Checkpoint

The compiler evaluates independent simulation lanes over:

```text
parameter set x subject x estimate
```

Trials execute inside a lane when state must persist between trials. The public
entry point is `psyneulink.core.batched.BatchedCompositionCompiler`; supported
models lower through GraphIR and backend-neutral KernelIR before Triton source
is emitted. Unsupported semantics are diagnosed and fail closed. There is no
silent fallback to Python, LLVM, or a hand-written model implementation.

The exact CSI research-model boundary currently compiles and runs. It covers:

- affine cue repeat/switch timing and fixed integral delayed ITI;
- persistent LCA state and finite numeric LCA noise;
- stochastic DDM execution with component-local, lane-local random streams;
- folded, persistent threshold control;
- runtime parameter lanes used by the three- and five-parameter PEC fits; and
- the two `WhenFinished(DDM)` output gates.

This is an authenticated specialization, not generic co-evolving scheduling.
The accepted CSI graph has an exact 11-node, six-consideration-set structure,
one registered drift UDF, one controlled LCA, and one terminating DDM. Changes
outside that boundary remain declaration-only or are rejected.

## Goal

Replace the CSI-specific path with a generic lane-local consideration-set
executor. CSI must remain a demanding regression fixture, but adding another
topology composed from supported mechanisms and predicates must not require a
new topology predicate, KernelIR builder, validator, or emitter loop.

The compiler should continue to be narrower than general PsyNeuLink execution.
Generality here means compositional support for explicitly represented
scheduler semantics, not accepting behavior that cannot be authenticated and
lowered exactly.

## Generic Co-Evolving Architecture

### Acceptance corpus

The generic path must cover these topology classes before the CSI
specialization is deleted:

1. Two independent stateful chains in one trial.
2. Branching and fan-in `WhenFinished` dependencies.
3. Multiple finished predicates and a distinct trial terminator.
4. Delayed `AtPass` execution and termination between consideration sets.
5. Multiple stochastic mechanisms with component-local RNG clocks.
6. CSI, including deterministic, stochastic, three-parameter, and
   five-parameter PEC acceptance.

Each topology fixture must lower without a production-code change after its
mechanisms and predicates are registered.

### Existing semantic foundation

GraphIR already represents most of the required scheduler contract:

- `BatchedSchedulerSpec` and ordered `BatchedConsiderationSetSpec` values;
- typed conditions, dependency component IDs, and finished-value IDs;
- run/trial termination declarations;
- mechanism state and reset declarations;
- effective parameters and modulation routes; and
- RNG/state metadata owned by the relevant component.

The next step should reuse and strengthen these declarations rather than add a
parallel scheduler model.

### KernelIR shape

Introduce one typed dynamic schedule carried by a
`ForPasses(trace_kind="lane_local_dynamic")` region. Its declaration should
contain:

- the ordered consideration sets;
- for each member, its component ID, typed condition, body operations, declared
  outputs, and post-execution effects;
- explicit loop-carried mechanism state, held effective/control values, current
  output bank, and diagnostics;
- typed scheduler slots for pass index, per-component execution count,
  `has_run`, finished values, and dependency credits; and
- component-local RNG clocks, advanced only when their owner executes.

The region must publish only declared results. Nested values may not escape
through emitter scope, and effectful state may not be reconstructed from source
order.

### Execution semantics

For each consideration set:

1. Check trial termination at the boundary before the set.
2. Snapshot condition inputs, counters, finished values, and the output bank.
3. Evaluate member predicates and produce lane masks from that frozen snapshot.
4. Execute eligible member bodies under their masks.
5. After the whole set, publish outputs and effects, update execution counts and
   `has_run`, spend and publish dependency credits, and refresh finished values.

Termination is observable between consideration sets, never between members of
one frozen set. This ordering is required for CSI's threshold-control cleanup
and for general `AllHaveRun` behavior.

Random draws must be addressed by `(seed, lane identity, component identity,
component-local execution count)`. Unrelated onset, divergence, or a safety cap
must not change another component's stream.

### Validation

Validation should be compositional:

- validate each condition and its referenced scheduler slots;
- validate each component body against its registered immutable spec;
- validate state, output, control, and RNG ownership independently;
- validate consideration-set ordering and frozen-input semantics;
- validate explicit region arguments, yields, effects, and diagnostics; and
- prove every executable GraphIR declaration has a corresponding KernelIR
  operation and vice versa.

Whole-topology node counts, CSI role names, and topology-shaped equality checks
must not be capability authorities in the generic path.

## Migration Sequence

### 1. Create the seam without changing execution

- Extract shared typed condition validation/evaluation from `schedule.py` so
  precomputed traces and dynamic execution use the same predicate semantics.
- Split `_trial_body_ops` into a per-component body builder plus an ordering
  layer; keep current precomputed lowering behavior unchanged.
- Add the typed dynamic-schedule KernelIR declarations and validators while the
  existing `lane_local_counted` and `lane_local_coevolving` emitters remain in
  service.
- Add declaration/lowering tests for the acceptance corpus.

### 2. Implement the generic region

- Lower ordered GraphIR consideration sets into the dynamic schedule.
- Allocate and carry scheduler slots explicitly.
- Emit masked member bodies and publish effects only at set boundaries.
- Add component-local RNG clocks and explicit state/output yields.
- Compare scheduler traces as well as outputs against fresh Python executions.

### 3. Migrate existing dynamic paths

- Move the controlled-finished producer/follower topology to the generic
  region.
- Move CSI without adding a CSI branch to the generic builder or emitter.
- Keep old and new paths temporarily selectable in tests until deterministic,
  stochastic, reset, truncation, and PEC parity are established.

### 4. Delete specialization

After parity, remove:

- `graph.py:_dynamic_controlled_coevolving_graph_eligible`;
- the CSI-specific canonical KernelIR builder and validator;
- `emit/ops.py:_emit_coevolving_trial_loop` and its partition/dispatch helpers;
- special `lane_local_coevolving` branches in modulation and pass-region
  validation; and
- redundant topology-specific tests that are covered by generic structural
  invariants plus end-to-end CSI acceptance.

Only then should the compiler claim support for arbitrary co-evolving graphs
within its registered mechanism and predicate subset.

## Deletion Gate

The old path can be removed when all of the following are true:

- no topology names or exact node counts occur in generic admission;
- a new supported topology requires fixture code only;
- Python scheduler traces match per lane, including same-set freezing,
  mid-pass termination, resets, persistent effects, and truncation;
- stochastic replay, common-random-number behavior, and stream independence
  hold across onset and cap changes;
- CSI interpreter and physical-GPU suites pass through the generic executor;
- both real CSI PEC objective surfaces compile and score through it; and
- generated source and KernelIR validation reject all tested direct-IR
  forgeries.

## Code Map

- `psyneulink/core/batched/graph.py`: Composition snapshot, capability
  diagnosis, scheduler/control/state declarations, and GraphIR construction.
- `psyneulink/core/batched/schedule.py`: typed scheduler predicates and static
  trace planning.
- `psyneulink/core/batched/ir.py`: GraphIR declarations.
- `psyneulink/core/batched/kernel_ir.py`: backend-neutral executable operations,
  lowering, and complete validation.
- `psyneulink/core/batched/components/`: registered component specifications and
  step semantics.
- `psyneulink/core/batched/backend/triton/emit/`: KernelIR-to-Triton emission.
- `psyneulink/core/batched/backend/triton/runtime.py`: launch, buffers, and
  result/diagnostic handling.
- `psyneulink/core/batched/prep.py`: input/parameter lane preparation and host
  validation.
- `psyneulink/core/batched/likelihood.py`: on-device histogram likelihood.
- `docs/source/BatchedCompilation.rst`: current user/developer contract.

Primary acceptance suites:

- `test_batched_scheduler_ir.py` and `test_batched_schedule_execution.py`;
- `test_batched_dynamic_control_kernel_ir.py`;
- `test_batched_controlled_finished_acceptance.py`;
- `test_batched_coevolving_graph_admission.py`;
- `test_batched_csi_coevolving_acceptance.py`;
- `test_batched_fail_closed_validation.py`; and
- `test_batched_lca_numeric_noise.py`.

## Design Invariants

- KernelIR remains backend-neutral; it contains no Triton expressions.
- Unsupported semantics fail closed with actionable diagnostics.
- GraphIR and KernelIR are immutable snapshots; backend validation does not
  trust mutable Composition objects.
- Dynamic regions have explicit inputs, outputs, state, and effects.
- Consideration-set input freezing and termination boundaries match the Python
  scheduler.
- RNG streams are component-local and independent of execution caps.
- Stateful values persist or reset only according to typed declarations.
- Generated kernels use float32, so host validation must protect discrete
  scheduler decisions from float64-to-float32 changes.
- Benchmarks report cold and warm timing separately, state all lane dimensions,
  and compare equivalent semantic workloads.

## Current Tools and Examples

- `csi_model_surrogate.py`: canonical CSI construction used by tests and
  benchmarks.
- `csi_batched_parameter_recovery.py`: end-to-end historical five-parameter
  recovery example.
- `csi_triton_vs_llvm.py`: apples-to-apples CSI Triton/LLVM benchmark.
- `gpu_batch_compile_benchmark.py`: broader cross-backend exploratory benchmark.
- `benchmarks/batched.py`: ASV performance cases.

These scripts are examples and measurement tools, not support authorities.
Tests and `BatchedCompilation.rst` define the accepted behavior.
