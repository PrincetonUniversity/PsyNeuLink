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

Controlled N-chain graphs and the real CSI research model now compile through
one typed `ForPasses(trace_kind="lane_local_dynamic")` program. The shared
program contains:

- ordered consideration sets with beginning-of-set-frozen predicates and
  member inputs;
- pass index, per-component execution count, `has_run`, usable-call, finished,
  and component-local RNG-clock scheduler slots;
- explicit carries for retained mechanism state, per-trial mechanism state,
  held effective parameters, current outputs, readouts, and diagnostics;
- per-component execution/truncation budgets plus an independent
  whole-schedule fuel bound; and
- termination checks and result/effect publication at consideration-set
  boundaries.

CSI exercises this machinery with:

- affine cue repeat/switch timing and fixed integral delayed ITI;
- persistent LCA state and finite numeric LCA noise;
- stochastic DDM execution with component-local, lane-local random streams;
- typed folded-affine, lane-persistent threshold control;
- runtime parameter lanes used by the three- and five-parameter PEC fits; and
- the two `WhenFinished(DDM)` output gates.

The old CSI-only graph recognizer, canonical KernelIR builder/validator,
`lane_local_coevolving` region, and private Triton emitter loop have been
deleted. Their typed replacements leave the migration as a net reduction in
production code. CSI remains an exact semantic acceptance fixture, but its
public role names and 11-node/six-set shape are no longer capability
authorities.

This checkpoint is more compositional, but it is not arbitrary PsyNeuLink
scheduling. Conditions, component bodies, state/reset contracts, control
forms, projections, RNG streams, and trial termination must all belong to the
registered dynamic subset described in `BatchedCompilation.rst`.

## Goal

Extend the shared lane-local executor until a new topology composed entirely
from supported mechanisms, controls, predicates, state/reset policies, and
projections requires only fixture code, with no new topology predicate,
KernelIR builder, validator, or emitter loop.

The compiler should continue to be narrower than general PsyNeuLink execution.
Generality here means compositional support for explicitly represented
scheduler semantics, not accepting behavior that cannot be authenticated and
lowered exactly.

## Generic Co-Evolving Architecture

### Implemented semantic foundation

GraphIR represents the scheduler contract with:

- `BatchedSchedulerSpec` and ordered `BatchedConsiderationSetSpec` values;
- typed conditions, dependency component IDs, and finished-value IDs;
- run/trial termination declarations;
- mechanism state and reset declarations;
- effective parameters and ordinary modulation routes;
- folded-affine scheduler control; and
- RNG/state metadata owned by the relevant component.

KernelIR carries one typed dynamic schedule in a
`ForPasses(trace_kind="lane_local_dynamic")` region. Its declaration contains:

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

### Implemented execution semantics

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

CSI's DDM threshold controller is represented by a typed folded-affine record.
Its member reads its own pre-execution count, computes
`base + delta * (count + 1)`, publishes the controller value, and updates
lane-persistent held storage. At trial entry, a typed finished-slot initializer
compares count zero with the previously held raw LCA threshold; the
minimum-one rule applies after the LCA executes. The DDM samples the held
threshold at its ParameterPort update. This preserves count-zero scheduling,
the one-step cleanup visit, and cross-trial threshold persistence without
embedding CSI behavior in the emitter.

### Validation boundary

The shared dynamic path validates compositionally:

- validate each condition and its referenced scheduler slots;
- validate each component body against its registered immutable spec;
- validate state, output, control, and RNG ownership independently;
- validate consideration-set ordering and frozen-input semantics;
- validate explicit region arguments, yields, effects, and diagnostics; and
- prove every executable GraphIR declaration has a corresponding KernelIR
  operation and vice versa.

Whole-topology node counts, CSI role names, and topology-shaped equality checks
must not be capability authorities in the generic path.

## Completed Migration

The CSI-specialization replacement is complete:

- shared typed condition validation/evaluation was extracted so
  precomputed traces and dynamic execution use the same predicate semantics;
- per-component body construction and ordering were separated without changing
  precomputed lowering;
- typed dynamic-schedule declarations, validators, scheduler slots, carries,
  budgets, and fuel were added;
- controlled N-chain and registered scheduled-terminator fixtures were moved to
  the generic region;
- deterministic, stochastic, delayed-ITI, persistent-state, folded-threshold,
  runtime-lane, and both real CSI PEC surfaces were moved to that same region;
  and
- the private co-evolving recognizer, KernelIR path, validation branches, and
  emitter were removed after interpreter and physical-GPU parity.

## Remaining Generalization Sequence

### 1. Broaden graph admission compositionally

- Replace the remaining exact controlled-chain shape check with reusable
  per-edge and per-component invariants.
- Extend the existing one-finished-value output split to fan-in dependencies
  and predicates with more than one finished dependency.
- Admit multiple independent and interacting stateful chains without requiring
  one count producer plus one dynamic terminator.

### 2. Broaden scheduler and dataflow semantics

- Add the next typed predicate/time-scale forms required by real models.
- Implement current/next output banks for safe same-set data projections.
- Generalize trial termination beyond exact default `AllHaveRun()` while
  retaining termination checks between consideration sets.
- Extend reset/state initialization beyond exact `Never` and `AtTrialStart`.

### 3. Broaden mechanisms, control, and RNG

- Register more stateful steppers and terminators with explicit state, readout,
  finished, and budget contracts.
- Cover multiple stochastic mechanisms and prove component-local stream
  independence when their execution histories diverge.
- Generalize ordinary modulation beyond the current scalar `OVERRIDE` chains
  and folded-affine control beyond the authenticated DDM threshold contract.

### 4. Keep CSI as a regression, not an authority

- A new topology built only from admitted pieces must lower without production
  code changes.
- Python scheduler traces must match per lane for same-set freezing, mid-pass
  termination, resets, persistent effects, and truncation.
- Direct-IR forgeries must continue to fail before source emission.
- Interpreter and physical-GPU parity remain required for each expanded
  semantic boundary.

The compiler should claim arbitrary co-evolving support only within the
explicitly registered subset, and only after the remaining branching, fan-in,
multiple-stochastic-component, control, and state/reset fixtures satisfy these
criteria.

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
- `test_batched_dynamic_terminator_acceptance.py`;
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
