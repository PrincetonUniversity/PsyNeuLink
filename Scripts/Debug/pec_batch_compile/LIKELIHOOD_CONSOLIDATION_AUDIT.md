# Likelihood compiler consolidation: reuse before expansion

**Historical exploration:** checkpoint `e2721cc518` preserves the code and
experiment described below. The subsequent implementation replaces the custom
algebra with direct SymPy expressions and removes the disposable adapter/script.
See [the current implementation notes](CONTINUOUS_EQUATION_COMPILATION.md) for
the migrated API, retained-domain guards, and validation. Reproduction commands
below apply to the checkpoint, not the migrated tree.

Exploratory audit and local experiment, 2026-09-14. The objective is to retain
features and execution speed while reducing the machinery we own. This is not
a new numerical backend registration, a source/continuous equivalence proof,
or a replacement for the existing CSI likelihood driver.

## Decision summary

1. Do not introduce a second, independently authored experiment schema. Derive
   numerical execution plans from the existing frozen PNL/compiler model.
2. SymPy is promising as a **replacement** for custom equation algebra, not as
   an additional permanent backend behind the current expression language.
3. Keep the fused C++ integration/PDE loops and their adjoints. The experiment
   found no material execution regression when only equation generation changed.
4. Do not promote the prototype yet: domain preservation, structural evidence,
   and larger-model compilation costs need explicit acceptance criteria.

No production code was changed for this exploration. It adds this note and
`experiment_sympy_dynamics.py`; it has not yet reduced production line count.
The script is disposable scaffolding, not a commitment to maintaining two
symbolic implementations. Existing uncommitted continuous-backend work remains
unchanged, as do CSI fitting defaults and cluster jobs.

## Existing ownership and the actual gaps

Paths below are relative to `psyneulink/core/` unless otherwise stated.

| Concern | Existing owner | Consolidation direction |
| --- | --- | --- |
| Nodes, ports, projections, parameter identities | PNL objects; `batched/graph.py`, `ir.py`, `specs.py` | Reuse frozen IDs, bindings, and primitive snapshots; do not maintain a parallel model graph. |
| Persistent and trial-local state | `BatchedStateSpec`, `MechanismOpSpec.states/trial_states`, kernel loop carries | Derive initial/reset/persistent roles from these declarations. |
| Reset policies | `graph._reset_ir_specs`, `BatchedResetSpec` | Reuse policies and state IDs; add a continuous interpretation only where source events do not determine one. |
| Scheduling and termination | `BatchedSchedulerSpec`, schedule regions, `BatchedTerminationSpec`, dynamic KernelIR programs | Analyze the existing representation; a numerical schedule should be a derived plan, not another user-maintained scheduler. |
| Observed fields, scoring versus conditioning, recording policy | `batched/observation.py` | Extend the existing observation contract when necessary; do not add another RT/data specification. |
| Event readout reconstruction | `batched/endpoints.py`, `EventCountReadout`, `EndpointWitness` | Reuse binding/readout evidence, but not integer-count arithmetic as an implicit physical-time interpretation. |
| Stateful likelihood classification and replay | `batched/likelihood_analysis.py`, `dependency.py`, `history.py` | Reuse structural checks and rejection diagnostics; source replay is not automatically continuous replay. |
| Primitive mathematical interpretation | Existing `LikelihoodEffectContract`, Gaussian/Wiener interpretations | Attach missing equation/clock interpretations to the same primitive snapshot, rather than register a separate model definition. |
| Continuous phase equations and stochastic coordinate dependencies | New `continuous_ir.py` | Keep genuinely necessary mathematical metadata; investigate replacing its custom algebra. |
| Batched integration and first-passage probability/adjoints | `batched/numerical/` | Preserve these numerical kernels and their validation. |

The two dependency analyses are related but not interchangeable. Execution-axis
analysis includes scheduling/termination effects at component granularity;
continuous analysis tracks drift/diffusion dependencies at state-coordinate
granularity. They should share identities and evidence, not be merged merely
because both traverse a graph.

PNL also has an RK4 helper in
`components/functions/stateful/integratorfunctions.py`, specialized compiled
integrators, an LLVM scheduler, and an AST-to-LLVM visitor for user functions.
These are not drop-in replacements for fused equation evaluation with the
current batch ABI and integration adjoint. Calling Python integrator helpers
inside each numerical cell would defeat the performance objective. Rebuilding
all of this work around LLVM would be a separate project, not a small cleanup.
MDF export hooks also exist; their presence alone does not establish the clock,
noise, observation, or conditioning contracts needed for direct likelihoods.

### Avoid a false unification

`EndpointExpression` represents source readout arithmetic used with explicit
floating-point/count reconstruction guards. A SymPy expression represents
mathematical algebra and may reassociate or cancel operations. Do not silently
replace source endpoint arithmetic with symbolic simplification. Separate
representations are justified where their semantics genuinely differ.

Likewise, the continuous target still needs an explicit interpretation of
physical clocks and continuous boundary crossing. A source scheduler cannot
uniquely supply that interpretation just by being reused.

## SymPy replacement experiment

`experiment_sympy_dynamics.py` adapts the current equation fixtures to SymPy,
forms weighted-output derivatives using `diff`, applies explicit `cse`, and
emits C++ with the library's C++ printer. A small custom `Sigmoid` retains the
existing overflow-safe numerical primitive and its derivative. User-provided
names remain bindings; only compiler-generated identifiers enter C++.

It substitutes only equation generation. The same `CompiledContinuousPhase`,
`rk4_cpu.h`, OpenMP loops, tensor ABI, integration adjoint, and first-passage
backend are used. There is no SymPy evaluation in the integration time loop.
The temporary adapter checks the current conservative stochastic analysis
before algebraic simplification; it is not a proposed permanent extra IR layer.

### Validation

The existing continuous-phase suite with generator substitution returned
**17 passed, 1 skipped** (the skip is the existing pycodestyle item). It covers
nonlinear equations, all five gradient input groups, finite differences,
parameter-dependent phase chaining, zero phases/padding, empty batches,
stochastic/latent rejection, the generated-drift-to-PDE gradient, and agreement
with the original handwritten CSI drift kernel. An additional script check
verified sigmoid values and derivatives at inputs from -1000 to +1000.

This validates those fixtures, not all algebraic domains or all PNL models.

### Runtime

Local WSL CPU, four threads, float64, 480 trial lanes, 1,000 cells at 1 ms,
two warmups and 21 interleaved measurements per route/mode. Tests and concurrent
compiler builds had finished before the reported run. Medians exclude source
generation and extension loading. This is a **deterministic drift-stage**
benchmark, not whole-subject scoring, fitting, or GPU sampling.

| Measurement | Custom generator | SymPy generator |
| --- | ---: | ---: |
| Forward | 30.583 ms | 30.583 ms |
| Forward + initial-state/input/parameter gradients | 89.067 ms | 86.880 ms |
| Source generation, one invocation | 2.63 ms | 167.98 ms |
| Generated source including unchanged RK4 header | 17,724 bytes / 456 lines | 13,093 bytes / 247 lines |

The gradient median was about 2.5% lower, but run-to-run variation is large
enough that this should be interpreted as comparable performance, not a proven
speedup. Maximum drift difference was 1.67e-16; final states matched exactly.
Maximum compared gradient difference was 1.14e-13.

Both extensions were cached in the reported run. The script separately reports
build-or-cache-load time, which must not be presented as a cold-build comparison.
The initial SymPy build took about 26 seconds while another test build was
running; no fair paired cold-build measurement was performed.

A small dense coupled sigmoid fixture with 2/8/16 states was also generated
(not integrated). Source-generation times were approximately 0.0005/0.0035/
0.0133 seconds for custom generation versus 0.061/0.177/0.797 seconds for SymPy.
Both generated source size and generation time grew with dimension. This is
not Dawa's model, nor a demonstration of acceptable worst-case scaling for
arbitrary nested expressions. Compilation is amortized only if a plan is reused
across parameter proposals rather than rebuilt for each score.

### Does it actually reduce code?

The four current continuous-backend production files total 688 physical lines:
268 in `continuous_ir.py`, 112 in `dynamics_codegen.py`, 134 in `dynamics.py`,
and 174 in `rk4_cpu.h`. These counts exclude tests/docs and the previously
extracted PDE implementation; moving existing PDE code is not inventing a
new solver.

The custom emitter and source wrapper occupy 67 lines. The experimental
SymPy primitive/printer/emitter/wrapper occupy 52 lines, plus a **32-line
temporary adapter**. Thus a generator-only substitution is not a meaningful
line-count win with the adapter retained. Generated C++ size is not maintained
Python source size and must not be counted as codebase reduction.

The larger potential saving is deleting the custom expression constructors,
operators, traversal, and handwritten elementary derivative rules, using
library expressions directly. Binding validation, supported-operation checks,
stable numerical primitives, domain requirements, and structural evidence
would remain. The exact net reduction needs to be measured on that replacement;
it has not been demonstrated by this experiment. Avoiding a new experiment
schema may prevent more technical debt than the initial symbolic line saving.

## Required safeguards before replacement

- **Domains:** The prototype turns `x/x` into `1`. At x=0 the original equation
  would produce a nonfinite value and be rejected, whereas the simplified
  expression would not. This is a known promotion blocker. Preserve/check
  denominator and other domain restrictions, or restrict admission to a
  demonstrably safe subset; do not just relax tests.
- **Structural evidence:** Ordinary SymPy arithmetic can erase `0*x` before
  analysis. Preserve source dependency/effect information independently of
  optimized equations, or use an explicitly unevaluated authoring boundary.
  Mathematical cancellation is not evidence of source effects or exact
  observed-history factorization.
- **Numerics:** Preserve stable sigmoid and other special primitives. Test
  saturation, cancellation, divisions, and reassociation in addition to nominal
  finite differences. Bitwise equivalence is not claimed for the continuous
  target, but floating-point endpoint guards must retain their own semantics.
- **Scaling:** Weighted symbolic differentiation avoids explicitly constructing
  a dense Jacobian, but can still expand shared expressions before CSE. Add a
  shared/nested-expression compilation budget and benchmark beyond this toy.
- **Dependencies:** SymPy 1.14.0 is installed locally alongside Torch
  2.13.0+cu130. PNL does not currently declare SymPy directly. A production
  adoption needs an explicit dependency policy and version coverage, not
  reliance on this environment's transitive dependencies.

## Smallest useful next implementation

1. Resolve the domain/evidence policy on a few primitive equations. Prototype
   direct library expressions with safe indexed bindings and measure **net**
   code deletion, including validation and reference evaluation.
2. If correctness and performance gates pass, replace the custom algebra in
   one path. Remove the conversion adapter and custom generator rather than
   add a permanent backend flag. Preserve tests as regression oracles.
3. Bind primitive continuous interpretations to existing op snapshots and
   source IDs. Derive reset/history/phase plans from existing graph and
   observation evidence; add only missing physical-clock/observation semantics.
   Keep source and continuous targets explicitly distinct.
4. Migrate the CSI subject scan only after that boundary is clear, retaining
   C++ loops and checking whole-subject values, gradients, and timing against
   the existing CSI driver. Do not change point-RT history assumptions during
   this architectural migration.

No new universal experiment DSL, no general symbolic solver, no migration of
the source GPU endpoint arithmetic, and no rewritten PDE are needed for this
next step. Dawa-like models should reuse the same source ownership and equation
declarations, but still require a suitable multi-dimensional stochastic backend.

## Reproduce

From the repository root, with the environment's Ninja/C++ compiler available:

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=1
python Scripts/Debug/pec_batch_compile/experiment_sympy_dynamics.py --check
python Scripts/Debug/pec_batch_compile/experiment_sympy_dynamics.py --repeats 21
```

The script prints diagnostics/timings and writes no result files. Torch may
populate its normal extension cache. Do not overlap the timing run with builds
or other benchmarks. SymPy reference: [code generation documentation](https://docs.sympy.org/latest/modules/codegen.html).
