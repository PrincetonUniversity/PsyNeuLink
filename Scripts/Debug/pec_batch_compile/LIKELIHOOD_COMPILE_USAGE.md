# Likelihood compilation: diagnosis, history replay, paths, and primitive sampling

The implementation on `feat/likelihood_compile` diagnoses model structure,
derives supported event readouts, and reconstructs history for a checked
single-event subset using the generated source scheduler. A checked boundary-path interface materializes
deterministic inputs in candidate/trial-parallel lanes. A generated primitive
sampler consumes these paths in candidate/trial/estimate lanes. Checked scalar
observation gates and an explicit empirical count-domain mass objective now run
on CPU interpreter or compiled GPU backends. Automatic PEC routing, broader
observation/estimator support, and production performance remain subsequent steps.

## Example: independent DDM trials

```python
import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    ObservationField,
    ObservationSpec,
)

decision = pnl.DDM(
    function=pnl.DriftDiffusionIntegrator(noise=0.2),
    output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
)
composition = pnl.Composition(pathways=decision)
observations = ObservationSpec((
    ObservationField(decision.output_ports[0], measure="counting"),
    ObservationField(
        decision.output_ports[1], measure="counting", role="event_time",
    ),
))
report = BatchedCompositionCompiler.diagnose_likelihood(
    composition, observations=observations, max_steps=256,
)
print(report.history_kind)          # independent_trials
print(report.factorization_status)  # eligible
print(report.codegen_ready)         # False: diagnosis does not select an estimator
```

Counting measure here describes exact outputs of the discrete simulator. A
floating-point RT output does not establish a continuous-time density. Asking
for `measure="lebesgue"` retains a density-existence obligation. The compiler
does not silently adopt the continuous CSI direct model.

Fields bind to exact OutputPort objects at the public boundary. Reports contain
only lowered component/port IDs, field widths, column offsets, and observation
semantics. Ports from another model and duplicate port declarations are rejected.
`report.to_dict()` exposes diagnostics and evidence for serialization without
live PNL objects. Field order defines the data-column order for lowering.

You can also use `simulation_plan.diagnose_likelihood(observations)` on an
existing plan. Later registry replacement cannot change the contracts used by
that plan. Diagnosis can report structural results without a usable GPU;
simulation backend availability is recorded separately in `report.simulation`.

## Persistent history

For CSI, declare choice and RT at the actual output gates, and register the
drift UDF with its explicit effect contract. The report then identifies a
`deterministic_history_candidate`. The same analysis applies after renaming the
nodes; it does not call the CSI-specialization validator.

This diagnosis is deliberately incomplete. A separate `compile_history_replay()`
step checks a restricted event substitution and generates scheduler replay
for state and held controls. `compile_boundary_trajectories()` checks the
within-trial input boundary and generates deterministic paths for this subset.
The separate `compile_stochastic_sampler()` step then generates raw primitive
sampling, not a score. A user-supplied `event_time` label cannot discharge these
obligations.

Current cases are:

| `history_kind` | Interpretation |
|---|---|
| `independent_trials` | No retained mechanism state or held controls after declared trial resets, under registered effect contracts. |
| `deterministic_history_candidate` | Deterministic reconstruction is a candidate; obligations remain before any execution transformation. |
| `unresolved_stochastic_history` | Randomness reaches history through unresolved data/control/scheduler dependencies, or a required endpoint observation is unavailable/ambiguous. |
| `unknown` | Source lowering, a primitive effect contract, or initial-state semantics is unavailable. |

Candidate analysis preserves the original simulation dependencies. In
particular, the LCA endpoint remains stochastic during forward simulation when
DDM stopping time controls execution duration. No simulation loops are split
by this diagnostic API.

`ObservationField` separates recording semantics and data availability:

- `recording="exact"` is the default.
- `recording="rounded", precision=0.01` declares recording precision and
  requires counting measure; it does not set a histogram bandwidth.
- `recording="noisy"` or `"censored"` records a need for a future measurement
  or censoring operator. Such declarations cannot enable endpoint reconstruction.
- `availability="may_be_missing"` prevents assuming that every event is known.
- `score=False, condition_history=True` preserves a known observation for
  history reconstruction without claiming a full joint score for all fields.

Rounded times might identify a unique discrete endpoint for particular data
and parameters. That data-dependent check is not implemented yet, so diagnosis
is conservative. Initial conditions default to the lowered model defaults;
`initial_state="latent"` produces an unresolved-initial-state diagnostic.
Inputs are conditioned on. Diagnosis does not evaluate trial data; the endpoint
reference interface below accepts one subject at a time. Multi-subject history
execution will require explicit subject boundaries.

## Checked event counts

Using the composition and observations above:

```python
endpoint_plan = BatchedCompositionCompiler.compile_observed_endpoints(
    composition, observations=observations, max_steps=256,
)
# Data columns follow ObservationSpec order: choice, then exact RT.
# Inputs and parameter_sets use the same conventions as simulation_plan.run().
counts = endpoint_plan.reconstruct(inputs, data, parameter_sets=parameter_sets)
# Read-only int64 array: [parameter candidate, trial, conditioning event].
```

An existing simulation plan also exposes
`simulation_plan.compile_observed_endpoints(observations)`. Compilation and
reconstruction are CPU-side operations and do not require a CUDA device.

The DDM primitive declares the readout `non_decision_time + steps *
time_step_size`. The compiler follows this rule through scalar dense
projections, supplied inputs, and stateless affine functions. In the coupled
scheduler tier it also checks publication order and termination dependencies.
Consequently, CSI's RT gate is derived from its actual graph, including the cue
input, projection weights, and output-function parameters—not from CSI node
names or a hard-coded RT formula. Unknown/nonlinear transformations and
scheduler-modulated readout parameters currently produce explicit rejection.

The plan stores an expression and its source identities in an `EndpointWitness`.
Validation rederives this expression from the frozen primitive contracts and
rejects changed witnesses. This is translation validation under trusted
primitive declarations, not an independent theorem checker or Lean certificate.
Changing the live registry after compilation does not change the snapshot.

For each candidate and trial, the reference evaluator checks all counts from
the primitive's minimum through `max_steps`, using bounded-memory chunks. It
re-evaluates candidate-dependent step size, nondecision time, and affine
parameters, including supplied trial-varying parameters. It returns a count
only when exactly one is compatible with the observed readout. It applies no
nearest-bin or ceiling policy. Zero or multiple compatible counts raise
`EndpointReconstructionError` with a diagnostic `code`; invalid parameter
inputs can also raise the shared simulation preparation errors.

Compatibility uses conservative outward float32 arithmetic enclosures, allowing
both contracted multiply-add and separate rounding in the declared expression.
This is a roundoff guard, **not a measurement-noise model or recording tolerance**.
Success establishes one compatible count within the configured cap; it does
not prove exact simulator support, positive likelihood, or uniqueness outside
that cap. Subnormal arithmetic and overflow are conservatively rejected. This
reference implementation enumerates the bounded count domain and is not yet an
optimized fitting hot path.

Only complete observations declared `recording="exact"` are supported here.
Rounded, noisy, censored, or potentially missing event observations cannot use
this reconstruction path. Its count is the primitive's **active integration
count**, not the total number of scheduler passes. In CSI, resolving that count
does not by itself determine the LCA's onset passes, trial reset behavior, or
held controls. Reports therefore retain history/trajectory obligations and
`codegen_ready=False`, even when `endpoint_witnesses` is populated. No simulation
dependency edges are removed and no likelihood implementation is replaced.

## Checked scheduler/history replay

For a supported coupled composition such as CSI, with choice and RT declared
at its actual output gates:

```python
# Use a fresh CPU-interpreter process: TRITON_INTERPRET=1.
history_plan = BatchedCompositionCompiler.compile_history_replay(
    composition, observations, max_steps=256,
)
history = history_plan.reconstruct(inputs, data, parameter_sets)
print(history.end_states)                  # [candidate, trial, state column]
print(history.end_effective_parameters)    # [candidate, trial, held-value column]
print(history.execution_counts)            # [candidate, trial, component]
print(history.scheduler_rounds)            # [candidate, trial]
print(history.event_counts)                # [candidate, trial]
```

Existing simulation plans expose `compile_history_replay(observations)` too.
This initial runtime supports `triton_cpu` (the default) and compiled `triton`
GPU plans. Use separate fresh processes for the two execution modes. It is not
yet a production fitting backend: endpoint inversion, checks, and public trace
arrays remain host-side, with transfers between stages.
It requires one nonempty contiguous subject, model-default
initial state, one observed stochastic event primitive, complete exact event
observations, and registered effects for every primitive. No particle lanes or
random draws are used in reconstruction. It supports scalar and trial-varying
parameter candidates through the shared input/parameter preparation code.

The event primitive must declare the stronger
`EventCountReadout(execution_rule="one_step_until_finished", ...)` contract:
zero counter and false finished flag at trial entry, one integration step per
scheduled execution while unfinished, and an absorbing finished flag/counter.
The DDM registers this rule. An algebraic RT readout alone is insufficient.

The checker rejects unresolved stochastic influences on state or held controls,
unobserved persistent state inside the event primitive, extra stochastic roots,
latent initial conditions, and unsupported scheduler programs. It checks all
state columns exposed by the trace, including those reset on the next trial.
It records a `HistoryReplayWitness` and rederives it before source generation
and execution. The ordinary simulation graph and its dependency edges remain
unchanged. This is checked translation under trusted primitive contracts, not
a formal proof of scheduler or component semantics.

Replay replaces only the stochastic event step with its observed count and
finished publication. The existing generated scheduler still runs the actual
deterministic component bodies, resets, recurrent values, control updates,
consideration-set publication, and termination tests. Thus CSI's LCA evolves
during DDM execution and on the terminating pass; there is no independent CSI
recurrence or hard-coded onset-to-pass formula in the implementation.

State columns follow the source `kernel_ir.states` order, flattened by width.
Held-value columns follow `witness.effective_parameter_ids` and represent the
stored modulation values, not necessarily the final sampled target parameter.
Component-count columns follow `witness.component_ids`. Start arrays are taken
at the trial boundary **before trial resets**; end arrays are taken after
termination. Active integration counts, scheduled component calls (which may
include post-finish calls), and scheduler rounds are deliberately separate.
Arrays are read-only. Truncation, incomplete termination, a mismatched event
count, or nonfinite reconstructed state causes an error.

The inspection interface deliberately provides no chunk-resume/initial-state
argument: both mechanism state and held controls must be preserved before that
can be added safely. Recompute history whenever candidate parameters, inputs,
or conditioning observations change.

For differential verification:

```python
forward = history_plan.simulate_reference(inputs, parameter_sets, seed=12)
# One simulation lane per candidate; forward.observations contains its outputs.
replayed = history_plan.reconstruct(inputs, forward.observations[0], parameter_sets)
# The example above assumes one candidate. Real data is shared across candidates.
```

`simulate_reference()` instruments ordinary, unforced coupled simulation using
the same trace layout. Replay traces never expose placeholder stochastic
outputs as data: `history.observations` is `None`. The reference comparison
checks the event substitution and execution transformation; it shares component
implementations and is not an independent oracle for their mathematics.
`history_plan.source(replay=True/False)` exposes both generated programs.

Successful replay does not establish that an exact observed value has nonzero
probability, supply a likelihood value, or validate independent stochastic
trials. The endpoint roundoff/support limitations above still apply. Reports
keep `codegen_ready=False`; existing CSI likelihood workflows are unchanged.

## Deterministic boundary paths

After compiling a history replay plan:

```python
path_plan = history_plan.compile_boundary_trajectories()
paths = path_plan.generate(inputs, data, parameter_sets, horizon=128)
print(paths.values.shape)        # [candidate, trial, active step, field column]
print(paths.valid.shape)         # [candidate, trial, active step]
print(paths.pass_indices.shape)  # zero-based scheduler pass at each step
print(paths.fields)              # source-value identities and column layout
```

The compiler derives the boundary from the actual scheduled `StepMechanism`:
its combined input vector and explicitly sampled held parameter values. For
CSI this includes the DDM input signal and held threshold modulation. These are
the values supplied to the primitive, **not** an independently expanded drift
or collapsing-bound formula. Candidate/trial parameter bindings remain in
`witness.static_parameter_ids`; “static” here means fixed within that trial,
not fixed across candidates or trials. The primitive's own counter, noise,
bound transformation, and update formula remain responsibilities of the
stochastic region evaluator described below.

The checked `BoundaryTrajectoryWitness` records typed input fields, the
consumer and consideration-set identities, parameter bindings, deterministic
dependency closure, and the history witness. Analysis includes dependencies
of the stochastic member's execution predicate, not only its data inputs.
Unobserved stochastic influences on these boundary values or their scheduling
are rejected. The current rule also requires the stochastic component's
scheduled execution to stop when it finishes. Source generation rederives the
witness from the frozen model; this remains checked translation under trusted
primitive contracts, not formal certification.

Generation has two separate executions:

1. Sequential observed-history replay supplies each trial's canonical starting
   mechanism state and held control values.
2. A generated path kernel assigns one lane to each candidate/trial pair,
   restores that complete start, applies the source's trial resets, and runs
   the actual deterministic scheduler/components to the requested horizon.

Each path is sampled immediately before its stochastic step, using the same
frozen consideration-set values that the coupled simulator consumes. No
stochastic draws are made during path generation. The registered drift UDF and
actual projections are used; there is no CSI drift transcription in this path.
The generated path kernel has no sequential trial loop.

Hypothetical tails are never carried into the next trial. Changing a trial's
observed RT can change subsequent trial starts, but does not change that
trial's own full input path when its start and parameters are unchanged.
Changing the path horizon likewise leaves canonical history unchanged.
`paths.history` contains the **observed** history, not endpoints of the
hypothetical path executions. Its event counts need not equal the path horizon.

For comparison, `path_plan.simulate_reference(inputs, parameter_sets, seed=12,
horizon=128)` captures boundary values from unmodified coupled simulation.
Reference entries after the actual stopping step are invalid (`valid=False`),
with `NaN` values and pass index `-1`; compare only the valid prefix. A generated
path has all requested steps valid, including hypothetical steps beyond the
observed stopping time. Arrays are read-only and the result's `mode`
distinguishes `coupled_reference` from `observed_history_paths`.

Both interfaces execute on `triton_cpu` or compiled `triton` GPU plans in fresh
processes. Candidate/trial parallelism is expressed in the generated lane layout;
this is not a CPU-threading or GPU-speedup claim. They accept one contiguous
subject from model defaults, with no external state injection or chunk resume.
`horizon` must be a positive integer no larger than the source step cap.
`max_buffer_bytes` (default 256 MiB) bounds the estimated inspection buffers
before history execution; it is not a process-memory limit and excludes
compiler/framework workspace. No likelihood workflow is automatically routed
to these inspection APIs.

## Conditional primitive sampling

```python
sampler = path_plan.compile_stochastic_sampler()
samples = sampler.sample(
    inputs, data, parameter_sets, num_estimates=16, seed=21,
    common_random_numbers=True, horizon=128,
)
print(samples.values.shape)       # [candidate, trial, estimate, raw output column]
print(samples.outputs)            # primitive port identities and column layout
print(samples.event_counts)       # active integration counts, [candidate, trial, estimate]
print(samples.truncated)          # same lane shape
```

This first sampler supports the checked single stochastic event primitive with
trial-local state. It generates boundary paths internally from the same inputs,
observations, and parameter candidates; it does not accept unchecked cached
paths. It then emits only the stochastic primitive's registered step and readout,
with its source parameter bindings, initial trial state, sampled held controls,
and random-number streams. Deterministic mechanisms are not re-executed for each
estimate. There is no CSI-specific drift, threshold, or update formula in the
sampler. In particular, a controller's already-transformed threshold is passed
to the existing primitive binding, not transformed a second time.
The source compiler's parameter constraints still apply; this interface does
not make frozen parameters (such as the current coupled DDM noise binding)
available for fitting.

`values` contains **raw primitive outputs**, not the composition's final output
gates. For CSI, raw DDM decision/RT should not be confused with the final
correct-response coding or RT after cue-timing adjustment. No density, histogram,
likelihood, or support claim is attached to these samples. Observation mapping
and scoring use the separate checked interfaces below.

Sampling preserves the source subject/trial/estimate/component execution RNG
addressing, including paired-normal caching and the common-random-number option.
Every estimate begins at the same reconstructed state for its candidate/trial.
Sampled endpoints never replace the observed history used by later trials.

For a coupled comparison using those **same conditional starts**:

```python
import numpy as np

reference = sampler.simulate_reference(
    inputs, data, parameter_sets, num_estimates=16, seed=21,
    common_random_numbers=True, horizon=128,
)
np.testing.assert_array_equal(samples.values, reference.values)
np.testing.assert_array_equal(samples.event_counts, reference.event_counts)
```

The reference restores both canonical mechanism states and held controls in
every trial/estimate lane, then runs the full source scheduler, deterministic
mechanisms, and stochastic primitive. This differs from unconditional
multi-trial simulation, where each estimate develops its own random RT history.
It checks the execution split but shares the primitive implementation; it is
not an independent mathematical validation of the primitive.
`sampler.source(reference=False/True)` exposes both generated programs.

Truncation raises `StochasticSamplingError(code="sampling.truncated")` by
default. `strict_truncation=False` returns explicit flags and partial outputs
for inspection; those outputs are not completed events and must not be silently
scored as such. The runtime supports `triton_cpu` and `triton`, one contiguous
subject, and a fresh process for the chosen backend. Positive horizon, estimate count,
buffer-budget, and lane-index bounds are checked before allocation. The buffer
budget excludes compiler/framework workspace. The local GPU validation is
described below. Structural diagnosis reports still say `codegen_ready=False`:
they do not select a numerical estimator. The explicit mass compiler below is
a separate opt-in interface, and existing CSI/PEC fitting workflows are unchanged.

## Checked observations and explicit empirical mass

```python
observed_sampler = sampler.compile_observation_sampler()
observed = observed_sampler.sample(inputs, data, parameter_sets, num_estimates=1024)
# observed.values: [candidate, trial, estimate, declared observation column]

# Equivalent one-call entry point, including automatic history/path/readout checks:
mass_plan = BatchedCompositionCompiler.compile_empirical_mass(
    composition, observations, backend="triton", max_steps=128,
)
result = mass_plan.score(inputs, data, parameter_sets, num_estimates=4097, seed=17)
print(result.successes)       # [candidate, trial], raw hit counts
print(result.probabilities)   # successes / num_estimates
print(result.log_factors)     # zero hits give -inf, without a floor
print(result.log_likelihood)  # [candidate], sum of log factors
```

Observation mapping derives scalar affine gates, dense scalar projections,
known inputs, and parameter bindings from the frozen graph. It shares the
endpoint derivation's publication checks: dependent gates must publish after
the event finishes, with their dependencies available. Sampled primitive ports
are leaves in this derivation. There are no CSI-specific gate formulas or node
names in production lowering. It rejects unsupported/nonlinear transformations,
modulated gate parameters, incomplete or non-exact recording, and vector fields.
The observation declaration fixes column order, including reordered gates.

`ObservationSamplerPlan.simulate_reference()` returns the actual declared ports
from the full coupled source scheduler, not the new arithmetic expressions.
This independently checks the readout translation against source execution.
All witness artifacts are rederived before emission/execution.

The mass estimator's target is explicitly
`checked_event_count_and_exact_fp32_observation_fields`:

- A scored event-time field matches the unique count returned by the existing
  endpoint compatibility guard. That guard is **not** proof of exact floating-point
  readout support; this is a count-domain target, not a density in real-valued RT.
- Other scored fields match exact values after FP32 conversion. All scored fields
  require counting measure. A second scored event-time field cannot silently
  reuse the first field's count; it is rejected pending a consistency check.
- Unscored event times still condition history. No sampled history is substituted.
- Every lane must finish; truncation raises, without discarding/renormalizing lanes.
- Zero hits remain zero. There is no bandwidth, smoothing, pseudocount, density
  conversion, or implicit log floor. `zero_hits` exposes sparse factors.

The sampler/readout kernels run on the chosen device. For GPU scoring, the
observation tensor stays on the GPU and Torch reduces the matching lanes there;
small count/status and result arrays still cross the host boundary. This is not
a fused, chunked, or fully device-resident objective. The logarithm of the Monte
Carlo mass estimate is biased and can be unstable. No gradient, uncertainty
certificate, or pseudo-marginal MCMC guarantee is supplied.

This narrow estimator is useful for correctness checks and exact discrete
observation experiments. It does **not** replace the existing smoothed CSI
likelihood or implement a recording model for real participant RTs. PEC routing
and such estimator/measurement choices need an explicit subsequent design.

## Registered effect contracts

Built-in compiler primitives declare complete effects. Custom implementations
must explicitly supply a `LikelihoodEffectContract`, for example:

```python
from psyneulink.core.batched import LikelihoodEffectContract, batched_node_op

@batched_node_op("my deterministic readout",
                 likelihood_contract=LikelihoodEffectContract())
def readout(x0, x1):
    return x0 - x1
```

The assertion covers absence of hidden state, external effects, and undeclared
randomness in the source implementation and its registered lowering. It is
trusted author input, not an automatically checked mathematical proof. Contracts
are frozen with op implementations and reported as `registered_contract` trust.
Omitting one does not change simulation support, but blocks likelihood analysis.

Mechanism primitives with declared RNG streams use
`LikelihoodEffectContract(randomness="declared_streams")`. Registration rejects
contracts that disagree with their stream declarations. Component-local
randomness is distinct from randomness inherited through data or scheduling.
For example, numeric LCA `noise` in the supported PNL implementation is an
additive deterministic term; classification uses RNG effects rather than the
parameter's name.

Readout semantics are a separate part of the contract. `EventCountReadout`
declares the counter, output port, offset parameter, step parameter, and minimum
count; registration checks that these bindings exist. Built-in Linear and
MappingProjection declarations provide the affine and dense-projection rules.
An effect-only contract on a custom function does not establish its algebraic
readout rule.

No current report certifies density existence, likelihood unbiasedness,
gradients, or suitability for pseudo-marginal MCMC. Structural factorization,
observation semantics, estimator properties, and kernel implementation remain
separate parts of the implementation plan.

## Verification

The current broad interpreter run, including boundary paths, history, endpoints,
diagnosis, axis dependencies, registry snapshots, resets, compiler regression,
stochastic sampling, and observed sampling, passed **185 tests with six GPU-mode
skips** (617 seconds). Additional renamed-observation and CSI/Python acceptance
checks passed **six tests with one opposite-backend skip** (71 seconds).

On the local RTX 2080 Ti, the new conditional/observation/mass suites passed
**11 tests with four interpreter-mode skips** (41 seconds), and existing GPU
comparisons against the transition oracle, LLVM, and fresh PNL Python passed
**four tests** (36 seconds). These run counts overlap; they are not a unique
aggregate test count. Ruff and `git diff --check` also passed. No cluster job
was submitted. See [the GPU validation report](LIKELIHOOD_COMPILE_GPU_VALIDATION.md)
for the larger 10 ms/1 ms experiment and reproducible commands.

Endpoint tests cover parameter-dependent clocks, affine transformations,
incompatible/ambiguous observations, forged witnesses, frozen registry rules,
and reconstruction against coupled CSI interpreter outputs. Metamorphic CSI
coverage renames the nodes and changes an RT projection and output gate.
Separate and contracted float32 readout calculations exercise the roundoff
enclosure independently of the simulator's arithmetic.

History tests compare both deterministic and stochastic coupled runs against
observed-count replay, including every exposed state, held control, and
execution count. Additional cases cover candidate/trial-varying timing,
renaming and altered gates, delayed and zero onset, numeric additive LCA noise,
trial resets, one-step and exactly-at-cap events, forged witnesses, missing
execution contracts, and stochastic-state rejection. The broad regression run
also includes existing comparisons of coupled CSI outputs against fresh PNL
Python execution for ordinary, zero-count, and affine-onset timing.

Boundary-path tests additionally verify exact pre-step input/modulation
agreement with coupled simulation, candidate/trial parameter lanes versus
separate runs, horizon-invariant canonical history, causal propagation of a
changed observed RT to subsequent trials only, altered registered drift code
and projection weights, delayed onset/reset restoration, forged artifacts,
dependency rejection, and buffer/horizon guards.

The stochastic-sampler suite checks exact raw-output and event-count agreement
with full coupled conditional execution across deterministic/noisy models,
multiple estimates/candidates, and both common-random-number policies. It also
compares against ordinary noisy forward simulation; verifies identical-candidate
draw sharing and independent streams; exercises delayed onset and trial-varying
parameters; and checks witness tampering, resource guards, explicit truncation,
and horizon-invariant canonical history.

The reset-state suite's fixtures previously changed Never reset declarations
to AtTrialStart without updating the matching LCA initialization policy (16
failures reproduced at `034f4f2679`). The fixtures now change those two related
declarations together; production reset semantics are unchanged.
