# Automatic compilation of observation-conditioned history likelihoods

Status: implementation started on `feat/likelihood_compile`, 2026-09-09.
This is the immediate engineering track for
[Proof-Carrying Likelihood Compilation](PROOF_CARRYING_LIKELIHOOD_COMPILER.md).

## Implementation progress

The first slice adds observation declarations, frozen primitive effect
contracts, and `BatchedCompositionCompiler.diagnose_likelihood()`. An existing
`BatchedSimulationPlan` also exposes `diagnose_likelihood()` so analysis uses
its exact implementation snapshot. See
[the current API and limitations](LIKELIHOOD_COMPILE_USAGE.md).

This classifies reset trials and deterministic-history candidates without CSI
name checks, accounts for held controls, and reports unresolved dependencies.

The next slice implements checked event readouts: frozen primitive counter
contracts, scalar affine/projection derivation, publication-order checks,
rederived witness validation, and a CPU reference runtime guard. The public
`compile_observed_endpoints()` interface reconstructs per-candidate active
integration counts from complete exact observations. The guard checks unique
compatibility within the step cap using conservative float32 roundoff
enclosures; it is not an exact-support or likelihood certificate. Renamed CSI
and altered RT gates exercise the generic derivation against coupled execution.

The following slice adds checked one-event scheduler replay and inspectable
generated history execution. A stronger primitive execution contract and
dependency checks authorize replacing the event step, while the original
generated scheduler and deterministic component emitters reconstruct state and
held controls. `compile_history_replay()` exposes a one-subject CPU reference
with trial-boundary states, held values, component calls, and scheduler rounds.
An instrumented, unmodified coupled simulation supplies differential traces.

The next slice derives the stochastic step's typed input/effective-parameter
boundary, checks its data and scheduler dependencies, and generates
candidate/trial-parallel deterministic paths from canonical observed-history
starts. It records pre-step values and pass indices, and compares their valid
prefixes with instrumented coupled execution. Hypothetical tails cannot update
another trial's canonical history. The path uses registered component bodies,
including altered drift UDFs and projections, without CSI-name recognition.

These slices implement the one-event reconstruction/boundary portion of change
2 and the initial inspection paths of changes 3–4, not an executable likelihood.
Generated stochastic-region execution/scoring, multi-subject and chunk-safe
execution, GPU validation/performance, and non-decision acceptance cases remain
outstanding. Consequently, likelihood reports always have
`codegen_ready=False` and `can_execute=False`. The existing CSI sampling
implementation continues to perform likelihood work.

## Objective and first release boundary

Compile the optimization currently implemented by the handwritten CSI GPU
likelihood from model structure and declared observation semantics. The user
supplies a Composition, parameter domain, and observation specification; the
compiler determines whether persistent state can be reconstructed from the
observed history and generates the appropriate execution plan.

The first release should support:

- independent trials whose relevant state resets;
- deterministic persistent state reconstructible from observed history;
- explicit diagnosis of residual latent history or unsupported semantics.

Automatic generation of the second case is the highest priority. General
particle filtering, direct equation synthesis, LLM derivation, and Lean proofs
are subsequent work. Their interfaces should fit this architecture, but their
implementation should not delay replacing the CSI-specific kernels.

The initial target preserves the source's discrete scheduler, component
updates, stopping rules, and output readouts. It does not automatically replace
Euler integration or endpoint detection with a continuous process. The existing
continuous CSI direct solver remains a separate numerical reference.

## Existing implementation to build on

Paths below are relative to the repository root.

| Existing code | Role in the implementation |
|---|---|
| `psyneulink/core/batched/ir.py` | Reuse graph identities, ports, state, resets, RNG declarations, and parameter constraints. |
| `psyneulink/core/batched/dependency.py` | Reuse axis analysis and dependency edges; add a separate observation-conditioned analysis. |
| `psyneulink/core/batched/kernel_ir.py` | Reuse scheduled step operations, publication order, effects, and validation. |
| `psyneulink/core/batched/specs.py` | Extend registered primitive contracts and freeze them with implementation snapshots. |
| `psyneulink/core/batched/backend/triton/emit/` | Generate transformed programs using existing component lowering. |
| `psyneulink/core/batched/compiler.py` | Expose likelihood diagnosis, compilation, and execution beside simulation plans. |
| `psyneulink/core/components/functions/nonstateful/fitfunctions.py` | Route PEC objectives through a cached likelihood plan. |
| `psyneulink/core/batched/backend/triton/csi_deterministic.py` | Keep temporarily as a differential oracle and performance baseline. |

Current axis analysis explicitly propagates stochastic trial-termination
dependencies back into repeatedly executed components. That is correct for
forward simulation: a noise-free LCA can have a random endpoint because its
execution duration is random. Do not weaken this analysis merely to classify
the LCA as deterministic. Add a distinct analysis under declared observations.

Current primitive declarations already distinguish persistent and trial-local
state and expose scheduled step/readout hooks. They do not yet constitute a
complete mathematical purity or observation contract for arbitrary UDFs.

## Legality condition

Let `h_t` be the relevant state at the start of trial `t`, `u_t` the known
inputs, and `y_t` the recorded observation. A candidate factorization is:

```text
h_1       = initialize(theta, declared initial conditions)
factor_t  = p_theta(y_t | h_t, u_t)
h_(t+1)   = reconstruct(theta, h_t, u_t, y_t)
likelihood = product_t factor_t
```

The reconstruction must determine every part of the next state that can affect
future observations. The first implementation should conservatively require
all retained state to be reconstructible; eliminating irrelevant hidden state
can follow with a separate liveness justification. Trial randomness must be
independent across trials conditional on this state and declared inputs.

The compiler must check two distinct conditions:

1. The observation determines the carried state after the trial.
2. At each execution point, the deterministic inputs needed by the stochastic
   region can be evaluated without knowing an unobserved stochastic trajectory.

The second condition permits precomputation of a time-varying input path.
Absence of direct feedback projections alone is insufficient: scheduler
conditions, held control values, execution counts, initialization, and
termination all carry dependencies.

For CSI, first reconstruct the state at the beginning of each trial from
preceding RTs. Then evolve the LCA throughout each possible current-trial
duration to supply the DDM's changing drift. Use the current observed endpoint
only to select the state carried to the next trial. Do not stop the current
trial's simulated paths at the observed RT or replace its LCA path with a
constant starting state.

The factors can be evaluated in parallel after history reconstruction. This
does not claim that the original trials are marginally independent.

## Observation and semantic contracts

Introduce a minimal `ObservationSpec` using port/event identities rather than
CSI names. It should declare:

- observed fields, their transformations, support, and mass/density semantics;
- event time or fixed observation horizon, units, recording precision, and
  any measurement noise;
- subject/sequence boundaries and initial conditions;
- which observations condition history and which contribute score terms;
- treatment of missing observations, censoring, and timeouts.

Keep the observation model separate from histogram/KDE estimator settings.
A density-estimation bandwidth is not the recording precision of the data.

Start with fixed known horizons and event readouts whose scheduler execution
count can be uniquely recovered through registered rules. For example, an
affine readout can bind a component's execution count, timestep, and known or
parameter-dependent offset. Recover the relevant scheduler phase and other
execution counts from the schedule, not merely from that component count.
Generic inversion of arbitrary Python output functions is outside the first
release.

The compiler may infer a declared primitive's readout relation, but cannot
infer how a researcher rounded, censored, or measured it. Examples and the PEC
adapter must make that distinction explicit.

Specific policies:

- A unique observed scheduler endpoint can enable deterministic history.
- Rounded/noisy RTs that admit several endpoints normally require latent-state
  inference. A singleton compatible endpoint is eligible after validation.
- Existing CSI ceil/snap endpoint reconstruction is an explicit compatibility
  policy. For off-grid recorded RTs it must be labeled as an approximation,
  not presented as exact conditioning on the source's discrete observation law.
- Excluding a trial from scoring does not erase its known history. Distinguish
  a score mask from missing conditioning data; describe masked scoring as a
  selected conditional objective rather than the full joint likelihood.
- Missing or censored times that leave relevant state uncertain must block this
  specialization. A declared reset may make a later segment eligible again.
- Runtime truncation is not automatically observed censoring. Report it or
  reject the evaluation unless a corresponding observation category is declared.

Extend primitive contracts with state effects, stochastic effects, readout
relations, and applicability predicates. Base stochasticity on actual RNG
effects, not parameter names: the current LCA implementation treats numeric
`noise` as a deterministic additive term and rejects distribution-valued noise.
Any zero-noise simplification needs a primitive rule and a parameter-domain or
runtime guard, including trial-varying and modulated parameter values.

Opaque UDFs need explicit trusted contracts or analyzable restricted bodies.
Record contract trust in the artifact; a declaration is not a formal proof.
Bind the contract to the exact frozen implementation. The generated CSI path
must use the registered drift implementation, removing the duplicated formula
from production likelihood code.

## Compiler passes and proposed internal representation

Use a small likelihood layer beside existing execution IR. Avoid a second
independent reconstruction of the Composition or a competing scheduler.

Proposed modules under `psyneulink/core/batched/`:

| Module | Responsibility |
|---|---|
| `observation.py` | Observation declarations, normalization, event-readout contracts. |
| `likelihood_ir.py` | Immutable factorization, reconstruction, trajectory, estimator, and evidence records. |
| `likelihood_analysis.py` | Applicability, conditioned dependency analysis, runtime guards, rejection reasons. |
| `likelihood_planner.py` | Strategy selection and lowering of validated partitions to execution programs. |
| `backend/triton/history.py` | Runtime orchestration of generated history/trajectory/sampling kernels. |

Names are provisional. Extend existing modules where their abstractions fit;
do not expand `kernel_ir.py` with CSI recognition code.

The pass sequence is:

1. Freeze the model, primitive contracts, observation specification, and
   parameter domain alongside the existing implementation snapshot.
2. Compute state, data, control, scheduler, and RNG dependencies to a fixed
   point across trial boundaries, including initial uncertainty.
3. Resolve observed events to execution facts. Preserve the distinction between
   an unknown event in simulation and its supplied value in history replay.
4. Check reconstruction of retained state and legality of deterministic path
   generation. Produce a witness for each removed stochastic dependency.
5. Construct a likelihood plan with history operations, stochastic regions,
   boundary values, scoring operations, runtime guards, and parallel axes.
6. Validate the plan independently of the discovery pass: every state read,
   write, scheduler effect, and cross-region input must be accounted for.
7. Lower its execution regions through existing KernelIR and component emitters.

First support one stochastic execution region and a reconstructible endpoint
per trial. Multiple stochastic nodes inside that region remain jointly
simulated; splitting trials does not justify splitting dependent nodes.
Arbitrary multiple stopping clocks or stochastic branches require further
rules and should receive specific diagnostics initially.

An evidence record should contain the source and observation identities,
referenced operation/state IDs, applicability guards, transformation rule IDs,
trusted contract versions, remaining obligations, and guarantee level. These
are checked compiler witnesses initially, not Lean certificates.

## Generated execution and performance

Separate history reconstruction from reusable within-trial paths:

```text
observations + inputs + candidate parameters
    -> sequential scan of observed history, parallel across candidates/subjects
    -> saved trial-start states
    -> deterministic path generation, parallel across trials/candidates/subjects
    -> stochastic trial simulation, additionally parallel across estimates
    -> observation counts/weights and log-score reduction
```

Replay the actual scheduler update/publication order, including the final
execution of persistent mechanisms on a terminating pass. Endpoint replay is
a distinct execution transformation; it cannot generally be implemented by
changing only the loop bound. Preserve initialization, reset state, recurrent
sender values, controls, and readout timing.

Initially materialize paths and sampled outputs for inspection. Then generate
fused simulation/scoring so fitting need not store every sampled response.
Path buffers represent generic values crossing the deterministic/stochastic
boundary: drift, bounds, inputs, or other coefficients as required by the graph.

For `P` candidates, `T` trials, `M` estimates, horizon `K`, and boundary width
`W`, the intended costs per subject are approximately:

- history reconstruction: proportional to `P * sum(observed trial steps)`;
- path generation/storage: `O(P*T*K*W)` without an estimate dimension;
- remaining stochastic work: `O(P*T*M*K)` worst case, with early termination.

This removes repeated deterministic work and per-trial particle resampling.
It does not remove the sampling cost or guarantee a speedup over the direct
numerical method. Keep a memory budget and chunk trials/candidates; continue
history state across chunks. An initial plan may combine scan and path
generation, but the separate scan enables parallel path generation and avoids
serially generating full unused tails for every trial.

Use semantic RNG addressing by subject, trial, estimate, component/stream,
and execution index. Preserve the declared common-random-number policy across
candidates. Chunking and launch geometry must not change the random experiment.
Different legacy RNG layouts need distributional comparison unless an explicit
replay mapping is provided; do not promise bitwise equality across backends.

Cache compiled programs by model/contract/observation identity and structural
settings. Recompute histories when parameters, inputs, or conditioning data
change. Changes to timestep or nondecision time can move the observed endpoint;
never reuse history merely because graph topology is unchanged.

## Public API and strategy selection

Proposed API shape, not currently executable:

```python
report = BatchedCompositionCompiler.diagnose_likelihood(
    composition, observations=observation_spec, parameter_domain=domain,
    strategy="auto", estimator=histogram_spec, backend="triton",
)
plan = BatchedCompositionCompiler.compile_likelihood(
    composition, observations=observation_spec, parameter_domain=domain,
    strategy="auto", estimator=histogram_spec, backend="triton",
)
scores = plan.log_likelihood(inputs=inputs, data=data,
                             parameter_sets=candidates, num_estimates=M)
```

The report should separate semantic eligibility, code-generation readiness,
and device availability, following existing capability diagnostics. Explain
why history is reconstructible or give the first blocking dependency path.

For this release, `auto` chooses between independently reset trials and the
generated deterministic-history plan. Otherwise return an unsupported-likelihood
diagnosis with the simulation capability and unresolved latent state. Do not
silently fall back to a product of unconditional trial marginals. A fallback
likelihood is eligible only when implemented for the same observation target
and permitted approximation policy.

PEC receives the observation specification and caches the chosen plan. Existing
explicit CSI flags initially remain compatibility adapters with their current
semantics labeled. Switch experiments to the new API only after differential
validation; changing observation assumptions must be a separate study decision.

Reports must distinguish:

- preservation of source execution under checked transformation assumptions;
- observation or endpoint approximations;
- finite-sample histogram, smoothing, pseudocount, and zero-probability handling;
- available gradient semantics and estimator guarantees.

An exact factorization does not make a histogram log-score exact, unbiased, or
suitable for pseudo-marginal MCMC. Even an unbiased density estimate does not
yield an unbiased log estimate. No such guarantee should be inferred by API
consumers from successful compilation.

## Implementation sequence and acceptance gates

| Change | Deliverable | Acceptance gate |
|---|---|---|
| 1. Contracts and diagnosis | Minimal observations, primitive semantic contracts, likelihood report, immutable plan records. | Renamed CSI and independent-reset examples are diagnosed from identities and semantics; rounded/missing endpoints and opaque effects produce specific reasons. |
| 2. Checked factorization | Conditioned dependency pass, reconstruction rules, plan validator, runtime guards. | CSI is admitted without model names; stochastic feedback, random initial state, and hidden stopping clocks are rejected; forged witnesses fail validation. |
| 3. Generic history execution | Generated endpoint replay and saved starts using existing scheduled component steps; CPU interpreter inspection path. | Trial starts, endpoints, phase counts, held values, and recurrent states agree with controlled source execution over multi-trial sequences. |
| 4. Generic stochastic execution | Generated boundary paths and parallel stochastic trials; PEC opt-in routing. | CSI scores/outcomes agree with the legacy specialization under matched semantics; a non-decision model uses the same passes. |
| 5. Performance and migration | Fused scoring, chunking, launch tuning, automatic selection, compatibility adapters. | Fused/materialized agreement, chunking invariance, explicit runtime/memory comparison, and no correctness regression; migrate CSI workflows. |
| 6. Retire duplicate production code | Remove legacy CSI kernels after retaining independent tests and reproducible baselines. | New production route contains no CSI-name checks, CSI drift transcription, or CSI-specific state layout. |

Changes 1–4 are the functional priority. Change 5 establishes whether the new
compiler-generated plan is an acceptable replacement for the current workload.
Lean integration is not an acceptance dependency for these changes.

## Correctness tests

Use several independent checks, because shared component code can share bugs.

- Structural/metamorphic: rename and reorder nodes, change valid projections,
  change registered drift functions, and vary deterministic subgraphs. Results
  must follow the supplied model rather than the old CSI formula. Use an
  identity-based UDF binding path where current name-based registration limits
  these tests.
- State and timing: first-trial initialization, trial resets, zero/one-step
  cases, overlapping onset/decision passes, output gates, held controls,
  parameter-dependent timing, support boundaries, and multi-subject boundaries.
- Negative cases: actual random persistent state, uncertain initial state,
  trajectory feedback, scheduler-only stochastic feedback unresolved by the
  observations, missing RT, ambiguous rounding, unsupported censoring, and
  runtime parameter rows that violate a specialization guard.
- Controlled replay: start the generic coupled simulator from a reconstructed
  state; compare its within-trial path and carried endpoint under controlled
  stopping schedules. Include LLVM/Python source execution on small cases.
- Probability oracle: use a tiny finite-state model with enumerable trial
  randomness to compare the entire sequence probability, including history
  conditioning. This catches errors that single-trial RT plots cannot.
- Non-decision admission: a deterministic adaptation state evolving over known
  trial durations, with a trial-reset noisy scalar readout. Start with an
  already-supported deterministic primitive; add a registered Gaussian
  readout if necessary. Verify known conditional distributions and histories.
- CSI regression: compare the legacy specialization, generic coupled execution,
  and generated likelihood under identical timestep, readout, observation,
  estimator, and truncation settings. Test both 10 ms and 1 ms configurations.
- Direct comparison: use the discrete endpoint oracle for matched-process
  comparisons where available. Treat continuous direct likelihood agreement
  as a separate refinement experiment, not finite-timestep equality.
- GPU checks: deterministic replay under a fixed RNG mapping, fused versus
  materialized scoring, chunk invariance, candidate ordering under the stated
  RNG policy, and uncertainty-aware distribution comparisons across seeds.

Run local analysis/interpreter tests first and small GPU validation next. Large
recovery studies are a later integration gate, after state and likelihood tests.

Benchmark compile time, warm objective time, history/path/sampling/reduction
time separately, peak device memory, score variability, and the amortization
point. Include the existing 11-candidate/100,000-estimate workload and both
timesteps on available hardware. A reasonable provisional migration target is
within 20% of the handwritten specialization's warm runtime at comparable peak
memory, with regressions investigated before retiring it; this is a target,
not an established result. Cluster jobs require a separate authorized run.

## Connection to broader likelihood compilation

Keep factorization independent of the numerical method used for each factor.
The generated history plan can eventually feed a sampling operator, registered
exact distribution, discrete recurrence, or continuous equation solver. Each
operator must declare its source semantics, observation measure, parameter
domain, approximation/error properties, and gradient support.

After the generic history route works:

1. Add a small exact likelihood operator to exercise interchangeable factor
   evaluators without changing history analysis.
2. Generalize residual-state inference with explicit transition/observation
   kernels and estimator guarantees; adapt the experimental conditioned path
   only where those contracts apply.
3. Expose compiler facts and typed obligations to equation/LLM synthesis.
   Proposed factorizations must pass the same deterministic validator.
4. Formalize the history-factorization rule and its IR semantics in Lean. The
   existing witness format supplies theorem inputs and documents trusted
   translation assumptions.
5. Add equation and numerical certification as separate guarantees. A proof
   of factorization does not certify a PDE discretization or a Triton kernel.

The first release milestone will be automatic, checked reconstruction and
parallel likelihood compilation for CSI and a non-decision example, with
explicit rejection when the observations do not determine the required history.
