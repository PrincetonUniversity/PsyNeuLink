The particle-filter approach can be added to LLVM/PEC independently of
`feat/likelihood_compile`. The appropriate unit of work is a general resumable
LLVM simulation interface plus a sequential PEC likelihood. The existing LLVM
prototype is useful scaffolding, but it is not ready to extract unchanged.

This assessment examines feature head `0fdfaa52be` and the locally available
`origin/devel` snapshot `5cf4270d70`. The local `devel` branch is older; start a
port from the reviewed upstream snapshot or a subsequently checked upstream
head. No production code was changed, and no branches were merged.

**The statistical change**

Ordinary PEC runs each estimate through the complete input sequence, then
scores each observed trial against that trial's simulated outcomes. Persistent
model state advances, but observed history never changes the population of
states. Multiplying those trial marginals generally does not give the sequence
likelihood when unobserved stochastic state persists between trials.

The desired factorization is

```text
p(y[1:T] | theta, inputs)
    = product_t p(y[t] | y[1:t-1], theta, inputs[1:t])
```

At each trial, simulate every particle from its current state, evaluate the
observation kernel for each simulated outcome, estimate the predictive density,
and resample the terminal states using those same weights. Continue from the
resampled states. With resampling every trial, particles enter the next trial
with equal weights. A later ESS-based resampling policy would also need to
carry unequal weights between trials.

For continuous outcomes, histogram bins or a smoothing kernel define an
observation approximation. Filtering fixes conditioning on history; it does
not make that approximation the exact likelihood of the original continuous
model. A scalar KDE density alone is insufficient: ancestry requires each
particle's contribution to the same observation model used for scoring.

**What can be reused**

The relevant prototype originated in `ad4fe14f21` ("Condition CSI simulator
likelihood on observed history"). That commit also changes the batched compiler
and model scripts, so extract selected changes instead of cherry-picking the
whole commit.

| Existing piece | Use in the independent port |
| --- | --- |
| `CompExecution.thread_evaluate_stateful()` in [execution.py](psyneulink/core/llvm/execution.py#L831) | Per-particle LLVM state/data buffers and threaded single-trial evaluation. The prototype adds 207 lines to this file, including resampling support. |
| `retain_simulation_state` in [optimizationcontrolmechanism.py](psyneulink/core/components/mechanisms/modulatory/control/optimizationcontrolmechanism.py#L3401) | Per-lane structure addressing and terminal-state publication. Correct the publication implementation. |
| `_resample_stateful_structures()` in [execution.py](psyneulink/core/llvm/execution.py#L49) | Gather model state and recurrent output data by ancestry while preserving destination RNG streams. It walks component layout identifiers rather than model-specific offsets. |
| `_llvm_conditioned_objective_func()` in [fitfunctions.py](psyneulink/core/components/functions/nonstateful/fitfunctions.py#L1433) | Basic PEC trial loop and cleanup pattern. Replace its older observation semantics and repair input/seed handling. |
| `BatchedSimulationPlan.conditioned_log_likelihood()` in [compiler.py](psyneulink/core/batched/compiler.py#L539) | Reference for the corrected observation semantics, systematic resampling, masks, zero-support checks, and diagnostics. Reimplement the small CPU algorithm without importing the batched compiler. |
| [test_batched_conditioned_likelihood.py](tests/composition/pec/test_batched_conditioned_likelihood.py) | Exact finite-state filtering examples and normalized observation-kernel tests to adapt into backend-independent tests. |

**Problems reproduced in the LLVM prototype**

The local audit uses a noisy `SimpleIntegrator` inside a PEC, three trials,
four particles, and one LLVM execution thread. For continuation checks it
substitutes neutral observation weights and identity ancestry, so differences
cannot be explained by conditioning or sampling different ancestors.

1. **Terminal state is not actually written back on the tested LLVM setup.**
   With `Never()` resets, uninterrupted execution produces approximately
   `[0.998082, 1.998645, 2.989625]` for particle 0. Split execution produces
   `[0.998082, 0.998082, 0.998082]`, and every returned integrator state stays
   zero. The publication calls at
   [optimizationcontrolmechanism.py:3556](psyneulink/core/components/mechanisms/modulatory/control/optimizationcontrolmechanism.py#L3556)
   use `helpers.memcpy`, which obtains its size from `llvm.objectsize` on the
   destination. These destinations are caller-owned pointers; the helper
   requests zero when their size is unknown. The usual copy *into* a local
   allocation does not have this problem. Replacing just these write-backs
   in memory with typed LLVM load/store operations restores exact split/full
   agreement for this model. The upstream snapshot's helper still queries the
   destination this way, despite having replaced its handwritten copy loop.

2. **Correcting write-back does not preserve run/scheduler semantics.**
   With that temporary correction and `AtTrial(0)` resets, the uninterrupted
   sequence remains `[0.998082, 1.998645, 2.989625]`, but split execution gives
   `[0.998082, 1.000563, 0.990980]`. Every resumed launch resets the integrator
   again. [gen_composition_run](psyneulink/core/llvm/codegen.py#L1165)
   resets RUN execution counts and allocates fresh scheduler metadata on each
   call. State/data transport alone is not a complete continuation contract.

3. **`depends_on` fails when assigning parameters after slicing inputs.**
   Three observations with conditions `a, b, a` raise `IndexError`: the input
   has one row, but the condition mask has three. The prototype calls
   `set_parameters_in_inputs()` on each one-row slice while that method uses
   full-sequence masks. Prepare all fitted parameter columns for the complete
   candidate first, then slice the prepared rows for execution.

4. **Zero observation support still returns a finite score.**
   Putting all three observations inside `[100, 101]` while simulated outcomes
   are near `[1, 2, 3]` returns `-69.07755275`, approximately `3*log(1e-10)`.
   The prototype floors the density and substitutes uniform ancestry. This
   continues without a supported posterior. The corrected path should stop
   with a clear zero-support diagnostic, or score a deliberately specified
   contamination model. An optimizer can explicitly map a support failure to
   an invalid candidate; it must not silently invent filtered ancestry.

5. **Its smoothing convention is not the corrected observation kernel.**
   LLVM calls `histogram_observation_weights()` with the default
   `source_normalized=False`. In a five-bin boundary example, integrating the
   observation density yields `0.94307547`; source normalization yields
   `0.99999999`. Use normalization around each simulated source bin, preserve
   overflow mass, and validate every assimilated observation. The prototype
   also rejects positive pseudocounts with a rationale superseded by the
   branch's explicit contamination interpretation.

6. **The fresh-seed policy is bypassed.**
   Two sequential objective evaluations produce identical simulations with
   `same_seed_for_all_parameter_combinations=False`. The sequential loop calls
   `_evaluate()` directly and omits the seed-grid update in `_run_simulations()`.
   Fresh streams must be selected between complete candidate evaluations;
   changing seeds between trials would restart trajectories instead.

The existing two LLVM-conditioned tests pass despite these failures: one checks
finite/repeatable scores on a trial-resetting DDM, and the other checks a small
synthetic structured-array gather. Neither establishes simulation continuation
or likelihood correctness.

With the temporary write-back correction, an additional compiled check forces
every particle to inherit particle 0's state. The next trial's offspring have
different outcomes (range `0.02357824`), consistent with the helper preserving
their distinct advanced RNG streams in this model. This supports reusing the
ancestry helper, while broader nested-component coverage remains necessary.

**Recommended implementation boundary**

1. **A reusable LLVM particle execution session.** Prepare the binary, canonical
   inputs, parameter structures, RNG layout, worker pool, and initial particle
   buffers once per evaluation/session. Provide an explicit trial-advance
   operation with caller-owned continuation buffers. Preserve integrator state,
   recurrent/output data, held controller values, sampled ParameterPort state,
   scheduler metadata, and relevant execution counters. Initialize a run once;
   resume it without repeating run initialization. Traverse nested compositions.
   Use typed copies or explicit ABI sizes for caller-owned structures. Ordinary
   run/evaluate behavior can retain its existing interface.

   The prototype's controller flags can bootstrap this work, but an explicit
   session avoids hidden shared mutable state and repeated setup. Share only
   immutable parameters. Any supported component that mutates additional
   parameter storage needs that storage included in continuation or an explicit
   unsupported-mode check. "General" should mean any composition satisfying the
   LLVM continuation contract, with no CSI/DAWA recognition.

2. **A small CPU observation/filter layer.** Implement a NumPy reference for
   normalized categorical/continuous histogram observation contributions,
   predictive density, systematic resampling, and diagnostics. Give it a
   backend-independent interface returning per-particle weights and the
   corresponding predictive density. The state gather follows ancestry while
   advanced RNG streams remain attached to destination particles, allowing
   duplicate offspring to diverge on subsequent trials.

   Match the newer branch semantics: observations excluded from the score still
   condition history; zero support is explicit; smoothing is source-normalized;
   simulated overflow probability is retained; and positive contamination
   affects both weights and score. For `N` particles, `K` joint histogram cells,
   cell volume `V`, contributions `q[i]`, and per-cell pseudocount `alpha`:

   ```text
   resampling weights[i] = q[i] + alpha/N
   predictive density  = (sum(q) + alpha) / ((N + K*alpha)*V)
   contamination probability = K*alpha / (N + K*alpha)
   ```

   Prefer an explicit contamination probability in a new public interface, or
   document that `alpha/N` must stay fixed when changing particle budgets.
   Otherwise particle-budget comparisons also change the observation model.
   Keep kernel configuration independent of Triton and `batched_*` options.

3. **PEC integration and regression coverage.** Route both optimization and
   `pec.log_likelihood()` through the same sequential scorer. Normalize all
   supported input forms and assign condition-specific parameters before
   slicing. Start a fresh population for each candidate/subject, respect PEC's
   seed policy, and restore state on exceptions. Return per-trial densities,
   ESS, and support diagnostics through a defined optional result interface.
   If simulated outcomes are exposed, label them as predictive particle clouds;
   fixed particle indices after resampling are not uninterrupted trajectories.

   An explicit conditioned-likelihood option is the conservative initial API.
   Changing the default for all existing PECs is a separate compatibility
   decision: independent-trial KDE fits need not change, and automatically
   proving when history conditioning is necessary requires additional analysis.
   The current upstream hierarchical wrapper already calls each subject PEC's
   public `log_likelihood()`, so that integration point should be retained and
   tested. Dask workers must build the same configured scorer and keep each
   subject's sequential filter together.

The first two stages should have correctness tests before optimizer plumbing.
Keeping observation weighting and resampling on the CPU initially is compatible
with LLVM-compiled simulation; compiling every filtering operation into LLVM
is optional optimization work. The main costs to measure are simulation time,
per-trial dispatch, and copying complete particle structures, including RNG
storage. The branch's GPU timings do not predict CPU LLVM performance.

**Related fixes and extraction scope**

`fix/llvm-reset-sampled-modulation` (`926e2a9b46`) separately fixes transfer resets
to use previously sampled modulation. It is relevant to models such as DAWA
and should precede claims of Python/LLVM parity for controlled resets. The
continuation format must include its additional ParameterPort state.

`fix/pec-independent-noise-streams` contains a separate correction for identical
seeds across different stochastic components. Correct ancestry RNG handling
does not repair correlated component streams. Coordinate with those small
fixes rather than merging the feature branch or treating either as a substitute
for particle filtering. Structural scheduler condition selection is already
present in the reviewed upstream snapshot via `#3661`; that change does not
solve continuation of scheduler clocks.

The core port should touch LLVM execution/code generation, OCM evaluation,
PEC optimization/input handling, one small observation/filter module, and
focused tests/documentation. It does not require the batched IR, Triton runtime,
likelihood planner, model-specific direct likelihoods, adaptive fitting, or
benchmark drivers. Preserve newer upstream hierarchical/Dask work when adapting
the PEC changes rather than replacing those files with feature-branch versions.

**Acceptance checks**

- Exact split-versus-uninterrupted simulations with identity ancestry: persistent
  stochastic integrators, recurrent outputs, nested compositions, trial-indexed
  resets, RUN counters, and held/sample-time modulation.
- Actual compiled duplicate-ancestor simulations: model states follow parents,
  destination RNG streams continue independently, and thread count does not
  change seeded results.
- Exact finite-state posterior/predictive examples and a linear-Gaussian model
  with a specified observation kernel checked against a Kalman-filter reference.
  Allow Monte Carlo error where appropriate; seeded replay alone is insufficient.
- An earlier observation changes later filtered predictions, including when
  that earlier observation is excluded from the accumulated score.
- Observation normalization, multiple outcomes, boundaries, overflow, explicit
  contamination, zero support, and very small positive densities.
- `depends_on`, node/InputPort/composition-keyed inputs, ragged node widths,
  alternate outcomes, empty/invalid sequences, and caller-input immutability.
- Common-random-number replay, fresh-stream mode, candidate and subject isolation,
  cleanup after errors, public scoring, and at least one optimizer evaluation.
- Existing ordinary LLVM PEC tests plus upstream hierarchical/Dask integration
  tests, with both supported floating-point configurations for the runtime work.

**Evidence and reproducibility**

The existing LLVM-conditioned tests passed **2/2**. The CPU observation/ancestry
suite passed **21/21**, with its two GPU tests deselected. These were run in the
workspace virtual environment with Python 3.13.3, NumPy 2.3.5, and llvmlite 0.48.0.
The full repository suite and GPU execution were not run for this assessment.

The supplementary audit is [llvm_particle_filter_audit.py](/tmp/llvm_particle_filter_audit.py),
with [results](/tmp/llvm_particle_filter_audit.json) and
[output](/tmp/llvm_particle_filter_audit.log). Its write-back workaround and
neutral observation kernels are temporary in-process patches, not source edits.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/composition/pec/test_parameterestimationcomposition.py \
  -k conditioned_llvm -o addopts='' -q --tb=short

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/composition/pec/test_batched_conditioned_likelihood.py \
  -m 'not triton_gpu' -o addopts='' -q --tb=short --disable-warnings

PYTHONPATH=/home/david/princeton/PsyNeuLink \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python /tmp/llvm_particle_filter_audit.py
```
