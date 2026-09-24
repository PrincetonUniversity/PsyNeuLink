# DAWA model experiments

This directory groups DAWA's source model, simulation drivers, direct
likelihoods, and recovery studies. Shared compiler benchmarks and design notes
are in the [parent directory](../README.md).

The DAWA LC/LCA network, including Gaussian noise in all four LCAs, can use the
ordinary batched compiler and PEC simulation objective on Triton.
`dawa_batched_simulation.py` loads the local
`dawa_lca_model/full_lca_model_lc.py`. The
[source model, original fitting scripts, and Slurm examples](dawa_lca_model/README.md)
are tracked; subject data and generated outputs remain ignored.
All three `flanker_fit_lc_part*.py` scripts use this same model builder.

The compiler additions cover scheduled Logistic LCAs of width 1–32, finite dense
recurrent matrices, scalar numeric or `NormalDist` noise, maximum-activity
termination, standard decision index/time/step and energy outputs, scalar Euler
FitzHugh–Nagumo integration in a Linear TransferMechanism, and elementwise
ObjectiveMechanisms. Scalar OVERRIDE controllers can fan out to multiple
registered parameters in a dynamic schedule. Both held control values and the
values last sampled by each target are represented explicitly; trial resets use
the latter. These are component and scheduler features, with no DAWA-specific
kernel or compiler graph recognizer.

Recurrent scheduling is now the default in both the shared model builder and
the driver. All four LCAs and the four weighted processing mechanisms use
`Always()`, so they keep advancing through the graph's existing execution order
until the response reaches threshold. Bias/weight controllers still use
`AtPass(0)` and hold their values throughout the trial; output gates still use
`WhenFinished(responseLayer)`. Native Python, LLVM, and the fitting scripts
inherit the fix directly from the shared builder.

Previously, the processing nodes inherited `EveryNCalls` dependencies on the
once-per-trial controllers. The stimulus, decision, and response LCAs could
therefore execute only once, stalling trials that needed further integration.
The driver also applies the fix to older/custom builders by default. Its
optional `--schedule source` setting uses the loaded builder's own conditions;
the bundled builder already contains the fix, so this option also runs
recurrently with the bundled model.

Run a stochastic simulation and evaluate two candidates through PEC:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py \
  --trials 4 --estimates 64 --max-steps 500 \
  --pec-smoke --output /tmp/dawa_samples.npz
```

The sample array has axes `[candidate, subject, trial, estimate, outcome]`, with
decision and response time as its final two columns. PEC smoke testing includes
all seven original fitting parameters and conditional parameters for subject
and previous congruency, matching the supplied fit scripts (12 coordinates for
the example's two subjects). It evaluates the existing histogram simulation
objective; it does not run an optimizer or infer an analytic likelihood.

Check deterministic results against native LLVM (or use `--reference python`):

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py \
  --deterministic --reference llvm \
  --trials 4 --estimates 3 --max-steps 500
```

The local RTX 2080 Ti audit matched LLVM choices and response times, with maximum
absolute error below `3e-8`. A separate Python comparison of all four LCA layers
and LC activity across four trials also passed. A warm
stochastic run of two candidates × four trials × 64 estimates took about 30 ms
on this machine; this small workload measurement includes host preparation and
result transfer and is not a fitting-scale throughput claim.

To use these features directly:

```python
from psyneulink.core.batched import BatchedCompositionCompiler

plan = BatchedCompositionCompiler.compile(model, backend="triton", max_steps=2000)
samples = plan.run(inputs, parameter_sets, num_estimates=1000,
                   seed=29, strict_truncation=True)
```

Existing PEC workflows select
`PECOptimizationFunction(..., batched_backend="triton", batched_max_steps=2000)`.
The shared builder supports repeated integration at positive fitting thresholds
by default. Keep strict truncation checks enabled during simulation audits.

The new LCA configurations require `execute_until_finished=False`; the existing
CSI run-to-completion path is retained. Gaussian scheduled LCAs support
`AtTrialStart` and `Never` resets, and the FHN adapter supports scalar Euler
integration with zero initializers and fixed per-pass internal execution counts. Custom LCA
matrices are frozen; scalar competition/self-excitation overrides on those
matrices are rejected. GPU random streams reproduce seeded GPU runs, not
NumPy/LLVM draws. Networks with scalar OVERRIDE controls must run each subject's
complete sequence in one call: `initial_states` does not yet restore held and
sampled control values, so resuming these networks is rejected explicitly.

## Noise in all four LCAs

The source builder and run helper expose `c_noise`, `s_noise`, `d_noise`, and
`r_noise` as zero-mean Gaussian standard deviations. The batch driver accepts
the corresponding `--c-noise`, `--s-noise`, `--d-noise`, and `--r-noise` flags;
`--deterministic` disables all four. Existing defaults are preserved (the batch
driver enables only response noise). An audit used standard deviation 0.1 in
each layer and the original 0.01-second integration step. Each integration
adds a draw scaled by `sqrt(dt)`, so these are diffusion amplitudes, not constant
additive inputs. Native Python PNL completed four trials with noise in all four
LCAs, producing finite choices and response times; each trial took multiple
integration passes (49–165 response steps in the audit).

To run the full-noise composition and evaluate the conditional PEC objective:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py \
  --c-noise .1 --s-noise .1 --d-noise .1 --r-noise .1 \
  --trials 4 --estimates 256 --max-steps 2000 --pec-smoke
```

Control uses `Never` and carries its state across trials; the other three LCAs
use `AtTrialStart`. Noise settings preserve those reset policies. The previous
compiler restriction on persistent Gaussian LCAs has been removed with explicit
support for their initial state.

PNL samples the initial recurrent **RESULT port** during construction; its value
can differ from the mechanism's separately initialized value. Native PEC copies
this constructed activity into each estimate. The compiler freezes that same
RESULT default in the plan and uses it before any processing executes, while
the integrated state starts at zero. Runtime gain/noise changes affect subsequent
steps, without resampling or transforming the constructed activity. This uses
the general `StateDecl.initial_attribute` facility for frozen vector initializers.
`AtTrialStart` LCAs retain their original reset behavior.

The existing random-stream allocation then supplies independent integration
draws by accumulator, estimate and trial, with common random numbers across
parameter candidates by default. Each estimate retains its own control state.
The original nonlinear Logistic dynamics and scheduler are unchanged. The
compiler starts a fresh sequence from the model's construction defaults; it
does not implicitly import a live composition's current state.

The all-four-noise configuration passed a GPU audit on the RTX 2080 Ti: two
candidates, four trials, 256 estimates, no truncation, exact seeded replay, and
finite conditional PEC scores. Component tests cover constructed activity,
runtime parameter changes, analytic noise moments and cross-trial covariance,
stream independence, and replay on both GPU and the Triton CPU interpreter.

The [benchmark results](dawa_benchmark_results.md) distinguish the original
response-only measurements from the full-noise configuration.
For a comparison with independent noise in all four LCAs, pass
`--independent-noise-streams` to `dawa_llvm_benchmark.py`. Native PEC normally
broadcasts the same seed to all random variables in an estimate, which can
correlate the LCAs' draws. The option applies distinct seed offsets to LLVM's
randomization projections; Triton already separates component streams. This is
an explicit benchmark configuration, not a change to the original fitting scripts.

## Benchmarks and direct likelihoods

For fitting-scale performance against PEC's threaded LLVM simulation path,
see [the benchmark results](dawa_benchmark_results.md) and
[reusable benchmark driver](dawa_llvm_benchmark.py). Measurements include
1,000 and 10,000 estimates over both 64-trial slices and a full 760-trial subject,
plus 100,000 estimates over 64 trials with noise in all LCAs. Setup and likelihood
scoring are separated from simulation timing.

The [full-subject objective benchmark](dawa_benchmark_results.md#full-subject-pec-objective-at-100000-estimates)
also evaluates four distinct candidates at 100,000 estimates over all 760 trials,
including conditional parameters and GPU histogram scoring. The
[profiling follow-up](dawa_benchmark_results.md#gpu-profiling-and-count-only-scoring)
adds general count-only scoring and measures about 2.5 seconds per candidate
with 32-lane/one-warp launches on the 2080 Ti. This implies approximately 3.5
hours for 5,000 evaluations before optimizer overhead. Live Torch fitting
buffers fall from 3.1 GiB per candidate to 0.23 MiB for four candidates together;
CUDA context/code and allocator reservations are additional. These are
throughput extrapolations; a complete optimizer run has not been measured.

A [differentiable direct-likelihood prototype](dawa_likelihood/README.md) is also
available. It propagates the joint response-state distribution and supports
gradients through all seven fitting parameters in its RT observation model.
The documentation distinguishes direct stopping-step scoring from the
empirical-RT mode's fixed history approximation and records validation results.

The separate [continuous-time DAWA likelihood](dawa_likelihood/CONTINUOUS_README.md)
uses coupled ODE dynamics and a two-dimensional absorbing Fokker–Planck solver.
It scores RT intervals without added measurement noise, differentiates all seven
parameters, and updates history using candidate-dependent decision durations.
Its instantaneous gain modulation defines a continuous extension of the source
model; validation uses an independent continuous SDE sampler rather than expecting
parity with the original 10 ms scheduler.

A [source-convergence and synthetic-recovery study](dawa_likelihood/STUDY_README.md)
now checks that connection explicitly, using 100,000 estimates per case and
joint refinement of the original LCA and LC time steps at a fixed clock ratio.

The [CPU and GPU performance audit](dawa_likelihood/PERFORMANCE_README.md) compares
threaded CPU and fused GPU direct likelihoods, including all parameter gradients,
against continuous GPU sampling. Both backends preserve sequential control-state
history. Run
[dawa_continuous_benchmark.py](dawa_continuous_benchmark.py) to reproduce the
two-condition workload, including 100,000 GPU estimates per condition.
