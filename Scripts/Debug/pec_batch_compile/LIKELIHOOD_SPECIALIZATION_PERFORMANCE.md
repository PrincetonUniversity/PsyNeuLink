# CSI handwritten versus compiler-generated specialization

Measured locally on 2026-09-09, WSL, NVIDIA GeForce RTX 2080 Ti (11 GiB),
at compiler commit `1b7d4ee7bd` for the baseline; the optimization
follow-ups below measure subsequent working-tree changes. This is a fitting-scale **end-to-end objective
benchmark**, not an isolated GPU kernel comparison or a complete optimization.
No cluster jobs were submitted. The baseline changed no core implementation;
the follow-ups successively optimize endpoint reconstruction, fused scoring,
and estimator-aware execution.

## Workload and comparison contract

- Subject 1's complete chronological task/stimulus sequence from
  `csi_fit/data fitting/data_to_fit_study3.csv`: 561 trials across NoInstruction,
  RealRare, and RealFrequent. All trials condition history; 485 contribute scores.
- **11 parameter candidates, 100,000 simulations per trial**, matching the
  population batch and simulation budget in
  `csi_fit/data fitting/csi_gpu1ms_comprehensive_recovery_gb300.slurm`.
- Both LCA and DDM use the indicated timestep; maximum duration is 12 seconds.
  LCA leak 12, competition 3, noise 0; DDM noise 0.1; ITI 1 second and switch
  cue interval 80 ms. Eleven nearby interior candidates vary condition-specific
  gain, threshold, collapse rate, and nondecision time. Physical durations and
  collapse rates are preserved when changing timestep.
- Both routes use the same joint response/RT histogram estimator: 100 bins,
  Gaussian smoothing sigma 0.5 bins, pseudocount 0.1, two response categories,
  and the same score-inclusion mask.
- Observed responses/RTs are **source-simulated on the real trial sequence**,
  not the participant's recorded outcomes. The generated compiler currently
  requires compatible event counts; arbitrary recorded RTs need an explicit
  recording/rounding contract. We did not silently round the participant data.
  Synthetic mean RT is 0.810 s at 10 ms and 0.769 s at 1 ms; the actual mean
  is 0.775 s. Candidate NDT shifts are whole timestep multiples.
- CRN is enabled with seed 17. The two implementations use different RNG
  mappings, so equal seeds do not mean identical draws. This benchmark does
  not establish statistical likelihood agreement. Each method's warm repeats
  exactly reproduced its own scores.

The generated route uses a benchmark-only adapter from device observation
samples to the existing histogram estimator. This does **not** add histogram
semantics to its public `compile_empirical_mass()` API, which has a different
statistical target.

## Warm objective times

Seconds per complete 11-candidate subject batch; median of two synchronized
warm runs after an excluded first call. Data generation is excluded.

| Implementation | 10 ms | 1 ms |
|---|---:|---:|
| Handwritten CSI: production fused fitting path | 0.831 | 7.531 |
| Handwritten CSI: strict full-endpoint sampling, fused scoring | 1.604 | 16.018 |
| Generated specialization + shared histogram adapter | 43.631 | 244.095 |
| Generated / handwritten production | 52.5x | 32.4x |
| Generated / handwritten strict | 27.2x | 15.2x |

Individual warm measurements were 0.828/0.833, 1.600/1.608, and 44.235/43.028
seconds at 10 ms; 7.507/7.555, 15.978/16.059, and 244.313/243.877 seconds at
1 ms, respectively. First-call times were 1.295, 1.853, 47.587 seconds at
10 ms and 7.698, 16.104, 245.607 seconds at 1 ms. These are not clean-cache
compilation timings.

Production CSI uses exact observation-window stopping for its histogram score.
The strict control disables that shortcut and runs until full decision endpoints
(or the cap). The production and strict handwritten scores were **exactly
equal** for all eleven candidates at both timesteps. Both handwritten modes
use fused histogram accumulation, not the older materializing debug path.

The handwritten route uses its tuned block size 32 / one warp; generated
sampling uses its current fixed block size 128 / four warps. The handwritten
route samples only the 485 scored trials after reconstructing all 561 trial
histories; the generated sampler currently samples all 561. Thus even the
strict control is not a pure instruction-for-instruction kernel comparison.

## Memory and batching

The generated full population would materialize 617,100,000 sample lanes.
Its observation, status, and diagnostic device arrays alone need **16.09 GiB**,
excluding paths and reduction temporaries. Its current conservative combined
host/device sample-budget lower bound is 35.06 GiB. It cannot run as one batch
on this 2080 Ti.

We therefore ran the generated route as eleven single-candidate microbatches,
each retaining the complete chronological sequence. CRN makes this candidate
partitioning valid; no trial history is split or restarted. The handwritten
route evaluates all eleven candidates together.

Peak Torch allocated device memory was 2.718 GiB for the microbatched generated
route at either timestep, versus 0.028 GiB / 0.277 GiB for handwritten CSI at
10 ms / 1 ms. These exclude CUDA context, display, allocator-reserved memory,
and host arrays; they are not total process/system memory measurements.

## Baseline interpretation and next steps

The generated specialization currently supplies the generic correctness
machinery, but is not a competitive replacement for the tuned CSI objective
at real fitting budgets. The small prior GPU validation did not establish
production performance.

The histogram adapter itself takes only about 0.32 seconds per eleven-candidate
batch at either timestep. Almost all generated time is in endpoint checks,
history/path preparation, sampling, transfers, and validation.

A separate synchronized phase-attribution run retained all 561 trials and
100,000 simulations, but used one candidate and one warm repeat:

| Generated phase, seconds per candidate | 10 ms | 1 ms |
|---|---:|---:|
| CPU endpoint-count inversion | 1.964 | 18.179 |
| History execution wrapper | 0.163 | 0.722 |
| Boundary-path execution wrapper | 0.011 | 0.021 |
| Remaining preparation, sampling, transfers, and validation | 1.721 | 3.328 |
| Histogram reduction | 0.030 | 0.030 |
| Total | 3.888 | 22.281 |

The wrapper measurements include preparation/transfers in those functions;
the residual is not an isolated sampling-kernel time. Extra synchronizations
are enabled only for this diagnostic run, not the headline comparison.
CPU endpoint inversion accounts for approximately **51% at 10 ms and 82%
at 1 ms**. The current implementation in `endpoints.py` scans the entire
1,200/12,000-count domain in 256-element chunks for each candidate/trial.
This is a measured host-side bottleneck, not an inference from GPU utilization.
Removing it alone would still leave a substantial gap; the remaining work
needs its own optimization and subsequent measurement.

Priority improvements, retaining the compiler's checked semantics:

1. Replace exhaustive endpoint-count enumeration with a fast checked inverse
   for supported readouts, retaining forward roundoff-envelope verification
   and uniqueness/incompatibility rejection. Vectorize candidate/trial work.
2. Compile fused estimator accumulation and compact diagnostic reductions,
   avoiding sample-sized observation/status transfers and the full-population
   memory barrier. Keep estimator/recording semantics explicit in the API.
3. Propagate the score mask and compile safe observation-window stopping where
   the estimator admits it, while preserving history for every trial.
4. Retain history/path intermediates on device, prepare populations together,
   and tune launch geometry after profiling. Candidate chunking should become
   a supported memory-planning decision rather than a benchmark adapter.

These are optimization targets, not measured speedups. Repeat this full-budget
benchmark and the source-equivalence tests after each change. Timings from one
sequence and an interior candidate neighborhood do not characterize difficult
parameter tails, all subjects, full optimizer convergence, or GB300 performance.

## Reproduce

```bash
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py

# Separate phase attribution, full sequence and 100k simulations, one candidate:
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --candidates 1 --methods generated_histogram --repeats 1 --profile-generated
```

The script prints JSON records to stdout and progress to stderr; it does not
save JSON files. Runs were sequential on a display-attached GPU. Two repeats
give a useful magnitude comparison, not a precise confidence interval.

## Follow-up: generic checked endpoint inversion

Implemented batched interval inversion in `psyneulink/core/batched/endpoints.py`.
The compiler evaluates conservative ranges through the original readout
expression, prunes impossible count intervals, and checks individual survivors
with the unchanged point-enclosure evaluator. It does not hard-code CSI,
reassociate the expression into a different floating-point formula, or change
the observation/recording model.

Before pruning it checks arithmetic safety across the complete count range.
Uncertified ranges and excessive subdivision fall back to the original
enumeration, also available explicitly as `reconstruct(..., method="exhaustive")`.
Candidate/trial inputs and parameters are prepared as arrays for batched
inversion. Source-witness validation and incompatible/ambiguous count rejection
remain active.

Same GPU, sequence, candidates, seed, estimator, 100,000 simulations per trial,
and candidate microbatching as the baseline. Again, synthetic source-lattice
outcomes on the real task sequence, not a fit to recorded participant RTs.
Median of two warm objective runs after an excluded first call:

| End-to-end objective, seconds per 11-candidate batch | 10 ms | 1 ms |
|---|---:|---:|
| Generated, before | 43.631 | 244.095 |
| Generated, after endpoint optimization | 18.489 | 44.329 |
| Measured speedup | 2.36x | 5.51x |
| Handwritten production CSI, rerun control | 0.866 | 7.480 |

Generated warm runs were 18.721/18.256 s and 44.031/44.627 s; first calls were
19.867 s and 44.017 s. All eleven generated scores at each timestep **exactly
match the baseline generated scores**, and each warm repeat matches its own
first call. The handwritten control scores also match their baseline exactly.
This is before/after reproducibility, not matched RNG draws between handwritten
and generated implementations.

The generated route remains approximately 21.3x / 5.9x slower than the rerun
handwritten production objective. Peak allocated GPU memory remains 2.718 GiB;
no sampling, scoring, launch, transfer, or buffer-allocation optimization was
included in this change. Fused generic scoring remains the next milestone.

Separate full-sequence, one-candidate verification compared **all 561 stopping
counts at each timestep** with exhaustive reconstruction: both matched exactly.
In the same process, inversion took 0.024 s versus 1.939 s at 10 ms, and
0.028 s versus 18.412 s at 1 ms. These are single diagnostic calls, outside
the objective timing loops.

The subsequent synchronized one-candidate warm profile measured endpoint
reconstruction at 0.023 s / 0.026 s, out of 1.549 s / 3.966 s total objective
time at 10 ms / 1 ms. It is now under 2% / 1% of that diagnostic runtime.
History wrappers took 0.065 s / 0.441 s, boundary wrappers 0.010 s / 0.024 s,
and histogram reduction 0.030 s / 0.032 s. The residual still combines
sampling, preparation, transfers, and validation; it is not isolated kernel
execution. No performance tests ran concurrently with the fitting-scale
benchmark or this profile.

Validation:

- 20 new inversion tests: randomized signed affine readouts, bounds and adjacent
  floats, incompatible/ambiguous events, whole-domain overflow/subnormal guards,
  cancellation, bounded fallback work, mixed safe/unsafe lanes, and the maximum
  supported count cap. An ordinary public-path test forbids exhaustive fallback.
- 52 existing endpoint/history tests passed with the Triton interpreter,
  including renamed models, altered gates, trial-varying parameters, resets,
  held controls, and endpoint scheduling.
- GPU regression selection: 11 passed, four backend-specific skips.
- Larger GPU source-equivalence check: 786,624 sample lanes across both timesteps,
  two seeds, and both CRN policies; zero count, response, or empirical-hit
  mismatches. Timing from this correctness check is not used above.

Reproduce the follow-up timing and the separate endpoint/profile diagnostic:

```bash
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --methods specialized_fit generated_histogram

env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --candidates 1 --methods generated_histogram --repeats 1 \
  --profile-generated --verify-endpoints
```

## Follow-up: generic fused scoring and memory-aware execution

The checked observation sampler now supports compact GPU reductions for two
distinct, explicit estimators: count-domain empirical mass and a histogram
density surrogate. The histogram tier supports one numeric dimension and
optional categorical dimensions. It reuses registered primitive/RNG/readout
implementations, with no CSI name or equation matching. Sample inspection
remains available as an oracle. No continuous-model, recording, or LLVM
likelihood semantics were changed.

Generated programs reduce integer successes/bin counts and nonfinite/truncation
diagnostics inside each estimate tile. Smoothing and normalization happen after
all simulation chunks accumulate. Candidate batches are chosen from a conservative
path/workspace memory budget; optional candidate and estimate chunk limits retain
global RNG identities under both CRN policies. Paths are prepared once per
candidate batch and reused across its estimate chunks. The existing public
history/path interfaces still transfer intermediate arrays through the host.

The same full fitting workload was rerun sequentially on the 2080 Ti: 561 trials,
485 scored, 11 candidates, 100,000 simulations per trial, 12-second cap, exact
source-simulated observations on the real sequence, and the same histogram
settings. Medians of two warm runs, excluding the first call:

| End-to-end seconds per full candidate batch | 10 ms | 1 ms |
|---|---:|---:|
| Materialized generated sampler + histogram adapter, rerun | 18.284 | 43.754 |
| Fused generated scorer, default block 128 / four warps | 2.730 | 24.373 |
| Fused generated scorer, block 32 / one warp | 2.197 | 19.245 |
| Handwritten production CSI, rerun control | 0.889 | 7.604 |

The block-32 setting is an explicit supported launch option, not a new global
default inferred from one model/GPU. Its scores exactly match the block-128
fused scores. Compared with the previous endpoint-optimized run (18.489 / 44.329
seconds), the tuned fused route is **8.42x / 2.30x faster**. Compared with the
original pre-optimization generated route (43.631 / 244.095 seconds), the two
optimization stages together yield **19.86x / 12.68x** on this workload.

At this stage, the fused route still ran all 561 trials to their own decision endpoints or
the cap, with strict truncation checks. It has no observation-window early
stopping and no sampling skip for unscored trials. The handwritten production
route has both optimizations. The earlier handwritten full-endpoint control
was 1.604 / 16.018 seconds; this is a useful comparison but was not rerun in
the fused experiment. These are objective batches, not whole fit runtimes or
comparisons with the direct CPU solver.

Peak Torch allocated GPU memory fell from **2.718 GiB** for a materialized
single-candidate microbatch to **0.091 / 0.898 GiB** for fused execution of all
eleven candidates together at 10 ms / 1 ms. Full-population sample buffers
alone would otherwise require 16.09 GiB. Context/display/allocator reservations
and host storage are excluded from these allocated-memory figures.

Default fused warm runs were 2.750/2.710 and 24.548/24.198 seconds; first calls
were 6.067 and 27.167 seconds. Block-32 warm runs were 2.199/2.194 and
19.349/19.141 seconds; first calls were 3.033 and 20.236 seconds. No tests or
other benchmark jobs ran concurrently with these measurements.

Total log scores differ from the materialized weighted-sample estimator by
at most **9.16e-5 / 7.63e-5** at 10 ms / 1 ms. Integer grouping changes FP32
summation order; estimator definitions, sample budgets, and random streams are
unchanged. The materialized sampler's scores still exactly match the previous
baseline at both timesteps.

Acceptance coverage includes exact mass/histogram statistics, both CRN policies,
non-power-of-two simulation counts, candidate and estimate chunking, GPU launch
geometry, score masks, out-of-range bins, floors, invalid configurations, forged
readouts, strict truncation, injected nonfinite outputs, automatic candidate
batching, and memory budgets that fit fused statistics but reject sample arrays.
Renamed components, altered gates/projections, and reversed observation columns
also pass. The larger 786,624-lane GPU check now validates fused empirical hits
and histogram counts in addition to source observation/count equivalence, with
zero mismatches at both timesteps, both seeds, and both CRN policies.

Reproduce:

```bash
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --methods specialized_fit generated_histogram generated_fused

env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --methods generated_fused --fused-block-size 32 --fused-num-warps 1 \
  --verify-fused-counts
```

The full-budget count check is outside the timing loops. It compares the full
fused population with materialized reference samples one candidate at a time,
preserving complete histories and CRN identities. No JSON files are saved by
the script and no cluster jobs were submitted.

That check **passed at both timesteps**: all 30,855 retained integer bin-count
cells per timestep matched exactly, for 617,100,000 simulated lanes per timestep
(1,234,200,000 across the two cases). These are checks of aggregated bin counts,
not individual lane-by-lane source comparisons. The separate 786,624-lane
source-equivalence experiment provides the latter coverage.

## Follow-up: generic score-only and observation-window execution

The histogram compiler now has three explicit execution policies. `strict`
remains the default and full-endpoint oracle. `score_only` compacts stochastic
launches to included trials while replaying **all** observed histories.
`window` additionally derives candidate/trial count cutoffs from the checked
conditioning-event expression and the contributing histogram bins.

The cutoff planner uses conservative FP32 interval evaluation over count
suffixes, retaining the original expression tree and supported FMA envelope.
Only an interval disjoint from the closed scoring window authorizes a stop;
unsupported numeric observations or uncertified arithmetic fall back to full
execution. This does not rely on CSI names, a hand-coded RT equation, or a
positive readout slope. The certificate covers the source's declared step cap,
not an unbounded continuous process. Reducing the requested runtime horizon
does not reduce that certification domain.

Skipped trials expose unavailable statistics (`bin_counts=-1`, densities/log
factors NaN, and `sampled_trials=False`). Deliberately stopped lanes have no
endpoint readout and contribute no bin counts, but remain in the original
simulation denominator. `window_stopped` reports these counts separately.
Unresolved lanes without a certified cutoff still raise truncation errors.
These score-only modes cannot diagnose unexecuted tails; they are not a
replacement for strict simulation validation.

The same full workload was rerun: 561 chronological trials / 485 scored,
11 candidates, 100,000 simulations per trial, 12-second cap, exact synthetic
outcomes on the real task sequence. Both implementations use block 32 / one
warp. Synchronized medians of two warm calls, excluding the first call:

| End-to-end seconds per full candidate batch | 10 ms | 1 ms |
|---|---:|---:|
| Generated strict fused, rerun | 2.108 | 18.921 |
| Generated score-only, unscored trials omitted | 1.878 | 16.884 |
| Generated observation-window execution | 1.223 | 9.582 |
| Handwritten production CSI, rerun control | 0.829 | 7.490 |

The new window mode is **1.72x / 1.97x faster** than the rerun strict generic
objective. The gap to the handwritten production objective is now **47% / 28%**,
down from approximately 2.5x in the preceding experiment. Skipping unscored
trials removes 83.6 million of 617.1 million stochastic lanes; the 533.5 million
remaining lanes include 221,515,781 / 238,323,648 deliberately stopped before
their endpoints at 10 ms / 1 ms. Neither shortcut changes canonical history,
random-stream identity, the histogram estimator, or its normalization.

Memory is effectively unchanged: peak Torch allocated device memory is
0.091 / 0.898 GiB for the generic modes, versus 0.028 / 0.277 GiB handwritten.
All deterministic boundary paths are still generated through the requested
horizon, including paths for unscored trials. Eliminating those unused paths
and host/device round trips remains possible future work; no such optimization
is included in these timings.

Strict warm runs were 2.127/2.088 and 18.904/18.939 seconds; score-only runs
were 1.878/1.879 and 16.831/16.937; window runs were 1.215/1.230 and
9.529/9.634; handwritten runs were 0.829/0.829 and 7.449/7.531. First calls
were 2.905/19.843, 2.417/17.668, 1.754/9.963, and 1.085/7.483 seconds,
respectively. No tests ran concurrently with these timing loops.

All eleven total log scores match **exactly** between the strict, score-only,
and window-generated objectives at both timesteps. The handwritten control
still uses different RNG mappings, so its individual scores are not expected
to match those of the generated sampler at an equal seed.

The full-budget integer checks passed at **both** timesteps: all 26,675 scored
bin-count cells from window execution matched strict fused execution, and all
30,855 strict fused cells matched materialized samples. This covers reference
populations of 617,100,000 lanes per timestep, or 1,234,200,000 across both.
It verifies sufficient statistics, not individual full endpoints for stopped
lanes (which intentionally do not exist in window execution).

The separate 786,624-lane source validation was rerun with both seeds, both CRN
policies, and both timesteps. Full generated and coupled source observations,
event counts, empirical-mass hits, and histogram counts had zero discrepancies.
Window scoring also matched the materialized histogram on an interleaved trial
mask while candidate and estimate chunking were enabled. The GPU-focused
regression selection passed (44 passed, 24 skipped, including style checks),
covering signed/transformed time gates, renamed nodes and reordered columns,
bin boundaries, empty masks/support, unsupported numeric-field fallback,
resource guards, witness tampering, nonfinite readouts, and true truncation.
The targeted interpreter selection also passed (9 passed, 9 opposite-backend
cases skipped), exercising both faster modes, chunk/RNG identity, signed time
gates, empty masks, numeric fallback, nonfinite outputs, and truncation.

Reproduce timing and full-budget integer-statistic checks:

```bash
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --methods specialized_fit generated_fused generated_score_only generated_window \
  --fused-block-size 32 --fused-num-warps 1 --verify-fused-counts
```

## Retirement follow-up: recorded-data compatibility

The next milestone adds opt-in PEC routing through `batched_observations` and
explicit `history_timing="ceil_fp32_8ulp"` for approximate positive-count
histories. Existing fitting defaults remain on their previous routes.
The recorded-data audit and limitations are documented in
[LIKELIHOOD_COMPILE_USAGE.md](LIKELIHOOD_COMPILE_USAGE.md#opt-in-pec-migration-and-approximate-history-timing).
That audit validates actual recorded RTs and continuously proposed NDT values;
it does **not** replace the fitting-budget timings above with its small
257-estimate correctness budget. Zero-step history support and complete
optimizer validation remain prerequisites for retiring the handwritten kernel.

## Default-route retirement check (2026-09-10)

The previous section is superseded: zero-count replay is supported for the
checked counted-prelude structure, and the legacy-named deterministic-history
API plus PEC's deterministic-history mode now select generated code by default.
The custom kernel remains available only by explicit `handwritten` selection.
This retires the custom *likelihood kernel's default routing*, not the registered
primitive operations, ordinary forward simulator, or continuous direct solver.

Final isolated RTX 2080 Ti timing used subject 1's **actual recorded choices and
RTs**, all 561 trials / 485 scored, 11 condition-dependent candidates, and
100,000 estimates per candidate per trial. NDT proposals were not quantized to
the simulation timestep. Both methods used 100 histogram bins, sigma 0.5,
pseudocount 0.1, seed 17, block size 32 / one warp, and observation-window
execution. These are warm objective evaluations, not optimizer or subject-fit
times. Each number is the median of two synchronized warm runs; no tests or
other task workloads ran concurrently with the timing loops.

| Default/API route | 10 ms | 1 ms |
|---|---:|---:|
| Handwritten CSI oracle, explicit opt-in | 0.799 s | 7.482 s |
| Generated deterministic-history API, 1 GiB cap | **0.869 s** | **6.343 s** |

The generated default is about 8.8% slower at 10 ms and 15.2% faster at 1 ms.
Final warm runs were 0.797/0.801 and 7.473/7.490 seconds for handwritten,
0.869/0.870 and 6.337/6.349 for generated. Peak Torch allocated GPU memory was
0.028/0.277 GiB for handwritten and 0.091/0.898 GiB for generated. First calls
were 2.231/7.531 seconds for handwritten and 1.070/6.362 for generated; JIT cache
reuse and method order mean those are not independent cold-compilation timings.

The lower-level generated histogram scorer took 6.287 seconds at 1 ms. An
initial compatibility-API run with a 256 MiB cap took 7.839 seconds, because it
microbatched candidates. The fitting route now exposes a configurable 1 GiB
default; standalone histogram scoring retains its 256 MiB default. The larger
cap closes that batching overhead without changing counts or scores. Users can
lower it on memory-constrained devices; framework/compiler workspace is extra.

Full-budget strict fused counts matched materialized generated samples exactly
for all 30,855 count cells at each timestep. Window execution matched all 26,675
scored count cells, retaining 221,223,174 intentional window stops at 10 ms and
203,989,657 at 1 ms in the estimate denominator. Each materialized reference
population contained 617,100,000 lanes (1,234,200,000 across both timesteps).
Both the 256 MiB and 1 GiB compatibility routes produced exactly the same scores
as the larger-budget generated window scorer. Handwritten and generated scores
are *not* matched-draw comparisons: their random-stream mappings differ.

The separate recorded-data audit passed eight candidates at both timesteps,
including the two high-NDT candidates with 55 and 33 zero-count trials. Endpoint
counts, LCA states and drift paths matched the handwritten oracle exactly for
that subject workload. Small zero-settling tests separately account for unused
uninitialized output placeholders and verify preservation of source states.
Broad GPU regression: 133 passed / 56 skipped. Targeted interpreter checks:
5 passed / 4 skipped, including zero settling and PEC scoring. No cluster jobs
were submitted and no JSON result files were added.

Reproduce timings and the full-budget integer check:

```bash
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --recorded --methods specialized_fit generated_window generated_default \
  --fused-block-size 32 --fused-num-warps 1 --verify-fused-counts
```

The final 10 ms default/API comparison was also rerun with `--dt .01 --methods
specialized_fit generated_default`; the table uses that final rerun. Individual
warm timings varied by a few percent across these short runs.

## Follow-up: GPU-resident boundary trajectories

The preceding work was committed as `6516590034`. Follow-up profiling of the
1 ms, 11-candidate, 100,000-simulation objective found that the generated GPU
sampling kernel itself was not slower: approximately 5.54 seconds generated
versus 5.75 handwritten. Generated history/path kernels took about 0.42 seconds
versus 1.58 handwritten. The generic complete path-generation/inspection stage,
however, took 3.43–3.48 seconds, followed by 0.27–0.31 seconds uploading large
NumPy buffers. These separate diagnostic timings motivated this change; they
are not additive to the earlier end-to-end benchmark measurements.

`BoundaryTrajectoryPlan.generate_device()` now returns backend tensor paths
directly. A generic Triton validation kernel checks consumed-value finiteness
and exact prefix coverage without path-sized temporary arrays, returning just
two error flags. Canonical-start comparisons and all history/scheduler guards
are retained. Fused empirical-mass and histogram scorers consume that tensor
directly; no full trajectory download, NumPy inspection, or re-upload occurs.
`generate()` remains the independent full NumPy inspection oracle. Device
tensors are owned by their result and must be treated as read-only; scoring
continues to construct its own paths rather than accepting external buffers.

The memory planner now accounts for one device trajectory allocation instead
of host/device copies and a second sampling-device allocation. Per-step validity
and scheduler-pass buffers are still produced, validated, and then released.
Canonical history snapshots and endpoint inversion remain host-side. All paths
still span the requested horizon, including paths for unscored trials; this
change only alters storage and validation, not generated dynamics or RNG.

The same 2080 Ti workload was rerun sequentially with block 32 / one warp:
561 chronological trials, 485 scored, 11 candidates, 100,000 simulations,
12-second cap, and exact synthetic observations on the real trial sequence.
Medians of two synchronized warm calls, excluding the first call:

| End-to-end seconds per full candidate batch | 10 ms | 1 ms |
|---|---:|---:|
| Generated window, prior host-inspection benchmark | 1.223 | 9.582 |
| Generated window, device-resident paths | 0.950 | 6.369 |
| Handwritten production CSI, rerun control | 0.845 | 7.502 |

Device-resident paths reduce the generated objective's time by **22% / 34%**
relative to the previous stage. At 1 ms the generated objective now takes
**15% less time** than the handwritten control (1.18x throughput). At 10 ms it
still takes about **12% more time**. These are workload-specific objective
measurements, not full fit times, direct CPU solver comparisons, or a claim
that every model/GPU benefits equally.

Generated warm runs were 0.955/0.945 and 6.361/6.377 seconds; first calls were
1.177 and 6.536 seconds. Handwritten warm runs were 0.845/0.845 and
7.473/7.531 seconds; first calls were 1.137 and 7.388 seconds. No tests ran
concurrently with these timing loops. Peak Torch allocated GPU memory remains
0.091 / 0.898 GiB generated versus 0.028 / 0.277 GiB handwritten: the inspection
buffers still exist transiently on GPU, although their host copies are gone.

All candidate scores and intentional-stop counts exactly match the previous
generated window benchmark at both timesteps. The handwritten scores match
their own previous baseline; the two implementations retain different random
stream mappings, so their scores should not be compared as matched draws.

Full-budget integer checks passed at both timesteps: all 26,675 scored window
bin-count cells matched strict device-path scoring, and all 30,855 strict cells
matched materialized inspection samples. Each reference population contains
617,100,000 lanes (1,234,200,000 across both timesteps). GPU regression tests
passed (32 passed, 27 skipped, including style checks), as did the targeted
interpreter selection (8 passed, 8 skipped, including style checks). New tests
check exact device/NumPy path equality, prohibit large `.cpu()` downloads,
inject nonfinite/prefix/start-state faults, check invalid suffix masking and
partial scan tiles, and retain witness/resource guards and scoring chunk/RNG
identity. The smaller automatic-batch budget test now targets the device-only
memory accounting rather than the removed host/device copies.
The 786,624-lane coupled-source validation also passed after the storage change:
both timesteps, both seeds, and both CRN policies had zero observation/count,
empirical-mass, or histogram discrepancies, including masked/chunked window
scoring. No JSON output files were added and no cluster jobs were submitted.

A separate synchronized phase profile confirms the intended bottleneck was
removed: complete device-path preparation takes **0.597–0.600 seconds**, down
from the earlier 3.43–3.48-second inspection stage, with no subsequent full-path
upload. Device finiteness/prefix validation takes only **1.69–1.74 ms**.
Endpoint inversion takes 0.083–0.085 seconds, history execution 0.428–0.433,
and the boundary execution wrapper 0.032–0.033. These are nested phase timers:
the complete preparation time includes the other listed preparation phases,
so they must not be added to it. The profiled objective median was 6.242 seconds;
the uninstrumented 6.369-second value above remains the headline measurement.

Reproduce:

```bash
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_likelihood_specializations.py \
  --methods specialized_fit generated_window \
  --fused-block-size 32 --fused-num-warps 1 --verify-fused-counts
```
