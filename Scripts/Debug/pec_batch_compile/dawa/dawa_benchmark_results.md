# DAWA simulation benchmarks

## Response noise only

Measured on 2026-09-23 under WSL2 with an NVIDIA RTX 2080 Ti (11 GB) and an
Intel Core i7-9700K using all eight CPU threads. The fit scripts specify 10,000
estimates. The data contain 760 retained trials per subject after filtering
missing previous-congruency values.

| Retained trials | Estimates | LLVM, 8 CPU threads | Triton, 2080 Ti | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1,000 | 1.738 s | 0.068 s | 25.6× |
| 64 | 10,000 | 15.587 s | 0.072 s | 215.6× |
| 760 | 1,000 | 19.329 s | 0.363 s | 53.2× |
| 760 | 10,000 | 178.651 s | 0.423 s | 422.6× |

These are simulation timings for one parameter candidate and one subject.
Numbers are medians of three warmed calls, except the full 760-trial, 10,000-estimate
LLVM row, which is one measured warmed call after an initial call. Both calls
actually simulated the full workload; no timing is extrapolated. Raw durations,
first-call times, software versions, input-file hash, and output summaries are
in `dawa_benchmark_results.json`.

Each backend/case ran in a fresh process, and timed cases ran serially. Setup and
JIT compilation are excluded from warmed timings. The timed region includes
input preparation, simulation, and returning decision/RT arrays to host memory.
LLVM uses PEC's actual threaded `grid_evaluate` path, not a Python loop over
estimates. Triton uses the ordinary batched compiler on the same PEC-wrapped
model, with PEC's fitting controls represented as runtime parameters. Native
LLVM produces float64 outputs; Triton produces float32 outputs. Density
estimation and optimization are excluded: LLVM's objective callback returns a
constant, while both paths return all simulated outcomes.

The inputs are subject 1 from `dawa_lca_model/flanker_data_part1.csv`, in original
order, filtered exactly as in the supplied fit script. The 64-trial cases use
the beginning of that sequence. Parameters are threshold 0.3, nondecision time
0.2, bias -0.45, control gain 10, LC mode 0.9, LC slope 1, and LC intercept 5.
These seven parameters are constant over the benchmark sequence. This measures
one representative candidate, not an optimizer run or the full multi-subject
conditional fit. Runtime will vary with candidate parameters, especially the
response threshold.

Both paths used the `recurrent` scheduler adjustment described in
`README.md`. That schedule is now the default in the shared model
builder and driver; it fixes the former stall for multi-pass responses. The
recorded timings and source hash describe the benchmark before this default
change, with the same recurrent execution behavior. All GPU runs used strict
truncation checking with a 2,000-pass cap;
none truncated. Each backend reproduces its seeded repeated samples exactly.
GPU and LLVM random streams differ, so their stochastic samples are compared
as distributions rather than matched draws.

For 760 trials and 10,000 estimates, mean RT is 0.973180 s on Triton
and 0.973245 s on LLVM. Their difference is
0.064 ms; the combined Monte Carlo standard
error, treating each simulated trial sequence as an independent estimate, is
0.120 ms. The choice-1 fractions are 0.498731
and 0.498695, respectively.

There is a numerical validation caveat: with response noise disabled, LLVM
returns 0.97 s for the first retained real-data trial, while Python and Triton
return 0.96 s. Triton's first-trial LCA/LC states match Python within 8e-8;
the next three trial RTs and all four choices agree with LLVM. The one-step
LLVM discrepancy also occurs outside PEC and remains unresolved. These timing
results do not establish exact backend equivalence.

To reproduce the full-subject GPU case:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_llvm_benchmark.py \
  --backend triton --trials 760 --estimates 10000 --repeats 3 \
  --output /tmp/dawa_triton_760_10000.json
```

Use `--backend llvm --repeats 1` for the measured full-subject LLVM case, and
`--trials 64` or `--estimates 1000` for the smaller cases. `--deterministic`
disables response noise for numerical checks. The reusable benchmark writes
progress and a JSON report after every completed run.

## Noise in all four LCAs

The September 24 audit uses Gaussian standard deviation 0.1 in control,
stimulus, decision, and response, with the original 0.01-second LCA step.
Control retains its state across trials; the other three layers reset. Both
backends use the same constructed control RESULT value, real subject-1 trial
sequence, nonlinear dynamics, and parameter candidate as above.

| Retained trials | Estimates | LLVM, 8 CPU threads | Triton, 2080 Ti | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 10,000 | 17.443 s | 0.089 s | 196.9× |
| 64 | 100,000 | 188.852 s | 0.336 s | 562.1× |
| 760 | 10,000 | 214.144 s | 0.574 s | 373.0× |

GPU numbers and the 64-trial/10,000-estimate LLVM number are medians of three
warmed calls. The two larger LLVM cases use one measured warmed call after an
initial call. Every workload ran in full, serially in a fresh process; setup
and first-call times are recorded separately. These are simulation timings,
including input preparation and host outputs, excluding density estimation
and optimization. Every GPU run passed strict truncation checks, and all
backends reproduced their seeded repeated samples exactly. Compact reports
and comparisons are in the `all_lca_noise` section of
[dawa_benchmark_results.json](dawa_benchmark_results.json).

The comparison explicitly uses **independent component noise**. Native PEC's
default randomization control broadcasts the same seed to all random variables
in an estimate. With multiple noisy LCAs this correlates their noise. The
benchmark's `--independent-noise-streams` option assigns distinct seed ranges to
LLVM's randomization projections while retaining PEC's threaded evaluator.
Triton already uses separate component streams. The original fitting scripts
are unchanged; they would need the same seed separation to fit this independent
noise model with native PEC.

The distinction materially affects behavior: with 64 trials and 10,000
estimates, default shared-seed LLVM has mean RT 0.94865 s and choice-1 fraction
0.50045. With separate seeds those become 0.98844 s and 0.53038, close to the
GPU's 0.98826 s and 0.53027. This is why shared-seed timings are excluded from
the independent-noise performance comparison.

The compiler's initialization audit constructs all four LCAs with noise, then
disables subsequent draws to isolate initialization and recurrence. Across
four real-data trials, GPU choices/RTs and all LCA/LC states match native Python
within 4e-7. Focused GPU and interpreter tests additionally check analytic noise
moments, independence across accumulators and trials, retained state, seeded
replay, and immunity to subsequent edits of the source graph.

There is still a **first-trial RT discrepancy with LLVM**, also observed in the
earlier response-only audit. In the 10,000-estimate independent-noise comparison,
the first-trial mean RT is 0.92728 s on GPU and 0.94174 s on LLVM (5.56 combined
standard errors). Overall mean RT and choice fraction agree within one combined
standard error. The raw report records first-trial discrepancies separately;
these measurements do not establish exact native LLVM equivalence.

At 100,000 estimates, the overall RT difference is 0.255 ms (1.57 combined
standard errors); after the first trial the maximum absolute per-trial RT
discrepancy is 2.31 standard errors. Across the full 760-trial sequence, mean RT
is 0.982601 s on GPU and 0.982715 s on LLVM, a 0.114 ms difference with combined
Monte Carlo standard error 0.149 ms. The choice-1 fractions are 0.498792 and
0.498913 (1.45 combined standard errors). Standard errors treat whole simulated
sequences as independent estimates, retaining within-sequence dependence.

To reproduce the independent-noise workload:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_llvm_benchmark.py \
  --backend triton --trials 64 --estimates 10000 --repeats 3 \
  --c-noise .1 --s-noise .1 --d-noise .1 --r-noise .1 \
  --independent-noise-streams --output /tmp/dawa_all_noise_triton.json
```

Use `--backend llvm` for the native comparison, `--estimates 100000` for the
larger estimate count, or `--trials 760` for the full subject sequence. The two
larger native measurements used `--repeats 1`.

## Full-subject PEC objective at 100,000 estimates

Measured September 24 on the same RTX 2080 Ti, using the ordinary PEC batched
GPU simulator and its histogram objective. Four distinct parameter candidates
each simulate subject 1's complete 760-trial sequence, with 100,000 estimates
per trial and candidate (76 million simulated trial outcomes per candidate).
The original include mask scores 720 trials; all 760 execute in order, retaining
control state. All four LCAs have Gaussian noise standard deviation 0.1 and
the original 0.01-second integration step.

The seven fitted parameters expand to eight coordinates for one subject,
because LC mode depends on previous congruency. The proposals vary all eight
coordinates, with response thresholds 0.30, 0.40, 0.50, and 0.60. Exact
coordinates, repeated timings, and scores are recorded in the
`full_subject_100k_pec_objective` section of
[dawa_benchmark_results.json](dawa_benchmark_results.json).

| Evaluation of the same four candidates | Median total | Seconds per candidate | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| One candidate at a time | 12.239 s | 3.060 s | 3.12 GiB |
| Two candidates at a time, two calls | 11.999 s | 3.000 s | 6.23 GiB |

Medians use three warmed repetitions after a first call for each batch size.
Timings include conditional-parameter preparation, simulation, strict
truncation checks, histogram density estimation, and log-score aggregation.
Samples remain on the GPU. Only likelihoods and scores return to the CPU.
Setup took 11.0 seconds, separately; the local Triton compilation cache was
already populated. Optimizer proposal generation and convergence are not timed.

Every simulated trajectory completed within the 2,000-pass cap. Per-trial
likelihoods match exactly across candidate batch sizes, and repeated calls at
the same batch size reproduce the total scores exactly. Total log scores differ
by at most 0.00110 between batch sizes because NumPy's masked float32 reduction
uses a different summation order for one versus multiple lanes; the underlying
trial probabilities are identical.

This uses 100 RT bins on [0, 3] seconds, zero histogram smoothing, and pseudocount
1. It measures the existing sum of trial-marginal histogram log densities,
with simulated persistent control state. It does not switch to native PEC's
CPU fastKDE or a sequentially observation-conditioned likelihood. It is a
throughput measurement, not a completed or validated subject fit.

At the measured average throughput, 1,000 candidate evaluations take about
50 minutes, 5,000 take 4.2 hours, and 10,000 take 8.3--8.5 hours, excluding
optimizer overhead and setup. The supplied Optuna/CMA-ES scripts budget 5,000
candidate evaluations, so roughly four hours per subject per optimization start
is a useful initial estimate for this configuration. Individual proposal
medians ranged from 2.29 to 3.87 seconds (3.2--5.4 hours if 5,000 evaluations
all cost that much). Other parameter regions, restarts, and required convergence
can change the total substantially. At 100,000 estimates, processing two
candidates together provides little additional throughput and doubles memory.

To reproduce:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --batch-sizes 1 2 --repeats 3 \
  --materialized \
  --output /tmp/dawa_pec_fit_100k.json
```

## GPU profiling and count-only scoring

The September 24 profile separates the two issues: simulation time and fitting
buffer size. For the first full-subject candidate, Torch/CUPTI measured 3.415
seconds in the generated simulator out of 3.462 seconds of GPU kernel time
(98.64%). Histogram operations and validation accounted for the remainder.
The compiled simulator used 255 registers per thread and reported eight spills.
Nsight Compute measured 24.78% achieved occupancy, a register-limited ceiling
of 25%, 51.83% compute throughput, and 1.23% DRAM throughput. This is primarily
a computation/register-pressure problem, rather than a bandwidth bottleneck.
Nsight's instrumented replay durations are excluded from the timing comparisons.

The old fitting path retains 760 × 100,000 × 2 float32 outcomes: 608 MB
(580 MiB) for choices and RTs alone, per candidate. Output selection, category
comparisons, bin indices, range masks, and finite checks create additional
arrays. Peak live Torch allocations reach 3.115 GiB. CUDA allocator reservations,
context/code storage, local spill storage, and the Windows desktop are separate
from this live-Torch figure.

`BatchedSimulationPlan.log_likelihood` now uses a general count-only path for
unsmoothed stateful histograms. It runs the same generated dynamics and random
streams, but reduces matches and diagnostic flags at each trial boundary rather
than writing every estimate's outcomes. Persistent state remains within each
simulated sequence. Masked trials still execute, and all outputs are checked
for nonfinite values even when they are not selected for scoring. There is no
observed-RT cutoff, change to integration steps, or replacement of latent history.
Gaussian smoothing of one continuous outcome now also uses count-only scoring,
as described in the extension below. Smoothing multiple continuous outcomes
and unsupported fusion kinds retain the existing materialized path.
`fused=False` on the plan or
`batched_fused_likelihood=False` on `PECOptimizationFunction` selects the
reference path explicitly; ordinary `plan.run` still returns raw samples.

Fresh unprofiled runs of the same four proposals, with medians of three warmed
repetitions on the 2080 Ti, gave:

| Scorer and launch | Candidates per call | Total for four | Seconds per candidate | Peak live Torch memory |
| --- | ---: | ---: | ---: | ---: |
| Materialized, 128 lanes / 4 warps | 1 | 11.625 s | 2.906 s | 3.115 GiB |
| Count-only, 128 lanes / 4 warps | 1 | 12.165 s | 3.041 s | 0.144 MiB |
| Count-only, 32 lanes / 1 warp | 1 | 10.096 s | 2.524 s | 0.144 MiB |
| Count-only, 32 lanes / 1 warp | 4 | 9.936 s | 2.484 s | 0.231 MiB |

Count reduction chiefly fixes memory; it does not itself accelerate the
dominant dynamics kernel. The smaller launch blocks provide the runtime gain:
about 15% less time than the fresh materialized baseline. The 5,000-evaluation
projection becomes approximately 3.45 hours, excluding optimizer overhead and
setup. These remain throughput projections for the four fixed proposals.
The global launch default remains unchanged; the measured DAWA configuration is
`batched_triton_launch_options={"block_size": 32, "num_warps": 1}`.

Forcing a 128-register cap at 64 lanes / 2 warps was counterproductive: reported
spills rose to 124 and the profiled candidate kernel took 13.56 seconds. No
register cap is recommended for this workload. Further runtime work should
target the live scheduler/parameter values and repeated Gaussian generation,
rather than histogram bandwidth. Those changes require additional numerical
and random-stream validation.

The CSI comparison also has substantive workload differences. The local CSI
fit benchmark uses 561 trials (485 scored), deterministic observed LCA history,
a scalar noisy DDM, and checked histogram-window stopping; it measured 6.20
seconds for eleven candidates on the same GPU. DAWA uses 760 trials (720 scored),
ten noisy accumulator coordinates across four LCAs plus LC dynamics, and
separate persistent control history for every estimate. That history prevents
the same precomputation and arbitrary per-trial stopping used by the CSI fitter.
The 1 ms CSI and 10 ms DAWA timesteps do not make their per-estimate computation
equivalent. The CSI timing is the existing recorded comparison, not a new
matched-model benchmark in this profiling run.

All 3,040 per-trial probabilities (four proposals × 760 trials) match the
materialized scorer exactly at 100,000 estimates. Same-batch scores replay
exactly; the previously documented small cross-batch float32 summation
differences remain. Focused CPU-interpreter and GPU tests cover persistent
state, partial blocks, multiple subjects, per-trial parameters, independent
candidate random streams, output reordering, multidimensional/category-only
histograms, interior bin edges, and masked/unselected-output diagnostics.
Compact profiling metrics and raw timing repetitions are in
`gpu_profile_and_optimization` in [dawa_benchmark_results.json](dawa_benchmark_results.json).

Reproduce the optimized timing and compare its trial probabilities with the
original scorer:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --batch-sizes 1 4 --repeats 3 \
  --block-size 32 --num-warps 1 --verify-materialized \
  --output /tmp/dawa_pec_fit_100k_fused.json
```

Add `--profile /tmp/dawa_profile.json` to profile a single warmed candidate
instead of running the timing loop. It writes a Chrome trace, operator timing
and memory tables, and compiler register/spill metadata. Use `--materialized`
and `--block-size 128 --num-warps 4` to profile the original configuration.

## Smoothing and pseudocounts with count-only scoring

The stateful scorer now supports Gaussian smoothing of one continuous outcome
plus optional categorical outcomes. Setting `batched_smoothing_sigma` on
`PECOptimizationFunction` uses this path automatically; `batched_pseudocount`
works with both smoothed and unsmoothed counts. Defaults are unchanged.
Smoothing multiple continuous outcomes still uses the materialized reference.

For positive sigma, the scorer retains `2 * radius + 1` integer counts per
candidate/subject/trial, where `radius = min(bins - 1, ceil(3 * sigma))`.
Sigma 0.5 therefore keeps five counts and sigma 1.0 keeps seven. Only estimates
matching the observed choice contribute. Gaussian weights are applied after
simulation and renormalized over valid neighbors at histogram boundaries.
All simulated trial histories, integration steps, and diagnostics are preserved.

For DAWA's choice/RT observations, the density is
`(weighted_count + alpha) / ((N + alpha * B) * RT_bin_width)`, where `alpha`
is the pseudocount and `B` is the number of joint choice/RT bins. With two
choices and 100 RT bins, `B = 200`. The prior is added once after weighting,
since the Gaussian weights sum to one; it is not multiplied by the number of
neighbor bins. An explicit `batched_categorical_cardinalities=[2]` preserves
both possible choices when scoring an observed subset containing only one.

Fresh sequential-process measurements on the 2080 Ti used the same four
proposals, 760 simulated trials (720 scored), 100,000 estimates per trial per
candidate, noise in all four LCAs, 10 ms LCA steps, and 32-lane/one-warp launches.
Each timing is the median of three warm repetitions, excluding compilation.
All rows use 100 RT bins over 0–3 seconds and a pseudocount of one per joint bin.

| Scorer | Sigma | Candidates per call | Seconds per candidate | Peak live Torch memory |
| --- | ---: | ---: | ---: | ---: |
| Count-only | 0 | 1 | 2.356 s | 0.157 MiB |
| Count-only | 0 | 4 | 2.332 s | 0.245 MiB |
| Count-only | 0.5 | 1 | 2.330 s | 0.322 MiB |
| Count-only | 0.5 | 4 | 2.238 s | 0.496 MiB |
| Count-only | 1.0 | 4 | 2.363 s | 0.632 MiB |
| Materialized | 0.5 | 1 | 2.675 s | 4.248 GiB |

Smoothing has comparable runtime to unsmoothed counting in these measurements.
The main benefit remains memory: sigma 0.5 uses 0.322 MiB for one candidate,
versus 4.248 GiB with the original smoothed scorer. Live Torch memory excludes
CUDA context/code, register-spill storage, and allocator reservations. These
fixed-proposal timings do not measure a complete fitting run.

Both sigma 0.5 and 1.0 pass comparisons of all 3,040 trial densities against
the original materialized scorer. Maximum relative differences are respectively
`3.35e-7` and `4.24e-7`, attributable to weighting integer bin counts instead of
summing a weight per estimate in FP32. Repeated fused runs are bit-for-bit
reproducible; sigma 0.5 trial densities also match exactly across candidate batch
sizes one and four. The unsmoothed comparison remains exact. CPU and GPU tests
cover pseudocounts of zero and positive values, boundary normalization, absent
choices, out-of-range observations, exact bin edges, masked trials, retained
state, multiple subjects, and invalid settings.

Reproduce the smoothed evaluation and reference comparison:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --batch-sizes 1 4 --repeats 3 \
  --block-size 32 --num-warps 1 --smoothing-sigma 0.5 --pseudocount 1 \
  --verify-materialized --output /tmp/dawa_pec_fit_smoothed.json
```

Use `--smoothing-sigma 0` for the unsmoothed baseline, or
`--materialized --batch-sizes 1` for the original smoothed scorer. Compact
measurements, source hashes, and validation errors are recorded under
`smoothed_count_only_scoring` in [dawa_benchmark_results.json](dawa_benchmark_results.json).
