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

## Post-smoothing profile and optimization plan

The smoothing extension is committed as `c7084025ad`. The following profile
uses that implementation, subject 1's full 760-trial sequence (720 scored),
100,000 estimates per trial, Gaussian SD 0.1 in every LCA, sigma 0.5 smoothing,
pseudocount 1, and 32-lane/one-warp launches on the 2080 Ti. The detailed
profilers measure the first of the four established proposals after warming.
The four-proposal throughput measurements above remain the fitting benchmark;
Nsight replay durations are excluded from those timings.

Torch/CUPTI measures 2.75048 seconds in the simulation kernel, out of 2.75058
seconds of GPU kernel work: **99.9966%**. The entire profiled objective range
takes 2.79527 seconds. All GPU copies together take approximately 57 microseconds.
The large CPU `copy_`/`cudaMemcpyAsync` entry mostly waits for simulation to
finish. It does not indicate that transferring the small count buffers is the
main cost. Peak live Torch allocations remain 0.322 MiB for this candidate.

Hardware counters make the remaining bottleneck clearer:

| Measurement | Result | Implication |
| --- | ---: | --- |
| Registers per thread | 238 | High demand for registers; Triton also reports eight spills |
| Theoretical / achieved occupancy | 25% / 24.23% | Register allocation limits resident warps |
| Scheduler cycles with no eligible warp | 46.94% | Frequently no resident warp is ready to issue |
| Eligible warps per scheduler | 0.72 | Limited ability to hide dependent-instruction latency |
| Compute throughput | 58.25% of peak | Arithmetic units still have unused capacity |
| DRAM throughput | 1.29% of peak | External-memory bandwidth has substantial headroom |

A separate hardware program-counter sampling pass attributes the sampled
instruction locations as follows. These are **sampling shares, not exact
exclusive wall-clock percentages**; inlining and dependency stalls affect the
attribution.

| Source region | Share of samples |
| --- | ---: |
| Gaussian RNG, total | 63.82% |
| LC Euler integration | 11.91% |
| Other generated model/scheduler code | 19.97% |
| Triton standard reductions | 2.59% |
| Histogram and diagnostic epilogue | 0.56% |
| Unattributed | 1.15% |

Within RNG, the normal transform accounts for 30.95% of all samples, Philox
rounds for 25.94%, and other RNG operations for 6.92%. The current LCA adapter
calls `tl.randn` separately for each accumulator: ten calls per scheduler pass
across widths 2, 4, 2, and 2. Triton's scalar call runs Philox and a normal-pair
transform while returning only one normal. The scalar DDM path already has a
normal-pair reuse facility; vector LCAs do not use it. LC also performs ten
Euler updates per scheduler pass, so its integration is another meaningful
arithmetic cost.

The sampling profiler auto-expanded its buffer to 512 MiB and aggregated two
passes; its overflow flag was set, with zero final reported dropped bytes.
That instrument memory is separate from ordinary fitting memory. The earlier
all-instruction instrumented attempt was stopped because of its overhead and
contributes no measurements to this report.

There is also wasted work from waiting for the slowest estimate within a block.
A separate diagnostic ran the unchanged sampler for all four proposals with
4,096 estimates across all 760 trials. It recovered response execution counts
from `(RT - nondecision_time) / 0.01`, checking that these were integer counts.
For each trial and group, useful step slots are the sum of individual counts;
issued block step slots are `group_size * max(counts)`. Grouping the same paths
after simulation gives:

| Proposal | Mean response steps | Useful slots, group 32 | Useful slots, group 128 |
| --- | ---: | ---: | ---: |
| 1 | 78.25 | 55.33% | 48.47% |
| 2 | 71.62 | 65.22% | 59.13% |
| 3 | 76.19 | 63.87% | 56.88% |
| 4 | 47.04 | 57.50% | 49.17% |

These are a model-work utilization proxy, not measured hardware utilization or
a prediction of achievable speedup. They explain part of the benefit of smaller
blocks and identify a possible larger scheduling improvement.

Recommended implementation order:

1. **Generate independent Gaussian values in groups for vector LCAs.** Extend
   the shared RNG interface to reuse the multiple Philox outputs and both
   members of normal pairs within an LCA execution. For DAWA, four Philox
   invocations could provide the ten required values instead of ten separate
   invocations. This targets the largest sampled cost without retaining large
   noise arrays or adding persistent spare draws for every coordinate. It
   changes the seed-to-draw mapping, so preserve the current RNG mode for
   reproducing older runs and record the selected mode in benchmarks. Validate
   noise moments, independence between accumulators/components, replay,
   common-random-number alignment, and full-sequence distribution comparisons.
   Fused and materialized scoring must still agree under the same RNG mode.

2. **Reduce live parameters and scheduler state.** The generated interface
   exposes 115 parameters, with 46 zero defaults and 30 unit defaults, while the
   fitting surface changes seven named parameters. Add general specialization
   for proven fixed values, preserving trial-dependent parameters and controller
   effects. Then simplify clocks and counters when the typed schedule proves
   that a component executes once every pass: this kernel currently emits five
   execution-count vectors and four RNG-clock vectors. These changes should
   preserve the RNG mapping. Measure registers, spills, and runtime after each
   change, and retune launch geometry once register use falls. The prior forced
   register cap caused severe spilling, so reducing required live values is the
   useful experiment.

3. **Prototype independent trial progression for each estimate.** Let an
   estimate start its next trial as soon as it finishes, carrying its own
   control state and RNG identity. This could recover some of the 35–45% idle
   step slots seen with groups of 32. It is a larger change: trial indices,
   conditional parameters, input loads, and histogram writes would differ
   across lanes. Preserve each estimate's ordered trial history and all
   diagnostics, then measure whether useful-work gains outweigh the additional
   indexing and scattered writes. Keep the existing scheduler as the reference.

Each stage should be a general compiler feature, with the full-subject
100,000-estimate workload as a regression benchmark. The next experiment I
recommend is grouped Gaussian generation; potential runtime gains remain
unmeasured. No additional compiler optimization was implemented in this profile.

Reproduce the Torch profile:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --block-size 32 --num-warps 1 \
  --smoothing-sigma 0.5 --pseudocount 1 \
  --profile /tmp/dawa_smoothed_profile.json \
  --output /tmp/dawa_smoothed_profile_summary.json
```

Reproduce the hardware counters, skipping the first four candidate launches
and collecting the first warmed candidate:

```bash
ncu --target-processes all --kernel-name pnl_batched_coevolving_graph_kernel \
  --launch-skip 4 --launch-count 1 --kill yes \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis \
  --force-overwrite --export /tmp/dawa_smoothed_hw \
  .venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --batch-sizes 1 --repeats 1 --block-size 32 --num-warps 1 \
  --smoothing-sigma 0.5 --pseudocount 1 --output /tmp/dawa_smoothed_hw_run.json
```

For source sampling, replace the section options with
`--section LaunchStats --metrics smsp__pcsamp_sample_count` and use another
export filename. Compact measurements, sampling metadata, and the plan are
recorded under `post_smoothing_profile_and_plan` in
[dawa_benchmark_results.json](dawa_benchmark_results.json). Large traces and
Nsight reports remain outside the repository.

## Grouped Gaussian generation

The shared Triton emitter now provides `ctx.normal_draws(node.name, step)` for
a complete declared vector of independent Gaussian draws. The scheduled LCA
adapter uses it for every supported width, with no DAWA-specific compiler
logic. `philox4x_v1` groups up to four coordinates within one component
execution, using both Box–Muller outputs and both Philox uniform pairs when
needed. DAWA's widths 2, 4, 2, and 2 require **four Philox invocations instead
of ten**, and five Box–Muller pairs instead of ten. All ten Gaussian values
are still generated. No unused draws are retained across steps or shared
between different component clocks.

The new mode is the default. Select `normal_rng="legacy"` in
`triton_launch_options` (or PEC's `batched_triton_launch_options`) to reproduce
the previous per-coordinate generator. Modes differ in seeded trajectories;
both preserve the intended distributions and common-random-number alignment.
Record the mode with the seed. Scalar DDM temporal pairing is unchanged.

### Full-subject throughput

Fresh measurements on the WSL RTX 2080 Ti, September 24, 2026: subject 1,
760 ordered trials (720 scored), **100,000 estimates per trial per candidate**,
the same four fitting proposals, Gaussian SD 0.1 in all LCAs, LCA dt 0.01 s,
100 RT bins on [0, 3], smoothing sigma 0.5, and pseudocount 1. Launches use
32 lanes and one warp. Each entry is the median of three warmed full-pool
evaluations divided by four candidates; a separate cold run is excluded.
Timings include PEC preparation and scoring, but not optimizer convergence.

| Candidate batch size | Legacy seconds/candidate | Grouped seconds/candidate | Speedup | Runtime reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.2312 | 1.7405 | 1.282× | 22.0% |
| 4 | 2.1876 | 1.6645 | 1.314× | 23.9% |

Live Torch buffer peaks are unchanged: 0.3218 MiB for batch size 1 and
0.4961 MiB for batch size 4. CUDA context/code and allocator reservations are
additional. At the batch-four rate, 5,000 candidate evaluations would take
about 2.31 hours instead of 3.04 hours, before optimizer overhead. This is a
throughput extrapolation, not a measured fit.

A separate Torch/CUPTI profile of the first proposal measured the simulation
kernel at 2.7267 s with legacy draws and 2.0861 s with grouped draws. These
instrumented runs are excluded from the table. Registers per thread fell
from 238 to 230; Triton still reports eight spills. Simulation remains over
99.99% of GPU kernel time. Source equations, time steps, trial synchronization,
and scheduling clocks were not changed by this optimization.

### Validation

- Legacy mode exactly reproduces the previous benchmark's hash of all 3,040
  trial densities. Both modes replay exactly and retain the same trial
  densities when changing candidate batch size from one to four.
- At 100,000 estimates, the grouped fused scorer matches the materialized
  sampler for all 3,040 densities: maximum relative difference
  `3.7463e-7`, within the established FP32 weighting tolerance of `2e-6`.
- Complete sequences were also sampled with 16,384 estimates per mode and
  independent seeds, checking every trial's choice/RT distribution, including
  unscored trials. Maximum choice/RT sub-CDF differences across the four
  proposals were 0.02008, 0.01819, 0.01898, and 0.01929, below the conservative
  simultaneous 99% bound of 0.04237. The bound uses two one-sample DKW bounds
  and a union bound across proposals, trials, and choices; it allows dependence
  across trials. These are checks of trial marginals after running complete
  histories, not a test of the entire high-dimensional joint sequence law.
- Analytic normal moments, tails, and correlations were checked for widths
  1, 2, 3, 4, 5, 7, and 32, including distinct RNG owners and execution steps.
  Replay tests cover common randomness, independent candidate/subject streams,
  changed seeds, launch geometry, persistent state, and split-sequence resume.
- The focused suites passed 59 GPU vector/launch/histogram checks, 70 GPU
  CSI/DDM/reduced-scoring checks, and 41 interpreter vector/histogram checks.
  Existing Gaussian LCA network and persistent-noise GPU checks also passed.

Reproduce the throughput and distribution comparison from the repository root:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --batch-sizes 1 4 --repeats 3 \
  --block-size 32 --num-warps 1 --smoothing-sigma 0.5 --pseudocount 1 \
  --normal-rng philox4x_v1 --verify-materialized --validate-normal-rngs \
  --output /tmp/dawa_gaussian_grouped.json
```

Use `--normal-rng legacy` for the baseline timing. `--validation-estimates`
controls the separate distribution comparison (default 16,384); it does not
change the 100,000-estimate timing workload. Add `--profile /tmp/trace.json`
to collect a separate warmed first-proposal profile instead of timing the pool.
The compact results are under `grouped_gaussian_generation` in
[dawa_benchmark_results.json](dawa_benchmark_results.json). Large traces and raw
simulation arrays are not tracked.

## Independent trial advancement

The general Triton dynamic scheduler now supports
`trial_schedule="independent"`. Each estimate advances through its own ordered
trial sequence as soon as it finishes a trial. It retains its LC/LCA state,
held control values and RNG identity, while resetting only its own per-trial
state and counters. The compiler uses that lane's trial index for inputs,
conditional parameters and output placement. Mechanism equations, dt, within-trial
execution order and random draws are unchanged. Fused scoring scatters integer
counts into the completed estimates' trial rows; Gaussian smoothing and
pseudocounts use the existing reductions.

This is an optional compiler setting for typed dynamic sequences. The global
default remains `"synchronized"`, which is also the reference implementation.
Static/atomic schedules and observed-history sampling reject the new setting.
CSI's full dynamic model supports it; CSI's optimized observed-history fitting
path already samples trials independently from replayed history and does not
use this loop. Performance should be measured for each model.

### Full-subject throughput

Fresh serial measurements on the WSL RTX 2080 Ti, September 24, 2026, using
the same four proposals and subject 1's 760 trials (720 scored), **100,000
estimates per trial per proposal**, noise SD 0.1 in all four LCAs, LCA dt 0.01 s,
100 RT bins on [0, 3], smoothing sigma 0.5 and pseudocount 1. Both modes use
`normal_rng="philox4x_v1"`, 32 lanes and one warp. Entries are medians of three
warmed full-pool evaluations divided by four, excluding a separate cold run.
They include PEC preparation and scoring, with strict truncation checking.

| Candidate batch size | Synchronized seconds/candidate | Independent seconds/candidate | Speedup | Runtime reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.6911 | 1.3991 | 1.209× | 17.3% |
| 4 | 1.6448 | 1.3193 | 1.247× | 19.8% |

Live Torch buffer peaks remain 0.3218 MiB for one candidate and 0.4961 MiB
for four. These exclude CUDA context/code and allocator reservations. At the
batch-four rate, 5,000 candidate evaluations extrapolate to **1.83 hours**,
versus 2.28 hours for the fresh synchronized baseline, before optimizer overhead.
This is a throughput estimate, not a measured fit or convergence guarantee.

A separate instrumented first-proposal pilot measured the simulation kernel
at 1.4928 s, versus 2.0861 s in the earlier grouped-Gaussian profile. Registers
per thread decreased from 230 to 213, with eight reported spills in both.
These profiles are excluded from the throughput table. The net gain includes
the cost of per-lane trial indexing, masked resets and scattered counts; it
does not recover every previously idle step slot.

### Validation and reproduction

All **3,040 trial densities match exactly** between trial schedules at 100,000
estimates, including excluded trials. Replay and changing candidate batch size
from one to four preserve them exactly. The independent fused scorer also
matches materialized sampling within the existing FP32 weighting tolerance:
maximum relative density difference `3.7463e-7`, below `2e-6`. No benchmark
trial truncated. The density hash remains
`565a2aa803797d23f50dc41547a27c9b8e2e9b464915032f00e6694b299c04df`.

Focused tests compare exact outputs and final states for persistent Gaussian
LCAs, the full noisy DAWA network, CSI's dynamic model, scalar DDM noise, delayed
execution and parallel controlled chains. They cover both Gaussian generators,
conditional parameters, multiple subjects, common and independent candidate
randomness, launch geometry, split-sequence resume, smoothed scores, truncation
and nonfinite unscored outcomes. Existing vector-RNG, histogram, dynamic-scheduler
and observed-history scoring suites also pass: 72 checks in the GPU regression
run and 40 in the interpreter run, in addition to 38 independent-trial/launch
checks in the GPU run and 16 independent-trial checks in the interpreter run.
These counts include the device-independent checks selected in each run.

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --batch-sizes 1 4 --repeats 3 \
  --block-size 32 --num-warps 1 --smoothing-sigma 0.5 --pseudocount 1 \
  --trial-schedule independent --verify-synchronized --verify-materialized \
  --output /tmp/dawa_trial_sync_independent.json
```

Use `--trial-schedule synchronized` for the baseline. PEC accepts the setting
in `batched_triton_launch_options`; direct simulation and likelihood plans use
`triton_launch_options`. Compact results are recorded under
`independent_trial_advancement` in
[dawa_benchmark_results.json](dawa_benchmark_results.json). Raw traces and
materialized sample arrays remain outside the repository.

## H100 and A100 subject benchmark

Measured September 24, 2026, using the same current compiler snapshot and
full-subject workload on three machines. Each proposal simulates subject 1's
760 ordered trials (720 scored), with **100,000 estimates per trial**, noise
SD 0.1 in all four LCAs, LCA dt 0.01 s, 100 RT bins on [0, 3], smoothing sigma
0.5 and pseudocount 1. The four fitting proposals and seed 29 are unchanged.
All use `trial_schedule="independent"`, `normal_rng="philox4x_v1"`, 32 lanes,
one warp and no register cap, on **one GPU** with eight host threads. No
architecture-specific tuning was performed.

Each entry is the median of **five warmed complete four-proposal evaluations**,
divided by four. A separate cold run is excluded. These end-to-end objective
times include PEC preparation, simulation and scoring. The 2080 Ti baseline
was rerun for this comparison.

| GPU | Serial proposals, s/proposal | Four proposals together, s/proposal | Speedup vs 2080 Ti, batch four | 5,000 evaluations, serial | 5,000 evaluations, batch four |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 2080 Ti, local WSL | 1.3888 | 1.3578 | 1.00× | 115.7 min | 113.2 min |
| H100 NVL, della-rse | 0.4501 | 0.4339 | 3.13× | 37.5 min | 36.2 min |
| A100 SXM4 80 GB, Della Slurm | 0.7988 | 0.7087 | 1.92× | 66.6 min | 59.1 min |

At batch four, the H100 is **1.63× faster than the A100** for this workload.
Ten thousand evaluations extrapolate to 72.3 minutes on H100, 118.1 minutes
on A100 and 226.3 minutes on the 2080 Ti. The original source fit scripts use
5,000 Optuna trials, so the 5,000-evaluation column is a useful budget estimate,
but those scripts default to only 10,000 estimates. This benchmark uses 100,000.
Evaluation counts are **parameter proposals, not optimizer generations**.
The serial column applies when the optimizer submits one proposal at a time;
the batch-four column requires submitting four together.

These are throughput extrapolations, not measured optimizer convergence.
Actual fit time depends on the number and values of proposals, particularly
parameters that change trial lengths. Queueing, environment setup, JIT and
optimizer overhead are additional. Software versions and launch geometry were
held constant where relevant, but these are whole-system timings and include
the different hosts' CPU preparation costs.

### Execution and validation

- The H100 run used idle GPU 0 on `della-rse.princeton.edu`, an H100 NVL with
  132 SMs and a configured 400 W power limit. Only one of the two H100s was used.
- Slurm job **14387819** ran on `della-l08g6` with one full A100 SXM4 80 GB,
  eight CPUs and 24 GiB host RAM. It completed successfully (`COMPLETED`, exit
  `0:0`) in 2 minutes 13 seconds, including startup, cold compilation and
  validation. The final request was `--constraint=a100 --gres=gpu:a100:1`
  with a five-minute limit; Slurm used `gputest`/`gpu-test`. This excludes MIG
  slices while allowing either PCIe or SXM A100 nodes.
- All three used PyTorch `2.13.0+cu130`, Triton `3.7.1`, CUDA runtime `13.0`,
  NumPy `2.3.5`, SciPy `1.18.0`, pandas `3.0.5`, llvmlite `0.48.0` and
  graph-scheduler `1.2.2`. Python was `3.13.3` locally and `3.13.13` remotely;
  both remote GPUs used driver `610.57.04` and the same isolated environment.
- All **3,040 saved trial densities match exactly across all three GPUs**,
  across candidate batch sizes and across repeats. Their common SHA-256 is
  `565a2aa803797d23f50dc41547a27c9b8e2e9b464915032f00e6694b299c04df`.
  On each remote GPU, independent and synchronized trial execution match
  exactly. The materialized reference comparison passed with maximum relative
  density difference `3.7463e-7`, within the established FP32 weighting
  tolerance. Strict truncation checks passed.
- Peak live Torch buffers remain 0.3218 MiB for one proposal and 0.4961 MiB for
  four on every GPU. CUDA context/code and allocator reservations are additional.

Both remote runs used an isolated scratch snapshot, including the uncommitted
independent-trial implementation, based on commit `5036399651`. File hashes
were verified before execution. Existing remote checkouts and environments
were not modified. Source, resolved dependencies, launch scripts, logs and raw
results are retained under:

```text
/scratch/gpfs/CSES/dmturner/dawa-benchmarks/trial-sync-20260924T212200Z
```

The benchmark now records GPU/software details and supports `--save-densities`
for cross-device comparisons. Reproduce the measured workload on an allocated
GPU with the prepared environment and source snapshot:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit_benchmark.py \
  --estimates 100000 --batch-sizes 1 4 --repeats 5 \
  --block-size 32 --num-warps 1 --smoothing-sigma 0.5 --pseudocount 1 \
  --trial-schedule independent --verify-synchronized --verify-materialized \
  --save-densities /scratch/path/device-densities.npz \
  --output /scratch/path/device.json
```

Compact measurements, raw warmed durations, source checksums, validation and
fit-time projections are recorded under `cross_gpu_subject_benchmark` in
[dawa_benchmark_results.json](dawa_benchmark_results.json).

## H100 profiling and further optimization experiments (2026-09-24)

Profiled the implementation committed as `ab0e681c75` on one idle H100 NVL
on `della-rse`. The existing isolated snapshot was reused; all 269 Python
files under `psyneulink/` and the DAWA benchmark directory matched the current
checkout. Nsight Compute **2026.2.1** was available through
`module load cudatoolkit/13.0`. These measurements used scratch implementations;
the compiler implementation and its retest are recorded in the next section.

The workload is unchanged: subject 1, 760 trials / 720 scored, **100,000
estimates per trial per proposal**, noise SD 0.1 in all four LCAs, dt 0.01,
100 RT bins, smoothing sigma 0.5, pseudocount 1, independent trial advancement
and grouped Philox normals. Timings below are end-to-end medians of five
unprofiled warmed evaluations of the same four-proposal batch, divided by four.
Startup/JIT is excluded. Except for the explicit launch experiments, all use
32 lanes, one warp and no register cap.

### Where the current kernel spends its work

For the first proposal, the Torch profile measured 0.49581 s in the simulation
kernel and 0.49589 s across all GPU kernels: **99.985% of GPU kernel time is
simulation**. Copies took 0.116 ms. The instrumented CPU objective range was
0.52554 s, so the GPU-only fraction should not be mistaken for a fraction of
end-to-end time.

Nsight measured **219 registers per thread**, 12.5% theoretical occupancy and
12.12% achieved occupancy. Instruction issue slots were busy 61.93% of the time;
36.75% of scheduler cycles had no eligible warp. DRAM bandwidth was only
19.2 MB/s. This points toward reducing arithmetic and live registers, rather
than bulk memory or transfer optimization. Triton reports eight static spills,
but Nsight measured zero dynamic local-memory spilling requests in this kernel.

Instruction-position sampling collected 56,350,066 samples with no dropped
bytes or buffer overflow:

| Source category | Share of samples |
| --- | ---: |
| Gaussian conversion / Box–Muller | 36.20% |
| Philox integer generator | 13.51% |
| Uniform conversion and other RNG helpers | 3.05% |
| LC Euler integration | 15.77% |
| Other model arithmetic and scheduler | 24.32% |
| Histogram and diagnostics | 4.94% |
| Block reductions | 2.21% |

These are shares of sampled warp instruction positions, **not exclusive
wall-time percentages**. They identify candidates for experiments; they do
not predict additive speedups. The sine/cosine source line includes general
range handling and branches in the generated machine code.

### Measured experiments

| Variant | Seconds/proposal | Speedup | Registers/thread | Estimated 5,000 evaluations |
| --- | ---: | ---: | ---: | ---: |
| Current compiler | 0.4343 | 1.00× | 219 | 36.2 min |
| Specialize fixed parameters | 0.3316 | **1.31×** | 128 | 27.6 min |
| Bounded-angle Gaussian transform | 0.3266 | **1.33×** | 219 | 27.2 min |
| Both experiments | 0.2707 | **1.60×** | 124 | 22.6 min |
| 64 lanes / 2 warps | 0.5134 | 0.85× | 240 | 42.8 min |
| 128 lanes / 4 warps | 0.5540 | 0.78× | 240 | 46.2 min |
| Register cap 168, 32 lanes / 1 warp | 0.4304 | 1.01× | 168 | 35.9 min |
| Register cap 128, 32 lanes / 1 warp | 0.4556 | 0.95× | 128 | 38.0 min |

The fit estimates count **parameter evaluations, not generations**. These
are throughput extrapolations for the fixed proposal pool, not measured
optimizer convergence; other parameter values can change simulation length.
The current-compiler median agrees with the earlier H100 measurement. One of
its five repetitions was slower (0.5140 s/proposal); the other four were
0.4336–0.4344 s/proposal. Raw durations are retained in the JSON summary.

**Fixed-parameter specialization is the first implementation recommendation.**
The model exposes 115 scalar parameters, but this fit changes only seven names
(eight fitting coordinates because LC mode depends on previous congruency).
The experiment emits explicit FP32 constants for the other 108, while asserting
that their supplied values equal their defaults. There are 46 zero and 30 unit
defaults in the full interface. Specialization lets Triton remove redundant
arithmetic and reduce live values. It does not linearize the LCAs, change time
steps, freeze controller outputs, or remove retained state. All **3,040 trial
densities and all four scores matched the baseline exactly** in this experiment.

A general implementation should explicitly declare which parameters are fixed,
include their values in the compilation cache key, and reject overrides or
recompile when they change. Fitted parameters, trial-dependent values and
controller modulation must retain their existing semantics. Equality within
one candidate batch is not sufficient evidence that a parameter stays fixed
throughout fitting. Broader model and seed coverage is still needed before
treating the prototype as a supported compiler feature.

**A bounded-angle Gaussian transform is the second recommendation.** The
experiment changes only sine and cosine in the five Box–Muller pairs to CUDA
`libdevice.fast_cosf` and `fast_sinf`. Their angles come from uniform draws
multiplied by 2π. Philox, its counter mapping, uniform conversion, clamp,
logarithm and square root are unchanged. A distinct emitted helper and compiled
kernel hash verify that the alternative was actually compiled.

This changes floating-point rounding. Across 4,194,304 paired draws, maximum
normal-value difference from the current transform was `3.10e-6`, with RMS
difference `2.63e-7`. In the complete subject benchmark, 121 of 3,040 densities
changed; maximum absolute density difference was `4.53e-4`, maximum relative
difference `8.58e-4` (0.086%), and maximum absolute log-likelihood difference
was **0.00055**. Adding fixed-parameter specialization produced exactly the
same densities as the fast-transform experiment alone. All runs completed
with strict truncation checks and repeatable results within each variant.

Those checks establish a promising performance experiment, not full numerical
or statistical validation. Before adoption, use a versioned or opt-in transform
and test multiple seeds, tails, component widths, retained-state trajectories
and GPUs. Preserve the existing mode for seeded replay.

Larger blocks and forced register caps offer little benefit here. After the
two substantive changes, reprofile before further scheduler simplification or
Gaussian stream packing: the relative costs will have changed. Histogram
smoothing and transfers are lower priorities on the current profile.

### Artifacts and reproduction

Raw reports, traces, benchmark JSON, saved densities and scratch experiment
scripts are retained remotely under:

```text
/scratch/gpfs/CSES/dmturner/dawa-benchmarks/trial-sync-20260924T212200Z/profile-h100-20260924
```

`env.sh` selects the isolated environment and GPU. `sweep.sh` runs the baseline,
launch experiments and fixed-parameter experiment; `fast-sweep.sh` runs the
alternative transform and combined experiment. `normal_validation.py` performs
the paired transform check. `hardware.ncu-rep`, `pc.ncu-rep` and their summaries
contain the profiling evidence. `summary.json` is the compact result copied to
`h100_optimization_profile` in [dawa_benchmark_results.json](dawa_benchmark_results.json).
The scratch scripts are experimental tools, not a public compiler API.

## Implemented specialization and fast Gaussian transform (2026-09-24)

Both optimizations now use the compiler's ordinary APIs. The H100 retest below
ran the working-tree implementation based on `ab0e681c75`, without monkeypatches
or experimental wrappers. The final source snapshot was verified by file hash
before execution. An initial pass and a repeat after tightening alias-override
validation gave consistent results; this table uses the final pass.

### Compiler interfaces

- `BatchedCompositionCompiler.compile(..., fixed_parameters={...})` and
  `plan.specialize_parameters({...})` produce a plan with explicit scalar input
  constants. The source defaults and controller semantics remain intact.
  Constants enter the generated source/cache key. Omitted values use the
  specialization; conflicting scalar, candidate-vector and trial-varying
  overrides are rejected, including overrides supplied through aliases.
- `PECOptimizationFunction(..., batched_specialize_fixed_parameters=True)`
  specializes all non-fitted defaults. The seven DAWA fitting parameters remain
  dynamic, including condition-dependent LC mode; the other 108 become constants.
- `normal_rng="philox4x_fast_v1"` selects the bounded-angle Box–Muller transform.
  It covers grouped vector draws, scalar/odd-width tails, cached scalar pairs,
  and direct `tl.randn` calls in registered scalar templates such as standalone
  DDM and NormalDist. Uniform streams and counter addressing are preserved.
  The CPU interpreter and handwritten CSI oracle explicitly reject this mode.

General defaults remain unspecialized with `philox4x_v1`. The DAWA subject
benchmark now defaults to specialization and `philox4x_fast_v1`; the previous
baseline is available with `--no-specialize-fixed-parameters --normal-rng
philox4x_v1`. The [DAWA README](README.md#fixed-parameters-and-faster-gaussian-conversion)
shows the PEC configuration and benchmark command.

### H100 measurements

One H100 NVL on `della-rse`, the same subject and four proposals, **100,000
estimates per trial per proposal**, 760 trials / 720 scored, noise SD 0.1 in all
LCAs, dt 0.01, smoothing sigma 0.5 and pseudocount 1. All use independent trial
advancement, 32 lanes, one warp and no register cap. Each entry is the median of
five warmed complete four-proposal evaluations, divided by four, with startup
excluded. Serial proposals are separate objective calls; batch four submits
the four proposals together. Times include PEC preparation and scoring.

| Implementation | Serial, s/proposal | Batch four, s/proposal | Batch-four speedup | 5,000 evaluations, batch four |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.4502 | 0.4336 | 1.00× | 36.1 min |
| Fixed parameters | 0.3589 | 0.3321 | 1.31× | 27.7 min |
| Fast Gaussian conversion | 0.3443 | 0.3273 | 1.32× | 27.3 min |
| Both | **0.3007** | **0.2718** | **1.60×** | **22.6 min** |

For a serial optimizer, the combined implementation gives a **1.50× speedup**
and projects to **25.1 minutes for 5,000 evaluations**, versus 37.5 minutes for
the baseline. These are throughput extrapolations, not convergence measurements;
candidate values affect trial lengths, and setup/JIT, optimizer overhead and
queueing are additional. Evaluations count proposals, not optimizer generations.

The optimized first-proposal kernel uses **124 registers/thread and no reported
spills**, versus 219 registers/thread in the earlier baseline profile. Peak
live Torch fitting buffers remain 0.3218 MiB for serial proposals and 0.4961 MiB
for batch four; context, compiled code and allocator reservations are additional.
PyTorch `2.13.0+cu130`, Triton `3.7.1`, CUDA `13.0` and Python `3.13.13` match
the earlier H100 environment. GPU tests ran separately from timing.

### Validation

- Specialization alone matches all **3,040 baseline trial densities exactly**.
  The fast-transform and combined variants also match each other exactly.
  Each variant reproduces its own densities across repeats and candidate batch
  sizes, and all strict truncation checks passed.
- The fast transform changes 121 densities relative to the baseline. Maximum
  absolute difference is `4.53e-4`; maximum relative difference is `8.58e-4`
  (0.086%). Maximum absolute log-likelihood difference is `0.00061` using the
  serial FP32 objective, or `0.000624` when summing logs of the saved densities
  in FP64. The earlier scratch table used batch-four FP32 reductions, whose
  summation order gives slightly different score differences.
- The combined variant matches synchronized trial execution exactly. Its fused
  smoothed densities match the materialized reference within `3.44e-7` relative
  error (tolerance `2e-6`).
- A full-sequence distribution comparison used **16,384 estimates per generator**
  with independent seeds 29 and 1,000,029. For each of four candidates, all
  760 trials and both choices were compared through joint choice/RT sub-CDFs.
  Maximum differences ranged from 0.01984 to 0.02204, below the conservative
  simultaneous bound 0.04237 at familywise alpha 0.01. This includes unscored
  trials and their retained-state histories.
- GPU unit checks cover normal moments, fourth moments, correlations and tails;
  two additional seeds each exercise 1,048,576 four-normal groups. Paired
  same-uniform comparisons stay within `5e-6`. Other checks cover widths 1–32,
  odd widths, high seed bits, multiple streams, replay, common randomness,
  resume, launch geometry, scalar templates, controller resets and full DAWA
  trajectories under both modes and two seeds.

The H100 regression suite passed **129 tests**; a targeted final implementation
follow-up passed 45, and the final specialization/override suite passed 16.
CPU/interpreter and IR checks passed 120 tests, the final parameter/guard suite
passed 28, and compilation/registry snapshot checks passed 29. These suites
overlap; counts are not additive. Opposite-backend cases and GPU-only transforms
were intentionally skipped. Ruff and `git diff --check` passed.

The fast mode is a numerical alternative, not bitwise replay of the old mode.
Neither optimization linearizes the model or changes its time discretization.

The source snapshot, manifest, launch scripts, raw timing/validation JSON,
densities, profile and test logs are retained at:

```text
/scratch/gpfs/CSES/dmturner/dawa-benchmarks/specialization-20260924
```

`env.sh` selects the isolated snapshot/environment and one H100;
`benchmark.sh` runs the four implementations and validation. Compact final
measurements are stored under `h100_implemented_optimizations` in
[dawa_benchmark_results.json](dawa_benchmark_results.json).


## H100 timestep refinement: 10 ms versus 1 ms (2026-09-24)

A fresh matched comparison used both compiler optimizations, the same four
parameter proposals and full 760-trial subject (720 scored), 100,000 estimates
per trial per proposal, all four LCA noise SDs 0.1, smoothing sigma 0.5 and
pseudocount 1. Medians of five warmed four-proposal pools, divided by four,
include PEC preparation, simulation and likelihood scoring; setup/JIT is excluded.

| Candidate batch size | 10 ms, s/evaluation | 1 ms, s/evaluation | Slowdown | 5,000 evaluations at 1 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 (serial) | 0.3008 | 2.6284 | **8.74×** | 219.0 min / 3.65 h |
| 4 | 0.2697 | 2.4834 | **9.21×** | 207.0 min / 3.45 h |

All LCA timesteps changed from 0.01 to 0.001. The LC integrator timestep also
changed proportionally, from 0.02 to 0.002, retaining ten internal LC updates
per scheduler pass. This preserves the relative LC/LCA clock rate. Leaving
the LC timestep fixed would accelerate it tenfold relative to the LCAs and
would be a different comparison. The trial step cap increased from 2,000 to
20,000, retaining a 20-second decision-time cap. Noise amplitudes were unchanged;
the integrators apply their ordinary square-root-of-dt scaling.

The configuration wrapper verified the constructed mechanism/integrator clocks;
the reports' five specialized timestep constants were checked independently.
Strict truncation checks passed, repeated runs and candidate batch sizes gave
identical trial densities within each timestep, and the new 10 ms densities
exactly reproduce the preceding optimized H100 run. Different timesteps are
not expected to produce identical likelihoods; this is a runtime comparison,
not a demonstration that the model's predictions have converged in timestep.
Fit times are throughput extrapolations, not optimizer convergence measurements.

Compiler and timed objective code were unchanged. A configuration-only wrapper,
reproduction script, raw reports, saved densities and logs are retained under:

```text
/scratch/gpfs/CSES/dmturner/dawa-benchmarks/specialization-20260924/timestep
```

The compact measurements are recorded as `h100_timestep_refinement` in
[dawa_benchmark_results.json](dawa_benchmark_results.json).
