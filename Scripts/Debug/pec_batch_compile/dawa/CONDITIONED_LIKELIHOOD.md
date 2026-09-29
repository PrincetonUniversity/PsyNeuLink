# Observation-conditioned GPU fitting

The DAWA fit and recovery drivers now default to `--likelihood conditioned`.
Each simulated particle carries its own latent state, and each observed
choice/RT updates the distribution of states entering the next trial. The
previous `--likelihood marginal` objective simulated complete stateful histories
but did not update them using earlier observations.

This uses the original composition's 10 ms LCA dynamics, with Gaussian noise in
all four LCA layers. It does not use the continuous direct-likelihood prototype.

## State and observation semantics

The ordinary generated simulator executes one trial per particle. The filter
then weights and systematically resamples complete terminal states. State
transport includes held controller outputs and the values last sampled by
their targets, because next-trial resets can use those sampled values before a
controller publishes again. DAWA's flat state buffer has 47 floats: 27 mechanism
states, 10 held control values, and 10 sampled values. The transport is a generic
compiler feature.

Conditioning uses a histogram observation kernel, rather than exact continuous
choice/RT densities. The default RT bin width is 30 ms and Gaussian smoothing
has standard deviation 0.5 bins. For in-range simulated outcomes, the Gaussian
kernel is normalized over possible **observed** bins, including at domain edges.
Simulated outcomes outside the finite domain retain implicit overflow mass;
they are not folded into an edge bin or removed by renormalizing the particles.

A symmetric pseudocount is interpreted as uniform observation contamination.
With `N` particles, `K` joint choice/RT bins, and pseudocount `alpha` per bin, the
contamination probability is `K*alpha/(N + K*alpha)`. Each particle receives its
ordinary observation weight plus `alpha/N`. This supplies a defined ancestry
distribution even when no simulated response matches the observation: the
filter then retains the predictive state distribution. At 100,000 particles,
100 RT bins, two choices, and alpha=1, contamination probability is about 0.2%.
Rescoring at another particle budget scales alpha to keep that probability fixed.

All retained observations condition state, including trials excluded from the
log score. The mask selects which conditional log densities to sum; it does not
mean an observation is missing. The result is the full sequence likelihood under
this observation model only when all terms are included. Masked RTs above 3 s
expand the domain in 30 ms increments. Missing observations are not supported.

The exact noisy-model likelihood is still approximated by finite particles and
the declared observation kernel. Effective sample size and posterior
contamination responsibility are available through
`conditioned_log_likelihood(..., return_diagnostics=True)`.
The [accuracy study](CONDITIONED_ACCURACY.md) tests the filtering recursion
against independent references and measures particle-budget sensitivity.
The follow-up [H100 recovery and profile study](CONDITIONED_RECOVERY.md) measures
complete fits with matched synthetic observations and examines LC identification.

## Execution and validation

The prepared CUDA path validates and emits the kernel once per likelihood call,
uploads the sequence's inputs and parameters once, and reuses output, diagnostic,
and state buffers between trials. It executes the ordinary simulation kernel;
there is no DAWA-specific dynamics implementation. `execution="reference"`
retains the slower one-trial `plan.run()` loop for differential tests.

Prepared CUDA filtering now also combines categorical matching, continuous
bin searches, and observation-table lookups into one general FP32 kernel.
The Gaussian tables and Torch score reductions are unchanged. Systematic
resampling keeps the reference FP64 cumulative sum and random offsets, then
combines ancestor search and complete-state gathering into a reusable buffer.
This avoids allocating another full state copy at every trial. It supports
contiguous FP32 state buffers up to 128 fields; other cases retain Torch
gathering. Multinomial resampling and CPU execution retain their reference
implementations. Unused normalized weights are omitted from score-only
systematic filtering; diagnostics still compute the original ESS values.

Tests cover exact split-versus-continuous seeded simulation, changing controller
values and trial parameters, both trial schedules, and prepared-versus-reference
filtering. Changing an earlier unscored observed RT changes subsequent filtered
predictions while leaving subsequent marginal predictions unchanged. Independent
finite-state tests compare the filtering recursion against an analytically
known posterior and sequence likelihood. Observation-kernel tests check boundary
normalization, overflow mass, contamination ancestry, and zero-support errors.

Candidate batches share systematic resampling offsets when common random
numbers are requested. Evaluating a candidate alone or in a batch therefore
preserves its resampling stream. The cumulative raw weights and systematic
positions use FP64: FP32 scan/reduction rounding changed particle ancestry
when the candidate batch size changed. Model simulation remains FP32; the
reported log scores can still differ at the level of FP32 reduction rounding.
Future simulation noise remains indexed by
destination particle and original trial number, so duplicate offspring do not
duplicate their future noise.

## Conditioned-loop optimization, 2026-09-29

The baseline was frozen in a separate worktree at **`b53de60067`**. The optimized
compiler was measured on the same RTX 2080 Ti, complete recorded subject 1
(760 retained / 720 scored trials), four fixed proposals, model-construction
seed 29, and likelihood seeds 29–31. Both use 100 RT bins over 0–3 s,
sigma 0.5 bins, pseudocount `N/100000`, all four noise SDs 0.1, strict cap
4,000, fixed-parameter specialization, and the existing 32-particle/one-warp
launch with `philox4x_fast_v1`. No dynamics, observations, or particle budgets
were changed.

These are medians of three warm **score-only** calls, including sequence
preparation and error checks, excluding compilation and optimizer overhead.
Diagnostics were collected separately. GPU tests did not run during timing.

| Particles | Candidates | Baseline s/batch | Optimized s/batch | Speedup | Baseline → optimized peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 1.618 | 1.082 | 1.50× | 6.35 → 4.78 |
| 10,000 | 4 | 1.446 | 1.052 | 1.37× | 23.29 → 16.58 |
| 100,000 | 1 | 1.889 | 1.795 | 1.05× | 57.23 → 41.59 |
| 100,000 | 4 | 5.490 | 5.398 | 1.02× | 227.39 → 159.97 |

Peak allocation is live Torch tensor memory, excluding context, allocator
reservation, and other processes. The four-candidate 100k case uses **30% less
tensor memory**. Its timing improvement is small; this does not establish a
substantial speedup for full-budget fitting. At the measured proposal mix,
5,000 evaluations in batches of four project to about **1.87 hours**, versus
1.91 hours for this baseline, before optimizer and validation overhead.

Profiling the 100k/four-candidate case reduced kernel launches from **48,588
to 25,032**. Recorded CUDA kernel durations were:

| Operation | Baseline ms | Optimized ms |
| --- | ---: | ---: |
| Trial simulation | 4,657.0 | 4,743.1 |
| FP64 cumulative sum | 284.7 | 290.3 |
| Ancestor search and state gathering | 196.6 | 169.5 |
| All recorded CUDA kernels | 5,370.7 | 5,335.5 |

Simulation accounts for about **89%** of the optimized kernel time. The changes
mainly reduce launch overhead and memory; the simulator itself is unchanged.
CUDA kernel durations and host cProfile times are different quantities: host
time attributed to a weighting call can include waiting for preceding device
work. It should not be interpreted as pure observation-weight computation.

Two launch experiments also preserved all recorded scores and diagnostics.
Neither changes the default:

| Experiment, 100k × four candidates | Median s/batch | Finding |
| --- | ---: | --- |
| 64 particles / two warps | 6.257 | Slower than the existing launch |
| 32 particles / one warp, register cap 128 | 5.207 | About 4% faster than the optimized default here, with two reported spills |

The register cap needs broader workloads before becoming a default. After
Della access was restored, the [H100 profile below](#h100-benchmark-and-profile-2026-09-29)
tested the same compiler changes and register cap. The existing independent
advancement *within* each trial remains; the observation update between trials
is still required.

Validation includes **57 passing tests**: exact per-particle observation weights
(multiple continuous/category outputs, boundaries, nonfinite/overflow outcomes,
and smoothing), reference systematic ancestors and RNG states, complete-state
gathers, finite-state filtering oracles, and DAWA prepared/reference parity.
On the complete subject, all **30 final scores** across particle budgets,
candidate batches, and three seeds match the frozen baseline exactly. All
**7,600 per-trial densities**, plus corresponding ESS, contamination fractions,
and zero-support indicators, match for the diagnostic seed 29. This establishes
execution equivalence for these cases, not new particle-convergence evidence.

The [compact optimization report](fitting_acceleration/conditioned_optimization_2080ti.json)
preserves timings, scores, diagnostic digests, source/data hashes, and profiler
summaries. Large raw traces remain outside the repository. To reproduce the
optimized measurement, use a fresh output path and the separately supplied data:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_conditioned_benchmark.py \
  --data "$DAWA_DATA" --subject 1 --likelihoods conditioned \
  --estimates 10000 100000 --batch-sizes 1 4 --max-steps 4000 --repeats 3 \
  --profile /tmp/dawa-optimized-profile --output /tmp/dawa-optimized.json
```

Run the same command against an isolated checkout of `b53de60067` for the
frozen baseline, with its checkout first on `PYTHONPATH`. The current
`--execution reference` mode is a correctness oracle; it includes repeated
per-trial preparation and is not the performance baseline. The new benchmark
options `--block-size`, `--num-warps`, and `--maxnreg` reproduce launch tuning.

## H100 benchmark and profile, 2026-09-29

The matched experiment ran on **one H100 NVL on `della-rse`**, using physical
GPU 0. GPU 1 was left available to other users throughout. Both source variants,
tests, and profilers ran sequentially on the same device; both GPUs were idle
after completion. For current daytime work on this shared host, use at most
one H100 and leave the second available to others.

The model, complete subject, proposals, seeds, 100k-particle observation model,
and execution cap are identical to the local comparison above. The baseline
compiler is frozen at `b53de60067`; the optimized compiler file hashes match
the local experiment. A common benchmark harness adds explicit source-revision
metadata for the isolated snapshots. Source/data manifests verify 395 baseline
and 397 optimized files. Both environments use Torch `2.13.0+cu130` and Triton
`3.7.1`. Warmup is excluded; a shared Triton cache means the reported warmup
times are not comparable cold-start measurements.

| Particles | Candidates | Baseline H100 s/batch | Optimized H100 s/batch | Compiler speedup | Optimized 2080 Ti s/batch | H100 system speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 0.547 | 0.390 | 1.40× | 1.082 | 2.77× |
| 10,000 | 4 | 0.566 | 0.460 | 1.23× | 1.052 | 2.29× |
| 100,000 | 1 | 0.922 | 0.845 | 1.09× | 1.795 | 2.12× |
| 100,000 | 4 | 2.271 | 2.188 | 1.04× | 5.398 | 2.47× |

These remain medians of three warm score-only evaluations. At 100k particles,
four candidates, peak tensor allocation is **159.97 MiB**, versus 227.39 MiB
before optimization. At this proposal mix, 5,000 evaluations in batches of four
project to **45.6 minutes** on H100, versus 47.3 minutes for its baseline and
112.5 minutes for the optimized 2080 Ti. These are throughput projections;
optimizer overhead, validation, compilation, and changing proposal difficulty
are additional. They are not a new full-fit measurement or convergence claim.

The GPU comparison also changes host and OS: Xeon Gold 6548Y+ under native
Linux versus Core i7-9700K under WSL2. Treat the wall-time ratios as comparisons
of the complete systems. For the same large case, recorded simulation kernel
time itself is about 2.54× lower on H100.

### Where the time goes

Nsight Systems measured the first timed score call after warmup, at 100k
particles and four candidates. Its optimized call took 2.197 s, close to the
uninstrumented median of 2.188 s. CUDA activity occupies **96.8%** of this
profiled interval. The profile therefore does not support expecting a large
full-budget gain from removing host launch gaps alone.

| Work in optimized call | Time, ms | Share of profiled wall time |
| --- | ---: | ---: |
| Trial simulation | 1,867.0 | 85.0% |
| FP64 cumulative resampling sum | 128.4 | 5.8% |
| Ancestor search and state gathering | 59.4 | 2.7% |
| Observation contributions and other CUDA kernels | 70.6 | 3.2% |
| Copies and memsets | 1.3 | 0.1% |
| Without recorded GPU activity | 70.0 | 3.2% |

Kernel launches decreased from **48,588 to 25,032**. In the baseline profile,
simulation took 1,867.8 ms; in the optimized profile it took 1,867.0 ms. The
filtering changes save work around an unchanged simulator. Host-to-device and
device-to-host copies total only **0.015 ms** in the optimized call. Memory
allocation and transfer volume are not the principal time bottleneck here.

### Simulation registers and launch tuning

Nsight Compute replayed trial 41 of the measured sequence, after a complete
warmup and the preceding conditioned trials. These hardware counters describe
one trial, not an average across the entire subject or fitting domain.

| Metric | Existing launch | Register cap 128 |
| --- | ---: | ---: |
| Registers per thread | 133 | 128 |
| Resident blocks allowed by registers, per SM | 12 | 16 |
| Theoretical occupancy for that limit | 18.75% | 25.0% |
| Achieved occupancy | 17.0% | 22.3% |
| Scheduler issue activity during active cycles | 67.1% | 72.9% |
| Eligible warps per scheduler per active cycle | 1.07 | 1.48 |
| DRAM throughput, percentage of peak | 1.08% | 1.15% |
| Measured local spill requests | 0 | 0 |
| Profiled single-kernel duration | 3.005 ms | 2.805 ms |

A separate **uninstrumented full-subject** check with `--maxnreg 128` took
**2.070 s/batch**, another **1.057×** speedup over the optimized default, or
about **43.1 minutes per 5,000 proposals** at this batch size and proposal mix.
All scores across three seeds and the seed-29 diagnostic arrays match exactly. The cap
changes generated instruction/register allocation, not the model or particle
budget. It remains an optional experiment; the fit driver's default launch
has not changed. The local GPU also benefited modestly, but reported spills
there, so the hardware tradeoff is not identical.

The next compiler target is reducing simultaneously live values and arithmetic
in the simulation kernel. The register cap provides a measured reason to
investigate register lifetimes; occupancy alone does not predict a proportional
speedup. The FP64 resampling scan is a secondary target at about 6% of wall
time. Its numerical stability and candidate-batch invariance must be preserved.
Further work on observation-weight fusion or bulk transfers has limited room
to improve this large-budget case.

### Checks and reproduction

All **57 tests passed on H100**. All 30 benchmark scores and all 7,600 per-trial
density entries, plus corresponding ESS, contamination and zero-support
arrays, match the frozen H100 baseline. They also match the local benchmark
exactly for these cases; that is not a general promise of replay across GPUs.
The Nsight Systems and both Nsight Compute replays preserve their matching
scores and diagnostics as well.

The [compact H100 report](fitting_acceleration/conditioned_optimization_h100.json)
contains timings, source hashes, comparisons, profiler summaries, and hardware
counters. Frozen sources, the complete manifests, raw traces, and launchers are
retained on `della-rse` under:

```text
/scratch/gpfs/CSES/dmturner/dawa-benchmarks/conditioned-optimization-20260929
```

`env.sh` selects GPU 0, the existing Python environment, and an isolated source
variant. `run.sh` runs tests and the matched benchmarks; `profile.sh` and
`profile-cap128.sh` reproduce the hardware captures. They use fixed artifact
paths, so copy the launchers and update their root paths for a new results
directory before another run.
The public benchmark command above works on H100 as well. Add
`--estimates 100000 --batch-sizes 4 --maxnreg 128` for the cap experiment and
use a fresh output path. `--source-revision b53de60067` records the base revision
when measuring an isolated source snapshot without Git metadata.

## RTX 2080 Ti measurements, 2026-09-27

The complete recorded subject 1 has 760 trials, of which 720 contribute to the
score. These are medians of three warm evaluations with independent seeds,
using the original 10 ms model, noise SD 0.1 in all four LCAs, 30 ms RT bins,
0.5-bin smoothing, and a constant approximately 0.2% contamination probability.
Both methods use the same parameter proposals and observations but evaluate
different objectives. Timings include simulation, observation updates,
resampling, and returning scores to the CPU. Compilation is excluded.

| Particles | Candidates per batch | Previous marginal, s/batch | Conditioned, s/batch | Conditioned, s/candidate | Conditioned peak allocation, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 0.169 | 1.626 | 1.626 | 6.35 |
| 10,000 | 4 | 0.357 | 1.481 | 0.370 | 23.29 |
| 100,000 | 1 | 0.913 | 1.938 | 1.938 | 57.23 |
| 100,000 | 4 | 3.011 | 5.594 | 1.399 | 227.39 |

At 100,000 particles the conditioning cost is **2.12x** for one candidate or
**1.86x** for batches of four. At the latter rate, 5,000 parameter evaluations
would take approximately **1.94 hours**, before optimizer and validation
overhead. These four proposals do not establish an average over an actual
optimization run. Peak allocation is Torch tensor memory in the benchmark
process, excluding CUDA context, allocator reservation, and other processes.
The compiled simulation kernels used 127–141 registers and reported no spills.

Preparation mattered substantially: a 64-trial, 10,000-particle evaluation
took 2.87 s with the initial per-trial preparation loop and 0.146 s with the
prepared runner. Profiling identified repeated source validation/emission and
thousands of small parameter uploads. The prepared path removes that repetition.
Small particle budgets still show substantial per-trial launch overhead; the
10,000-particle case is not a tenfold speedup over 100,000 particles.

This establishes execution correctness and timing, not particle convergence
or fitting accuracy. At 100,000 particles, the first proposal's median ESS was
2,645 and minimum ESS was 56; across the four proposals the minimum was about
5. Low-support observations can therefore still produce noisy filtering and
scores. Particle-budget and independent-seed checks remain necessary before
drawing conclusions from a fitted subject. The subsequent
[accuracy study](CONDITIONED_ACCURACY.md) measures this uncertainty, and the
[conditioned recovery pilot](CONDITIONED_RECOVERY.md) exercises complete fits.
The earlier marginal recovery studies do not validate this conditioned objective.

The [compact results](fitting_acceleration/conditioned_2080ti.json) include all
timings and scores, ESS and contamination summaries, model/data/implementation
hashes, and warmup times. Large profiler traces are not included in the repo.

## Reproduction

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_conditioned_benchmark.py \
  --subject 1 --estimates 10000 100000 --batch-sizes 1 4 --repeats 3 \
  --output /tmp/dawa-conditioned-benchmark.json
```

The benchmark measures both objectives on the same ordered subject, parameter
proposals, model clocks, smoothing width, and contamination fraction. Warmup
and compilation are reported separately. Its fit-time projection assumes 5,000
parameter evaluations at the measured batch size; it is not a convergence claim.
Use `--execution reference` to measure the original per-trial setup overhead.

Cached NDT profiling and the existing adaptive pooling of per-trial density
blocks require `--likelihood marginal`. They cannot be reused unchanged when
the observations and NDT affect subsequent particle histories.
