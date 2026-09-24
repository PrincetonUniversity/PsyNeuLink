# Matched CSI and DAWA first-passage benchmarks

The reversal survives a matched workload: direct evaluation is faster for CSI,
while warmed GPU sampling is faster for DAWA. The two-dimensional density grid
is the main explanation, compounded by the current explicit stability-limited
DAWA time integrator. The conclusion also survives checking numerical refinement;
DAWA's cheaper coarse grids have materially larger distribution error.

The [benchmark driver](benchmark_csi_dawa_first_passage.py) and
[curated measurements](benchmark_results/csi_dawa_first_passage.json) record
the workload, parameter values, source hashes, repeated timings, and checks.

## What is matched

One candidate, two conditional trials, a **1.4 s decision horizon**, noise
amplitude **0.1**, and **100,000 GPU samples per trial**. Both samplers use
**0.25 ms Euler steps with Brownian-bridge corrections** and float32 stochastic
states. Both direct solvers use float64 and 1 ms coefficient intervals. Each
method reports both choice probabilities, censoring, and a final 25 ms choice-0
interval probability. Censored samples remain in the denominator.

Each model's nonlinear deterministic paths are computed using its existing
native equations. Their construction is timed separately. The timings below
cover the stochastic stage, including transfers and host probability summaries.
These are medians of three synchronized warm calls on an i7-9700K and RTX 2080 Ti;
compilation and CUDA graph capture are excluded. Runs did not overlap.

The two CSI conditions use gains 10/20, thresholds 0.12/0.12, and collapse rates
-0.01/-0.02 per second, after a 1 s ITI and 50 ms cue. DAWA uses the prior
performance candidate and its two incongruent conditions. Full values are in
the JSON. Reaction times need not match across these different models: mean
decision durations including censoring were 0.84/0.61 s for CSI and 0.64/0.56 s
for DAWA. At this horizon the CSI censored fractions were 19.35%/3.46%; DAWA's
were 0.002%/0%. The same finite-horizon probabilities are computed by both
methods within each model.

The 16-trial test repeats each pair eight times, computing every distribution
separately. It checks workload scaling without any distribution caching. These
are conditional first-passage benchmarks, not sequential subject fits. Neither
model's history handling or resets were changed.

## Matched timings

The table uses **129 spatial points per coordinate**: CSI has 127 interior
density values, while DAWA has 129×129 = 16,641 cells. The fastest measured
129-point direct backend for each model is shown.

| Model | Conditional trials | Direct backend | Direct time | GPU sampling | Faster method |
| --- | ---: | --- | ---: | ---: | --- |
| CSI | 2 | CPU, 4 threads | 4.23 ms | 52.41 ms | Direct, 12.4× |
| DAWA | 2 | GPU | 410.66 ms | 50.78 ms | Sampling, 8.1× |
| CSI | 16 | CPU, 4 threads | 16.25 ms | 326.01 ms | Direct, 20.1× |
| DAWA | 16 | GPU | 3,350.18 ms | 414.36 ms | Sampling, 8.1× |

For two trials, deterministic-path construction took 2.04/3.10 ms for the
CSI/DAWA direct paths and 3.60/9.66 ms for their finer sampling paths. At 16
trials these were 11.72/25.05 ms and 25.47/75.79 ms. Those costs do not explain
the reversal. Adding them to stochastic-stage medians is only an estimate of
combined time, not a separately measured full-objective timing.

The CPU-only comparison isolates the large direct-solver difference on the same
processor. At 129 points, one thread takes **7.91 ms for CSI** and **3.743 s for
DAWA**, about **473×** longer for DAWA. At 33 points, where DAWA needs no extra
stability substeps, its single-thread solve is still about **69×** slower.

| Points per coordinate | CSI CPU, 1 thread | DAWA CPU, 1 thread | DAWA GPU | DAWA substeps per 1 ms |
| ---: | ---: | ---: | ---: | ---: |
| 33 | 1.99 ms | 137.97 ms | 124.22 ms | 1 |
| 65 | 3.86 ms | 596.11 ms | 158.85 ms | 2 |
| 129 | 7.91 ms | 3,743.40 ms | 410.66 ms | 5 |

CSI parallelizes independent density lanes across CPU threads. DAWA's CPU
backend parallelizes cells within a trial, and its GPU backend currently
processes trial densities consecutively. This contributes to the different
batch scaling. Even the single-thread, two-trial comparison has a very large
gap; trial batching alone is not its cause.

## Numerical accuracy

We compared joint choice/time CDFs at 100 ms intervals through 1.4 s against a
257-point-per-coordinate direct reference with 0.5 ms coefficient intervals.
The reference is a finer numerical approximation, not exact truth. Entries
below are maximum absolute errors in **percentage points**, across both choices,
both trials, and all cutoffs.

| Direct grid | CSI CDF error | DAWA CDF error |
| ---: | ---: | ---: |
| 33 | 0.163 | 4.241 |
| 65 | 0.039 | 1.138 |
| 129 | 0.0077 | 0.240 |

At 257 points, changing coefficient intervals from 1 ms to 0.5 ms changes the
CDFs by at most 0.00017 percentage points for CSI and 0.000041 for DAWA. This
isolates temporal refinement at that grid; it does not bound remaining spatial
error in the reference.

Across three independent seeds with 100,000 samples per trial, 0.25 ms sampling
errors were **0.180–0.400 percentage points for CSI** and **0.217–0.250 for DAWA**.
Halving the sampling step to 0.125 ms gave **0.119–0.265** and **0.135–0.371**,
respectively. At this sample count, seed variation remains substantial relative
to the change in time step.

Thus CSI's 33-point solve already has a CDF error of roughly the Monte Carlo
scale in this audit. DAWA needs the 129×129 grid to reach that range among the
timed meshes. Using DAWA's 33×33 or 65×65 timings as equally accurate alternatives
would obscure a substantial numerical difference. CDF agreement does not by
itself establish equal accuracy of narrow RT-bin likelihoods, especially tails.

## Why the cost grows

* **Grid size:** a scalar CSI DDM needs O(N) density values; the two coupled
  DAWA response accumulators need O(N²). At the 129-point setting there are about
  131× as many active density cells in DAWA.
* **Time integration:** CSI uses an implicit tridiagonal solve. DAWA uses explicit
  SSP-RK2 with two stages per substep. Here its stability check requires five
  substeps per millisecond, so it performs ten grid stages for each CSI macrostep.
  That is about 1,310× as many cell-stage visits before accounting for the different
  arithmetic per visit; it is not a prediction of the measured wall-time ratio.
* **Refinement:** doubling DAWA's resolution roughly quadruples its cells and can
  also increase the number of stability substeps. In the diffusion-dominated
  asymptotic regime, this explicit scheme can approach O(N⁴) work at fixed horizon.
* **Sampling:** another stochastic coordinate adds per-trajectory state and
  arithmetic, not a tensor-product grid. Independent trajectories keep their small
  states local throughout a fused GPU loop. The direct GPU solver exchanges
  neighboring-cell mass through memory between ordered stages and uses float64.

This is the dimensionality penalty the model comparison suggested. Nonlinear
spatial coefficients, the chosen time integrator, hardware, and implementation
overhead also affect the ratio. The measurements are not a lower bound on what
a better 2D solver could achieve. An implicit or split-direction method that
reduces stability substeps is a more targeted next investigation than additional
CPU threads; it would need its own conservation, positivity, and gradient checks.

## Fitting and compilation limits

The CSI timing uses a new research-local
[continuous GPU sampler](csi/csi_fit/direct_likelihood/continuous_monte_carlo.py),
validated against the existing independent first-passage solver. It uses the same
continuous crossing convention as the DAWA validation sampler, rather than CSI's
scheduled PEC fitting sampler. The original production fitting route is unchanged.

The DAWA validation sampler currently specializes its kernel on `BIAS`. A separate
probe changed bias by 0.000137: the first sampling call took **0.753 s**, followed
by **0.061/0.053 s** on repeated calls. This includes new-specialization overhead,
which is excluded from the warmed table. Changing bias during fitting can incur
that cost repeatedly. Moving fitted values to runtime kernel inputs would be
necessary before treating these warmed times as a prediction for optimization.
The direct GPU backend can likewise need fresh CUDA graph capture when its
numerical topology changes. No optimizer or full-fit speedup was measured here.

The new sampler's two functional tests and two style checks passed. The benchmark
also checks CPU-thread/GPU numerical parity, probability conservation, positive
density, and DAWA lower-domain truncation. Refinement arrays are retained in the
full output; the curated JSON keeps errors, timings, and settings.

## Reproduction

From the repository root, with CUDA, Triton, Ninja, and an OpenMP-capable compiler:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONWARNINGS=ignore .venv/bin/python \
  Scripts/Debug/pec_batch_compile/benchmark_csi_dawa_first_passage.py \
  --repeats 3 --estimates 100000 --trials 2 16 \
  --output /tmp/csi_dawa_first_passage.json

PATH="$PWD/.venv/bin:$PATH" PYTHONWARNINGS=ignore .venv/bin/pytest -n 0 \
  tests/composition/pec/test_csi_continuous_sampler.py -q
```

`--skip-refinement` omits the separate numerical audit. `--horizon` can change
the common decision duration (it must align with 1 ms). Repeating the full
command with another horizon or model candidate is needed to generalize the
timing ratios beyond these cases.
