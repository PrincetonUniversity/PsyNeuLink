# Compiled NDT profiling: implementation and H100 fits

Nondecision-time profiling is implemented and available with `--profile-ndt`
in both adaptive empirical fits and parameter recovery. Two full-subject H100
fits with reduced proposal budgets took **6.10 and 5.82 minutes overall**, versus
**8.47 and 8.08 minutes** for the previous adaptive fits: **1.39× faster**.
Their fresh-seed likelihoods were very close to the corresponding adaptive
fits. Several LC parameter estimates still differed.

The gain includes reducing search/refinement budgets from 2,000/600 to
1,000/300. Profiling alone does not make simulation faster: the standard-budget
profiled run took 8.49 minutes. Against the prior 27.54-minute fixed-100k
benchmark, the shorter profiled fits are **4.51–4.73× faster overall**. These
results are still short of the 10× target and cover only two starts on one
synthetic subject.

A control with the same smaller budgets but profiling disabled took 4.81
minutes, but lost **0.945 ± 0.111 log units** relative to the profiled fit.
This supports profiling as a way to preserve quality with fewer proposals in
this example; reducing proposal counts alone did not preserve it.

The earlier adaptive changes were committed as `2031e35b73`. This report
describes the subsequent NDT implementation. Full measurements, source hashes,
launch scripts, and configuration are in [ndt_h100.json](ndt_h100.json).
The [follow-up profile](ndt_profile_h100.md) breaks down the fastest successful
configuration and tests concurrent independent sampling blocks.

## What the implementation does

The new general compiler method `BatchedSimulationPlan.discrete_output_counts`
accumulates exact counts for a caller-declared finite numeric support, matched
to each trial's observed categories. It runs the same ordered histories and
retains the same latent state. It does not materialize every sample. Unsupported
numeric values, nonfinite outputs, and truncated histories are detected.

`ShiftedHistogramScorer` applies an explicit FP32 additive shift to each support
value, then uses the existing histogram boundaries, Gaussian smoothing, and
pseudocount rule. It precomputes sparse gathers and deduplicates identical bin
maps over the entire support. It does not assume that arbitrary model parameters
can be removed from simulation.

The DAWA adapter checks that NDT is a passive additive RT readout. At the tested
10 ms response timestep, the support contains 2,001 possible decision times.
The original 2,001-value NDT grid from 0.1 to 0.3 seconds produces only **21
distinct bin maps**. Each seven-coordinate CMA proposal simulates at NDT zero
and scores these maps from the resulting counts.

Adaptive blocks pool densities before choosing **one NDT for the entire
subject**, then take logs and rank candidates. Reference checks and refinement
also profile NDT. Final selection profiles again on three fresh 100k blocks,
pooled into a 300k-estimate comparison, and returns a full eight-parameter
vector. Validation evaluates that fixed vector using the ordinary scorer.
The ranking-uncertainty heuristic conditions on the selected NDT; it is not a
calibrated uncertainty bound after nuisance-parameter selection.

This preserves the existing histogram objective, rather than introducing a
continuous RT density. Equal-scoring shifts report the lowest grid
representative; the decimal precision of that value is not estimation precision.

## Correctness and evaluation cost

The H100 check used all 760 trials and compared compiled counts against
materialized samples: **exact equality**. Ordinary and profiled scoring agreed
at five NDT shifts, including shifts between integration steps, at both 5k and
100k estimates. Maximum density difference was `7.15e-7`; maximum total log-score
difference was `4.85e-5`, from floating-point summation order.

Median of three warm calls for ten candidates on an H100 NVL:

| Estimates per candidate | Ordinary score at one NDT | Profile all 2,001 NDT values | Added time |
| --- | ---: | ---: | ---: |
| 5,000 | 0.1796 s | 0.1837 s | 2.26% |
| 100,000 | 2.7640 s | 2.7884 s | 0.88% |

The count buffer is **60.8 MB** for ten candidates and all 760 trials,
independent of the number of estimates. This is a buffer size, not total GPU
memory use. Materializing both outcomes at 100k would require about 6.08 GB.
The ordinary fused scorer remains smaller because it only retains counts near
one observed RT bin.

GPU tests also cover retained state across trials, multiple subjects, both
trial schedules, common and independent candidate random streams, smoothing,
pseudocounts, finite-range boundaries, invalid support, nonfinite outputs, and
candidate-wide truncation rejection. Adaptive tests exercise pooling before
profiling and final selection at a different NDT from the reference winner.
Both empirical and recovery driver smoke tests pass.

## Full-fit comparison

All fits use the same synthetic observations: 760 ordered trials, 720 scored,
10 ms LCA steps, LC 20 ms steps with ten steps per pass, and noise SD 0.1 on
all four LCAs. The histogram has 100 RT bins over 0–3 seconds, Gaussian sigma
0.5 bins, and pseudocount 1 at 100k. Smaller sampling blocks scale pseudocounts
to retain that prior fraction. Adaptive search starts at 5k and can reach 100k.

Likelihoods below were rescored using the ordinary scorer at 100k on the same
five fresh seeds, **9401–9405**, with float64 log summation. Higher is better.
These resample the same observations; they are not held-out data.

| Policy | Start | CMA proposals | Fit time | Total time | Mean fresh log likelihood |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 100k | 0 | 5,000 | 26.88 min | 27.54 min | 254.999 |
| Previous adaptive | 0 | 2,600 | 7.86 min | 8.47 min | 254.907 |
| Previous adaptive | 1 | 2,600 | 7.51 min | 8.08 min | 254.775 |
| Profile NDT, standard budgets | 0 | 1,851 | 7.88 min | 8.49 min | 254.558 |
| No profiling, smaller budgets | 0 | 1,300 | 4.20 min | 4.81 min | 253.953 |
| **Profile NDT, smaller budgets** | **0** | **1,300** | **5.49 min** | **6.10 min** | **254.898** |
| **Profile NDT, smaller budgets** | **1** | **1,300** | **5.20 min** | **5.82 min** | **254.769** |

The smaller profiled fits differ from their previous adaptive counterparts by
**−0.009 ± 0.079** and **−0.007 ± 0.127** log units (paired mean ± Monte Carlo
standard error). They started 65.03 and 60.07 million candidate histories,
versus 97.5 and 93.3 million previously. Sampling-work totals include screening,
reference checks, refinement, and final selection outside the CMA proposal count.

The smaller-budget control used the same start, optimizer/simulation seeds,
and 1,000/300 proposal caps as profiled start 0. Its score was lower on all five
fresh seeds: **253.953 versus 254.898**, a paired difference of **0.945 ± 0.111**.
Profiling cost 27% more time at this fixed proposal budget. Adaptive search
averaged 30,425 estimates per proposal with profiling versus 8,695 without it.
The search trajectories and their sampling allocations differ, explaining why
the runtime difference exceeds the per-evaluation profiling overhead.
This is one controlled starting point, not evidence of a universal advantage.

The standard-budget profiled run transitioned at the plateau rule after 1,251
search proposals and then used all 600 refinement proposals. Its fresh score
was 0.350 ± 0.160 below the previous adaptive start 0. It is not a continuation
of the shorter run: the refinement starts, covariance, and trajectories differ.
This single result does not show that additional refinement generally hurts.

Total time includes setup/compilation, fitting, validation, and predictions;
the separate common rescoring above is excluded. New runs use in-memory Optuna
storage and separate initially empty caches, as do the previous adaptive timing
baselines. The older fixed benchmark includes three validation seeds rather
than five. Both overall speedups against fixed sampling use its start-0 timing.

Other users joined GPU0 during the first standard-budget profiled run, so its
timing is excluded. A repeat on GPU1 reproduced **every proposal and score**
and supplied the 8.49-minute timing above. Process monitoring every ten seconds
observed only our job on GPU1. The shorter profiled runs used GPU1 sequentially;
it was free at launch and during spot checks, without continuous monitoring.
The smaller-budget unprofiled control was also monitored every ten seconds;
only our process appeared on its GPU.

## Parameters

| Parameter | Truth | Previous adaptive 0 | Profiled 0 | Previous adaptive 1 | Profiled 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Response threshold | 0.400 | 0.383 | 0.384 | 0.388 | 0.385 |
| Nondecision time, s | 0.2200 | 0.2197 | 0.2101 | 0.2056 | 0.2001 |
| Automaticity bias | −0.400 | −0.409 | −0.413 | −0.415 | −0.416 |
| Control gain | 12.00 | 12.94 | 14.05 | 14.58 | 13.91 |
| LC mode 0 | 0.650 | 0.367 | 0.374 | 0.428 | 0.268 |
| LC mode 1 | 0.800 | 0.657 | 0.627 | 0.726 | 0.612 |
| LC scaling | 1.500 | 2.218 | 2.973 | 2.170 | 1.921 |
| LC base | 5.500 | 5.058 | 4.794 | 4.958 | 5.027 |

The NDT pairs 0.2197/0.2101 and 0.2056/0.2001 have **identical bin maps over
the full decision-time support**. Their reported differences arise from the
choice of plateau representative. Threshold and bias remain similar; some LC
coordinates move substantially despite almost equal likelihoods. Profiling
does not resolve the existing parameter-recovery weakness.

## Reproduce

From the repository root in the CUDA environment used for ordinary fits:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --fit-strategy adaptive --profile-ndt --estimates 100000 --evaluations 5000 \
  --adaptive-search-evaluations 1000 --adaptive-refine-evaluations 300 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --optimizer-storage memory --validation-estimates 100000 \
  --validation-seeds 9401 9402 9403 9404 9405 \
  --output "$DAWA_RESULTS/ndt-profile-start0"
```

For start 1, use `--start 1 --optimizer-seed 202 --simulation-seed 37` and a
new output directory. Keep data/model seeds unchanged. For the standard-budget
comparison use search 2,000 and refinement 600. `--profile-ndt` alone does not
reduce those budgets. The empirical driver accepts the same options.
To reproduce the smaller-budget control, remove `--profile-ndt` from start 0
and use a new output directory.

The standalone `dawa_ndt_benchmark.py --run PATH --output NEW_PATH` verifies
counts and times ordinary/profiling evaluations from a saved recovery run.
Its optional `--fits label=PATH ... --skip-timing` compares fitted vectors on
common seeds. The JSON report records the exact commands and source hashes.
The remote environment used Optuna 4.9.0, cmaes 0.13.1, and CUDA toolkit 13.0;
raw artifacts are under
`/scratch/gpfs/CSES/dmturner/dawa-benchmarks/ndt-20260926` on della-rse.
