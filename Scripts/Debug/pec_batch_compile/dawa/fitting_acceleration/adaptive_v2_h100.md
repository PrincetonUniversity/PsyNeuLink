# Adaptive policy 2: quality repair and H100 retest

The revised policy substantially improved both tested fits. On five fresh
simulation seeds, its log likelihood differed from the corresponding fixed-100k
fits by **−0.038 ± 0.056** and **+0.139 ± 0.141** (mean ± paired Monte Carlo
standard error). The earlier adaptive policy trailed by 0.859 and 2.773 units.

The repair costs time: **8.47 and 8.08 minutes overall**, compared with the
27.54-minute fixed benchmark. That is **3.25–3.41× overall speedup**, not 10×.
The earlier adaptive fits were faster but had worse likelihoods. These two
starts on one synthetic subject support the quality repair; they do not establish
equivalent fitting performance or parameter recovery across subjects.

## What changed

The changes are implemented in [dawa_adaptive_fit.py](../dawa_adaptive_fit.py)
and used by both the empirical and recovery runners with `--fit-strategy adaptive`.
The model, noise, integration, trial dependencies, and histogram estimator are unchanged.

1. **Preserve learned correlations.** Refinement reuses the broad search's
   covariance in the same normalized, alphabetically ordered parameter space.
   It restarts at the checked incumbent with sigma 0.03 and reset adaptation
   state. The learned covariance condition numbers in these runs were 618 and
   713. Optuna has no public covariance accessor, so the small private interface
   is isolated and tested; its serialized state can lag one population.
2. **Check consequential ordering throughout the population.** Signed logarithmic
   rank weights approximate the importance of an inversion to CMA updates,
   including inversions within the selected half. Four independent blocks now
   start each race; promotions add two more blocks while retaining earlier samples.
   A 2.5-SE margin and tolerance 1 allocate sampling effort. This remains a
   heuristic, not a calibrated confidence bound or the exact CMA weight formula.
3. **Use a plateau to change precision, not declare convergence.** Broad search
   has a 2,000-proposal cap. Two sufficiently flat checks after 1,000 proposals
   can transition earlier, but refinement then gets its full 600-proposal budget.
   Both H100 runs reached the search cap; neither stopped on the plateau rule.
4. **Separate screening, selection, and validation.** Nominees from different
   generations are screened at one common budget before reference checks.
   Eight finalists are compared on three fresh 100k blocks, pooling densities
   before taking logs (300k estimates per finalist in total). Validation uses
   another, disjoint set of seeds. The reported fit may therefore differ from
   the candidate with the best original training-seed score.

All blocks simulate complete ordered subject histories. Smaller blocks scale
pseudocounts to keep the prior fraction constant. Extra screening, reference,
and selection evaluations are included in fit time and sampling-work counters,
but are outside the CMA proposal count.

## Full-fit results

Same synthetic subject: 760 trials, 720 scored observations, 10 ms LCA timestep,
noise 0.1 on all four LCAs, 100-bin RT histogram over 0–3 seconds, smoothing sigma
0.5 bins, pseudocount 1 at 100k. Both new runs used an H100 NVL on della-rse,
in-memory optimizer storage, and separate initially empty compilation caches.

Every likelihood in this table was rescored on **seeds 9301–9305, with 100,000
estimates per seed**, after fitting. These are fresh Monte Carlo draws on the
same observations, not held-out data. Higher is better.

| Policy | Start | Proposals | Fit time | Total time | Mean fresh log likelihood |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 100k | 0 | 5,000 | 26.88 min | 27.54 min | 254.964 |
| Fixed 100k, historical journal run | 1 | 5,000 | 28.03 min | 28.55 min | 254.648 |
| Adaptive policy 1 | 0 | 2,701 | 4.79 min | 5.40 min | 254.106 |
| Adaptive policy 1, warm cache | 1 | 1,701 | 2.56 min | 2.81 min | 251.876 |
| **Adaptive policy 2** | **0** | **2,600** | **7.86 min** | **8.47 min** | **254.926** |
| **Adaptive policy 2** | **1** | **2,600** | **7.51 min** | **8.08 min** | **254.788** |

The 3.25–3.41× overall speedups use the prior cold-cache, in-memory **fixed start-0
runtime** as the reference for both new starts; the start-0 comparison is matched
on starting point and optimizer seed. Fit-only speedups against that reference
are 3.42–3.58×. The older fixed start-1 journal run supplies a second quality
comparison, not a matched timing baseline. Old total times include three validation
seeds; new totals include five. Additional common rescoring is outside these times.

The independent selection step chose the sixth- and fourth-ranked reference-seed
candidates. Their fresh mean scores were 0.140 and 0.090 above the respective
reference winners, with paired Monte Carlo SEs 0.099 and 0.122. That is a favorable
observation here, not strong standalone evidence that this selection rule always helps.

Search averaged 15,348 and 13,248 estimates per proposal. Full fitting started
97.5 million and 93.3 million candidate histories, including screening, checks,
refinement, and selection. Refinement alone took about 176–178 seconds of sampling
time. The extra precision work and additional block launches explain why the
quality repair reduced the earlier speed advantage.

## Parameter estimates

| Parameter | Generating truth | Fixed 0 | Revised 0 | Fixed 1 | Revised 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Response threshold | 0.400 | 0.382 | 0.383 | 0.383 | 0.388 |
| Nondecision time | 0.2200 | 0.2116 | 0.2197 | 0.1960 | 0.2056 |
| Bias | −0.400 | −0.410 | −0.409 | −0.423 | −0.415 |
| Control gain | 12.00 | 12.52 | 12.94 | 14.70 | 14.58 |
| LC mode 0 | 0.650 | 0.379 | 0.367 | 0.309 | 0.428 |
| LC mode 1 | 0.800 | 0.693 | 0.657 | 0.533 | 0.726 |
| LC scaling | 1.500 | 2.489 | 2.218 | 2.421 | 2.170 |
| LC base | 5.500 | 5.099 | 5.058 | 4.864 | 4.958 |

Threshold and bias estimates are close, and the control-gain discrepancy is much
smaller than before. LC estimates still vary between starts and differ from the
generating values. Better optimization does not resolve the parameter-recovery
issues diagnosed earlier. This experiment bundles the policy changes; the earlier
[covariance ablation](quality_diagnosis.md) isolates that particular change.

## Reproduce and inspect

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --fit-strategy adaptive --estimates 100000 --evaluations 5000 \
  --adaptive-search-evaluations 2000 --adaptive-refine-evaluations 600 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --optimizer-storage memory --validation-seeds 9301 9302 9303 9304 9305 \
  --output "$DAWA_RESULTS/adaptive-policy2-start0"
```

For start 1, use `--start 1 --optimizer-seed 202 --simulation-seed 37` and a new
output directory. The ordinary fixed-budget strategy remains the default.
`--evaluations` caps combined proposal counts; it need not be exhausted.

[The compact result archive](adaptive_v2_h100.json) contains parameters, all five
scores, paired differences, stage diagnostics, source hashes, the launch script,
and the common-rescoring script. Raw artifacts remain under
`/scratch/gpfs/CSES/dmturner/dawa-benchmarks/adaptive-20260925/`, with new runs in
`adaptive-v3-start0` and `adaptive-v3-start1`. The directory's third snapshot is
policy version 2; snapshot v2 had only added truncation rejection to policy 1.
The tested optimization dependencies were Optuna 4.9.0 and cmaes 0.13.1.

The common comparison uses float64 log reduction. Separately applying PEC's
float32 reduction reproduces the runners' validation scores within 1e-5; the two
reduction precisions differ by about 0.0003 log units at most here. Older CSVs have
a different column order, so the comparison verifies equality of every ordered
stimulus, condition, mask, and observation rather than comparing CSV byte hashes.

Regression checks cover retained sample blocks and disjoint seeds, inversions
within the elite, covariance mapping and sampling correlations, fresh selection
overriding a biased reference winner, and the empirical/recovery GPU runners.
The adaptive unit, driver, and budget-study suites passed (30 checks in aggregate,
including lint checks; one cached style check skipped). No compiler kernel changes
were needed for this policy revision.
