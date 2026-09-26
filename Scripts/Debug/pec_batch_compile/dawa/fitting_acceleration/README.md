# Local fitting acceleration study

These are experiments with the existing nonlinear, noisy PEC sampler. The
[ordinary fitting guide](../README.md) remains the handoff entry point. No
adaptive-budget policy or particle filter is enabled by default.
The optional [adaptive fitting workflow](adaptive_h100.md) now implements
sample accumulation, uncertainty-based budgets, and high-budget refinement.
The [fit-quality diagnosis](quality_diagnosis.md) investigates its remaining
quality gap and tests retaining learned covariance during refinement.
The [NDT profiling implementation and H100 comparison](ndt_h100.md) extend
that workflow with compiled decision-time counts and seven-parameter search.
The [subsequent H100 profile](ndt_profile_h100.md) identifies small sampling
blocks and repeated compiler preparation as the next optimization targets.
The [sampling-block implementation](sampling_blocks_h100.md) addresses those
targets and checks exact replay of the complete fit.
The [updated H100 profile](sampling_blocks_profile_h100.md) finds that GPU
simulation now takes 91% of fitting time, split almost equally between adaptive
search and 100k refinement.
The local prototype results below predate those implementations.

The study uses the complete synthetic subject from the first H100 recovery
run: 760 trials, 720 scored, all four LCA noise SDs 0.1, LCA steps 10 ms, and
the original retained control/controller history. Candidate populations come
from proposals 2–11, 1002–1011, and 4991–5000 of that run. These populations
represent early, middle, and late search; they are not a random sample of all
possible fits or subjects.

## Measured on the local RTX 2080 Ti, 2026-09-25

Full scores and timings are in [local_2080ti.json](local_2080ti.json), with the
selected proposals in [candidates.json](candidates.json). The synthetic choices,
RTs, design, and scoring mask regenerated locally match the original H100
recovery subject exactly.

| Population | Estimates | Seconds/candidate | Speedup vs 100k | Pairwise ranking disagreements | Largest winner loss, log units |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early | 2,000 | 0.0178 | 22.3× | 0% | 0 |
| Early | 5,000 | 0.0277 | 14.4× | 0% | 0 |
| Middle | 2,000 | 0.0264 | 31.6× | 1.5% | 0 |
| Middle | 5,000 | 0.0503 | 16.6× | 0.7% | 0 |
| Late | 2,000 | 0.0265 | 30.6× | 38.5% | 0.466 |
| Late | 5,000 | 0.0483 | 16.8× | 36.3% | 0.466 |
| Late | 25,000 | 0.2128 | 3.8× | 32.6% | 0.245 |
| Late | 100,000 | 0.7990 | 1.0× | 28.1% | 0.245 |

These rows use a constant prior fraction. The 100k row uses independent test
seeds against the three reference seeds; its nonzero disagreement shows that
the reference ordering of very similar late candidates is itself uncertain.
Only 17 of the 45 late-population pairs pass the reference separation diagnostic.
On those pairs, disagreement is 19.6% at 5k and 9.8% at 100k.

Small budgets preserve useful rankings much better than absolute scores. In
the early population, 5k scores average 83.6 log units below the 100k reference
despite preserving every ranking. The corresponding middle/late differences
are approximately −3.1/−2.0 log units. The fixed-pseudocount comparison shifts
the objective substantially more; it is not an interchangeable low-budget
approximation to the default 100k regularization.

For nondecision time, **all 2,001 grid values from 0.1 to 0.3 s took 0.102 s
to score** after a 0.209 s simulation/compression at 10,000 estimates. A single
ordinary fused evaluation took approximately 0.184 s at the same dynamic
parameters and estimate budget. The compressed cache was 2.41 MB versus
60.8 MB of materialized samples. It contained 198 distinct decision times;
the 2,001 shifts produced only 21 distinct bin maps/scores.
The NDT equivalence check uses pseudocount 1 at its 10k budget; it tests reuse
of the existing score, independently of the prior-scaling audit above.

The prototype's map deduplication took 0.105 s including discovery, so it did
not improve on the direct grid scan here. The gain comes from reusing dynamics.
This is a very large saving for an entire NDT sweep, but an optimizer normally
does not run 2,001 such sweeps per candidate; it is **not** a fitting speedup
of that magnitude. Pathwise equality passed for all 7.6 million trial outcomes
at each of five shifts. Maximum score discrepancy was below 0.000004 log units.

## Complete smaller-budget fit

[recovery_5k.json](recovery_5k.json) records a complete 5,000-proposal recovery
fit with 5,000 estimates, pseudocount 0.05, and the same initial point and
optimizer seed as the first original recovery run. It took **562.3 s (9.37 min)**
to fit on the local 2080 Ti, or 609.5 s including setup, validation, and
predictions. Nineteen truncating proposals were penalized; final validation
completed without truncation.

All entries below were rescored locally at 100,000 estimates, pseudocount 1,
and independent seeds 8101–8103. The original H100 fits reproduce their earlier
validation scores in this local comparison.

| Parameter vector | Mean independent log likelihood |
| --- | ---: |
| Generating parameters | 252.177 |
| Original 100k-estimate fit, start 0 | 255.060 |
| Original 100k-estimate fit, start 1 | 254.884 |
| Best 5k proposal through evaluation 500 | 250.019 |
| Best 5k proposal through evaluation 1,000 | 251.019 |
| Best 5k proposal through evaluation 2,000 | 252.867 |
| Best 5k proposal through evaluation 5,000 | 252.877 |

Uniformly reducing the budget costs approximately **2.18 log units** relative
to the first original fit in this pilot. Moreover, proposals 2,001–5,000 took
another 296 s and improved this independent score by only 0.010, while the
training score increased by 1.86. This is evidence for switching precision or
checking convergence sooner; it does not establish an optimal switch point.
The final generating-parameter recovery is still weak in the LC coordinates.

A post-fit NDT sweep at 100k estimates selected 0.2501 s instead of 0.2548 s.
Both lie on the same histogram plateau and gave **identical validation
scores**. A sweep at the end alone did not recover the remaining likelihood
gap. The potential benefit is removing redundant NDT searches while fitting
the other coordinates.

For the final parameter vector, a matched batch-of-ten benchmark measured
0.04665 s/candidate at 5k and 0.77102 s/candidate at 100k: **16.5×** faster
objective evaluations. The complete fit spent 303.9 s inside objective calls
and 258.4 s on everything else, predominantly journal persistence. An in-memory
replay reproduced every one of 4,071 checked proposals exactly, taking 8.72 s
for optimizer work versus 210.59 s outside objective calls in that part of the
actual fit. The replay reused measured scores; it was not another GPU fit.

The earlier complete 100k fits ran on H100s, so their 28-minute runtimes and
this local 9.37-minute runtime are **not a matched end-to-end speedup**.
Candidate timing, reduced optimizer overhead, and earlier precision changes
are promising ingredients for a 10× improvement. Comparable fit quality and
an end-to-end 10× gain still need a matched adaptive-fit experiment.

These results motivated the [adaptive implementation](adaptive_h100.md):
in-memory optimizer storage, small estimates budgets
for broad search, and independent accumulated simulation blocks to refine
close comparisons and verify the incumbent. It should compare final fits at
100k or more across several starts. Cached NDT profiling is now implemented
and [tested inside that search](ndt_h100.md). Bootstrap particle filtering remains a
lower-priority experiment because it changes the objective and does not by
itself address the low-budget ranking problem observed here.

## Reproduce the candidate-budget audit

From the repository root, using the CUDA environment described in the fitting
guide:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_fitting_budget_study.py \
  --data Scripts/Debug/pec_batch_compile/dawa/dawa_lca_model/flanker_data_part1.csv \
  --generate-seed 20260925 \
  --candidates Scripts/Debug/pec_batch_compile/dawa/fitting_acceleration/candidates.json \
  --output /tmp/dawa-budget-study
```

`--generate-seed` replaces recorded responses with one trajectory at the
original generating parameters. Omit it to audit an existing observation CSV.
`--history PATH/evaluations.jsonl` can replace `--candidates` to extract the
three populations from a complete 5,000-proposal fit. Outputs must be new
directories. A 16-trial smoke check is available with `--trials 16`.

For each population, three independent 100,000-estimate reference scores are
compared against three other seeds at 2k, 5k, 10k, 25k, and 100k estimates.
Candidates share random draws within each seed. Trials are never treated as
independent Monte Carlo replicates. Timings exclude warmup and report the
median time per candidate in a batch of ten, including scoring and transfers.

The report contains full scores, ranking disagreements, top-five overlap,
the reference-score loss from selecting each low-budget winner, and runtime.
Reference rankings also contain Monte Carlo error. The additional “resolved
pairs” metric only includes pairs whose reference mean difference exceeds
twice its estimated paired standard error. With three reference seeds this
is a diagnostic, not a calibrated confidence guarantee. The reported CRN
variance ratio estimates independent-seed difference variance divided by
paired-seed difference variance; it is noisy with only three repeats.

## Keep regularization comparable

The production estimator has 200 joint cells (two choices, 100 RT bins),
Gaussian smoothing SD 0.5 bins, and pseudocount 1 at 100,000 estimates.
Keeping pseudocount 1 at 5,000 estimates raises the prior fraction from about
0.20% to 3.85%. The study compares that setting with
`pseudocount = estimates / 100000`, which holds the prior fraction constant.
This does **not** remove finite-sample log-likelihood bias.

Independent simulation blocks can be accumulated without discarding earlier
work: add their weighted counts, or average their densities with weights
proportional to block sizes when pseudocount/estimates is constant. Take logs
after pooling. Averaging the blocks' log likelihoods is a different estimator.
The study reports pooled scores and tests pooling against concatenating the
actual samples. This audit script does not implement automatic promotion;
the subsequent adaptive fitter does.

## Nondecision time

Nondecision time is an additive RT readout with no influence on model dynamics.
The prototype samples at zero nondecision time, compresses the actual FP32
decision times into per-trial/per-choice counts, then applies each proposed
shift and the existing histogram/smoothing rule to those counts. It retains
exact decision times; shifting a coarse 30 ms histogram is not equivalent.

The full-subject check compares choices and RTs path by path at five shifts,
including shifts that are not multiples of the 10 ms timestep. It also checks
cached scores against the production fused scorer. Bin-membership maps are
deduplicated when several shifts produce exactly the same histogram score.

This prototype materializes samples before compressing them. The subsequent
[compiler implementation](ndt_h100.md) accumulates decision-time counts directly
and profiles NDT inside optimization. Its report compares fit quality against
the eight-coordinate optimizer.

## Smaller-budget fit with independent validation

The shared fitting driver accepts explicit fitting pseudocount and validation
budget controls. Existing defaults remain 100,000 estimates and pseudocount 1.
For a 5,000-estimate recovery fit with the same prior weight:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --estimates 5000 --pseudocount .05 --evaluations 5000 \
  --validation-estimates 100000 --output /tmp/dawa-recovery-5k
```

Rescoring scales the pseudocount with its budget, so this command validates
with pseudocount 1. Training scores from different budgets must not be compared
as though they were exact values of the same objective. Compare fits on the
same independent validation seeds and budget instead.

The optional `--optimizer-storage memory` avoids Optuna's per-update journal
flushes. It still writes `evaluations.jsonl`, progress, fit results, and the
final `optimizer_trials.csv`, but does not persist Optuna's internal state in
`optimizer.journal`. Journal storage remains the default. This tradeoff becomes
relevant at smaller simulation budgets; on the local WSL filesystem a CPU-only
201-proposal audit took 11.64 s with journal storage versus 1.28 s in memory.
The profile attributed 9.79 s to 2,780 `fsync` calls. These timings include
profiling overhead and a synthetic objective, and do not measure full-fit
speedup. Filesystem performance on Della can differ substantially.

Particle filtering and conditional integration of threshold crossings remain
separate follow-ups. A filter would condition retained state on observed
history, changing the present trial-marginal objective. Neither a filter
speedup nor an end-to-end 10× fit speedup follows from candidate timing alone.
