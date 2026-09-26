# Why the adaptive fits lost quality

This report diagnoses the original adaptive policy. The subsequent implementation
and full-fit retest are documented in [adaptive policy 2](adaptive_v2_h100.md).
Descriptions of the "current" transition below refer to the original policy.

The H100 follow-up identified three weaknesses in the fitting policy: restarting
refinement without the learned parameter correlations, an aggressive stopping
rule, and a sampling rule that overlooks rankings within the selected half of
the population. **Retaining the learned covariance improved both tested starts
without increasing the 200-proposal refinement budget.**

These are diagnostic experiments with the original nonlinear, noisy model,
760 trials/720 scored observations, 10 ms LCA timesteps, and the original
histogram likelihood. The ordinary fitter has not been changed by this study.
The reproducible helper is [dawa_adaptive_diagnosis.py](../dawa_adaptive_diagnosis.py);
measurements, parameter vectors, and covariance matrices are in
[quality_diagnosis.json](quality_diagnosis.json).

## Refinement discards useful information

The current stage transition constructs a fresh CMA-ES sampler with normalized
sigma 0.03 and identity covariance. It keeps the incumbent parameter vector,
but discards the relationships between parameters learned during broad search.
Those learned covariance matrices had condition numbers about **805 and 374**
for starts 0 and 1: their search distributions were far from spherical.

I compared three ways to refine exactly the same broad-search results:

| Refinement | Proposals at 100k | Start 0 score | Start 1 score |
| --- | ---: | ---: | ---: |
| Current refinement, covariance reset | 200 | 253.802 | 252.044 |
| Continue that same optimizer | 1,200 | 254.337 | 253.433 |
| Initialize with learned covariance | 200 | **254.651** | **253.925** |
| Fixed-100k complete fit, for context | 5,000 total fit proposals | 255.145 | 254.829 |

Scores are mean log likelihoods over **new seeds 9201–9203 at 100,000 estimates**;
higher is better. They use the same observations, not held-out trials. These
seeds differ from the earlier benchmark's 8101–8103, so the absolute scores also
differ. No diagnostic evaluation score selected an optimizer candidate or
controlled a stopping decision.

For the continuation, all 200 saved ask/tell proposals were replayed exactly
before adding another 1,000 proposals. The covariance experiment changed only
the initial covariance, using the last serialized broad-search covariance in
Optuna's normalized coordinate order. Its center, scalar sigma, optimizer seed,
population size, learning-rate adaptation, parameter grids, reference seed,
and 200-proposal budget matched the original refinement.

Retaining covariance gained **0.849 and 1.882 log units** relative to the original
200-proposal refinements. It outperformed the 1,200-proposal continuations while
using one sixth as many refinement proposals. Those 200-proposal experiments
took 58.4 and 58.2 seconds to optimize on H100. This is a phase-level experiment,
not a new end-to-end fitting speedup benchmark. The diagnostic subclass uses
Optuna internals to initialize covariance; it is not wired into the fit runner.

The parameter geometry explains why this matters. Starting from the original
adaptive start-0 fit, changing only control gain to its fixed-fit value lowered
the fresh-seed score by **39.48**. Changing only LC base gain lowered it by
**36.97**. Moving all eight parameters together to the fixed fit improved it by
**1.34**. Intermediate joint moves also improved the score. Thus individually
reasonable parameter changes can be harmful unless coupled changes accompany
them. These perturbations are not profile likelihoods and do not establish
nonidentifiability, but they directly demonstrate strong parameter coupling.

## The stopping rule mistakes plateaus for convergence

The adaptive rule permits stopping after two 250-proposal windows with reference
improvement below 0.25 log units, once 1,000 proposals have been evaluated.
Applying exactly that rule to the saved **fixed-100k** histories gives:

| Fixed run | Would stop at proposal | Subsequent training-score gain missed |
| --- | ---: | ---: |
| Start 0 | 2,251 | 1.071 |
| Start 1 | 1,001 | 3.202 |

This is evidence against the stopping rule even when every score uses 100k
estimates. CMA-ES can have long plateaus before finding another improvement.
The refinement continuation provides another example: start 0 found no better
reference score through 1,001 refinement proposals, then improved between
1,001 and 1,200. A short plateau alone is insufficient evidence of convergence.

Longer refinement helped, but simply spending another 1,000 proposals added
about 4.6–4.9 minutes and did less than retaining covariance in this experiment.
The next change should improve the transition and convergence checks, rather
than just increase every simulation budget.

## The budget rule protects the wrong part of the ranking

`ranking_uncertainty()` checks possible inversions between the top and bottom
halves. CMA-ES also uses the **ordering within each half**. In the installed
implementation, its largest positive mean-update weights are approximately
0.456 and 0.271, so swapping first and second changes the search direction.

A concrete observed example from start 1, proposals 1302–1311:

| Candidate | 10k search score | 100k reference score |
| --- | ---: | ---: |
| Chosen first, proposal 1306 | 237.130 | 234.648 |
| Chosen second, proposal 1307 | 237.019 | **240.272** |

The top-five membership was completely correct, yet the leading pair was
reversed by **5.624 reference log units**. The race reported uncertainty 0.699,
below its tolerance of 1, and stopped sampling. The current diagnostic does not
protect against this kind of mistake by design.

Two initial Monte Carlo blocks also provide a noisy uncertainty estimate. In
20 repeated low-budget measurements of one late start-1 population, five would
have stopped sampling immediately. One reported zero consequential uncertainty
while accepting a cross-boundary inversion of **3.334 log units** relative to a
three-seed 100k reference. The pair's differences on the individual reference
seeds were 3.145, 3.702, and 3.156. The analogous start-0 population had no such
failure among its 12 accepted repetitions. These are counterexamples to the
heuristic's reliability, not estimates of a general failure rate.

The repeated 5k measurements averaged 2.18 and 2.69 log units below their 100k
references for the two populations. Pooling densities and keeping the prior
fraction fixed does not remove finite-sample bias after taking logs. This is
another reason to avoid comparing raw scores from different budgets as though
they were interchangeable.

## What the audit ruled out

I rescored **all 1,000 proposals** in the final two broad-search windows of both
starts at the original reference budget and seed. In all four windows, the
three-candidate shortlist included the best candidate on that reference
objective. It did not miss an improvement that would have changed those
stopping decisions. Mixed-budget shortlisting remains a methodological concern,
but it was not the explanation for these four flat checkpoints.

Independent model/plan reconstruction reproduced the saved reference scores.
The continuations replayed every saved refinement proposal exactly, and the
CPU reconstruction reproduced all broad-search proposals before extracting
covariance. The earlier truncation optimization also preserved every proposal
and score. There is no evidence here of a likelihood pooling or compiler
correctness error causing the quality gap.

CMA's mean learning rate decreased in both methods: at 2,501 proposals it was
0.045 for adaptive start 0 and 0.047 for fixed start 0. A learning-rate decrease
alone is not evidence of an adaptive-specific failure.

## Recommended changes

1. Retain the learned parameter correlations when entering high-precision
   refinement. This is the most promising tested repair and needs no additional
   refinement simulations in these runs.
2. Use a low-budget plateau to trigger a precision/stage transition; require
   stronger evidence before ending the fit. Validate the stopping rule against
   longer runs instead of interpreting two flat checks as convergence.
3. Make sample allocation account for consequential changes in CMA's weighted
   ranking, including the leading candidates. Improve the uncertainty estimate
   beyond two blocks before treating it as a reliable budget decision.
4. Rescore a small final shortlist with additional independent selection seeds,
   reserving separate seeds for evaluation. Longer refinement sometimes improves
   its fixed-seed training score while worsening fresh-seed scores, so final
   selection also has Monte Carlo uncertainty.

The covariance change has not closed the whole gap: it remains about 0.49 and
0.90 log units below the matched fixed starts on these fresh seeds. LC mode
estimates also remain different. Better optimization is not the same as
successful parameter recovery. This study covers two starts of one synthetic
subject and does not yet establish a reliable 10× speedup at equal fit quality.

## Reproduce the diagnostics

Use saved recovery directories containing `manifest.json`, `recovery.json`,
`evaluations.jsonl`, and `synthetic_subject.csv`. Run from the repository root:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_adaptive_diagnosis.py \
  --mode audit --run /path/to/adaptive-v2-start0 \
  --output /path/to/new-audit

python Scripts/Debug/pec_batch_compile/dawa/dawa_adaptive_diagnosis.py \
  --mode refine --run /path/to/adaptive-v2-start0 --refine-total 1200 \
  --output /path/to/new-refinement-continuation

python Scripts/Debug/pec_batch_compile/dawa/dawa_adaptive_diagnosis.py \
  --mode covariance --run /path/to/adaptive-v2-start0 --refine-total 200 \
  --covariance-source Scripts/Debug/pec_batch_compile/dawa/fitting_acceleration/quality_diagnosis.json \
  --output /path/to/new-covariance-test
```

The embedded covariance data apply to the saved benchmark starts with those
basenames; other fits require their own covariance reconstruction. The helper's
`perturb` mode compares joint and individual coordinate moves, with
`--comparator /path/to/fixed-run`. All modes default to evaluation seeds
9201–9203. Raw diagnostics and the exact scripts used on H100 remain under
`/scratch/gpfs/CSES/dmturner/dawa-benchmarks/adaptive-20260925/diagnosis*`.
