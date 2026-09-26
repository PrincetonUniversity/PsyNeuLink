# Adaptive fitting with the compiled DAWA sampler

This is the historical **policy 1** benchmark. The adaptive runner now uses
[policy 2](adaptive_v2_h100.md), which addresses the diagnosed quality issues.
Commands and defaults below describe the original experiment.

The empirical and recovery runners implement this workflow with
`--fit-strategy adaptive`. The model, timestep, noise, ordered trial history,
histogram bins, and smoothing are unchanged. This is an opt-in fitting strategy,
not a particle filter or density approximation. Cached nondecision-time sweeps
remain an experimental helper; this workflow fits all eight original coordinates.

## H100 results, 2026-09-25

**The matched comparison achieved 5.1× overall speedup, with a modest loss in
likelihood and appreciable differences in some fitted parameters. It does not
establish a reliable 10× speedup at equivalent fit quality.**

The [follow-up quality diagnosis](quality_diagnosis.md) tests the causes of the
gap. Retaining CMA-ES's learned covariance improved both starts at the same
refinement budget; stopping and ranking checks also need revision. These are
diagnostic experiments; the subsequent policy-2 implementation is linked above.

Both runs used the same synthetic full subject, start, optimizer seed, model,
estimator, and 100k final validation budget. Both used in-memory optimizer
storage. They ran on separate H100 NVL GPUs on della-rse with isolated, initially
empty compilation caches. Full figures and source hashes are in
[h100_comparison.json](h100_comparison.json).

| Run | Proposals | Fit time | Total time | Mean independent log likelihood |
| --- | ---: | ---: | ---: | ---: |
| Fixed 100k, start 0 | 5,000 | 26.88 min | 27.54 min | 255.060 |
| Adaptive, start 0 | 2,701 | 4.79 min | 5.40 min | 254.173 |
| Adaptive, start 1, warm cache | 1,701 | 2.56 min | 2.81 min | 252.085 |

The first adaptive run is **5.61× faster during fitting and 5.10× faster overall**.
The second start changed the initial point, optimizer seed (202), and simulation
seed (37), and reused the compiled kernels. Its roughly 9.8× total-time ratio
against the fixed start-0 run therefore is not a matched comparison; it also
loses 2.97 log units. For context, the earlier fixed-100k run from start 1 scored
254.884 on the same validation seeds, 2.80 above this adaptive start-1 result.
That historical run is not used as the timing baseline here.

Independent validation uses three new random seeds, each with 100,000 estimates,
on the same observed dataset. It measures simulation-score stability, not
held-out-trial predictive accuracy. Start 0 loses 0.887 log units on average;
the three paired losses are 1.085, 0.927, and 0.648. Thus its small loss is
consistent across these seeds. All methods use the same nonlinear noisy model.
Reduced-budget search, shortlisting, and early stopping can miss candidates
that a longer, more precise search finds.

| Parameter | Generating value | Fixed, start 0 | Adaptive, start 0 | Adaptive, start 1 |
| --- | ---: | ---: | ---: | ---: |
| Response threshold | 0.400 | 0.382 | 0.386 | 0.363 |
| Nondecision time, seconds | 0.2200 | 0.2116 | 0.1999 | 0.1956 |
| S/D/R bias | −0.400 | −0.410 | −0.423 | −0.417 |
| Control gain | 12.00 | 12.52 | 14.49 | 9.19 |
| LC mode, previous condition 0 | 0.650 | 0.379 | 0.279 | 0.336 |
| LC mode, previous condition 1 | 0.800 | 0.693 | 0.421 | 0.460 |
| LC scaling | 1.500 | 2.489 | 2.990 | 2.783 |
| LC base gain | 5.500 | 5.099 | 4.765 | 5.634 |

Threshold, nondecision time, and bias are relatively close; control gain and LC
coordinates vary substantially. For start 0, the largest difference from fixed
is 34% of the allowed range for LC mode 1. Similar likelihoods do not imply
successful parameter recovery. Use multiple starts and high-budget checks;
keep the fixed strategy when comparing methods until the quality tradeoff has
been assessed for the study's data.

Start 0 used 2,501 broad-search proposals and 200 refinement proposals, plus
reference checks. Its average search budget was 16,475 estimates; 69% of search
proposals stayed at 5,000. Including reference checks and refinement, it started
64.205 million simulated subject histories, compared with 500 million nominal
histories for fixed search. Sampling work decreases more than wall time because
small blocks incur launch and scoring overhead, and do not use the H100 as
fully as large batches. This is a plausible remaining performance limitation,
not a separately measured breakdown of GPU occupancy.

The first adaptive implementation took 7.00 minutes to fit (7.50 total).
Rejecting truncated candidates within the fused kernel reduced fitting time to
4.79 minutes, saving 132 seconds. **All 2,701 proposal vectors, scores, budgets,
and block seeds remained exactly equal**, as did final validation scores.
Sampling time for the first 251 proposals fell from 149.7 to 27.5 seconds.
This improvement combines avoiding serial retries and stopping failed histories
early. Valid histories still simulate the complete sequence. The fixed comparator
retains its existing serial retry handling; that optimization could also benefit
fixed-budget fitting.

The fixed run reproduced all 5,000 proposal vectors and scores from the earlier
journal-backed run exactly. Removing journal storage is therefore excluded from
the headline speedup. Raw outputs, logs, and immutable source snapshots are
retained under `/scratch/gpfs/CSES/dmturner/dawa-benchmarks/adaptive-20260925`.

## How it works

1. Start each CMA-ES population with two independent blocks of 2,500 complete
   simulated subject trajectories per candidate. Candidates share random draws
   within a block; new blocks and generations use distinct seeds.
2. Pool the per-trial densities in proportion to block sizes **before taking
   logs**. Scale each block's pseudocount with its size, preserving the prior
   fraction of pseudocount 1 at 100,000 estimates.
3. Estimate uncertainty in paired candidate score differences from the
   independent blocks. Retain covariance across trials and between candidates.
   If ranking across the top-half/bottom-half selection boundary is uncertain
   by more than one log unit, add a block to double the sample budget, up to
   100,000. Earlier samples are retained. The two-standard-error rule is a
   budget heuristic, not a calibrated confidence guarantee.
4. Every 250 search proposals, compare up to three shortlisted candidates with
   the incumbent on the same 100,000-estimate, fixed-seed reference objective.
   Mixed-budget scores do not select the reported final fit.
5. After at least 1,000 proposals, stop broad search if two consecutive checks
   improve the reference incumbent by less than 0.25 log units. The overall
   proposal cap can also stop search. Start a local CMA-ES refinement from the
   incumbent, with normalized initial sigma 0.03 and up to 200 proposals at
   100,000 estimates. Report the best reference-scored candidate.
6. Validate with fresh seeds 8101–8103, then generate predictive trajectories.
   Validation outcomes never influence racing, stopping, or final selection.

The compiler's new `BatchedSimulationPlan.histogram_likelihood()` method returns
densities shaped `[candidate, subject, trial]`. It supports both the fused
reduction and materialized reference. Thus independent simulation blocks can
be pooled without storing trajectories or using a logging hook. Masked trials
still execute and carry state. The adaptive runner uses
`invalid_candidates="nan"` to reject an entire candidate if any of its histories
truncates, including on masked trials. Failed histories stop immediately under
independent trial scheduling; valid candidates retain the same dynamics and
scores. This avoids rerunning a whole population one candidate at a time to
identify failures. Other numerical errors still abort the fit. Ordinary
`log_likelihood()` truncation behavior is unchanged.

The default maximum is set by `--estimates`; the default starting budget is
`--adaptive-min-estimates 5000`. Further controls appear in `--help`.
`--evaluations` caps the combined broad-search and refinement proposals;
incumbent reference checks and independent final validation are additional work
and included in the measured runtimes.

## Reproduce the comparison

Run from the repository root in the CUDA environment described in the
[fitting guide](../README.md). Use complete subject sequences, new output
directories, and the same seeds for each pair:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --fit-strategy fixed --estimates 100000 --evaluations 5000 \
  --optimizer-storage memory --validation-estimates 100000 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --output /path/to/results/fixed-start0

python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --fit-strategy adaptive --estimates 100000 --evaluations 5000 \
  --optimizer-storage memory --validation-estimates 100000 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --output /path/to/results/adaptive-start0
```

The fixed comparator also uses in-memory storage, so the speed comparison does
not credit adaptive fitting with removing journal overhead. Both commands
generate the same synthetic subject (data seed 20260925, model seed 29), use
760 trials/720 scored observations, 10 ms LCA timesteps, and noise SD 0.1 in all
four LCAs. Fit time includes sampling, scoring, selection, refinement, optimizer,
and logging work. Total driver time also includes setup/compilation, independent
validation, and predictive simulation.

Compare normalized parameter differences as well as fresh-seed log likelihoods.
The earlier 100k pilots already showed variability in LC modes and scaling
between starts; similar likelihoods do not establish equivalent parameters or
successful recovery. One synthetic subject cannot establish performance across
a study population.


## What is implemented

The empirical fitter and recovery runner share the complete adaptive workflow.
The general compiler provides per-trial densities for sample accumulation and
optional rejection of truncated candidates within a batch. The search still
uses the original eight parameters and the full nonlinear noisy model.
Nondecision-time caching remains a separate experimental helper, and particle
filtering has not been implemented in this workflow.

Regression checks cover density pooling against concatenated simulation
histories, unequal block sizes, masks and multiple subjects, invalid candidates,
shared and separate random streams, retention of earlier samples, separation of
selection from independent validation, and an end-to-end GPU recovery smoke
fit. The local regression run passed 64 tests (34 skipped); the expanded
truncation/random-stream check subsequently passed all four GPU cases.
