# Legacy marginal adaptive fitting

These commands reproduce the earlier **trial-marginal objective**. They do not
condition latent state on observed responses and are not the current DAWA
fitting workflow. Use the [main fitting guide](../README.md) for conditioned
fits. The timing gains below do not transfer directly to particle filtering.

The [fit and recovery runners](../README.md) support `--likelihood marginal --fit-strategy adaptive`. This starts with small simulation
budgets, adds independent samples when candidate rankings are uncertain, and
checks promising candidates at the maximum budget. A final refinement retains
the parameter correlations learned during the search:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit.py \
  --data "$DAWA_DATA" --subject 1 --likelihood marginal --fit-strategy adaptive \
  --estimates 100000 --evaluations 5000 \
  --output "$DAWA_RESULTS/subject1-adaptive"
```

For recovery, use the same options with `dawa_pec_recovery.py`. Conditioned
fitting currently rejects adaptive pooling and NDT profiling: independent
trial-density blocks cannot be pooled as before, and changing NDT changes
the observation weights and therefore subsequent state history. These
shortcuts need separate sequential algorithms before they can be enabled.
In marginal mode,
`--estimates` is the maximum/reference budget and `--evaluations` is a cap on
search plus refinement proposals. The current experimental policy defaults to:

- Up to 2,000 search proposals, starting with four independent blocks totaling
  5,000 estimates. Uncertainty in ordering within the leading candidates also
  triggers extra samples. This is a heuristic, not a confidence guarantee.
- 600 proposals at the maximum budget, retaining the learned covariance.
  A coarse-search plateau triggers this stage; it does not imply convergence.
- A comparison of eight finalists on three fresh maximum-budget blocks,
  pooling densities before taking logs. Separate validation seeds then assess
  the selected fit.

Use `--adaptive-search-evaluations` and `--adaptive-refine-evaluations` to change
the stage budgets. Screening, reference checks, and final selection add scoring
work outside the proposal count; their time is included in reported fit time.
Smaller simulation blocks scale pseudocounts to keep the prior weight constant.
`best_training_log_likelihood` is the selected fit's original reference-seed
score; `adaptive.best_reference_score` can be higher. Final selection details
are saved under `adaptive.final_selection`.

Adaptive fits default to in-memory optimizer storage and save evaluation logs,
trial tables, and results. Add `--optimizer-storage journal` to persist the
broad-search optimizer internals too. `--fit-strategy fixed` remains the default.
See [the original adaptive experiment](adaptive_h100.md)
and [its quality diagnosis](quality_diagnosis.md) for the
motivation behind this revision. Use multiple starts when comparing parameter estimates.

The [revised H100 test](adaptive_v2_h100.md) took **8.1–8.5
minutes**, versus a **27.5-minute** fixed-budget benchmark (about **3.3× faster**).
Both revised fits had fresh-seed likelihoods close to their corresponding fixed
fits. The earlier policy was faster but fit less well. This is still experimental:
two starts on one synthetic subject do not establish recovery across subjects,
and some LC parameter estimates still differ noticeably.

## Profile nondecision time

Add `--profile-ndt` to an adaptive fit or recovery run to optimize nondecision
time inside each proposal. CMA-ES then searches seven dynamic parameters.
The compiler accumulates exact decision-time counts during the same complete
trial histories; the fitter evaluates the 0.1–0.3 s NDT grid from those counts.
It retains the existing histogram, smoothing, and pseudocount rules.

This is experimental and requires `--likelihood marginal --fit-strategy adaptive`. Equal
histogram scores can cover an interval of NDT values; the reported value is the
lowest grid representative, not evidence of 0.1 ms estimation precision.
Reference checks, refinement, and final selection also optimize NDT, while
independent validation scores the chosen full eight-parameter vector.

In the [H100 experiment](ndt_h100.md), profiling NDT with
smaller search/refinement budgets took **5.8–6.1 minutes overall**, about
**1.39× faster** than the previous adaptive fits, with similar fresh-seed
likelihoods in two starts. A smaller-budget control without profiling fit less
well. LC parameter estimates still varied. To use the
tested budgets, add these options to a fit or recovery command:

```bash
  --likelihood marginal --fit-strategy adaptive --profile-ndt --estimates 100000 \
  --adaptive-search-evaluations 1000 --adaptive-refine-evaluations 300 \
  --optimizer-storage memory
```

`--profile-ndt` alone keeps the existing search/refinement budgets. The measured
gain includes reducing those budgets; it is not a 1.39× faster simulator.

NDT-profiled adaptive fits now run their independent sampling blocks together
by default. This keeps the same samples and fitting decisions while reducing
GPU launches and repeated preparation. Add `--no-batch-sampling-blocks` for
separate-block execution, which uses less count-buffer memory. See the
[H100 comparison](sampling_blocks_h100.md) for complete
fit timings and exact replay checks.
