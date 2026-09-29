# Adaptive particle budgets for conditioned fitting

The experimental `--fit-strategy adaptive` option selects a staged particle-budget
policy for `--likelihood conditioned`, restoring inexpensive exploration while
keeping the corrected observation-conditioned likelihood. The same strategy
with `--likelihood marginal` selects the legacy simulation-block policy. Fixed-count
fitting remains the default. This policy uses the existing generic PEC
conditioned objective; it changes neither the nonlinear model nor its clocks,
noise, trial resets, or state carried across trials.

## What the policy does

1. Explore with a fixed smaller count, by default 10,000 particles. Each CMA-ES
   population uses one common count and seed.
2. Every 250 exploration proposals, check up to four best recent candidates
   and the incumbent at the reference count (`--estimates`, normally 100,000).
   Each check restarts the complete subject filter. Low-count scores never
   compete directly with reference-count scores.
3. Start reference-count refinement when the exploration allowance runs out,
   or after two checks improve by less than 0.25 log units once at least 1,000
   exploration proposals have run. This is a refinement trigger, not a
   convergence test. Restart near the reference incumbent while retaining the
   learned CMA covariance and fit all eight coordinates, including NDT.
4. Compare up to eight reference-score finalists with three fresh, common
   reference-count seeds. Select by their **mean complete-run masked log score**.
5. Run the ordinary independent-seed validation at the requested validation
   count. These reserved seeds never guide optimization or final selection.

`--evaluations` caps exploration plus refinement proposals; by default 600
proposals are reserved for refinement. Checkpoint and final-selection filter
runs are additional work, included in `fit_seconds` and the work counters.
`--adaptive-refine-evaluations` controls the reservation. Check intervals and the
minimum exploration count are checked after a population; the actual transition
can be a few proposals beyond the nominal count.

Each filter includes **all retained observations**, including the masked rows
that condition the next trial but do not add a score. At count N, the driver
uses `pseudocount * N / estimates`, keeping the observation contamination law
constant. It reruns filtering from the model's initial state whenever a score
is needed at a different count or seed. It never appends particles to an old
history or pools per-trial factors from independent filters.

Averaging complete masked log scores estimates a finite-particle score
criterion. It does not remove count-dependent bias, and we do not claim an
unbiased masked likelihood. The high-count checks and final validation remain
necessary. Cached NDT profiling is still rejected because changing NDT changes
particle weights and subsequent ancestry.

## Run it

After the setup in the [fitting guide](README.md):

```bash
python "$DAWA_SCRIPTS/dawa_pec_fit.py" \
  --data "$DAWA_DATA" --subject 1 --likelihood conditioned \
  --fit-strategy adaptive --estimates 100000 --pseudocount 1 \
  --adaptive-min-estimates 10000 --evaluations 3000 \
  --adaptive-refine-evaluations 600 --population 10 --max-steps 4000 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --validation-estimates 1000000 --validation-seeds 91001 91002 91003 91004 91005 \
  --output "$DAWA_RESULTS/subject1-adaptive-start0"
```

Use `dawa_pec_recovery.py` with the same options for matched synthetic recovery.
Small smoke tests must override both particle counts, the refinement count,
and the minimum exploration count. They test execution, not fit quality.

The manifest records every policy setting under `adaptive_config` and the
policy driver hash under `adaptive_driver_sha256`. `fit_strategy` is `adaptive`;
`fit_policy` identifies `staged` for conditioned fits and `block_racing` for
marginal fits. These are internal algorithm identifiers, not CLI strategy choices.
`evaluations.jsonl` records the phase, count, seed, and complete score of each
optimizer proposal. `adaptive.json` saves reference checkpoints, covariance
transfer, finalist replicate scores, the selected candidate, and work counters
before independent validation, including a `policy` identifier. Shared controls
use `--adaptive-*`: for example, `--adaptive-refine-evaluations` and
`--adaptive-selection-repeats`. Marginal racing controls are marked as such in
`--help`; `--adaptive-checkpoint-candidates` applies to conditioned fitting.
`candidate_particles` counts initial particle
lanes summed across candidate filter runs, including attempted truncating
batches and retries; it is not a GPU-kernel count or a sum of integration steps.

## Ranking calibration on the H100

The calibration selected 16 candidates before running new seeds: initial,
prefix-best, and near-optimum candidates from the two saved full-subject
fixed-100k fits. It used the same synthetic subject (760 retained, 720 scored
trials), five fresh seeds 12001–12005, and 10k, 25k, 100k, and 400k particles.
All four noise SDs were 0.1; the 30 ms histogram bins, smoothing, and
contamination law were held constant. The reference is the five-seed **mean at
400k**, not an exact likelihood.

| Search count | Mean rank correlation, all 16 | Mean rank correlation, 10 near-optimum | Largest loss from selected winner, reference log units |
| --- | ---: | ---: | ---: |
| 10k | 0.804 | 0.324 | 0.909 |
| 25k | 0.871 | 0.505 | 0.346 |
| 100k | 0.898 | 0.578 | 0.202 |

The final column is the worst across five seeds of the reference score lost by
selecting that seed's best candidate rather than the reference-best member of
this fixed candidate set. It says nothing about unexplored parameter space.
Including the poor starting points makes the full-set correlation look much
better than the fine ordering near the optimum. This supports low-count
exploration followed by full-count refinement; it does not justify returning a
10k-particle winner directly. Five seeds and one subject are a pilot, not a
universal particle-count recommendation.

## Full-subject fitting pilot

One H100 NVL ran the staged recovery from start 0, optimizer seed 101 and filter
seed 29, on the **same synthetic observations** as the earlier fixed-100k
pilot. Generation used seed 20261001 and measurement seed 20261002. Observation
hash, initial parameters, bounds, source model hash, all four noise settings,
model clocks, and estimator settings match exactly.

| Measurement | Earlier fixed fit | Staged fit |
| --- | ---: | ---: |
| Optimizer proposals | 3,000 at 100k | 1,251 at 10k + 600 at 100k |
| Search, checks, and selection | 35.94 min | **13.03 min** |
| Complete run, including setup, validation and predictions | 38.15 min | **15.26 min** |
| Mean validation log score, 1m particles | 233.406 | **233.542** |

Five reserved seeds, 11001–11005, gave staged-minus-fixed differences of
0.1164, −0.0009, 0.1717, 0.2095, and 0.1853. The mean difference was
**+0.1364**, with Monte Carlo standard error **0.0376**. These are complete
masked log scores on the same observations, not held-out-subject evidence or
parameter-recovery intervals. The result supports comparable fit quality in
this pilot; it does not establish general convergence or recovery.

The search switched to refinement after the reference-check plateau. It made
2,035 candidate filter attempts in 335 batch calls, including checks, final
selection, and retries of truncating batches, using 78.31 million initial
particle lanes across those attempts. Sixteen proposals hit the execution cap
and were penalized. No partial simulation was accepted as a valid score.

The observed search-time ratio is **2.76×**, and the complete-run ratio is
**2.50×**. This includes fewer optimizer proposals as well as smaller exploration
budgets. The historical fixed run also predates the generic compiler
optimizations in `54b6dedf2b`, so this is not an isolated policy-only timing
comparison. The staged run includes first-use compilation for its extra
particle counts and candidate batch shapes.

To separate the kernel effect, ten predetermined, nontruncating populations
from the historical fixed fit were replayed twice with the current compiler.
All **200 scores matched exactly**. The first historical population included
first-use compilation; excluding it gives nine comparable warm populations:
**5.370 s previously versus 5.250 s now** per batch of ten,
only **1.023×** faster. Most of this pilot's full-fit improvement therefore
comes from the staged search. Raw startup timings remain in the report and
are explicitly excluded from this warm comparison. This replay is a throughput
check, not a rerun of the full fixed-budget optimization.

Fixed-count fitting stays the default. More subjects, starting points, and
optimizer/filter seeds are needed before making the staged policy the default.
In particular, a cheap-search plateau need not mean the broader parameter
search is complete.

## Artifacts and reproduction

The [compact report](fitting_acceleration/conditioned_staged_h100.json) contains
all calibration scores, candidate coordinates, policy checkpoints, finalist
replicate scores, validation differences, provenance hashes, and matched
fixed-population replay timings. It is generated by
[dawa_conditioned_staged_report.py](dawa_conditioned_staged_report.py).

The raw runs and source snapshots are at:

```text
/scratch/gpfs/CSES/dmturner/dawa-benchmarks/conditioned-staged-20260929/
  source/          # committed 54b6dedf2b, used for ranking calibration
  source-staged/   # same compiler plus the staged fitting driver
  ranking_candidates.json
  ranking.json
  staged-0/
  fixed-replay.json
  rank.sh
  pilot.sh
  replay.sh
  replay_fixed.py
```

All GPU work used GPU 0, UUID
`GPU-372ffaa5-87b1-a620-d84f-4a2d0427a8c4`; the second H100 was left available.
No datasets or full optimization logs were added to the repository.

Reproduce calibration using the same `ranking_candidates.json` and prior
pilot's `synthetic_subject.csv`:

```bash
python "$DAWA_SCRIPTS/dawa_conditioned_accuracy.py" \
  --data /path/to/fixed-0/synthetic_subject.csv --subject 1 \
  --candidates /path/to/ranking_candidates.json \
  --estimates 10000 25000 100000 400000 --seed-start 12001 --repeats 5 \
  --batch-size 4 --max-steps 4000 --output /path/to/new-ranking.json
```

The full pilot used the equivalent policy and settings above through `dawa_pec_recovery.py`,
with `--data-seed 20261001 --observation-seed 20261002` and validation seeds
11001–11005. The candidate calibration and pilot used separate filter seeds. The archived
pilot predates the public-interface rename: its commands, metadata, and
`staged.json` retain the provisional spelling. Current runs use `adaptive` and
`adaptive.json`; the reporter reads both formats. Historical measurements and
source hashes have been preserved.
Render the compact report from completed results:

```bash
python "$DAWA_SCRIPTS/dawa_conditioned_staged_report.py" \
  --accuracy /path/to/ranking.json \
  --fixed /path/to/fixed-0 --staged /path/to/staged-0 \
  --replay /path/to/fixed-replay.json --output /path/to/summary.json
```
