# Full-sequence PEC sampler recovery

For setup, fitting commands, and a short execution check, start with the
[fitting and recovery guide](../README.md). This page records the completed
pilot and its interpretation.

The [recovery driver](../dawa_pec_recovery.py) generates one synthetic subject
and fits all eight original coordinates with CMA-ES through PEC's compiled
GPU objective. The input sequence and scoring mask come from the selected
subject's design; empirical responses are discarded. One simulated trajectory
supplies one response per trial, including the retained control state.
The shared pipeline now lives in [dawa_pec_fit.py](../dawa_pec_fit.py), whose
own command fits recorded responses without generating a synthetic subject.

## First H100 pilot: completed

Two H100 NVLs on `della-rse` each ran one CMA-ES start on the same synthetic
760-trial subject (720 scored), with 100,000 estimates per proposal. Each
completed **5,000 optimizer proposals**. Fitting took **27.85 and 28.03 minutes**;
total driver times including setup and validation were 28.47 and 28.55 minutes.
The runs executed concurrently on separate GPUs. Base compiler commit:
`6dd18b4202`; the new driver and all input/source hashes are recorded separately.

| Parameter | Generating value | Start 1 fit | Start 2 fit |
| --- | ---: | ---: | ---: |
| Response threshold | 0.400 | 0.382 | 0.383 |
| Nondecision time (s) | 0.2200 | 0.2116 | 0.1960 |
| SDR bias | -0.400 | -0.410 | -0.423 |
| Control gain | 12.00 | 12.52 | 14.70 |
| LC mode, previous condition 0 | 0.650 | 0.379 | 0.309 |
| LC mode, previous condition 1 | 0.800 | 0.693 | 0.533 |
| LC scaling | 1.500 | 2.489 | 2.421 |
| LC base gain | 5.500 | 5.099 | 4.864 |

Recovery is mixed. Threshold, bias, and several other values are relatively
close, particularly in start 1, while LC modes and scaling are appreciably
different from the generating values. Start 2 also has a larger control-gain
error. A good likelihood score does not establish recovery of every parameter.

| Score on the same synthetic observations | Generating values | Start 1 fit | Start 2 fit |
| --- | ---: | ---: | ---: |
| Mean log likelihood over fresh seeds 8101–8103 | 252.177 | 255.060 | 254.884 |
| Mean improvement over generating values | — | 2.883 | 2.707 |
| SD of paired improvement across the three seeds | — | 0.396 | 0.509 |

Both fits beat the generating parameters on every rescoring seed. The two fits
differ by only 0.176 log units on average; the paired difference has SD 0.280
across these seeds, so this small comparison is insufficient to establish a
reliable ranking. These are Monte Carlo repeats on the same observations,
not a confidence interval for parameter recovery or held-out-subject scores.

The finite synthetic subject also differs from its generating distribution in
a relevant summary. For previous conditions 0 and 1, respectively:

| Mean RT (s), scored trials | Previous 0 | Previous 1 | Previous 1 minus 0 |
| --- | ---: | ---: | ---: |
| Synthetic observations (358 / 362 trials) | 0.93645 | 0.94296 | +0.00650 |
| Generating model, fresh simulations | 0.94121 | 0.92836 | -0.01285 |
| Start 1 fit, fresh simulations | 0.92758 | 0.93562 | +0.00804 |
| Start 2 fit, fresh simulations | 0.93137 | 0.93972 | +0.00835 |

Predictions use 4,096 full-sequence trajectories per parameter vector. The
observed condition contrast reverses relative to the generating expectation,
and both fits reproduce its direction. This is consistent with finite-sample
variation contributing to the LC recovery errors. The contrast also reflects
the fixed input design and history, so it does not isolate an LC-mode causal
effect or prove nonidentifiability. More synthetic subjects and generating
vectors are needed for a general recovery assessment.

The final 1,000 proposals improved the best training scores by only 0.111 and
0.091 log units. The best proposals occurred at evaluations 4,932 and 4,225.
This indicates little late improvement at this budget, not a proof of optimizer
convergence. There were 24 and 32 invalid proposals that exceeded the step cap
(0.48% and 0.64%); these were explicitly penalized and recorded. Final fits and
all generating/fitted rescoring and predictive runs passed strict checks.

The preliminary 16-trial execution smoke test completed 21 proposals, checked
generator/PEC sample parity, and completed all three fresh-seed rescores.
Both full runs verified identical generated datasets, their requested clocks,
initial points, and evaluation budgets. Ruff, Python compilation, and
`git diff --check` passed for the driver/docs changes.

See the [compact results](h100_10ms_results.json) and
[parameter recovery / search plots](h100_10ms_recovery.png). Raw histories,
synthetic observations, optimizer journals, and source snapshots remain in
the remote artifact directory listed below.

This pilot uses 10 ms LCA updates, 20 ms LC internal updates with ten updates
per scheduler pass, and independent Gaussian noise SD 0.1 in each of the four
LCAs. Both fixed-parameter specialization and `philox4x_fast_v1` are enabled.
The original model equations and recurrent scheduler are used. The gradient
ideas in the design notes remain deferred.

The fitting objective is the existing **trial-marginal histogram score**, with
100 RT bins spanning 0–3 seconds, smoothing sigma 0.5 bins, and pseudocount 1
per joint choice/RT cell. Every estimate executes the complete sequence;
masked observations still advance state. This objective does not condition
latent control state on observed responses as a particle filter would.

## Run

From the repository root, on a CUDA GPU:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --estimates 100000 --evaluations 5000 --population 10 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --output /tmp/dawa_recovery_start0
```

For the second start, use `--start 1 --optimizer-seed 202 --simulation-seed 37`
and a different output directory. The default generation/model seeds are
shared, so both starts use exactly the same synthetic subject. The output
directory must not already exist. `--source-revision` records the base commit
when executing a copied snapshot without `.git`; the driver also records its
own hash. `--trials 16 --estimates 128 --evaluations 21 --predictive-estimates 64`
is an execution smoke test, not a recovery benchmark.

The generating coordinates, in fitting order, are threshold 0.40, nondecision
time 0.22, SDR bias -0.40, control gain 12, LC modes 0.65/0.80 for previous
congruency 0/1, LC scaling 1.5, and LC base gain 5.5. The optimizer starts do not
use the generating values. Bounds match the source fitting surface; fine grids
are used instead of treating just two endpoints as the available allocations.
The default population is ten, and the 5,000 budget counts proposals rather
than generations. These are two optimizer runs on one generating parameter
vector and one synthetic subject, not a population-wide recovery study.

## Validation and artifacts

Before fitting, the driver requires exact agreement between generation and
the specialized PEC plan at the generating parameters and seed. It also checks
all five timestep constants, the optimizer's initial point, and the completed
evaluation count. Strict truncation checks remain enabled. If a candidate
truncates, the driver identifies it individually, records the failure, and
assigns an explicit invalid-candidate penalty; its paths are not scored as
valid completed responses. Other numerical/programming errors abort the run.

Final generating, initial, and fitted values are rescored at the requested
estimate count with independent seeds 8101, 8102, and 8103. This checks Monte
Carlo score stability on the same synthetic observations; it is not held-out
data validation. Predictive summaries use 4,096 fresh full-sequence simulations
per generating/fitted parameter vector. A finite sample need not be best fit
by its generating parameters.

Each run saves the synthetic CSV, manifest, every evaluated proposal, Optuna
journal, progress checkpoint, optimizer CSV, final parameter errors, fresh-seed
scores, predictive summaries, and elapsed fitting time. The journal is durable
provenance; automatic optimizer resume is not implemented by this driver.
Finishing the budget alone does not assert optimizer convergence or successful
parameter recovery. Compare actual parameter errors, independent score gaps,
and agreement between starts before drawing conclusions.

The first H100 pilot uses the isolated source snapshot and artifacts under:

```text
/scratch/gpfs/CSES/dmturner/dawa-benchmarks/recovery-20260925
```

`source_manifest.json` records every source/input file hash. `env.sh` and
`run.sh` record the environment and the two GPU assignments. Raw outputs stay
outside the repository; the compact results and plot above summarize the
completed first pilot.
