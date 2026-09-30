# Fitting Dawa's LC/LCA model

Start here to fit recorded choices and response times with the compiled GPU
model. The supported workflow uses the original nonlinear model, Gaussian
noise in all four LCA layers, and **observation-conditioned particle filtering**.
Control state carries across trials, so every retained observation updates the
state distribution before the next trial.

| Task | Entry point |
| --- | --- |
| Fit one subject's recorded responses | [dawa_pec_fit.py](dawa_pec_fit.py) |
| Generate and fit one synthetic subject | [dawa_pec_recovery.py](dawa_pec_recovery.py) |
| Submit either workflow on Della | [dawa_gpu.slurm](dawa_gpu.slurm) |
| Inspect or modify the PNL composition | [full_lca_model_lc.py](dawa_lca_model/full_lca_model_lc.py) |

Both fitting commands use the same compiler, bounds, and CMA-ES optimization.
Recovery replaces the recorded responses with synthetic observations.
Use `--fit-strategy adaptive` to vary simulation effort during fitting. The
likelihood selects the policy: [conditioned fitting](CONDITIONED_STAGED_FITTING.md)
uses smaller-particle exploration followed by full-count filtering and refinement;
marginal fitting uses the earlier simulation-block approach. The
[legacy fitting guide](fitting_acceleration/LEGACY_FITTING.md) documents that
marginal policy and NDT profiling, which requires the marginal likelihood.

Fixed fits and conditioned adaptive fits now run through `pec.run(inputs=inputs)`.
The adaptive policy lives in `PECOptimizationFunction`, configured with
`fit_strategy="adaptive"` and `adaptive_options`; the Dawa driver supplies the
model, data, budgets, and reporting callback. Final rescoring uses the public
`pec.log_likelihood_batch(...)` method with independent seeds and a requested
particle count. The marginal adaptive research policy remains in the Dawa scripts.
See [the PEC interface](CONDITIONED_STAGED_FITTING.md#pec-interface) for details.

For external optimizers, `pec.log_likelihood_batch(..., adaptive=True,
adaptive_options={...})` selects a particle budget and returns mean log scores,
Monte Carlo standard errors, and precision/work diagnostics. This is separate
from the adaptive fitting strategy; ordinary batch calls still return score
arrays. See [adaptive likelihood evaluation](CONDITIONED_STAGED_FITTING.md#adaptive-likelihood-evaluation)
for paired comparisons, particle limits, and the distinction between Monte Carlo
precision and likelihood bias.

## Install and check the GPU

Use this checkout's `feat/likelihood_compile` branch, Python 3.10 or newer, and
Linux or WSL with an NVIDIA CUDA GPU. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[triton]'
python -c 'import torch, triton; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'

export DAWA_SCRIPTS="$PWD/Scripts/Debug/pec_batch_compile/dawa"
export DAWA_RESULTS=/absolute/path/to/your/dawa-results
```

An existing compatible environment can be used instead. The GPU check must
succeed before fitting; use CUDA-enabled PyTorch compatible with the NVIDIA
driver. On a cluster, run the check and fits inside a GPU allocation. The local
checks use an RTX 2080 Ti; complete conditioned fits have also run on H100 and A100.

## First run: a self-contained execution check

The tracked [smoke CSV](examples/smoke_subject.csv) contains **fabricated** inputs
and responses for eight trials. It is only an execution fixture, not behavioral
data or a recovery benchmark. No private data is needed for this check:

```bash
python "$DAWA_SCRIPTS/dawa_pec_fit.py" \
  --data "$DAWA_SCRIPTS/examples/smoke_subject.csv" --subject 1 \
  --estimates 128 --pseudocount 0.00128 --evaluations 21 \
  --validation-estimates 128 --validation-seeds 91001 91002 --predictive-estimates 64 \
  --output "$DAWA_RESULTS/fit-smoke"
```

Expect `manifest.json` and `fit.json` with `status: complete`. First use compiles
GPU kernels. The small particle count tests execution, not fitting accuracy.
The explicit pseudocount keeps the same contamination fraction as the full
100k-particle fit below; leaving it at 1 with 128 particles would make
contamination dominate the observation model.

Every run needs a **new output directory**. Runs do not automatically resume.
Completed search parameters and each finished validation repetition are saved
before prediction checks, so a late failure preserves those results.

## Supply the behavioral data

Obtain the behavioral CSV separately; private data are not tracked in Git:

```bash
export DAWA_DATA=/absolute/path/to/flanker_data_part1.csv
```

| Required columns | Meaning |
| --- | --- |
| `subject_nr` | Subject ID selected by `--subject` |
| `T1, T2, S1, S2, S3, S4` | Task and stimulus inputs, in original trial order |
| `PrevCongruency` | Previous condition, coded 0 or 1; rows missing this value are excluded |
| `likelihood_include_mask` | 1 adds the observation's log score; 0 still conditions state history |
| `decision`, `response_time` | Observed choice (0/1) and RT in **seconds**; only empirical fitting requires these columns |

Both previous-congruency levels must have scored trials. Keep masked trials and
preserve row order. All retained inputs and empirical outcomes must be finite;
masked rows are not missing observations. RTs must be positive, and scored RTs
must be within 0–3 s. A masked RT above 3 s expands the histogram range in 30 ms
increments. The runner validates these requirements before creating output.
One invocation fits one subject. A separate process and directory is needed
for each additional subject or start.

## Fit a complete subject

```bash
python "$DAWA_SCRIPTS/dawa_pec_fit.py" \
  --data "$DAWA_DATA" --subject 1 \
  --likelihood conditioned --estimates 100000 --pseudocount 1 \
  --evaluations 5000 --population 10 --max-steps 4000 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --validation-estimates 1000000 --validation-seeds 91001 91002 91003 91004 91005 \
  --output "$DAWA_RESULTS/subject1-start0"
```

For a second start, use `--start 1 --optimizer-seed 202 --simulation-seed 37`
and a new directory. Omit `--trials` for full fits; that option selects a prefix
for execution checks. The particle count is a working search budget, not a
guarantee of likelihood precision or optimizer convergence.

| Option | Interpretation / default |
| --- | --- |
| `--estimates` | Particles per candidate and trial; default 100,000; reference/refinement count for adaptive fitting |
| `--fit-strategy` | `fixed` by default; `adaptive` selects the budget policy appropriate to the likelihood |
| `--evaluations` | Total parameter proposals, not generations; default 5,000 |
| `--adaptive-min-estimates` | Exploration count for conditioned fits (default 10,000); minimum race budget for marginal fits (default 5,000) |
| `--population` | CMA-ES population and candidate batch size; default 10 |
| `--optimizer-storage` | `memory` by default; `journal` additionally saves optimizer internals but does not enable automatic resume |
| `--max-steps` | Strict execution cap per trial; default 4,000, matching the conditioned pilot |
| `--validation-estimates` | Fresh-seed rescoring budget; default 1,000,000 particles per candidate and repetition |
| `--validation-seeds` | Five distinct independent repetitions by default: 91001, 91002, 91003, 91004, 91005 |
| `--pseudocount` | Per-cell contamination weight at the fitting budget; default 1 |

Defaults changed during handoff cleanup: fixed fits now use memory storage,
the execution cap is 4,000, and validation seeds avoid those used in the earlier
accuracy study. Both fixed and adaptive fits now validate with one million
particles across five seeds by default; recovery and the Slurm launcher inherit
these defaults. Override both validation settings for short execution checks.
The manifest records resolved options. Explicit old options
remain available for reproducing previous runs.

### Complete-subject timings, 2026-09-30

The [handoff benchmark](fitting_acceleration/conditioned_handoff_20260930.json)
fit empirical subject 1 with **760 retained trials, 720 scored**, using
`--fit-strategy adaptive` and all other fitting/validation defaults. Both runs
used the same source snapshot, observations, initial parameters, and seeds.
Each used one GPU and started with empty compilation caches.

| Device | Search, refinement, and selection | Complete driver run |
| --- | ---: | ---: |
| H100 NVL on `della-rse` | 593.0 s / **9.88 min** | 672.9 s / **11.22 min** |
| A100 SXM4 80 GB through Slurm | 776.1 s / **12.94 min** | 880.9 s / **14.68 min** |

The complete driver includes setup, first-use compilation, final validation at
**1,000,000 particles × five independent seeds**, and 4,096 predictive
simulations. Each validation seed scores both the fitted and initial parameter
vectors. Driver timing excludes Python startup and queue wait. Including launch
overhead, the H100 process took 11.39 minutes; A100 Slurm job `14755287` took
15.17 minutes after a 7-second queue wait and exited successfully.

Both searches switched to refinement after 2,001 proposals at 10k particles,
then completed 600 proposals at 100k, plus reference checks and final selection.
The 5,000-proposal option is a maximum budget for adaptive fitting; these runs
used **2,601 optimizer proposals**. All proposal scores, selected parameters,
and five final validation results matched exactly across GPUs. Neither run had
an invalid/truncated search proposal. The mean independently rescored fitted
log likelihood was **720.289**, with Monte Carlo standard error **0.057**;
its improvement over the initial parameters was 159.446 ± 0.066 (one MC SE).
Pseudocount scales from 1 during 100k refinement to 10 during 1M validation,
preserving the observation model's contamination fraction.

The H100 was **1.31× faster** for this complete workload. These are single-run,
single-subject measurements, not an optimizer convergence result or an
across-subject runtime guarantee. Monte Carlo precision does not measure bias
from the binned, smoothed observation model. Fixed-count fitting remains the
default; explicitly select the adaptive strategy to reproduce this workload:

```bash
python "$DAWA_SCRIPTS/dawa_pec_fit.py" \
  --data "$DAWA_DATA" --subject 1 --fit-strategy adaptive \
  --output "$DAWA_RESULTS/subject1-adaptive"
```

For Slurm, add `--fit-strategy adaptive` to the submission example below. The
timed A100 run overrode the launcher's time limit with `sbatch --time=01:00:00`.
Source/data hashes, work counts, exact settings, and remote artifact locations
are recorded in the benchmark JSON; raw observations are not tracked here.

Historical comparisons remain in the [fixed conditioned recovery pilot](CONDITIONED_RECOVERY.md),
the [adaptive conditioned pilot](CONDITIONED_STAGED_FITTING.md), and the
[matched compiler throughput benchmark](CONDITIONED_LIKELIHOOD.md#h100-benchmark-and-profile-2026-09-29).
They used different workloads and cannot establish the speedup of adaptive
fitting on this empirical subject. Earlier marginal-fit benchmarks also
evaluate a different objective.

## Read the results and diagnose failures

| Output | Purpose |
| --- | --- |
| `manifest.json` | Resolved settings, hashes, device, and phase: `preparing`, `fitting`, `validating`, `predicting`, `complete`, or `failed` |
| `progress.json` | Completed proposals, best training candidate, elapsed fitting time, invalid-candidate count |
| `adaptive.json`, `optimizer_refinement_trials.csv` | Adaptive policy, reference checkpoints, final selection, work counts, and refinement trials; saved before validation |
| `fit_checkpoint.json` | Completed search result, saved before validation; not a resumable optimizer checkpoint |
| `validation.json` | Each completed fresh-seed evaluation, budget, pseudocount, and final summary |
| `fit.json` / `recovery.json` | Final parameters, timing, independent validation, and predictive summaries |
| `observed_subject.csv` / `synthetic_subject.csv` | Exact observations fitted, including masked rows |
| `latent_subject.csv` | Recovery only: simulated responses before measurement noise |
| `evaluations.jsonl`, `optimizer_trials.csv` | Proposal history and optimizer trial table |
| `optimizer.journal` | Additional optimizer state when journal storage is requested |

Higher scores are better. `validation_summary` reports the mean and
single-evaluation SD of each complete-run log score and paired candidate
comparison. Its `mc_standard_error` describes uncertainty in the mean over
seeds. With one seed, SD and standard error are null. These are simulation
uncertainties on the same observations, not held-out validation or parameter
confidence intervals. Per-trial factors from different filters are never pooled.

Predictive summaries describe **unconditional latent choice/RT simulations**
before the observation kernel. They are not predictions conditioned on each
observed trial, and matched synthetic observations include additional measurement
noise. Compare these quantities with that distinction in mind.

A failure after output creation records `failed_phase` and the error in the
manifest when Python can handle the exception. Abrupt termination, node failure,
or SIGKILL can leave the last recorded phase. Inspect the job log too. Existing
output is never overwritten. If validation or prediction fails, the completed
search remains in `fit_checkpoint.json`; an incomplete run is not a validated fit.

Truncated proposals are recorded and penalized during search. Final validation
and prediction require all simulations to finish normally. Investigate frequent
truncation before increasing the cap. The cap does not change the model's 10 ms
LCA timestep. A zero-support error with `--pseudocount 0` means no simulated
particle supported that observation; it is not silently replaced by a valid score.

## Check synthetic recovery

Replace the fit entry point with `dawa_pec_recovery.py`, keeping the same budget
and validation options. It uses only the CSV's input design and mask. A short
self-contained check is:

```bash
python "$DAWA_SCRIPTS/dawa_pec_recovery.py" \
  --data "$DAWA_SCRIPTS/examples/smoke_subject.csv" --subject 1 \
  --estimates 128 --pseudocount 0.00128 --evaluations 21 \
  --validation-estimates 128 --validation-seeds 91001 91002 --predictive-estimates 64 \
  --output "$DAWA_RESULTS/recovery-smoke"
```

By default, recovery generates a complete latent history and then applies the
same observation law used for scoring. Measurement noise never changes the
simulated state history. `--observation-model latent` explicitly selects raw
model responses instead. Latent overflow outside the generation domain is an
error, not a reason to discard and regenerate a history.

For repeated fits to the same synthetic observations, keep `--data-seed`
(default 20260925), `--observation-seed` (20260926), `--model-seed` (29), input
design, and observation settings unchanged. Keep the pseudocount/particle ratio
unchanged if changing the particle budget. Change `--data-seed` to generate a
new subject on the same design. Generation, observation, fitting, prediction,
and validation streams must use distinct seeds; validation repetitions must
also have distinct seeds. The model-construction seed is a separate stream.

## Submit a GPU job on Della

The launcher requests one full A100, eight CPUs, 24 GB RAM, and two hours. It
uses an existing environment and does not install packages. Set absolute paths
on your scratch allocation before submitting:

```bash
export DAWA_REPO_ROOT=/absolute/path/to/scratch/PsyNeuLink
export DAWA_PYTHON="$DAWA_REPO_ROOT/.venv/bin/python"
export DAWA_DATA=/absolute/path/to/flanker_data_part1.csv
export DAWA_RESULTS=/absolute/path/to/scratch/dawa-results
mkdir -p "$DAWA_RESULTS/logs"

sbatch --chdir="$DAWA_RESULTS" --output="$DAWA_RESULTS/logs/fit-%j.log" \
  "$DAWA_REPO_ROOT/Scripts/Debug/pec_batch_compile/dawa/dawa_gpu.slurm" \
  fit --subject 1 --estimates 100000 --evaluations 5000 \
  --validation-estimates 1000000 --validation-seeds 91001 91002 91003 91004 91005
```

Use `recovery` instead of `fit` for synthetic data. Slurm options go before the
script path; Python options go after the mode. Results default to
`fit-JOB_ID` or `recovery-JOB_ID`; `--output` can select a new directory.
For a smoke job, point `DAWA_DATA` to the tracked fixture and use the small
budgets and pseudocount above. Allow for first-use compilation.

For subject arrays, add `--array=1,2,3%2` before the script path and omit
`--subject`. Array IDs are actual subject IDs; each task receives its own GPU
and output directory. The `%2` limits concurrency. Default caches live under
`DAWA_RESULTS/.work`, overridable with `DAWA_WORK_ROOT`. If needed, export
`DAWA_CUDA_MODULE` for your environment; leave `CUDA_VISIBLE_DEVICES` to Slurm.
The launcher inherits the driver's conditioned-likelihood, memory-storage,
and final-validation defaults. It also retains `--fit-strategy fixed` unless
overridden; the complete-subject timings above explicitly used `adaptive`.

For current daytime experiments on the shared `della-rse` host, use at most
one H100 and leave the second available to other users.

## Model and likelihood contract

| Fitted coordinate | Bounds | Recovery generating value |
| --- | --- | --- |
| Response threshold | 0.25–0.70 | 0.40 |
| Nondecision time | 0.10–0.30 s | 0.22 s |
| Stimulus/decision/response bias | −0.50–0 | −0.40 |
| Control gain | 5–20 | 12 |
| LC mode for previous condition 0 / 1 | 0.10–0.90 each | 0.65 / 0.80 |
| LC scaling | 1–4 | 1.5 |
| LC base gain | 3–10 | 5.5 |

The runner fixes Gaussian noise SD at 0.1 in each LCA. LCAs integrate at 10 ms;
the LC executes ten internal 20 ms steps per scheduler pass. Control state
persists; the other LCAs reset. The recurrent schedule keeps processing layers
advancing while bias and weight controllers publish once per trial.

The observation kernel uses 30 ms RT bins, Gaussian smoothing SD 15 ms, and
optional uniform contamination. With `N` particles, `K` joint choice/RT cells,
and pseudocount `alpha`, contamination probability is `K*alpha/(N+K*alpha)`.
At 100k particles, 100 RT bins, two choices, and alpha=1, this is about 0.2%.
Validation automatically scales alpha with its particle budget. If changing
`--estimates` between fits, scale `--pseudocount` proportionally to preserve
the observation model. Expanding the histogram for masked RTs changes `K` and
therefore the total contamination fraction, which is recorded in the manifest.

This is the likelihood under a specified observation model, estimated with
finite particles. It is not an exact unsmoothed RT density. Every retained row
conditions state; the mask selects which log factors contribute to the score.
With masked rows, this is not the full joint likelihood or generally the joint
likelihood of scored observations conditional on all masked observations.
See [CONDITIONED_LIKELIHOOD.md](CONDITIONED_LIKELIHOOD.md) for boundary, overflow,
resampling, and state-transport semantics.

The general `PECOptimizationFunction` does not enable conditioning automatically.
These runners set `conditioned_likelihood=True` explicitly. The preserved
[partition-wide LLVM scripts](dawa_lca_model/README.md) are an older workflow.
A known LLVM reset/modulation mismatch remains; Python is the simulation
reference used to check the GPU implementation. See the
[reset diagnosis](dawa_benchmark_results.md#first-trial-llvm-rt-discrepancy-reset-diagnosis-2026-09-25).

## Validation and remaining engineering work

- [Compiler notes](COMPILER_NOTES.md): supported components, reset and schedule tests.
- [Conditioned accuracy](CONDITIONED_ACCURACY.md): exact-reference checks and particle-budget uncertainty.
- [Conditioned performance](CONDITIONED_LIKELIHOOD.md#conditioned-loop-optimization-2026-09-29): preserved baseline, compiler optimizations, and current GPU measurements.
- [Conditioned recovery](CONDITIONED_RECOVERY.md): full fits, independent rescoring, and measured H100 runtimes.
- [Adaptive conditioned fitting](CONDITIONED_STAGED_FITTING.md): full-history budget stages, ranking calibration, and fitting pilot.
- [Legacy acceleration](fitting_acceleration/README.md): historical marginal benchmarks and reproductions.

Observation weighting and state gathering now use general fused GPU operations,
with exact baseline comparisons and lower tensor memory use. Large particle
batches remain dominated by simulation; smaller budgets benefit more from the
reduced launch overhead.

Remaining work includes simulation-kernel optimization and further launch reduction,
broader validation of staged particle budgets, the LLVM reset fix, automatic fit resume,
and observation-kernel sensitivity on recorded data. Missing outcomes and
multiple disjoint subject sequences in one filter call are not supported.
Parameter-identifiability and direct-likelihood research are separate from this
handoff. Fixed budgets remain the default; adaptive fitting is opt-in. Both fit
NDT jointly with the other coordinates.
