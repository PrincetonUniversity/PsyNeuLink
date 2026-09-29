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
Recovery replaces the recorded responses with synthetic observations. The
older adaptive and NDT-profiling commands are documented separately in the
[legacy fitting guide](fitting_acceleration/LEGACY_FITTING.md); they require a
different likelihood and are **not supported shortcuts for conditioned fits**.

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
checks use an RTX 2080 Ti; complete conditioned fits have also run on H100.

## First run: a self-contained execution check

The tracked [smoke CSV](examples/smoke_subject.csv) contains **fabricated** inputs
and responses for eight trials. It is only an execution fixture, not behavioral
data or a recovery benchmark. No private data is needed for this check:

```bash
python "$DAWA_SCRIPTS/dawa_pec_fit.py" \
  --data "$DAWA_SCRIPTS/examples/smoke_subject.csv" --subject 1 \
  --estimates 128 --pseudocount 0.00128 --evaluations 21 \
  --validation-seeds 91001 91002 --predictive-estimates 64 \
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
| `--estimates` | Particles per candidate and trial; default 100,000 |
| `--evaluations` | Total parameter proposals, not generations; default 5,000 |
| `--population` | CMA-ES population and candidate batch size; default 10 |
| `--optimizer-storage` | `memory` by default; `journal` additionally saves optimizer internals but does not enable automatic resume |
| `--max-steps` | Strict execution cap per trial; default 4,000, matching the conditioned pilot |
| `--validation-estimates` | Fresh-seed rescoring budget; defaults to the fitting budget if omitted |
| `--validation-seeds` | Distinct independent repetitions; defaults 91001, 91002, 91003 |
| `--pseudocount` | Per-cell contamination weight at the fitting budget; default 1 |

Defaults changed during handoff cleanup: fixed fits now use memory storage,
the execution cap is 4,000, and validation seeds avoid those used in the earlier
accuracy study. The manifest records resolved options. Explicit old options
remain available for reproducing previous runs.

The [conditioned H100 pilot](CONDITIONED_RECOVERY.md) took **34.5–35.9 minutes
for 3,000 search proposals**, or 36.8–38.2 minutes including setup and final
validation. These are measurements for one synthetic subject, not a runtime
promise for the 5,000-proposal command above. Earlier 6–28 minute marginal-fit
benchmarks evaluate a different objective.

The [latest matched compiler benchmark](CONDITIONED_LIKELIHOOD.md#h100-benchmark-and-profile-2026-09-29)
measured **2.19 s per batch of four candidates** at 100k particles on one H100,
versus 5.40 s on the local 2080 Ti. That projects to about **46 minutes for
5,000 evaluations** for those proposals, before optimizer and validation
overhead. This is a throughput measurement, separate from the complete recovery
fits above.

## Read the results and diagnose failures

| Output | Purpose |
| --- | --- |
| `manifest.json` | Resolved settings, hashes, device, and phase: `preparing`, `fitting`, `validating`, `predicting`, `complete`, or `failed` |
| `progress.json` | Completed proposals, best training candidate, elapsed fitting time, invalid-candidate count |
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
  --validation-seeds 91001 91002 --predictive-estimates 64 \
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
The launcher inherits the driver's conditioned-likelihood and memory-storage
defaults. Historical A100 Slurm measurements used the marginal objective;
conditioned H100 measurements are documented in the recovery pilot.

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
- [Legacy acceleration](fitting_acceleration/README.md): historical marginal benchmarks and reproductions.

Observation weighting and state gathering now use general fused GPU operations,
with exact baseline comparisons and lower tensor memory use. Large particle
batches remain dominated by simulation; smaller budgets benefit more from the
reduced launch overhead.

Remaining work includes simulation-kernel optimization and further launch reduction,
sequential adaptive particle budgets, the LLVM reset fix, automatic fit resume,
and observation-kernel sensitivity on recorded data. Missing outcomes and
multiple disjoint subject sequences in one filter call are not supported.
Parameter-identifiability and direct-likelihood research are separate from this
handoff. The current driver uses fixed budgets and fits NDT jointly with the
other coordinates.
