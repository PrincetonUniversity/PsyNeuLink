# Fitting Dawa's LC/LCA model

This directory contains the model and tools for fitting choices and response
times, and for checking whether a fit can recover known parameters. The model
has control, stimulus, decision, and response LCA layers. An LC mechanism
modulates gain in the three downstream layers. Control state carries over
between trials, so trial order matters.

There are currently two fitting workflows:

| Task | Script | Runs on |
| --- | --- | --- |
| Fit one subject's recorded choices and RTs | [dawa_pec_fit.py](dawa_pec_fit.py) | NVIDIA GPU |
| Generate a synthetic subject and recover its parameters | [dawa_pec_recovery.py](dawa_pec_recovery.py) | NVIDIA GPU |

Both commands use the same model, parameter bounds, and CMA-ES fitting pipeline.
Use the fit command for empirical data: recovery replaces the CSV's recorded
responses with simulated ones. The similarly named `dawa_pec_fit_benchmark.py`
only times fixed parameter proposals.

## Environment and data

These instructions apply to the `feat/likelihood_compile` working branch.
Use Python 3.10 or newer on Linux or WSL with an NVIDIA GPU. From the repository root,
activate your existing PsyNeuLink environment, or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[triton]'

# Check that this environment can see a CUDA GPU.
python -c 'import torch, triton; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
```

The GPU check must succeed before running fits. On a cluster, run it and
the fits inside a GPU allocation. Use a CUDA-enabled PyTorch installation
compatible with the node's NVIDIA driver. On Della, keep the checkout,
environment, and results in your scratch allocation.

Obtain the behavioral CSV separately; it is not tracked in Git. Choose a data
file and a writable results directory:

```bash
export DAWA_DATA="$PWD/Scripts/Debug/pec_batch_compile/dawa/dawa_lca_model/flanker_data_part1.csv"
export DAWA_RESULTS=/absolute/path/to/your/dawa-results
```

The runners use these columns:

| Columns | Meaning |
| --- | --- |
| `subject_nr` | Subject ID selected by `--subject` |
| `T1, T2, S1, S2, S3, S4` | Task and stimulus inputs in their original order |
| `PrevCongruency` | Previous condition, coded 0 or 1; rows with missing values are excluded |
| `likelihood_include_mask` | 1 to score the observation, 0 to retain the trial only for state history |
| `decision`, `response_time` | Recorded choice (0/1) and RT in **seconds**; required only for empirical fitting |

Both previous-congruency levels must have scored trials. Keep masked trials
and preserve row order. Inputs and empirical outcomes must be finite on all
retained rows, including masked trials. RTs must be positive; scored RTs must
lie within the configured 0–3 s histogram range. The runners check these
requirements before creating a run directory.

## Fit a subject's recorded responses

Start with a short execution check, from the repository root:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit.py \
  --data "$DAWA_DATA" --subject 1 \
  --trials 16 --estimates 128 --evaluations 21 --predictive-estimates 64 \
  --output "$DAWA_RESULTS/fit-smoke"
```

Then fit the complete subject:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit.py \
  --data "$DAWA_DATA" --subject 1 \
  --estimates 100000 --evaluations 5000 --population 10 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --output "$DAWA_RESULTS/subject1-start0"
```

`--subject` is the actual `subject_nr` value in the CSV. Each command fits one
subject. Use a new process and output directory for each additional subject.
For a second start, use `--start 1 --optimizer-seed 202 --simulation-seed 37`
and another output directory, such as `subject1-start1`.

The short check verifies execution, not fit quality. Omit `--trials` for real
fits. First use compiles GPU kernels. Every run needs a **new output directory**;
the runner refuses to overwrite one and does not automatically resume an
interrupted fit.

`--estimates` counts simulated trajectories per parameter proposal, with one
response per trial in each trajectory. `--evaluations` counts parameter
proposals, not generations. `--population` controls the CMA-ES population and
candidate batch size. These meanings are the same for recovery.

## Try adaptive fitting

Both runners support `--fit-strategy adaptive`. This starts with small simulation
budgets, adds independent samples when candidate rankings are uncertain, and
checks promising candidates at the maximum budget. A final refinement retains
the parameter correlations learned during the search:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit.py \
  --data "$DAWA_DATA" --subject 1 --fit-strategy adaptive \
  --estimates 100000 --evaluations 5000 \
  --output "$DAWA_RESULTS/subject1-adaptive"
```

For recovery, use the same options with `dawa_pec_recovery.py`. In this mode,
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
See [the original adaptive experiment](fitting_acceleration/adaptive_h100.md)
and [its quality diagnosis](fitting_acceleration/quality_diagnosis.md) for the
motivation behind this revision. Use multiple starts when comparing parameter estimates.

The [revised H100 test](fitting_acceleration/adaptive_v2_h100.md) took **8.1–8.5
minutes**, versus a **27.5-minute** fixed-budget benchmark (about **3.3× faster**).
Both revised fits had fresh-seed likelihoods close to their corresponding fixed
fits. The earlier policy was faster but fit less well. This is still experimental:
two starts on one synthetic subject do not establish recovery across subjects,
and some LC parameter estimates still differ noticeably.

### Profile nondecision time

Add `--profile-ndt` to an adaptive fit or recovery run to optimize nondecision
time inside each proposal. CMA-ES then searches seven dynamic parameters.
The compiler accumulates exact decision-time counts during the same complete
trial histories; the fitter evaluates the 0.1–0.3 s NDT grid from those counts.
It retains the existing histogram, smoothing, and pseudocount rules.

This is experimental and currently requires `--fit-strategy adaptive`. Equal
histogram scores can cover an interval of NDT values; the reported value is the
lowest grid representative, not evidence of 0.1 ms estimation precision.
Reference checks, refinement, and final selection also optimize NDT, while
independent validation scores the chosen full eight-parameter vector.

In the [H100 experiment](fitting_acceleration/ndt_h100.md), profiling NDT with
smaller search/refinement budgets took **5.8–6.1 minutes overall**, about
**1.39× faster** than the previous adaptive fits, with similar fresh-seed
likelihoods in two starts. A smaller-budget control without profiling fit less
well. LC parameter estimates still varied. To use the
tested budgets, add these options to a fit or recovery command:

```bash
  --fit-strategy adaptive --profile-ndt --estimates 100000 \
  --adaptive-search-evaluations 1000 --adaptive-refine-evaluations 300 \
  --optimizer-storage memory
```

`--profile-ndt` alone keeps the existing search/refinement budgets. The measured
gain includes reducing those budgets; it is not a 1.39× faster simulator.

NDT-profiled adaptive fits now run their independent sampling blocks together
by default. This keeps the same samples and fitting decisions while reducing
GPU launches and repeated preparation. Add `--no-batch-sampling-blocks` for
separate-block execution, which uses less count-buffer memory. See the
[H100 comparison](fitting_acceleration/sampling_blocks_h100.md) for complete
fit timings and exact replay checks.

## Run a short recovery check

From the repository root, with the environment activated:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --data "$DAWA_DATA" --subject 1 \
  --trials 16 --estimates 128 --evaluations 21 --predictive-estimates 64 \
  --output "$DAWA_RESULTS/recovery-smoke"
```

This generates responses, runs a small optimization, and checks the resulting
parameters with fresh simulation seeds. As with the empirical smoke test,
these settings verify execution rather than scientific recovery.

## Run a full parameter recovery fit

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --data "$DAWA_DATA" --subject 1 \
  --estimates 100000 --evaluations 5000 --population 10 \
  --start 0 --optimizer-seed 101 --simulation-seed 29 \
  --output "$DAWA_RESULTS/recovery-start0"
```

This creates one synthetic subject using the selected subject's complete input
sequence, then fits it with CMA-ES. Omit `--trials` for a full subject.

For a second start on the **same synthetic observations**:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --data "$DAWA_DATA" --subject 1 \
  --estimates 100000 --evaluations 5000 --population 10 \
  --start 1 --optimizer-seed 202 --simulation-seed 37 \
  --output "$DAWA_RESULTS/recovery-start1"
```

Keep `--data-seed` and `--model-seed` unchanged to reuse the synthetic dataset.
To generate a new synthetic subject on the same design, change `--data-seed`
(default `20260925`) and use another output directory. Keep generation, fitting,
and validation seeds distinct. Changing only the optimizer seed tests search
variability, not recovery across different synthetic datasets.

The completed H100 pilot took about **28 minutes per start** for 760 trials,
720 scored observations, 100,000 estimates, and 5,000 proposals. Runtime depends
on the device, input sequence, and parameters. LC modes and scaling were weakly
recovered in that pilot; compare multiple starts and synthetic datasets before
interpreting parameter estimates. See the [pilot results](pec_recovery/README.md).

## Submit a GPU job on Della

[dawa_gpu.slurm](dawa_gpu.slurm) runs either workflow on one full A100, with
eight CPU cores, 24 GB of host RAM, and a two-hour limit. It uses an existing
Python environment; jobs do not install packages. Log into `della-gpu` and
set these paths to your own scratch checkout and results directory:

```bash
export DAWA_REPO_ROOT=/absolute/path/to/your/scratch/PsyNeuLink
export DAWA_PYTHON="$DAWA_REPO_ROOT/.venv/bin/python"
export DAWA_DATA="$DAWA_REPO_ROOT/Scripts/Debug/pec_batch_compile/dawa/dawa_lca_model/flanker_data_part1.csv"
export DAWA_RESULTS=/absolute/path/to/your/scratch/dawa-results
mkdir -p "$DAWA_RESULTS/logs"
```

The commands use your default Slurm account. Submit an empirical fit or recovery
run with the same driver options used locally:

```bash
sbatch --chdir="$DAWA_RESULTS" \
  --output="$DAWA_RESULTS/logs/fit-%j.log" \
  "$DAWA_REPO_ROOT/Scripts/Debug/pec_batch_compile/dawa/dawa_gpu.slurm" \
  fit --subject 1 --estimates 100000 --evaluations 5000

sbatch --chdir="$DAWA_RESULTS" \
  --output="$DAWA_RESULTS/logs/recovery-%j.log" \
  "$DAWA_REPO_ROOT/Scripts/Debug/pec_batch_compile/dawa/dawa_gpu.slurm" \
  recovery --subject 1 --estimates 100000 --evaluations 5000
```

Results go to `fit-JOB_ID` or `recovery-JOB_ID` under `DAWA_RESULTS`. Use
`--output /absolute/path/to/new/run` after `fit` or `recovery` to choose another
directory. Slurm options belong **before** the script path; Python options
belong **after** the mode. Create the log directory before submitting.

For a short Slurm test, add `--time=00:05:00` before the script path and use
`--trials 16 --estimates 128 --evaluations 21 --predictive-estimates 64` after
the mode. Check `squeue -u "$USER"`, then inspect the log and final `fit.json`
or `recovery.json`. A successful run also sets `manifest.json` status to
`complete`.

For subject arrays, add e.g. `--array=1,2,3` before the script path and omit
`--subject`: array IDs are used as actual `subject_nr` values. Each task gets
its own GPU and output directory. To limit concurrent tasks, use
`--array=1,2,3%2`. For multiple starts on one subject, submit separate jobs with
explicit `--subject`, `--start`, and seeds.

The launcher keeps caches and temporary files under `DAWA_RESULTS/.work`;
override this with `DAWA_WORK_ROOT`. If the Python environment needs a CUDA
module, export `DAWA_CUDA_MODULE` before submission (the tested environment
uses `cudatoolkit/13.0`). Leave `CUDA_VISIBLE_DEVICES` to Slurm. Della selects
the partition from the resource request, so no explicit partition is needed.

Both modes passed a [Slurm A100 test](dawa_benchmark_results.md#slurm-fitting-and-recovery-handoff-test-2026-09-25)
on the full 760-trial subject at 100,000 estimates and 21 proposals. These short
runs validate execution; use the full budget and multiple starts for fitting.

## Parameters and current settings

Both runners fit eight coordinates: seven parameter types, with a
separate LC mode for each previous-congruency level.

| Parameter | Search bounds | Value used to generate synthetic data |
| --- | --- | --- |
| Response threshold | 0.25–0.70 | 0.40 |
| Nondecision time | 0.10–0.30 s | 0.22 s |
| Stimulus/decision/response bias | −0.50–0 | −0.40 |
| Control gain | 5–20 | 12 |
| LC mode, previous congruency 0 / 1 | 0.10–0.90 each | 0.65 / 0.80 |
| LC scaling | 1–4 | 1.5 |
| LC base gain | 3–10 | 5.5 |

Both runners use the tested recovery configuration:

- **10 ms LCA timesteps**; the LC performs ten internal 20 ms steps per model pass.
- **Noise SD 0.1 in each of the four LCAs**, fixed throughout fitting.
- A simulated choice/RT histogram with **100 RT bins over 0–3 seconds**,
  Gaussian smoothing of **0.5 bins (15 ms)**, and **pseudocount 1** per choice/RT cell.
- A separate simulated control-state history for every estimate. Masked
  observations still advance that history.

These are the pilot's settings, not established best choices for every dataset.
The objective scores each trial's simulated choice/RT distribution; it does
not condition latent control state on the observed responses.

Budget and seed options are exposed by `--help` on either command. Generating parameters
(`TRUTH`), starting points (`STARTS`), noise, timestep checks, and histogram
settings are currently specified in [the shared fitting script](dawa_pec_fit.py).
Bounds come from `fit_surface()` in [dawa_batched_simulation.py](dawa_batched_simulation.py).
Changing those model/estimator settings currently requires editing the code;
the exception is the histogram pseudocount, exposed as `--pseudocount`.

## Read the results

| File in the output directory | What to look for |
| --- | --- |
| `progress.json` | Completed proposals, current best parameters/score, elapsed time, invalid-candidate count |
| `fit.json` (empirical fit) | Final fitted values, scores at fresh seeds, observed/predicted summaries, fitting time |
| `recovery.json` | Final fitted values, errors from truth, fitting time, fresh-seed scores, predictive summaries |
| `manifest.json` | Settings, parameter order/bounds, seeds, data/source hashes, device, completion status |
| `observed_subject.csv` or `synthetic_subject.csv` | The selected empirical or generated observations actually fitted, including masked rows |
| `evaluations.jsonl`, `optimizer_trials.csv`, `optimizer.journal` | Search history and optimizer records |
| `optimizer_refinement_trials.csv` (adaptive) | The separate local refinement at the reference simulation budget |

`optimizer.journal` is present only with journal storage. Adaptive reports also
record simulation budgets, independent block seeds, incumbent checks, stopping
reason, and total sampled trajectories. Low-budget search scores may not be
comparable across generations; use the final reference score and fresh-seed
validation to compare fits.

Compare agreement between starts and observed/predicted choice/RT summaries;
for recovery, also compare parameter errors. Higher log likelihood is better,
but a recovery fit can score better than the generating parameters on a finite
synthetic dataset. Fresh-seed rescoring
checks simulation variability on the same observations; it is not held-out
validation. Reaching the evaluation budget does not establish convergence.

Proposals that exceed `--max-steps` are recorded and penalized. If many proposals
fail, inspect their parameters and the model's response durations before
increasing the cap. Final scoring and prediction checks must finish normally.

## Further reading

- [Recovery pilot](pec_recovery/README.md): completed fits, parameter errors, and interpretation.
- [Compiler notes](COMPILER_NOTES.md): simulation checks, scheduling, noise, and GPU configuration.
- [Benchmark results](dawa_benchmark_results.md): measured performance and reproduction commands.
- [Original fitting scripts](dawa_lca_model/README.md): the earlier partition-wide LLVM workflow and Slurm examples.

The older direct-likelihood experiments are indexed in the compiler notes.
For the model with noise in all four LCAs, use the simulation-based workflow
above.
