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
there are no CLI flags for them yet.

## Read the results

| File in the output directory | What to look for |
| --- | --- |
| `progress.json` | Completed proposals, current best parameters/score, elapsed time, invalid-candidate count |
| `fit.json` (empirical fit) | Final fitted values, scores at fresh seeds, observed/predicted summaries, fitting time |
| `recovery.json` | Final fitted values, errors from truth, fitting time, fresh-seed scores, predictive summaries |
| `manifest.json` | Settings, parameter order/bounds, seeds, data/source hashes, device, completion status |
| `observed_subject.csv` or `synthetic_subject.csv` | The selected empirical or generated observations actually fitted, including masked rows |
| `evaluations.jsonl`, `optimizer_trials.csv`, `optimizer.journal` | Search history and optimizer records |

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
