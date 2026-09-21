# CSI fitting handoff

Use the `feat/likelihood_compile` branch. The scripts in [handoff/](handoff/)
run one participant per process or Slurm array task, with no existing fit or
warm-start files required. They use the Study 3 real-sequence model with fixed
LCA leak 12, competition 3, LCA noise 0, and DDM noise 0.1.

| Runner | Implementation | Default fitting configuration |
| --- | --- | --- |
| `cpu` | Continuous direct likelihood; native C++/OpenMP LCA and DDM PDE kernels; exact-gradient L-BFGS-B | Float64; 1 ms DDM mesh, 65 spatial points, RK4 LCA step at most 10 ms; 4 starts, 32 screened random candidates, 200 iterations/start plus polishing |
| `gpu` | PsyNeuLink PEC, Triton **generated batched likelihood**, deterministic observed LCA history, simulated DDM, CMA-ES | 1 ms model step; 12 s horizon with checked histogram-window stopping; 100,000 estimates/candidate; batches of 11 candidates; 5,000 candidate evaluations; 100 RT bins, smoothing sigma 0.5 bins, pseudocount 0.1/cell |

Both fit 13 parameters: three gains, one switch CSI, three thresholds, three
collapse rates, and three nondecision times. Both runners use the expanded
bounds from the archived GB300 population run `direct-all-subjects-expanded-1867`:

| Parameter | CPU and GPU bounds |
| --- | --- |
| Gain | 5–120 |
| Switch CSI | 0–0.3 s |
| Threshold | 0.05–0.30 |
| Collapse rate | −0.3–0 per second |
| Nondecision time | 0.1–0.50 s |

Repeat CSI is zero. Both runners accept `--gain-upper-bound`,
`--threshold-upper-bound`, and `--non-decision-time-upper-bound` overrides.
CPU fitting searches continuously; the GPU retains its original grid spacings
of 0.1 for gain, 0.0005 for threshold, and 0.001 s for nondecision time, using
more grid points for the wider bounds. GPU upper-bound overrides must lie on
those grids. GPU CSVs record the three upper bounds alongside estimator settings.
These are starting fitting configurations, not a guarantee of convergence or
an equivalence between the two objectives. See the caveats below before
interpreting results.

## Checkout, storage, and data

On Della, place the checkout and all large files on scratch. David's checkout is
`/scratch/gpfs/CSES/dmturner/PsyNeuLink`; substitute your own writable allocation.
The same scratch checkout can be accessed from `della` and `della-gpu`.

```bash
export CSI_REPO_ROOT=/scratch/gpfs/CSES/dmturner/PsyNeuLink
export CSI_WORK_ROOT=/scratch/gpfs/CSES/dmturner/csi-handoff

# For a new checkout (choose your own paths above first):
git clone --branch feat/likelihood_compile --single-branch \
  https://github.com/PrincetonUniversity/PsyNeuLink.git "$CSI_REPO_ROOT"

export CSI_DATA_FILE="$CSI_REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting/data_to_fit_study3.csv"
source "$CSI_REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/handoff/environment.sh"
```

For an existing checkout, fetch before selecting the branch:

```bash
cd "$CSI_REPO_ROOT" &&
git fetch origin &&
git switch feat/likelihood_compile &&
git pull --ff-only
```

If Git reports local or untracked files that would be overwritten, preserve
those files before switching; do not force checkout or delete them blindly.
After switching, source the handoff environment as above.
**Obtain the behavioral CSV separately from the data owner;
it is intentionally not in git.** Its default location, if `CSI_DATA_FILE` is
unset, is `csi_fit/data fitting/data_to_fit_study3.csv` in the checkout.

Set these exports in each new login session, or keep them in your own small
configuration file and source it. Set them **before** sourcing `environment.sh`.
All paths must be absolute. The repository defaults to the checkout containing
the handoff scripts; the raw `.slurm` files default to David's path because
Slurm executes a spooled copy of the job script.

| Variable | Default / purpose |
| --- | --- |
| `CSI_REPO_ROOT` | Checkout containing the scripts; override for another checkout |
| `CSI_WORK_ROOT` | `/scratch/gpfs/CSES/$USER/csi-handoff`; Python downloads, environments, caches, temporary files |
| `CSI_VENV` | `$CSI_WORK_ROOT/venv` |
| `CSI_RESULTS_ROOT` | `$CSI_WORK_ROOT/results`; runs and Slurm logs |
| `CSI_DATA_FILE` | Behavioral CSV, as described above |
| `CSI_PYTHON` | `$CSI_VENV/bin/python`; override to use an existing compatible environment |
| `CSI_CPUS` | Local OpenMP threads, default 4; Slurm uses `SLURM_CPUS_PER_TASK` |

The environment sets both `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to that CPU
count. Keep them aligned: setting `MKL_NUM_THREADS=1` makes PyTorch use one
thread even when OpenMP requests more. Each run records the effective Torch
thread count in `run.json`.

The environment file redirects uv/Python downloads, pip, Torch extensions,
TorchInductor, Triton, CUDA, Matplotlib, and temporary storage to scratch, even
if the login environment has cache variables pointing at home. It refuses
storage/output paths under home, including symlinks into home. Setup puts the
virtual environment on scratch. Scratch is working storage: archive results,
data provenance, and environment records in your group's durable storage.

## Python and dependencies with uv

Use Linux x86-64 for this handoff. A CUDA environment supports both runners;
alternatively install a smaller CPU-only environment. Use a recent `uv`
supporting `uv pip install --torch-backend`. If `uv` is already installed, reuse
it. Otherwise, after sourcing `environment.sh` above:

```bash
mkdir -p "$CSI_WORK_ROOT/bin" "$TMPDIR"
curl -LsSf https://astral.sh/uv/install.sh -o "$TMPDIR/install-uv.sh"
UV_INSTALL_DIR="$CSI_WORK_ROOT/bin" UV_NO_MODIFY_PATH=1 sh "$TMPDIR/install-uv.sh"
export PATH="$CSI_WORK_ROOT/bin:$PATH"
uv --version
```

`setup.sh` installs Python 3.12 with uv, creates the environment, installs the
editable checkout and Ninja, checks dependencies, and saves
`$CSI_VENV/requirements-resolved.txt`. Override `CSI_PYTHON_VERSION` if needed.
No conda activation or system Python changes are needed. Python installations
stay under `UV_PYTHON_INSTALL_DIR=$CSI_WORK_ROOT/python`; keep that directory
because the virtual environment refers to it. See uv's
[Python guide](https://docs.astral.sh/uv/guides/install-python/) and
[storage settings](https://docs.astral.sh/uv/reference/storage/).

```bash
# On della-gpu, for one environment that can run BOTH CPU and GPU fits:
bash "$CSI_HANDOFF_DIR/setup.sh" gpu

# ALTERNATIVE: on della, for a CPU-only environment:
# bash "$CSI_HANDOFF_DIR/setup.sh" cpu
```

A C++ compiler with OpenMP support must be on `PATH` at setup **and job time**.
The RHEL system GCC may suffice; otherwise inspect `module avail gcc` and load
an available GCC module before setup and submission. Set `CXX` to its executable
if necessary. Native kernels compile on first use and are cached on scratch;
Ninja is installed by setup. Build parallelism is limited to two processes
(`CSI_BUILD_JOBS` overrides it). Avoid compiler flags such as `-march=native`
that can make a login-node build incompatible with a compute node.

GPU setup explicitly requests CUDA 12.8 Torch wheels (`CSI_TORCH_BACKEND=cu128`),
so setup does not require a visible GPU. Check compatibility with the allocated
node's NVIDIA driver. If necessary, select a different supported CUDA wheel
backend in a **new** environment, e.g. export `CSI_TORCH_BACKEND=cu126` before
setup. Torch and the checkout's `triton` extra must resolve together. Installing
a CUDA module alone does not convert CPU-only Torch into a GPU build. See
[uv's PyTorch guidance](https://docs.astral.sh/uv/guides/integration/pytorch/).

Setup refuses to overwrite an existing environment. To use your existing
scratch `.venv`, set `CSI_PYTHON="$CSI_REPO_ROOT/.venv/bin/python"` and skip setup
after checking it contains the checkout dependencies, Triton (for GPU), and
Ninja. Batch jobs only use that interpreter; they never install packages or run
`uv sync`. The root `uv.lock` is not required. This is a resolved installation,
not a pinned dependency lock; retain the package snapshot and `run.json`, and
do not update the checkout or environment while jobs are using it.

## Validate and run locally or in an allocation

All wrapper subject arguments are **actual `subject_nr` values**, including
Slurm array IDs. The runner translates these into the GPU driver's internal
one-based, first-seen CSV subject index. Never assume those two numbers agree
for a reordered or subset CSV.

```bash
# Read-only preflight: validate data, subject mapping, and show exact commands.
bash "$CSI_HANDOFF_DIR/run.sh" cpu --subject 1 --dry-run
bash "$CSI_HANDOFF_DIR/run.sh" gpu --subject 1 --dry-run
bash "$CSI_HANDOFF_DIR/run.sh" --help

# In a suitable compute allocation (or on a local workstation):
bash "$CSI_HANDOFF_DIR/run.sh" cpu --subject 1 --smoke
bash "$CSI_HANDOFF_DIR/run.sh" gpu --subject 1 --smoke

# Full fits, each in a new output directory:
bash "$CSI_HANDOFF_DIR/run.sh" cpu --subject 1 --seed 17 --starts 8
bash "$CSI_HANDOFF_DIR/run.sh" gpu --subject 1 --seed 17 --simulation-seed 31
```

Dry-run does not import Torch, compile kernels, allocate a GPU, create output,
or fit. It needs the configured interpreter and CSV. Smoke mode keeps the data
and model resolution but uses one CPU start and iteration without polishing, or
64 GPU estimates and 22 candidate evaluations. First-run compilation still
takes time. Smoke results are **not usable scientific fits**. Run real fits and
GPU smoke tests on allocated compute nodes, not login nodes.

CSV columns required by both paths:
`subject_nr, sequence, T1, T2, S1, S2, S3, S4, correct_response, decision,
response_time, likelihood_include_mask`. Use integer subject IDs, finite inputs,
choices 0/1, correct response −1/+1, and RTs in **seconds**, including masked
rows. Encode the entire mask column consistently as `0/1` or `True/False` with
no blanks. This wrapper requires included observations in all three conditions
(`NoInstruction`, `RealRare`, `RealFrequent`). It preserves CSV row order and
retains masked rows in the state history; do not drop them or sort by RT.
Other sequence conditions are filtered out by both drivers. Optional GPU
predictive output also requires `task_transition` and `congruence` columns.

## Submit on Della

Log into `della.princeton.edu` for CPU work and `della-gpu.princeton.edu` for GPU
work. Export your configuration and source `environment.sh` on that host.
The helper creates scratch log directories **before** calling `sbatch`, sets
the job working directory to scratch, and exports the configuration to the job.
It works from any current directory. Options before `--` go to Slurm; options
after `--` go to the fitting runner.

```bash
# Inspect only: this does NOT submit a job.
bash "$CSI_HANDOFF_DIR/submit.sh" cpu --dry-run --account=cses --array=1 -- --smoke
bash "$CSI_HANDOFF_DIR/submit.sh" gpu --dry-run --account=cses --array=1 -- --smoke

# Submit one smoke task first, then inspect its logs and outputs:
bash "$CSI_HANDOFF_DIR/submit.sh" cpu --account=cses --array=1 -- --smoke
bash "$CSI_HANDOFF_DIR/submit.sh" gpu --account=cses --array=1 -- --smoke

# Full arrays: ONLY use 1-97 if these are the actual IDs in your CSV.
bash "$CSI_HANDOFF_DIR/submit.sh" cpu --account=cses --array=1-97%8
bash "$CSI_HANDOFF_DIR/submit.sh" gpu --account=cses --array=1-97%2

# Sparse subject IDs and a different fit budget:
bash "$CSI_HANDOFF_DIR/submit.sh" gpu --account=cses --array=4,12,27%2 \
  --time=12:00:00 -- --estimates 20000 --iterations 10000 --seed 43
```

Replace `cses` with an account authorized for you, or omit `--account` to use
your site's default. Account and QOS are deliberately not hard-coded in the
templates. Do not pass a fixed `--subject` or fixed `--output` to an array:
every task would select that subject or compete for that directory. Use
`--array` for IDs and `CSI_RESULTS_ROOT` for a shared results parent.

You can also submit the `.slurm` files directly. Slurm does not expand shell
variables in `#SBATCH` lines, and it opens logs before running the script.
The raw templates default their working directory and logs to David's scratch
checkout, so even a submission from home writes logs to scratch. For another
checkout, override `--chdir` and the log paths as below (`submit.sh` does this
automatically):

```bash
mkdir -p "$CSI_RESULTS_ROOT/logs" "$CSI_WORK_ROOT"
sbatch --export=ALL --account=cses --array=1 \
  --chdir="$CSI_WORK_ROOT" \
  --output="$CSI_RESULTS_ROOT/logs/cpu-%A_%a.out" \
  --error="$CSI_RESULTS_ROOT/logs/cpu-%A_%a.err" \
  "$CSI_HANDOFF_DIR/della_cpu.slurm"
# For GPU use della_gpu.slurm and distinct gpu-%A_%a log names.
```

CPU template: 8 cores, 8 GB RAM, 30 minutes. GPU template: one `gpu40` GPU,
4 cores, 16 GB host RAM, 1 hour. Della selects the GPU partition from the
resource request; its submission policy rejects an explicit `--partition=gpu`.
One-hour GPU submissions are assigned to `gpu-test`; longer requests use a
different scheduling class. Changing an existing job's walltime alone did not
change its class in our live check, so a new submission was needed for that move.
`--mem` controls host RAM, not GPU memory. These are initial requests, not
runtime guarantees. Start with
one participant, inspect `jobstats JOBID` and `sacct -j JOBID`, then tune memory,
time, and array concurrency. Each GPU process batches candidates on **one**
GPU; allocating more GPUs does not accelerate that process. Arrays distribute
participants. Lower `--buffer-mib` to microbatch with less likelihood buffer
memory; changing `--batch-size` also changes the CMA-ES population.

Princeton's [Della documentation](https://researchcomputing.princeton.edu/systems/della)
describes the CPU/GPU login hosts, `gpu40` constraint, and automatically selected
QOS. It also explains scratch storage and differences between node CPUs.
The templates also incorporate live submission checks on Della: leave the GPU
partition and QOS unspecified. Check `sinfo`, `qos`, and your account permissions
on the cluster before the first submission. These scripts target ordinary
x86-64 Della nodes, not the ARM Grace Hopper or restricted H100/H200 partitions.

Preparation checks passed locally on subject 1: CPU and GPU smoke fits, exact
CPU fresh-score agreement, and GPU rescoring with two independent seeds. That
workstation used Python 3.13.3, Torch 2.13.0+cu130, Triton 3.7.1, and an RTX
2080 Ti. On Della, the setup script also successfully created a scratch
environment with Python 3.12.14, Torch 2.11.0+cu128, and Triton 3.6.0. Full CPU
and GPU fits subsequently completed, as described below. Those historical fits
predate the threshold scheduling fix (`0b15c416f8`); they do not validate the
corrected GPU model. On 2026-09-21, the corrected checkout passed 40 focused
scheduling/compiler regressions, CPU and GPU smoke fits, exact CPU fresh-score
agreement, and GPU rescoring at the new 100,000-estimate default with two
independent seeds. Fresh full-fit validation on Della is pending. Inspect the
job results before starting an array.

A full local CPU fit of Study 3 subject 1 took **4 minutes 7 seconds**, including
first-use native compilation and the independent fresh-score check, on an
Intel Core i7-9700K with eight Torch/OpenMP threads. This used the original,
narrower upper bounds (gain 35, threshold 0.25, nondecision time 0.4 s) with a
1 ms DDM mesh, four starts, 32 screened candidates, up to 200 iterations/start,
and polishing, with 561 retained rows and 485 included observations. The run
recorded 815 evaluations and approximately 1.61 GiB peak resident memory;
the fresh score reproduced the fitted likelihood exactly. This is a timing
reference for one participant, not a Della runtime guarantee. The optimizer
reported success but its stricter stationarity checks were false, so completion
and reproducible scoring alone do not establish convergence. The 30-minute CPU
request leaves headroom for this configuration; increase `--time` for larger
fit budgets or participants that need longer. Reassess runtime after expanding
bounds; the timing above predates the expanded CPU defaults.

The archived GB300 expanded-bound subject-1 fit reached gain 38.61, threshold
0.275, and nondecision time 0.4385 s, exceeding each of those earlier ceilings.
Its fitting schedule also used eight starts, previous fits as initial points,
and more polishing; matching its bounds alone does not reproduce that schedule.
To refit an existing CPU result within the expanded bounds, keep it as one
starting point using `--initial-parameters /path/to/fit.json`.

With the expanded bounds and corrected threading, Della CPU job `13971325_1`
completed subject 1 in **3 minutes 25 seconds**, including startup, fitting,
and independent rescoring, on `della-h17n6`. It used eight allocated CPUs,
four starts (one initialized from the earlier narrow-bound fit), 32 screened
candidates, up to 200 iterations/start, and default polishing. The run recorded
865 evaluations and approximately 2.54 GiB peak resident memory. Its manifest
confirmed eight Torch threads, and a process sample confirmed CPU work on all
eight solver threads. All four starts reported optimizer success. The final
log likelihood was -3063.504140, with exact fresh-score agreement and no invalid
or zero-probability included rows. The earlier upper-bound hits were resolved;
the NoInstruction collapse rate still reached its lower bound of -0.3, and the
stricter stationarity checks remained false. This supports the 30-minute CPU
request for this configuration, but does not establish convergence or predict
all subjects' runtimes. Changed bounds, initial points, and node hardware mean
this is not a controlled speedup comparison with the earlier one-thread run.

For a historical timing reference, the archived GB300 recovery study
`gpu1ms-comprehensive-recovery-1475` contains 384 completed fits with 100,000
estimates/candidate, 5,000 candidate evaluations, and a 1 ms model step.
Their recorded `fit_duration` values range from 18.8 to 25.0 minutes, with a
21.1-minute median and a 24.0-minute 95th percentile. These were synthetic
recovery fits with strict truncation disabled and a 12 s horizon. Their code
predated the generated likelihood default. Hardware, data, and compiler changes
make this a reference rather than a prediction for Della.

Della job `13983447_1` ran on a full A100 40 GB with the earlier narrow GPU
bounds, a 50 s horizon, and strict trajectory completion. It timed out after
logging 4,874 of 5,000 evaluations at 100,000 estimates/candidate; Slurm recorded
65 minutes 23 seconds elapsed against a one-hour request. No final fit was saved.
That timing does not measure the current window-scoring configuration.

The later generated-path Della A100 job `13988705_1` completed in **23 minutes
15 seconds** (22 minutes 47 seconds inside fitting), using 100,000 estimates,
5,000 candidate evaluations, 1 ms steps, expanded bounds, a 12 s maximum
horizon, and checked histogram-window stopping. It predates the scheduling
fix, so its fitted parameters require refitting. The separate historical
handwritten-path replay took 39 minutes 31 seconds inside fitting on A100
versus 21 minutes 4 seconds on GB300; that hardware comparison uses a different
code path from this handoff and is not a runtime prediction for it.

A controlled local RTX 2080 Ti comparison used subject 1's recorded data,
11 fixed representative candidates, 100,000 estimates, 1 ms steps, and a 1 GiB
buffer budget. Medians of two warm calls per setting were:

| Horizon | Strict completion | Window scoring |
| --- | --- | --- |
| 50 s | 18.09 s/batch | 9.00 s/batch |
| 12 s | 15.64 s/batch | 6.20 s/batch |

All eleven scores matched exactly across all four settings. These objective
batch timings support the faster default; they are not full-fit timings or a
validation of every candidate in the expanded search space. Tune the one-hour
GPU request using completed Della fits and increase `--time` when needed.

## Outputs and follow-up checks

Each invocation creates a distinct directory under `CSI_RESULTS_ROOT`, prints
its location, and records `run.json`: exact commands, subject mapping, data
SHA-256, Git revision/status, package versions, host, and job settings.
`source.diff` records tracked changes relative to HEAD; untracked source files
are listed in status but are not archived. For reproducibility use a committed
checkout. `--output /absolute/scratch/path` chooses an exact **new** directory;
existing directories are rejected to prevent accidental overwrites.

- CPU: `fit.json` and `fresh-score.json`. The runner initially writes
  `fit.partial.json`, independently scores it in a fresh process on the same
  mesh, checks finite likelihood, agreement within `1e-8`, and absence of
  invalid/zero-probability included rows, then renames it to `fit.json`.
- GPU: `fit.csv`, including parameter and estimator settings. Add
  `--predictive-simulations 100` for a potentially large free-running predictive
  CSV. This is a plug-in simulation at the fitted vector, not Bayesian
  posterior sampling. It is off by default.
- Slurm stdout/stderr: `CSI_RESULTS_ROOT/logs/csi-{cpu,gpu}-JOB_TASK.{out,err}`.
  Local runs print to the terminal; redirect or use `tee` if you need a log.

`run.json` status `complete` means the workflow finished its checks; it does
not certify optimizer convergence. Inspect CPU `success`, `message`,
`stationary`, `coordinate_stationary`, `run_results`, and bound hits. A killed
or timed-out process can leave `status=running` and partial files. There is no
automatic optimizer checkpoint/resume; rerun in a new directory. A completed
CPU fit can be reused with `--initial-parameters /path/to/fit.json`.

Rescore GPU parameters using larger simulation counts and independent seeds:

```bash
bash "$CSI_HANDOFF_DIR/run.sh" gpu --subject 1 \
  --rescore /absolute/path/to/fit.csv --estimates 200000 \
  --rescore-seeds 101 102 103

# Optional full-trajectory validation, using a longer horizon:
bash "$CSI_HANDOFF_DIR/run.sh" gpu --subject 1 \
  --rescore /absolute/path/to/fit.csv --estimates 200000 \
  --rescore-seeds 101 --strict-truncation --horizon 50

# CPU mesh-refinement fit, initialized at an existing solution:
bash "$CSI_HANDOFF_DIR/run.sh" cpu --subject 1 \
  --initial-parameters /absolute/path/to/fit.json --starts 1 \
  --ddm-time-step 0.0005 --ddm-spatial-points 129 --lca-max-step 0.005
```

Use these commands in compute allocations, or pass the runner options after
`--` to `submit.sh`. GPU rescoring reads parameter columns only: explicitly
repeat the original `--time-step`, `--bins`, `--smoothing-sigma`, `--pseudocount`,
and horizon if they differed from the handoff defaults. Keep the same data and
subject. Compare scores under the same settings across multiple seeds.

## Local likelihood parameter sweep

`csi_likelihood_parameter_sweep.py` rescores saved direct solutions with both
objectives; it does not run new fits. It retains complete observed histories,
uses the original inclusion masks, compares densities in the same units, and
records individual trial scores across seeds. The original PNL/LLVM model
remains the reference for simulation behavior.

From an activated CUDA environment at the repository root:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export CSI_AUDIT_ROOT=/tmp/csi-likelihood-audit

# Fit JSON files must be arranged as subject-N/fit.json beneath --fits.
python Scripts/Debug/pec_batch_compile/csi_fit/csi_likelihood_parameter_sweep.py \
  --data "$CSI_DATA_FILE" --fits /path/to/saved-direct-fits --subjects all \
  --estimates 20000 --repeats 3 --output "$CSI_AUDIT_ROOT/screen"

python Scripts/Debug/pec_batch_compile/csi_fit/csi_likelihood_parameter_sweep.py \
  --data "$CSI_DATA_FILE" --fits /path/to/saved-direct-fits \
  --subjects 1,4,7,42,71,81 --deep --synthetic \
  --estimates 20000 --repeats 3 --output "$CSI_AUDIT_ROOT/deep"

# Recheck anchors, selected local discrepancies, and one poor joint point.
python Scripts/Debug/pec_batch_compile/csi_fit/csi_likelihood_parameter_sweep.py \
  --data "$CSI_DATA_FILE" --refine-from "$CSI_AUDIT_ROOT/deep" \
  --estimates 100000 --repeats 3 \
  --output "$CSI_AUDIT_ROOT/refinement"

python Scripts/Debug/pec_batch_compile/csi_fit/csi_likelihood_parameter_sweep.py \
  --data "$CSI_DATA_FILE" --direct-convergence-from "$CSI_AUDIT_ROOT/deep" \
  --output "$CSI_AUDIT_ROOT/convergence"

python Scripts/Debug/pec_batch_compile/csi_fit/csi_likelihood_sweep_report.py \
  "$CSI_AUDIT_ROOT"
```

Saved population fits and audit artifacts are local research outputs, not part
of a clean checkout. Supply `--fits` explicitly when using another account.
Use `--expanded-subject-one --expanded-fit /path/to/fit.json` to substitute an
expanded-bound subject-1 anchor. Use `--replay-from /path/to/archived/deep` with
a new `--output` to rescore every archived candidate and dataset exactly.
Completed dataset directories are skipped on resume; use a new output root
when changing settings. Native direct scoring requires `ninja` on `PATH`.
Use the handoff environment's scratch cache settings on Della.

The report separates local probes from broad search-bound stress cases and
records zero direct probabilities and empty GPU bins. Close per-trial scores
do not establish identical optimization surfaces: histogram smoothing,
pseudocounts, and model discretization can change nearby parameter rankings.

## Scientific and operational caveats

- **The original PNL composition executed with LLVM defines model behavior.**
  Compare simulated choices and RTs directly when checking compiler fidelity;
  the legacy LLVM likelihood implementation is not the reference likelihood.
  A subject-1 audit at the expanded direct-fit solution, with 1 ms steps and
  all 561 trial inputs, found identical deterministic choices and a maximum
  RT difference of `3.25e-8` s. Independent stochastic sequences also agreed
  (256 GPU and 32 LLVM replicates). See the
  [scheduling-fix audit](DIRECT_LIKELIHOOD_NOTES.md#threshold-scheduling-fix-2026-09-16).
- **Refit GPU results obtained before the threshold scheduling fix.** The source
  now publishes a fresh threshold before the first DDM step and requires a
  fresh DDM call before publishing a finished response. Previously, held
  thresholds could carry over between trials and produce spurious responses.
  The generated path follows the corrected source schedule. At the saved
  subject-1 direct solution, this reduced the comparable log-density gap from
  89.68 to 3.42 (100,000 estimates, 1 ms). It does not make the two objectives
  identical. Do not pool pre-fix and post-fix GPU scores or fit results.
- **The CPU direct solver does not reproduce every PNL scheduling convention.**
  Continuous versus discrete LCA history timing can matter at the expanded
  gains, particularly in RealRare trials. These differences
  can affect fitted parameters; close aggregate scores or a Brownian-bridge
  check alone do not establish that the fitted models are equivalent. Do not
  reset or clamp the compiler's threshold independently of the source model.
- **CPU and GPU are different numerical objectives.** The direct solver models
  a continuous LCA/diffusion with moving boundaries and integrates choice flux
  over the default 1 ms RT recording interval. GPU fitting uses discrete Euler
  simulation, endpoint boundary checks, and a smoothed choice/RT histogram.
  Do not compare their raw likelihoods, AIC, or BIC as if on a common scale.
  Use within-objective rescoring, predictions, and recovery checks.
- Both handoff paths condition persistent LCA state on observed RT history.
  GPU `generated` history uses the batched compiler, not the handwritten CSI
  history oracle. This assumes deterministic LCA noise; adding LCA noise needs
  a different likelihood strategy. Neither wrapper uses the legacy LLVM
  likelihood as the CPU fitting route.
- The GPU default is **1 ms**, finer and more expensive than the older 10 ms
  experiments. `--time-step 0.01` selects that coarser model. The driver rescales
  ITI, switch CSI, boundary increments, and maximum steps together. CPU CSI is
  continuous seconds; GPU CSI is scheduler steps, and collapse is per step in
  GPU CSVs. Never copy parameter numbers between formats without conversions.
- GPU fitting defaults to a 12 s DDM horizon and checked histogram-window
  stopping (`--no-strict-truncation`). A simulated trajectory can stop early
  only when later outcomes cannot contribute to the required histogram bins;
  it remains in the original probability denominator. Stochastic sampling of
  unscored trials is skipped, but all observed state history is retained.
  Nonfinite samples and unresolved horizon truncations without a valid window
  cutoff still fail. This mode does not validate unexecuted tails. Use
  `--strict-truncation --horizon 50` for full-trajectory validation of selected
  fits; even that finite cap need not accommodate every simulated tail.
- Smoothing and pseudocounts stabilize the GPU objective but change it; these
  handoff defaults intentionally differ from the raw driver's unsmoothed
  defaults. Finite Monte Carlo scores can hide poorly supported histogram
  cells. Repeat optimizer seeds and independently rescore, especially near
  ties; a single optimizer result is insufficient evidence of a best fit.
- The direct solver remains a research prototype. Multi-start optimization,
  bound checks, mesh refinement, and parameter recovery remain necessary.
  Upper-bound overrides change the fitting study; pass the same values to both
  runners when matched bounds are intended. The older population job scripts
  also used private warm starts and different optimization budgets; matching
  bounds alone does not reproduce those runs.
- Trial ordering and masking affect the persistent state. The existing
  convention compares the first retained task with the last retained task to
  determine its switch status; it does not introduce block resets. Confirm
  this and the assumed 1 ms RT recording resolution match your experiment.
- Keep package/compiler versions fixed across a campaign. The first native
  extension or Triton compilation can be slow; warm one task before launching
  a large array sharing the cache. Avoid mixing incompatible environments in
  one cache root. Review storage growth and archive successful fits before
  scratch cleanup.

Implementation background and validation limits are in
[DIRECT_LIKELIHOOD_NOTES.md](DIRECT_LIKELIHOOD_NOTES.md). The underlying CLIs
offer additional diagnostics: `csi_direct_likelihood.py --help` (including
`staged-fit` and `grid-refinement`) and
`data fitting/expectation_fit_study3.2_real_sequences_single_csi_leak12.py --help`.
