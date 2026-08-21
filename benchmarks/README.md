# Batched compiler benchmarks (asv)

[airspeed velocity](https://asv.readthedocs.io) benchmarks tracking the GPU
batched Triton simulator across commits (`benchmarks/batched.py`): `DDM`,
`DDMGraph` (transfer→DDM), deterministic `LCA` (width-2 with a static trial
threshold), and `StabilityFlexibility` (the narrow scalar identity
cue→`OVERRIDE` threshold path feeding the deterministic LCA, followed by a
stochastic DDM; `stateful_graph`), and `CSISurrogate` (the stochastic
co-evolving CSI research model with realistic mixed cues and its five-parameter
recovery surface; `coevolving_graph`) — each `time_run` + `track_checksum`,
swept over the number of estimates (GPU lanes). Runtime LCA step counts and CSI
cue-slope rows must be exact nonnegative integers no larger than `2**24`;
static thresholds are host-discretized, and either form honors the LCA node's
execution cap. `CSISurrogate` fixes DDM noise at the historical `0.1` and
sweeps gain, CSI switch duration, threshold, threshold collapse, and
non-decision time as runtime parameter rows.

## Current LCA and CSI boundaries

Results recorded before the exact deterministic LCA migration are not directly
comparable for `LCA` or `StabilityFlexibility`. Historical `CSISurrogate`
results are also not directly comparable with the newly re-enabled full CSI
workload:

- recurrent activation now starts from the PsyNeuLink value `Logistic(0)` rather
  than from zero;
- the supported LCA subset accepts finite numeric (constant) noise but not a
  stochastic/callable noise source, so it declares no RNG stream;
- removing that stream changes the DDM stream slot, and hence stochastic draws,
  in mixed LCA+DDM kernels;
- the standalone LCA benchmark is now deterministic; and
- the CSI benchmark now executes through typed scheduler/control semantics with
  fixed stochastic DDM noise, mixed repeat/switch cues, and five runtime
  recovery parameters.

The first result recorded after this migration establishes the new LCA and
stability-flexibility baseline. The first result from the re-enabled full CSI
series likewise establishes its new baseline. Historical result files remain
useful for the older implementation and should not be rewritten.

## Baseline reset at `504319d70d`

**Results recorded before `504319d70d` are not comparable to later ones** and
should not be read as regressions or improvements:

- the RNG stream layout was decoupled from the step caps, which changes every
  stochastic draw — so every `track_checksum` shifted;
- bounded loops gained an early exit, which changes every `time_run`;
- `CSISurrogate` results are additionally **void**: it uses `PARAM_SETS = 8`,
  and the co-evolving kernel was computing only parameter set 0, so those runs
  measured an eighth of the work and checksummed uninitialised memory.

The re-enabled full CSI series runs all eight parameter rows and should be
treated as a fresh baseline, not a continuation of those void measurements.

Older points also sit in a separate asv series: the environment key includes the
interpreter path, and they were recorded under `.venv/bin/python` where current
runs use `.venv/bin/python3`. Invoke asv exactly as documented below so future
points stay in one series.

## Constraints

- Runs the compiled **`triton` GPU** path → needs **CUDA + triton**; benchmarks
  skip otherwise. Run on the GPU box.
- Uses the project's existing (uv) venv (`environment_type: existing`) so the
  heavy torch/triton/CUDA stack is reused, not reinstalled per commit.
- asv itself is declared in `dev_requirements.txt`. Note that `uv sync` will not
  notice edits to that file on its own — the extras are `dynamic` in
  `pyproject.toml` and read from it, but uv keys lockfile staleness on
  `pyproject.toml`, so it reports "Resolved ... in 1ms" and does nothing. Use
  `uv lock --refresh`, then `uv sync --extra dev --extra triton` (one name per
  `--extra`; `--extra dev,triton` is a single unknown extra, not two).
- `existing` environments only benchmark the current checkout, and asv won't save
  results unless given a commit hash — so **forward-track**: after each commit,
  run with `--set-commit-hash`. (Historical back-fill isn't supported here.)

## Record a data point for the current commit

```bash
.venv/bin/asv run --set-commit-hash $(git rev-parse HEAD)
.venv/bin/asv publish              # regenerate the HTML dashboard (.asv/html)
.venv/bin/asv preview              # serve it locally
```

`.asv/results/`, `.asv/env/`, and `.asv/html/` are local, regenerable artifacts
and are git-ignored. Store benchmark results in the PR description or another
explicitly reviewed report instead of committing machine-specific ASV data.

## Inspect

```bash
.venv/bin/asv show <commit>        # stored timings/checksums for a commit
.venv/bin/asv compare <c1> <c2>    # regression/improvement between two commits
```

Check `nvidia-smi` before a run. These benchmarks are sensitive to competing GPU
work. Report cold compilation and warm execution separately, state every lane
dimension, exclude correctness-only repeats from timed work, and compare
semantically equivalent workloads. The CSI Triton/LLVM comparison in
`Scripts/Debug/pec_batch_compile/csi_triton_vs_llvm.py` follows that convention.
