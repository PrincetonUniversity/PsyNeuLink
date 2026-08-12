# Batched compiler benchmarks (asv)

[airspeed velocity](https://asv.readthedocs.io) benchmarks tracking the GPU
batched Triton simulator across commits (`benchmarks/batched.py`): `DDM`,
`DDMGraph` (transfer→DDM), `LCA` (isolated width-2, cue-driven),
`StabilityFlexibility` (LCA+DDM, `stateful_graph`) and `CSISurrogate` (the
co-evolving research model, `coevolving_graph`) — each `time_run` +
`track_checksum`, swept over the number of estimates (GPU lanes).

## Baseline reset at `504319d70d`

**Results recorded before `504319d70d` are not comparable to later ones** and
should not be read as regressions or improvements:

- the RNG stream layout was decoupled from the step caps, which changes every
  stochastic draw — so every `track_checksum` shifted;
- bounded loops gained an early exit, which changes every `time_run`;
- `CSISurrogate` results are additionally **void**: it uses `PARAM_SETS = 8`,
  and the co-evolving kernel was computing only parameter set 0, so those runs
  measured an eighth of the work and checksummed uninitialised memory.

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

`.asv/results/` is committed (the data); `.asv/env/` and `.asv/html/` are
git-ignored (regenerable). Results are machine-tagged (`DavePC`).

## Inspect

```bash
.venv/bin/asv show <commit>        # stored timings/checksums for a commit
.venv/bin/asv compare <c1> <c2>    # regression/improvement between two commits
```

Check `nvidia-smi` before a run. These benchmarks are sensitive to other GPU
work — see "Benchmarking Methodology" in
`Scripts/Debug/pec_batch_compile/BATCH_COMPILE_WIP.md` for the traps, several of
which produced credible-looking numbers that were wrong by 2.5x to 30x.
