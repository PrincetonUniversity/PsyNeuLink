# Batched compiler benchmarks (asv)

[airspeed velocity](https://asv.readthedocs.io) benchmarks tracking the GPU
batched Triton simulator across commits (`benchmarks/batched.py`): `DDM`,
`DDMGraph` (transfer→DDM), `LCA` (isolated width-2, cue-driven), and
`StabilityFlexibility` (LCA+DDM) — each `time_run` + `track_checksum`, swept
over the number of estimates (GPU lanes).

## Constraints

- Runs the compiled **`triton` GPU** path → needs **CUDA + triton**; benchmarks
  skip otherwise. Run on the GPU box.
- Uses the project's existing (uv) venv (`environment_type: existing`) so the
  heavy torch/triton/CUDA stack is reused, not reinstalled per commit.
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
