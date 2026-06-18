# PEC dask benchmark

Model-agnostic harness for benchmarking parallel `ParameterEstimationComposition`
(PEC) maximum-likelihood fits over Dask. One process runs one fit configuration
and appends a JSON metrics row; SLURM launchers fan a sweep out across configs.

## Layout

```
bench.py              entrypoint: run ONE config, append a metrics row to a .jsonl
bench_core.py         importable Dask worker (evaluate_loglik) + ask/tell driver
                      (run_fit) + the Optuna sampler factory (make_sampler)
models/               pluggable model providers (ddm, stabflex) -- the contract
                      is documented in models/__init__.py
summarize.py          per-results-file summary table (median per config + speedup)
plot_core_scaling.py  plots for the CMA-ES core-scaling study (historical)
plot_samplers.py      plots for the per-sampler study (sweep_samplers.sh)

slurm/                SLURM scripts
  run_config.slurm        pinned/exclusive run of ONE config (fair timing); the
                          building block every sweep launcher submits
  run_config_quick.slurm  unpinned/non-exclusive variant for quick checks
  sweep_core_grid.sh      sweep: worker x core grid, CMA-ES, fixed popsize=32
  sweep_samplers.sh       sweep: per-sampler worker x core grid, popsize=NW
  sweep_singlenode.slurm  single-node regular vs dask-local sweep
  run_stabflex.slurm      heavy-model parallel-vs-serial fit (multinode verify)
  smoke.slurm             tiny dask-srun mechanics smoke test

results/              one .jsonl (or a .d/ of them) per study
logs/                 SLURM .out logs land here
plots/                figures + CSVs written by the plot_*.py scripts
```

## Modes (`bench.py --mode`)

- `regular`    — standard `pec.run()`: serial trials, estimates over all cores.
- `dask-local` — one-node `LocalCluster`, `n_workers` x `worker_cores`.
- `dask-srun`  — `SLURMRunner` inside an allocation: `srun -n (n_workers+2)`
  launches scheduler (rank 0) + driver (rank 1) + workers (ranks 2+) at once.

## Samplers (`bench.py --sampler`)

`cmaes` (default), `tpe`, `tpe_noliar`, `random`, `qmc`, `gp`, `nsga2` — built by
`bench_core.make_sampler`. The driver's ask/tell loop is identical for every
sampler; `--optimizer-popsize` is the per-round batch size (and the CMA-ES /
NSGA-II population). The sweeps differ in how they set it:

- `sweep_core_grid.sh` pins **popsize = 32** (one fixed optimizer trajectory; the
  grid isolates the hardware throughput surface). Peak throughput sits at
  `n_workers = popsize`.
- `sweep_samplers.sh` pins **popsize = n_workers** (the "popsize = num_workers"
  regime), so each config records both throughput and recovery error and the
  cross-sampler comparison shows whether that heuristic generalises.

## Run a sweep

```bash
# CMA-ES worker x core grid (fixed popsize):
./slurm/sweep_core_grid.sh

# Per-sampler grid (popsize = num_workers); override the set if you like:
SAMPLERS="cmaes gp" ./slurm/sweep_samplers.sh

# Inspect anytime (sweeps also submit a dependent summary/plot job):
.venv/bin/python3 summarize.py results/samplers.d/*.jsonl
.venv/bin/python3 plot_samplers.py
```