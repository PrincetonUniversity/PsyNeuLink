# Distributed PEC fitting (Dask)

Run `ParameterEstimationComposition` fits with candidate evaluations distributed
across a Dask cluster, behind a single constructor flag. You do **no** Dask
administration: no scheduler to start, no addresses to pass.

## Enabling it

Pass `distributed=True` and a `pec_factory` to the optimizer (or the PEC):

```python
optimizer = PECOptimizationFunction(
    method=optuna.samplers.CmaEsSampler(seed=0, popsize=8),
    max_iterations=480,
    distributed=True,
    distributed_options={"pec_factory": pec_factory},
)
```

`pec_factory(data) -> (pec, inputs)` is a top-level, picklable callable that
rebuilds a fresh serial PEC plus its inputs. A live PEC is never shipped to
workers; each worker builds and caches its own from this recipe and reuses the
compiled LLVM binary across evaluations. Keep the factory self-contained (imports
and literals inside) so Dask can ship it by value.

`distributed_options` keys (all optional except `pec_factory`):

| key | meaning | default |
|---|---|---|
| `pec_factory` | worker recipe `(data) -> (pec, inputs)` | **required** |
| `worker_cores` | LLVM threads per worker | `$SLURM_CPUS_PER_TASK`, else cores |
| `max_concurrent_evaluations` | candidates per ask/tell round | live worker count |
| `client` / `scheduler_address` / `scheduler_file` / `n_workers` | advanced cluster escape hatch | — |

Optimizers: any optuna sampler/study and `differential_evolution`. LLVM execution
and a single scalar objective only.

## Running

**Single node (zero config).** A `LocalCluster` is formed automatically:

```bash
python study.py
```

**Multiple nodes (the launcher).** PsyNeuLink forms the cluster from the SLURM
ranks of one srun step — rank 0 scheduler, rank 1 driver (runs your script),
ranks 2+ workers — so workers = tasks − 2:

```bash
srun -n <workers+2> python -m psyneulink.dask_run study.py
```

See [submit_dask.slurm](submit_dask.slurm) for a ready-to-edit batch script and
[study.py](study.py) for a complete DDM example.

## Reproducibility

With common random numbers — `same_seed_for_all_parameter_combinations=True` and a
fixed `initial_seed` — every candidate is scored against identical simulation
noise, so a distributed fit with a tell-order-independent sampler (e.g.
`RandomSampler`) matches the serial fit. Stateful samplers (CMA-ES, QMC) explore
different points under batched ask/tell and are not bit-identical to the serial
driver. Without common random numbers the fit is still valid but not reproducible,
and a warning is issued.

## Requirements

Install the extra: `pip install "psyneulink[dask]"` (adds `dask`, `distributed`,
`dask-jobqueue`). With `distributed=False` (the default) Dask is never imported.
