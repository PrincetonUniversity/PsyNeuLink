# Distributed PEC fitting (Dask)

Run `ParameterEstimationComposition` fits with candidate evaluations distributed
across a Dask cluster.

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
rebuilds a serial PEC plus its inputs. Each worker caches its PEC and reuses the
compiled LLVM binary across evaluations. PsyNeuLink must be importable in the
worker environment.

`distributed_options` keys (all optional except `pec_factory`):

| key | meaning | default |
|---|---|---|
| `pec_factory` | worker recipe `(data) -> (pec, inputs)` | **required** |
| `worker_cores` | LLVM threads per worker | `$SLURM_CPUS_PER_TASK`, else available cores divided by workers |
| `max_concurrent_evaluations` | candidates dispatched per ask/tell round | live worker count |

*Live worker count* is the number of workers registered with the scheduler when
the fit starts (srun tasks - 2 with the launcher; the `LocalCluster` size on a
single node).

Optimizers: any optuna sampler/study and `differential_evolution`. LLVM execution
and a single scalar objective only.

### Choosing the cluster (optional)

If neither option is set, PsyNeuLink uses the launcher client when present;
otherwise it creates a single-node `LocalCluster`.

| key | use it to |
|---|---|
| `client` | run the fit on a `dask.distributed.Client` you provide instead of an auto-resolved cluster |
| `n_workers` | set how many workers the auto-created single-node `LocalCluster` spawns |

Pass `client` to use an existing `dask.distributed.Client`, including a notebook,
a non-SLURM cluster, or an existing scheduler:

```python
client = Client("tcp://head-node:8786")          # or Client(KubeCluster(...)), etc.
distributed_options={"client": client, "pec_factory": pec_factory}
```

PsyNeuLink does not close a client supplied by the caller.

## Running

**Single node -**

```bash
python study.py
```

**Multiple nodes -** Use one SLURM `srun` step: rank 0 is the scheduler,
rank 1 is the driver, and ranks 2+ are workers.

```bash
srun -n <workers+2> python -m psyneulink.dask_run study.py
```

See [submit_dask.slurm](submit_dask.slurm) for a batch template and
[study.py](study.py) for a DDM example.

For a larger Stability-Flexibility example, see
[../stability_flexibility/stability_flexibility_dask.py](../stability_flexibility/stability_flexibility_dask.py)
and its [submit_stabflex_dask.slurm](../stability_flexibility/submit_stabflex_dask.slurm).
That `pec_factory` imports `make_stab_flex` from the co-located
`stability_flexibility.py`, so the batch script puts that directory on
`PYTHONPATH`.

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
