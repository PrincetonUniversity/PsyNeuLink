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
rebuilds a fresh serial PEC plus its inputs. Each worker (where the likelihood evaluation happens) 
builds and caches its own from this recipe and reuses the compiled LLVM binary across evaluations. 
Define it at module level and have it depend only on its data argument and
importable names, so Dask can ship it to workers by value. PsyNeuLink must be
importable in the worker environment.

`distributed_options` keys (all optional except `pec_factory`):

| key | meaning | default |
|---|---|---|
| `pec_factory` | worker recipe `(data) -> (pec, inputs)` | **required** |
| `worker_cores` | LLVM threads per worker | `$SLURM_CPUS_PER_TASK`, else cores |
| `max_concurrent_evaluations` | candidates dispatched per ask/tell round | live worker count |

*Live worker count* = the number of workers registered with the scheduler when the
fit starts (srun tasks − 2 with the launcher; the `LocalCluster` size on a single
node). Defaulting `max_concurrent_evaluations` to it means each round sends one
candidate per worker — every worker busy, nothing idle or queued.

Optimizers: any optuna sampler/study and `differential_evolution`. LLVM execution
and a single scalar objective only.

### Choosing the cluster (optional)

You normally set neither of the options below: with the launcher PsyNeuLink uses the
cluster it formed, and otherwise it creates a single-node `LocalCluster` for you.

| key | use it to |
|---|---|
| `client` | run the fit on a `dask.distributed.Client` you provide instead of an auto-resolved cluster |
| `n_workers` | set how many workers the auto-created single-node `LocalCluster` spawns |

`client` is the door to anything beyond the SLURM launcher or a single node: pass a
`Client` you built yourself — for a notebook, a non-SLURM cluster, or a warm cluster
reused across many fits. It covers connecting to an existing scheduler too, since
that is just `Client("tcp://host:port")` or `Client(scheduler_file="…")`:

```python
client = Client("tcp://head-node:8786")          # or Client(KubeCluster(...)), etc.
distributed_options={"client": client, "pec_factory": pec_factory}
```

You own a `client` you pass (PsyNeuLink never shuts it down), which is exactly what
you want for reuse across fits.

## Running

**Single node -** A `LocalCluster` is formed automatically:

```bash
python study.py
```

**Multiple nodes -** PsyNeuLink forms the cluster from the SLURM
ranks of one srun step — 
rank 0 scheduler, rank 1 driver (runs your script),
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
