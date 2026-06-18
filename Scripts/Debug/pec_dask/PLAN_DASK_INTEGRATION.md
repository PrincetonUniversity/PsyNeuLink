# Dask Integration into Core PEC — Design Plan

Design-of-record for folding the benchmark-validated Dask MLE backend into core
PsyNeuLink. Sibling to `PLAN.md` (the hierarchical/EM plan). This documents the
agreed design **before** any `psyneulink/` code is written.

## Context

The evaluation-level Dask MLE backend is built and benchmark-validated under
`Scripts/Debug/pec_dask/` (the `benchmark/bench_core.py` ask/tell driver,
the `pec_dask_mle/` prototype, dask-srun verified multinode, per-sampler sweep
done). It currently lives outside the package and drives an unmodified
`pec.log_likelihood`. The goal is to fold it into core PsyNeuLink so a user
enables distributed PEC fitting with a single constructor flag (`distributed=True`)
and **does no Dask administration** — no spinning up a scheduler, no handling IPs.
Outcome: candidate evaluations distribute across a Dask cluster that PNL forms
from the SLURM allocation, with results matching serial when common random
numbers are used.

## Dask topology (orientation)

A cluster is three roles: a **scheduler** (one lightweight coordinator; assigns
tasks, brokers data, runs no computation), **workers** (processes that run the PEC
evaluations), and a **client/driver** (the user's script). PNL forms and wires all
three so the user never launches a scheduler or sees an address.

## Locked decisions

- **User-facing API = `distributed: bool` + `distributed_options: Mapping`** (Dask is the hidden engine).
- **PNL owns cluster setup.** Primary multi-node path is a **launcher**:
  `srun -n N python -m psyneulink.dask_run study.py`. The launcher splits the
  SLURM ranks into scheduler/driver/workers, runs `study.py` only on the driver
  rank, and `distributed=True` auto-detects the launcher-formed cluster. The
  user's `study.py` is a normal serial-looking script + `distributed=True`. No IP,
  no gate line, no manual scheduler.
- **Worker reconstruction = a `pec_factory` callable.** A live PEC is never shipped
  (Compositions aren't safely picklable); each worker rebuilds one PEC locally from
  the factory and caches it. Dask ships the *factory function* (and the task) by
  value via its own cloudpickle, so `build_pec` can live in `study.py`; workers
  only need PNL importable in the same env. We never pickle a PEC.
- **Optimizers = optuna samplers AND `differential_evolution`**, both distributed.
- **Seeding = allow either; warn without CRN.** With CRN
  (`same_seed_for_all_parameter_combinations=True`) distributed == serial
  bit-for-bit; without it the fit is still valid but not reproducible/serial-
  matching, and we warn.
- **`pnllvm.cleanup()` at `log_likelihood` is removed outright** (no flag) — the
  serial optimization loop never calls it (it goes through `evaluate_agent_rep`,
  not `log_likelihood`), so removal only affects the diagnostic `log_likelihood`
  and the Dask worker, both of which want the compiled binary reused.
- **Driver/worker code folds into `fitfunctions.py`**; the launcher is its own tiny
  runnable module (it must be `python -m`-invokable).
- LLVM execution mode only (already enforced); single scalar objective
  (MLE/likelihood or a float-returning `objective_function`; vector/multi-objective
  NSGA-II excluded).

## API

On `ParameterEstimationComposition.__init__` (forwarded to `PECOptimizationFunction`)
and equivalently on `PECOptimizationFunction.__init__`:

- `distributed: bool = False`
- `distributed_options: Optional[Mapping] = None` — keys (all optional except the factory for multi-node):
  - `"pec_factory": callable` (required for multi-node) — returns a fresh `(pec, inputs)`.
  - `"worker_cores": int` — LLVM threads per worker; **defaults to `$SLURM_CPUS_PER_TASK`** (else available cores) so it mirrors `--cpus-per-task` without retyping.
  - `"max_concurrent_evaluations": int` — batch size (candidates per ask/tell round); **defaults to the live worker count**, or the sampler's `popsize`/DE population for generational samplers.
  - optional explicit connection (advanced escape hatch): `{"client": <Client>}` | `{"scheduler_address": …}` | `{"scheduler_file": …}` | `{"cluster": "local", "n_workers": N}`.

Plain attributes (like `self.method`/`self.direction`), all default off → existing code untouched.

**Sizing — workers vs cores vs batch (one source of truth each):**
- **Worker count** is the SLURM allocation, NOT a Python value: `srun -n N`
  (= `--ntasks=N`) launches N ranks → rank 0 scheduler, rank 1 driver, ranks
  2…N−1 workers, so **workers = N − 2**. (`n_workers` in Python applies *only* to
  the single-node `LocalCluster` path.)
- **`worker_cores`** mirrors `--cpus-per-task` (auto-detected).
- **`max_concurrent_evaluations`** is the batch size, a *distinct* concept that
  auto-defaults to the live worker count.
- Common case: the user sets none of these — they derive from the allocation + sampler.

**Client resolution order** (in the folded helper `_dask_client(...)`): explicit
`client` → explicit `scheduler_address`/`scheduler_file` → **active launcher
client** (module global set by `psyneulink.dask_run`) → else a `LocalCluster` on
the current node (the zero-config single-node default).

## Launch models

1. **Launcher (primary, multi-node).** `srun -n N python -m psyneulink.dask_run study.py`
   (N = workers + 2: rank 0 scheduler, rank 1 driver, rest workers). PNL forms
   scheduler+workers from the ranks, runs `study.py` only on the driver, manages
   the scheduler-file in a temp dir. `study.py` = normal PNL + `distributed=True`.
   Zero Dask admin; worker count = the allocation, `worker_cores` auto from
   `--cpus-per-task`.
2. **`LocalCluster` (single node / CI).** Plain `python study.py` with
   `distributed=True` and no connection info → PNL starts scheduler + N worker
   subprocesses on the current node, runs, tears down. Never spans nodes.
3. **Connect to an existing cluster (advanced escape hatch).** For users already
   running a persistent Dask cluster: `distributed_options={"client": …}` or
   `{"scheduler_address"/"scheduler_file": …}`. Not the documented default.

## Architecture

Two nested parallelism levels: **candidates across workers**, **`num_estimates`
across `worker_cores` LLVM threads inside each worker**. The driver owns the
study; each worker rebuilds one PEC via `pec_factory`, caches it, returns
`float(pec.log_likelihood(*params, inputs=...))`. Directly lifts `bench_core.py`
(`run_fit`, `evaluate_loglik`) and `pec_dask_mle/worker.py` — the *code*, not the
benchmark SLURM scripts (no `--exclusive`/node pinning carried over; shipped
templates are plain).

## Changes by file

1. **`psyneulink/core/components/functions/nonstateful/fitfunctions.py`** (driver + folded worker/helper code)
   - `PECOptimizationFunction.__init__` (~`:310`): add `distributed`/`distributed_options`; resolve factory/worker_cores/batch.
   - `_fit` (~`:549`): if `distributed`, build the client once via `_dask_client(...)` and dispatch to the dask variant; else serial path byte-for-byte unchanged.
   - `_fit_optuna` (~`:773`): add a `distributed` branch mirroring `bench_core.run_fit` — `create_study(sampler=opt_func, direction=self.direction)`, loop `max_iterations // batch` rounds of `ask` → `client.submit(_dask_evaluate_loglik, …)` → `gather` → `tell`. Reuse `fit_param_bounds`/`fit_param_names`; return the **same** `{"fitted_params","optimal_value"}` dict so `_function`/`PEC.run` are unchanged.
   - `_fit_differential_evolution` (~`:751`): add a `distributed` branch calling `differential_evolution(obj, bounds, popsize=15, polish=False, updating="deferred", workers=_dask_map, seed=…)`, where `obj` is a **top-level factory objective** (`functools.partial(_dask_evaluate_loglik_de, pec_factory, worker_cores, data_f, inputs_f, direction)`) — never the PEC-capturing closure. `updating="deferred"` is mandatory with `workers`.
   - New module-level helpers (importable-by-reference): `_dask_evaluate_loglik(...)` (per-worker cache, `set_num_threads`, factory rebuild, scalar; from `pec_dask_mle/worker.py`), `_dask_evaluate_loglik_de(...)` (sign-wrapped for scipy minimize), `_dask_client(...)` (resolution order above; LocalCluster + active-launcher-client global), `_dask_map(...)` for DE. Data/inputs sent once via `client.scatter(..., broadcast=True, hash=False)` (`hash=False` avoids "lost dependencies" cancellations on a second fit of the same data in one process).

2. **`psyneulink/dask_run.py`** (new, tiny launcher; `python -m psyneulink.dask_run study.py`)
   - Uses `dask_jobqueue.slurm.SLURMRunner` (from the prototype's `bench.py maybe_start_runner`) to assign rank roles by `SLURM_PROCID` (0 scheduler, 1 driver, 2+ workers); non-driver ranks serve until shutdown.
   - On the driver rank: create the `Client`, stash it as the module-global "active dask client" PNL reads, then `runpy.run_path(study_py, run_name="__main__")`; on exit, `client.shutdown()` so all ranks end. Lazy dask import.

3. **`psyneulink/core/compositions/parameterestimationcomposition.py`**
   - `__init__` (~`:465`): accept `distributed`/`distributed_options`, forward to the optimizer.
   - `log_likelihood` (~`:1020`): **delete** the `pnllvm.cleanup()` call (serial fitting unaffected; diagnostic + workers now reuse the binary). Leave the once-per-`run()` cleanup at `:909`.
   - When `distributed` and CRN is off: emit a clear warning (valid but not reproducible/serial-identical).

4. **`setup.py` (~`:82`) + new `dask_requirements.txt`**: add `'dask': get_requirements('dask')` pinning `dask`, `distributed`, `dask-jobqueue` (cloudpickle is transitive via `distributed`, not pinned). All dask imports lazy; `distributed=False` never imports dask; missing dask → `ImportError("install psyneulink[dask]")`.

## Seeding / reproducibility

Seeds come from the OCM's `gen_new_seed_sequence()` advancing a per-OCM
`_seed_counter`. Each worker has its own PEC/counter, so without CRN the seeds a
candidate gets depend on worker placement (not reproducible, possible cross-worker
collisions). With CRN every candidate uses identical seeds → distributed ==
serial. Both are supported and we **warn** when CRN is off. (Possible later
enhancement: driver-assigned per-candidate seeds by global index → serial-matching
without CRN.)

## Scope & limits

optuna samplers + `differential_evolution`; LLVM-only; single-objective;
`pec_factory` required for multi-node; CRN optional (warn); launcher + LocalCluster
+ existing-cluster escape hatch; clear errors for dask missing, non-LLVM mode,
multi-objective sampler, factory missing on a multi-node cluster.

## Verification

- **Parity (headline):** small DDM, tiny `num_trials`, fixed `initial_seed`, CRN
  on. Serial vs a 2-worker **process** `LocalCluster` (`threads_per_worker=1` —
  PNL's LLVM asserts on multi-thread in-process cleanup), identical seeded sampler
  + `max_iterations`. Assert identical `optimized_parameter_values`/`optimal_value`
  and per-candidate `log_likelihood` `rtol=1e-10`. Run for **both** an optuna
  sampler and `differential_evolution`.
- **No-CRN behavior:** distributed run with CRN off completes and emits the warning.
- **Driver bookkeeping (FakeClient, no dask/PNL):** exactly `n_rounds*batch` trials
  asked/told in order, candidates within bounds, vectors in `fit_param_names`
  order, `worker_cores` passed, generational sampler rejects `batch<2`. Port
  `tests/test_driver_logic.py`.
- **Worker cache reuse:** `client.run(...)` shows the per-worker cache; factory
  called once/worker. Port `tests/test_distributed.py`.
- **Launcher:** a 2-rank `srun` smoke (or a mocked SLURMRunner unit test) confirms
  the driver runs the script once, `distributed=True` finds the active client, and
  teardown exits all ranks.
- **Guard tests:** dask missing → `ImportError`; non-LLVM → error; factory missing
  (multi-node) → error.
- **Scale:** point a thin variant of `benchmark/bench.py` at the new
  `distributed=True` API; confirm parity with the prototype + recovery-vs-truth.
- **Templates:** ship a plain SBATCH (no pinning) using
  `srun python -m psyneulink.dask_run study.py`.

New core test module: `tests/composition/test_pec_dask.py`, `pytest.importorskip("dask.distributed")`.

## Riskiest assumptions

1. Removing the `:1020` cleanup is safe — confirmed: serial fitting never calls it; the prototype proved bit-identical reuse.
2. Launcher correctness: `SLURMRunner` rank-splitting + `runpy` driver + active-client global + teardown must exit cleanly on all ranks (the prototype's `maybe_start_runner` is the validated basis).
3. DE: the objective handed to scipy must be a top-level picklable factory function (no PEC closure); `updating="deferred"` mandatory with `workers`.
4. `scatter(hash=False)` required to avoid intermittent "lost dependencies" on repeat fits.
5. PNL must be installed on workers; the `pec_factory`'s body may only reference importable things (the model build), not unpicklable captured state.

## Critical files

- `psyneulink/core/components/functions/nonstateful/fitfunctions.py` — knobs, `_fit` dispatch, `_fit_optuna`/`_fit_differential_evolution` dask branches, folded worker task + `_dask_client`/`_dask_map`
- `psyneulink/dask_run.py` (new) — `python -m psyneulink.dask_run` launcher
- `psyneulink/core/compositions/parameterestimationcomposition.py` — forward knobs, delete `:1020` cleanup, no-CRN warning
- `setup.py` + `dask_requirements.txt` — `dask` extra
- Reference (reuse, don't re-derive): `Scripts/Debug/pec_dask/benchmark/bench_core.py`, `pec_dask_mle/worker.py`, `bench.py` (`run_dask_local`, `maybe_start_runner`), `tests/test_driver_logic.py`/`test_distributed.py`
