# Dask Backend Plan For PEC MLE And Hierarchical Reuse

## Summary
Add a Dask backend to current PEC MLE first, with Optuna as the primary v1 path using `ask`/`tell`. Design the backend as a reusable execution layer so hierarchical EM can later distribute subject E-steps without changing the user’s cluster setup.

## Public API
- Extend `PECOptimizationFunction`:
  ```python
  pnl.PECOptimizationFunction(
      method=optuna.samplers.TPESampler,
      parallel_backend=None,
      parallel_backend_options=None,
      parallel_scope="auto",
      max_concurrent_evaluations=None,
      evaluations_per_job=1,
      worker_cores=None,
      pec_factory=None,
  )
  ```
- Use **evaluation** to mean one optimizer objective call for one parameter proposal.
- Do not expose Optuna’s “trial” terminology in PEC-facing API.
- Add optional Dask extra: `pip install psyneulink[dask]`.

## MLE Backend Behavior
- For current PEC MLE, `parallel_scope="auto"` resolves to `"evaluations"`.
- Driver owns the Optuna study, asks for parameter proposals, submits likelihood evaluations to Dask, and tells completed scalar values back to Optuna.
- Dask process/SLURM execution requires `pec_factory`, returning a fresh serial PEC configured for the same MLE problem, because constructed PEC objects are not assumed to be safely picklable.
- Worker-local PEC instances are cached to avoid rebuilding the model for every evaluation.
- SciPy differential evolution may also use Dask through its map-like `workers` interface, but Optuna is the v1 priority.

## Serialization And Worker Construction
- Dask Futures serialize submitted task functions, arguments, and results to move work from the driver to workers.
- Do not submit a constructed `ParameterEstimationComposition`, bound PEC methods, or closures that capture a PEC:
  ```python
  client.submit(pec.log_likelihood, *params, inputs=inputs)  # avoid
  client.submit(lambda p: pec.log_likelihood(*p, inputs=inputs), params)  # avoid
  ```
- Dask process/SLURM execution should submit only serializable lightweight payloads:
  - parameter proposal values;
  - immutable input/data references or scattered arrays;
  - a `pec_factory` import string or top-level callable;
  - scalar likelihood results and warning metadata.
- Workers construct a fresh local PEC from `pec_factory`, set it to serial execution with `parallel_backend=None`, configure LLVM/thread settings, and cache it for reuse.
- Worker-local PEC caching is required because rebuilding the model for every objective evaluation would add large overhead.
- The driver remains responsible for Optuna `ask`/`tell`; workers only evaluate objective values.

## Hierarchical Interface
- For `fit_method="hierarchical"`, `parallel_scope="auto"` resolves to `"subjects"`.
- One Dask task should run one subject E-step:
  ```text
  subject data/input slice -> subject MAP optimization -> finite-difference Hessian -> subject posterior summary
  ```
- The subject E-step reuses the same MAP/MLE likelihood machinery, but runs locally inside the worker.
- Do not nest Dask by default:
  ```text
  Dask over subjects -> local Optuna/MAP inside worker -> LLVM threads
  ```
- Add a future hierarchy-specific concurrency name:
  ```python
  max_concurrent_subject_fits
  ```
  rather than reusing `max_concurrent_evaluations`, because each subject fit contains many objective evaluations.

## Core Scheduling
- Prefer LLVM parallelism within each SLURM job/node.
- Default SLURM policy:
  ```text
  evaluations_per_job = 1
  worker_cores = cores_per_job
  slurm_jobs = max_concurrent_evaluations      # MLE/MAP
  slurm_jobs = max_concurrent_subject_fits     # hierarchical EM
  ```
- Multiple evaluations or subject fits per job are opt-in only.
- Worker setup must call PsyNeuLink thread control, e.g. `pnl.set_num_threads(worker_cores)`, before LLVM evaluation.
- Use Dask worker resources to prevent oversubscription.

## Tests And Smoke Scripts
- MLE tests cover serial preservation, local Dask Optuna, LLVM-only errors, missing Dask dependency, `pec_factory` validation, existing-client ownership, and worker-core setup.
- Dask tasks do not require pickling a constructed PEC; process/SLURM mode validates `pec_factory` and uses worker-local PEC construction.
- Hierarchical tests later verify that Dask distributes subject E-steps while inner subject optimization remains local.
- Add `Scripts/Debug/pec_hierarchical/dask_optuna_mle_smoke.py` with local Dask and commented SLURM examples.
- Add hierarchical smoke coverage later showing one subject E-step per Dask task.

## Assumptions
- Dask is optional and never required for normal PEC use.
- V1 Dask supports current data-fitting MLE first.
- Hierarchical EM reuses the backend later at the subject E-step level.
- Constructed PEC objects are not part of the Dask wire protocol; only factory specs, data/input payloads, params, and scalar results are serialized.
- Ray, `mpi4py`, nested Dask parallelism, and Optuna shared-storage studies are out of scope for v1.
