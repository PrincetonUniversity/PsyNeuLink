.. _DistributedFitting:

Distributed Fitting
===================

`ParameterEstimationComposition` fits can evaluate candidate parameterizations in
parallel across a `Dask <https://www.dask.org>`_ cluster instead of serially. The
optimizer (an optuna sampler or ``differential_evolution``) runs on a driver, while
each candidate's log-likelihood is scored on a worker. Distributed fitting requires
the ``psyneulink[dask]`` extra (see `Distributed_Fitting_Requirements`), LLVM
execution, and a single scalar objective.


.. _Distributed_Fitting_Enabling:

Enabling distributed fitting
----------------------------

Pass ``distributed=True`` and a ``pec_factory`` to the `PECOptimizationFunction`, or
to the `ParameterEstimationComposition`, which forwards them::

    optimizer = PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0, popsize=8),
        max_iterations=480,
        distributed=True,
        distributed_options={"pec_factory": pec_factory},
    )

``pec_factory(data) -> (pec, inputs)`` is a top-level, picklable callable that
rebuilds a serial `ParameterEstimationComposition` and its inputs. Each worker
caches its own PEC and reuses the compiled LLVM binary across evaluations.
PsyNeuLink must be importable in the worker environment.

``distributed_options`` accepts the following keys:

* ``pec_factory`` (required)
    Worker recipe ``(data) -> (pec, inputs)``. Required for distributed fitting.

* ``worker_cores``
    LLVM threads per worker. Defaults to ``$SLURM_CPUS_PER_TASK``, otherwise to the
    available cores divided by the worker count.

* ``max_concurrent_evaluations``
    Candidates dispatched per ask/tell round. Defaults to the live worker count.

* ``client``
    Existing Dask ``Client`` to use instead of creating or resolving a cluster.

* ``n_workers``
    Number of workers for the automatically created single-node ``LocalCluster``.
    Ignored when ``client`` is supplied or the SLURM launcher is used.

The *live worker count* is the number of workers registered with the scheduler when
the fit starts: ``srun`` tasks minus two with the launcher (see
`Distributed_Fitting_Running`), or the ``LocalCluster`` size on a single node.

Any optuna sampler or study and ``differential_evolution`` are supported.


.. _Distributed_Fitting_Running:

Running
-------

To run across multiple nodes, a single SLURM ``srun`` step launches the cluster
through the ``psyneulink.dask_run`` module, which builds the scheduler, driver, and
workers from the SLURM tasks.

.. code-block:: bash

    srun -n <ntasks> python -m psyneulink.dask_run study.py

Two of the tasks are reserved for the scheduler and the driver, so the number of
workers is ``ntasks - 2``. Request the tasks the usual way in an SBATCH script (with
``--nodes`` and ``--ntasks-per-node``); the workers are spread across whatever nodes
the allocation spans, and each uses ``$SLURM_CPUS_PER_TASK`` LLVM threads. For
example, ``--nodes=2 --ntasks-per-node=5`` requests 10 tasks and so runs 8 workers.

:download:`submit_dask.slurm <../../Scripts/Debug/pec_dask/submit_dask.slurm>` is a
complete batch template, and
:download:`study.py <../../Scripts/Debug/pec_dask/study.py>` is the DDM study it
runs. A larger Stability-Flexibility example is provided as
:download:`stability_flexibility_dask.py <../../Scripts/Debug/stability_flexibility/stability_flexibility_dask.py>`
and :download:`submit_stabflex_dask.slurm <../../Scripts/Debug/stability_flexibility/submit_stabflex_dask.slurm>`;
its ``pec_factory`` imports ``make_stab_flex`` from the co-located
``stability_flexibility.py``, so the batch script adds that directory to
``PYTHONPATH``.

To try a fit on a single machine, run the study script directly. It forms a
one-node cluster automatically, with no launcher; ``n_workers`` in
``distributed_options`` sets how many workers it spawns.

.. code-block:: bash

    python study.py


Using an existing cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already run your own Dask cluster -- for example in a notebook or outside
SLURM -- pass it as ``client`` and PsyNeuLink uses it instead of creating one::

    from dask.distributed import Client
    client = Client("tcp://scheduler-address:8786")
    distributed_options = {"client": client, "pec_factory": pec_factory}

This is optional; the launcher above covers the usual SLURM case. PsyNeuLink does
not shut down a client you supply.


.. _Distributed_Fitting_Reproducibility:

Reproducibility
---------------

With common random numbers -- ``same_seed_for_all_parameter_combinations=True`` and
a fixed ``initial_seed`` -- every candidate is scored against identical simulation
noise, so a distributed fit with a tell-order-independent sampler (such as
``RandomSampler``) matches the serial fit. Stateful samplers (CMA-ES, QMC) explore
different points under batched ask/tell and are not bit-identical to the serial
driver. Without common random numbers the fit is still valid but not reproducible,
and a warning is issued.


.. _Distributed_Fitting_Requirements:

Requirements
------------

Install the extra with ``pip install "psyneulink[dask]"``, which adds ``dask``,
``distributed``, and ``dask-jobqueue``. With ``distributed=False``, the default,
Dask is never imported.
