.. _HierarchicalFitting:

Hierarchical Fitting
====================

`ParameterEstimationComposition` normally fits one participant's data at a time, which
treats each participant as unrelated to the others, so every estimate is only as good as
that participant's own trial count allows.

Hierarchical fitting instead fits the group jointly: participants are drawn from a
population, and each participant's estimate is informed by the rest of the group.
Estimation is by empirical-Bayes Laplace EM, with the group modelled as

.. math::

   z_s \sim \mathcal{N}(\beta,\ \mathrm{diag}(\sigma))

where :math:`z_s` is participant :math:`s`'s parameter vector in an unconstrained space,
and :math:`\beta` and :math:`\sigma` are estimated from the group.


.. _Hierarchical_Fitting_Enabling:

Enabling hierarchical fitting
-----------------------------

Pass ``fit_method="hierarchical"``, name the column of ``data`` that identifies
participants, and supply a ``pec_factory``::

    pec = ParameterEstimationComposition(
        nodes=[model],
        parameters={("rate", decision): np.linspace(-1.5, 1.5, 1000)},
        outcome_variables=[decision.output_ports[DECISION_OUTCOME],
                           decision.output_ports[RESPONSE_TIME]],
        data=stacked,
        optimization_function=PECOptimizationFunction(method="differential_evolution",
                                                      max_iterations=1),
        fit_method="hierarchical",
        hierarchical_options={"subject_id": "subject"},
        distributed_options={"pec_factory": build_subject_pec},
    )

    results = pec.run()
    results.group_parameters

The model passed to the constructor declares which parameters are fitted, over what
ranges, and which outputs are compared against the data. It is not itself simulated.


.. _Hierarchical_Fitting_Data:

Data
----

``data`` is one table with every participant's trials stacked, plus a column naming who
produced each row::

    subject   decision   response_time
    S01       1          0.512
    S01       0          0.734
    S02       1          0.488
    ...

Apart from that column, the table holds the outcome variables in the order given by
``outcome_variables``, exactly as for a single-participant fit.

Participants may have different trial counts. They are ordered by first appearance rather
than sorted, and that order is used for every per-participant array and frame in the
results, so ``results.subject_labels[i]`` always identifies row ``i``.

Two or more participants are required, and every trial must name one.

``likelihood_include_mask`` is not accepted; drop the rows you want excluded from ``data``
instead.


.. _Hierarchical_Fitting_Factory:

The participant factory
-----------------------

``pec_factory(data, subject_index=None) -> (pec, inputs)`` is a top-level, picklable
callable that builds one participant's model from their rows::

    def build_subject_pec(data, subject_index=None):
        comp, decision = build_model()
        pec = ParameterEstimationComposition(
            nodes=[comp],
            parameters={("rate", decision): np.linspace(-1.5, 1.5, 1000)},
            outcome_variables=[decision.output_ports[DECISION_OUTCOME],
                               decision.output_ports[RESPONSE_TIME]],
            data=data,
            optimization_function=PECOptimizationFunction(
                method="differential_evolution", max_iterations=1),
            num_estimates=300,
            initial_seed=100 + subject_index,
            same_seed_for_all_parameter_combinations=True,
        )
        pec.controller.parameters.comp_execution_mode.set("LLVM")
        return pec, {comp: trial_inputs(len(data))}

A `Composition` cannot be copied, so each participant's model is built rather than cloned.
The factory lives in ``distributed_options``, the same key distributed maximum-likelihood
fitting uses (see :ref:`DistributedFitting`).

Requirements on what it returns:

* **Common random numbers**
    ``same_seed_for_all_parameter_combinations=True`` with a fixed ``initial_seed``.
    Posterior curvature comes from finite differences, which measure simulation noise
    rather than curvature if the likelihood is not deterministic in its parameters.

* **A distinct seed per participant**
    Use ``subject_index``. A shared seed gives every participant the same stream of
    simulation noise, which is absorbed into the group variance rather than averaging out.

* **LLVM execution, and the same parameters and ranges as the constructor's model**
    The group model is defined in terms of those ranges, so ranges that varied between
    participants would mean different things for different people. This is checked for
    every participant, in-process and on a worker alike, before it is scored.


.. _Hierarchical_Fitting_Options:

Options
-------

``hierarchical_options`` accepts the following keys. An unrecognised key raises rather than
being ignored.

* ``subject_id`` (required)
    Column of ``data`` identifying participants.

* ``max_iterations``
    Most EM iterations to run. Defaults to ``50``.

* ``tol``
    Stop once no group parameter moves by more than this. Defaults to ``1e-4``.

* ``damping``
    Fraction of the previous group estimate retained each iteration. Slows the fit but
    steadies it when participant posteriors are noisy. Defaults to ``0.0``.

* ``variance_floor``
    Smallest posterior variance to report. Defaults to ``1e-6``.

* ``hessian_step``
    Finite-difference step for posterior curvature, in unconstrained units. Derived per
    parameter from the group variance when omitted.

* ``estep_method``
    Any method accepted by `scipy.optimize.minimize`. Defaults to ``"Nelder-Mead"``,
    which is derivative-free, since a simulated likelihood has no gradient.

* ``estep_options``
    Passed through to `scipy.optimize.minimize`.


.. _Hierarchical_Fitting_Running:

Running
-------

By default every participant is fitted in the calling process. A participant's model is
constructed and compiled before it can be scored, so this is appropriate for small groups.

Setting ``distributed=True`` fits participants across a Dask cluster, one per task, with
the group update still performed by the caller. The cluster is resolved exactly as for
distributed maximum-likelihood fitting (see :ref:`Distributed_Fitting_Running`): an
explicit ``client``, a cluster formed by ``python -m psyneulink.dask_run``, or a
single-node ``LocalCluster`` created on demand. Each worker caches the models it builds and
participants are pinned to the worker holding theirs, so a model is built once rather than
once per iteration.

Results are collected by participant index rather than in completion order, so a
distributed fit and an in-process one agree exactly.

:download:`hierarchical_fitting.py <../../Scripts/Debug/pec_hierarchical/hierarchical_fitting.py>`
is a complete example,
:download:`make_example_data.py <../../Scripts/Debug/pec_hierarchical/make_example_data.py>`
writes a synthetic table for it to fit, and
:download:`submit_hierarchical.slurm <../../Scripts/Debug/pec_hierarchical/submit_hierarchical.slurm>`
is a multi-node batch template.


.. _Hierarchical_Fitting_Results:

Results
-------

``run()`` returns a `HierarchicalPECResults`, also available afterwards as
``pec.fit_results``.

``group_parameters`` has one row per parameter: ``mean_z`` and ``sd_z`` are the group
estimate and spread in the unconstrained space, and ``value`` is that mean mapped into the
model's units. Because the transform is monotone, ``value`` is the **median** of the
implied distribution of the parameter, not the mean of ``subject_parameters``. Spread is
reported only as ``sd_z``: a single standard deviation in the model's units would
misrepresent an interval the transform makes asymmetric near a bound.

``subject_parameters`` gives one row per participant in the model's units, and
``subject_posteriors`` one row per participant and parameter, with uncertainty in both
spaces and whether that participant's fit converged. ``em_history`` records each iteration
alongside the group estimate that produced it.

Convergence is judged by how far the group estimate moves, not by the objective, which is
not monotone under an approximate E-step.


.. _Hierarchical_Fitting_Limitations:

Limitations
-----------

* Group covariance is diagonal: each parameter's spread across the population is estimated on
  its own, so a tendency for two of them to move together -- participants with a high drift rate
  also tending to have a high threshold -- is not represented.
* Participant uncertainty is diagonal too, and is the spread of one parameter with the
  others held at the mode rather than integrated out. Where two parameters trade off
  against each other, that is the narrower of the two quantities, so reported intervals err
  towards being too tight.
* Participant estimates are posterior modes with a Gaussian approximation around them, not
  posterior means.
* Interval width tracks the quality of the likelihood. A likelihood estimated from too few
  simulations gives intervals that are too narrow, and no amount of fitting corrects that.
* A parameter the data barely constrain is shrunk toward the group mean. The point estimate
  alone does not distinguish that from a well-estimated parameter; ``subject_posteriors``
  reports the spread that does.
* ``depends_on`` is not supported together with hierarchical fitting.
* The group model is an intercept only; group-level predictors are not yet available.


.. _Hierarchical_Fitting_Requirements:

Requirements
------------

Fitting in one process needs nothing beyond PsyNeuLink itself. ``distributed=True``
requires the same extra as distributed maximum-likelihood fitting, installed with
``pip install "psyneulink[dask]"``.
