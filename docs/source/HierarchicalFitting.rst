.. _HierarchicalFitting:

Hierarchical Fitting
====================

`ParameterEstimationComposition` normally fits one participant's data at a time.  Fitting a group
that way treats each participant as unrelated to the others, so every estimate is only as good as
that participant's own trial count allows.

Hierarchical fitting instead fits the group jointly: participants are drawn from a population, and
each participant's estimate is informed by the rest of the group.  Estimation is by empirical-Bayes
Laplace EM, with the group modelled as

.. math::

   z_s \sim \mathcal{N}(\beta,\ \mathrm{diag}(\sigma))

where :math:`z_s` is participant :math:`s`'s parameter vector in an unconstrained space, and
:math:`\beta` and :math:`\sigma` are estimated from the group.


.. _Hierarchical_Fitting_Enabling:

Enabling hierarchical fitting
-----------------------------

Pass ``fit_method="hierarchical"``, name the column of ``data`` that identifies participants, and
supply a ``pec_factory``::

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

The model passed to the constructor declares which parameters are fitted, over what ranges, and
which outputs are compared against the data.  It is not itself simulated; each participant's model
comes from the factory.


.. _Hierarchical_Fitting_Data:

Data
----

``data`` is one table with every participant's trials stacked, plus a column saying who produced
each row::

    subject   decision   response_time
    S01       1          0.512
    S01       0          0.734
    S02       1          0.488
    ...

The column named by ``subject_id`` is read at construction and then removed; the remaining columns
are the outcome variables, in the order given by ``outcome_variables``, exactly as for a
single-participant fit.  The original table stays available as ``pec.hierarchical_data``.

Participants may have different trial counts.  They are ordered by first appearance rather than
sorted, and that order is used for every per-participant array and frame in the results, so
``results.subject_labels[i]`` always identifies row ``i``.

At least two participants are required: with one, the group variance is not identified.


.. _Hierarchical_Fitting_Factory:

The participant factory
-----------------------

``pec_factory`` builds one participant's model::

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

It is required, and takes this form, because a `Composition` cannot be copied: there is no
``Composition.copy()`` and no ``__deepcopy__``, so each participant's model has to be built rather
than cloned.  It lives in ``distributed_options`` because distributed maximum-likelihood fitting
reads the same key, so there is one place for it whichever kind of fit is being run.

The factory receives that participant's rows and their index, and returns the model together with
the inputs to run it with.  It must be defined at module level so that it can be sent to a worker
process.

Requirements on what it returns:

* **Common random numbers.**  ``same_seed_for_all_parameter_combinations=True`` with a fixed
  ``initial_seed``.  Posterior curvature comes from finite differences, which measure simulation
  noise rather than curvature if the likelihood is not deterministic in its parameters.
* **A distinct seed per participant.**  Use ``subject_index``.  A shared seed gives every
  participant the same stream of simulation noise, and that common component is absorbed into the
  group variance rather than averaging out.
* **LLVM execution**, and the same parameters over the same ranges for every participant.  The
  group model is defined in terms of those ranges, so ranges that varied between participants would
  mean different things for different people.  This is checked.


.. _Hierarchical_Fitting_Options:

Options
-------

``hierarchical_options`` accepts:

.. list-table::
   :widths: 22 16 62
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``subject_id``
     - *required*
     - Column of ``data`` identifying participants.
   * - ``max_iterations``
     - ``50``
     - Most EM iterations to run.
   * - ``tol``
     - ``1e-4``
     - Stop once no group parameter moves by more than this.
   * - ``damping``
     - ``0.0``
     - Fraction of the previous group estimate retained each iteration.  Slows the fit but steadies
       it when participant posteriors are noisy.
   * - ``variance_floor``
     - ``1e-6``
     - Smallest posterior variance to report.
   * - ``hessian_step``
     - ``None``
     - Finite-difference step for posterior curvature, in unconstrained units.  Derived per
       parameter from the group variance when omitted.
   * - ``estep_method``
     - ``"Nelder-Mead"``
     - Any method accepted by `scipy.optimize.minimize`.  Derivative-free by default, since a
       simulated likelihood has no gradient.
   * - ``estep_options``
     - ``None``
     - Passed through to `scipy.optimize.minimize`.

An unrecognised key raises rather than being ignored.


.. _Hierarchical_Fitting_Running:

Running
-------

By default every participant is fitted in the calling process.  A participant's model is constructed
and compiled before it can be scored, so this is appropriate for small groups.

Setting ``distributed=True`` fits participants across a Dask cluster, one per task, with the group
update still performed by the caller.  The cluster is resolved exactly as for distributed
maximum-likelihood fitting (see :ref:`DistributedFitting`): an explicit ``client``, a cluster formed
by ``python -m psyneulink.dask_run``, or a single-node ``LocalCluster`` created on demand.  Each
worker caches the models it builds and participants are pinned to the worker holding theirs, so a
model is built once rather than once per iteration.

Distributed and in-process fits are otherwise identical: results are collected by participant index
rather than in completion order, so the two agree exactly.


.. _Hierarchical_Fitting_Results:

Results
-------

``run()`` returns a `HierarchicalPECResults`, also available afterwards as ``pec.fit_results``.

``group_parameters`` has one row per parameter.  ``mean_z`` and ``sd_z`` are the group estimate and
spread in the unconstrained space; ``value`` is that mean mapped into the model's units.

Three group-level quantities are easy to confuse.  ``value`` is the group mean mapped through a
monotone transform, which makes it the **median** of the implied distribution of the parameter.
``subject_parameters.mean()`` is the mean of the participants' own estimates.  ``mean_z`` is the
estimate itself.  Spread is reported only as ``sd_z``, because a single standard deviation in the
model's units would misrepresent an interval that the transform makes asymmetric near a bound.

``subject_parameters`` gives one row per participant in the model's units, and
``subject_posteriors`` one row per participant and parameter, with uncertainty in both spaces and
whether that participant's fit converged.  ``em_history`` records each iteration alongside the group
estimate that produced it.

Convergence is judged by how far the group estimate moves, not by the objective, which is not
monotone under an approximate E-step.


.. _Hierarchical_Fitting_Limitations:

Limitations
-----------

* Group covariance is diagonal: parameters are modelled as varying independently across
  participants.
* Participant estimates are posterior modes with a Gaussian approximation around them, not
  posterior means.
* Interval width tracks the quality of the likelihood.  A likelihood estimated from too few
  simulations gives intervals that are too narrow, and no amount of fitting corrects that.
* A parameter the data barely constrain will be shrunk toward the group mean.  That is the model
  behaving correctly, and is not distinguishable from a well-estimated parameter by looking at the
  point estimate alone; ``subject_posteriors`` reports the spread that says which it is.
* ``depends_on`` is not supported together with hierarchical fitting.
* Group-level predictors are not yet available: the group model is an intercept only.


.. _Hierarchical_Fitting_Requirements:

Requirements
------------

Fitting in one process needs nothing beyond PsyNeuLink itself.  ``distributed=True`` requires the
``dask`` extra, as for distributed maximum-likelihood fitting::

    pip install "psyneulink[dask]"
