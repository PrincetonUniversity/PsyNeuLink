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

   z_s \sim \mathcal{N}(\beta,\ \Sigma)

where :math:`z_s` is participant :math:`s`'s parameter vector in an unconstrained space,
and :math:`\beta` and :math:`\Sigma` are estimated from the group. :math:`\Sigma` is the
parameter-by-parameter covariance; by default its off-diagonals are held at zero, and the
``covariance`` option below lets the fit estimate them.


.. _Hierarchical_Fitting_Enabling:

Enabling hierarchical fitting
-----------------------------

Pass ``fit_method="hierarchical"``, name the column of ``data`` that identifies
participants, and supply a ``pec_factory``::

    pec = ParameterEstimationComposition(
        data=stacked,
        fit_method="hierarchical",
        hierarchical_options={"subject_id": "subject"},
        distributed_options={"pec_factory": build_subject_pec},
    )

    results = pec.run()
    results.group_parameters

No model is given here. What is fitted, over what ranges, and which outputs are compared
against the data are declared once, by the factory: it builds a participant's model and
this composition holds them all to the first one it builds.


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

* **LLVM execution, and the same parameters and ranges for every participant**
    The group model is defined in terms of those ranges, so ranges that varied between
    participants would mean different things for different people. The first participant's
    model settles what they are, and every other one is held to it, in-process and on a
    worker alike, before it is scored.


.. _Hierarchical_Fitting_Options:

Options
-------

``hierarchical_options`` accepts the following keys. An unrecognised key raises rather than
being ignored.

* ``subject_id`` (required)
    Column of ``data`` identifying participants.

* ``covariance``
    ``"diagonal"`` (the default) or ``"full"``. See :ref:`Hierarchical_Fitting_Covariance`.

* ``sampler``
    ``None`` (the default) fits by EM. ``"nuts"`` samples the posterior instead; see
    :ref:`Hierarchical_Fitting_Sampling`.

* ``sampler_options``
    Passed to `NUTSConfig`, and accepted only when ``sampler`` is set.

* ``max_iterations``
    Most EM iterations to run. Defaults to ``50``.

* ``tol``
    Stop once no group parameter moves by more than this. Defaults to ``1e-4``.

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


.. _Hierarchical_Fitting_Covariance:

Group covariance
----------------

The group covariance :math:`\Sigma` is one parameter-by-parameter matrix. What
``covariance`` chooses is how much of it the fit is allowed to use.

``"diagonal"``, the default, holds the off-diagonals at zero: the group model asserts that
the parameters vary independently across the population. Each participant's curvature is
measured the same way, one parameter at a time with the others held at their mode.

``"full"`` estimates the whole matrix, so the fit can report that participants with a high
threshold tend to have a low drift rate, and each participant's posterior covariance is
measured in full and inverted. This matters for the reported intervals as much as for the
group: where two parameters trade off against each other, the width of one with the other
held fixed is smaller than its width with the other integrated out, so a diagonal fit
reports intervals that are too tight. How much too tight depends on how strongly the
parameters trade off, and a fit cannot tell you that without measuring it.

The cost is the curvature probe. Measuring the diagonal takes :math:`2P` evaluations of a
participant's objective per EM iteration; measuring the whole matrix takes :math:`2P^2`.
For a four-parameter model that is 32 evaluations instead of 8. Where an evaluation means
simulating a model, that is the dominant cost of the fit; where it means calling a trained
network (see :ref:`Neural Likelihoods <NeuralLikelihood>`) it is largely free.

``fit_results.group_correlation`` reports the correlations implied by :math:`\Sigma`. A
diagonal fit reports the identity, which records that it assumed the parameters were
independent -- not that it measured them to be.


.. _Hierarchical_Fitting_Sampling:

Sampling instead of EM
----------------------

EM reports each participant's single best parameter values and summarizes the uncertainty
around them with a Gaussian placed at that peak. That summary is only as good as the
assumption behind it. A posterior that is skewed, or pressed against a bound, or has a
ridge running through it is not a Gaussian, and no amount of fitting makes the reported
interval right.

``sampler="nuts"`` draws from the posterior instead, so intervals come out of the draws
themselves::

    pec = pnl.ParameterEstimationComposition(
        data=data,
        fit_method="hierarchical",
        hierarchical_options={
            "subject_id": "subject",
            "sampler": "nuts",
            "covariance": "full",
            "sampler_options": {"draws": 1000, "warmup": 1000, "chains": 4},
        },
        distributed_options={"pec_factory": build_participant},
    )
    results = pec.run()

Two things follow from how it works, and both are requirements rather than preferences.

**Every participant's model must be scored by a trained estimator** (see
:ref:`Neural Likelihoods <NeuralLikelihood>`). The sampler needs the gradient of the score
with respect to the parameters, which simulating a model does not give; a fit whose
participants are scored by simulation is refused rather than run. It also needs tens of
thousands of evaluations where EM needs hundreds, which is affordable only because an
evaluation is one network call.

**The fit runs in one process.** A single evaluation of the posterior involves every
participant at once, so unlike EM there is no point at which one participant can be fitted
apart from the rest, and ``distributed=True`` is refused. Participants sharing one estimator
object are scored in a single call, so a factory that loads the artifact once and reuses it
is markedly faster than one that loads it again for each participant.

Read ``results.convergence`` and ``results.diagnostics`` before the estimates. ``r_hat``
above about 1.01, or ``ess`` in the low hundreds, means the draws do not yet describe the
posterior; any divergences at all mean the sampler could not follow it somewhere, and the
draws are biased in a direction it cannot report. Raising ``target_accept`` is the usual
response to divergences, and more draws to the rest. Hierarchical posteriors mix more slowly
than the number of parameters suggests, so expect to need more draws than for a fit of the
same size that is not hierarchical.


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

``run()`` returns a `HierarchicalPECResults`, or a `HierarchicalSamplingResults` when a
sampler was used, also available afterwards as ``pec.fit_results``. The two report the same
group and participant estimates; a sampled fit reports intervals taken from the draws in
place of a Gaussian width, and adds the draws and the convergence diagnostics described in
:ref:`Hierarchical_Fitting_Sampling`.

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

``group_covariance`` is the full :math:`P \times P` matrix and ``group_correlation`` the
correlations implied by it; ``posterior_covariance`` is one such matrix per participant.
``sigma`` and ``posterior_variance`` are their diagonals, which is what a diagonal fit
estimated in the first place.

Convergence is judged by how far the group estimate moves, not by the objective, which is
not monotone under an approximate E-step.


.. _Hierarchical_Fitting_Limitations:

Limitations
-----------

* With ``covariance="diagonal"``, the default, a tendency for two parameters to move together
  is not represented, and reported intervals err towards being too tight; see
  :ref:`Hierarchical_Fitting_Covariance`.
* Participant estimates are posterior modes with a Gaussian approximation around them, not
  posterior means.
* Interval width tracks the quality of the likelihood. A likelihood estimated from too few
  simulations gives intervals that are too narrow, and no amount of fitting corrects that.
* A parameter the data barely constrain is shrunk toward the group mean. The point estimate
  alone does not distinguish that from a well-estimated parameter; ``subject_posteriors``
  reports the spread that does.
* Sampling requires a trained estimator for every participant, and runs in one process; see
  :ref:`Hierarchical_Fitting_Sampling`.
* ``depends_on`` is not supported together with hierarchical fitting.
* The group model is an intercept only; group-level predictors are not yet available.


.. _Hierarchical_Fitting_Requirements:

Requirements
------------

Fitting in one process needs nothing beyond PsyNeuLink itself. ``distributed=True``
requires the same extra as distributed maximum-likelihood fitting, installed with
``pip install "psyneulink[dask]"``.
