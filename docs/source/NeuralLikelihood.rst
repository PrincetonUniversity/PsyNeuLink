.. _NeuralLikelihood:

Neural Likelihoods
==================

In `data fitting <ParameterEstimationComposition_Data_Fitting>`, a `ParameterEstimationComposition`
computes the likelihood of its **data** under each set of parameter values it considers by simulating
the model ``num_estimates`` times and estimating a density from the simulated outcomes. A neural
likelihood is a density estimator that is trained beforehand, on data simulated from the model, and then
used in place of those simulations. Each evaluation of the likelihood then requires a single evaluation
of the estimator, and the likelihood varies smoothly with the parameter values.

An estimator is trained once for a given model and range of parameter values, and can be reused for
any fit of that model within that range, including a :ref:`hierarchical fit <HierarchicalFitting>`.


.. _Neural_Likelihood_Training:

Training an Estimator
---------------------

`train_neural_likelihood` simulates the model at parameter values drawn from within the ranges specified
in **bounds**, and trains an estimator on the results. The model can be specified directly, together with
the inputs used to run it::

    likelihood = pnl.train_neural_likelihood(
        bounds={"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)},
        outcome_names=("decision", "response_time"),
        pec=pec,
        inputs={comp: trial_inputs},
        n_parameter_samples=20000,
    )
    likelihood.save("ddm_nle.pt")

The keys of **bounds** must name the fitted parameters in the order in which the model lists them, and
**outcome_names** must name the outcome variables in the order of its ``outcome_variables``. The number of
trials simulated for each set of parameter values is the number of trials in **inputs**.

To distribute the simulations over a `Dask <https://www.dask.org>`_ cluster, specify a **pec_factory**
in place of **pec**, together with **distributed_options** (see :ref:`Distributed Fitting
<DistributedFitting>`)::

    likelihood = pnl.train_neural_likelihood(
        bounds={"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)},
        outcome_names=("decision", "response_time"),
        pec_factory=build_pec,
        n_parameter_samples=20000,
        n_trials_per_sample=100,
        distributed_options={"n_workers": 8},
    )

``build_pec(data) -> (pec, inputs)`` is a top-level function that builds the model, as for distributed
fitting. Each worker calls it with a table of **n_trials_per_sample** rows to build its own copy of the
model.


.. _Neural_Likelihood_Fitting:

Fitting with an Estimator
-------------------------

To use a trained estimator, specify ``likelihood_estimator="neural"``, and the estimator, or the path to
one saved with its `save <NeuralLikelihood.save>` method, as the ``"artifact"`` of
**likelihood_estimator_kwargs**::

    pec = pnl.ParameterEstimationComposition(
        nodes=[comp],
        parameters={("rate", decision): np.linspace(-1.5, 1.5, 1000),
                    ("threshold", decision): np.linspace(0.3, 1.5, 1000)},
        outcome_variables=[decision.output_ports[pnl.DECISION_OUTCOME],
                           decision.output_ports[pnl.RESPONSE_TIME]],
        data=data,
        optimization_function="differential_evolution",
        likelihood_estimator="neural",
        likelihood_estimator_kwargs={"artifact": "ddm_nle.pt"},
    )
    pec.run(inputs={comp: trial_inputs})

The model is not simulated in the fit, so it is not compiled. `log_likelihood
<ParameterEstimationComposition.log_likelihood>` also uses the estimator, but cannot return simulated data
(``return_sim_data``).

In a :ref:`hierarchical fit <HierarchicalFitting>`, ``likelihood_estimator`` is specified for the
participant models built by the ``pec_factory``, and not for the group.


.. _Neural_Likelihood_Trial_Features:

Trial Features
--------------

Where trials differ from one another (for example, congruent and incongruent trials), an estimator
represents the distribution of outcomes on each kind of trial, which it distinguishes by the values of the
model's inputs on each trial. Those inputs that vary across the trials simulated in training are recorded
with the estimator.

When fitting, the same inputs are taken from those specified for `run <Composition.run>` or
`log_likelihood <ParameterEstimationComposition.log_likelihood>`, including any that do not vary in the
data being fit. These must therefore be the same inputs, in the same order, as those used for training;
otherwise an error is generated.


.. _Neural_Likelihood_Matching:

Matching an Estimator to a Model
--------------------------------

An estimator records the model for which it was trained, and generates an error if it is used to fit one
that differs in any of the following:

* the fitted parameters, or their order;
* the range of any parameter, which must lie within the range used for training;
* the outcome variables, their order, or which of them are categorical;
* the values of a categorical outcome, which must be among those simulated in training.

Other properties of the model, such as the values of parameters that are not fit, are not recorded; an
estimator should be retrained if any of these are changed.


.. _Neural_Likelihood_Validation:

Validation
----------

`train_neural_likelihood` generates an error if the estimator's negative log-likelihood on the data held
out from training is not finite, or if it assigns a finite density to fewer than 99.9% of a sample of the
simulated data. Specifying ``strict=False`` issues a `NeuralLikelihoodWarning` instead.

These checks identify an estimator that failed to train, but not one that is inaccurate. The accuracy of
an estimator for a given model can be assessed by fitting data simulated at known parameter values, and
comparing the estimates with those values.


.. _Neural_Likelihood_Limitations:

Limitations
-----------

* An estimator is valid only for the model, and the ranges of parameter values, for which it was trained
  (see `Neural_Likelihood_Matching`).
* The accuracy of a fit is limited by that of the estimator, which depends on the amount of simulated
  data and training.
* Parameter values are drawn evenly from within **bounds** for training, so regions of the parameter space
  in which the model's behavior changes rapidly are not represented in more detail than others.
* A fit that uses a neural likelihood cannot be distributed (``distributed=True``).


.. _Neural_Likelihood_Requirements:

Requirements
------------

Neural likelihoods require the ``nle`` extra, installed with ``pip install "psyneulink[nle]"``, which
includes `sbi <https://sbi-dev.github.io/sbi/>`_ and PyTorch.

:download:`train_neural_likelihood.py
<../../Scripts/Examples/ParameterEstimation/neural_likelihood/train_neural_likelihood.py>` trains an
estimator for a drift-diffusion model, and uses it to fit simulated data.


.. _Neural_Likelihood_Class_Reference:

Class Reference
---------------

.. autofunction:: psyneulink.core.components.functions.nonstateful.neurallikelihoodfunctions.train_neural_likelihood

.. autoclass:: psyneulink.core.components.functions.nonstateful.neurallikelihoodfunctions.NeuralLikelihood
   :members: log_likelihood, trial_log_prob, save, load

.. autoexception:: psyneulink.core.components.functions.nonstateful.neurallikelihoodfunctions.NeuralLikelihoodError

.. autoexception:: psyneulink.core.components.functions.nonstateful.neurallikelihoodfunctions.NeuralLikelihoodWarning
