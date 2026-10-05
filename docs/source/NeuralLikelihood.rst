.. _NeuralLikelihood:

Neural Likelihoods
==================

In `data fitting <ParameterEstimationComposition_Data_Fitting>`, a `ParameterEstimationComposition`
computes the likelihood of its **data** under each set of parameter values it considers by simulating
the model ``num_estimates`` times and estimating a density from the simulated outcomes. A neural
likelihood is a density estimator that is trained beforehand, on data simulated from the model, and then
used in place of those simulations. Each evaluation of the likelihood then requires a single evaluation
of the estimator, and the likelihood varies smoothly with the parameter values.


.. _Neural_Likelihood_Training:

Training an Estimator
---------------------

`train_neural_likelihood` simulates the model at parameter values drawn uniformly from within the ranges
specified in **bounds**, and trains an estimator on the results. The model can be specified directly,
together with the inputs used to run it::

    likelihood = pnl.train_neural_likelihood(
        bounds={"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)},
        outcome_names=("decision", "response_time"),
        pec=pec,
        inputs={comp: trial_inputs},
    )
    likelihood.save("ddm_nle.pt")

The keys of **bounds** must name the fitted parameters in the order in which the model lists them, and
**outcome_names** must be the column names of the **data** that the estimator will be used to fit, which
are in the order of the model's ``outcome_variables``. Each set of parameter values is simulated on the
trials in **inputs**, each as many times as the model's ``num_estimates``.

To distribute the simulations over a `Dask <https://www.dask.org>`_ cluster, specify a **pec_factory**
in place of **pec**, together with **distributed_options** (see :ref:`Distributed Fitting
<DistributedFitting>`)::

    likelihood = pnl.train_neural_likelihood(
        bounds={"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)},
        outcome_names=("decision", "response_time"),
        pec_factory=build_pec,
        n_trials_per_sample=500,
        distributed_options={"n_workers": 8},
    )

``build_pec(data) -> (pec, inputs)`` is a top-level function that builds the model, as for distributed
fitting. Each worker calls it with a table of **n_trials_per_sample** rows to build its own copy of the
model.

`train_neural_likelihood` generates an error if the estimator's negative log-likelihood on the data held
out from training is not finite, and records it per trial as ``metadata.val_nll``. This shows that training
worked, not that the estimator is accurate; for that, fit data simulated at known parameter values.


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

The composition being fitted is not simulated when fitting with an estimator, so it is not compiled.
`log_likelihood <ParameterEstimationComposition.log_likelihood>` also uses the estimator.

A parameter that depends on a condition (specified in **depends_on**) is fit separately for each condition,
and each trial is scored with the value for its condition. The estimator for such a fit is trained on the
model without **depends_on**, over the range of each parameter.

In a :ref:`hierarchical fit <HierarchicalFitting>`, ``likelihood_estimator`` is specified for the
participant models built by the ``pec_factory``, and not for the group.


.. _Neural_Likelihood_Trial_Features:

Trial Features
--------------

Where trials differ from one another (for example, congruent and incongruent trials), an estimator
represents the distribution of outcomes on each kind of trial, which it distinguishes by the values of the
model's inputs on each trial.

When fitting, the same inputs are taken from those specified for `run <Composition.run>` or
`log_likelihood <ParameterEstimationComposition.log_likelihood>`, including any that do not vary in the
data being fit. These must be laid out as they were for training, though the nodes to which they are
given can be listed in any order, and an input that did not vary in training must have the same value;
otherwise an error is generated, since the estimator has no information about such trials.


.. _Neural_Likelihood_Matching:

Matching an Estimator to a Model
--------------------------------

An estimator is valid only for the model, and the ranges of parameter values, for which it was trained;
within those ranges, it can be reused for any fit of that model. It records the following, and generates
an error if it is used to fit a model that differs in any of them:

* the fitted parameters, or their order;
* the range of any parameter, which must lie within the range used for training;
* the outcome variables, their order, or which of them are categorical;
* the values of a categorical outcome, which must be among those simulated in training.

Other properties of the model, such as the values of parameters that are not fit, are not recorded; an
estimator should be retrained if any of these are changed.


.. _Neural_Likelihood_Limitations:

Limitations
-----------

* At least one outcome must be continuous, such as a response time: an estimator cannot be trained for a
  model whose outcomes are all categorical.


.. _Neural_Likelihood_Requirements:

Requirements
------------

Neural likelihoods require the ``nle`` extra, installed with ``pip install "psyneulink[nle]"``, which
includes `sbi <https://sbi-dev.github.io/sbi/>`_ and PyTorch.

:download:`train_neural_likelihood.py
<../../Scripts/Examples/ParameterEstimation/neural_likelihood/train_neural_likelihood.py>` trains an
estimator for a drift-diffusion model, in one process or, with ``--distributed``, over a Dask cluster, and
uses it to fit simulated data.


.. _Neural_Likelihood_Class_Reference:

Class Reference
---------------

.. autofunction:: psyneulink.core.components.functions.nonstateful.neurallikelihoodfunctions.train_neural_likelihood

.. autoclass:: psyneulink.core.components.functions.nonstateful.neurallikelihoodfunctions.NeuralLikelihood
   :members: log_likelihood, trial_log_prob, save, load

.. autoexception:: psyneulink.core.components.functions.nonstateful.neurallikelihoodfunctions.NeuralLikelihoodError
