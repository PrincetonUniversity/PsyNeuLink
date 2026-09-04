.. _NeuralLikelihood:

Neural Likelihoods
==================

`ParameterEstimationComposition` scores a candidate parameter setting by simulating the
model and turning the simulated outcomes into a density by kernel density estimation.
Every evaluation therefore costs ``num_estimates`` simulations, and the density it
produces carries the Monte Carlo noise of those simulations.

A neural likelihood replaces that pipeline with a conditional density model
``p(outcomes | parameters)``, trained once on simulated data. Fitting afterwards costs a
network forward pass rather than a batch of simulations, and the surface is smooth and
differentiable rather than noisy.

The simulation cost is paid once instead of on every evaluation, and the trained estimator
can be reused across fits of the same model. How much cheaper an evaluation becomes depends
on how many simulations the density would otherwise need: scoring 400 trials of a
drift-diffusion model takes roughly 820 ms with ``num_estimates=300`` and roughly 4 ms with
a trained estimator.


.. _Neural_Likelihood_Training:

Training an estimator
---------------------

Training is explicit and offline: it takes minutes to hours, and reusing a saved estimator
is the point::

    from psyneulink import train_neural_likelihood

    likelihood = train_neural_likelihood(
        build_pec,
        bounds={"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)},
        outcome_names=("decision", "response_time"),
        n_parameter_samples=20000,
        n_trials_per_sample=100,
    )
    likelihood.save("ddm_nle.pt")

``build_pec`` is a ``pec_factory(data) -> (pec, inputs)``, the same contract distributed
and hierarchical fitting use (see :ref:`DistributedFitting`). It is called with a
placeholder table, since training simulates rather than fits.

Parameter draws are taken across the box given by **bounds**, which is also the region the
estimator is valid over. Generation is embarrassingly parallel; passing
``distributed_options`` spreads it over a Dask cluster, resolved exactly as for
distributed fitting.


.. _Neural_Likelihood_Fitting:

Fitting with one
----------------

``likelihood_estimator="neural"`` replaces the likelihood for any fit, hierarchical or
not::

    pec = ParameterEstimationComposition(
        nodes=[model],
        parameters={("rate", decision): np.linspace(-1.5, 1.5, 1000)},
        outcome_variables=[decision.output_ports[DECISION_OUTCOME],
                           decision.output_ports[RESPONSE_TIME]],
        data=data,
        optimization_function=PECOptimizationFunction(method="differential_evolution"),
        likelihood_estimator="neural",
        likelihood_estimator_kwargs={"artifact": "ddm_nle.pt"},
    )

**artifact** is either a trained `NeuralLikelihood` or the path to one. Nothing is
simulated during the fit, so the model is never compiled and ``comp_execution_mode`` does
not apply.

A single-participant fit benefits as much as a group one: the estimator is a property of
the model, not of how many participants are being fitted.


.. _Neural_Likelihood_Conditioning:

What the estimator is conditioned on
------------------------------------

The conditioning vector is the fitted parameters followed by the per-trial features, which
are the values entering the composition's input nodes on that trial. A model whose trials
differ -- congruent against incongruent, switch against repeat -- is therefore conditioned
on which trial it is scoring rather than on the parameters alone.

Features that do not vary across trials carry no information and are dropped, so a model
driven by a constant input is conditioned on its parameters alone.

The features are read from the ``inputs`` the factory returns during training, and from
the ``inputs`` passed to `run <Composition.run>` when fitting. They have to describe trials
the same way in both, and a fit that cannot supply them raises rather than scoring against
a different conditioning.


.. _Neural_Likelihood_Provenance:

What an estimator is valid for
------------------------------

An estimator trained for one model will score a different one without complaint, returning
plausible numbers that mean nothing. Each `NeuralLikelihood` therefore records what it was
trained for, and refuses to be used against anything else:

* the fitted parameters **and their order**, since the conditioning vector is positional;
* the range of each parameter;
* the outcome variables, their order, and which of them are categorical;
* the categories a categorical outcome took while training;
* the model it was trained on, and the versions of PsyNeuLink and ``sbi`` that produced it.

Ranges are checked for containment rather than equality. Fitting inside the trained box is
supported; fitting outside it is extrapolation, whose error is unbounded and silent, so it
raises.


.. _Neural_Likelihood_Gates:

Validation
----------

`train_neural_likelihood` refuses to return an estimator that did not train: the held-out
negative log-likelihood must be finite, and the estimator must assign a finite density to
essentially all held-out data. ``strict=False`` downgrades these to a
`NeuralLikelihoodWarning`.

These gates catch an estimator that failed, not one that is merely mediocre. Whether a
trained estimator is good enough for a given model is a question about that model, and is
answered by comparing recovered parameters against known values on simulated data.


.. _Neural_Likelihood_Limitations:

Limitations
-----------

* An estimator is only valid inside the box it was trained on, and only for the model that
  produced its training data.
* Interval width tracks the quality of the likelihood. An estimator trained on too few
  simulations gives intervals that are too narrow, and no amount of fitting corrects that.
* Training draws parameters uniformly across the box. A model whose behaviour changes
  sharply in a small region of that box is represented no more finely there than anywhere
  else.
* A parameter the data barely constrain stays barely constrained: a better likelihood
  estimates a flat surface more faithfully, it does not make it informative.
* ``return_sim_data`` is not available, since nothing is simulated.


.. _Neural_Likelihood_Requirements:

Requirements
------------

Neural likelihoods require an extra, installed with ``pip install "psyneulink[nle]"``.
Nothing is imported from it unless ``likelihood_estimator="neural"`` is requested, or an
estimator is trained.

:download:`train_neural_likelihood.py <../../Scripts/Debug/pec_nle/train_neural_likelihood.py>`
trains an estimator for a drift-diffusion model and fits with it.
