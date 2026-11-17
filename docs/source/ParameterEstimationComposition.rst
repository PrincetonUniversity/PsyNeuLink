ParameterEstimationComposition
==============================

.. container:: related

    *Related*

    .. toctree::
       :maxdepth: 1

       OptimizationControlMechanism
       DistributedFitting
       HierarchicalFitting
       NeuralLikelihood

.. toctree::
   :maxdepth: 2

.. _PEC_Parameter_Space_Migration:

Migrating Parameter Spaces
--------------------------

Each entry in ``parameters`` now requires an explicit search-space dictionary.
Bare lists, NumPy arrays, and tuples are rejected, including when building a PEC
only to evaluate ``log_likelihood()``. A two-element tuple is accepted inside
``"bounds"``, but does not by itself declare a continuous range.

For continuous fitting, replace sample arrays with bounds::

    # Previously:
    parameters = {("threshold", decision): np.linspace(0.1, 1.0, 100)}

    # Now:
    parameters = {("threshold", decision): {"bounds": (0.1, 1.0)}}

Differential evolution has always searched continuously between the endpoints;
the intermediate samples did not restrict its candidates. Hierarchical fitting
also requires continuous bounds. Optuna supports continuous bounds as well.

For a grid, specify the spacing directly::

    parameters = {("threshold", decision): {"bounds": (0.1, 1.0), "step": 0.1}}

The grid starts at the lower bound and ends at the largest grid point no greater
than the upper bound. If the step does not divide the interval, Optuna warns and
reduces the grid's upper endpoint. PEC passes the specified step through without
inferring or rounding it.

For exact allowed values, including irregularly spaced values, use::

    parameters = {("threshold", decision): {"values": [0.1, 0.25, 0.9]}}

NumPy arrays are accepted inside ``"values"``. These are categorical choices to
Optuna, which changes sampler behavior compared with a numeric grid. Choose
``"step"`` when numeric grid spacing is intended. Currently, grids and exact choices
are supported only by the Optuna backend; differential evolution and hierarchical
fitting reject them.

Code that reads ``PECOptimizationFunction.fit_param_bounds`` should now unpack
``(lower, upper)`` pairs. Optuna distributions, including steps or choices, are
available through ``fit_param_distributions``.

.. automodule:: psyneulink.core.compositions.parameterestimationcomposition
   :members:
   :private-members:
   :exclude-members: _validate_params, run, evaluate
