"""The PEC factory -- the "recipe" workers use to build a fresh serial PEC.

This is the pec_factory of PLAN_DASK: it depends only on serializable inputs
(the observed-data DataFrame) and never on a driver-side object, so any worker
can build its own local PEC. A constructed PEC is never shipped across the wire;
only this recipe (imported by reference) and lightweight payloads are.
"""

import numpy as np
import psyneulink as pnl

from . import config


def build_model():
    """Build the DDM composition (used for both data generation and the PEC)."""
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(**config.TRUE_PARAMS),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision), decision


def build_pec(data_to_fit):
    """Construct a fresh *serial* PEC for the DDM MLE problem (the pec_factory).

    Returns ``(pec, model_comp)``. Configured for:

      * **LLVM execution** -- required by ``log_likelihood``.
      * **Common random numbers** -- ``same_seed_for_all_parameter_combinations=
        True`` with a fixed ``initial_seed`` means every candidate is scored
        against *identical* simulation noise draws. This keeps the likelihood
        surface smooth (only the parameters differ between candidates) and makes
        the result reproducible regardless of which worker / scheduling order
        evaluated it. See ../CONCEPTS.md section 4 and the seed discussion.
    """
    comp, decision = build_model()

    fit_parameters = {
        ("rate", decision): np.linspace(*config.FIT_BOUNDS["rate"], 1000),
        ("threshold", decision): np.linspace(*config.FIT_BOUNDS["threshold"], 1000),
        ("non_decision_time", decision): np.linspace(*config.FIT_BOUNDS["non_decision_time"], 1000),
    }

    pec = pnl.ParameterEstimationComposition(
        name="pec_dask_mle",
        nodes=[comp],
        parameters=fit_parameters,
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data_to_fit,
        # The driver owns the real Optuna study; this inner optimizer is unused
        # for likelihood-only evaluation but is required by the PEC constructor.
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=1,
        ),
        num_estimates=config.NUM_ESTIMATES,
        initial_seed=config.INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,  # common random numbers
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, comp
