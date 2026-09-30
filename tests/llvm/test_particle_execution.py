"""A trial boundary must not restart a compiled simulation's history."""

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.llvm.particle import ParticleExecution

pytestmark = [pytest.mark.llvm, pytest.mark.usefixtures("set_threads_to_one")]


def make_population(reset, *, nested=False, recurrent=False):
    if recurrent:
        node = pnl.RecurrentTransferMechanism(
            input_shapes=1,
            auto=0.2,
            function=pnl.Linear,
            integrator_mode=True,
            integration_rate=0.5,
            noise=pnl.NormalDist(standard_deviation=0.01, seed=5),
            reset_stateful_function_when=reset,
        )
        parameter = "slope"
    else:
        node = pnl.IntegratorMechanism(
            function=pnl.SimpleIntegrator(
                rate=1.0, noise=pnl.NormalDist(standard_deviation=0.01, seed=5)
            ),
            reset_stateful_function_when=reset,
        )
        parameter = "rate"
    model = pnl.Composition(pathways=[node])
    outcome = node
    if nested:
        outcome = pnl.ProcessingMechanism()
        model = pnl.Composition(pathways=[model, outcome])
    pec = pnl.ParameterEstimationComposition(
        model=model,
        parameters={(parameter, node): [0.5, 1.0]},
        outcome_variables=[outcome.output_port],
        data=pd.DataFrame({"value": [1.0, 2.0, 3.0]}),
        optimization_function=pnl.PECOptimizationFunction(method=None),
        num_estimates=4,
        initial_seed=42,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.function.set_pec_objective_function(
        lambda samples: float(np.mean(samples))
    )
    context = Context(
        execution_id=None, composition=pec, execution_phase=ContextFlags.PROCESSING
    )
    pec._prepare_pec_inputs_for_simulation({model: np.ones((3, 1))}, context)
    return pec, context


def run_split(pec, context, *, ancestors=None):
    controller = pec.controller
    full_inputs = controller._pec_input_values
    trials = controller.parameters.num_trials_per_estimate.get(context)
    controller.parameters.num_trials_per_estimate.set(1, context)
    outcomes = []
    try:
        with ParticleExecution(pec, context, 4) as session:
            for trial in range(3):
                controller._pec_input_values = {
                    pec.model: full_inputs[pec.model][trial : trial + 1]
                }
                inputs, length = pec._parse_run_inputs(
                    controller.parameters.state_feature_values._get(context), context
                )
                assert length == 1
                outcomes.append(session.advance(inputs))
                session.resample(np.arange(4) if ancestors is None else ancestors)
            # Closing and invalid ancestry are explicit, not native memory errors.
            with pytest.raises(ValueError, match="ancestors"):
                session.resample([-1, 0, 0, 0])
        with pytest.raises(RuntimeError, match="closed"):
            session.advance(inputs)
    finally:
        controller._pec_input_values = full_inputs
        controller.parameters.num_trials_per_estimate.set(trials, context)
    return np.array(outcomes)[..., pec.controller.function.outcome_variable_indices]


@pytest.mark.parametrize("reset", [pnl.Never, lambda: pnl.AtTrial(0), pnl.AtTrialStart])
@pytest.mark.parametrize(
    "nested,recurrent", [(False, False), (True, False), (False, True)]
)
def test_split_execution_matches_complete_history(reset, nested, recurrent):
    pec, context = make_population(reset(), nested=nested, recurrent=recurrent)
    full = pec.controller.function._run_simulations(1.0, context=context)
    full = full[..., pec.controller.function.outcome_variable_indices]
    split = run_split(pec, context)
    np.testing.assert_array_equal(split, full)


def test_duplicate_ancestors_preserve_independent_advanced_rngs():
    pec, context = make_population(pnl.Never(), nested=True)
    full = pec.controller.function._run_simulations(1.0, context=context)
    full = full[..., pec.controller.function.outcome_variable_indices]
    split = run_split(pec, context, ancestors=np.zeros(4, dtype=int))
    # Every child starts from parent zero but keeps its own next innovation.
    expected_second = full[0, 0] + full[1] - full[0]
    np.testing.assert_allclose(split[1], expected_second, atol=1e-6)
    assert np.ptp(split[1, :, 0]) > 0
    pnl.set_num_threads(2)
    np.testing.assert_array_equal(
        run_split(pec, context, ancestors=np.zeros(4, dtype=int)), split
    )


def test_run_execution_counts_continue_across_trial_launches():
    pec, context = make_population(pnl.Never())
    node = pec.model.nodes[0]
    node.reset_stateful_function_when = pnl.AtNCalls(
        node, 2, time_scale=pnl.TimeScale.RUN
    )
    full = pec.controller.function._run_simulations(1.0, context=context)
    full = full[..., pec.controller.function.outcome_variable_indices]
    split = run_split(pec, context)
    np.testing.assert_array_equal(split, full)
    assert np.mean(split[2]) < np.mean(split[1])
