import logging

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.combinationfunctions import \
    LinearCombination, Concatenate
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import GridSearch
from psyneulink.core.compositions.parameterestimationcomposition import ParameterEstimationComposition

logger = logging.getLogger(__name__)


# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for ParameterEstimationComposition

# objective_function = {None: 2, Concatenate: 2, LinearCombination: 1}
# expected

pec_test_args = [(None, 2, True, False),
                 (None, 2, False, True),
                 (Concatenate, 2, True, False),
                 (LinearCombination, 1, True, False),
                 # (None, 2, True, True), <- USE TO TEST ERROR
                 # (None, 2, False, False), <- USE TO TEST ERROR
                 ]

@pytest.mark.parametrize(
    'objective_function_arg, expected_input_len, model_spec, node_spec',
    pec_test_args,
    ids=[f"{x[0]}-{'model' if x[2] else None}-{'nodes' if x[3] else None})" for x in pec_test_args]
)
def test_parameter_estimation_composition(objective_function_arg, expected_input_len, model_spec, node_spec):
    Input = pnl.TransferMechanism(name='Input')
    reward = pnl.TransferMechanism(output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE],
                                   name='reward',
                                   # integrator_mode=True,
                                   # noise=NormalDist  # <- FIX 11/3/31: TEST ALLOCATION OF SEED FOR THIS WHEN WORKING
                                   )
    Decision = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0,
                                                                         pnl.ControlProjection(function=pnl.Linear,
                                                                                               control_signal_params={
                                                                                                   pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3),
                                                                                               })),
                                                             threshold=(1.0,
                                                                        pnl.ControlProjection(function=pnl.Linear,
                                                                                              control_signal_params={
                                                                                                  pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3),
                                                                                              })),
                                                             noise=0.5,
                                                             starting_point=0,
                                                             t0=0.45),
                       output_ports=[pnl.DECISION_VARIABLE,
                                    pnl.RESPONSE_TIME,
                                    pnl.PROBABILITY_UPPER_THRESHOLD],
                       name='Decision')
    Decision2 = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=1.0,
                                                             threshold=1.0,
                                                             noise=0.5,
                                                             starting_point=0,
                                                             t0=0.45),
                       output_ports=[pnl.DECISION_VARIABLE,
                                    pnl.RESPONSE_TIME,
                                    pnl.PROBABILITY_UPPER_THRESHOLD],
                       name='Decision')


    comp = pnl.Composition(name="evc", retain_old_simulation_data=True)
    comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
    comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
    comp.add_node(Decision2, required_roles=[pnl.NodeRole.OUTPUT])
    task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision, Decision2]
    comp.add_linear_processing_pathway(task_execution_pathway)

    pec = ParameterEstimationComposition(name='pec',
                                         model = comp if model_spec else None,
                                         nodes = comp if node_spec else None,
                                         # data = [1,2,3],    # For testing error
                                         parameters={('drift_rate',Decision):[1,2],
                                                     ('threshold',Decision2):[1,2],},
                                         # parameters={('shrimp_boo',Decision):[1,2],   # For testing error
                                         #             ('scripblat',Decision2):[1,2],}, # For testing error
                                         outcome_variables=[Decision.output_ports[pnl.DECISION_VARIABLE],
                                                            Decision.output_ports[pnl.RESPONSE_TIME]],
                                         objective_function=objective_function_arg,
                                         optimization_function=GridSearch,
                                         num_estimates=3,
                                         # controller_mode=AFTER,   # For testing error
                                         # enable_controller=False  # For testing error
                                         )
    ctlr = pec.controller
    if objective_function_arg:
        assert ctlr.objective_mechanism                         # For objective_function specified
    else:
        assert not ctlr.objective_mechanism                         # For objective_function specified
    assert len(ctlr.input_ports[pnl.OUTCOME].variable) == expected_input_len
    assert len(ctlr.control_signals) == 3
    assert pnl.RANDOMIZATION_CONTROL_SIGNAL_NAME in ctlr.control_signals.names
    assert ctlr.control_signals[pnl.RANDOMIZATION_CONTROL_SIGNAL_NAME].allocation_samples.base.num == 3
    # pec.run()
