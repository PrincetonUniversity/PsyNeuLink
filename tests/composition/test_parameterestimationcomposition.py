import logging
import timeit as timeit

import numpy as np

import pytest

import psyneulink as pnl

from psyneulink.core.compositions.parameterestimationcomposition import ParameterEstimationComposition
from psyneulink.core.components.functions.nonstateful.combinationfunctions import \
    LinearCombination, Concatenate
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import GridSearch

logger = logging.getLogger(__name__)


# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for ParameterEstimationComposition

def test_parameter_estimation_composition(autodiff_mode):
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
                                         model=comp,
                                         # nodes = comp,  # For testing alternative
                                         # data = [1,2,3],       # For testing error
                                         parameters={('drift_rate',Decision):[1,2],
                                                     ('threshold',Decision2):[1,2],},
                                         # parameters={('shrimp_boo',Decision):[1,2],   # For testing error
                                         #             ('scripblat',Decision2):[1,2],}, # For testing error
                                         outcome_variables=[Decision.output_ports[pnl.DECISION_VARIABLE],
                                                            Decision.output_ports[pnl.RESPONSE_TIME]],
                                         # objective_function=LinearCombination, <- FIX: TEST WITH AND WITHOUT
                                         optimization_function=GridSearch,
                                         num_estimates=3,
                                         # controller_mode=AFTER,   # For testing error
                                         # enable_controller=False  # For testing error
                                         )
    # pec.run()
