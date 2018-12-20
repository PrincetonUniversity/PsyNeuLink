import psyneulink as pnl
import numpy as np

import psyneulink.core.components.functions.distributionfunctions
from psyneulink import *

# TEST GaussianProcess in context of EVC-Gratton model

# Mechanisms
Input = pnl.TransferMechanism(
    name='Input',
)
Reward = pnl.TransferMechanism(
    output_states=[pnl.RESULT, pnl.OUTPUT_MEAN, pnl.OUTPUT_VARIANCE],
    name='Reward'
)
Decision = pnl.DDM(
    function=psyneulink.core.components.functions.distributionfunctions.DriftDiffusionAnalytical(
        drift_rate=(
            1.0,
            pnl.ControlProjection(
                function=pnl.Linear,
                control_signal_params={
                    pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                },
            ),
        ),
        threshold=(
            1.0,
            pnl.ControlProjection(
                function=pnl.Linear,
                control_signal_params={
                    pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                },
            ),
        ),
        noise=(0.5),
        starting_point=(0),
        t0=0.45
    ),
    output_states=[
        pnl.DECISION_VARIABLE,
        pnl.RESPONSE_TIME,
        pnl.PROBABILITY_UPPER_THRESHOLD
    ],
    name='Decision',
)

comp = pnl.Composition(name="evc")
comp.add_c_node(Reward, required_roles=[pnl.CNodeRole.TERMINAL])
comp.add_c_node(Decision, required_roles=[pnl.CNodeRole.TERMINAL])
task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
comp.add_linear_processing_pathway(task_execution_pathway)

ocm = pnl.OptimizationControlMechanism(features={pnl.SHADOW_EXTERNAL_INPUTS: [Input, Reward]},
                                       feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                                       agent_rep=comp,
                                       # function=pnl.GaussianProcess,
                                       function=pnl.GaussianProcess,
                                       control_signals=[("drift_rate", Decision), ("threshold", Decision)],
                                       objective_mechanism=pnl.ObjectiveMechanism(
                                                                         monitor_for_control=[
                                                                                 Reward,
                                                                                 Decision.PROBABILITY_UPPER_THRESHOLD,
                                                                                 (Decision.RESPONSE_TIME, -1, 1)]))
comp.add_model_based_optimizer(optimizer=ocm)

comp.enable_model_based_optimizer = True

# Stimuli
comp._analyze_graph()

stim_list_dict = {
    Input: [0.5, 0.123],
    Reward: [20, 20]
}
# print("- - - - - - - - RUN - - - - - - - -")
# comp.show_graph()
print (comp.run(inputs=stim_list_dict))
