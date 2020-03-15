import numpy as np
import psyneulink as pnl

# Mechanisms
Input = pnl.TransferMechanism(name='Input')
reward = pnl.TransferMechanism(
    output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE], name='reward'
)
Decision = pnl.DDM(
    function=pnl.DriftDiffusionAnalytical(
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
        noise=0.5,
        starting_point=0,
        t0=0.45,
    ),
    output_ports=[
        pnl.DECISION_VARIABLE,
        pnl.RESPONSE_TIME,
        pnl.PROBABILITY_UPPER_THRESHOLD,
    ],
    name='Decision',
)

comp = pnl.Composition(name='comp')
comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
comp.add_linear_processing_pathway(task_execution_pathway)

comp.add_controller(
    controller=pnl.OptimizationControlMechanism(
        agent_rep=comp,
        features=[Input.input_port, reward.input_port],
        feature_function=pnl.AdaptiveIntegrator(rate=0.5),
        objective_mechanism=pnl.ObjectiveMechanism(
            function=pnl.LinearCombination(operation=pnl.PRODUCT),
            monitor=[
                reward,
                Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                (Decision.output_ports[pnl.RESPONSE_TIME], -1, 1),
            ],
        ),
        function=pnl.GridSearch,
        control=[
            {
                pnl.PROJECTIONS: ('drift_rate', Decision),
                pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3),
            },
            {
                pnl.PROJECTIONS: (pnl.THRESHOLD, Decision),
                pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3),
            },
        ],
    )
)
