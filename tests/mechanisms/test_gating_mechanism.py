import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions

from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AccumulatorIntegrator
from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.keywords import \
    DEFAULT_VARIABLE, FUNCTION, FUNCTION_PARAMS, INITIALIZER, RATE, TARGET_MECHANISM, VALUE
from psyneulink.core.compositions.composition import Composition

def test_gating_with_composition():
    """Tests same configuration as control of InputPort in tests/mechansims/test_identicalness_of_control_and_gating
    """
    Input_Layer = TransferMechanism(name='Input Layer', function=Logistic, size=2)
    Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1', function=Logistic, size=5)
    Hidden_Layer_2 = TransferMechanism(name='Hidden Layer_2', function=Logistic, size=4)
    Output_Layer = TransferMechanism(name='Output Layer', function=Logistic, size=3)

    Gating_Mechanism = GatingMechanism(size=[1], gate=[Hidden_Layer_1, Hidden_Layer_2, Output_Layer])

    Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
    Middle_Weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
    Output_Weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)
    # This projection is specified in add_backpropagation_learning_pathway method below
    Input_Weights = MappingProjection(name='Input Weights',matrix=Input_Weights_matrix)
    # This projection is "discovered" by add_backpropagation_learning_pathway method below
    Middle_Weights = MappingProjection(name='Middle Weights',sender=Hidden_Layer_1,receiver=Hidden_Layer_2,
        matrix={
            VALUE: Middle_Weights_matrix,
            FUNCTION: AccumulatorIntegrator,
            FUNCTION_PARAMS: {
                DEFAULT_VARIABLE: Middle_Weights_matrix,
                INITIALIZER: Middle_Weights_matrix,
                RATE: Middle_Weights_matrix
            },
        }
    )
    Output_Weights = MappingProjection(sender=Hidden_Layer_2, receiver=Output_Layer, matrix=Output_Weights_matrix)

    pathway = [Input_Layer, Input_Weights, Hidden_Layer_1, Hidden_Layer_2, Output_Layer]
    comp = Composition()
    backprop_pathway = comp.add_backpropagation_learning_pathway(pathway=pathway,
                                                                 loss_function=None)
    # c.add_linear_processing_pathway(pathway=z)
    comp.add_node(Gating_Mechanism)

    stim_list = {
        Input_Layer: [[-1, 30]],
        Gating_Mechanism: [1.0],
        backprop_pathway.target: [[0, 0, 1]]}

    comp.learn(num_trials=3, inputs=stim_list)

    expected_results = [[[0.81493513, 0.85129046, 0.88154205]],
                        [[0.81331773, 0.85008207, 0.88157851]],
                        [[0.81168332, 0.84886047, 0.88161468]]]

    assert np.allclose(comp.results, expected_results)

    stim_list[Gating_Mechanism]=[0.0]
    results = comp.learn(num_trials=1, inputs=stim_list)
    expected_results = [[[0.5, 0.5, 0.5]]]
    assert np.allclose(results, expected_results)

    stim_list[Gating_Mechanism]=[2.0]
    results = comp.learn(num_trials=1, inputs=stim_list)
    expected_results = [[0.96941429, 0.9837254 , 0.99217549]]
    assert np.allclose(results, expected_results)

def test_gating_with_UDF_with_composition():
    def my_linear_fct(
        x,
        m=2.0,
        b=0.0,
        params={
            pnl.ADDITIVE_PARAM: 'b',
            pnl.MULTIPLICATIVE_PARAM: 'm'
        }
    ):
        return m * x + b

    def my_simple_linear_fct(
        x,
        m=1.0,
        b=0.0
    ):
        return m * x + b

    def my_exp_fct(
        x,
        r=1.0,
        # b=pnl.CONTROL,
        b=0.0,
        params={
            pnl.ADDITIVE_PARAM: 'b',
            pnl.MULTIPLICATIVE_PARAM: 'r'
        }
    ):
        return x**r + b

    def my_sinusoidal_fct(
        input,
        phase=0,
        amplitude=1,
        params={
            pnl.ADDITIVE_PARAM: 'phase',
            pnl.MULTIPLICATIVE_PARAM: 'amplitude'
        }
    ):
        frequency = input[0]
        t = input[1]
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    Input_Layer = pnl.TransferMechanism(
        name='Input_Layer',
        default_variable=np.zeros((2,)),
        function=psyneulink.core.components.functions.transferfunctions.Logistic
    )

    Output_Layer = pnl.TransferMechanism(
        name='Output_Layer',
        default_variable=[0, 0, 0],
        function=psyneulink.core.components.functions.transferfunctions.Linear,
        # function=pnl.Logistic,
        # output_ports={pnl.NAME: 'RESULTS USING UDF',
        #                pnl.VARIABLE: [(pnl.OWNER_VALUE,0), pnl.TIME_STEP],
        #                pnl.FUNCTION: my_sinusoidal_fct}
        output_ports={
            pnl.NAME: 'RESULTS USING UDF',
            # pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
            pnl.FUNCTION: psyneulink.core.components.functions.transferfunctions.Linear(slope=pnl.GATING)
        }
    )

    Gating_Mechanism = pnl.GatingMechanism(
        size=[1],
        gating_signals=[
            # Output_Layer
            Output_Layer.output_port,
        ]
    )

    comp = Composition()
    comp.add_linear_processing_pathway(pathway=[Input_Layer, Output_Layer])
    comp.add_node(Gating_Mechanism)

    stim_list = {
        Input_Layer: [[-1, 30], [-1, 30], [-1, 30], [-1, 30]],
        Gating_Mechanism: [[0.0], [0.5], [1.0], [2.0]]
    }

    comp.run(num_trials=4, inputs=stim_list)

    expected_results = [
        [np.array([0., 0., 0.])],
        [np.array([0.63447071, 0.63447071, 0.63447071])],
        [np.array([1.26894142, 1.26894142, 1.26894142])],
        [np.array([2.53788284, 2.53788284, 2.53788284])]
    ]

    np.testing.assert_allclose(comp.results, expected_results)
