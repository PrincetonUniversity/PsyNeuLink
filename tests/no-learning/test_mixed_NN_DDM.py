import numpy as np

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Functions.Function import Linear, Logistic, BogaczEtAl
from PsyNeuLink.Globals.Keywords import RANDOM_CONNECTIVITY_MATRIX, FULL_CONNECTIVITY_MATRIX


def test_mixed_NN_DDM():
    input_layer = TransferMechanism(
        name='Input Layer',
        function=Linear(),
        default_input_value=[0, 0],
    )

    hidden_layer = TransferMechanism(
        name='Hidden Layer 1',
        function=Logistic(gain=1.0, bias=0),
        default_input_value=np.zeros((5,)),
    )

    ddm = DDM(
        name='My_DDM',
        function=BogaczEtAl(
            drift_rate=0.5,
            threshold=1,
            starting_point=0.0,
        ),
    )

    p = process(
        name='Neural Network DDM Process',
        default_input_value=[0, 0],
        pathway=[
            input_layer,
            RANDOM_CONNECTIVITY_MATRIX,
            hidden_layer,
            FULL_CONNECTIVITY_MATRIX,
            ddm,
        ],
    )

    s = system(processes=[p])

    stim_list = {input_layer: [[-1, 2], [2, 3], [5, 5]]}
    result = s.run(inputs=stim_list)

    expected_output = [
        (input_layer.output_states[0].value, np.array([5., 5.])),
        (hidden_layer.input_states[0].value, np.array([3.5030505, 4.88571699, 5.24563838, 6.57282596, 5.01986775])),
        (hidden_layer.output_states[0].value, np.array([0.97077444, 0.99250292, 0.99475718, 0.99860411, 0.99343794])),
        (ddm.input_states[0].value, np.array([4.9500766])),
        (ddm.output_states[0].value, np.array([1.])),
        (result, [[np.array([1.]), np.array([0.89585249])], [np.array([1.]), np.array([0.63678208])], [np.array([1.]), np.array([0.60403415])]]),
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
