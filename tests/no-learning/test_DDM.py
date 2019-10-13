import numpy as np

from psyneulink.core.components.functions.distributionfunctions import DriftDiffusionAnalytical
from psyneulink.core.components.process import Process
from psyneulink.core.globals.keywords import FULL_CONNECTIVITY_MATRIX, IDENTITY_MATRIX
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM


# does not run a system, can be used to ensure that running processes alone still works
def test_DDM():
    myMechanism = DDM(
        function=DriftDiffusionAnalytical(
            drift_rate=(1.0),
            threshold=(10.0),
            starting_point=0.0,
        ),
        name='My_DDM',
    )

    myMechanism_2 = DDM(
        function=DriftDiffusionAnalytical(
            drift_rate=2.0,
            threshold=20.0),
        name='My_DDM_2'
    )

    myMechanism_3 = DDM(
        function=DriftDiffusionAnalytical(
            drift_rate=3.0,
            threshold=30.0
        ),
        name='My_DDM_3',
    )

    z = Process(
        default_variable=[[30], [10]],
        pathway=[
            myMechanism,
            (IDENTITY_MATRIX),
            myMechanism_2,
            (FULL_CONNECTIVITY_MATRIX),
            myMechanism_3
        ],
    )

    result = z.execute([[30], [10]])

    expected_output = [
        (myMechanism.input_ports[0].parameters.value.get(z), np.array([40.])),
        (myMechanism.output_ports[0].parameters.value.get(z), np.array([10.])),
        (myMechanism_2.input_ports[0].parameters.value.get(z), np.array([10.])),
        (myMechanism_2.output_ports[0].parameters.value.get(z), np.array([20.])),
        (myMechanism_3.input_ports[0].parameters.value.get(z), np.array([20.])),
        (myMechanism_3.output_ports[0].parameters.value.get(z), np.array([30.])),
        (result, np.array([30.])),
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
