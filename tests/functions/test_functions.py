import numpy as np
import psyneulink as pnl
import pytest

import psyneulink.core.components.functions.transferfunctions


@pytest.mark.parametrize(
    'output_type, variable, expected_output',
    [
        (pnl.FunctionOutputType.RAW_NUMBER, 1, 1.),
        (pnl.FunctionOutputType.RAW_NUMBER, [1], 1.),
        (pnl.FunctionOutputType.RAW_NUMBER, [[1]], 1.),
        (pnl.FunctionOutputType.RAW_NUMBER, [[[1]]], 1.),
        (pnl.FunctionOutputType.NP_1D_ARRAY, 1, np.array([1.])),
        (pnl.FunctionOutputType.NP_1D_ARRAY, [1], np.array([1.])),
        (pnl.FunctionOutputType.NP_1D_ARRAY, [[1]], np.array([1.])),
        (pnl.FunctionOutputType.NP_1D_ARRAY, [[[1]]], np.array([1.])),
        (pnl.FunctionOutputType.NP_2D_ARRAY, 1, np.array([[1.]])),
        (pnl.FunctionOutputType.NP_2D_ARRAY, [1], np.array([[1.]])),
        (pnl.FunctionOutputType.NP_2D_ARRAY, [[1]], np.array([[1.]])),
        (pnl.FunctionOutputType.NP_2D_ARRAY, [[[1]]], np.array([[1.]])),
    ]
)
def test_output_type_conversion(output_type, variable, expected_output):
    f = psyneulink.core.components.functions.transferfunctions.Linear()
    f.output_type = output_type
    f.enable_output_type_conversion = True

    assert f.execute(variable) == expected_output


@pytest.mark.parametrize(
    'output_type, variable',
    [
        (pnl.FunctionOutputType.RAW_NUMBER, [1, 1]),
        (pnl.FunctionOutputType.RAW_NUMBER, [[1, 1]]),
        (pnl.FunctionOutputType.RAW_NUMBER, [[[1], [1, 1]]]),
        (pnl.FunctionOutputType.NP_1D_ARRAY, [[1, 1], [1, 1]]),
    ]
)
def test_output_type_conversion_failure(output_type, variable):
    with pytest.raises(pnl.FunctionError):
        f = psyneulink.core.components.functions.transferfunctions.Linear()
        f.convert_output_type(variable, output_type=output_type)
