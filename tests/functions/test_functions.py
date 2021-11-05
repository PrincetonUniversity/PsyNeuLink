import numpy as np
import psyneulink as pnl
import pytest

import psyneulink.core.components.functions.nonstateful.transferfunctions


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
    f = psyneulink.core.components.functions.nonstateful.transferfunctions.Linear()
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
        f = psyneulink.core.components.functions.nonstateful.transferfunctions.Linear()
        f.convert_output_type(variable, output_type=output_type)


@pytest.mark.function
@pytest.mark.parametrize(
    "obj",
    [
        pnl.NormalDist,
        pnl.UniformToNormalDist,
        pnl.ExponentialDist,
        pnl.UniformDist,
        pnl.GammaDist,
        pnl.WaldDist,
        pnl.GaussianDistort,
    ]
)
def test_seed_setting_results(obj):
    obj = obj()
    new_seed = 1

    obj.parameters.seed.set(new_seed, context='c1')

    c1res1 = obj.execute(context='c1')
    c1res2 = obj.execute(context='c1')
    assert c1res1 != c1res2
    assert obj.execute(context='c2') != c1res1

    obj.parameters.seed.set(new_seed, context='c2')
    assert obj.execute(context='c2') == c1res1
    assert obj.execute(context='c2') == c1res2

    obj.parameters.seed.set(new_seed, context='c1')
    assert obj.execute(context='c1') == c1res1


# these functions use seed but the results from .execute may not directly change
# with different seeds, unlike those in test_seed_setting_results
@pytest.mark.function
@pytest.mark.parametrize(
    "obj",
    [
        pnl.DDM,
        pnl.DriftDiffusionIntegrator,
        pnl.DriftOnASphereIntegrator(dimension=3),
        pnl.OrnsteinUhlenbeckIntegrator,
        pnl.DictionaryMemory,
        pnl.ContentAddressableMemory,
    ]
)
def test_seed_setting_params(obj):
    if not isinstance(obj, pnl.Component):
        obj = obj()
    new_seed = 1

    obj._initialize_from_context(pnl.Context(execution_id='c1'))
    obj._initialize_from_context(pnl.Context(execution_id='c2'), pnl.Context(execution_id='c1'))
    obj.parameters.seed.set(new_seed, context='c1')

    c1res1 = obj.parameters.random_state.get('c1').rand()
    c1res2 = obj.parameters.random_state.get('c1').rand()
    assert c1res1 != c1res2
    assert obj.parameters.random_state.get('c2').rand() != c1res1

    obj.parameters.seed.set(new_seed, context='c2')
    assert obj.parameters.random_state.get('c2').rand() == c1res1
    assert obj.parameters.random_state.get('c2').rand() == c1res2

    obj.parameters.seed.set(new_seed, context='c1')
    assert obj.parameters.random_state.get('c1').rand() == c1res1


def test_runtime_params_reset():
    f = pnl.Linear()
    assert f.function(1) == 1
    assert f.function(1, params={'slope': 2}) == 2
    assert f.function(1) == 1
