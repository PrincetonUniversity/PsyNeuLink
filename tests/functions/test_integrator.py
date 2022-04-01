
import numpy as np
import pytest

import psyneulink as pnl
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.stateful.integratorfunctions as Functions
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.functions.nonstateful.transferfunctions import Angle
from psyneulink.core.globals.parameters import ParameterError

np.random.seed(0)
SIZE=10
test_var = np.random.rand(SIZE)
test_initializer = np.random.rand(SIZE)
test_noise_arr = np.random.rand(SIZE)

RAND0_1 = np.random.random()
RAND2 = np.random.rand()
RAND3 = np.random.rand()

def SimpleIntFun(init, value, iterations, noise, rate, offset, **kwargs):
    assert iterations == 3
    if np.isscalar(noise):
        if "initializer" in kwargs:
            return [4.91845218, 4.78766907, 4.73758993, 5.04920442, 4.09842889,
                    4.2909061,  4.05866892, 5.23154257, 5.23413599, 4.86548903]
        else:
            return [4.12672714, 4.25877415, 4.16954537, 4.12360778, 4.02739283,
                    4.2037768,  4.03845052, 4.39892272, 4.45597924, 3.99547688]
    elif isinstance(noise, pnl.DistributionFunction):
        if "initializer" in kwargs:
            return [6.07047464, 1.45183492, 2.13615798, 3.22296925, 3.29867927,
                    0.9734048, 2.54011924, 3.21213761, 1.54651058, 2.7026355, ]
        else:
            return [5.2787496, 0.92294, 1.56811342, 2.29737262, 3.22764321,
                    0.8862755, 2.51990084, 2.37951776, 0.76835383, 1.83262335]
    else:
        if "initializer" in kwargs:
            return [5.53160614, 4.86244369, 3.79932695, 5.06809088, 2.1305511,
                    3.8879681,  2.16602771, 5.74284825, 4.47697989, 3.78677378]
        else:
            return [4.7398811, 4.33354877, 3.23128239, 4.14249424, 2.05951504,
                    3.8008388, 2.14580932, 4.9102284,  3.69882314, 2.91676163]

def AdaptiveIntFun(init, value, iterations, noise, rate, offset, **kwargs):
    assert iterations == 3
    if np.isscalar(noise):
        if "initializer" in kwargs:
            return [3.44619156, 3.44183529, 3.38970396, 3.49707692, 3.08413924,
                    3.22437653, 3.07231498, 3.66899395, 3.69062231, 3.37774376]
        else:
            return [3.13125441, 3.23144828, 3.16374378, 3.12888752, 3.05588209,
                    3.18971771, 3.06427238, 3.33778941, 3.38108243, 3.03166509]
    elif isinstance(noise, pnl.DistributionFunction):
        if "initializer" in kwargs:
            return [4.18870661, 1.3561085, 1.69287182, 1.94643064, 2.12581409,
                    1.05242466, 2.05628752, 1.90164378, 1.18394637, 1.39578569]
        else:
            return [3.87376946, 1.14572149, 1.46691163, 1.57824123, 2.09755694,
                    1.01776584, 2.04824492, 1.57043925, 0.8744065, 1.04970702]
    else:
        if "initializer" in kwargs:
            return [3.91143701, 3.49857235, 2.67777415, 3.51140748, 1.59096419,
                    2.91863753, 1.63622751, 4.05695955, 3.11611173, 2.55924237]
        else:
            return [3.59649986, 3.28818534, 2.45181396, 3.14321808, 1.56270704,
                    2.88397872, 1.62818492, 3.72575501, 2.80657186, 2.2131637]


def DriftIntFun(init, value, iterations, noise, **kwargs):
    assert iterations == 3
    if np.isscalar(noise):
        if "initializer" not in kwargs:
            return ([0.35782281, 4.03326927, 4.90427264, 0.90944534, 1.45943493,
                     2.31791882, 3.05580281, 1.20089146, 2.8408554 , 1.93964773],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
        else:
            return ([1.14954785, 4.56216419, 5.4723172 , 1.83504198, 1.53047099,
                     2.40504812, 3.07602121, 2.0335113 , 3.61901215, 2.80965988],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
    else:
        if "initializer" not in kwargs:
            return ([0.17810305, 4.06675934, 4.20730295, 0.90582833, 1.60883329,
                     2.27822395, 2.2923697 , 1.10933472, 2.71418965, 1.86808107],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
        else:
            return ([0.96982809, 4.59565426, 4.77534751, 1.83142497, 1.67986935,
                     2.36535325, 2.3125881 , 1.94195457, 3.4923464 , 2.73809322],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

def LeakyFun(init, value, iterations, noise, **kwargs):
    assert iterations == 3

    if np.isscalar(noise):
        if "initializer" not in kwargs:
            return [2.20813608, 2.25674001, 2.22389663, 2.2069879,  2.17157305, 2.23649656, 2.17564317, 2.30832598, 2.32932737, 2.15982541]
        else:
            return [2.93867224, 2.74475902, 2.74803958, 3.06104933, 2.23711905, 2.31689203, 2.19429898, 3.07659637, 3.04734388, 2.96259823]
    elif isinstance(noise, pnl.DistributionFunction):
        if "initializer" not in kwargs:
            return [2.55912037, 1.24455938, 1.43417309, 1.638423, 1.91298882, 1.22700281, 1.71226825, 1.67794471, 1.20395947, 1.48326449]
        else:
            return [3.28965653, 1.73257839, 1.95831604, 2.49248443, 1.97853482, 1.30739828, 1.73092406, 2.4462151, 1.92197598, 2.28603731]
    else:
        if "initializer" not in kwargs:
            return [2.39694798, 2.27976578, 1.9349721, 2.21280371, 1.5655935, 2.11241762, 1.59283164, 2.46577518, 2.09617208, 1.82765063]
        else:
            return [3.12748415, 2.76778478, 2.45911505, 3.06686514, 1.6311395, 2.19281309, 1.61148745, 3.23404557, 2.81418859, 2.63042344]


def AccumulatorFun(init, value, iterations, noise, **kwargs):
    assert iterations == 3

    if np.isscalar(noise):
        if "initializer" not in kwargs:
            # variable is not used in Accumulator
            return [[1.38631136, 1.38631136, 1.38631136, 1.38631136, 1.38631136,
                     1.38631136, 1.38631136, 1.38631136, 1.38631136, 1.38631136]]
        else:
            return [[1.40097107, 1.39610447, 1.39682937, 1.40344986, 1.38762668,
                     1.38792466, 1.38668573, 1.40172829, 1.40071984, 1.40242065]]
    elif isinstance(noise, pnl.DistributionFunction):
        if "initializer" not in kwargs:
            return [[1.46381634, 0.97440038, 0.54931704, 0.28681701, 0.26162584,
                     0.66800459, 1.1010486, 0.02587729, 0.38761176, -0.56452977]]
        else:
            return [[1.47847605, 0.98419348, 0.55983505, 0.30395551, 0.26294116,
                     0.66961789, 1.10142297, 0.04129421, 0.40202024, -0.54842049]]
    else:
        if "initializer" not in kwargs:
            return [[1.65907194, 1.41957474, 0.96892655, 1.39471298, 0.51090402,
                     1.20706503, 0.5443729, 1.61376489, 1.04949166, 0.90644658]]
        else:
            return [[1.67373165, 1.42936784, 0.97944456, 1.41185147, 0.51221934,
                     1.20867833, 0.54474727, 1.62918182, 1.06390014, 0.92255587]]


GROUP_PREFIX="IntegratorFunction "


@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("variable, params", [
    (test_var, {'rate':RAND0_1, 'offset':RAND3}),
    (test_var, {'initializer':test_initializer, 'rate':RAND0_1, 'offset':RAND3}),
    ], ids=["Default", "Initializer"])
@pytest.mark.parametrize("noise", [RAND2, test_noise_arr, pnl.NormalDist],
                         ids=["SNOISE", "VNOISE", "FNOISE"])
@pytest.mark.parametrize("func", [
    (Functions.AdaptiveIntegrator, AdaptiveIntFun),
    (Functions.SimpleIntegrator, SimpleIntFun),
    (Functions.DriftDiffusionIntegrator, DriftIntFun),
    (Functions.LeakyCompetingIntegrator, LeakyFun),
    (Functions.AccumulatorIntegrator, AccumulatorFun),
    ], ids=lambda x: x[0])
@pytest.mark.benchmark
def test_execute(func, func_mode, variable, noise, params, benchmark):
    benchmark.group = GROUP_PREFIX + func[0].componentName
    try:
        noise = noise()
    except TypeError as e:
        if "object is not callable" not in str(e):
            raise e from None
    else:
        assert isinstance(noise, pnl.DistributionFunction)
        if func[1] == DriftIntFun:
            pytest.skip("DriftDiffusionIntegrator doesn't support functional noise")

    if 'DriftOnASphereIntegrator' in func[0].componentName:
        if func_mode != 'Python':
            pytest.skip("DriftDiffusionIntegrator not yet compiled")
        params.update({'dimension':len(variable) + 1})
    else:
        if 'dimension' in params:
            params.pop('dimension')

    if 'AccumulatorIntegrator' in func[0].componentName:
        params = {
            **params,
            'increment': RAND0_1,
        }
        params.pop('offset')

    # If we are dealing with a DriftDiffusionIntegrator, noise and time_step_size defaults
    # have changed since this test was created. Hard code their old values.
    if 'DriftDiffusionIntegrator' in str(func[0]):
        f = func[0](default_variable=variable, noise=np.sqrt(noise), time_step_size=1.0, **params)
    else:
        f = func[0](default_variable=variable, noise=noise, **params)

    ex = pytest.helpers.get_func_execution(f, func_mode)

    ex(variable)
    ex(variable)
    res = ex(variable)
    expected = func[1](f.initializer, variable, 3, noise, **params)
    for r, e in zip(res, expected):
        assert np.allclose(r, e)

    if benchmark.enabled:
        benchmark(ex, variable)


def test_integrator_function_no_default_variable_and_params_len_more_than_1():
    I = Functions.AdaptiveIntegrator(rate=[.1, .2, .3])
    I.defaults.variable = np.array([0,0,0])

def test_integrator_function_default_variable_len_1_but_user_specified_and_params_len_more_than_1():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[1], rate=[.1, .2, .3])
    error_msg_a = 'The length of the array specified for the rate parameter'
    error_msg_b = 'must match the length of the default input'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_default_variable_and_params_len_more_than_1_error():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[0,0], rate=[.1, .2, .3])
    error_msg_a = 'The length of the array specified for the rate parameter'
    error_msg_b = 'must match the length of the default input'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_with_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(rate=[.1, .2, .3], offset=[.4,.5])
    error_msg_a = "The parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "(['offset', 'rate']) don't all have the same length"
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_with_default_variable_and_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[0,0,0], rate=[.1, .2, .3], offset=[.4,.5])
    error_msg_a = "The following parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "don't have the same length as its 'default_variable' (3): ['offset']."
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)


err_msg_initializer = "'initializer' must be a list or 1d array of length 3 (the value of the 'dimension' parameter minus 1)"
err_msg_angle_func = 'Variable shape incompatibility between (DriftOnASphereIntegrator DriftOnASphereIntegrator'
err_msg_noise = "must be a list or 1d array of length 3 (the value of the 'dimension' parameter minus 1)"

test_vars = [
    ({'initializer': 0.1}, err_msg_initializer, FunctionError),
    ({'initializer': [0.1,0.1]}, err_msg_initializer, FunctionError),
    ({'initializer': [0.1,0.1,0.1]}, None, None),
    ({'angle_function': Angle}, None, None),
    ({'angle_function': Angle()}, None, None),
    ({'angle_function': Angle([1,1])}, err_msg_angle_func, ParameterError),
    ({'angle_function': Angle([1,1,1])}, None, None),
    ({'noise': .01}, None, None),
    ({'noise': [.01, .5]}, err_msg_noise, FunctionError),
    ({'noise': [.01, .5, .99]}, None, None),
    ({'noise': [.01, .5, .99, .1]}, err_msg_noise, FunctionError)
]

names = [
    "INITIALIZER_SCALAR", "INITIALIZER_2", "INITIALIZER_3",
    "ANGLE_CLASS", "ANGLE_NONE", "ANGLE_2", "ANGLE_3",
    "NOISE_SCALAR", "NOISE_2", "NOISE_3", "NOISE_4"
]

# FIX: CROSS WITH INITIALIZER SIZE:
@pytest.mark.parametrize("params, error_msg, error_type", test_vars, ids=names)
def test_drift_on_a_sphere_errors(params, error_msg, error_type):
    if error_type:
        with pytest.raises(error_type) as error_text:
            Functions.DriftOnASphereIntegrator(dimension=4, params=params)
        assert error_msg in str(error_text.value)
    else:
        Functions.DriftOnASphereIntegrator(dimension=4, params=params)
