import numpy as np
import pytest

import psyneulink as pnl
import psyneulink.core.components.functions.stateful.integratorfunctions as Functions
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.globals.parameters import ParameterError

np.random.seed(0)
SIZE = 10
test_var = np.random.rand(SIZE)
test_initializer = np.random.rand(SIZE)
test_noise_arr = np.random.rand(SIZE)

RAND0_1 = np.random.random()
RAND2 = np.random.rand()
RAND3 = np.random.rand()


def SimpleIntFun(_init, _value, iterations, noise, **kwargs):
    assert iterations == 3

    if np.isscalar(noise):
        if "initializer" in kwargs:
            return [4.91845218, 4.78766907, 4.73758993, 5.04920442, 4.09842889,
                    4.2909061, 4.05866892, 5.23154257, 5.23413599, 4.86548903]

        else:
            return [4.12672714, 4.25877415, 4.16954537, 4.12360778, 4.02739283,
                    4.2037768, 4.03845052, 4.39892272, 4.45597924, 3.99547688]
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
                    3.8879681, 2.16602771, 5.74284825, 4.47697989, 3.78677378]

        else:
            return [4.7398811, 4.33354877, 3.23128239, 4.14249424, 2.05951504,
                    3.8008388, 2.14580932, 4.9102284, 3.69882314, 2.91676163]


def AdaptiveIntFun(_init, _value, iterations, noise, **kwargs):
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


def DriftIntFun(_init, _value, iterations, noise, **kwargs):
    assert iterations == 3

    if np.isscalar(noise):
        if "initializer" not in kwargs:
            return [[0.53150387, 3.78140754, 4.53709231, 1.01650495, 1.48888893,
                     2.26545636, 2.89486977, 1.3060138, 2.75587927, 1.90759788],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]]

        else:
            return [[1.32322891, 4.31030246, 5.10513687, 1.94210158, 1.55992499,
                     2.35258566, 2.91508817, 2.13863365, 3.53403602, 2.77761003],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]]

    else:
        if "initializer" not in kwargs:
            return [[0.19557944, 3.84081432, 3.4503575, 1.01012678, 1.67172503,
                     2.1987747, 1.93406955, 1.1364648, 2.55292322, 1.79854117],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]]

        else:
            return [[0.98730448, 4.36970924, 4.01840206, 1.93572342, 1.74276108,
                     2.285904, 1.95428795, 1.96908464, 3.33107997, 2.66855331],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]]


def LeakyFun(_init, _value, iterations, noise, **kwargs):
    assert iterations == 3

    if np.isscalar(noise):
        if "initializer" not in kwargs:
            return [2.20813608, 2.25674001, 2.22389663, 2.2069879, 2.17157305, 2.23649656, 2.17564317, 2.30832598,
                    2.32932737, 2.15982541]
        else:
            return [2.93867224, 2.74475902, 2.74803958, 3.06104933, 2.23711905, 2.31689203, 2.19429898, 3.07659637,
                    3.04734388, 2.96259823]

    elif isinstance(noise, pnl.DistributionFunction):
        if "initializer" not in kwargs:
            return [2.55912037, 1.24455938, 1.43417309, 1.638423, 1.91298882, 1.22700281, 1.71226825, 1.67794471,
                    1.20395947, 1.48326449]
        else:
            return [3.28965653, 1.73257839, 1.95831604, 2.49248443, 1.97853482, 1.30739828, 1.73092406, 2.4462151,
                    1.92197598, 2.28603731]

    else:
        if "initializer" not in kwargs:
            return [2.39694798, 2.27976578, 1.9349721, 2.21280371, 1.5655935, 2.11241762, 1.59283164, 2.46577518,
                    2.09617208, 1.82765063]
        else:
            return [3.12748415, 2.76778478, 2.45911505, 3.06686514, 1.6311395, 2.19281309, 1.61148745, 3.23404557,
                    2.81418859, 2.63042344]


def AccumulatorFun(_init, _value, iterations, noise, **kwargs):
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


def DriftOnASphereFun(_init, _value, iterations, noise, **kwargs):
    assert iterations == 3

    if np.isscalar(noise):
        if "initializer" not in kwargs:
            return [-0.015710035765, -0.052577778859, 0.681218795793, 0.110947944152,
                    0.386134081139, -0.266532800135, 0.134110115733, -0.030188100977,
                    0.245868626971, 0.470058912262, -0.013319475244]

        else:
            return [-1.32269048e-01, 4.35051787e-05, 3.87398441e-05, -3.95620568e-06,
                    1.27324586e-04, -5.01625256e-04, -8.37794371e-04, 1.25048720e-01,
                    7.47570336e-01, -6.52303943e-01, -6.57270465e-05]

    else:
        if "initializer" not in kwargs:
            return [0.23690849474294814, 0.0014011543771184686, 0.0020071969614023914, -0.0012806262650772564,
                    -0.0009626666466757963, -0.016204753263919822, -0.026448355473615546, 0.4609067174067295,
                    0.828755706263852, -0.3158426068946889, -0.0013253357638719173]

        else:
            return [-3.72900858e-03, -3.38148799e-04, -6.43154678e-04, 4.36274120e-05,
                    6.67038983e-04, -2.87440868e-03, -2.08163440e-03, 4.41976901e-01,
                    5.31162110e-01, -7.22848147e-01, 4.66808385e-04]


GROUP_PREFIX = "IntegratorFunction "


@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("variable, params", [
    (test_var, {'rate': RAND0_1, 'offset': RAND3}),
    (test_var, {'initializer': test_initializer, 'rate': RAND0_1, 'offset': RAND3}),
], ids=["Default", "Initializer"])
@pytest.mark.parametrize("noise", [RAND2, test_noise_arr, pnl.NormalDist],
                         ids=["SNOISE", "VNOISE", "FNOISE"])
@pytest.mark.parametrize("func", [
    (pnl.AdaptiveIntegrator, AdaptiveIntFun, {}),
    (pnl.SimpleIntegrator, SimpleIntFun, {}),
    (pnl.DriftDiffusionIntegrator, DriftIntFun, {'time_step_size': 1.0}),
    (pnl.LeakyCompetingIntegrator, LeakyFun, {}),
    (pnl.AccumulatorIntegrator, AccumulatorFun, {'increment': RAND0_1}),
], ids=lambda x: x[0])
@pytest.mark.parametrize("mode", ["test_execution", "test_reset"])
@pytest.mark.benchmark
def test_execute(func, func_mode, variable, noise, params, mode, benchmark):
    func_class, func_res, func_params = func
    benchmark.group = GROUP_PREFIX + func_class.componentName + " " + mode

    if callable(noise):
        if issubclass(func_class, (pnl.DriftDiffusionIntegrator, pnl.DriftOnASphereIntegrator)):
            pytest.skip("{} doesn't support functional noise".format(func_class.componentName))

        # Instantiate the noise Function using explicit seed
        noise = noise(seed=0)

    params = {**params, **func_params}

    if issubclass(func_class, pnl.AccumulatorIntegrator) \
            or issubclass(func_class, pnl.DriftOnASphereIntegrator):
        params.pop('offset', None)

    f = func_class(default_variable=variable, noise=noise, **params)
    ex = pytest.helpers.get_func_execution(f, func_mode)

    # Execute few times to update the internal state
    ex(variable)
    ex(variable)

    if mode == "test_execution":
        res = benchmark(ex, variable)

        expected = func_res(f.initializer, variable, 3, noise, **params)

        tolerance = {} if pytest.helpers.llvm_current_fp_precision() == 'fp64' else {'rtol': 1e-5, 'atol': 1e-8}
        np.testing.assert_allclose(res, expected, **tolerance)

    elif mode == "test_reset":
        ex_res = pytest.helpers.get_func_execution(f, func_mode, tags=frozenset({'reset'}), member='reset')

        # Compiled mode ignores input variable, but python uses it if it's provided
        post_reset = benchmark(ex_res, None if func_mode == "Python" else variable)

        # Python implementations return 2d arrays,
        # while most compiled variants return 1d
        if func_mode != "Python":
            post_reset = np.atleast_2d(post_reset)

        # The order in which the reinitialized values are returned
        # is hardcoded in kwargs of the reset() methods of the respective
        # Function classes. The first one is 'initializer' in all cases.
        # The other ones are reset to 0 in the test cases.
        reset_expected = np.zeros_like(post_reset)
        reset_expected[0] = f.parameters.initializer.get()

        np.testing.assert_allclose(post_reset, reset_expected)

    else:
        assert False, "Unknown test mode: {}".format(mode)


def test_integrator_function_no_default_variable_and_params_len_more_than_1():
    I = Functions.AdaptiveIntegrator(rate=[.1, .2, .3])
    I.defaults.variable = np.array([0, 0, 0])


def test_integrator_function_default_variable_len_1_but_user_specified_and_params_len_more_than_1():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[1], rate=[.1, .2, .3])
    error_msg_a = 'The length of the array specified for the rate parameter'
    error_msg_b = 'must match the length of the default input'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)


def test_integrator_function_default_variable_and_params_len_more_than_1_error():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[0, 0], rate=[.1, .2, .3])
    error_msg_a = 'The length of the array specified for the rate parameter'
    error_msg_b = 'must match the length of the default input'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)


def test_integrator_function_with_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(rate=[.1, .2, .3], offset=[.4, .5])
    error_msg_a = "The parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "(['offset', 'rate']) don't all have the same length"
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)


def test_integrator_function_with_default_variable_and_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[0, 0, 0], rate=[.1, .2, .3], offset=[.4, .5])
    error_msg_a = "The following parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "don't have the same length as its 'default_variable' (3): ['offset']."
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)


@pytest.mark.parametrize("dim", [3, 4, 7])
def test_drift_on_a_sphere_initializer_rules(dim):
    # scalar initializer → always invalid
    with pytest.raises(FunctionError) as err:
        Functions.DriftOnASphereIntegrator(dimension=dim, initializer=0.1)
    assert f"list or 1d array of length {dim}" in str(err.value)

    # wrong length → invalid
    with pytest.raises(FunctionError):
        Functions.DriftOnASphereIntegrator(dimension=dim, initializer=[0.1] * (dim - 1))

    # zero vector → invalid
    with pytest.raises(FunctionError):
        Functions.DriftOnASphereIntegrator(dimension=dim, initializer=[0.0] * dim)

    # correct length → OK
    f = Functions.DriftOnASphereIntegrator(dimension=dim, initializer=[1.0] + [0.0] * (dim - 1))
    assert np.isclose(np.linalg.norm(f.parameters.previous_value.get(None)), 1.0)


@pytest.mark.parametrize("dim", [3, 4, 7])
def test_drift_on_a_sphere_noise_rules(dim):
    # scalar noise → always OK
    Functions.DriftOnASphereIntegrator(dimension=dim, noise=0.01)

    # wrong length (too short or too long) → error
    wrong_lengths = [dim - 2, dim]  # tangent length should be dim-1
    for L in wrong_lengths:
        with pytest.raises(FunctionError) as err:
            Functions.DriftOnASphereIntegrator(dimension=dim, noise=[0.1] * L)
        assert f"list or 1d array of length {dim - 1}" in str(err.value)

    # correct length → OK
    Functions.DriftOnASphereIntegrator(dimension=dim, noise=[0.1] * (dim - 1))


@pytest.mark.parametrize("dim", [3, 5, 7])
def test_drift_on_a_sphere_dimension_inference(dim):
    # Infer from initializer (full vector) ---
    init = np.zeros(dim)
    init[0] = 1.0
    f = Functions.DriftOnASphereIntegrator(initializer=init)
    assert f.parameters.dimension.get(None) == dim

    # All zero initializer → error
    theta = np.zeros(dim)
    with pytest.raises(FunctionError):
        Functions.DriftOnASphereIntegrator(initializer=theta)

    # Infer from noise vector (anisotropic, length dim-1) ---
    noise = [0.1] * (dim - 1)
    f = Functions.DriftOnASphereIntegrator(noise=noise)
    assert f.parameters.dimension.get(None) == dim

    # Here it is allowed to have zero noise in all directions
    noise = [0] * (dim - 1)
    f = Functions.DriftOnASphereIntegrator(noise=noise)
    assert f.parameters.dimension.get(None) == dim

    # Infer from default_variable (full vector) ---
    dv = np.zeros(dim)
    dv[0] = 1.0
    f = Functions.DriftOnASphereIntegrator(default_variable=dv)
    assert f.parameters.dimension.get(None) == dim + 1

    # Scalar initializer gives no dimension info → error
    with pytest.raises(FunctionError):
        Functions.DriftOnASphereIntegrator(initializer=0.5)

    # Scalar noise gives no dimension info → default
    f = Functions.DriftOnASphereIntegrator(noise=0.5)
    assert f.parameters.dimension.get(None) == 3  # defaults to 3


@pytest.mark.parametrize("dim", [3, 4, 7])
def test_drift_on_a_sphere_combined_rules(dim):
    good_init = [1.0] + [0.0] * (dim - 1)
    good_noise = [0.1] * (dim - 1)
    bad_init = [1.0] * (dim - 2)
    bad_noise = [0.1] * (dim - 2)

    # both OK → OK
    Functions.DriftOnASphereIntegrator(dimension=dim, initializer=good_init, noise=good_noise)

    # bad init → throws before noise matters
    with pytest.raises(FunctionError):
        Functions.DriftOnASphereIntegrator(dimension=dim, initializer=bad_init, noise=good_noise)

    # bad noise → init OK but noise fails
    with pytest.raises(FunctionError):
        Functions.DriftOnASphereIntegrator(dimension=dim, initializer=good_init, noise=bad_noise)


@pytest.mark.parametrize("dim", [3, 5, 7])
def test_drift_on_sphere(dim):
    """
    Tests DriftOnASphereIntegrator for correct deterministic and stochastic behavior:

    (1) With noise=0 and scalar drift input, motion follows a great-circle with constant angular step.
    (2) With rate=0 and noise>0, motion is stochastic but always remains on the sphere.
    (3) Drift direction used is the internal transported drift_dir, and motion is tangent.
    """

    drift_strength = 0.1
    dt = 0.01
    n_steps = 60

    # --- Deterministic drift case ---
    init = np.zeros(dim, float)
    init[0] = 1.0

    f = Functions.DriftOnASphereIntegrator(
        dimension=dim,
        initializer=init,
        noise=0.0,
        rate=.1,
        time_step_size=dt,
        seed=123,
    )

    xs = [f(variable=drift_strength)]
    for _ in range(n_steps):
        xs.append(f(variable=drift_strength))
    xs = np.stack(xs, axis=0)

    # (1) Always remain on the sphere
    norms = np.linalg.norm(xs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # (2) Angular step is constant
    cosθ = np.sum(xs[1:] * xs[:-1], axis=1)
    thetas = np.arccos(np.clip(cosθ, -1.0, 1.0))
    tol = thetas[0] * 1e-6
    assert np.allclose(thetas, thetas[0], atol=tol)

    # (3) Drift direction matches internally transported drift_dir
    context = f.most_recent_context or None
    d0 = f.parameters.drift_dir.get(context)
    d0 /= np.linalg.norm(d0)

    first_step = xs[1] - xs[0]
    assert np.isclose(first_step @ xs[0], 0.0, atol=1e-6)  # tangent
    step_dir = first_step / np.linalg.norm(first_step)
    assert np.isclose(step_dir @ d0, 1.0, atol=5e-2)

    # --- Stochastic case ---
    init2 = np.zeros(dim, float)
    init2[1] = 1.0

    f = Functions.DriftOnASphereIntegrator(
        dimension=dim,
        initializer=init2,
        noise=1.0,
        rate=0.0,
        time_step_size=dt,
        seed=999,
    )

    xs = [f(variable=0.0)]
    for _ in range(n_steps):
        xs.append(f(variable=0.0))
    xs = np.stack(xs, axis=0)

    norms = np.linalg.norm(xs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    sq_dists = np.sum((xs[1:] - xs[:-1]) ** 2, axis=1)
    assert not np.allclose(sq_dists, 0.0, atol=1e-6)


@pytest.mark.parametrize("dim", [3, 5, 7])
def test_drift_on_sphere_reset(dim):
    init = np.zeros(dim)
    init[0] = 1.0
    f = Functions.DriftOnASphereIntegrator(dimension=dim, initializer=init, noise=0.0)
    f(variable=0.1)
    f(variable=0.1)
    reset_out = f.reset()
    assert np.allclose(reset_out[0], init / np.linalg.norm(init))
    assert np.allclose(np.linalg.norm(reset_out[0]), 1.0)


@pytest.mark.parametrize("dim", [3, 5])
def test_target_mode_moves_toward_target(dim):
    init = np.zeros(dim)
    init[0] = 1.0
    target = np.zeros(dim)
    target[1] = 1.0  # orthogonal pole

    f = Functions.DriftOnASphereIntegrator(
        dimension=dim,
        initializer=init,
        rate=0.1,
        noise=0.0,
        input_space="target",
    )

    x1 = f(variable=target)
    assert np.linalg.norm(x1) == pytest.approx(1.0, abs=1e-6)

    # angle to target decreases
    ang_before = np.arccos(np.clip(init @ target, -1, 1))
    ang_after = np.arccos(np.clip(x1 @ target, -1, 1))
    assert ang_after < ang_before


@pytest.mark.parametrize("start", [[0, 0, 1], [.5, .5, .70710678]])
@pytest.mark.parametrize("end", [[0, 0, 1], [0, .5, -.5]])
def test_target_mode_rate1_dt1_reaches_target(start, end):
    target = np.array(end, dtype=float)

    integ = Functions.DriftOnASphereIntegrator(
        dimension=len(start),
        initializer=np.array(start, dtype=float),
        input_space="target",
        rate=1.0,
        noise=0.0,
        time_step_size=1.0,
    )

    # One step toward target
    x1 = integ.execute(variable=target)

    # Normalize both to be safe
    x1 /= np.linalg.norm(x1)
    target /= np.linalg.norm(target)

    assert np.allclose(x1, target, atol=1e-6), f"Expected to reach target, got {x1}"


def test_target_mode_length_mismatch_error():
    f = Functions.DriftOnASphereIntegrator(dimension=4)
    wrong = np.zeros(10)
    with pytest.raises(Exception):
        f(variable=wrong)
