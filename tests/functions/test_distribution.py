import numpy as np
import pytest
import sys

import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.nonstateful.distributionfunctions as Functions
from psyneulink.core.globals.utilities import _SeededPhilox

np.random.seed(0)
test_var = np.random.rand()

RAND1 = np.random.rand()
RAND2 = np.random.rand()
RAND3 = np.random.rand()
RAND4 = np.random.rand()
RAND5 = np.random.rand()

dda_expected_default = (1.9774974807292212, 0.012242689689501842, 1.9774974807292207, 1.3147677945132479, 1.7929299891370192, 1.9774974807292207, 1.3147677945132479, 1.7929299891370192)
dda_expected_random = (0.4236547993389047, -2.7755575615628914e-17, 0.5173675420165031, 0.06942854144616283, 6.302631815990666, 1.4934079600147951, 0.4288991185241868, 1.7740760781361433)
dda_expected_negative = (0.42365479933890504, 0.0, 0.5173675420165031, 0.06942854144616283, 6.302631815990666, 1.4934079600147951, 0.4288991185241868, 1.7740760781361433)
dda_expected_small = (0.5828813465336954, 0.04801236718458773,
                      0.532471083815943, 0.09633801362499317, 6.111833139205608,
                      1.5821207676710864, 0.5392724012504414, 1.8065252817609618)
# Different libm implementations produce slightly different results
if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
    dda_expected_small = (0.5828813465336954, 0.04801236718458773,
                          0.5324710838150166, 0.09633802135385469, 6.119380538293901,
                          1.58212076767016, 0.5392724012504414, 1.8065252817609618)

normal_expected_mt = (1.0890232855122397)
uniform_expected_mt = (0.6879771504250405)
normal_expected_philox = (0.5910357654927911)
uniform_expected_philox = (0.6043448764869507)

llvm_expected = {}
llvm_expected[dda_expected_small] = (0.5828813465336954, 0.04801236718458773,
                                     0.5324710838085324, 0.09633787836991654, 6.0158766570416775,
                                     1.5821207675877176, 0.5392731045768397, 1.8434859117411773)

test_data = [
    pytest.param(Functions.DriftDiffusionAnalytical, test_var, {}, None,
                 dda_expected_default,
                 id="DriftDiffusionAnalytical-DefaultParameters"),
    pytest.param(Functions.DriftDiffusionAnalytical, test_var,
                 {"drift_rate": RAND1, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, None,
                 dda_expected_random, id="DriftDiffusionAnalytical-RandomParameters"),
    pytest.param(Functions.DriftDiffusionAnalytical, -test_var,
                 {"drift_rate": RAND1, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, None,
                 dda_expected_negative, id="DriftDiffusionAnalytical-NegInput"),
    pytest.param(Functions.DriftDiffusionAnalytical, 1e-4,
                 {"drift_rate": 1e-5, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, "Rounding Errors",
                 dda_expected_small, id="DriftDiffusionAnalytical-SmallDriftRate"),
    pytest.param(Functions.DriftDiffusionAnalytical, -1e-4,
                 {"drift_rate": 1e-5, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, "Rounding Errors",
                 dda_expected_small, id="DriftDiffusionAnalytical-SmallDriftRate-NegInput"),
    pytest.param(Functions.DriftDiffusionAnalytical, 1e-4,
                 {"drift_rate": -1e-5, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, "Rounding Errors",
                 dda_expected_small, id="DriftDiffusionAnalytical-SmallNegDriftRate"),
    # Two tests with different inputs to show that input is ignored.
    pytest.param(Functions.NormalDist, 1e14, {"mean": RAND1, "standard_deviation": RAND2}, None, normal_expected_mt,
                 id="NormalDist"),
    pytest.param(Functions.NormalDist, 1e-4, {"mean": RAND1, "standard_deviation": RAND2}, None, normal_expected_mt,
                 id="NormalDist Small Input"),
    pytest.param(Functions.UniformDist, 1e14, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)}, None,
                 uniform_expected_mt, id="UniformDist"),
    pytest.param(Functions.UniformDist, 1e-4, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)}, None,
                 uniform_expected_mt, id="UniformDist"),
    # Inf inputs select Philox PRNG, test_var should never be inf
    pytest.param(Functions.NormalDist, np.inf, {"mean": RAND1, "standard_deviation": RAND2}, None,
                 normal_expected_philox, id="NormalDist Philox"),
    pytest.param(Functions.NormalDist, -np.inf, {"mean": RAND1, "standard_deviation": RAND2}, None,
                 normal_expected_philox, id="NormalDist Philox"),
    pytest.param(Functions.UniformDist, np.inf, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)}, None,
                 uniform_expected_philox, id="UniformDist Philox"),
    pytest.param(Functions.UniformDist, -np.inf, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)}, None,
                 uniform_expected_philox, id="UniformDist Philox"),
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, llvm_skip, expected", test_data)
def test_execute(func, variable, params, llvm_skip, expected, benchmark, func_mode):
    benchmark.group = "TransferFunction " + func.componentName
    if func_mode != 'Python':
        expected = llvm_expected.get(expected, expected)

    f = func(default_variable=variable, **params)
    if np.isinf(variable):
        f.parameters.random_state.set(_SeededPhilox([0]))

    ex = pytest.helpers.get_func_execution(f, func_mode)
    res = ex(variable)

    assert np.allclose(res, expected)

    if benchmark.enabled:
        benchmark(ex, variable)
