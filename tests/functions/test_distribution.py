import numpy as np
import pytest
import sys

from packaging import version as pversion

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

dda_expected_default = (1.9774974807292212, 0.012242689689501842,
                        1.9774974807292207, 1.3147677945132479, 1.7929299891370192,
                        1.9774974807292207, 1.3147677945132479, 1.7929299891370192)
dda_expected_random = (0.4236547993389047, -2.7755575615628914e-17,
                       0.5173675420165031, 0.06942854144616283, 6.302631815990666,
                       1.4934079600147951, 0.4288991185241868, 1.7740760781361433)
dda_expected_negative = (0.42365479933890504, 0.0,
                         0.5173675420165031, 0.06942854144616283, 6.302631815990666,
                         1.4934079600147951, 0.4288991185241868, 1.7740760781361433)
dda_expected_small = (0.5828813465336954, 0.04801236718458773,
                      0.532471083815943, 0.09633801555720854, 6.1142591416669765,
                      1.5821207676710864, 0.5392724051148722, 1.806647390875747)

# Different libm implementations produce slightly different results
# Numpy 1.22+ uses new/optimized implementation of FP routines
# on processors that support AVX512 since 1.22 [0]
# [0] https://github.com/numpy/numpy/commit/1eff1c543a8f1e9d7ea29182b8c76db5a2efc3c2
if sys.platform.startswith("win") or \
   sys.platform.startswith("darwin") or \
   (pversion.parse(np.version.version) >= pversion.parse('1.22') and pytest.helpers.numpy_uses_avx512()):
    dda_expected_small = (0.5828813465336954, 0.04801236718458773,
                          0.5324710838150166, 0.09633802135385469, 6.117763080882898,
                          1.58212076767016, 0.5392724012504414, 1.8064031532265)

# Numpy 1.23+ reimplemented tanh to roughly follow SVML but not enough to match
# the above results [1,2]
# [1] https://github.com/numpy/numpy/pull/20363
# [2] https://github.com/numpy/numpy/commit/75edab9f7a7d95ecc62efc1b3b92642b6d45762d
elif pversion.parse(np.version.version) >= pversion.parse('1.23'):
    dda_expected_small = (0.5828813465336954, 0.04801236718458773,
                          0.5324710838150166, 0.09633801169277778, 6.111024594252574,
                          1.58212076767016, 0.5392724012504414, 1.8064031532265)

normal_expected_mt = (1.0890232855122397)
uniform_expected_mt = (0.6879771504250405)
normal_expected_philox = (0.5910357654927911)
uniform_expected_philox = (0.6043448764869507)

llvm_expected = {'fp64': {}, 'fp32': {}}

# LLVM computes slightly different results from Python/numpy even for fp64
llvm_expected['fp64'][dda_expected_small] = (0.5828813465336954, 0.04801236718458773,
                                             0.5324710838085324, 0.09633788030213193, 6.0183026674990625,
                                             1.5821207675877176, 0.5392731084412705, 1.843608020219776)

# DDA fp32 results are noticeably different due to lower precision.
llvm_expected['fp32'][dda_expected_default] = (1.9774975776672363, 0.012242687866091728,
                                               1.9774975776672363, 1.3147677183151245, 1.792930245399475,
                                               1.9774975776672363, 1.3147677183151245, 1.792930245399475)
llvm_expected['fp32'][dda_expected_random] = (0.42365485429763794, 0.0,
                                              0.5173675417900085, 0.069428451359272, 6.302595138549805,
                                              1.4934077262878418, 0.42889538407325745, 1.7739042043685913)
llvm_expected['fp32'][dda_expected_negative] = (0.4236549735069275, 5.960464477539063e-08,
                                                0.5173678398132324, 0.06942932307720184, 6.302994251251221,
                                                1.4934080839157104, 0.4288962781429291, 1.7739406824111938)
llvm_expected['fp32'][dda_expected_small] = None

# Philox uses different algorithm for fp32 (consumes fewer bits of entropy)
llvm_expected['fp32'][normal_expected_philox] = (0.5655658841133118)
llvm_expected['fp32'][uniform_expected_philox] = (0.6180108785629272)

test_data = [
    pytest.param(Functions.DriftDiffusionAnalytical, test_var, {}, None, None,
                 dda_expected_default, id="DriftDiffusionAnalytical-DefaultParameters"),
    pytest.param(Functions.DriftDiffusionAnalytical, test_var,
                 {"drift_rate": RAND1, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, None, None,
                 dda_expected_random, id="DriftDiffusionAnalytical-RandomParameters"),
    pytest.param(Functions.DriftDiffusionAnalytical, -test_var,
                 {"drift_rate": RAND1, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, None, None,
                 dda_expected_negative, id="DriftDiffusionAnalytical-NegInput"),
    pytest.param(Functions.DriftDiffusionAnalytical, 1e-4,
                 {"drift_rate": 1e-5, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, None, "Rounding Errors",
                 dda_expected_small, id="DriftDiffusionAnalytical-SmallDriftRate"),
    pytest.param(Functions.DriftDiffusionAnalytical, -1e-4,
                 {"drift_rate": 1e-5, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, None, "Rounding Errors",
                 dda_expected_small, id="DriftDiffusionAnalytical-SmallDriftRate-NegInput"),
    pytest.param(Functions.DriftDiffusionAnalytical, 1e-4,
                 {"drift_rate": -1e-5, "threshold": RAND2, "starting_value": RAND3,
                  "non_decision_time":RAND4, "noise": RAND5}, None, "Rounding Errors",
                 dda_expected_small, id="DriftDiffusionAnalytical-SmallNegDriftRate"),
    # Two tests with different inputs to show that input is ignored.
    pytest.param(Functions.NormalDist, 1e14, {"mean": RAND1, "standard_deviation": RAND2},
                 None, None, normal_expected_mt, id="NormalDist"),
    pytest.param(Functions.NormalDist, 1e-4, {"mean": RAND1, "standard_deviation": RAND2},
                 None, None, normal_expected_mt, id="NormalDist Small Input"),
    pytest.param(Functions.UniformDist, 1e14, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)},
                 None, None, uniform_expected_mt, id="UniformDist"),
    pytest.param(Functions.UniformDist, 1e-4, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)},
                 None, None, uniform_expected_mt, id="UniformDist"),
    # Inf inputs select Philox PRNG, test_var should never be inf
    pytest.param(Functions.NormalDist, 1e14, {"mean": RAND1, "standard_deviation": RAND2},
                 _SeededPhilox, None, normal_expected_philox, id="NormalDist Philox"),
    pytest.param(Functions.NormalDist, 1e-4, {"mean": RAND1, "standard_deviation": RAND2},
                 _SeededPhilox, None, normal_expected_philox, id="NormalDist Philox"),
    pytest.param(Functions.UniformDist, 1e14, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)},
                 _SeededPhilox, None, uniform_expected_philox, id="UniformDist Philox"),
    pytest.param(Functions.UniformDist, 1e-4, {"low": min(RAND1, RAND2), "high": max(RAND1, RAND2)},
                 _SeededPhilox, None, uniform_expected_philox, id="UniformDist Philox"),
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, prng, llvm_skip, expected", test_data)
def test_execute(func, variable, params, prng, llvm_skip, expected, benchmark, func_mode):
    benchmark.group = "TransferFunction " + func.componentName

    if func_mode != 'Python':
        precision = pytest.helpers.llvm_current_fp_precision()
        # PTX needs only one special case, this is not worth adding it to the mechanism above
        if func_mode == "PTX" and precision == 'fp32' and expected is dda_expected_negative:
            expected = (0.4236549735069275, 5.960464477539063e-08,
                        0.5173678398132324, 0.06942932307720184, 6.302994728088379,
                        1.4934064149856567, 0.4288918972015381, 1.7737658023834229)

        expected = llvm_expected.get(precision, {}).get(expected, expected)

    if expected is None:
        pytest.skip(llvm_skip)

    f = func(default_variable=variable, **params)
    if prng is not None:
        f.parameters.random_state.set(prng([0]))

    ex = pytest.helpers.get_func_execution(f, func_mode)
    res = benchmark(ex, variable)

    np.testing.assert_allclose(res, expected)
