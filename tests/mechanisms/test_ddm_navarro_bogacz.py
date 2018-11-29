import numpy as np
import pytest
import typecheck
from math import isclose

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.function import BogaczEtAl, NavarroAndFuss, DriftDiffusionIntegrator, FunctionError, NormalDist
from psyneulink.core.components.process import Process
from psyneulink.core.components.system import System

from psyneulink.core.scheduling.condition import Never, WhenFinished
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.mechanisms.processing.integrator.ddm import ARRAY, DDM, DDMError, SELECTED_INPUT_ARRAY

def test_nf_vs_bogacz():

    NF = DDM(
        name='DDM',
        function=NavarroAndFuss()
    )

    B = DDM(
        name='DDM',
        function=BogaczEtAl(shenhav_et_al_compat_mode=True)
    )

    NUM_CHECKS = 1000
    rng = np.random.RandomState(100)
    for i in range(NUM_CHECKS):
        r_stim = rng.uniform(-5, 5)
        r_drift_rate = rng.uniform(-5,5)
        r_threshold = rng.uniform(0, 10)
        r_starting_point = rng.uniform(-r_threshold, r_threshold)
        r_bias = (r_starting_point + r_threshold) / (2 * r_threshold)
        r_t0 = rng.uniform(0, 3)
        r_noise = rng.uniform(0,1)

        print("\n{} of {}".format(i, NUM_CHECKS))
        print("\tr_stim = {}".format(r_stim))
        print("\tr_drift_rate = {}".format(r_drift_rate))
        print("\tr_threshold = {}".format(r_threshold))
        print("\tr_starting_point = {}".format(r_starting_point))
        print("\tr_bias = {}".format(r_bias))
        print("\tr_noise = {}".format(r_noise))
        print("\tr_t0 = {}".format(r_t0))
        print("ddm_params.z = {threshold}; ddm_params.c = {noise}; ddm_params.T0 = {t0}; [meanERs,meanRTs,meanDTs,condRTs,condVarRTs,condSkewRTs] = ddmSimFRG({drift_rate},{bias}, ddm_params, 1)".format(
                threshold=r_threshold, noise=r_noise, bias=r_bias, drift_rate=r_stim * r_drift_rate, t0=r_t0))
        NF.function_object.drift_rate=r_stim*r_drift_rate
        NF.function_object.threshold=r_threshold
        NF.function_object.starting_point=r_bias
        NF.function_object.t0=r_t0
        NF.function_object.noise=r_noise

        B.function_object.drift_rate = r_drift_rate
        B.function_object.threshold = r_threshold
        B.function_object.starting_point = r_starting_point
        B.function_object.t0 = r_t0
        B.function_object.noise = r_noise

        results_nf = NF.execute(r_stim)
        results_b = B.execute(r_stim)

        ABS_TOL = 1e-10
        assert np.isclose(results_b[1], results_nf[1], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[2], results_nf[2], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[3], results_nf[3], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[4], results_nf[4], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[5], results_nf[5], atol=ABS_TOL, equal_nan=True)
