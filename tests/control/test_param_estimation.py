import pytest

import numpy as np
import scipy.stats

from psyneulink.core.compositions import Composition
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.globals.sampleiterator import SampleSpec
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import OptimizationControlMechanism
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.globals.keywords import OVERRIDE
from psyneulink.core.components.functions.optimizationfunctions import ParamEstimationFunction, GridSearch, MINIMIZE

@pytest.mark.parametrize("mode", ['elfi', 'GridSearch'])
def test_moving_average(mode):

    # Set an arbitrary seed and a global random state to keep the randomly generated quantities the same between runs
    seed = 20170530  # this will be separately given to ELFI
    np.random.seed(seed)

    # true parameters
    t1_true = 0.6
    t2_true = 0.2

    # Define a function that simulates a 2nd order moving average, assuming mean zero:
    # y_t = w_t + t1*w_t-1 + t2*w_t-2
    # where t1 and t2 are real and w_k is i.i.d sequence white noise with N(0,1)
    def MA2(input=[0], t1=0.5, t2=0.5, n_obs=100, batch_size=1, random_state=None):
        # FIXME: Convert arguments to scalar if they are not. Why is this nescessary?
        # PsyNeuLink, when creating a user defined function, seems to expect the function
        # to support inputs of type np.ndarray even when they are only allowed to be
        # scalars.
        n_obs = n_obs[0] if (type(n_obs) is np.ndarray) else n_obs
        batch_size = batch_size[0] if (type(batch_size) is np.ndarray) else batch_size

        # Make inputs 2d arrays for numpy broadcasting with w
        t1 = np.asanyarray(t1).reshape((-1, 1))
        t2 = np.asanyarray(t2).reshape((-1, 1))
        random_state = random_state or np.random

        w = random_state.randn(int(batch_size), int(n_obs) + 2)  # i.i.d. sequence ~ N(0,1)
        x = w[:, 2:] + t1 * w[:, 1:-1] + t2 * w[:, :-2]
        return x

    # Lets make some observed data. This will be the data we try to fit parameters for.
    y_obs = MA2(t1=t1_true, t2=t2_true)

    # Make a processing mechanism out of our simulator.
    ma_mech = ProcessingMechanism(function=MA2,
                                      size=1,
                                      name='Moving Average (2nd Order)')

    # Now lets add it to a composition
    comp = Composition(name="Moving_Average")
    comp.add_node(ma_mech)

    # Now lets setup some control signals for the parameters we want to
    # infer. This is where we would like to specify priors.
    signalSearchRange = SampleSpec(start=0.1, stop=2.0, step=0.2)
    t1_control_signal = ControlSignal(projections=[('t1', ma_mech)],
                                      allocation_samples=signalSearchRange,
                                      cost_options=[],
                                      modulation=OVERRIDE)
    t2_control_signal = ControlSignal(projections=[('t2', ma_mech)],
                                      allocation_samples=signalSearchRange,
                                      cost_options=[],
                                      modulation=OVERRIDE)

    # A function to calculate the auto-covariance with specific lag for a
    # time series. We will use this function to compute the summary statistics
    # for generated and observed data so that we can compute a metric between the
    # two. In PsyNeuLink terms, this will be part of an ObjectiveMechanism.
    def autocov(agent_rep, x=None, lag=1):
        if x is None:
            return np.asarray(0.0)

        C = np.mean(x[:, lag:] * x[:, :-lag], axis=1)
        return C

    # # Lets make one function that computes all the summary stats in one go because PsyNeuLink
    # # objective mechanism expect a single function.
    # def objective_function(x):
    #     return np.concatenate((autocov(x), autocov(x, lag=2)))
    #
    # # Objective Mechanism and its function currently need to be specified in the script.
    # # (In future versions, this will be set up automatically)
    # objective_mech = ObjectiveMechanism(function=objective_function,
    #                                         monitor=[ma_mech])

    # Setup the controller with the ParamEstimationFunction
    if mode == 'elfi':
        comp.add_controller(
            controller=OptimizationControlMechanism(
                agent_rep=comp,
                function=ParamEstimationFunction(
                    priors={'t1': (scipy.stats.uniform, 0, 2),
                            't2': (scipy.stats.uniform, 0, 2)},
                    observed=y_obs,
                    summary=[(autocov, 1), (autocov, 2)],
                    discrepancy='euclidean',
                    n_samples=3, quantile=0.01, # Set very small now cause things are slow.
                    seed=seed),
                objective_mechanism=False,
                control_signals=[t1_control_signal, t2_control_signal]))
    elif mode == 'GridSearch':
        observed_C = np.array([autocov(None, y_obs, 1), autocov(None, y_obs, 2)])
        def objective_f(val):
            C = np.array([autocov(None, val, 1), autocov(None, val, 2)])
            ret = np.linalg.norm(C - observed_C)
            return ret

        objective_mech = ObjectiveMechanism(function=objective_f,
                                            size=len(y_obs[0]),
                                            monitor=[ma_mech],
                                            name='autocov - observed autocov')
        comp.add_controller(
            controller=OptimizationControlMechanism(
                agent_rep=comp,
                function=GridSearch(save_values=True, direction=MINIMIZE),
                objective_mechanism=objective_mech,
                control_signals=[t1_control_signal, t2_control_signal]))

    comp.disable_all_history()

    # Lets setup some input to the mechanism, not that it uses it for anything.
    stim_list_dict = {ma_mech: [0]}

    # # FIXME: Show graph fails when the controller doesn't have an objective mechanism.
    # comp.show_graph(show_controller=True,
    #                 show_projection_labels=True,
    #                 show_node_structure=True,
    #                 show_cim=True,
    #                 show_dimensions=True)

    comp.run(inputs=stim_list_dict)

    if mode == 'elfi':
        assert np.allclose(comp.controller.value, [[0.5314349], [0.19140103]])
    if mode == 'GridSearch':
        assert np.allclose(comp.controller.value, [[0.5], [0.3]])
