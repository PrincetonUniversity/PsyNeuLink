import numpy as np

from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.scheduling.condition import AfterTrial, Any, AtTrial
from psyneulink.core.globals.keywords import INPUT_PORT_PARAMS, FUNCTION_PARAMS

class TestMechanismRuntimeParams:

    def test_mechanism_execute_mechanism_runtime_param(self):

        # Construction
        T = TransferMechanism()
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # runtime param used for noise
        T.execute(runtime_params={"noise": 10.0}, input=2.0)
        assert T.value == 12.0

        # defalut values are restored
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0
        T.execute(input=2.0)
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0
        assert T.value == 2.0

    def test_mechanism_execute_function_runtime_param(self):

        # Construction
        T = TransferMechanism()
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.value == 20.0

        # defalut values are restored
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        T.execute(input=2.0)
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        assert T.value == 2.0

    def test_runtime_params_use_and_reset_not_affect_other_assigned_vals(self):

        T = TransferMechanism()

        # Intercept attr assigned
        T.function.intercept = 2.0
        assert T.function.intercept == 2.0

        # runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        # Assigned intercept and runtime_param for slope are used:
        assert T.value == 22.0

        # slope restored to default, but intercept retains assigned value
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        assert T.function.intercept == 2.0

        # previous runtime_param for slope not used again
        T.execute(input=2.0)
        assert T.value == 4.0
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

    def test_runtime_params_reset_to_previously_assigned_val(self):

        # Construction
        T = TransferMechanism()
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # set slope directly
        T.function.slope = 2.0
        assert T.function.slope == 2.0

        # runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.value == 20.0

        # slope restored to previously assigned value
        assert T.function.slope == 2.0
        assert T.parameter_ports['slope'].value == 2.0
        T.execute(input=2.0)
        assert T.value == 4.0
        assert T.function.slope == 2.0

class TestCompositionRuntimeParams:

    def test_composition_run_mechanism_runtime_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # runtime param used for noise
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": 10.0}})
        assert T.parameters.value.get(C.default_execution_id) == 12.0
        # noise restored to default
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].parameters.value.get(C) == 0.0

        # previous runtime_param for noise not used again
        C.run(inputs={T: 2.0}, )
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].parameters.value.get(C) == 0.0
        assert T.parameters.value.get(C.default_execution_id) == 2.0

    def test_composition_run_function_runtime_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        C.run(inputs={T: 2.0}, runtime_params={T: {"slope": 10.0}})
        # runtime param used for slope
        assert T.parameters.value.get(C.default_execution_id) == 20.0
        # slope restored to default
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].parameters.value.get(C) == 1.0

        # previous runtime_param for slope not used again
        C.run(inputs={T: 2.0})
        assert T.parameters.value.get(C.default_execution_id) == 2.0
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].parameters.value.get(C) == 1.0

    def test_composition_run_mechanism_inputport_runtime_param_no_condition(self):

        # Construction
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        C = Composition(pathways=[T1,T2])

        T1.function.slope = 5
        T2.input_port.function.scale = 4
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T1: {'slope': 3},                         # Mechanism's function (Linear) parameter
                  T2: {
                      'noise': 0.5,                        # Mechanism's parameter
                      # 'glorp': 22,  # FIX: MECHANISM PARAM STILL PASSES BUT SHOULDN'T
                      'intercept': 1,                       # Mechanism's function parameter
                      # FIX: WHAT ABOUT PROJECTION PARAMS?
                      INPUT_PORT_PARAMS: {
                          'weight':5,                      # InputPort's parameter
                          'scale':20,                      # InputPort's function (LinearCombination) parameter
                          # 'trigot':16,  # THIS FAILS
                          FUNCTION_PARAMS:{'weights':10,
                                           # 'flurb': 12,   # FIX: FUNCTION PARAM STILL PASSES BUT SHOULDN'T
                                           }}  # InputPort's function (LinearCombination) parameter
                  }
              })
        assert T2.parameters.value.get(C.default_execution_id) == [1201.5]

        # all parameters restored to previous values (assigned or defaults)
        assert T1.function.parameters.slope.get(C) == 5.0
        assert T1.parameter_ports['slope'].parameters.value.get(C) == 5.0
        assert T2.parameters.noise.get(C) == 0.0
        assert T2.parameter_ports['noise'].parameters.value.get(C) == 0.0
        assert T2.function.intercept == 0.0
        assert T2.function.parameters.intercept.get(C) == 0.0
        assert T2.input_port.weight == None
        assert T2.input_port.function.scale == 4.0
        assert T2.input_port.function.parameters.scale.get(C) == 4.0
        assert T2.input_port.function.weights == None
        assert T2.input_port.function.parameters.weights.get(C) == None

        C.run(inputs={T1: 2.0}, )
        assert C.results == [[[1201.5]], [[40.]]]
        assert T1.function.slope == 5.0
        assert T1.parameter_ports['slope'].parameters.value.get(C) == 5.0
        assert T2.input_port.function.parameters.scale.get(C.default_execution_id) == 4.0

    def test_composition_run_mechanism_runtime_param_with_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # run with runtime param used for noise only on trial 1
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AtTrial(1))}},
              # scheduler=S,
              num_trials=4)
        # noise restored to default
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].parameters.value.get(C) == 0.0

        # run again to insure restored default for noise after last run
        C.run(inputs={T: 2.0})

        # results reflect runtime_param used for noise only on trial 1
        assert np.allclose(C.results, [np.array([[2.]]),     # Trial 0 - condition not satisfied yet
                                       np.array([[12.]]),    # Trial 1 - condition satisfied
                                       np.array([[2.]]),     # Trial 2 - condition no longer satisfied (not sticky)
                                       np.array([[2.]]),     # Trial 3 - condition no longer satisfied (not sticky)
                                       np.array([[2.]])])    # New run (runtime param no longer applies)

    def test_composition_run_mechanism_runtime_param_with_sticky_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # run with runtime param used for noise after trial 1 (i.e., trials 2 and 3)
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AfterTrial(1))}},
              num_trials=4)
        # noise restored to default
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].parameters.value.get(C) == 0.0
        # run again to insure restored default for noise after last run
        C.run(inputs={T: 2.0})

        # results reflect runtime_param used for noise only on trials 2 and 3
        assert np.allclose(C.results, [np.array([[2.]]),      # Trial 0 - condition not satisfied yet
                                       np.array([[2.]]),      # Trial 1 - condition not satisfied yet
                                       np.array([[12.]]),     # Trial 2 - condition satisfied
                                       np.array([[12.]]),     # Trial 3 - condition satisfied (sticky)
                                       np.array([[2.]])])     # New run (runtime param no longer applies)

    def test_composition_run_mechanism_runtime_param_with_combined_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        # run with runtime param used for noise only on trial 1 and after 2 (i.e., 3 and 4)
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, Any(AtTrial(1), AfterTrial(2)))}},
              num_trials=5)
        # noise restored to default
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].parameters.value.get(C) == 0.0

        # run again to insure restored default for noise after last run
        C.run(inputs={T: 2.0})

        # results reflect runtime_param used for noise only on trials 1, 3 and 4
        assert np.allclose(C.results,[np.array([[2.]]),      # Trial 0 - NOT condition 0, NOT condition 1
                                      np.array([[12.]]),     # Trial 1 - condition 0, NOT condition 1
                                      np.array([[2.]]),      # Trial 2 - NOT condition 0, NOT condition 1
                                      np.array([[12.]]),     # Trial 3 - NOT condition 0, condition 1
                                      np.array([[12.]]),     # Trial 4 - NOT condition 0, condition 1
                                      np.array([[2.]])])     # New run (runtime param no longer applies)

    def test_composition_run_function_runtime_param_with_combined_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # run with runtime param used for slope only on trial 1 and after 2 (i.e., 3 and 4)
        C.run(inputs={T: 2.0},
              runtime_params={T: {"slope": (10.0, Any(AtTrial(1), AfterTrial(2)))}},
              num_trials=5)
        # slope restored to default
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # run again to insure restored default for slope after last run
        C.run(inputs={T: 2.0})

        # results reflect runtime_param used for slope only on trials 1, 3 and 4
        assert np.allclose(C.results,[np.array([[2.]]),      # Trial 0 - NOT condition 0, NOT condition 1
                                      np.array([[20.]]),     # Trial 1 - condition 0, NOT condition 1
                                      np.array([[2.]]),      # Trial 2 - NOT condition 0, NOT condition 1
                                      np.array([[20.]]),     # Trial 3 - NOT condition 0, condition 1
                                      np.array([[20.]]),     # Trial 4 - NOT condition 0, condition 1
                                      np.array([[2.]])])     # New run (runtime param no longer applies)

