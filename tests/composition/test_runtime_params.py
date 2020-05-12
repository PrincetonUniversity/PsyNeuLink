import numpy as np
import pytest

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.scheduling.condition import AfterTrial, Any, AtTrial, Never
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

    def test_mechanism_execute_mechanism_runtime_param_error(self):
        T = TransferMechanism()
        with pytest.raises(ComponentError) as error_text:
            T.execute(runtime_params={"glunfump": 10.0}, input=2.0)
        assert ("Invalid specification in runtime_params arg for TransferMechanism" in error_text.value.error_value and
                "'glunfump'" in error_text.value.error_value)

    # def test_mechanism_execute_mechanism_fuction_runtime_param_errors(self):
    #     # FIX 5/8/20 [JDC]: SHOULD FAIL BUT DOESN'T:
    #     T = TransferMechanism()
    #     with pytest.raises(ComponentError) as error_text:
    #         T.function.execute(runtime_params={"spranit": 23})
    #     assert ("Invalid specification in runtime_params arg for TransferMechanism" in error_text.value.error_value and
    #             "'spranit'" in error_text.value.error_value)

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
                      'intercept': 1,                       # Mechanism's function parameter
                      # FIX 5/8/20 [JDC]: WHAT ABOUT PROJECTION PARAMS?
                      INPUT_PORT_PARAMS: {
                          'weight':5,                      # InputPort's parameter
                          'scale':20,                      # InputPort's function (LinearCombination) parameter
                          FUNCTION_PARAMS:{'weights':10,   # InputPort's function (LinearCombination) parameter
                                           }}
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

    def test_composition_run_mechanism_inputport_runtime_param_errors(self):

        # Construction
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        C = Composition(pathways=[T1,T2])

        T1.function.slope = 5
        T2.input_port.function.scale = 4

        # Bad Mechanism param specified
        with pytest.raises(ComponentError) as error_text:
            C.run(inputs={T1: 2.0},
                  runtime_params={
                      T2: {
                          'noise': 0.5,
                          'glorp': 22,                        # Bad Mechanism arg
                          'intercept': 1,
                          INPUT_PORT_PARAMS: {
                              'weight':5,
                              'scale':20,
                              FUNCTION_PARAMS:{'weights':10}}
                      }
                  })
        assert ("Invalid specification in runtime_params arg for TransferMechanism" in error_text.value.error_value and
                "'glorp'" in error_text.value.error_value)

        with pytest.raises(ComponentError) as error_text:
            C.run(inputs={T1: 2.0},
                  runtime_params={
                      T1: {'slope': 3},
                      T2: {
                          'noise': 0.5,
                          'intercept': 1,
                          INPUT_PORT_PARAMS: {
                              'weight':5,
                              'scale':20,
                              'trigot':16,                    # Bad InputPort arg
                              FUNCTION_PARAMS:{'weights':10,
                                               }}
                      }
                  })
        assert ("Invalid specification in runtime_params arg for InputPort" in error_text.value.error_value and
                "of TransferMechanism" in error_text.value.error_value and "'trigot'" in error_text.value.error_value)

        with pytest.raises(ComponentError) as error_text:
            C.run(inputs={T1: 2.0},
                  runtime_params={
                      T1: {'slope': 3},
                      T2: {
                          'noise': 0.5,
                          'intercept': 1,
                          INPUT_PORT_PARAMS: {
                              'weight':5,
                              'scale':20,
                              FUNCTION_PARAMS:{'weights':10,
                                               'flurb': 12,   # Bad InputPort function arg
                                               }}
                      }
                  })
        assert ("Invalid specification in runtime_params arg for InputPort" in error_text.value.error_value and
                "of TransferMechanism" in error_text.value.error_value and "'flurb'" in error_text.value.error_value)

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

    def test_composition_run_function_runtime_params_with_different_but_overlapping_conditions(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # run with runtime param used for slope only on trial 1 and after 2 (i.e., 3 and 4)
        C.run(inputs={T: 2.0},
              runtime_params={T: {"slope": (10.0, Any(AtTrial(1), AfterTrial(2))),
                                  "intercept": (1.0, AfterTrial(1))}},
              num_trials=4)
        # slope restored to default
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        assert T.function.intercept == 0.0
        assert T.parameter_ports['intercept'].value == 0.0

        # run again to insure restored default for slope after last run
        C.run(inputs={T: 2.0})

        # results reflect runtime_param used for slope only on trials 1, 3 and 4
        assert np.allclose(C.results,[np.array([[2.]]),      # Trial 0 - neither condition met
                                      np.array([[20.]]),     # Trial 1 - slope condition met, intercept not met
                                      np.array([[3.]]),      # Trial 2 - slope condition not met, intercept met
                                      np.array([[21.]]),      # Trial 3 - both conditions met
                                      np.array([[2.]])])     # New run (runtime param no longer applies)

    def test_composition_run_mechanism_runtime_params_with_combined_conditions_for_all_INPUT_PORT_PARAMS(self):
        # Construction
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        C = Composition(pathways=[T1,T2])

        T1.function.slope = 5
        T2.input_port.function.scale = 4
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T1: {'slope': (3, AtTrial(1))},             # Condition on Mechanism's function (Linear) parameter
                  T2: {
                      'noise': 0.5,
                      'intercept': (1, AtTrial(2)),           # Condition on Mechanism's function parameter
                      # FIX 5/8/20 [JDC]: WHAT ABOUT PROJECTION PARAMS?
                      INPUT_PORT_PARAMS: ({
                          'weight':5,
                          'scale':20,
                          FUNCTION_PARAMS:{'weights':10,
                                           }}, AtTrial(3))    # Condition on INPUT_PORT_PARAMS
                  }
              },
              num_trials=4
              )

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

        # run again to insure restored default for noise after last run
        C.run(inputs={T1: 2.0}, )

        assert np.allclose(C.results,[np.array([[40.5]]),   # Trial 0 - no conditions met (2*5*4)+0.5
                                      np.array([[24.5]]),   # Trial 1 - only T1.slope condition met (2*3*4)+0.5
                                      np.array([[41.5]]),   # Trial 2 - only T2.intercept condition met (2*5*4)+1+0.5
                                      np.array([[2000.5]]), # Trial 3 - only T2 INPUT_PORT_PARAMS conditions met
                                                            #               (2*5*20*10) + 0.5
                                      np.array([[40.]])]) # New run - revert to assignments before previous run (2*5*4)

    def test_composition_run_mechanism_runtime_params_with_combined_conditions_for_individual_INPUT_PORT_PARAMS(self):
        # Construction
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        C = Composition(pathways=[T1,T2])

        T1.function.slope = 5
        T2.input_port.function.scale = 4
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T1: {'slope': (3, AtTrial(1))},             # Condition on Mechanism's function (Linear) parameter
                  T2: {
                      'noise': 0.5,
                      'intercept': (1, AtTrial(2)),           # Condition on Mechanism's function parameter
                      # FIX 5/8/20 [JDC]: WHAT ABOUT PROJECTION PARAMS?
                      INPUT_PORT_PARAMS: {
                          'weight':5,
                          # FIX 5/8/20 [JDC] ADD TEST FOR THIS ERROR:
                          # 'scale': (20, AtTrial(3), 3 ),
                          'scale': (20, AtTrial(3)),
                          FUNCTION_PARAMS:{'weights':(10, AtTrial(4))}
                      }
                  },
              },
              num_trials=5
              )
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T2: {
                      'noise': 0.5,
                      INPUT_PORT_PARAMS: ({
                          'scale': (20, AtTrial(0)),
                          FUNCTION_PARAMS:{'weights':(10, AtTrial(1))}
                      }, Never())
                  },
              },
              num_trials=2
              )
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T2: {
                      'noise': 0.5,
                      'intercept': (1, AtTrial(0)),
                      INPUT_PORT_PARAMS: ({
                          'scale': (20, AtTrial(0)),
                          FUNCTION_PARAMS:{'weights':(10, AtTrial(1))}
                      }, AtTrial(1))
                  },
              },
              num_trials=2
              )
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

        # run again to insure restored default for noise after last run
        C.run(inputs={T1: 2.0}, )

        assert np.allclose(C.results,[   # Conditions satisfied:
            np.array([[40.5]]), # Trial 0: no conditions (2*5*4)+0.5
            np.array([[24.5]]), # Trial 1: only T1.slope condition (2*3*4)+0.5
            np.array([[41.5]]), # Trial 2: only T2.intercept condition (2*5*4)+1+0.5
            np.array([[200.5]]),# Trial 3: only T2 scale condition (2*5*20) + 0.5
            np.array([[400.5]]),# Trial 4: only T2.function.weights condition (2*5*4*10)+0.5
            np.array([[40.5]]), # Run 2, Tria1 1: INPUT_PORT_PARAMS Never() takes precedence over scale (2*5*4)+0.5
            np.array([[40.5]]), # Run 2: Trial 2: INPUT_PORT_PARAMS Never() takes precedence over weights (2*5*4)+0.5
            np.array([[41.5]]), # Run 3, Tria1 1: INPUT_PORT_PARAMS AtTrial(1) takes precedence over scale (2*5*4)+1+0.5
            np.array([[400.5]]),# Run 3: Trial 2: INPUT_PORT_PARAMS AtTrial(1) consistent with weights (2*5*4*10)+0.5
            np.array([[40.]])   # Final run: revert to assignments before previous run (2*5*4)
        ])
