import numpy as np
import pytest

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.scheduling.condition import AfterTrial, Any, AtTrial, Never
from psyneulink.core.globals.keywords import CONTROL_PROJECTION_PARAMS, INPUT_PORT_PARAMS, FUNCTION_PARAMS, \
    OVERRIDE, PARAMETER_PORT_PARAMS, MAPPING_PROJECTION_PARAMS, SAMPLE, TARGET
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism

class TestMechanismRuntimeParams:

    def test_mechanism_runtime_param(self):

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

    def test_function_runtime_param(self):

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

    def test_use_and_reset_not_affect_other_assigned_vals(self):

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

    def test_reset_to_previously_assigned_val(self):

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

    def test_runtime_param_error(self):
        T = TransferMechanism()
        with pytest.raises(ComponentError) as error_text:
            T.execute(runtime_params={"glunfump": 10.0}, input=2.0)
        assert ("Invalid specification in runtime_params arg for TransferMechanism" in error_text.value.error_value and
                "'glunfump'" in error_text.value.error_value)

    # FIX 5/8/20 [JDC]:  ADDD TEST FOR INVALID FUNCTION PARAM
    # def test_mechanism_execute_mechanism_fuction_runtime_param_errors(self):
    #     # FIX 5/8/20 [JDC]: SHOULD FAIL BUT DOESN'T:
    #     T = TransferMechanism()
    #     with pytest.raises(ComponentError) as error_text:
    #         T.function.execute(runtime_params={"spranit": 23})
    #     assert ("Invalid specification in runtime_params arg for TransferMechanism" in error_text.value.error_value and
    #             "'spranit'" in error_text.value.error_value)

class TestCompositionRuntimeParams:

    def test_mechanism_param_no_condition(self):

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

    def test_function_param_no_condition(self):

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

    def test_input_port_param_no_condition(self):

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

    # FIX 5/8/20 [JDC]: ADD TESTS FOR PARAMETERPORTS AND OUTPUTPORTS

    def test_mechanism_param_with_AtTrial_condition(self):

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

    def test_mechanism_param_with_AfterTrial_condition(self):

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

    def test_mechanism_param_with_combined_condition(self):

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

    def test_function_param_with_combined_condition(self):

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

    def test_function_params_with_different_but_overlapping_conditions(self):

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

    def test_mechanism_params_with_combined_conditions_for_all_INPUT_PORT_PARAMS(self):
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

    def test_mechanism_params_with_combined_conditions_for_individual_INPUT_PORT_PARAMS(self):
        # Construction
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        P = MappingProjection(sender=T1, receiver=T2, name='MY PROJECTION')
        C = Composition(pathways=[[T1,P,T2]])

        T1.function.slope = 5
        T2.input_port.function.scale = 4
        # Run 1: Test INPUT_PORT_PARAMS for InputPort function directly (scale) and in FUNCTION_PARAMS dict (weights)
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T1: {'slope': (3, AtTrial(1))},             # Condition on Mechanism's function (Linear) parameter
                  T2: {
                      'noise': 0.5,
                      'intercept': (1, AtTrial(2)),           # Condition on Mechanism's function parameter
                      INPUT_PORT_PARAMS: {
                          'weight':5,
                          # FIX 5/8/20 [JDC] ADD TEST FOR THIS ERROR:
                          # 'scale': (20, AtTrial(3), 3 ),
                          'scale': (20, AtTrial(3)),
                          FUNCTION_PARAMS:{'weights':(10, AtTrial(4))},
                      }
                  },
              },
              num_trials=5
              )
        # Run 2:  Test INPUT_PORT_PARAMS override by Never() Condition
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
        # Run 3:  Test INPUT_PORT_PARAMS constraint to Trial 1 assignements
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T2: {
                      'noise': 0.5,
                      'intercept': (1, AtTrial(0)),
                      INPUT_PORT_PARAMS: ({
                          'scale': (20, AtTrial(0)),
                          FUNCTION_PARAMS:{'weights':(10, AtTrial(1))},
                      }, AtTrial(1))
                  },
              },
              num_trials=2
              )
        # Run 4: Test Projection params
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T2: {
                      'noise': 0.5,
                      INPUT_PORT_PARAMS: {
                          MAPPING_PROJECTION_PARAMS:{
                              'variable':(1000, AtTrial(0)),
                              'value':(2000, AtTrial(1)),
                          },
                          P:{'value':(3000, AtTrial(2))},
                          'MY PROJECTION':{'value':(4000, AtTrial(3))}
                      }
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

        # Final Run: insure restored default for noise after last run
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
            np.array([[4000.5]]),# Run 4: Trial 0: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*1000)+0.5
            np.array([[8000.5]]),# Run 4: Trial 1: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*2000)+0.5
            np.array([[12000.5]]),# Run 4: Trial 2: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*3000)+0.5
            np.array([[16000.5]]),# Run 4: Trial 3: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*4000)+0.5
            np.array([[40.]])   # Final run: revert to assignments before previous run (2*5*4)
        ])


    def test_params_for_modulatory_projection(self):
        # Construction
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        CTL = ControlMechanism(control=ControlSignal(projections=('slope',T2)))
        C = Composition(pathways=[[T1,T2,CTL]])

        # Run 1
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T2: {
                      PARAMETER_PORT_PARAMS: {
                          CONTROL_PROJECTION_PARAMS: {
                              'variable':(5, AtTrial(3)), # variable of all Projection to all ParameterPorts
                              'value':(10, AtTrial(4)),
                              'value':(21, AtTrial(5)),
                          },
                          # Test individual Projection specifications outside of type-specific dict
                          CTL.control_signals[0].efferents[0]: {'value':(32, AtTrial(6))},
                          'ControlProjection for TransferMechanism-1[slope]': {'value':(43, AtTrial(7))},
                      }
                  },
              },
              num_trials=8
              )
        CTL.control_signals[0].modulation = OVERRIDE
        # Run 2
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T2: {
                      PARAMETER_PORT_PARAMS: {
                          CONTROL_PROJECTION_PARAMS: {
                              'value':(5, Any(AtTrial(0), AtTrial(2))),
                              'variable':(10, AtTrial(1)),
                              # Test individual Projection specifications inside of type-specific dict
                              'ControlProjection for TransferMechanism-1[slope]': {'value':(19, AtTrial(3))},
                              CTL.control_signals[0].efferents[0]: {'value':(33, AtTrial(4))},
                          },
                      }
                  },
              },
              num_trials=5
              )

        assert np.allclose(C.results,[         # Conditions satisfied:
            np.array([[2]]),   # Run 1, Trial 0: None (2 input; no control since that requires a cycle)
            np.array([[4]]),   # Run 1, Trial 1: None (2 input * 2 control gathered last cycle)
            np.array([[8]]),   # Run 1, Trial 2: None (2 input * 4 control gathered last cycle)
            np.array([[10]]),  # Run 1, Trial 3: ControlProjection variable (2*5)
            np.array([[20]]),  # Run 1, Trial 4: ControlProjection value (2*10)
            np.array([[42]]),  # Run 1, Trial 5: ControlProjection value using Projection type-specific keyword (2*210)
            np.array([[64]]),  # Run 1, Trial 6: ControlProjection value using individual Projection (2*32)
            np.array([[86]]),  # Run 1, Trial 7: ControlProjection value using individual Projection by name (2*43)
            np.array([[10]]),  # Run 2, Tria1 0: ControlProjection value with OVERRIDE using value (2*5)
            np.array([[20]]),  # Run 2, Tria1 1: ControlProjection value with OVERRIDE using variable (2*10)
            np.array([[10]]),  # Run 2, Tria1 2: ControlProjection value with OVERRIDE using value again (in Any) (2*5)
            np.array([[38]]),  # Run 2, Tria1 3: ControlProjection value with OVERRIDE using individ Proj by name (2*19)
            np.array([[66]]),  # Run 2: Trial 4: ControlProjection value with OVERRIDE using individ Proj  (2*33)
        ])


    def test_composition_runtime_param_errors(self):

        # Construction
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        CM = ComparatorMechanism()
        P1 = MappingProjection(sender=T1, receiver=CM.input_ports[SAMPLE], name='SAMPLE PROJECTION')
        P2 = MappingProjection(sender=T2, receiver=CM.input_ports[TARGET], name='TARGET PROJECTION')
        C = Composition(nodes=[T1,T2,CM], projections=[P1,P2])

        T1.function.slope = 3
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

        # Bad param specified in INPUT_PORT_PARAMS
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

        # Bad param specified in FUNCTION_PARAMS of INPUT_PORT_PARAMS
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


        # Bad param specified in Projection in <TYPE>_PROJECTION_PARAMS
        with pytest.raises(ComponentError) as error_text:
            C.run(inputs={T1: 2.0,
                         T2: 4.0},
                 # runtime_params=rt_dict,
                 runtime_params={
                     CM: {
                         # 'variable' : 1000
                         CM.input_ports[TARGET] : {'variable':(1000, AtTrial(0))},
                         CM.output_port : {'value':(1000, AtTrial(0))},
                         INPUT_PORT_PARAMS: {
                             MAPPING_PROJECTION_PARAMS:{'value':(2000, AtTrial(0)),
                                                        'glarfip' : 2,                    # Bad arg in Projection type
                                                        P1:{'value':(3000, AtTrial(2))},
                                                        'MY PROJECTION':{'value':(4000, AtTrial(3))}
                                                        }
                         }
                     }
                 },
                 num_trials=2
                 )
        assert ("Invalid specification in runtime_params arg for matrix of SAMPLE PROJECTION: 'glarfip'."
                in error_text.value.error_value)

        # Bad param specified in Projection entry within <TYPE>_PROJECTION_PARAMS
        with pytest.raises(ComponentError) as error_text:
            C.run(inputs={T1: 2.0,
                         T2: 4.0},
                 runtime_params={
                     CM: {
                         # 'variable' : 1000
                         CM.input_ports[TARGET] : {'variable':(1000, AtTrial(0))},
                         CM.output_port : {'value':(1000, AtTrial(0))},
                         INPUT_PORT_PARAMS: {
                             MAPPING_PROJECTION_PARAMS:{'value':(2000, AtTrial(0)),
                                                        P1:{'value':(3000, AtTrial(2)),
                                                            'scrulip' : 2,              # Bad Projection specific arg
                                                            },
                                                        'MY PROJECTION':{'value':(4000, AtTrial(3))}
                                                        }
                         }
                     }
                 },
                 num_trials=2
                 )
        assert ("Invalid specification in runtime_params arg for matrix of SAMPLE PROJECTION: 'scrulip'."
                in error_text.value.error_value)

        # Bad param specified in Projection specified by name \within <TYPE>_PROJECTION_PARAMS
        with pytest.raises(ComponentError) as error_text:
            C.run(inputs={T1: 2.0,
                         T2: 4.0},
                 runtime_params={
                     CM: {
                         # 'variable' : 1000
                         CM.input_ports[TARGET] : {'variable':(1000, AtTrial(0))},
                         CM.output_port : {'value':(1000, AtTrial(0))},
                         INPUT_PORT_PARAMS: {
                             MAPPING_PROJECTION_PARAMS:{'value':(2000, AtTrial(0)),
                                                        P1:{'value':(3000, AtTrial(2)),
                                                            },
                                                        'TARGET PROJECTION':{'value':(4000, AtTrial(3)),
                                                                             'amiby': 4}
                                                        }
                         }
                     }
                 },
                 num_trials=2
                 )
        assert ("Invalid specification in runtime_params arg for matrix of TARGET PROJECTION: 'amiby'."
                in error_text.value.error_value)

