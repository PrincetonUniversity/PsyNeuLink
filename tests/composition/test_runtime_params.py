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
    OUTPUT_PORT_PARAMS, OVERRIDE, PARAMETER_PORT_PARAMS, MAPPING_PROJECTION_PARAMS, SAMPLE, TARGET
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism

class TestMechanismRuntimeParams:

    def test_mechanism_runtime_param(self):

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
        assert ("Invalid specification in runtime_params arg for TransferMechanism" in str(error_text.value) and
                "'glunfump'" in str(error_text.value))

    # FIX 5/8/20 [JDC]:  ADDD TEST FOR INVALID FUNCTION PARAM
    # def test_mechanism_execute_mechanism_fuction_runtime_param_errors(self):
    #     # FIX 5/8/20 [JDC]: SHOULD FAIL BUT DOESN'T:
    #     T = TransferMechanism()
    #     with pytest.raises(ComponentError) as error_text:
    #         T.function.execute(runtime_params={"spranit": 23})
    #     assert ("Invalid specification in runtime_params arg for TransferMechanism" in str(error_text.value) and
    #             "'spranit'" in str(error_text.value))

class TestCompositionRuntimeParams:

    def test_mechanism_param_no_condition(self):

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
                      INPUT_PORT_PARAMS: {
                          'weight':5,                      # InputPort's parameter (NOT USED)
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
        assert T2.input_port.weight is None
        assert T2.input_port.function.scale == 4.0
        assert T2.input_port.function.parameters.scale.get(C) == 4.0
        assert T2.input_port.function.weights is None
        assert T2.input_port.function.parameters.weights.get(C) is None

        C.run(inputs={T1: 2.0}, )
        assert C.results == [[[1201.5]], # (2*3*20*10)+1+0.5
                             [[40.]]]    # 2*5*4
        assert T1.function.slope == 5.0
        assert T1.parameter_ports['slope'].parameters.value.get(C) == 5.0
        assert T2.input_port.function.parameters.scale.get(C.default_execution_id) == 4.0

    # FIX 5/8/20 [JDC]: ADD TESTS FOR PARAMETERPORTS AND OUTPUTPORTS

    def test_mechanism_param_with_AtTrial_condition(self):

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
        assert T2.input_port.weight is None
        assert T2.input_port.function.scale == 4.0
        assert T2.input_port.function.parameters.scale.get(C) == 4.0
        assert T2.input_port.function.weights is None
        assert T2.input_port.function.parameters.weights.get(C) is None

        # run again to insure restored default for noise after last run
        C.run(inputs={T1: 2.0}, )

        assert np.allclose(C.results,[np.array([[40.5]]),   # Trial 0 - no conditions met (2*5*4)+0.5
                                      np.array([[24.5]]),   # Trial 1 - only T1.slope condition met (2*3*4)+0.5
                                      np.array([[41.5]]),   # Trial 2 - only T2.intercept condition met (2*5*4)+1+0.5
                                      np.array([[2000.5]]), # Trial 3 - only T2 INPUT_PORT_PARAMS conditions met
                                                            #               (2*5*20*10) + 0.5
                                      np.array([[40.]])]) # New run - revert to assignments before previous run (2*5*4)

    def test_mechanism_params_with_combined_conditions_for_individual_INPUT_PORT_PARAMS(self):

        T1 = TransferMechanism()
        T2 = TransferMechanism()
        P = MappingProjection(sender=T1, receiver=T2, name='MY PROJECTION')
        C = Composition(pathways=[[T1,P,T2]])

        T1.function.slope = 5
        T2.input_port.function.scale = 4
        # Run 0: Test INPUT_PORT_PARAMS for InputPort function directly (scale) and in FUNCTION_PARAMS dict (weights)
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
        # Run 1:  Test INPUT_PORT_PARAMS override by Never() Condition
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
        # Run 2:  Test INPUT_PORT_PARAMS constraint to Trial 1 assignements
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
        # Run 3: Test Projection params
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
        assert T2.input_port.weight is None
        assert T2.input_port.function.scale == 4.0
        assert T2.input_port.function.parameters.scale.get(C) == 4.0
        assert T2.input_port.function.weights is None
        assert T2.input_port.function.parameters.weights.get(C) is None

        # Final Run: insure restored default for noise after last run
        C.run(inputs={T1: 2.0}, )

        assert np.allclose(C.results,[   # Conditions satisfied:
            np.array([[40.5]]),  # Run 0 Trial 0: no conditions (2*5*4)+0.5
            np.array([[24.5]]),  # Run 0 Trial 1: only T1.slope condition (2*3*4)+0.5
            np.array([[41.5]]),  # Run 0 Trial 2: only T2.intercept condition (2*5*4)+1+0.5
            np.array([[200.5]]), # Run 0 Trial 3: only T2 scale condition (2*5*20) + 0.5
            np.array([[400.5]]), # Run 0 Trial 4: only T2.function.weights condition (2*5*4*10)+0.5
            np.array([[40.5]]),  # Run 1 Tria1 0: INPUT_PORT_PARAMS Never() takes precedence over scale (2*5*4)+0.5
            np.array([[40.5]]),  # Run 1 Trial 1: INPUT_PORT_PARAMS Never() takes precedence over weights (2*5*4)+0.5
            np.array([[41.5]]),  # Run 2 Tria1 0: INPUT_PORT_PARAMS AtTrial(1) takes precedence over scale (2*5*4)+1+0.5
            np.array([[400.5]]), # Run 2 Trial 1: INPUT_PORT_PARAMS AtTrial(1) consistent with weights (2*5*4*10)+0.5
            np.array([[4000.5]]), # Run 3 Trial 0: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*1000)+0.5
            np.array([[8000.5]]), # Run 3 Trial 1: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*2000)+0.5
            np.array([[12000.5]]),# Run 3 Trial 2: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*3000)+0.5
            np.array([[16000.5]]),# Run 3 Trial 3: INPUT_PORT_PARAMS AtTrial(0) Projection variable (2*5*4*4000)+0.5
            np.array([[40.]])   # Final run: revert to assignments before previous run (2*5*4)
        ])


    def test_params_for_input_port_and_projection_variable_and_value(self):

        SAMPLE_INPUT = TransferMechanism()
        TARGET_INPUT = TransferMechanism()
        CM = ComparatorMechanism()
        P1 = MappingProjection(sender=SAMPLE_INPUT, receiver=CM.input_ports[SAMPLE], name='SAMPLE PROJECTION')
        P2 = MappingProjection(sender=TARGET_INPUT, receiver=CM.input_ports[TARGET], name='TARGET PROJECTION')
        C = Composition(nodes=[SAMPLE_INPUT, TARGET_INPUT, CM], projections=[P1,P2])

        SAMPLE_INPUT.function.slope = 3
        CM.input_ports[SAMPLE].function.scale = 2

        TARGET_INPUT.input_port.function.scale = 4
        CM.input_ports[TARGET].function.scale = 1.5

        C.run(inputs={SAMPLE_INPUT: 2.0,
                      TARGET_INPUT: 5.0},
              runtime_params={
                  CM: {
                      CM.input_ports[SAMPLE]: {'variable':(83,AtTrial(0))}, # InputPort object outside INPUT_PORT_PARAMS
                      'TARGET': {'value':(999, Any(AtTrial(1),AtTrial(2)))},# InputPort by name outsideINPUT_PORT_PARAMS
                      INPUT_PORT_PARAMS: {
                          'scale': (15, AtTrial(2)),                       # all InputPorts
                          MAPPING_PROJECTION_PARAMS:{'value':(20, Any(AtTrial(3), AtTrial(4))), # all MappingProjections
                                                     'SAMPLE PROJECTION': {'value':(42, AfterTrial(3))}, # By name
                                                     P2:{'value':(156, AtTrial(5))}}                     # By Projection
                      }}},
              num_trials=6
              )
        assert np.allclose(C.results,[   # Conditions satisfied:          CM calculates: TARGET-SAMPLE:
            np.array([[-136.0]]), # Trial 0: CM SAMPLE InputPort variable (5*4*2.5 - 83*2)
            np.array([[987]]),    # Trial 1: CM TARGET InputPort value    (999 - 2*3*2)
            np.array([[909]]),    # Trial 2: CM TARGET InputPort value + CM Inputports SAMPLE fct scale: (999 - 2*3*15)
            np.array([[-10]]),    # Trial 3: Both CM MappingProjections value, scale default (20*1.5 - 20*2)
            np.array([[-54]]),    # Trial 4: Same as 3, but superceded by value for SAMPLE Projection (20*1.5 - 42*2)
            np.array([[150]]),    # Trial 5: Same as 4, but superceded by value for TARGET Projection ((156*1.5-42*2))
        ])

    def test_params_for_modulatory_projection_in_parameter_port(self):

        T1 = TransferMechanism()
        T2 = TransferMechanism()
        CTL = ControlMechanism(control=ControlSignal(projections=('slope',T2)))
        C = Composition(pathways=[[T1,T2,CTL]])

        # Run 0
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
        # Run 1
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
            np.array([[2]]),   # Run 0, Trial 0: None (2 input; no control since that requires a cycle)
            np.array([[4]]),   # Run 0, Trial 1: None (2 input * 2 control gathered last cycle)
            np.array([[8]]),   # Run 0, Trial 2: None (2 input * 4 control gathered last cycle)
            np.array([[10]]),  # Run 0, Trial 3: ControlProjection variable (2*5)
            np.array([[20]]),  # Run 0, Trial 4: ControlProjection value (2*10)
            np.array([[42]]),  # Run 0, Trial 5: ControlProjection value using Projection type-specific keyword (2*210)
            np.array([[64]]),  # Run 0, Trial 6: ControlProjection value using individual Projection (2*32)
            np.array([[86]]),  # Run 0, Trial 7: ControlProjection value using individual Projection by name (2*43)
            np.array([[10]]),  # Run 1, Tria1 0: ControlProjection value with OVERRIDE using value (2*5)
            np.array([[20]]),  # Run 1, Tria1 1: ControlProjection value with OVERRIDE using variable (2*10)
            np.array([[10]]),  # Run 1, Tria1 2: ControlProjection value with OVERRIDE using value again (in Any) (2*5)
            np.array([[38]]),  # Run 1, Tria1 3: ControlProjection value with OVERRIDE using individ Proj by name (2*19)
            np.array([[66]]),  # Run 1: Trial 4: ControlProjection value with OVERRIDE using individ Proj  (2*33)
        ])

    def test_params_for_output_port_variable_and_value(self):

        T1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T2 = TransferMechanism()
        T3 = TransferMechanism()
        # C = Composition(pathways=[[T1.output_ports['FIRST'],T2],
        #                           [T1.output_ports['SECOND'],T3]])
        # FIX 5/8/20 [JDC]: NEED TO ADD PROJECTIONS SINCE CAN'T SPECIFIY OUTPUT PORT IN PATHWAY
        P1 = MappingProjection(sender=T1.output_ports['FIRST'], receiver=T2)
        P2 = MappingProjection(sender=T1.output_ports['SECOND'], receiver=T2)
        C = Composition(nodes=[T1,T2], projections=[P1,P2])

        T1.output_ports['SECOND'].function.slope = 1.5

        # Run 0: Test of both OutputPort variables assigned
        C.run(inputs={T1: 10.0},
              runtime_params={
                  T1: {OUTPUT_PORT_PARAMS: {'variable': 2}}}
              )
        assert T1.value == 0.0 # T1 did not execute since both of its OutputPorts were assigned a variable
        assert T2.value == 5   # (2*1 + 2*1.5)

        # Run 1: Test of both OutputPort values assigned
        C.run(inputs={T1: 11.0},
              runtime_params={
                  T1: {OUTPUT_PORT_PARAMS: {'value': 3}}}
              )
        assert T1.value == 0.0 # T1 did not execute since both of its OutputPorts were assigned a value
        assert T2.value == 6   # (3 + 3)

        # Run 2: Test of on OutputPort variable and the other value assigned
        C.run(inputs={T1: 12.0},
              runtime_params={
                  T1: {OUTPUT_PORT_PARAMS: {
                          'FIRST': {'value': 5},
                          'SECOND': {'variable': 13}}}}
              )
        assert T1.value == 0.0 # T1 did not execute since both of its OutputPorts were assigned a variable or value
        assert T2.value == 24.5   # (5 + 13*1.5)

        # Run 3: Tests of numerical accuracy over all permutations of assignments
        C.run(inputs={T1: 2.0},
              runtime_params={
                  T1: {
                      OUTPUT_PORT_PARAMS: {
                          'variable':(1.7, AtTrial(1)), # variable of all Projection to all ParameterPorts
                          'value':(3, AtTrial(2)),
                          'FIRST': {'variable':(5, Any(AtTrial(3),AtTrial(5),AtTrial(9),AtTrial(11))),
                                    'value':(7, Any(AtTrial(6),AtTrial(8),AtTrial(10),AtTrial(12)))
                                    },
                          'SECOND': {'variable': (11, Any(AtTrial(4),AtTrial(5),AtTrial(10),AtTrial(11),AtTrial(12))),
                                     'value': (13, Any(AtTrial(7),AtTrial(8),AtTrial(9),AtTrial(11),AtTrial(12)))
                                     },
                      },
                  },
                  T2: {
                      'slope': 3
                  },
              },
              num_trials=13
              )
        assert np.allclose(C.results,[         # OutputPort Conditions satisfied:
            np.array([[5]]),      # Run 0, Trial 0:  See above
            np.array([[6]]),      # Run 1, Trial 0:  See above
            np.array([[24.5]]),   # Run 2, Trial 0:  See above
            np.array([[15]]),     # Run 3, Trial 0:  None (2*1 + 2*1.5) * 3
            np.array([[12.75]]),  # Run 3, Trial 1:  variable general (1.7*1 + 1.7*1.5) * 3
            np.array([[18]]),     # Run 3, Trial 2:  value general (3*1 + 3*1) * 3
            np.array([[24]]),     # Run 3, Trial 3:  FIRST variable (5*1 + 2*1.5) * 3
            np.array([[55.5]]),   # Run 3, Trial 4:  SECOND variable (2*1 + 11*1.5) * 3
            np.array([[64.5]]),   # Run 3, Trial 5:  FIRST and SECOND variable (5*1 + 11*1.5) * 3
            np.array([[30]]),     # Run 3, Trial 6:  FIRST value (7 + 2*1.5) * 3
            np.array([[45]]),     # Run 3, Trial 7:  SECOND value (2*1 + 13) * 3
            np.array([[60]]),     # Run 3, Trial 8:  FIRST and SECOND value (7+13) * 3
            np.array([[54]]),     # Run 3, Trial 9:  FIRST variable and SECOND value (5*1 + 13) * 3
            np.array([[70.5]]),   # Run 3, Trial 10: FIRST value and SECOND variable (7 + 11*1.5) * 3
            np.array([[54]]),     # Run 3, Trial 11: FIRST and SECOND variable and SECOND value (5*1 + 13) * 3
            np.array([[60]]),     # Run 3, Trial 12: FIRST and SECOND value and SECOND variable (7+13) * 3
        ])

    def test_composition_runtime_param_errors(self):

        T1 = TransferMechanism()
        T2 = TransferMechanism()
        CM = ComparatorMechanism()
        P1 = MappingProjection(sender=T1, receiver=CM.input_ports[SAMPLE], name='SAMPLE PROJECTION')
        P2 = MappingProjection(sender=T2, receiver=CM.input_ports[TARGET], name='TARGET PROJECTION')
        C = Composition(nodes=[T1,T2,CM], projections=[P1,P2])

        T1.function.slope = 3
        T2.input_port.function.scale = 4

        # Bad param specified for Mechanism
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
        assert ("Invalid specification in runtime_params arg for TransferMechanism" in str(error_text.value)
                and "'glorp'" in str(error_text.value))

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
        assert ("Invalid specification in runtime_params arg for InputPort" in str(error_text.value) and
                "of TransferMechanism" in str(error_text.value) and "'trigot'" in str(error_text.value))

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
        assert ("Invalid specification in runtime_params arg for InputPort" in str(error_text.value) and
                "of TransferMechanism" in str(error_text.value) and "'flurb'" in str(error_text.value))


        # Bad param specified in <TYPE>_PROJECTION_PARAMS
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
                in str(error_text.value))

        # Bad param specified for Projection specified within <TYPE>_PROJECTION_PARAMS
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
                in str(error_text.value))

        # Bad param specified in Projection specified by name within <TYPE>_PROJECTION_PARAMS
        with pytest.raises(ComponentError) as error_text:
            C.run(inputs={T1: 2.0,
                         T2: 4.0},
                 runtime_params={
                     CM: {
                         CM.input_ports[TARGET] : {'variable':(1000, AtTrial(0))},
                         CM.output_port : {'value':(1000, AtTrial(0))},
                         INPUT_PORT_PARAMS: {
                             MAPPING_PROJECTION_PARAMS:{'value':(2000, AtTrial(0)),
                                                        P1:{'value':(3000, AtTrial(2)),
                                                            },
                                                        'TARGET PROJECTION':{'value':(4000, AtTrial(3)),
                                                                             'amiby': 4}         # Bad Projection param
                                                        }
                         }
                     }
                 },
                 num_trials=2
                 )
        assert ("Invalid specification in runtime_params arg for matrix of TARGET PROJECTION: 'amiby'."
                in str(error_text.value))
