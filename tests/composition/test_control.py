import re

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.globals.keywords import ALLOCATION_SAMPLES, CONTROL, PROJECTIONS
from psyneulink.core.globals.log import LogCondition
from psyneulink.core.globals.sampleiterator import SampleIterator, SampleIteratorError, SampleSpec
from psyneulink.core.globals.utilities import _SeededPhilox
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import \
    _deferred_agent_rep_input_port_name, _deferred_state_feature_spec_msg, \
    _state_input_port_name, _numeric_state_input_port_name, _shadowed_state_input_port_name

@pytest.mark.control
class TestControlSpecification:
    # These test the coordination of adding a node with a control specification to a Composition
    #    with adding a controller that may also specify control of that node.
    # Principles:
    #    1) there should be no redundant ControlSignals or ControlProjections created;
    #    2) specification of control in controller supercedes any conflicting specification on a node;
    #    3) order of addition to the composition does not matter (i.e., Principle 2 always applies)

    def test_add_node_with_control_specified_then_add_controller(self):
        # First add Mechanism with control specification to Composition,
        #    then add controller with NO control specification to Composition
        # ControlProjection specified on Mechanism should initially be in deferred_init,
        #    but then initialized and added to controller when the Mechanism is added.
        ddm = pnl.DDM(function=pnl.DriftDiffusionAnalytical(
                                drift_rate=(1.0,
                                            pnl.ControlProjection(
                                                  function=pnl.Linear,
                                                  control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}))))
        ctl_mech = pnl.ControlMechanism()
        comp = pnl.Composition()
        comp.add_node(ddm)
        comp.add_controller(ctl_mech)
        assert ddm.parameter_ports['drift_rate'].mod_afferents[0].sender.owner == comp.controller
        assert comp.controller.control_signals[0].efferents[0].receiver == ddm.parameter_ports['drift_rate']
        assert np.allclose(comp.controller.control[0].allocation_samples(),
                           [0.1, 0.4, 0.7000000000000001, 1.0000000000000002])

    def test_add_controller_in_comp_constructor_then_add_node_with_control_specified(self):
        # First create Composition with controller that has NO control specification,
        #    then add Mechanism with control specification to Composition;
        # ControlProjection specified on Mechanism should initially be in deferred_init,
        #    but then initialized and added to controller when the Mechanism is added.
        ddm = pnl.DDM(function=pnl.DriftDiffusionAnalytical(
                                # drift_rate=(1.0,
                                #             pnl.ControlProjection(
                                #                   function=pnl.Linear,
                                #                   control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}))))
                                drift_rate=(1.0,
                                            pnl.ControlSignal(allocation_samples=np.arange(0.1, 1.01, 0.3),
                                                              intensity_cost_function=pnl.Linear))))
        ctl_mech = pnl.ControlMechanism()
        comp = pnl.Composition(controller=ctl_mech)
        comp.add_node(ddm)
        comp._analyze_graph()
        assert comp.controller.control[0].efferents[0].receiver == ddm.parameter_ports['drift_rate']
        assert ddm.parameter_ports['drift_rate'].mod_afferents[0].sender.owner == comp.controller
        assert np.allclose(comp.controller.control[0].allocation_samples(),
                           [0.1, 0.4, 0.7000000000000001, 1.0000000000000002])

    def test_redundant_control_spec_add_node_with_control_specified_then_controller_in_comp_constructor(self):
        # First add Mechanism with control specification to Composition,
        #    then add controller WITH redundant control specification to Composition
        # Control specification on controller should replace one on Mechanism
        ddm = pnl.DDM(function=pnl.DriftDiffusionAnalytical(
                                drift_rate=(1.0,
                                            pnl.ControlProjection(
                                                  function=pnl.Linear,
                                                  control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}))))
        comp = pnl.Composition()
        comp.add_node(ddm)
        comp.add_controller(pnl.ControlMechanism(control_signals=("drift_rate", ddm)))
        assert comp.controller.control_signals[0].efferents[0].receiver == ddm.parameter_ports['drift_rate']
        assert ddm.parameter_ports['drift_rate'].mod_afferents[0].sender.owner == comp.controller
        assert comp.controller.control_signals[0].allocation_samples is None

    def test_redundant_control_spec_add_controller_in_comp_constructor_then_add_node_with_control_specified(self):
        # First create Composition with controller that has HAS control specification,
        #    then add Mechanism with control specification to Composition;
        # Control specification on controller should supercede one on Mechanism (which should be ignored)
        ddm = pnl.DDM(function=pnl.DriftDiffusionAnalytical(
                                drift_rate=(1.0,
                                            pnl.ControlProjection(
                                                  function=pnl.Linear,
                                                  control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}))))
        expected_warning = "The controller of 'Composition-0' has been specified to project to 'DDM-0', but 'DDM-0' " \
                           "is not in 'Composition-0' or any of its nested Compositions. This projection will be " \
                           "deactivated until 'DDM-0' is added to' Composition-0' in a compatible way."
        with pytest.warns(UserWarning, match=expected_warning):
            comp = pnl.Composition(controller=pnl.ControlMechanism(control_signals=("drift_rate", ddm)))
        comp.add_node(ddm)
        assert comp.controller.control_signals[0].efferents[0].receiver == ddm.parameter_ports['drift_rate']
        assert ddm.parameter_ports['drift_rate'].mod_afferents[0].sender.owner == comp.controller
        assert comp.controller.control_signals[0].allocation_samples is None

    @pytest.mark.parametrize("control_spec", [CONTROL, PROJECTIONS])
    def test_redundant_control_spec_add_controller_in_comp_constructor_then_add_node_with_alloc_samples_specified(self,control_spec):
        # First create Composition with controller that has HAS control specification that includes allocation_samples,
        #    then add Mechanism with control specification to Composition;
        # Control specification on controller should supercede one on Mechanism (which should be ignored)
        ddm = pnl.DDM(function=pnl.DriftDiffusionAnalytical(
                                drift_rate=(1.0,
                                            pnl.ControlProjection(
                                                  function=pnl.Linear,
                                                  control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01,0.3)}))))
        expected_warning = "The controller of 'Composition-0' has been specified to project to 'DDM-0', but 'DDM-0' " \
                           "is not in 'Composition-0' or any of its nested Compositions. This projection will be " \
                           "deactivated until 'DDM-0' is added to' Composition-0' in a compatible way."
        with pytest.warns(UserWarning, match=expected_warning):
            comp = pnl.Composition(controller=pnl.ControlMechanism(control_signals={ALLOCATION_SAMPLES:np.arange(0.2,1.01, 0.3),
                                                                                    control_spec:('drift_rate', ddm)}))
        comp.add_node(ddm)
        assert comp.controller.control_signals[0].efferents[0].receiver == ddm.parameter_ports['drift_rate']
        assert ddm.parameter_ports['drift_rate'].mod_afferents[0].sender.owner == comp.controller
        assert np.allclose(comp.controller.control[0].allocation_samples(), [0.2, 0.5, 0.8])

    # def test_missing_mech_referenced_by_controller_warning(self):
    #     mech = pnl.ProcessingMechanism()
    #     warning_msg_1 = ''
    #     with pytest.warns(UserWarning) as warning:
    #         comp = pnl.Composition(controller=pnl.ControlMechanism(objective_mechanism=mech))
    #     assert repr(warning[1].message.args[0]) == warning_msg_1

    def test_bad_objective_mechanism_spec(self):
        mech = pnl.ProcessingMechanism()
        expected_error = 'Specification of objective_mechanism arg for \'ControlMechanism-0\' ' \
                         '(ProcessingMechanism-0) must be an ObjectiveMechanism or a list of Mechanisms ' \
                         'and/or OutputPorts to be monitored for control.'
        with pytest.raises(pnl.ControlMechanismError) as error:
            pnl.Composition(controller=pnl.ControlMechanism(objective_mechanism=mech))
        error_msg = error.value.error_value
        assert expected_error in error_msg

    def test_objective_mechanism_spec_as_monitor_for_control_error(self):
        expected_error = 'The \'monitor_for_control\' arg of \'ControlMechanism-0\' contains a specification ' \
                         'for an ObjectiveMechanism ([(ObjectiveMechanism ObjectiveMechanism-0)]).  ' \
                         'This should be specified in its \'objective_mechanism\' argument.'
        with pytest.raises(pnl.ControlMechanismError) as error:
            pnl.Composition(controller=pnl.ControlMechanism(monitor_for_control=pnl.ObjectiveMechanism()))
        error_msg = error.value.error_value
        assert expected_error in error_msg

    @pytest.mark.state_features
    @pytest.mark.parametrize("control_spec", [CONTROL, PROJECTIONS])
    @pytest.mark.parametrize("state_features_arg", [
        'none',
        'default_none',
        'list_none',
        'list_ports',
        'list_reversed',
        'list_numeric',
        'list_partial',
        'dict',
        'dict_reversed',
        'dict_partial',
    ])
    def test_deferred_init(self, control_spec, state_features_arg):
        # Test to insure controller works the same regardless of whether it is added to a composition before or after
        # the nodes it connects to

        # Mechanisms
        Input = pnl.TransferMechanism(name='Input')
        reward = pnl.TransferMechanism(output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE],
                                       name='reward')
        Decision = pnl.DDM(
            function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0,
                                                              pnl.ControlProjection(
                                                                  function=pnl.Linear,
                                                                  control_signal_params={
                                                                      pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                                                                  })),
                                                  threshold=(1.0,
                                                             pnl.ControlProjection(
                                                                 function=pnl.Linear,
                                                                 control_signal_params={
                                                                     pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                                                                 })),
                                                  noise=0.5,
                                                  starting_value=0,
                                                  non_decision_time=0.45),
            output_ports=[pnl.DECISION_VARIABLE,
                          pnl.RESPONSE_TIME,
                          pnl.PROBABILITY_UPPER_THRESHOLD],
            name='Decision')

        comp = pnl.Composition(name="evc", retain_old_simulation_data=True)

        state_features = {
            'none' : None,
            'default_none' : 'DEFAULT NONE',
            'list_none': [None, None],
            'list_ports': [reward.input_port, Input.input_port],
            'list_reversed': [Input.input_port, reward.input_port],
            'list_partial': [reward.input_port],
            'list_numeric': [[1.1],[2.2]],
            'dict': {reward: reward.input_port,
                     Input: Input.input_port},
            'dict_reversed': {reward: Input.input_port,
                              Input: reward.input_port},
            'dict_partial': {reward: reward.input_port}
        }[state_features_arg]

        if state_features == 'DEFAULT NONE':
            state_feature_default = None
        else:
            state_feature_default = pnl.SHADOW_INPUTS

        if state_features_arg in {'none', 'default_none',
                                  'list_none', 'list_ports', 'list_reversed', 'list_numeric', 'list_partial'}:
            expected_warning = f"The '{pnl.STATE_FEATURES}' arg for 'OptimizationControlMechanism-0' has been specified " \
                               f"before any Nodes have been assigned to its agent_rep ('evc').  Their order must " \
                               f"be the same as the order of the corresponding INPUT Nodes for 'evc' once they are " \
                               f"added, or unexpected results may occur.  It is safer to assign all Nodes to the " \
                               f"agent_rep of a controller before specifying its 'state_features'."
        elif state_features_arg in {'dict', 'dict_reversed'}:
            # expected_warning = f"The 'state_features' specified for 'OptimizationControlMechanism-0' " \
            #                    f"contains items (Input, reward) that are not in its agent_rep ('evc'). " \
            #                    f"Executing 'evc' before they are added will generate an error ."
            expected_warning = f"that are not in its agent_rep ('evc'). " \
                               f"Executing 'evc' before they are added will generate an error ."
        elif state_features_arg == 'dict_partial':
            expected_warning = f"The '{pnl.STATE_FEATURES}' specified for 'OptimizationControlMechanism-0' " \
                               f"contains an item (reward) that is not in its agent_rep ('evc'). " \
                               f"Executing 'evc' before it is added will generate an error ."
        else:
            assert False, f"TEST ERROR: unrecognized state_features_arg '{state_features_arg}'"

        with pytest.warns(UserWarning) as warning:
            # add the controller to the Composition before adding the relevant Mechanisms
            if 'default_none' in state_features_arg:
                comp.add_controller(controller=pnl.OptimizationControlMechanism(
                    agent_rep=comp,
                    # state_features = state_features, # Don't specify in order to test default assignments
                    state_feature_default=state_feature_default,
                    state_feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                    objective_mechanism=pnl.ObjectiveMechanism(
                        function=pnl.LinearCombination(operation=pnl.PRODUCT),
                        monitor=[reward,
                                 Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                                 (Decision.output_ports[pnl.RESPONSE_TIME], -1, 1)]),
                    function=pnl.GridSearch(),
                    control_signals=[{control_spec: ("drift_rate", Decision),
                                      ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                                     {control_spec: ("threshold", Decision),
                                      ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}])
                )
            else:
                comp.add_controller(controller=pnl.OptimizationControlMechanism(
                    agent_rep=comp,
                    state_features = state_features,
                    state_feature_default=state_feature_default,
                    state_feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                    objective_mechanism=pnl.ObjectiveMechanism(
                        function=pnl.LinearCombination(operation=pnl.PRODUCT),
                        monitor=[reward,
                                 Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                                 (Decision.output_ports[pnl.RESPONSE_TIME], -1, 1)]),
                    function=pnl.GridSearch(),
                    control_signals=[{control_spec: ("drift_rate", Decision),
                                      ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                                     {control_spec: ("threshold", Decision),
                                      ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}])
                )
        assert any(expected_warning in repr(w.message) for w in warning.list)

        deferred_reward_input_port = _deferred_state_feature_spec_msg('reward[InputPort-0]', 'evc')
        deferred_Input_input_port = _deferred_state_feature_spec_msg('Input[InputPort-0]', 'evc')
        deferred_node_0 = _deferred_agent_rep_input_port_name('0','evc')
        deferred_node_1 = _deferred_agent_rep_input_port_name('1','evc')
        deferred_shadowed_0 = _shadowed_state_input_port_name('reward[InputPort-0]' , deferred_node_0)
        deferred_shadowed_1 = _shadowed_state_input_port_name('Input[InputPort-0]' , deferred_node_1)
        deferred_shadowed_0_rev = _shadowed_state_input_port_name('Input[InputPort-0]' , deferred_node_0)
        deferred_shadowed_1_rev = _shadowed_state_input_port_name('reward[InputPort-0]' , deferred_node_1)
        deferred_numeric_input_port_0 = _numeric_state_input_port_name(deferred_node_0)
        deferred_numeric_input_port_1 = _numeric_state_input_port_name(deferred_node_1)
        deferred_reward_node = _deferred_agent_rep_input_port_name('reward[InputPort-0]', 'evc')
        deferred_Input_node = _deferred_agent_rep_input_port_name('Input[InputPort-0]', 'evc')
        shadowed_reward_node = _shadowed_state_input_port_name('reward[InputPort-0]', 'reward[InputPort-0]')
        shadowed_Input_node = _shadowed_state_input_port_name('Input[InputPort-0]', 'Input[InputPort-0]')
        shadowed_reward_node_rev = _shadowed_state_input_port_name('reward[InputPort-0]', 'Input[InputPort-0]')
        shadowed_Input_node_rev = _shadowed_state_input_port_name('Input[InputPort-0]', 'reward[InputPort-0]')
        numeric_reward_node = _numeric_state_input_port_name('reward[InputPort-0]')
        numeric_Input_node = _numeric_state_input_port_name('Input[InputPort-0]')

        assert comp._controller_initialization_status == pnl.ContextFlags.DEFERRED_INIT

        if state_features_arg in {'none', 'default_none'}:
            assert comp.controller.state_input_ports.names == []
            assert comp.controller.state_features == {}
            assert comp.controller.state_feature_values == {}
        elif state_features_arg == 'list_none':
            assert comp.controller.state_input_ports.names == []
            assert comp.controller.state_features == {deferred_node_0: None, deferred_node_1: None}
            assert comp.controller.state_feature_values == {}
        elif state_features_arg == 'list_ports':
            assert comp.controller.state_input_ports.names == [deferred_shadowed_0, deferred_shadowed_1]
            assert comp.controller.state_features == {deferred_node_0: deferred_reward_input_port,
                                                      deferred_node_1: deferred_Input_input_port}
            assert comp.controller.state_feature_values == {deferred_node_0: deferred_reward_input_port,
                                                            deferred_node_1: deferred_Input_input_port}
        elif state_features_arg == 'list_reversed':
            assert comp.controller.state_input_ports.names == [deferred_shadowed_0_rev, deferred_shadowed_1_rev]
            assert comp.controller.state_features == {deferred_node_0: deferred_Input_input_port,
                                                      deferred_node_1: deferred_reward_input_port}
            assert comp.controller.state_feature_values == {deferred_node_0: deferred_Input_input_port,
                                                            deferred_node_1: deferred_reward_input_port}
        elif state_features_arg == 'list_partial':
            assert comp.controller.state_input_ports.names == [deferred_shadowed_0]
            assert comp.controller.state_features == {deferred_node_0: deferred_reward_input_port}
            assert comp.controller.state_feature_values == {deferred_node_0: deferred_reward_input_port}
        elif state_features_arg == 'list_numeric':
            assert comp.controller.state_input_ports.names == [deferred_numeric_input_port_0,
                                                               deferred_numeric_input_port_1]
            assert comp.controller.state_features == {deferred_node_0: [1.1], deferred_node_1: [2.2]}
            assert np.allclose(list(comp.controller.state_feature_values.values()), [[0.9625],[1.925]])
            assert list(comp.controller.state_feature_values.keys()) == [deferred_node_0, deferred_node_1]
        elif state_features_arg == 'dict':
            assert comp.controller.state_input_ports.names == [deferred_shadowed_0, deferred_shadowed_1]
            assert comp.controller.state_features == {deferred_reward_node: deferred_reward_input_port,
                                                      deferred_Input_node: deferred_Input_input_port}
            assert comp.controller.state_feature_values == {deferred_reward_node: deferred_reward_input_port,
                                                            deferred_Input_node: deferred_Input_input_port}
        elif state_features_arg == 'dict_reversed':
            assert comp.controller.state_input_ports.names == [deferred_shadowed_0_rev, deferred_shadowed_1_rev]
            assert comp.controller.state_features == {deferred_reward_node: deferred_Input_input_port,
                                                      deferred_Input_node: deferred_reward_input_port}
            assert comp.controller.state_feature_values == {deferred_reward_node: deferred_Input_input_port,
                                                            deferred_Input_node: deferred_reward_input_port}
        elif state_features_arg == 'dict_partial':
            assert comp.controller.state_input_ports.names == [deferred_shadowed_0]
            assert comp.controller.state_features == {deferred_reward_node: deferred_reward_input_port}
            assert comp.controller.state_feature_values == {deferred_reward_node: deferred_reward_input_port}
        else:
            assert False, f"TEST ERROR: unrecognized state_features_arg '{state_features_arg}'"

        comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
        comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
        task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
        comp.add_linear_processing_pathway(task_execution_pathway)

        comp.enable_controller = True

        if state_features_arg == 'default_none':
            assert not any(p.path_afferents for p in comp.controller.state_input_ports)
            assert comp.controller.state_features == {}
            assert comp.controller.state_feature_values == {}
            assert comp.controller.state_input_ports.names == []
        elif state_features_arg == 'list_none':
            assert not any(p.path_afferents for p in comp.controller.state_input_ports)
            assert comp.controller.state_features == {'reward[InputPort-0]': None,
                                                      'Input[InputPort-0]': None}
            assert comp.controller.state_feature_values == {}
            assert comp.controller.state_input_ports.names == []
        elif state_features_arg == 'list_numeric':
            assert not any(p.path_afferents for p in comp.controller.state_input_ports)
            assert comp.controller.state_input_ports.names == [numeric_reward_node, numeric_Input_node]
            assert comp.controller.state_features == {'reward[InputPort-0]': [1.1],
                                                      'Input[InputPort-0]': [2.2]}
            assert np.allclose(list(comp.controller.state_feature_values.values()), [[1.065625],[2.13125]])
            assert list(comp.controller.state_feature_values.keys()) == [reward.input_port, Input.input_port]
        elif state_features_arg in {'list_reversed', 'dict_reversed'}:
            assert all(p.path_afferents for p in comp.controller.state_input_ports)
            assert comp.controller.state_features == {'reward[InputPort-0]': 'Input[InputPort-0]',
                                                      'Input[InputPort-0]': 'reward[InputPort-0]'}
            assert comp.controller.state_feature_values == {Input.input_port: [0], reward.input_port: [0]}
            assert comp.controller.state_input_ports.names == [shadowed_Input_node_rev, shadowed_reward_node_rev]
        else:
            assert all(p.path_afferents for p in comp.controller.state_input_ports)
            assert comp.controller.state_features == {'reward[InputPort-0]': 'reward[InputPort-0]',
                                                      'Input[InputPort-0]': 'Input[InputPort-0]'}
            assert comp.controller.state_feature_values == {reward.input_port: [0], Input.input_port: [0]}
            assert comp.controller.state_input_ports.names == [shadowed_reward_node, shadowed_Input_node]

        # comp._analyze_graph()

        stim_list_dict = {
            Input: [0.5, 0.123],
            reward: [20, 20]
        }

        comp.run(inputs=stim_list_dict)

        if state_features_arg in {'default_none', 'list_none'}:
            expected_sim_results_array = [
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]],
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]],
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]],
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]],
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]],
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]],
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]],
                [[0.], [0.], [0.], [0.49], [0.5]],
                [[0.], [0.], [0.], [1.09], [0.5]],
                [[0.], [0.], [0.], [2.41], [0.5]],
                [[0.], [0.], [0.], [4.45], [0.5]]]
        elif state_features_arg == 'list_numeric':
            expected_sim_results_array = [
                [[1.09785156], [1.09785156], [0.], [0.48989747], [0.5438015]],
                [[1.09946289], [1.09946289], [0.], [1.06483807], [0.66899791]],
                [[1.09986572], [1.09986572], [0.], [2.19475384], [0.77414214]],
                [[1.09996643], [1.09996643], [0.], [3.66103375], [0.85320293]],
                [[1.09999161], [1.09999161], [0.], [0.48842594], [0.66907284]],
                [[1.0999979], [1.0999979], [0.], [0.85321354], [0.94353405]],
                [[1.09999948], [1.09999948], [0.], [1.23401798], [0.99281107]],
                [[1.09999987], [1.09999987], [0.], [1.58437432], [0.99912464]],
                [[1.09999997], [1.09999997], [0.], [0.48560629], [0.77416842]],
                [[1.09999999], [1.09999999], [0.], [0.70600576], [0.99281108]],
                [[1.1], [1.1], [0.], [0.90438208], [0.99982029]],
                [[1.1], [1.1], [0.], [1.09934486], [0.99999554]],
                [[1.1], [1.1], [0.], [0.48210997], [0.85320966]],
                [[1.1], [1.1], [0.], [0.63149987], [0.99912464]],
                [[1.1], [1.1], [0.], [0.76817898], [0.99999554]],
                [[1.1], [1.1], [0.], [0.90454543], [0.99999998]],
                [[1.1], [1.1], [0.], [0.48989707], [0.54388677]],
                [[1.1], [1.1], [0.], [1.06481464], [0.66907403]],
                [[1.1], [1.1], [0.], [2.19470819], [0.77416843]],
                [[1.1], [1.1], [0.], [3.66099691], [0.85320966]],
                [[1.1], [1.1], [0.], [0.48842592], [0.66907403]],
                [[1.1], [1.1], [0.], [0.85321303], [0.94353433]],
                [[1.1], [1.1], [0.], [1.23401763], [0.99281108]],
                [[1.1], [1.1], [0.], [1.58437418], [0.99912464]],
                [[1.1], [1.1], [0.], [0.48560629], [0.77416843]],
                [[1.1], [1.1], [0.], [0.70600576], [0.99281108]],
                [[1.1], [1.1], [0.], [0.90438208], [0.99982029]],
                [[1.1], [1.1], [0.], [1.09934486], [0.99999554]],
                [[1.1], [1.1], [0.], [0.48210997], [0.85320966]],
                [[1.1], [1.1], [0.], [0.63149987], [0.99912464]],
                [[1.1], [1.1], [0.], [0.76817898], [0.99999554]],
                [[1.1], [1.1], [0.], [0.90454543], [0.99999998]]]
        elif state_features_arg in {'list_reversed', 'dict_reversed'}:
            expected_sim_results_array = [
                [[0.25], [0.25], [0.], [0.4879949], [0.68997448]],
                [[0.25], [0.25], [0.], [0.81866742], [0.96083428]],
                [[0.25], [0.25], [0.], [1.14484206], [0.99631576]],
                [[0.25], [0.25], [0.], [1.4493293], [0.99966465]],
                [[0.25], [0.25], [0.], [0.47304171], [0.96083428]],
                [[0.25], [0.25], [0.], [0.54999945], [0.99999724]],
                [[0.25], [0.25], [0.], [0.625], [1.]],
                [[0.25], [0.25], [0.], [0.7], [1.]],
                [[0.25], [0.25], [0.], [0.46418045], [0.99631576]],
                [[0.25], [0.25], [0.], [0.50714286], [1.]],
                [[0.25], [0.25], [0.], [0.55], [1.]],
                [[0.25], [0.25], [0.], [0.59285714], [1.]],
                [[0.25], [0.25], [0.], [0.45999329], [0.99966465]],
                [[0.25], [0.25], [0.], [0.49], [1.]],
                [[0.25], [0.25], [0.], [0.52], [1.]],
                [[0.25], [0.25], [0.], [0.55], [1.]],
                [[0.1865], [0.1865], [0.], [0.4858033], [0.76852478]],
                [[0.1865], [0.1865], [0.], [0.7123133], [0.99183743]],
                [[0.1865], [0.1865], [0.], [0.91645684], [0.99977518]],
                [[0.1865], [0.1865], [0.], [1.11665847], [0.99999386]],
                [[0.1865], [0.1865], [0.], [0.46639458], [0.99183743]],
                [[0.1865], [0.1865], [0.], [0.51666667], [1.]],
                [[0.1865], [0.1865], [0.], [0.56666667], [1.]],
                [[0.1865], [0.1865], [0.], [0.61666667], [1.]],
                [[0.1865], [0.1865], [0.], [0.45951953], [0.99977518]],
                [[0.1865], [0.1865], [0.], [0.48809524], [1.]],
                [[0.1865], [0.1865], [0.], [0.51666667], [1.]],
                [[0.1865], [0.1865], [0.], [0.5452381], [1.]],
                [[0.1865], [0.1865], [0.], [0.45666658], [0.99999386]],
                [[0.1865], [0.1865], [0.], [0.47666667], [1.]],
                [[0.1865], [0.1865], [0.], [0.49666667], [1.]],
                [[0.1865], [0.1865], [0.], [0.51666667], [1.]]]
        else:
            # Note: Removed decision variable OutputPort from simulation results because sign is chosen randomly
            expected_sim_results_array = [
                [[10.], [10.0], [0.0], [0.48999867], [0.50499983]],
                [[10.], [10.0], [0.0], [1.08965888], [0.51998934]],
                [[10.], [10.0], [0.0], [2.40680493], [0.53494295]],
                [[10.], [10.0], [0.0], [4.43671978], [0.549834]],
                [[10.], [10.0], [0.0], [0.48997868], [0.51998934]],
                [[10.], [10.0], [0.0], [1.08459402], [0.57932425]],
                [[10.], [10.0], [0.0], [2.36033556], [0.63645254]],
                [[10.], [10.0], [0.0], [4.24948962], [0.68997448]],
                [[10.], [10.0], [0.0], [0.48993479], [0.53494295]],
                [[10.], [10.0], [0.0], [1.07378304], [0.63645254]],
                [[10.], [10.0], [0.0], [2.26686573], [0.72710822]],
                [[10.], [10.0], [0.0], [3.90353015], [0.80218389]],
                [[10.], [10.0], [0.0], [0.4898672], [0.549834]],
                [[10.], [10.0], [0.0], [1.05791834], [0.68997448]],
                [[10.], [10.0], [0.0], [2.14222978], [0.80218389]],
                [[10.], [10.0], [0.0], [3.49637662], [0.88079708]],
                [[15.], [15.0], [0.0], [0.48999926], [0.50372993]],
                [[15.], [15.0], [0.0], [1.08981011], [0.51491557]],
                [[15.], [15.0], [0.0], [2.40822035], [0.52608629]],
                [[15.], [15.0], [0.0], [4.44259627], [0.53723096]],
                [[15.], [15.0], [0.0], [0.48998813], [0.51491557]],
                [[15.], [15.0], [0.0], [1.0869779], [0.55939819]],
                [[15.], [15.0], [0.0], [2.38198336], [0.60294711]],
                [[15.], [15.0], [0.0], [4.33535807], [0.64492386]],
                [[15.], [15.0], [0.0], [0.48996368], [0.52608629]],
                [[15.], [15.0], [0.0], [1.08085171], [0.60294711]],
                [[15.], [15.0], [0.0], [2.32712843], [0.67504223]],
                [[15.], [15.0], [0.0], [4.1221271], [0.7396981]],
                [[15.], [15.0], [0.0], [0.48992596], [0.53723096]],
                [[15.], [15.0], [0.0], [1.07165729], [0.64492386]],
                [[15.], [15.0], [0.0], [2.24934228], [0.7396981]],
                [[15.], [15.0], [0.0], [3.84279648], [0.81637827]]]

        for simulation in range(len(expected_sim_results_array)):
            assert np.allclose(expected_sim_results_array[simulation],
                               # Note: Skip decision variable OutputPort
                               comp.simulation_results[simulation][0:3] + comp.simulation_results[simulation][4:6])

        expected_results_array = [
            [[20.0], [20.0], [0.0], [1.0], [2.378055160151634], [0.9820137900379085]],
            [[20.0], [20.0], [0.0], [-0.1], [0.48999967725112503], [0.5024599801509442]]
        ]

        for trial in range(len(expected_results_array)):
            np.testing.assert_allclose(comp.results[trial], expected_results_array[trial], atol=1e-08,
                                       err_msg='Failed on expected_output[{0}]'.format(trial))

    @pytest.mark.state_features
    @pytest.mark.parametrize('state_features_option', ['list','set','dict','shadow_inputs_dict'])
    def test_partial_deferred_init(self, state_features_option):
        initial_node_a = pnl.TransferMechanism(name='ia')
        initial_node_b = pnl.ProcessingMechanism(name='ib')
        deferred_node = pnl.ProcessingMechanism(name='deferred')
        ocomp = pnl.Composition(name='ocomp',
                                pathways=[initial_node_a, initial_node_b],
                                controller_mode=pnl.BEFORE)

        member_node_control_signal = pnl.ControlSignal(control=[(pnl.SLOPE, initial_node_a)],
                                                       variable=1.0,
                                                       intensity_cost_function=pnl.Linear(slope=0.0),
                                                       allocation_samples=pnl.SampleSpec(start=1.0, stop=5.0, num=5))

        deferred_node_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, deferred_node)],
                                                         variable=1.0,
                                                         intensity_cost_function=pnl.Linear(slope=0.0),
                                                         allocation_samples=pnl.SampleSpec(start=1.0, stop=5.0, num=5))
        state_features = {
            'list': [initial_node_a.input_port,
                     deferred_node.input_port],
            'set': {initial_node_a,
                    deferred_node},
            'dict': {initial_node_a: initial_node_a.input_port,
                     deferred_node: deferred_node.input_port},
            'shadow_inputs_dict': {pnl.SHADOW_INPUTS: [initial_node_a, deferred_node]}
        }[state_features_option]

        ocomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=ocomp,
                state_features = state_features,
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=initial_node_b.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[
                    member_node_control_signal,
                    deferred_node_control_signal
                ])
        )
        deferred_input_port = _deferred_state_feature_spec_msg('deferred[InputPort-0]', 'ocomp')
        deferred_node_0 = _deferred_agent_rep_input_port_name('0','ocomp')
        deferred_shadowed_0 = _shadowed_state_input_port_name('deferred[InputPort-0]' , deferred_node_0)
        deferred_node_deferred = _deferred_agent_rep_input_port_name('deferred[InputPort-0]','ocomp')
        shadowed_ia_node = _shadowed_state_input_port_name('ia[InputPort-0]' ,'ia[InputPort-0]')
        shadowed_deferred_node_0 = _shadowed_state_input_port_name('deferred[InputPort-0]', deferred_node_0)
        shadowed_deferred_node_deferred = _shadowed_state_input_port_name('deferred[InputPort-0]',
                                                                          'deferred[InputPort-0]')

        assert ocomp.controller.state_input_ports.names == [shadowed_ia_node, shadowed_deferred_node_0]

        if state_features_option in {'list', 'shadow_inputs_dict'}:
            assert ocomp.controller.state_features == {'ia[InputPort-0]': 'ia[InputPort-0]',
                                                       deferred_node_0: deferred_input_port}
            assert ocomp.controller.state_feature_values == {initial_node_a.input_port: [0.],
                                                             deferred_node_0: deferred_input_port}
        elif state_features_option in {'dict', 'set'}:
            assert ocomp.controller.state_features == {'ia[InputPort-0]': 'ia[InputPort-0]',
                                                       deferred_node_deferred: deferred_input_port}
            assert ocomp.controller.state_feature_values == {initial_node_a.input_port: [0.],
                                                             deferred_node_deferred: deferred_input_port}
        else:
            assert False, f"TEST ERROR: unrecognized option '{state_features_option}'"


        if state_features_option in {'list', 'shadow_inputs_dict'}:
            # expected_text = 'The number of \'state_features\' specified for Controller (2) is more than the ' \
            #                 'number of INPUT Nodes (1) of the Composition assigned as its agent_rep (\'ocomp\').'
            #
            expected_text = 'The number of \'state_features\' specified for Controller (2) is more than the ' \
                            'number of INPUT Nodes (1) of the Composition assigned as its agent_rep (\'ocomp\'), ' \
                            'that includes the following: \'deferred\' missing from ocomp.'

        else:
            expected_text = 'The \'state_features\' specified for \'Controller\' contains an item (deferred) ' \
                            'that is not an INPUT Node within its agent_rep (\'ocomp\'); only INPUT Nodes can be ' \
                            'in a set or used as keys in a dict used to specify \'state_features\'.'

        with pytest.raises(pnl.OptimizationControlMechanismError) as error_text:
            ocomp.run({initial_node_a: [1]})
        assert expected_text in error_text.value.error_value

        ocomp.add_linear_processing_pathway([deferred_node, initial_node_b])
        assert ocomp.controller.state_features == {'ia[InputPort-0]': 'ia[InputPort-0]',
                                                   'deferred[InputPort-0]': 'deferred[InputPort-0]'}
        assert ocomp.controller.state_feature_values == {initial_node_a.input_port: [0.],
                                                         deferred_node.input_port: [0.]}
        assert all(p.path_afferents for p in ocomp.controller.state_input_ports)
        assert ocomp.controller.state_input_ports.names == [shadowed_ia_node, shadowed_deferred_node_deferred]

        result = ocomp.run({
            initial_node_a: [1],
            deferred_node: [1]
        })
        # result = 10, the sum of the input (1) multiplied by the value of the ControlSignals projecting,
        #              respectively, to Node "ia" and Node "deferred_node"
        # Control Signal "ia": Maximizes over the search space consisting of ints 1-5
        # Control Signal "deferred_node": Maximizes over the search space consisting of ints 1-5
        assert result == [[10]]

    def test_deferred_objective_mech(self):
        initial_node = pnl.TransferMechanism(name='initial_node')
        deferred_node = pnl.ProcessingMechanism(name='deferred')
        ocomp = pnl.Composition(name='ocomp',
                                pathways=[initial_node],
                                controller_mode=pnl.BEFORE)

        initial_node_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, initial_node)],
                                                        variable=1.0,
                                                        intensity_cost_function=pnl.Linear(slope=0.0),
                                                        allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                          stop=5.0,
                                                                                          num=5))
        ocomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=ocomp,
                state_features=[initial_node.input_port],
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=deferred_node.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[
                    initial_node_control_signal
                ])
        )

        text = '"Controller has \'outcome_ouput_ports\' that receive Projections from the following Components ' \
               'that do not belong to its agent_rep (ocomp): [\'deferred\']."'
        with pytest.raises(pnl.OptimizationControlMechanismError) as error:
            ocomp.run({initial_node: [1]})
        assert text == str(error.value)

        # The objective Mechanism is disabled because one of its aux components is a projection to
        # deferred_node, which is not currently a member node of the composition. Therefore, the Controller
        # has no basis to determine which set of values it should use for its efferent ControlProjections and
        # simply goes with the first in the search space, which is 1.

        # add deferred_node to the Composition
        ocomp.add_linear_processing_pathway([initial_node, deferred_node])

        # The objective mechanism's aux components are now all legal, so it will be activated on the following run
        result = ocomp.run({initial_node: [[1]]})
        assert result == [[5]]
        # result = 5, the input (1) multiplied by the value of the ControlSignal projecting to Node "ia"
        # Control Signal "ia": Maximizes over the search space consisting of ints 1-5

    def test_warning_for_add_controller_twice(self):
        mech = pnl.ProcessingMechanism()
        ctlr_1 = pnl.ControlMechanism()
        comp = pnl.Composition()
        comp.add_node(mech)
        comp.add_controller(ctlr_1)
        with pytest.warns(UserWarning, match="ControlMechanism-0 has already been assigned as the controller "
                                             "for Composition-0; assignment ignored."):
            comp.add_controller(ctlr_1)

    def test_warning_for_controller_assigned_to_another_comp(self):
        mech_1 = pnl.ProcessingMechanism()
        ctlr_1 = pnl.ControlMechanism()
        comp_1 = pnl.Composition()
        comp_1.add_node(mech_1)
        comp_1.add_controller(ctlr_1)
        mech_2 = pnl.ProcessingMechanism()
        comp_2 = pnl.Composition()
        comp_2.add_node(mech_2)
        with pytest.warns(UserWarning, match="'ControlMechanism-0' has already been assigned as the controller "
                                             "for 'Composition-0'; assignment to 'Composition-1' ignored."):
            comp_2.add_controller(ctlr_1)

    def test_warning_for_replacement_of_controller(self):
        mech = pnl.ProcessingMechanism()
        ctlr_1 = pnl.ControlMechanism()
        comp = pnl.Composition()
        comp.add_node(mech)
        comp.add_controller(ctlr_1)
        ctlr_2 = pnl.ControlMechanism()
        expected_warning = "The existing controller for 'Composition-0' ('ControlMechanism-0') " \
                           "is being replaced by 'ControlMechanism-1'."
        with pytest.warns(UserWarning) as warning:
            comp.add_controller(ctlr_2)
        assert expected_warning in repr(warning[0].message.args[0])

    def test_controller_has_no_input(self):
        mech = pnl.ProcessingMechanism()
        ctlr = pnl.ControlMechanism()
        comp = pnl.Composition()
        comp.add_node(mech)
        expected_warning = 'ControlMechanism-0 for Composition-0 is enabled but has no inputs.'
        with pytest.warns(UserWarning) as warning:
            comp.enable_controller = True
            comp.add_controller(ctlr)
        assert expected_warning in repr(warning[0].message.args[0])

    def test_agent_rep_assignment_as_controller_and_replacement(self):
        mech = pnl.ProcessingMechanism()
        comp = pnl.Composition(name='comp',
                               pathways=[mech],
                               controller=pnl.OptimizationControlMechanism(name="old_ocm",
                                                                           agent_rep=None,
                                                                           control_signals=(pnl.SLOPE, mech),
                                                                           search_space=[1]))
        assert comp.controller.composition == comp
        comp._analyze_graph()
        assert comp.controller.state_input_ports[0].shadow_inputs == mech.input_port
        assert comp.controller.state_input_ports[0].path_afferents[0].sender == mech.input_port.path_afferents[0].sender
        assert any(pnl.SLOPE in p_name for p_name in comp.projections.names)
        assert not any(pnl.INTERCEPT in p_name for p_name in comp.projections.names)
        old_ocm = comp.controller

        new_ocm = pnl.OptimizationControlMechanism(name='new_ocm',
                                                   agent_rep=None,
                                                   control_signals=(pnl.INTERCEPT, mech),
                                                   search_space=[1])
        comp.add_controller(new_ocm)
        comp._analyze_graph()

        #Confirm that components of new_ocm have been added
        assert comp.controller == new_ocm
        assert any(pnl.INTERCEPT in p_name for p_name in comp.projections.names)
        assert comp.controller.state_input_ports[0].shadow_inputs == mech.input_port
        assert comp.controller.state_input_ports[0].path_afferents[0].sender == mech.input_port.path_afferents[0].sender

        # Confirm all components of old_ocm have been removed
        assert old_ocm.composition is None
        assert old_ocm.state_input_ports[0].path_afferents == []
        assert not any(pnl.SLOPE in p_name for p_name in comp.projections.names)

    def test_hanging_control_spec_outer_controller(self):
        internal_mech = pnl.ProcessingMechanism(
            name='internal_mech',
            function=pnl.Linear(slope=pnl.CONTROL)
        )

        inner_comp = pnl.Composition(
            name='inner',
            pathways=[internal_mech],
        )

        controller = pnl.ControlMechanism(
            name='controller',
            default_variable=5
        )

        outer_comp = pnl.Composition(
            name='outer_with_controller',
            pathways=[inner_comp],
            controller=controller
        )

        result = outer_comp.run([1])
        assert result == [[5]]
        assert internal_mech.mod_afferents[0].sender.owner == inner_comp.parameter_CIM

    def test_hanging_control_spec_nearest_controller(self):
        inner_controller = pnl.ControlMechanism(
            name='inner_controller',
            default_variable=5
        )

        inner_comp = pnl.Composition(
            name='inner_comp',
            controller=inner_controller
        )

        outer_controller = pnl.ControlMechanism(
            name='outer_controller',
            default_variable=10
        )

        outer_comp = pnl.Composition(
            name='outer_with_controller',
            pathways=[inner_comp],
            controller=outer_controller
        )

        internal_mech = pnl.ProcessingMechanism(
            name='internal_mech',
            function=pnl.Linear(slope=pnl.CONTROL)
        )

        inner_comp.add_node(internal_mech)

        result = outer_comp.run([1])
        assert result == [[5]]
        assert internal_mech.mod_afferents[0].sender.owner == inner_comp.controller

    def test_state_input_ports_for_two_input_nodes(self):
        # Inner Composition
        ia = pnl.TransferMechanism(name='ia')
        icomp = pnl.Composition(name='icomp', pathways=[ia])

        # Outer Composition
        oa = pnl.TransferMechanism(name='oa')
        ob = pnl.TransferMechanism(name='ob')
        oc = pnl.TransferMechanism(name='oc')
        ctl_mech = pnl.ControlMechanism(name='ctl_mech',
                                    control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)])])
        ocomp = pnl.Composition(name='ocomp', pathways=[[ob],[oa, icomp, oc, ctl_mech]])
        # ocomp.add_nodes(ob)
        ocm = pnl.OptimizationControlMechanism(name='ocm',
                                           agent_rep=ocomp,
                                           control_signals=[
                                               pnl.ControlSignal(projections=[(pnl.NOISE, ia)]),
                                               pnl.ControlSignal(projections=[(pnl.INTERCEPT, ia)]),
                                               pnl.ControlSignal(projections=[(pnl.SLOPE, oa)]),
                                           ],
                                           search_space=[[1],[1],[1]])
        ocomp.add_controller(ocm)
        result = ocomp.run({oa: [[1]], ob: [[2]]})
        assert result == [[2.], [1.]]
        assert len(ocomp.controller.state_input_ports) == 2
        assert all([node in [input_port.shadow_inputs.owner for input_port in ocomp.controller.state_input_ports]
                    for node in {oa, ob}])

    @pytest.mark.parametrize(
        'ocm_control_signals',
        [
            'None',
            "[pnl.ControlSignal(modulates=('slope', a))]",
            "[pnl.ControlSignal(modulates=('slope', a), allocation_samples=[1, 2])]",
        ]
    )
    @pytest.mark.parametrize('ocm_num_estimates', [None, 1, 2])
    @pytest.mark.parametrize(
        'slope, intercept',
        [
            ((1.0, pnl.CONTROL), None),
            ((1.0, pnl.CONTROL), (1.0, pnl.CONTROL)),
        ]
    )
    def test_transfer_mechanism_and_ocm_variations(
        self,
        slope,
        intercept,
        ocm_num_estimates,
        ocm_control_signals,
    ):
        a = pnl.TransferMechanism(
            name='a',
            function=pnl.Linear(
                slope=slope,
                intercept=intercept,
            )
        )

        comp = pnl.Composition()
        comp.add_node(a)

        ocm_control_signals = eval(ocm_control_signals)

        search_space_len = len(
            set([
                # value of parameter name in 'modulates' kwarg
                p[0]
                for cs in ocm_control_signals
                for p in cs._init_args['projections']
            ]) if ocm_control_signals is not None else set()
            .union({'slope'} if slope is not None else set())
            .union({'intercept'} if intercept is not None else set())
        )
        search_space = [[0, 1]] * search_space_len

        ocm = pnl.OptimizationControlMechanism(
            agent_rep=comp,
            search_space=search_space,
            num_estimates=ocm_num_estimates,
            control_signals=ocm_control_signals,
        )
        comp.add_controller(ocm)

        # assume tuple is a control spec
        if (
            isinstance(slope, tuple)
            or (
                ocm_control_signals is not None
                and any(cs.name == 'slope' for cs in ocm_control_signals)
            )
        ):
            assert 'a[slope] ControlSignal' in ocm.control.names
        else:
            assert 'a[slope] ControlSignal' not in ocm.control.names

        if (
            isinstance(intercept, tuple)
            or (
                ocm_control_signals is not None
                and any(cs.name == 'intercept' for cs in ocm_control_signals)
            )
        ):
            assert 'a[intercept] ControlSignal' in ocm.control.names
        else:
            assert 'a[intercept] ControlSignal' not in ocm.control.names

@pytest.mark.control
class TestControlMechanisms:

    # id, agent_rep, state_feat, mon_for_ctl, allow_probes, obj_mech err_type, error_msg
    params = [
        ("allowable1",
         "icomp", "I", "I", True, None, None, None
         ),
        ("allowable2",
         "mcomp", "Ii A", "I B", True, None, None, None
         ),
        ("state_features_test_internal",
         "icomp", "B", "I", True, None, pnl.CompositionError,
         "Attempt to shadow the input to a node (B) in a nested Composition of OUTER COMP "
         "that is not an INPUT Node of that Composition is not currently supported."
         ),
        ("state_features_test_not_in_agent_rep",
         "icomp", "A", "I", True, None, pnl.OptimizationControlMechanismError,
        '\'OCM\' has \'state_features\' specified ([\'SHADOWED INPUT OF A[InputPort-0] FOR I[InputPort-0]\']) '
        'that are missing from both its `agent_rep` (\'INNER COMP\') as well as \'OUTER COMP\' '
        'and any Compositions nested within it.'
         ),
        ("monitor_for_control_test_not_in_agent_rep",
         "icomp", "I", "B", True, None, pnl.OptimizationControlMechanismError,
         "OCM has 'outcome_ouput_ports' that receive Projections from the following Components "
         "that do not belong to its agent_rep (INNER COMP): ['B']."
         ),
        ("monitor_for_control_with_obj_mech_test",
         "icomp", "I", None, True, True, pnl.OptimizationControlMechanismError,
         "OCM has 'outcome_ouput_ports' that receive Projections from the following Components "
         "that do not belong to its agent_rep (INNER COMP): ['B']."
         ),
        ("probe_error_test",
         "mcomp", "I", "B", False, None, pnl.CompositionError,
         "B found in nested Composition of OUTER COMP (MIDDLE COMP) but without "
         "required NodeRole.OUTPUT. Try setting 'allow_probes' argument of OCM to 'True'."
         ),
        ("probe_error_obj_mech_test",
         "mcomp", "I", None, False, True, pnl.CompositionError,
         "B found in nested Composition of OUTER COMP (MIDDLE COMP) but without required NodeRole.OUTPUT. "
         "Try setting 'allow_probes' argument of ObjectiveMechanism for OCM to 'True'."
         ),
        ("cfa_as_agent_rep_error",
         "cfa", "dict", None, False, True, pnl.OptimizationControlMechanismError,
         'The agent_rep specified for OCM is a CompositionFunctionApproximator, so its \'state_features\' argument '
         'must be a list, not a dict ({(ProcessingMechanism A): (InputPort InputPort-0), '
         '(ProcessingMechanism B): (InputPort InputPort-0)}).'
         )
    ]
    @pytest.mark.parametrize('id, agent_rep, state_features, monitor_for_control, allow_probes, '
                             'objective_mechanism, error_type, err_msg',
                             params, ids=[x[0] for x in params])
    def test_args_specific_to_ocm(self, id, agent_rep, state_features, monitor_for_control,
                                  allow_probes, objective_mechanism, error_type,err_msg):
        """Test args specific to OptimizationControlMechanism
        NOTE: state_features and associated warning and errors tested more fully in
              test_ocm_state_feature_specs_and_warnings_and_errors() below
        - state_feature must be in agent_rep
        - monitor_for_control must be in agent_rep, whether specified directly or for ObjectiveMechanism
        - allow_probes allows INTERNAL Nodes of nested comp to be monitored, otherwise generates and error
        - probes are not included in Composition.results
        """

        # FIX: ADD VERSION WITH agent_rep = CompositionFuntionApproximator
        #      ADD TESTS FOR SEPARATE AND CONCATENATE

        from psyneulink.core.globals.utilities import convert_to_list

        I = pnl.ProcessingMechanism(name='I')
        icomp = pnl.Composition(nodes=I, name='INNER COMP')

        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')
        C = pnl.ProcessingMechanism(name='C')
        mcomp = pnl.Composition(pathways=[[A,B,C],icomp],
                                name='MIDDLE COMP')
        ocomp = pnl.Composition(nodes=[mcomp], name='OUTER COMP', allow_probes=allow_probes)
        cfa = pnl.RegressionCFA

        agent_rep = {"mcomp":mcomp,
                     "icomp":icomp,
                     "cfa": cfa
                     }[agent_rep]

        state_features = {"I":I,
                          "Ii A":[I.input_port, A],
                          "A":A,
                          "B":B,
                          "dict":{A:A.input_port, B:B.input_port}
                          }[state_features]

        if monitor_for_control:
            monitor_for_control = {"I":I,
                                   "I B":[I, B],
                                   "B":B,
                                   }[monitor_for_control]

        if objective_mechanism:
            objective_mechanism = pnl.ObjectiveMechanism(monitor=B)

        if not err_msg:
            ocm = pnl.OptimizationControlMechanism(name='OCM',
                                                   agent_rep=agent_rep,
                                                   state_features=state_features,
                                                   monitor_for_control=monitor_for_control,
                                                   objective_mechanism=objective_mechanism,
                                                   allow_probes=allow_probes,
                                                   function=pnl.GridSearch(),
                                                   control_signals=pnl.ControlSignal(modulates=(pnl.SLOPE,I),
                                                                                     allocation_samples=[10, 20, 30])
                                                   )
            ocomp.add_controller(ocm)
            ocomp._analyze_graph()
            if allow_probes and B in convert_to_list(monitor_for_control):
                # If this fails, could be due to ordering of ports in ocomp.output_CIM (current assumes probe is on 0)
                assert ocomp.output_CIM._sender_is_probe(ocomp.output_CIM.output_ports[0])
                # Affirm that PROBE (included in ocomp's output_ports via its output_CIM
                #    but is *not* included in Composition.output_values (which is used for Composition.results)
                assert len(ocomp.output_values) == len(ocomp.output_ports) - 1

        else:
            with pytest.raises(error_type) as err:
                ocm = pnl.OptimizationControlMechanism(name='OCM',
                                                       agent_rep=agent_rep,
                                                       state_features=state_features,
                                                       monitor_for_control=monitor_for_control,
                                                       objective_mechanism=objective_mechanism,
                                                       allow_probes=allow_probes,
                                                       function=pnl.GridSearch(),
                                                       control_signals=pnl.ControlSignal(modulates=(pnl.SLOPE,
                                                                                                    I),
                                                                                         allocation_samples=[10, 20, 30])
                                                       )
                ocomp.add_controller(ocm)
                ocomp._analyze_graph()
                ocomp.run()
            assert err.value.error_value == err_msg

    messages = [
        # 0
        f"There are fewer '{pnl.STATE_FEATURES}' specified for 'OptimizationControlMechanism-0' than the number "
        f"of InputPort's for all of the INPUT Nodes of its agent_rep ('OUTER COMP'); the remaining inputs will be "
        f"assigned default values when 'OUTER COMP`s 'evaluate' method is executed. If this is not the desired "
        f"behavior, use its get_inputs_format() method to see the format for its inputs.",

        # 1
        f'\'Attempt to shadow the input to a node (IB) in a nested Composition of OUTER COMP '
        f'that is not an INPUT Node of that Composition is not currently supported.\'',

        # 2
        f'"\'OptimizationControlMechanism-0\' has \'state_features\' specified ([\'SHADOWED INPUT OF EXT[InputPort-0] '
        f'FOR IA[InputPort-0]\']) that are missing from \'OUTER COMP\' and any Compositions nested within it."',

        # 3
        '"\'OptimizationControlMechanism-0\' has \'state_features\' specified ([\'INPUT FROM EXT[OutputPort-0] '
        'FOR IA[InputPort-0]\']) that are missing from \'OUTER COMP\' and any Compositions nested within it."',

        # 4
        f"The '{pnl.STATE_FEATURES}' argument has been specified for 'OptimizationControlMechanism-0' that is using "
        f"a Composition ('OUTER COMP') as its agent_rep, but some of the specifications are not compatible with the "
        f"inputs required by its 'agent_rep': 'Input stimulus ([0.0]) for OB is incompatible with the shape of its "
        f"external input ([0.0 0.0 0.0]).' Use the get_inputs_format() method of 'OUTER COMP' to see the required "
        f"format, or remove the specification of 'state_features' from the constructor for "
        f"OptimizationControlMechanism-0 to have them automatically assigned.",

        # 5
        f"The '{pnl.STATE_FEATURES}' specified for OptimizationControlMechanism-0 is associated with a number of "
        f"InputPorts (4) that is greater than for the InputPorts of the INPUT Nodes (3) for the Composition assigned "
        f"as its agent_rep ('OUTER COMP'). Executing OptimizationControlMechanism-0 before the additional item(s) are "
        f"added as (part of) INPUT Nodes will generate an error.",

        # 6
        f"The '{pnl.STATE_FEATURES}' specified for OptimizationControlMechanism-0 is associated with a number of "
        f"InputPorts (4) that is greater than for the InputPorts of the INPUT Nodes (3) for the Composition assigned "
        f"as its agent_rep ('OUTER COMP'), which includes the following that are not (yet) in 'OUTER COMP': 'EXT'. "
        f"Executing OptimizationControlMechanism-0 before the additional item(s) are added as (part of) INPUT Nodes "
        f"will generate an error.",

        # 7
        f'"The number of \'state_features\' specified for OptimizationControlMechanism-0 (4) is more than the number '
        f'of INPUT Nodes (3) of the Composition assigned as its agent_rep (\'OUTER COMP\')."',

        # 8
        f'The \'state_features\' specified for \'OptimizationControlMechanism-0\' contains an item (OC) '
        f'that is not an INPUT Node within its agent_rep (\'OUTER COMP\'); only INPUT Nodes can be in a set or '
        f'used as keys in a dict used to specify \'state_features\'.',

        # 9
        f'The \'state_features\' specified for \'OptimizationControlMechanism-0\' contains an item (IA) '
        f'that is not an INPUT Node within its agent_rep (\'OUTER COMP\'); only INPUT Nodes can be in a set '
        f'or used as keys in a dict used to specify \'state_features\'.',

        # 10
        f"The '{pnl.STATE_FEATURES}' argument for 'OptimizationControlMechanism-0' includes one or more Compositions "
        f"('INNER COMP') in the list specified for its '{pnl.STATE_FEATURES}' argument; these must be replaced by "
        f"direct references to the Mechanisms (or their InputPorts) within them to be shadowed.",

        # 11
        f"The '{pnl.STATE_FEATURES}' argument for 'OptimizationControlMechanism-0' includes one or more Compositions "
        f"('INNER COMP') in the SHADOW_INPUTS dict specified for its '{pnl.STATE_FEATURES}' argument; these must be "
        f"replaced by direct references to the Mechanisms (or their InputPorts) within them to be shadowed.",

        # 12
        f"The '{pnl.STATE_FEATURES}' argument for 'OptimizationControlMechanism-0' has one or more items in the "
        f"list specified for 'SHADOW_INPUTS' ('IA') that are not (part of) any INPUT Nodes of its 'agent_rep' "
        f"('OUTER COMP').",

        # 13
        f"'OptimizationControlMechanism-0' has '{pnl.STATE_FEATURES}' specified "
        f"(['SHADOWED INPUT OF EXT[InputPort-0] FOR IA[InputPort-0]', "
        f"'SHADOWED INPUT OF EXT[InputPort-0] FOR OA[InputPort-0]', "
        f"'SHADOWED INPUT OF EXT[InputPort-0] FOR OB[InputPort-0]']) "
        f"that are missing from 'OUTER COMP' and any Compositions nested within it."
    ]
    state_feature_args = [
        # STATE_FEATURE_ARGS, STATE_FEATURE_DEFAULT, ERROR_OR_WARNING_MSG, EXCEPTION_TYPE
        ('single_none_spec', pnl.SHADOW_INPUTS, None, None),
        ('single_shadow_spec', pnl.SHADOW_INPUTS, None, None),
        ('single_tuple_shadow_spec', pnl.SHADOW_INPUTS, None, None),
        ('partial_legal_list_spec', pnl.SHADOW_INPUTS, messages[0], UserWarning),
        ('full_list_spec', pnl.SHADOW_INPUTS, None, None),
        ('list_spec_with_none', pnl.SHADOW_INPUTS, None, None),
        ('input_dict_spec', pnl.SHADOW_INPUTS, None, None),
        ('input_dict_spec_short', pnl.SHADOW_INPUTS, None, None),
        ('set_spec_short', None, None, None),
        ('set_spec', pnl.SHADOW_INPUTS, None, None),
        ('set_spec_port', pnl.SHADOW_INPUTS, None, None),
        ('no_specs', None, None, None),
        ('shadow_inputs_dict_spec', pnl.SHADOW_INPUTS, None, None),
        ('shadow_inputs_dict_spec_w_none', pnl.SHADOW_INPUTS, None, None),
        ('misplaced_shadow', pnl.SHADOW_INPUTS, messages[1], pnl.CompositionError),
        ('ext_shadow', pnl.SHADOW_INPUTS, messages[2], pnl.OptimizationControlMechanismError),
        ('ext_output_port', pnl.SHADOW_INPUTS, messages[3], pnl.OptimizationControlMechanismError),
        ('input_format_wrong_shape', pnl.SHADOW_INPUTS, messages[4], pnl.OptimizationControlMechanismError),
        ('too_many_inputs_warning', pnl.SHADOW_INPUTS, messages[5], UserWarning),
        ('too_many_w_node_not_in_composition_warning', pnl.SHADOW_INPUTS, messages[6], UserWarning),
        ('too_many_inputs_error', pnl.SHADOW_INPUTS, messages[7], pnl.OptimizationControlMechanismError),
        ('bad_single_spec', pnl.SHADOW_INPUTS, messages[13], pnl.OptimizationControlMechanismError),
        ('bad_dict_spec_warning', pnl.SHADOW_INPUTS, messages[8], UserWarning),
        ('bad_dict_spec_error', pnl.SHADOW_INPUTS, messages[8], pnl.OptimizationControlMechanismError),
        ('bad_shadow_inputs_dict_spec_error', pnl.SHADOW_INPUTS, messages[12], pnl.OptimizationControlMechanismError),
        ('comp_in_list_spec', pnl.SHADOW_INPUTS, messages[10], pnl.OptimizationControlMechanismError),
        ('comp_in_shadow_inupts_spec', pnl.SHADOW_INPUTS, messages[11], pnl.OptimizationControlMechanismError)
    ]
    if len(state_feature_args) != 27:
        print("\n\n************************************************************************************************")
        print("*** UNCOMMENT state_feature_args IN test_ocm_state_feature_specs_and_warnings_and_errors() *****")
        print("************************************************************************************************")
    @pytest.mark.state_features
    @pytest.mark.control
    @pytest.mark.parametrize('state_feature_args', state_feature_args, ids=[x[0] for x in state_feature_args])
    @pytest.mark.parametrize('obj_mech', ['obj_mech', 'mtr_for_ctl', None])
    def test_ocm_state_feature_specs_and_warnings_and_errors(self, state_feature_args, obj_mech):
        """See test_nested_composition_as_agent_rep() for additional tests of state_features specification."""

        test_condition = state_feature_args[0]
        state_feature_default = state_feature_args[1]
        error_or_warning_message = state_feature_args[2]
        exception_type = state_feature_args[3]

        ia = pnl.ProcessingMechanism(name='IA')
        ib = pnl.ProcessingMechanism(name='IB')
        ic = pnl.ProcessingMechanism(name='IC')
        oa = pnl.ProcessingMechanism(name='OA')
        ob = pnl.ProcessingMechanism(name='OB', size=3)
        oc = pnl.ProcessingMechanism(name='OC')
        ext = pnl.ProcessingMechanism(name='EXT')
        icomp = pnl.Composition(pathways=[ia,ib,ic], name='INNER COMP')
        ocomp = pnl.Composition(pathways=[icomp], name='OUTER COMP')
        ocomp.add_linear_processing_pathway([oa,oc])
        ocomp.add_linear_processing_pathway([ob,oc])

        state_features_dict = {

            # Legal state_features specifications
            'single_none_spec': None,
            'single_shadow_spec': pnl.SHADOW_INPUTS,
            'single_tuple_shadow_spec': (pnl.SHADOW_INPUTS, pnl.Logistic), # state_feature_values should be 0.5
            'partial_legal_list_spec': [oa.output_port], # only specifies ia;  oa and ob assigned default inputs
            'full_list_spec': [ia.input_port, oa.output_port, [3,1,2]],
            'list_spec_with_none': [ia.input_port, None, [3,1,2]],
            'input_dict_spec': {oa:oc.input_port, icomp:ia, ob:ob.output_port}, # use icomp & out of order is OK
            'input_dict_spec_short': {ob:ob.output_port, oa:oc.input_port}, # missing ia spec and out of order
            'set_spec_short': {oa},
            'set_spec': {ob, icomp, oa.input_port},  # out of order, use of Nested comp and InputPort as specs all OK
            'set_spec_port': {ob.input_port, icomp, oa},
            'no_specs': 'THIS IS IGNORED',
            'shadow_inputs_dict_spec': {pnl.SHADOW_INPUTS:[ia, oa, ob]}, # ia & ob OK because just for shadowing
            'shadow_inputs_dict_spec_w_none': {pnl.SHADOW_INPUTS:[ia, None, ob]},

            # Illegal state_features specifications
            'misplaced_shadow': [ib.input_port],
            'ext_shadow': [ext.input_port],
            'ext_output_port': [ext.output_port],
            'input_format_wrong_shape': [ia.input_port, oa.output_port, oc.output_port],
            'too_many_inputs_warning': [ia.input_port, oa.output_port, ob.output_port, oc.output_port],
            'too_many_inputs_error': [ia.input_port, oa.output_port, ob.output_port, oc.output_port],
            'too_many_w_node_not_in_composition_warning': [ia, oa, ob, ext],
            'bad_single_spec': ext.input_port,
            'bad_single_numeric_spec': [3],
            'bad_dict_spec_warning': {oa:oc.input_port, ia:ia, oc:ob.output_port}, # oc is not an INPUT Node
            'bad_dict_spec_error': {oa:oc.input_port, ia:ia, oc:ob.output_port}, # ia & oc are not *ocomp* INPUT Nodes
            'bad_shadow_inputs_dict_spec_error': {pnl.SHADOW_INPUTS:[ia.output_port, None, ob]}, # not INPUT InputPort
            'comp_in_list_spec':[icomp, oa.output_port, [3,1,2]],  # FIX: REMOVE ONCE TUPLE FORMAT SUPPORTED
            'comp_in_shadow_inupts_spec':{pnl.SHADOW_INPUTS:[icomp, oa, ob]}
        }
        objective_mechanism = [ic,ib] if obj_mech == 'obj_mech' else None
        monitor_for_control = [ic] if obj_mech == 'mtr_for_ctl' else None # Needs to be a single item for GridSearch
        state_features = state_features_dict[test_condition]

        ia_node = _state_input_port_name('OA[OutputPort-0]', 'IA[InputPort-0]')
        oa_node = _state_input_port_name('OA[OutputPort-0]', 'OA[InputPort-0]')
        ob_node = _state_input_port_name('OB[OutputPort-0]', 'OB[InputPort-0]')
        numeric_ob = _numeric_state_input_port_name('OB[InputPort-0]')
        shadowed_ia_node = _shadowed_state_input_port_name('IA[InputPort-0]', 'IA[InputPort-0]')
        shadowed_oa_node = _shadowed_state_input_port_name('OA[InputPort-0]', 'OA[InputPort-0]')
        shadowed_oa_oc = _shadowed_state_input_port_name('OC[InputPort-0]', 'OA[InputPort-0]')
        shadowed_ob_node = _shadowed_state_input_port_name('OB[InputPort-0]', 'OB[InputPort-0]')

        if test_condition == 'no_specs':
            ocm = pnl.OptimizationControlMechanism(objective_mechanism=objective_mechanism,
                                                   monitor_for_control=monitor_for_control,
                                                   function=pnl.GridSearch(),
                                                   control_signals=[pnl.ControlSignal(modulates=(pnl.SLOPE,ia),
                                                                                      allocation_samples=[10, 20, 30]),
                                                                    pnl.ControlSignal(modulates=(pnl.INTERCEPT,oc),
                                                                                      allocation_samples=[10, 20, 30])])

        else:
            ocm = pnl.OptimizationControlMechanism(state_features=state_features,
                                                   state_feature_default=state_feature_default,
                                                   objective_mechanism=objective_mechanism,
                                                   monitor_for_control=monitor_for_control,
                                                   function=pnl.GridSearch(),
                                                   control_signals=[pnl.ControlSignal(modulates=(pnl.SLOPE,ia),
                                                                                      allocation_samples=[10, 20, 30]),
                                                                    pnl.ControlSignal(modulates=(pnl.INTERCEPT,oc),
                                                                                      allocation_samples=[10, 20, 30])])
        if not exception_type:

            ocomp.add_controller(ocm)
            ocomp.run()

            if test_condition == 'single_none_spec':
                assert len(ocm.state_input_ports) == 0
                assert ocm.state_features == {'IA[InputPort-0]': None,
                                              'OA[InputPort-0]': None,
                                              'OB[InputPort-0]': None}
                assert ocm.state_feature_values == {}

            if test_condition in {'single_shadow_spec',
                                  'set_spec',
                                  'set_spec_port',
                                  'shadow_inputs_dict_spec'
                                  'no_specs'}:
                assert len(ocm.state_input_ports) == 3
                assert ocm.state_input_ports.names == [shadowed_ia_node, shadowed_oa_node, shadowed_ob_node]
                assert ocm.state_features == {'IA[InputPort-0]': 'IA[InputPort-0]',
                                              'OA[InputPort-0]': 'OA[InputPort-0]',
                                              'OB[InputPort-0]': 'OB[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {ia.input_port: [0.],
                                                                                      oa.input_port: [0.],
                                                                                      ob.input_port: [0., 0., 0.]}

            if test_condition == 'single_tuple_shadow_spec':
                assert ocm.state_input_ports.names == [shadowed_ia_node, shadowed_oa_node, shadowed_ob_node]
                assert ocm.state_features == {'IA[InputPort-0]': 'IA[InputPort-0]',
                                              'OA[InputPort-0]': 'OA[InputPort-0]',
                                              'OB[InputPort-0]': 'OB[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {ia.input_port: [0.5],
                                                                                      oa.input_port: [0.5],
                                                                                      ob.input_port: [0.5, 0.5, 0.5]}
                assert all('Logistic' in port.function.name for port in ocm.state_input_ports)

            if test_condition == 'full_list_spec':
                assert len(ocm.state_input_ports) == 3
                assert ocm.state_input_ports.names == [shadowed_ia_node, oa_node, numeric_ob]
                assert ocm.state_features == {'IA[InputPort-0]': 'IA[InputPort-0]',
                                              'OA[InputPort-0]': 'OA[OutputPort-0]',
                                              'OB[InputPort-0]': [3, 1, 2]}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {ia.input_port: [0.0],
                                                                                      oa.input_port: [0.0],
                                                                                      ob.input_port: [3.0, 1.0, 2.0]}

            if test_condition == 'list_spec_with_none':
                assert len(ocm.state_input_ports) == 2
                assert ocm.state_input_ports.names == [shadowed_ia_node, numeric_ob]
                assert ocm.state_features == {'IA[InputPort-0]': 'IA[InputPort-0]',
                                              'OA[InputPort-0]': None,
                                              'OB[InputPort-0]': [3, 1, 2]}
                assert all(np.allclose(expected, actual)
                           for expected, actual in zip(list(ocm.state_feature_values.values()),
                                                       [[0.], [3, 1, 2]]))

            elif test_condition in {'input_dict_spec', 'input_dict_spec_short'}:
                assert len(ocm.state_input_ports) == 3
                assert ocm.state_input_ports.names == [shadowed_ia_node, shadowed_oa_oc, ob_node]
                assert ocm.state_features == {'IA[InputPort-0]': 'IA[InputPort-0]',
                                              'OA[InputPort-0]': 'OC[InputPort-0]',
                                              'OB[InputPort-0]': 'OB[OutputPort-0]'}
                assert all(np.allclose(expected, actual)
                           for expected, actual in zip(list(ocm.state_feature_values.values()),
                                                       [[0.], [0.], [0, 0, 0]]))

            elif test_condition == 'set_spec_short':
                assert len(ocm.state_input_ports) == 1
                assert ocm.state_input_ports.names == [shadowed_oa_node]
                # 'set_spec': {ob, icomp, oa},  # Note: out of order is OK
                assert ocm.state_features == {'IA[InputPort-0]': None,
                                              'OA[InputPort-0]': 'OA[InputPort-0]',
                                              'OB[InputPort-0]': None}
                assert all(np.allclose(expected, actual)
                           for expected, actual in zip(list(ocm.state_feature_values.values()),
                                                       [[0.], [0.], [0, 0, 0]]))

        elif test_condition == 'shadow_inputs_dict_spec_w_none':
            assert len(ocm.state_input_ports) == 2
            assert ocm.state_input_ports.names == [shadowed_ia_node, shadowed_ob_node]
            assert ocm.state_features == {'IA[InputPort-0]': 'IA[InputPort-0]',
                                          'OA[InputPort-0]': None,
                                          'OB[InputPort-0]': 'OB[InputPort-0]'}
            assert all(np.allclose(expected, actual)
                       for expected, actual in zip(list(ocm.state_feature_values.values()),
                                                   [[0.], [0.], [0, 0, 0]]))

        elif exception_type is UserWarning:
            # These also produce errors, tested below
            if test_condition in {'too_many_inputs_warning',
                                  'too_many_w_node_not_in_composition_warning',
                                  'bad_dict_spec_warning'}:
                with pytest.warns(UserWarning) as warning:
                    ocomp.add_controller(ocm)
                    assert error_or_warning_message in [warning[i].message.args[0] for i in range(len(warning))]
            else:
                with pytest.warns(UserWarning) as warning:
                    ocomp.add_controller(ocm)
                    ocomp.run()
                    if test_condition == 'partial_legal_list_spec':
                        assert len(ocm.state_input_ports) == 3
                        assert ocm.state_input_ports.names == [ia_node, shadowed_oa_node, shadowed_ob_node]
                        # Note: oa is assigned to icomp due to ordering:
                        assert ocm.state_features == {'IA[InputPort-0]': 'OA[OutputPort-0]',
                                                      'OA[InputPort-0]': 'OA[InputPort-0]',
                                                      'OB[InputPort-0]': 'OB[InputPort-0]'}
                        assert all(np.allclose(expected, actual)
                                   for expected, actual in zip(list(ocm.state_feature_values.values()),
                                                               [[0.], [0.], [0, 0, 0]]))

                assert error_or_warning_message in [warning[i].message.args[0] for i in range(len(warning))]

        else:
            with pytest.raises(exception_type) as error:
                ocomp.add_controller(ocm)
                ocomp.run()
            assert error_or_warning_message in str(error.value)

    state_features_arg = [
        'single_numeric_spec',        # <- same numeric input for all INPUT Node InputPorts
        'single_tuple_numeric_spec',  # <- same value and function assigned to all INPUT Node InputPorts
        'single_port_spec',           # <- same Port for all INPUT Node InputPorts
        'single_mech_spec',           # <- same Mech's InputPort for INPUT Node InputPorts
        'nested_partial_list',        # <- specify 1st 3 INPUT Node InputPorts; 4th (I2) should get shaddowed
        'nested_partial_set',         # <- only 2 of 3 INPUT Nodes of nested Comp in set format;
        'nested_partial_dict',        # <- only 2 of 3 INPUT Nodes of nested Comp in dict format
        'nested_full_set',            # <- all 3 INPUT Nodes of nested Comp in set format
        'nested_full_dict',           # <- both of 2 INPUT Nodes of nested Comp in dict format
        'nested_comp_set',            # <- nested Comp as itself in set format
        'nested_comp_dict',           # <- nested Comp as itself in dict format with a single spec for all INPUT Nodes
        'no_spec',                    # <- Assign state_feature_default to all Nodes
        'bad'                         # <- Mechanism not in agent_rep
    ]
    if len(state_feature_args) != 13:
        print("\n\n**********************************************************************************************")
        print("*** RESTORE state_feature_args IN test_state_features_in_nested_composition_as_agent_rep() *****")
        print("***********************************************************************************************")
    @pytest.mark.state_features
    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.parametrize('nested_agent_rep',[(False, 'OUTER COMP'),(True, 'MIDDLE COMP')],
                             ids=['unnested','nested'])
    @pytest.mark.parametrize('state_features_arg', state_features_arg,
                             ids= [f"state_feature-{x}" for x in state_features_arg]
                             )
    def test_state_features_in_nested_composition_as_agent_rep(self, nested_agent_rep, state_features_arg):
        """Test state_features for agent_rep that is a nested Composition and also has one nested with in.
        Also test for single state_feature_spec and INPUT Node with multiple InputPorts, not tested in
        test_ocm_state_feature_specs_and_warnings_and_errors() (see that for tests of basic state_features specs).
        """

        I1 = pnl.ProcessingMechanism(name='I1')
        I2 = pnl.ProcessingMechanism(name='I2')
        icomp = pnl.Composition(nodes=[I1,I2], name='INNER COMP')
        A = pnl.ComparatorMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')
        C = pnl.ProcessingMechanism(name='C', size=3)
        D = pnl.ProcessingMechanism(name='D')
        mcomp = pnl.Composition(pathways=[[A,B,C], icomp], name='MIDDLE COMP')
        ocomp = pnl.Composition(nodes=[mcomp], name='OUTER COMP')

        # Test args:
        if nested_agent_rep is True:
            agent_rep = mcomp
            error_text = f"'OCM' has '{pnl.STATE_FEATURES}' specified (['D[OutputPort-0]']) that are missing from both " \
                         f"its `agent_rep` ('{nested_agent_rep[1]}') as well as 'OUTER COMP' and any " \
                         f"Compositions nested within it."
        else:
            agent_rep = None
            error_text = f"'OCM' has '{pnl.STATE_FEATURES}' specified (['INPUT FROM D[OutputPort-0] FOR A[SAMPLE]']) " \
                         f"that are missing from 'OUTER COMP' and any Compositions nested within it."

        state_features = {
            'single_numeric_spec': [3],
            'single_tuple_numeric_spec': ([3], pnl.Linear(slope=5)),
            'single_port_spec': I1.output_port,
            'single_mech_spec': I1,
            'nested_partial_list': [I1.output_port, [2], I2],
            'nested_partial_set': {A.input_ports[pnl.SAMPLE], I2},
            'nested_full_set': {A, I1, I2},
            'nested_partial_dict': {A.input_ports[pnl.SAMPLE]:[3.5], I2:I1.input_port},
            'nested_full_dict': {A:A.input_port, I1:I2.input_port, I2:I1.input_port},
            'nested_comp_set': {mcomp},
            'nested_comp_dict': {mcomp: I1},
            'no_spec': 'SHOULD SHADOW INPUT Node InputPorts',
            'bad': [D.output_port]
        }[state_features_arg]

        if state_features_arg == 'nested_partial_set':
            state_feature_default = None  # Test assignment of SHADOW_INPUTS to specs in set, and None to others
        else:
            state_feature_default = pnl.SHADOW_INPUTS

        if state_features_arg == 'no_spec':
            ocm = pnl.OptimizationControlMechanism(name='OCM',
                                                   agent_rep=agent_rep,
                                                   objective_mechanism=pnl.ObjectiveMechanism(monitor=[B]),
                                                   allow_probes=True,
                                                   function=pnl.GridSearch(),
                                                   control_signals=pnl.ControlSignal(modulates=(pnl.SLOPE,I1),
                                                                                     allocation_samples=[10, 20, 30]))
        else:
            ocm = pnl.OptimizationControlMechanism(name='OCM',
                                                   agent_rep=agent_rep,
                                                   state_features=state_features,
                                                   state_feature_default=state_feature_default,
                                                   objective_mechanism=pnl.ObjectiveMechanism(monitor=[B]),
                                                   allow_probes=True,
                                                   function=pnl.GridSearch(),
                                                   control_signals=pnl.ControlSignal(modulates=(pnl.SLOPE,I1),
                                                                                     allocation_samples=[10, 20, 30]))
        if state_features_arg == 'bad':
            with pytest.raises(pnl.OptimizationControlMechanismError) as error:
                ocomp.add_controller(ocm)
                ocomp.run()
            assert error_text in str(error.value)
        else:
            ocomp.add_controller(ocm)
            ocomp.run()

            if state_features_arg == 'single_numeric_spec':
                assert ocm.state_features == {'A[SAMPLE]': [3],
                                              'A[TARGET]': [3],
                                              'I1[InputPort-0]': [3],
                                              'I2[InputPort-0]': [3]}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [3],
                                                                                      A.input_ports[pnl.TARGET]: [3],
                                                                                      I1.input_port: [3],
                                                                                      I2.input_port: [3]}
            elif state_features_arg == 'single_tuple_numeric_spec':
                assert ocm.state_features == {'A[SAMPLE]': [3],
                                              'A[TARGET]': [3],
                                              'I1[InputPort-0]': [3],
                                              'I2[InputPort-0]': [3]}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [15],
                                                                                      A.input_ports[pnl.TARGET]: [15],
                                                                                      I1.input_port: [15],
                                                                                      I2.input_port: [15]}
            elif state_features_arg in {'single_port_spec'}:
                assert ocm.state_features == {'A[SAMPLE]': 'I1[OutputPort-0]',
                                              'A[TARGET]': 'I1[OutputPort-0]',
                                              'I1[InputPort-0]': 'I1[OutputPort-0]',
                                              'I2[InputPort-0]': 'I1[OutputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [0],
                                                                                      A.input_ports[pnl.TARGET]: [0],
                                                                                      I1.input_port: [0],
                                                                                      I2.input_port: [0]}
            elif state_features_arg in {'single_mech_spec'}:
                assert ocm.state_features == {'A[SAMPLE]': 'I1[InputPort-0]',
                                              'A[TARGET]': 'I1[InputPort-0]',
                                              'I1[InputPort-0]': 'I1[InputPort-0]',
                                              'I2[InputPort-0]': 'I1[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [0],
                                                                                      A.input_ports[pnl.TARGET]: [0],
                                                                                      I1.input_port: [0],
                                                                                      I2.input_port: [0]}
            elif state_features_arg in 'nested_partial_list':
                assert ocm.state_features == {'A[SAMPLE]': 'I1[OutputPort-0]',
                                              'A[TARGET]': [2],
                                              'I1[InputPort-0]': 'I2[InputPort-0]',
                                              'I2[InputPort-0]': 'I2[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [0],
                                                                                      A.input_ports[pnl.TARGET]: [2],
                                                                                      I1.input_port: [0],
                                                                                      I2.input_port: [0]}
            elif state_features_arg in 'nested_partial_set':
                assert ocm.state_features == {'A[SAMPLE]': 'A[SAMPLE]',
                                              'A[TARGET]': None,
                                              'I1[InputPort-0]': None,
                                              'I2[InputPort-0]': 'I2[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [0],
                                                                                      I2.input_port: [0]}

            elif state_features_arg == 'nested_partial_dict':
                assert ocm.state_features == {'A[SAMPLE]': [3.5],
                                              'A[TARGET]': 'A[TARGET]',
                                              'I1[InputPort-0]': 'I1[InputPort-0]',
                                              'I2[InputPort-0]': 'I1[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [3.5],
                                                                                      A.input_ports[pnl.TARGET]: [0],
                                                                                      I1.input_port: [0],
                                                                                      I2.input_port: [0]}
            elif state_features_arg in {'nested_full_set', 'nested_comp_set', 'no_spec'}:
                assert ocm.state_features == {'A[SAMPLE]': 'A[SAMPLE]',
                                              'A[TARGET]': 'A[TARGET]',
                                              'I1[InputPort-0]': 'I1[InputPort-0]',
                                              'I2[InputPort-0]': 'I2[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [0],
                                                                                      A.input_ports[pnl.TARGET]: [0],
                                                                                      I1.input_port: [0],
                                                                                      I2.input_port: [0]}
            elif state_features_arg == 'nested_full_dict':
                assert ocm.state_features == {'A[SAMPLE]': 'A[SAMPLE]',
                                              'A[TARGET]': 'A[SAMPLE]',
                                              'I1[InputPort-0]': 'I2[InputPort-0]',
                                              'I2[InputPort-0]': 'I1[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[0]: [0],
                                                                                      A.input_ports[1]: [0],
                                                                                      I1.input_port: [0],
                                                                                      I2.input_port: [0]}
            elif state_features_arg == 'nested_comp_dict':
                assert ocm.state_features == {'A[SAMPLE]': 'I1[InputPort-0]',
                                              'A[TARGET]': 'I1[InputPort-0]',
                                              'I1[InputPort-0]': 'I1[InputPort-0]',
                                              'I2[InputPort-0]': 'I1[InputPort-0]'}
                assert {k:v.tolist() for k,v in ocm.state_feature_values.items()} == {A.input_ports[pnl.SAMPLE]: [0],
                                                                                      A.input_ports[pnl.TARGET]: [0],
                                                                                      I1.input_port: [0],
                                                                                      I2.input_port: [0]}

    @pytest.mark.state_features
    @pytest.mark.control
    @pytest.mark.parametrize('state_fct_assignments', [
        'partial_w_dict',
        'partial_w_params_dict',
        'tuple_override_dict',
        'tuple_override_params_dict',
        'port_spec_dict_in_feat_dict',
        'all',
        None
    ])
    def test_state_feature_function_specs(self, state_fct_assignments):
        """Test assignment of state_feature_functions in various configurations
        Also test use of InputPort specification dictionary as state_feature_specification"""

        fct_a = pnl.AdaptiveIntegrator
        fct_b = pnl.Buffer(history=2)
        fct_c = pnl.SimpleIntegrator
        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')
        C = pnl.ProcessingMechanism(name='C')
        R = pnl.ProcessingMechanism(name='D')

        if state_fct_assignments == 'partial_w_dict':
            state_features = [{pnl.PROJECTIONS: A, # Note: specification of A in dict is still interpreted as shadowing
                               pnl.FUNCTION: fct_a},
                              (B, fct_b),
                              C]
            state_feature_function = fct_c
        elif state_fct_assignments == 'partial_w_params_dict':
            state_features = [{pnl.PARAMS: {pnl.PROJECTIONS: A,
                                            pnl.FUNCTION: fct_a}},
                              (B, fct_b),
                              C]
            state_feature_function = fct_c
        elif state_fct_assignments == 'tuple_override_dict':
            state_features = [({pnl.PROJECTIONS: A,
                                pnl.FUNCTION: pnl.Buffer},
                               fct_a),
                              (B, fct_b),
                              C]
            state_feature_function = fct_c
        elif state_fct_assignments == 'tuple_override_params_dict':
            state_features = [({pnl.PARAMS: {pnl.PROJECTIONS: A,
                                             pnl.FUNCTION: pnl.Buffer}}, fct_a),
                              (B, fct_b),
                              C]
            state_feature_function = fct_c
        elif state_fct_assignments == 'port_spec_dict_in_feat_dict':
            state_features = {A:{pnl.PROJECTIONS: A,
                                 pnl.FUNCTION: fct_a},
                              B: ({pnl.PROJECTIONS: B}, fct_b),
                              C: C}
            state_feature_function = fct_c
        elif state_fct_assignments == 'all':
            state_features = [(A, fct_a), (B, fct_b), (C, fct_c)]
            state_feature_function = None
        else:
            state_features = [A, B, C]
            state_feature_function = None

        comp = pnl.Composition(name='comp', pathways=[[A,R],[B,R],[C,R]])
        ocm = pnl.OptimizationControlMechanism(state_features=state_features,
                                               state_feature_function=state_feature_function,
                                               function=pnl.GridSearch(),
                                               # monitor_for_control=A,
                                               control_signals=[pnl.ControlSignal(modulates=(pnl.SLOPE, A),
                                                                                  allocation_samples=[10, 20, 30])])
        comp.add_controller(ocm)
        if state_fct_assignments:
            assert isinstance(ocm.state_input_ports[0].function, fct_a)
            assert isinstance(ocm.state_input_ports[1].function, fct_b.__class__)
            assert isinstance(ocm.state_input_ports[2].function, fct_c)
            inputs = {A:[1,2], B:[1,2], C:[1,2]}
            result = comp.run(inputs=inputs, context='test')
            assert result == [[24.]]
            assert all(np.allclose(actual, expected)
                       for actual, expected in zip(list(ocm.parameters.state_feature_values.get('test').values()),
                                                   [[2],[[1],[2]],[3]]))
        else:
            assert isinstance(ocm.state_input_ports[0].function, pnl.LinearCombination)
            assert isinstance(ocm.state_input_ports[1].function, pnl.LinearCombination)
            assert isinstance(ocm.state_input_ports[2].function, pnl.LinearCombination)
            inputs = {A:[1,2], B:[1,2], C:[1,2]}
            result = comp.run(inputs=inputs, context='test')
            assert result == [[24.]]
            assert all(np.allclose(expected, actual)
                       for actual, expected in zip(list(ocm.parameters.state_feature_values.get('test').values()),
                                                   [[2],[2],[2]]))

    @pytest.mark.state_features
    @pytest.mark.control
    def test_ocm_state_and_state_dict(self):
        ia = pnl.ProcessingMechanism(name='IA')
        ib = pnl.ProcessingMechanism(name='IB')
        ic = pnl.ProcessingMechanism(name='IC')
        oa = pnl.ProcessingMechanism(name='OA')
        ob = pnl.ProcessingMechanism(name='OB', size=3)
        oc = pnl.ProcessingMechanism(name='OC')
        icomp = pnl.Composition(pathways=[ia,ib,ic], name='INNER COMP')
        ocomp = pnl.Composition(pathways=[icomp], name='OUTER COMP')
        ocomp.add_linear_processing_pathway([oa,oc])
        ocomp.add_linear_processing_pathway([ob,oc])
        ocm = pnl.OptimizationControlMechanism(
            state_features=[ia.input_port,
                            oa.output_port,
                            [3,1,2]], # <- ob
            objective_mechanism=[ic,ib],
            function=pnl.GridSearch(),
            allow_probes=True,
            control_signals=[pnl.ControlSignal(modulates=(pnl.SLOPE,ia),
                                          allocation_samples=[10, 20, 30]),
                             pnl.ControlSignal(modulates=[(pnl.INTERCEPT,oc),(pnl.SLOPE, oc)],
                                          allocation_samples=[10, 20, 30]),
                             ]
        )
        ocomp.add_controller(ocm)
        assert all(np.allclose(x,y) for x,y in zip(ocm.state, [[0.0], [0.0], [3.0, 1.0, 2.0], [1.0], [1.0]]))
        assert len(ocm.state_distal_sources_and_destinations_dict) == 6
        keys = list(ocm.state_distal_sources_and_destinations_dict.keys())
        values = list(ocm.state_distal_sources_and_destinations_dict.values())
        for key, value in ocm.state_distal_sources_and_destinations_dict.items():
            ocm.state[key[3]] == value
        assert keys[0] == (ia.input_port, ia, icomp ,0)
        assert keys[1] == (oa.output_port, oa, ocomp, 1)
        assert keys[2] == ('default_variable', None, None, 2)
        assert keys[3] == (ia.parameter_ports[pnl.SLOPE], ia, icomp, 3)
        assert keys[4] == (oc.parameter_ports[pnl.INTERCEPT], oc, ocomp, 4)
        assert keys[5] == (oc.parameter_ports[pnl.SLOPE], oc, ocomp, 4)
        ocomp.run()
        assert all(np.allclose(expected, actual)
                   for expected, actual in zip(list(ocm.state_feature_values.values()),
                                               [[0.], [0.], [3, 1, 2]]))

    def test_modulation_of_control_signal_intensity_cost_function_MULTIPLICATIVE(self):
        # tests multiplicative modulation of default intensity_cost_function (Exponential) of
        #    a ControlMechanism's default function (TransferWithCosts);
        #    intensity_cost should = e ^ (allocation (3) * value of ctl_mech_B (also 3)) = e^9
        mech = pnl.ProcessingMechanism()
        ctl_mech_A = pnl.ControlMechanism(monitor_for_control=mech,
                                      control_signals=pnl.ControlSignal(modulates=(pnl.INTERCEPT,mech),
                                                                        cost_options=pnl.CostFunctions.INTENSITY))
        ctl_mech_B = pnl.ControlMechanism(monitor_for_control=mech,
                                          control_signals=pnl.ControlSignal(
                                                              modulates=ctl_mech_A.control_signals[0],
                                                              modulation=pnl.INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM))
        comp = pnl.Composition()
        comp.add_linear_processing_pathway(pathway=[mech,
                                                    ctl_mech_A,
                                                    ctl_mech_B
                                                    ])

        comp.run(inputs={mech:[3]}, num_trials=2)
        assert np.allclose(ctl_mech_A.control_signals[0].intensity_cost, 8103.083927575384008)

    def test_feedback_assignment_for_multiple_control_projections_to_same_mechanism(self):
        """Test that multiple ControlProjections from a ControlMechanism to the same Mechanism are treated
        same as a single ControlProjection to that Mechanism.
        Note: Even though both mech and control_mech don't receive pathway inputs, since control_mech projects to mech,
        control_mech is assigned as NodeRole.INPUT (can be overridden with assignments in add_nodes)
        """
        mech = pnl.ProcessingMechanism(input_ports=['A','B','C'])
        control_mech = pnl.ControlMechanism(control=mech.input_ports[0])
        comp = pnl.Composition()
        comp.add_nodes([mech, control_mech])
        result = comp.run(inputs={control_mech:[2]}, num_trials=3)
        # assert np.allclose(result, [[2],[2],[2]])
        assert pnl.NodeRole.INPUT not in comp.get_roles_by_node(mech)
        assert pnl.NodeRole.INPUT in comp.get_roles_by_node(control_mech)

        # Should produce same result as above
        mech = pnl.ProcessingMechanism(input_ports=['A','B','C'])
        control_mech = pnl.ControlMechanism(control=mech.input_ports) # Note multiple parallel ControlProjections
        comp = pnl.Composition()
        comp.add_nodes([mech, control_mech])
        comp.run(inputs={control_mech:[2]}, num_trials=3)
        assert pnl.NodeRole.INPUT not in comp.get_roles_by_node(mech)
        assert pnl.NodeRole.INPUT in comp.get_roles_by_node(control_mech)

    def test_modulation_of_control_signal_intensity_cost_function_ADDITIVE(self):
        # tests additive modulation of default intensity_cost_function (Exponential) of
        #    a ControlMechanism's default function (TransferWithCosts)
        #    intensity_cost should = e ^ (allocation (3) + value of ctl_mech_B (also 3)) = e^6
        mech = pnl.ProcessingMechanism()
        ctl_mech_A = pnl.ControlMechanism(monitor_for_control=mech,
                                      control_signals=pnl.ControlSignal(modulates=(pnl.INTERCEPT,mech),
                                                                        cost_options=pnl.CostFunctions.INTENSITY))
        ctl_mech_B = pnl.ControlMechanism(monitor_for_control=mech,
                                          control_signals=pnl.ControlSignal(
                                                              modulates=ctl_mech_A.control_signals[0],
                                                              modulation=pnl.INTENSITY_COST_FCT_ADDITIVE_PARAM))
        comp = pnl.Composition()
        comp.add_linear_processing_pathway(pathway=[mech,
                                                    ctl_mech_A,
                                                    ctl_mech_B
                                                    ])

        comp.run(inputs={mech:[3]}, num_trials=2)
        assert np.allclose(ctl_mech_A.control_signals[0].intensity_cost, 403.428793492735123)

    def test_lvoc(self):
        m1 = pnl.TransferMechanism(input_ports=["InputPort A", "InputPort B"])
        m2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_node(m1, required_roles=pnl.NodeRole.INPUT)
        c.add_node(m2, required_roles=pnl.NodeRole.INPUT)
        c._analyze_graph()
        lvoc = pnl.OptimizationControlMechanism(agent_rep=pnl.RegressionCFA,
                                                state_features=[m1.input_ports[0], m1.input_ports[1], m2.input_port],
                                                objective_mechanism=pnl.ObjectiveMechanism(
                                                    monitor=[m1, m2]),
                                                function=pnl.GridSearch(max_iterations=1),
                                                control_signals=[
                                                    {CONTROL: (pnl.SLOPE, m1),
                                                     ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                                                    {CONTROL: (pnl.SLOPE, m2),
                                                     ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}])
        c.add_node(lvoc)
        input_dict = {m1: [[1], [1]], m2: [1]}

        c.run(inputs=input_dict)

        assert len(lvoc.input_ports) == 4

    def test_lvoc_both_predictors_specs(self):
        m1 = pnl.TransferMechanism(input_ports=["InputPort A", "InputPort B"])
        m2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_node(m1, required_roles=pnl.NodeRole.INPUT)
        c.add_node(m2, required_roles=pnl.NodeRole.INPUT)
        c._analyze_graph()
        lvoc = pnl.OptimizationControlMechanism(agent_rep=pnl.RegressionCFA,
                                                state_features=[m1.input_ports[0], m1.input_ports[1], m2.input_port, m2],
                                                objective_mechanism=pnl.ObjectiveMechanism(
                                                    monitor=[m1, m2]),
                                                function=pnl.GridSearch(max_iterations=1),
                                                control_signals=[
                                                    {CONTROL: (pnl.SLOPE, m1),
                                                     ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                                                    {CONTROL: (pnl.SLOPE, m2),
                                                     ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}])
        c.add_node(lvoc)
        input_dict = {m1: [[1], [1]], m2: [1]}

        c.run(inputs=input_dict)

        assert len(lvoc.input_ports) == 5

    def test_lvoc_features_function(self):
        m1 = pnl.TransferMechanism(input_ports=["InputPort A", "InputPort B"])
        m2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_node(m1, required_roles=pnl.NodeRole.INPUT)
        c.add_node(m2, required_roles=pnl.NodeRole.INPUT)
        c._analyze_graph()
        lvoc = pnl.OptimizationControlMechanism(agent_rep=pnl.RegressionCFA,
                                                state_features=[m1.input_ports[0], m1.input_ports[1], m2.input_port, m2],
                                                state_feature_function=pnl.LinearCombination(offset=10.0),
                                                objective_mechanism=pnl.ObjectiveMechanism(
                                                    monitor=[m1, m2]),
                                                function=pnl.GradientOptimization(max_iterations=1),
                                                control_signals=[(pnl.SLOPE, m1), (pnl.SLOPE, m2)])
        c.add_node(lvoc)
        input_dict = {m1: [[1], [1]], m2: [1]}

        c.run(inputs=input_dict)

        assert len(lvoc.input_ports) == 5

        for i in range(1,5):
            assert lvoc.input_ports[i].function.offset == 10.0

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Multilevel GridSearch")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python])
    def test_multilevel_ocm_gridsearch_conflicting_directions(self, mode, benchmark):
        oa = pnl.TransferMechanism(name='oa')
        ob = pnl.TransferMechanism(name='ob')
        ocomp = pnl.Composition(name='ocomp', controller_mode=pnl.BEFORE)
        ia = pnl.TransferMechanism(name='ia')
        ib = pnl.ProcessingMechanism(name='ib',
                                     function=lambda x: abs(x - 75))
        icomp = pnl.Composition(name='icomp', controller_mode=pnl.BEFORE)
        ocomp.add_node(oa, required_roles=pnl.NodeRole.INPUT)
        ocomp.add_node(ob)
        ocomp.add_node(icomp)
        icomp.add_node(ia, required_roles=pnl.NodeRole.INPUT)
        icomp.add_node(ib)
        ocomp._analyze_graph()
        icomp._analyze_graph()
        ocomp.add_projection(pnl.MappingProjection(), sender=oa, receiver=ia)
        icomp.add_projection(pnl.MappingProjection(), sender=ia, receiver=ib)
        ocomp.add_projection(pnl.MappingProjection(), sender=ib, receiver=ob)

        ocomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=ocomp,
                state_features=[oa.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MINIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0, stop=5.0, num=5))])
        )
        icomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=icomp,
                state_features=[ia.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0, stop=5.0, num=5))])
        )
        results = ocomp.run([5], execution_mode=mode)
        assert np.allclose(results, [[50]])

        if benchmark.enabled:
            benchmark(ocomp.run, [5], execution_mode=mode)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Multilevel GridSearch")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python])
    def test_multilevel_ocm_gridsearch_maximize(self, mode, benchmark):
        oa = pnl.TransferMechanism(name='oa')
        ob = pnl.TransferMechanism(name='ob')
        ocomp = pnl.Composition(name='ocomp', controller_mode=pnl.BEFORE)
        ia = pnl.TransferMechanism(name='ia')
        ib = pnl.ProcessingMechanism(name='ib',
                                     function=lambda x: abs(x - 75))
        icomp = pnl.Composition(name='icomp', controller_mode=pnl.BEFORE)
        ocomp.add_node(oa, required_roles=pnl.NodeRole.INPUT)
        ocomp.add_node(ob)
        ocomp.add_node(icomp)
        icomp.add_node(ia, required_roles=pnl.NodeRole.INPUT)
        icomp.add_node(ib)
        ocomp._analyze_graph()
        icomp._analyze_graph()
        ocomp.add_projection(pnl.MappingProjection(), sender=oa, receiver=ia)
        icomp.add_projection(pnl.MappingProjection(), sender=ia, receiver=ib)
        ocomp.add_projection(pnl.MappingProjection(), sender=ib, receiver=ob)

        ocomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=ocomp,
                state_features=[oa.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                     stop=5.0,
                                                                                     num=5))])
        )
        icomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=icomp,
                state_features=[ia.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                     stop=5.0,
                                                                                     num=5))])
        )
        results = ocomp.run([5], execution_mode=mode)
        assert np.allclose(results, [[70]])

        if benchmark.enabled:
            benchmark(ocomp.run, [5], execution_mode=mode)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Multilevel GridSearch")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python])
    def test_multilevel_ocm_gridsearch_minimize(self, mode, benchmark):
        oa = pnl.TransferMechanism(name='oa')
        ob = pnl.TransferMechanism(name='ob')
        ocomp = pnl.Composition(name='ocomp', controller_mode=pnl.BEFORE)
        ia = pnl.TransferMechanism(name='ia')
        ib = pnl.ProcessingMechanism(name='ib',
                                     function=lambda x: abs(x - 70))
        icomp = pnl.Composition(name='icomp', controller_mode=pnl.BEFORE)
        ocomp.add_node(oa, required_roles=pnl.NodeRole.INPUT)
        ocomp.add_node(ob)
        ocomp.add_node(icomp)
        icomp.add_node(ia, required_roles=pnl.NodeRole.INPUT)
        icomp.add_node(ib)
        ocomp._analyze_graph()
        icomp._analyze_graph()
        ocomp.add_projection(pnl.MappingProjection(), sender=oa, receiver=ia)
        icomp.add_projection(pnl.MappingProjection(), sender=ia, receiver=ib)
        ocomp.add_projection(pnl.MappingProjection(), sender=ib, receiver=ob)

        ocomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=ocomp,
                state_features=[oa.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="oController",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MINIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                     stop=5.0,
                                                                                     num=5))])
        )
        icomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=icomp,
                state_features=[ia.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="iController",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="iController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MINIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                     stop=5.0,
                                                                                     num=5))])
        )
        results = ocomp.run([5], execution_mode=mode)
        assert np.allclose(results, [[5]])

        if benchmark.enabled:
            benchmark(ocomp.run, [5], execution_mode=mode)

    def test_two_tier_ocm(self):
        integrationConstant = 0.8  # Time Constant
        DRIFT = 0.25  # Drift Rate
        STARTING_VALUE = 0.0  # Starting Point
        THRESHOLD = 0.05  # Threshold
        NOISE = 0.1  # Noise
        T0 = 0.2  # T0
        congruentWeight = 0.2

        # Task Layer: [Color, Motion] {0, 1} Mutually Exclusive
        taskLayer = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                          # size=2,
                                          function=pnl.Linear(slope=1, intercept=0),
                                          output_ports=[pnl.RESULT],
                                          name='Task Input [I1, I2]')

        # Stimulus Layer: [Color Stimulus, Motion Stimulus]
        stimulusInfo = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                             # size=2,
                                             function=pnl.Linear(slope=1, intercept=0),
                                             output_ports=[pnl.RESULT],
                                             name="Stimulus Input [S1, S2]")

        congruenceWeighting = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                                    size=2,
                                                    function=pnl.Linear(slope=congruentWeight, intercept=0),
                                                    name='Congruence * Automatic Component')

        # Activation Layer: [Color Activation, Motion Activation]
        activation = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
                                                    function=pnl.Logistic(gain=1.0),
                                                    matrix=[[1.0, -1.0],
                                                            [-1.0, 1.0]],
                                                    integrator_mode=True,
                                                    integrator_function=pnl.AdaptiveIntegrator(
                                                        rate=integrationConstant),
                                                    initial_value=np.array([[0.0, 0.0]]),
                                                    output_ports=[pnl.RESULT],
                                                    name='Task Activations [Act1, Act2]')

        activation.set_log_conditions([pnl.RESULT, "mod_gain"])

        # Hadamard product of Activation and Stimulus Information
        nonAutomaticComponent = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                                      size=2,
                                                      function=pnl.Linear(slope=1, intercept=0),
                                                      input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                                      output_ports=[pnl.RESULT],
                                                      name='Non-Automatic Component')

        # Summation of nonAutomatic and Automatic Components
        ddmCombination = pnl.TransferMechanism(size=1,
                                               function=pnl.Linear(slope=1, intercept=0),
                                               input_ports=pnl.InputPort(combine=pnl.SUM),
                                               output_ports=[pnl.RESULT],
                                               name="Drift = Wa*(S1 + S2) + (S1*Act1 + S2*Act2)")

        decisionMaker = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=DRIFT,
                                                                      starting_value=STARTING_VALUE,
                                                                      threshold=THRESHOLD,
                                                                      noise=NOISE,
                                                                      non_decision_time=T0),
                                output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME,
                                               pnl.PROBABILITY_UPPER_THRESHOLD,
                                               pnl.PROBABILITY_LOWER_THRESHOLD],
                                name='DDM')

        weightingFunction = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                                  size=2,
                                                  function=pnl.Linear(slope=1, intercept=0),
                                                  input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                                  output_ports=[pnl.RESULT],
                                                  name='Bias')

        topCorrect = pnl.TransferMechanism(size=1,
                                           function=pnl.Linear(slope=1, intercept=0),
                                           input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                           output_ports=[pnl.RESULT],
                                           name="weightDDMInput")

        stabilityFlexibility = pnl.Composition(name='inner', controller_mode=pnl.BEFORE)

        # Linear pathway from the task input origin node to the DDM
        stabilityFlexibility.add_linear_processing_pathway(pathway=[taskLayer,
                                                                    activation,
                                                                    nonAutomaticComponent,
                                                                    ddmCombination,
                                                                    topCorrect,
                                                                    decisionMaker])

        # Linear pathway from the stimulus input origin node to the DDM
        stabilityFlexibility.add_linear_processing_pathway(pathway=[stimulusInfo,
                                                                    nonAutomaticComponent,
                                                                    ddmCombination,
                                                                    topCorrect,
                                                                    decisionMaker])

        # Linear pathway from the stimulus input origin node to the DDM with congruence
        stabilityFlexibility.add_linear_processing_pathway(pathway=[stimulusInfo,
                                                                    congruenceWeighting,
                                                                    ddmCombination,
                                                                    topCorrect,
                                                                    decisionMaker])

        stabilityFlexibility.add_linear_processing_pathway(pathway=[taskLayer,
                                                                    weightingFunction,
                                                                    topCorrect,
                                                                    decisionMaker])

        stabilityFlexibility.add_linear_processing_pathway(pathway=[stimulusInfo,
                                                                    weightingFunction,
                                                                    topCorrect,
                                                                    decisionMaker])

        stabilityFlexibility.add_controller(
            pnl.OptimizationControlMechanism(agent_rep=stabilityFlexibility,
                                             state_features=[taskLayer.input_port,
                                                             stimulusInfo.input_port],
                                             name="Controller",
                                             objective_mechanism=pnl.ObjectiveMechanism(
                                                 monitor=[(pnl.PROBABILITY_UPPER_THRESHOLD,
                                                           decisionMaker)],
                                                 function=pnl.SimpleIntegrator,
                                                 name="Controller Objective Mechanism"),
                                             function=pnl.GridSearch(),
                                             control_signals=[pnl.ControlSignal(
                                                 projections=[(pnl.GAIN, activation)],
                                                 variable=1.0,
                                                 intensity_cost_function=pnl.Linear(
                                                     slope=0.0),
                                                 allocation_samples=pnl.SampleSpec(
                                                     start=1.0,
                                                     stop=5.0,
                                                     num=2))]
                                             )
        )
        outerComposition = pnl.Composition(name='outer',
                                           controller_mode=pnl.AFTER,
                                           retain_old_simulation_data=True)
        outerComposition.add_node(stabilityFlexibility)
        outerComposition.add_controller(
            pnl.OptimizationControlMechanism(agent_rep=stabilityFlexibility,
                                             state_features=[taskLayer.input_port, stimulusInfo.input_port],
                                             name="OuterController",
                                             objective_mechanism=pnl.ObjectiveMechanism(
                                                 monitor=[(pnl.PROBABILITY_UPPER_THRESHOLD, decisionMaker)],
                                                 function=pnl.SimpleIntegrator,
                                                 name="Controller Objective Mechanism"),
                                             function=pnl.GridSearch(),
                                             control_signals=[
                                                 pnl.ControlSignal(
                                                     projections=[(pnl.THRESHOLD, decisionMaker)],
                                                     variable=1.0,
                                                     intensity_cost_function=pnl.Linear(
                                                         slope=0.0),
                                                     allocation_samples=pnl.SampleSpec(
                                                         start=0.5,
                                                         stop=2.0,
                                                         num=3))
                                             ])
        )
        taskTrain = [[0, 1], [0, 1], [0, 1]]
        stimulusTrain = [[1, -1], [1, -1], [1, -1]]
        zipTrain = list(zip(taskTrain, stimulusTrain))
        outerComposition.run(zipTrain)
        assert np.allclose(outerComposition.results,
                           [[[0.05], [0.42357798], [0.76941918], [0.23058082]],
                            [[0.1], [0.64721378], [0.98737278], [0.01262722]],
                            [[0.1], [0.60232676], [0.9925894], [0.0074106]]])

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Multilevel")
    def test_multilevel_control(self, comp_mode, benchmark):
        oA = pnl.TransferMechanism(
            name='OuterA',
        )
        oB = pnl.TransferMechanism(
            name='OuterB',
        )
        iA = pnl.TransferMechanism(
            name='InnerA',
        )
        iB = pnl.TransferMechanism(
            name='InnerB',
        )
        iComp = pnl.Composition(name='InnerComp')
        iComp.add_node(iA)
        iComp.add_node(iB)
        iComp.add_projection(pnl.MappingProjection(), iA, iB)
        oComp = pnl.Composition(name='OuterComp')
        oComp.add_node(oA)
        oComp.add_node(oB)
        oComp.add_node(iComp)
        oComp.add_projection(pnl.MappingProjection(), oA, iComp)
        oComp.add_projection(pnl.MappingProjection(), iB, oB)
        oController = pnl.ControlMechanism(
            name='Outer Controller',
            control_signals=[
                pnl.ControlSignal(
                    name='ControllerTransfer',
                    transfer_function=pnl.Linear(slope=2),
                    modulates=(pnl.SLOPE, iA),
                )
            ],
        )
        oComp.add_controller(oController)
        assert oComp.controller == oController
        iController = pnl.ControlMechanism(
            name='Inner Controller',
            control_signals=[
                pnl.ControlSignal(
                    name='ControllerTransfer',
                    transfer_function=pnl.Linear(slope=4),
                    modulates=(pnl.SLOPE, iB)
                )
            ],
        )
        iComp.add_controller(iController)
        assert iComp.controller == iController
        assert oComp.controller == oController
        res = oComp.run(inputs=[5], execution_mode=comp_mode)
        assert np.allclose(res, [40])

        if benchmark.enabled:
            benchmark(oComp.run, [5], execution_mode=comp_mode)

    @pytest.mark.control
    @pytest.mark.composition
    def test_recurrent_control(self, comp_mode):
        monitor = pnl.TransferMechanism(default_variable=[[0.0]],
                                    size=1,
                                    function=pnl.Linear(slope=1, intercept=0),
                                    output_ports=[pnl.RESULT],
                                    name='monitor')

        rtm = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
                                            function=pnl.Logistic(gain=1.0),
                                            matrix=[[1.0, -1.0],
                                                    [-1.0, 1.0]],
                                            integrator_mode=True,
                                            integrator_function=pnl.AdaptiveIntegrator(rate=(1)),
                                            initial_value=np.array([[0.0, 0.0]]),
                                            output_ports=[pnl.RESULT],
                                            name='rtm')

        controller = pnl.ControlMechanism(
            monitor_for_control=monitor,
            control_signals=[(pnl.NOISE, rtm)])

        comp = pnl.Composition()
        roles = [pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT]
        comp.add_node(monitor, required_roles=roles)
        comp.add_node(rtm, required_roles=roles)
        comp.add_node(controller)
        val = comp.run(inputs = {
                    monitor: [[1], [5], [1], [5]],
                    rtm: [[1,0], [1,0] ,[1,0], [1,0]]
                },
            execution_mode=comp_mode
        )
        assert np.allclose(val[0], [5])
        assert np.allclose(val[1], [0.7978996, 0.40776362])

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.parametrize("modulation, expected", [
                              (pnl.OVERRIDE, 0.375),
                              (pnl.DISABLE, 0.4375),
                              (pnl.MULTIPLICATIVE, 0.484375),
                              (pnl.ADDITIVE, 0.25),
                             ])
    def test_control_of_mech_port(self, comp_mode, modulation, expected):
        mech = pnl.TransferMechanism(termination_threshold=0.1,
                                     execute_until_finished=True,
                                     integrator_mode=True)
        control_mech = pnl.ControlMechanism(
                control_signals=pnl.ControlSignal(modulation=modulation,
                                                  modulates=(pnl.TERMINATION_THRESHOLD, mech)))
        comp = pnl.Composition()
        comp.add_nodes([(mech, pnl.NodeRole.INPUT), (control_mech, pnl.NodeRole.INPUT)])
        inputs = {mech:[[0.5]], control_mech:[0.2]}
        results = comp.run(inputs=inputs, num_trials=1, execution_mode=comp_mode)
        assert np.allclose(results, expected)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.parametrize("modulation, expected", [
                              (pnl.OVERRIDE, 0.2),
                              (pnl.DISABLE, 0.5),
                              (pnl.MULTIPLICATIVE, 0.1),
                              (pnl.ADDITIVE, 0.7),
                             ])
    def test_control_of_mech_input_port(self, comp_mode, modulation, expected):
        mech = pnl.TransferMechanism()
        control_mech = pnl.ControlMechanism(
                control_signals=pnl.ControlSignal(modulation=modulation,
                                                  modulates=mech.input_port))
        comp = pnl.Composition()
        comp.add_nodes([(mech, pnl.NodeRole.INPUT), (control_mech, pnl.NodeRole.INPUT)])
        inputs = {mech:[[0.5]], control_mech:[0.2]}
        results = comp.run(inputs=inputs, num_trials=1, execution_mode=comp_mode)
        assert np.allclose(results, expected)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.parametrize("modulation, expected", [
                              (pnl.OVERRIDE, 0.2),
                              (pnl.DISABLE, 0.5),
                              (pnl.MULTIPLICATIVE, 0.1),
                              (pnl.ADDITIVE, 0.7),
                             ])
    @pytest.mark.parametrize("specification", [ pnl.OWNER_VALUE, (pnl.OWNER_VALUE, 0)])
    def test_control_of_mech_output_port(self, comp_mode, modulation, expected, specification):
        mech = pnl.TransferMechanism(output_ports=[pnl.OutputPort(variable=specification)])
        control_mech = pnl.ControlMechanism(
                control_signals=pnl.ControlSignal(modulation=modulation,
                                                  modulates=mech.output_port))
        comp = pnl.Composition()
        comp.add_nodes([(mech, pnl.NodeRole.INPUT), (control_mech, pnl.NodeRole.INPUT)])
        inputs = {mech:[[0.5]], control_mech:[0.2]}
        results = comp.run(inputs=inputs, num_trials=1, execution_mode=comp_mode)
        assert np.allclose(results, expected)

    @pytest.mark.control
    @pytest.mark.composition
    def test_add_node_with_controller_spec_and_control_mech_but_not_a_controller(self):
        mech = pnl.ProcessingMechanism(name='MECH', function=pnl.Linear(slope=(2, pnl.CONTROL)))
        ctl = pnl.ControlMechanism(name='CONTROL MECHANISM')
        warning_msg_1 = '"OutputPort (\'ControlSignal-0\') of \'CONTROL MECHANISM\' doesn\'t have any efferent ' \
                        'Projections in \'COMPOSITION\'."'
        warning_msg_2 = '"The \'slope\' parameter of \'MECH\' is specified for control, but the Composition it is in ' \
                        '(\'COMPOSITION\') does not have a controller; if a controller is not added to COMPOSITION ' \
                        'the control specification will be ignored."'
        warning_msg_3 = '"\\nThe following Projections were specified but are not being used by Nodes in ' \
                        '\'COMPOSITION\':\\n\\tControlProjection for MECH[slope]"'
        with pytest.warns(UserWarning) as warning:
            comp = pnl.Composition(name='COMPOSITION', pathways=[ctl])
            comp.add_node(mech)
            comp.run()
        assert all(msg in [repr(w.message.args[0]) for w in warning]
                   for msg in {warning_msg_1, warning_msg_2, warning_msg_3})

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.parametrize("cost, expected, exp_values", [
        (pnl.CostFunctions.NONE, 7.0, [3, 4, 5, 6, 7]),
        (pnl.CostFunctions.INTENSITY, 3, [0.2817181715409549, -3.3890560989306495, -15.085536923187664, -48.59815003314423, -141.41315910257657]),
        (pnl.CostFunctions.ADJUSTMENT, 3, [3, 3, 3, 3, 3] ),
        (pnl.CostFunctions.INTENSITY | pnl.CostFunctions.ADJUSTMENT, 3, [0.2817181715409549, -4.389056098930649, -17.085536923187664, -51.59815003314423, -145.41315910257657]),
        (pnl.CostFunctions.DURATION, 3, [-17, -20, -23, -26, -29]),
        # FIXME: combinations with DURATION are broken
        # (pnl.CostFunctions.DURATION | pnl.CostFunctions.ADJUSTMENT, ,),
        # (pnl.CostFunctions.ALL, ,),
        pytest.param(pnl.CostFunctions.DEFAULTS, 7, [3, 4, 5, 6, 7], id="CostFunctions.DEFAULT")],
        ids=lambda x: x if isinstance(x, pnl.CostFunctions) else "")
    def test_modulation_simple(self, cost, expected, exp_values, comp_mode):
        if comp_mode != pnl.ExecutionMode.Python and cost not in {pnl.CostFunctions.NONE, pnl.CostFunctions.INTENSITY}:
            pytest.skip("Not implemented!")

        obj = pnl.ObjectiveMechanism()
        mech = pnl.ProcessingMechanism()

        comp = pnl.Composition(controller_mode=pnl.BEFORE)
        comp.add_node(mech, required_roles=pnl.NodeRole.INPUT)
        comp.add_linear_processing_pathway([mech, obj])

        comp.add_controller(
            pnl.OptimizationControlMechanism(
                objective_mechanism=obj,
                state_features=[mech.input_port],
                control_signals=pnl.ControlSignal(
                    modulates=('intercept', mech),
                    modulation=pnl.OVERRIDE,
                    allocation_samples=pnl.SampleSpec(start=1, stop=5, step=1),
                    cost_options=cost,
                ),

                # Need to specify GridSearch since save_values is False by default and we
                # going to check these values later in the test.
                function=pnl.GridSearch(save_values=True)
            )
        )

        ret = comp.run(inputs={mech: [2]}, num_trials=1, execution_mode=comp_mode)
        assert np.allclose(ret, expected)
        if comp_mode == pnl.ExecutionMode.Python:
            assert np.allclose(comp.controller.function.saved_values.flatten(), exp_values)

    @pytest.mark.benchmark
    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.parametrize('prng', ['Default', 'Philox'])
    def test_modulation_of_random_state_direct(self, comp_mode, benchmark, prng):
        # set explicit seed to make sure modulation is different
        mech = pnl.ProcessingMechanism(function=pnl.UniformDist(seed=0))
        if prng == 'Philox':
            mech.function.parameters.random_state.set(_SeededPhilox([0]))
        ctl_mech = pnl.ControlMechanism(control_signals=pnl.ControlSignal(modulates=('seed', mech),
                                                                          modulation=pnl.OVERRIDE))
        comp = pnl.Composition()
        comp.add_node(mech)
        comp.add_node(ctl_mech)

        seeds = [13, 13, 14]
        # cycle over the seeds twice setting and resetting the random state
        benchmark(comp.run, inputs={ctl_mech:seeds}, num_trials=len(seeds) * 2, execution_mode=comp_mode)

        if prng == 'Default':
            prngs = {s:np.random.RandomState([s]) for s in seeds}
            def get_val(s, dty):
                return prngs[s].uniform()
        elif prng == 'Philox':
            prngs = {s:_SeededPhilox([s]) for s in seeds}
            def get_val(s, dty):
                return prngs[s].random(dtype=dty)

        dty = np.float32 if pytest.helpers.llvm_current_fp_precision() == 'fp32' else np.float64
        expected = [get_val(s, dty) for s in seeds] * 2
        assert np.allclose(np.squeeze(comp.results[:len(seeds) * 2]), expected)

    @pytest.mark.benchmark
    @pytest.mark.control
    @pytest.mark.composition
    # 'LLVM' mode is not supported, because synchronization of compiler and
    # python values during execution is not implemented.
    @pytest.mark.usefixtures("comp_mode_no_llvm")
    @pytest.mark.parametrize('prng', ['Default', 'Philox'])
    def test_modulation_of_random_state_DDM(self, comp_mode, benchmark, prng):
        # set explicit seed to make sure modulation is different
        mech = pnl.DDM(function=pnl.DriftDiffusionIntegrator(noise=np.sqrt(5.0)),
                       reset_stateful_function_when=pnl.AtPass(0),
                       execute_until_finished=True)
        if prng == 'Philox':
            mech.function.parameters.random_state.set(_SeededPhilox([0]))
        ctl_mech = pnl.ControlMechanism(control_signals=pnl.ControlSignal(modulates=('seed-function', mech),
                                                                          modulation=pnl.OVERRIDE))
        comp = pnl.Composition()
        comp.add_node(mech, required_roles=pnl.NodeRole.INPUT)
        comp.add_node(ctl_mech)

        # Seeds are chosen to show difference in results below.
        seeds = [13, 13, 14]

        # cycle over the seeds twice setting and resetting the random state
        benchmark(comp.run, inputs={ctl_mech:seeds, mech:5.0}, num_trials=len(seeds) * 2, execution_mode=comp_mode)

        precision = pytest.helpers.llvm_current_fp_precision()
        if prng == 'Default':
            assert np.allclose(np.squeeze(comp.results[:len(seeds) * 2]), [[100, 21], [100, 23], [100, 20]] * 2)
        elif prng == 'Philox' and precision == 'fp64':
            assert np.allclose(np.squeeze(comp.results[:len(seeds) * 2]), [[100, 19], [100, 21], [100, 21]] * 2)
        elif prng == 'Philox' and precision == 'fp32':
            assert np.allclose(np.squeeze(comp.results[:len(seeds) * 2]), [[100, 17], [100, 22], [100, 20]] * 2)
        else:
            assert False, "Unknown PRNG!"

    @pytest.mark.benchmark
    @pytest.mark.control
    @pytest.mark.composition
    # 'LLVM' mode is not supported, because synchronization of compiler and
    # python values during execution is not implemented.
    @pytest.mark.usefixtures("comp_mode_no_llvm")
    @pytest.mark.parametrize('prng', ['Default', 'Philox'])
    def test_modulation_of_random_state_DDM_Analytical(self, comp_mode, benchmark, prng):
        # set explicit seed to make sure modulation is different
        mech = pnl.DDM(function=pnl.DriftDiffusionAnalytical())
        if prng == 'Philox':
            mech.parameters.random_state.set(_SeededPhilox([0]))
        ctl_mech = pnl.ControlMechanism(control_signals=pnl.ControlSignal(modulates=('seed', mech),
                                                                          modulation=pnl.OVERRIDE))
        comp = pnl.Composition()
        comp.add_node(mech, required_roles=pnl.NodeRole.INPUT)
        comp.add_node(ctl_mech)

        # Seeds are chosen to show difference in results below.
        seeds = [3, 3, 4]

        # cycle over the seeds twice setting and resetting the random state
        benchmark(comp.run, inputs={ctl_mech:seeds, mech:0.1}, num_trials=len(seeds) * 2, execution_mode=comp_mode)

        precision = pytest.helpers.llvm_current_fp_precision()
        if prng == 'Default':
            assert np.allclose(np.squeeze(comp.results[:len(seeds) * 2]), [[-1, 3.99948962], [1, 3.99948962], [-1, 3.99948962]] * 2)
        elif prng == 'Philox' and precision == 'fp64':
            assert np.allclose(np.squeeze(comp.results[:len(seeds) * 2]), [[-1, 3.99948962], [-1, 3.99948962], [1, 3.99948962]] * 2)
        elif prng == 'Philox' and precision == 'fp32':
            assert np.allclose(np.squeeze(comp.results[:len(seeds) * 2]), [[1, 3.99948978], [-1, 3.99948978], [1, 3.99948978]] * 2)
        else:
            assert False, "Unknown PRNG!"

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.parametrize("num_generators", [5])
    def test_modulation_of_random_state(self, comp_mode, num_generators):
        obj = pnl.ObjectiveMechanism()
        # Set original seed that is not used by any evaluation
        # this prevents dirty state from initialization skewing the results.
        # The alternative would be to set:
        # mech.functions.seed.base = mech.functions.seed.base
        # to reset the PRNG
        mech = pnl.ProcessingMechanism(function=pnl.UniformDist(seed=num_generators))

        comp = pnl.Composition(retain_old_simulation_data=True,
                               controller_mode=pnl.BEFORE)
        comp.add_node(mech, required_roles=pnl.NodeRole.INPUT)
        comp.add_linear_processing_pathway([mech, obj])

        comp.add_controller(
            pnl.OptimizationControlMechanism(
                state_features=[mech.input_port],
                objective_mechanism=obj,
                control_signals=pnl.ControlSignal(
                    modulates=('seed', mech),
                    modulation=pnl.OVERRIDE,
                    allocation_samples=pnl.SampleSpec(start=0, stop=num_generators - 1, step=1),
                    # FIX: 11/3/21 DELETE: [NOT NEEDED ANYMORE]
                    cost_options=pnl.CostFunctions.NONE
                )
            )
        )

        comp.run(inputs={mech: [1]},
                 num_trials=2,
                 report_output=pnl.ReportOutput.FULL,
                 report_params=pnl.ReportParams.MONITORED,
                 execution_mode=comp_mode)

        # Construct expected results.
        # First all generators rest their sequence.
        # In the second trial, the "winning" seed from the previous one continues its
        # random sequence
        all_generators = [np.random.RandomState([seed]) for seed in range(num_generators)]
        first_generator_samples = [g.uniform(0, 1) for g in all_generators]
        best_first = max(first_generator_samples)
        index_best = first_generator_samples.index(best_first)
        second_generator_samples = [g.uniform(0, 1) for g in all_generators]
        second_considerations = first_generator_samples[:index_best] + \
                                second_generator_samples[index_best:index_best + 1] + \
                                first_generator_samples[index_best + 1:]
        best_second = max(second_considerations)
        # Check that we select the maximum of generated values
        assert np.allclose(best_first, comp.results[0])
        assert np.allclose(best_second, comp.results[1])


@pytest.mark.control
class TestModelBasedOptimizationControlMechanisms_Execution:
    def test_ocm_default_function(self):
        a = pnl.ProcessingMechanism()
        comp = pnl.Composition(
            controller_mode=pnl.BEFORE,
            nodes=[a],
            controller=pnl.OptimizationControlMechanism(
                control=pnl.ControlSignal(
                    modulates=(pnl.SLOPE, a),
                    intensity_cost_function=lambda x: 0,
                    adjustment_cost_function=lambda x: 0,
                    allocation_samples=[1, 10]
                ),
                state_features=[a.input_port],
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=[a.output_port]
                ),
            )
        )
        assert type(comp.controller.function) == pnl.GridSearch
        assert comp.run([1]) == [10]

    @pytest.mark.parametrize("nested", [True, False])
    @pytest.mark.parametrize("format", ["list", "tuple", "SampleIterator", "SampleIteratorArray", "SampleSpec", "ndArray"])
    @pytest.mark.parametrize("mode", pytest.helpers.get_comp_execution_modes() +
                                     [pytest.helpers.cuda_param('Python-PTX'),
                                      pytest.param('Python-LLVM', marks=pytest.mark.llvm)])
    def test_ocm_searchspace_format_equivalence(self, format, nested, mode):
        if str(mode).startswith('Python-'):
            ocm_mode = mode.split('-')[1]
            mode = pnl.ExecutionMode.Python
        else:
            # OCM default mode is Python
            ocm_mode = 'Python'

        if format == "list":
            search_space = [1, 10]
        elif format == "tuple":
            search_space = (1, 10)
        elif format == "SampleIterator":
            search_space = SampleIterator((1, 10))
        elif format == "SampleIteratorArray":
            search_space = SampleIterator([1, 10])
        elif format == "SampleSpec":
            search_space = SampleSpec(1, 10, 9)
        elif format == "ndArray":
            search_space = np.array((1, 10))

        if nested:
            search_space = [search_space]

        a = pnl.ProcessingMechanism()
        comp = pnl.Composition(
            controller_mode=pnl.BEFORE,
            nodes=[a],
            controller=pnl.OptimizationControlMechanism(
                control=pnl.ControlSignal(
                    modulates=(pnl.SLOPE, a),
                    cost_options=None
                ),
                state_features=[a.input_port],
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=[a.output_port]
                ),
                search_space=search_space
            )
        )
        comp.controller.comp_execution_mode = ocm_mode

        assert type(comp.controller.function) == pnl.GridSearch
        assert comp.run([1], execution_mode=mode) == [[10]]

    def test_evc(self):
        # Mechanisms
        Input = pnl.TransferMechanism(name='Input')
        reward = pnl.TransferMechanism(output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE],
                                       name='reward')
        Decision = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0,
                                                                             pnl.ControlProjection(function=pnl.Linear,
                                                                                                   control_signal_params={pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})),
                                                                 threshold=(1.0,
                                                                            pnl.ControlProjection(function=pnl.Linear,
                                                                                                  control_signal_params={pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})),
                                                                 noise=0.5,
                                                                 starting_value=0,
                                                                 non_decision_time=0.45),
                           output_ports=[pnl.DECISION_VARIABLE,
                                        pnl.RESPONSE_TIME,
                                        pnl.PROBABILITY_UPPER_THRESHOLD],
                           name='Decision')

        comp = pnl.Composition(name="evc", retain_old_simulation_data=True)
        comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
        comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
        task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
        comp.add_linear_processing_pathway(task_execution_pathway)

        comp.add_controller(controller=pnl.OptimizationControlMechanism(
                                                agent_rep=comp,
                                                state_features=[reward.input_port, Input.input_port],
                                                state_feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                                                objective_mechanism=pnl.ObjectiveMechanism(
                                                        function=pnl.LinearCombination(operation=pnl.PRODUCT),
                                                        monitor=[reward,
                                                                 Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                                                                 (Decision.output_ports[pnl.RESPONSE_TIME], -1, 1)]),
                                                function=pnl.GridSearch(),
                                                control_signals=[{CONTROL: ("drift_rate", Decision),
                                                                  ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                                                                 {CONTROL: ("threshold", Decision),
                                                                  ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}])
                                       )

        comp.enable_controller = True

        comp._analyze_graph()

        stim_list_dict = {
            Input: [0.5, 0.123],
            reward: [20, 20]
        }

        comp.run(inputs=stim_list_dict)

        # Note: Removed decision variable OutputPort from simulation results because sign is chosen randomly
        expected_sim_results_array = [
            [[10.], [10.0], [0.0], [0.48999867], [0.50499983]],
            [[10.], [10.0], [0.0], [1.08965888], [0.51998934]],
            [[10.], [10.0], [0.0], [2.40680493], [0.53494295]],
            [[10.], [10.0], [0.0], [4.43671978], [0.549834]],
            [[10.], [10.0], [0.0], [0.48997868], [0.51998934]],
            [[10.], [10.0], [0.0], [1.08459402], [0.57932425]],
            [[10.], [10.0], [0.0], [2.36033556], [0.63645254]],
            [[10.], [10.0], [0.0], [4.24948962], [0.68997448]],
            [[10.], [10.0], [0.0], [0.48993479], [0.53494295]],
            [[10.], [10.0], [0.0], [1.07378304], [0.63645254]],
            [[10.], [10.0], [0.0], [2.26686573], [0.72710822]],
            [[10.], [10.0], [0.0], [3.90353015], [0.80218389]],
            [[10.], [10.0], [0.0], [0.4898672], [0.549834]],
            [[10.], [10.0], [0.0], [1.05791834], [0.68997448]],
            [[10.], [10.0], [0.0], [2.14222978], [0.80218389]],
            [[10.], [10.0], [0.0], [3.49637662], [0.88079708]],
            [[15.], [15.0], [0.0], [0.48999926], [0.50372993]],
            [[15.], [15.0], [0.0], [1.08981011], [0.51491557]],
            [[15.], [15.0], [0.0], [2.40822035], [0.52608629]],
            [[15.], [15.0], [0.0], [4.44259627], [0.53723096]],
            [[15.], [15.0], [0.0], [0.48998813], [0.51491557]],
            [[15.], [15.0], [0.0], [1.0869779], [0.55939819]],
            [[15.], [15.0], [0.0], [2.38198336], [0.60294711]],
            [[15.], [15.0], [0.0], [4.33535807], [0.64492386]],
            [[15.], [15.0], [0.0], [0.48996368], [0.52608629]],
            [[15.], [15.0], [0.0], [1.08085171], [0.60294711]],
            [[15.], [15.0], [0.0], [2.32712843], [0.67504223]],
            [[15.], [15.0], [0.0], [4.1221271], [0.7396981]],
            [[15.], [15.0], [0.0], [0.48992596], [0.53723096]],
            [[15.], [15.0], [0.0], [1.07165729], [0.64492386]],
            [[15.], [15.0], [0.0], [2.24934228], [0.7396981]],
            [[15.], [15.0], [0.0], [3.84279648], [0.81637827]]
        ]

        for simulation in range(len(expected_sim_results_array)):
            assert np.allclose(expected_sim_results_array[simulation],
                               # Note: Skip decision variable OutputPort
                               comp.simulation_results[simulation][0:3] + comp.simulation_results[simulation][4:6])

        expected_results_array = [
            [[20.0], [20.0], [0.0], [1.0], [2.378055160151634], [0.9820137900379085]],
            [[20.0], [20.0], [0.0], [-0.1], [0.48999967725112503], [0.5024599801509442]]
        ]

        for trial in range(len(expected_results_array)):
            np.testing.assert_allclose(comp.results[trial], expected_results_array[trial], atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(trial))

    def test_evc_gratton(self):
        # Stimulus Mechanisms
        target_stim = pnl.TransferMechanism(name='Target Stimulus',
                                            function=pnl.Linear(slope=0.3324))
        flanker_stim = pnl.TransferMechanism(name='Flanker Stimulus',
                                             function=pnl.Linear(slope=0.3545221843))

        # Processing Mechanisms (Control)
        Target_Rep = pnl.TransferMechanism(name='Target Representation')
        Flanker_Rep = pnl.TransferMechanism(name='Flanker Representation')

        # Processing Mechanism (Automatic)
        Automatic_Component = pnl.TransferMechanism(name='Automatic Component')

        # Decision Mechanism
        Decision = pnl.DDM(name='Decision',
                           function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0),
                                                                 threshold=(0.2645),
                                                                 noise=(0.5),
                                                                 starting_value=(0),
                                                                 non_decision_time=0.15),
                           output_ports=[pnl.DECISION_VARIABLE,
                                          pnl.RESPONSE_TIME,
                                          pnl.PROBABILITY_UPPER_THRESHOLD]
                           )

        # Outcome Mechanism
        reward = pnl.TransferMechanism(name='reward')

        # Pathways
        target_control_pathway = [target_stim, Target_Rep, Decision]
        flanker_control_pathway = [flanker_stim, Flanker_Rep, Decision]
        target_automatic_pathway = [target_stim, Automatic_Component, Decision]
        flanker_automatic_pathway = [flanker_stim, Automatic_Component, Decision]
        pathways = [target_control_pathway, flanker_control_pathway, target_automatic_pathway,
                    flanker_automatic_pathway]

        # Composition
        evc_gratton = pnl.Composition(name="EVCGratton")
        evc_gratton.add_node(Decision, required_roles=pnl.NodeRole.OUTPUT)
        for path in pathways:
            evc_gratton.add_linear_processing_pathway(path)
        evc_gratton.add_node(reward, required_roles=pnl.NodeRole.OUTPUT)

        # Control Signals
        signalSearchRange = pnl.SampleSpec(start=1.0, stop=1.8, step=0.2)

        target_rep_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, Target_Rep)],
                                                      variable=1.0,
                                                      intensity_cost_function=pnl.Exponential(rate=0.8046),
                                                      allocation_samples=signalSearchRange)

        flanker_rep_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, Flanker_Rep)],
                                                       variable=1.0,
                                                       intensity_cost_function=pnl.Exponential(rate=0.8046),
                                                       allocation_samples=signalSearchRange)

        objective_mech = pnl.ObjectiveMechanism(function=pnl.LinearCombination(operation=pnl.PRODUCT),
                                                monitor=[reward,
                                                                         (Decision.output_ports[
                                                                              pnl.PROBABILITY_UPPER_THRESHOLD], 1, -1)])
        # Model Based OCM (formerly controller)
        evc_gratton.add_controller(controller=pnl.OptimizationControlMechanism(agent_rep=evc_gratton,
                                                                               state_features=[target_stim.input_port,
                                                                                               flanker_stim.input_port,
                                                                                               reward.input_port],
                                                                               state_feature_function=pnl.AdaptiveIntegrator(
                                                                                             rate=1.0),
                                                                               objective_mechanism=objective_mech,
                                                                               function=pnl.GridSearch(),
                                                                               control_signals=[
                                                                                             target_rep_control_signal,
                                                                                             flanker_rep_control_signal]))
        evc_gratton.enable_controller = True

        targetFeatures = [1, 1, 1]
        flankerFeatures = [1, -1, 1]
        rewardValues = [100, 100, 100]

        stim_list_dict = {target_stim: targetFeatures,
                          flanker_stim: flankerFeatures,
                          reward: rewardValues}

        evc_gratton.run(inputs=stim_list_dict)

        expected_results_array = [[[0.32257752863413636], [0.9481940753514433], [100.]],
                                  [[0.42963678062444666], [0.47661180945923376], [100.]],
                                  [[0.300291026852769], [0.97089165101931], [100.]]]

        expected_sim_results_array = [
            [[0.32257753], [0.94819408], [100.]],
            [[0.31663196], [0.95508757], [100.]],
            [[0.31093566], [0.96110142], [100.]],
            [[0.30548947], [0.96633839], [100.]],
            [[0.30029103], [0.97089165], [100.]],
            [[0.3169957], [0.95468427], [100.]],
            [[0.31128378], [0.9607499], [100.]],
            [[0.30582202], [0.96603252], [100.]],
            [[0.30060824], [0.9706259], [100.]],
            [[0.29563774], [0.97461444], [100.]],
            [[0.31163288], [0.96039533], [100.]],
            [[0.30615555], [0.96572397], [100.]],
            [[0.30092641], [0.97035779], [100.]],
            [[0.2959409], [0.97438178], [100.]],
            [[0.29119255], [0.97787196], [100.]],
            [[0.30649004], [0.96541272], [100.]],
            [[0.30124552], [0.97008732], [100.]],
            [[0.29624499], [0.97414704], [100.]],
            [[0.29148205], [0.97766847], [100.]],
            [[0.28694892], [0.98071974], [100.]],
            [[0.30156558], [0.96981445], [100.]],
            [[0.29654999], [0.97391021], [100.]],
            [[0.29177245], [0.97746315], [100.]],
            [[0.28722523], [0.98054192], [100.]],
            [[0.28289958], [0.98320731], [100.]],
            [[0.42963678], [0.47661181], [100.]],
            [[0.42846471], [0.43938586], [100.]],
            [[0.42628176], [0.40282965], [100.]],
            [[0.42314468], [0.36732207], [100.]],
            [[0.41913221], [0.333198], [100.]],
            [[0.42978939], [0.51176048], [100.]],
            [[0.42959394], [0.47427693], [100.]],
            [[0.4283576], [0.43708106], [100.]],
            [[0.4261132], [0.40057958], [100.]],
            [[0.422919], [0.36514906], [100.]],
            [[0.42902209], [0.54679323], [100.]],
            [[0.42980788], [0.50942101], [100.]],
            [[0.42954704], [0.47194318], [100.]],
            [[0.42824656], [0.43477897], [100.]],
            [[0.42594094], [0.3983337], [100.]],
            [[0.42735293], [0.58136855], [100.]],
            [[0.42910149], [0.54447221], [100.]],
            [[0.42982229], [0.50708112], [100.]],
            [[0.42949608], [0.46961065], [100.]],
            [[0.42813159], [0.43247968], [100.]],
            [[0.42482049], [0.61516258], [100.]],
            [[0.42749136], [0.57908829], [100.]],
            [[0.42917687], [0.54214925], [100.]],
            [[0.42983261], [0.50474093], [100.]],
            [[0.42944107], [0.46727945], [100.]],
            [[0.32257753], [0.94819408], [100.]],
            [[0.31663196], [0.95508757], [100.]],
            [[0.31093566], [0.96110142], [100.]],
            [[0.30548947], [0.96633839], [100.]],
            [[0.30029103], [0.97089165], [100.]],
            [[0.3169957], [0.95468427], [100.]],
            [[0.31128378], [0.9607499], [100.]],
            [[0.30582202], [0.96603252], [100.]],
            [[0.30060824], [0.9706259], [100.]],
            [[0.29563774], [0.97461444], [100.]],
            [[0.31163288], [0.96039533], [100.]],
            [[0.30615555], [0.96572397], [100.]],
            [[0.30092641], [0.97035779], [100.]],
            [[0.2959409], [0.97438178], [100.]],
            [[0.29119255], [0.97787196], [100.]],
            [[0.30649004], [0.96541272], [100.]],
            [[0.30124552], [0.97008732], [100.]],
            [[0.29624499], [0.97414704], [100.]],
            [[0.29148205], [0.97766847], [100.]],
            [[0.28694892], [0.98071974], [100.]],
            [[0.30156558], [0.96981445], [100.]],
            [[0.29654999], [0.97391021], [100.]],
            [[0.29177245], [0.97746315], [100.]],
            [[0.28722523], [0.98054192], [100.]],
            [[0.28289958], [0.98320731], [100.]],
        ]

        for trial in range(len(evc_gratton.results)):
            assert np.allclose(expected_results_array[trial],
                               # Note: Skip decision variable OutputPort
                               evc_gratton.results[trial][1:])
        for simulation in range(len(evc_gratton.simulation_results)):
            assert np.allclose(expected_sim_results_array[simulation],
                               # Note: Skip decision variable OutputPort
                               evc_gratton.simulation_results[simulation][1:])

    @pytest.mark.control
    @pytest.mark.composition
    def test_laming_validation_specify_control_signals(self):
        # Mechanisms
        Input = pnl.TransferMechanism(name='Input')
        reward = pnl.TransferMechanism(
            output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE],
            name='reward'
        )
        Decision = pnl.DDM(
            function=pnl.DriftDiffusionAnalytical(
                drift_rate=1.0,
                threshold=1.0,
                noise=0.5,
                starting_value=0,
                non_decision_time=0.45
            ),
            output_ports=[
                pnl.DECISION_VARIABLE,
                pnl.RESPONSE_TIME,
                pnl.PROBABILITY_UPPER_THRESHOLD
            ],
            name='Decision'
        )

        comp = pnl.Composition(name="evc", retain_old_simulation_data=True)
        comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
        comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
        task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
        comp.add_linear_processing_pathway(task_execution_pathway)

        comp.add_controller(
            controller=pnl.OptimizationControlMechanism(
                agent_rep=comp,
                state_features=[reward.input_port, Input.input_port],
                state_feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                objective_mechanism=pnl.ObjectiveMechanism(
                    function=pnl.LinearCombination(operation=pnl.PRODUCT),
                    monitor=[
                        reward,
                        Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                        (Decision.output_ports[pnl.RESPONSE_TIME], -1, 1)
                    ]
                ),
                function=pnl.GridSearch(),
                control_signals=[
                    {
                        CONTROL: (pnl.DRIFT_RATE, Decision),
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                    {
                        CONTROL: (pnl.THRESHOLD, Decision),
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    }
                ],
            )
        )

        stim_list_dict = {
            Input: [0.5, 0.123],
            reward: [20, 20]
        }

        comp.run(inputs=stim_list_dict)

        # Note: Removed decision variable OutputPort from simulation results
        # because sign is chosen randomly
        expected_sim_results_array = [
            [[10.], [10.0], [0.0], [0.48999867], [0.50499983]],
            [[10.], [10.0], [0.0], [1.08965888], [0.51998934]],
            [[10.], [10.0], [0.0], [2.40680493], [0.53494295]],
            [[10.], [10.0], [0.0], [4.43671978], [0.549834]],
            [[10.], [10.0], [0.0], [0.48997868], [0.51998934]],
            [[10.], [10.0], [0.0], [1.08459402], [0.57932425]],
            [[10.], [10.0], [0.0], [2.36033556], [0.63645254]],
            [[10.], [10.0], [0.0], [4.24948962], [0.68997448]],
            [[10.], [10.0], [0.0], [0.48993479], [0.53494295]],
            [[10.], [10.0], [0.0], [1.07378304], [0.63645254]],
            [[10.], [10.0], [0.0], [2.26686573], [0.72710822]],
            [[10.], [10.0], [0.0], [3.90353015], [0.80218389]],
            [[10.], [10.0], [0.0], [0.4898672], [0.549834]],
            [[10.], [10.0], [0.0], [1.05791834], [0.68997448]],
            [[10.], [10.0], [0.0], [2.14222978], [0.80218389]],
            [[10.], [10.0], [0.0], [3.49637662], [0.88079708]],
            [[15.], [15.0], [0.0], [0.48999926], [0.50372993]],
            [[15.], [15.0], [0.0], [1.08981011], [0.51491557]],
            [[15.], [15.0], [0.0], [2.40822035], [0.52608629]],
            [[15.], [15.0], [0.0], [4.44259627], [0.53723096]],
            [[15.], [15.0], [0.0], [0.48998813], [0.51491557]],
            [[15.], [15.0], [0.0], [1.0869779], [0.55939819]],
            [[15.], [15.0], [0.0], [2.38198336], [0.60294711]],
            [[15.], [15.0], [0.0], [4.33535807], [0.64492386]],
            [[15.], [15.0], [0.0], [0.48996368], [0.52608629]],
            [[15.], [15.0], [0.0], [1.08085171], [0.60294711]],
            [[15.], [15.0], [0.0], [2.32712843], [0.67504223]],
            [[15.], [15.0], [0.0], [4.1221271], [0.7396981]],
            [[15.], [15.0], [0.0], [0.48992596], [0.53723096]],
            [[15.], [15.0], [0.0], [1.07165729], [0.64492386]],
            [[15.], [15.0], [0.0], [2.24934228], [0.7396981]],
            [[15.], [15.0], [0.0], [3.84279648], [0.81637827]]
        ]

        for simulation in range(len(expected_sim_results_array)):
            assert np.allclose(
                expected_sim_results_array[simulation],
                # Note: Skip decision variable OutputPort
                comp.simulation_results[simulation][0:3] + comp.simulation_results[simulation][4:6]
            )

        expected_results_array = [
            [[20.0], [20.0], [0.0], [1.0], [2.378055160151634], [0.9820137900379085]],
            [[20.0], [20.0], [0.0], [-0.1], [0.48999967725112503], [0.5024599801509442]]
        ]

        for trial in range(len(expected_results_array)):
            np.testing.assert_allclose(
                comp.results[trial],
                expected_results_array[trial],
                atol=1e-08,
                err_msg='Failed on expected_output[{0}]'.format(trial)
            )

    @pytest.mark.control
    @pytest.mark.composition
    def test_stateful_mechanism_in_simulation(self):
        # Mechanisms
        Input = pnl.TransferMechanism(name='Input', integrator_mode=True)
        reward = pnl.TransferMechanism(
            output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE],
            name='reward'
        )
        Decision = pnl.DDM(
            function=pnl.DriftDiffusionAnalytical(
                drift_rate=(
                    1.0,
                    pnl.ControlProjection(
                        function=pnl.Linear,
                        control_signal_params={
                            ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                        },
                    ),
                ),
                threshold=(
                    1.0,
                    pnl.ControlProjection(
                        function=pnl.Linear,
                        control_signal_params={
                            ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                        },
                    ),
                ),
                noise=(0.5),
                starting_value=(0),
                non_decision_time=0.45
            ),
            output_ports=[
                pnl.DECISION_VARIABLE,
                pnl.RESPONSE_TIME,
                pnl.PROBABILITY_UPPER_THRESHOLD
            ],
            name='Decision',
        )

        comp = pnl.Composition(name="evc", retain_old_simulation_data=True)
        comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
        comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
        task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
        comp.add_linear_processing_pathway(task_execution_pathway)

        comp.add_controller(
            controller=pnl.OptimizationControlMechanism(
                agent_rep=comp,
                state_features=[reward.input_port, Input.input_port],
                state_feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                objective_mechanism=pnl.ObjectiveMechanism(
                    function=pnl.LinearCombination(operation=pnl.PRODUCT),
                    monitor=[
                        reward,
                        Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                        (Decision.output_ports[pnl.RESPONSE_TIME], -1, 1)
                    ]
                ),
                function=pnl.GridSearch(),
                control_signals=[
                    {
                        CONTROL: (pnl.DRIFT_RATE, Decision),
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                    {
                        CONTROL: (pnl.THRESHOLD, Decision),
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    }
                ],
            )
        )

        stim_list_dict = {
            Input: [0.5, 0.123],
            reward: [20, 20]
        }
        Input.reset_stateful_function_when = pnl.Never()

        comp.run(inputs=stim_list_dict)

        # Note: Removed decision variable OutputPort from simulation results
        # because sign is chosen randomly
        expected_sim_results_array = [
            [[10.], [10.0], [0.0], [0.48999867], [0.50499983]],
            [[10.], [10.0], [0.0], [1.08965888], [0.51998934]],
            [[10.], [10.0], [0.0], [2.40680493], [0.53494295]],
            [[10.], [10.0], [0.0], [4.43671978], [0.549834]],
            [[10.], [10.0], [0.0], [0.48997868], [0.51998934]],
            [[10.], [10.0], [0.0], [1.08459402], [0.57932425]],
            [[10.], [10.0], [0.0], [2.36033556], [0.63645254]],
            [[10.], [10.0], [0.0], [4.24948962], [0.68997448]],
            [[10.], [10.0], [0.0], [0.48993479], [0.53494295]],
            [[10.], [10.0], [0.0], [1.07378304], [0.63645254]],
            [[10.], [10.0], [0.0], [2.26686573], [0.72710822]],
            [[10.], [10.0], [0.0], [3.90353015], [0.80218389]],
            [[10.], [10.0], [0.0], [0.4898672], [0.549834]],
            [[10.], [10.0], [0.0], [1.05791834], [0.68997448]],
            [[10.], [10.0], [0.0], [2.14222978], [0.80218389]],
            [[10.], [10.0], [0.0], [3.49637662], [0.88079708]],
            [[15.], [15.0], [0.0], [0.48999926], [0.50372993]],
            [[15.], [15.0], [0.0], [1.08981011], [0.51491557]],
            [[15.], [15.0], [0.0], [2.40822035], [0.52608629]],
            [[15.], [15.0], [0.0], [4.44259627], [0.53723096]],
            [[15.], [15.0], [0.0], [0.48998813], [0.51491557]],
            [[15.], [15.0], [0.0], [1.0869779], [0.55939819]],
            [[15.], [15.0], [0.0], [2.38198336], [0.60294711]],
            [[15.], [15.0], [0.0], [4.33535807], [0.64492386]],
            [[15.], [15.0], [0.0], [0.48996368], [0.52608629]],
            [[15.], [15.0], [0.0], [1.08085171], [0.60294711]],
            [[15.], [15.0], [0.0], [2.32712843], [0.67504223]],
            [[15.], [15.0], [0.0], [4.1221271], [0.7396981]],
            [[15.], [15.0], [0.0], [0.48992596], [0.53723096]],
            [[15.], [15.0], [0.0], [1.07165729], [0.64492386]],
            [[15.], [15.0], [0.0], [2.24934228], [0.7396981]],
            [[15.], [15.0], [0.0], [3.84279648], [0.81637827]]
        ]

        for simulation in range(len(expected_sim_results_array)):
            assert np.allclose(
                expected_sim_results_array[simulation],
                # Note: Skip decision variable OutputPort
                comp.simulation_results[simulation][0:3] + comp.simulation_results[simulation][4:6]
            )

        expected_results_array = [
            [[20.0], [20.0], [0.0], [1.0], [3.4963766238230596], [0.8807970779778824]],
            [[20.0], [20.0], [0.0], [-0.1], [0.4899992579951842], [0.503729930808051]]
        ]

        for trial in range(len(expected_results_array)):
            np.testing.assert_allclose(
                comp.results[trial],
                expected_results_array[trial],
                atol=1e-08,
                err_msg='Failed on expected_output[{0}]'.format(trial)
            )

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Model Based OCM")
    @pytest.mark.parametrize("mode", pytest.helpers.get_comp_execution_modes() +
                                     [pytest.helpers.cuda_param('Python-PTX'),
                                      pytest.param('Python-LLVM', marks=pytest.mark.llvm)])
    def test_model_based_ocm_after(self, benchmark, mode):
        if str(mode).startswith('Python-'):
            ocm_mode = mode.split('-')[1]
            mode = pnl.ExecutionMode.Python
        else:
            # OCM default mode is Python
            ocm_mode = 'Python'

        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')

        comp = pnl.Composition(name='comp',
                               controller_mode=pnl.AFTER)
        comp.add_linear_processing_pathway([A, B])

        search_range = pnl.SampleSpec(start=0.25, stop=0.75, step=0.25)
        control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, A)],
                                           variable=1.0,
                                           allocation_samples=search_range,
                                           cost_options=pnl.CostFunctions.INTENSITY,
                                           intensity_cost_function=pnl.Linear(slope=0.))

        objective_mech = pnl.ObjectiveMechanism(monitor=[B])
        ocm = pnl.OptimizationControlMechanism(agent_rep=comp,
                                               state_features=[A.input_port],
                                               objective_mechanism=objective_mech,
                                               function=pnl.GridSearch(save_values=True),
                                               control_signals=[control_signal],
                                               comp_execution_mode=ocm_mode)
        # objective_mech.log.set_log_conditions(pnl.OUTCOME)

        comp.add_controller(ocm)

        inputs = {A: [[[1.0]], [[2.0]], [[3.0]]]}

        comp.run(inputs=inputs, execution_mode=mode)

        # objective_mech.log.print_entries(pnl.OUTCOME)
        assert np.allclose(comp.results, [[np.array([1.])], [np.array([1.5])], [np.array([2.25])]])
        if mode == pnl.ExecutionMode.Python:
            assert np.allclose(np.asfarray(ocm.function.saved_values).flatten(), [0.75, 1.5, 2.25])

        if benchmark.enabled:
            benchmark(comp.run, inputs, execution_mode=mode)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Model Based OCM")
    @pytest.mark.parametrize("mode", pytest.helpers.get_comp_execution_modes() +
                                     [pytest.helpers.cuda_param('Python-PTX'),
                                      pytest.param('Python-LLVM', marks=pytest.mark.llvm)])
    def test_model_based_ocm_before(self, benchmark, mode):
        if str(mode).startswith('Python-'):
            ocm_mode = mode.split('-')[1]
            mode = pnl.ExecutionMode.Python
        else:
            # OCM default mode is Python
            ocm_mode = 'Python'

        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')

        comp = pnl.Composition(name='comp',
                               controller_mode=pnl.BEFORE)
        comp.add_linear_processing_pathway([A, B])

        search_range = pnl.SampleSpec(start=0.25, stop=0.75, step=0.25)
        control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, A)],
                                           variable=1.0,
                                           allocation_samples=search_range,
                                           cost_options=pnl.CostFunctions.INTENSITY,
                                           intensity_cost_function=pnl.Linear(slope=0.))

        objective_mech = pnl.ObjectiveMechanism(monitor=[B])
        ocm = pnl.OptimizationControlMechanism(agent_rep=comp,
                                               state_features=[A.input_port],
                                               objective_mechanism=objective_mech,
                                               function=pnl.GridSearch(save_values=True),
                                               control_signals=[control_signal],
                                               comp_execution_mode=ocm_mode)
        # objective_mech.log.set_log_conditions(pnl.OUTCOME)

        comp.add_controller(ocm)

        inputs = {A: [[[1.0]], [[2.0]], [[3.0]]]}

        comp.run(inputs=inputs, execution_mode=mode)

        # objective_mech.log.print_entries(pnl.OUTCOME)
        assert np.allclose(comp.results, [[np.array([0.75])], [np.array([1.5])], [np.array([2.25])]])
        if mode == pnl.ExecutionMode.Python:
            assert np.allclose(np.asfarray(ocm.function.saved_values).flatten(), [0.75, 1.5, 2.25])

        if benchmark.enabled:
            benchmark(comp.run, inputs, execution_mode=mode)

    def test_model_based_ocm_with_buffer(self):

        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')

        comp = pnl.Composition(name='comp',
                               controller_mode=pnl.BEFORE,
                               retain_old_simulation_data=True,
                               )
        comp.add_linear_processing_pathway([A, B])

        search_range = pnl.SampleSpec(start=0.25, stop=0.75, step=0.25)
        control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, A)],
                                           variable=1.0,
                                           allocation_samples=search_range,
                                           intensity_cost_function=pnl.Linear(slope=0.))

        objective_mech = pnl.ObjectiveMechanism(monitor=[B])
        ocm = pnl.OptimizationControlMechanism(agent_rep=comp,
                                               state_features=[A.input_port],
                                               state_feature_function=pnl.Buffer(history=2),
                                               objective_mechanism=objective_mech,
                                               function=pnl.GridSearch(),
                                               control_signals=[control_signal])
        objective_mech.log.set_log_conditions(pnl.OUTCOME)

        comp.add_controller(ocm)

        inputs = {A: [[[1.0]], [[2.0]], [[3.0]]]}

        for i in range(1, len(ocm.input_ports)):
            ocm.input_ports[i].function.reset()
        comp.run(inputs=inputs)

        log = objective_mech.log.nparray_dictionary()

        # "outer" composition
        assert np.allclose(log["comp"][pnl.OUTCOME], [[0.75], [1.5], [2.25]])

        # preprocess to ignore control allocations
        log_parsed = {}
        for key, value in log.items():
            cleaned_key = re.sub(r'comp-sim.*num: (\d).*', r'\1', key)
            log_parsed[cleaned_key] = value

        # First round of simulations is only one trial.
        # (Even though the feature fn is a Buffer, there is no history yet)
        for i in range(0, 3):
            assert len(log_parsed[str(i)]["Trial"]) == 1

        # Second and third rounds of simulations are two trials.
        # (The buffer has history = 2)
        for i in range(3, 9):
            assert len(log_parsed[str(i)]["Trial"]) == 2

    def test_stability_flexibility_susan_and_sebastian(self):

        # computeAccuracy(trialInformation)
        # Inputs: trialInformation[0, 1, 2, 3]
        # trialInformation[0] - Task Dimension : [0, 1] or [1, 0]
        # trialInformation[1] - Stimulus Dimension: Congruent {[1, 1] or [-1, -1]} // Incongruent {[-1, 1] or [1, -1]}
        # trialInformation[2] - Upper Threshold: Probability of DDM choosing upper bound
        # trialInformation[3] - Lower Threshold: Probability of DDM choosing lower bound

        def computeAccuracy(trialInformation):

            # Unload contents of trialInformation
            # Origin Node Inputs
            taskInputs = trialInformation[0]
            stimulusInputs = trialInformation[1]

            # DDM Outputs
            upperThreshold = trialInformation[2]
            lowerThreshold = trialInformation[3]

            # Keep Track of Accuracy
            accuracy = []

            # Beginning of Accuracy Calculation
            colorTrial = (taskInputs[0] == 1)
            motionTrial = (taskInputs[1] == 1)

            # Based on the task dimension information, decide which response is "correct"
            # Obtain accuracy probability from DDM thresholds in "correct" direction
            if colorTrial:
                if stimulusInputs[0] == 1:
                    accuracy.append(upperThreshold)
                elif stimulusInputs[0] == -1:
                    accuracy.append(lowerThreshold)

            if motionTrial:
                if stimulusInputs[1] == 1:
                    accuracy.append(upperThreshold)
                elif stimulusInputs[1] == -1:
                    accuracy.append(lowerThreshold)

            # Accounts for initialization runs that have no variable input
            if len(accuracy) == 0:
                accuracy = [0]

            # print("Accuracy: ", accuracy[0])
            # print()

            return [accuracy]

        # BEGIN: Composition Construction

        # Constants as defined in Musslick et al. 2018
        tau = 0.9  # Time Constant
        DRIFT = 1  # Drift Rate
        STARTING_VALUE = 0.0  # Starting Point
        THRESHOLD = 0.0475  # Threshold
        NOISE = 0.04  # Noise
        T0 = 0.2  # T0

        # Task Layer: [Color, Motion] {0, 1} Mutually Exclusive
        # Origin Node
        taskLayer = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                          size=2,
                                          function=pnl.Linear(slope=1, intercept=0),
                                          output_ports=[pnl.RESULT],
                                          name='Task Input [I1, I2]')

        # Stimulus Layer: [Color Stimulus, Motion Stimulus]
        # Origin Node
        stimulusInfo = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                             size=2,
                                             function=pnl.Linear(slope=1, intercept=0),
                                             output_ports=[pnl.RESULT],
                                             name="Stimulus Input [S1, S2]")

        # Activation Layer: [Color Activation, Motion Activation]
        # Recurrent: Self Excitation, Mutual Inhibition
        # Controlled: Gain Parameter
        activation = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
                                                    function=pnl.Logistic(gain=1.0),
                                                    matrix=[[1.0, -1.0],
                                                            [-1.0, 1.0]],
                                                    integrator_mode=True,
                                                    integrator_function=pnl.AdaptiveIntegrator(rate=(tau)),
                                                    initial_value=np.array([[0.0, 0.0]]),
                                                    output_ports=[pnl.RESULT],
                                                    name='Task Activations [Act 1, Act 2]')

        # Hadamard product of Activation and Stimulus Information
        nonAutomaticComponent = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                                      size=2,
                                                      function=pnl.Linear(slope=1, intercept=0),
                                                      input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                                      output_ports=[pnl.RESULT],
                                                      name='Non-Automatic Component [S1*Activity1, S2*Activity2]')

        # Summation of nonAutomatic and Automatic Components
        ddmCombination = pnl.TransferMechanism(size=1,
                                               function=pnl.Linear(slope=1, intercept=0),
                                               input_ports=pnl.InputPort(combine=pnl.SUM),
                                               output_ports=[pnl.RESULT],
                                               name="Drift = (S1 + S2) + (S1*Activity1 + S2*Activity2)")

        decisionMaker = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=DRIFT,
                                                                      starting_value=STARTING_VALUE,
                                                                      threshold=THRESHOLD,
                                                                      noise=NOISE,
                                                                      non_decision_time=T0),
                                output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME,
                                               pnl.PROBABILITY_UPPER_THRESHOLD,
                                               pnl.PROBABILITY_LOWER_THRESHOLD],
                                name='DDM')

        taskLayer.set_log_conditions([pnl.RESULT])
        stimulusInfo.set_log_conditions([pnl.RESULT])
        activation.set_log_conditions([pnl.RESULT, "mod_gain"])
        nonAutomaticComponent.set_log_conditions([pnl.RESULT])
        ddmCombination.set_log_conditions([pnl.RESULT])
        decisionMaker.set_log_conditions([pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD,
                                          pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])

        # Composition Creation

        stabilityFlexibility = pnl.Composition(controller_mode=pnl.BEFORE)

        # Node Creation
        stabilityFlexibility.add_node(taskLayer)
        stabilityFlexibility.add_node(activation)
        stabilityFlexibility.add_node(nonAutomaticComponent)
        stabilityFlexibility.add_node(stimulusInfo)
        stabilityFlexibility.add_node(ddmCombination)
        stabilityFlexibility.add_node(decisionMaker)

        # Projection Creation
        stabilityFlexibility.add_projection(sender=taskLayer, receiver=activation)
        stabilityFlexibility.add_projection(sender=activation, receiver=nonAutomaticComponent)
        stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=nonAutomaticComponent)
        stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=ddmCombination)
        stabilityFlexibility.add_projection(sender=nonAutomaticComponent, receiver=ddmCombination)
        stabilityFlexibility.add_projection(sender=ddmCombination, receiver=decisionMaker)

        # Beginning of Controller

        # Grid Search Range
        searchRange = pnl.SampleSpec(start=1.0, stop=1.9, num=10)

        # Modulate the GAIN parameter from activation layer
        # Initalize cost function as 0
        signal = pnl.ControlSignal(projections=[(pnl.GAIN, activation)],
                                   variable=1.0,
                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                   allocation_samples=searchRange)

        # Use the computeAccuracy function to obtain selection values
        # Pass in 4 arguments whenever computeRewardRate is called
        objectiveMechanism = pnl.ObjectiveMechanism(monitor=[taskLayer, stimulusInfo,
                                                             (pnl.PROBABILITY_UPPER_THRESHOLD, decisionMaker),
                                                             (pnl.PROBABILITY_LOWER_THRESHOLD, decisionMaker)],
                                                    function=computeAccuracy,
                                                    name="Controller Objective Mechanism")

        #  Sets trial history for simulations over specified signal search parameters
        metaController = pnl.OptimizationControlMechanism(agent_rep=stabilityFlexibility,
                                                          state_features=[taskLayer.input_port, stimulusInfo.input_port],
                                                          state_feature_function=pnl.Buffer(history=10),
                                                          name="Controller",
                                                          objective_mechanism=objectiveMechanism,
                                                          function=pnl.GridSearch(),
                                                          control_signals=[signal])

        stabilityFlexibility.add_controller(metaController)
        stabilityFlexibility.enable_controller = True
        # stabilityFlexibility.model_based_optimizer_mode = pnl.BEFORE

        for i in range(1, len(stabilityFlexibility.controller.input_ports)):
            stabilityFlexibility.controller.input_ports[i].function.reset()
        # Origin Node Inputs
        taskTrain = [[1, 0], [0, 1], [1, 0], [0, 1]]
        stimulusTrain = [[1, -1], [-1, 1], [1, -1], [-1, 1]]

        inputs = {taskLayer: taskTrain, stimulusInfo: stimulusTrain}
        stabilityFlexibility.run(inputs)
        assert np.allclose(stabilityFlexibility.results,
                           [[[0.0475], [0.33695222], [1.], [1.13867062e-09]],
                            [[0.0475], [1.13635091], [0.93038951], [0.06961049]],
                            [[0.0475], [0.35801411], [0.99999998], [1.77215519e-08]],
                            [[0.0475], [0.89706881], [0.97981972], [0.02018028]]])

    @pytest.mark.parametrize('num_estimates',[None, 1, 2] )
    @pytest.mark.parametrize('rand_var',[False, True] )
    def test_model_based_num_estimates(self, num_estimates, rand_var):

        A = pnl.ProcessingMechanism(name='A')
        if rand_var:
            B = pnl.DDM(name='B',
                        function=pnl.DriftDiffusionAnalytical)
        else:
            B = pnl.ProcessingMechanism(name='B',
                                        function=pnl.SimpleIntegrator(rate=1))

        comp = pnl.Composition(name='comp')
        comp.add_linear_processing_pathway([A, B])

        search_range = pnl.SampleSpec(start=0.25, stop=0.75, step=0.25)
        control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, A)],
                                           variable=1.0,
                                           allocation_samples=search_range,
                                           intensity_cost_function=pnl.Linear(slope=0.))

        objective_mech = pnl.ObjectiveMechanism(monitor=[B])
        warning_type = None
        if num_estimates and not rand_var:
            warning_type = UserWarning
        warning_msg = f'"\'OptimizationControlMechanism-0\' has \'num_estimates = {num_estimates}\' specified, ' \
                      f'but its \'agent_rep\' (\'comp\') has no random variables: ' \
                      f'\'RANDOMIZATION_CONTROL_SIGNAL\' will not be created, and num_estimates set to None."'
        with pytest.warns(warning_type) as warning:
            ocm = pnl.OptimizationControlMechanism(agent_rep=comp,
                                                   state_features=[A.input_port],
                                                   objective_mechanism=objective_mech,
                                                   function=pnl.GridSearch(),
                                                   num_estimates=num_estimates,
                                                   control_signals=[control_signal])
            if warning_type:
                assert repr(warning[5].message.args[0]) == warning_msg

        comp.add_controller(ocm)
        inputs = {A: [[[1.0]]]}

        comp.run(inputs=inputs,
                 num_trials=2)

        if not num_estimates or not rand_var:
            assert pnl.RANDOMIZATION_CONTROL_SIGNAL not in comp.controller.control_signals # Confirm no estimates
        elif num_estimates:
            assert len(comp.controller.control_signals[pnl.RANDOMIZATION_CONTROL_SIGNAL].efferents) == 1
            # noise

        if rand_var: # results for DDM (which has random variables)
            assert np.allclose(comp.simulation_results,
                               [[np.array([2.25])], [np.array([3.5])], [np.array([4.75])], [np.array([3.])], [np.array([4.25])], [np.array([5.5])]])
            assert np.allclose(comp.results,
                               [[np.array([1.]), np.array([1.1993293])], [np.array([1.]), np.array([3.24637662])]])
        else:  # results for ProcessingMechanism (which does not have any random variables)
            assert np.allclose(comp.simulation_results,
                               [[np.array([2.25])], [np.array([3.5])], [np.array([4.75])], [np.array([3.])], [np.array([4.25])], [np.array([5.5])]])
            assert np.allclose(comp.results,
                               [[np.array([1.])], [np.array([1.75])]])

    def test_model_based_ocm_no_simulations(self):
        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B', function=pnl.SimpleIntegrator(rate=1))

        comp = pnl.Composition(name='comp')
        comp.add_linear_processing_pathway([A, B])

        control_signal = pnl.ControlSignal(
            projections=[(pnl.SLOPE, A)],
            variable=1.0,
            allocation_samples=[1, 2, 3],
            intensity_cost_function=pnl.Linear(slope=0.)
        )

        objective_mech = pnl.ObjectiveMechanism(monitor=[B])
        ocm = pnl.OptimizationControlMechanism(
            agent_rep=comp,
            state_features=[A.input_port],
            objective_mechanism=objective_mech,
            function=pnl.GridSearch(),
            num_estimates=1,
            control_signals=[control_signal],
            search_statefulness=False,
        )

        comp.add_controller(ocm)

        inputs = {A: [[[1.0]]]}

        comp.run(inputs=inputs, num_trials=1)

        # initial 1 + each allocation sample (1, 2, 3) integrated
        assert B.parameters.value.get(comp) == 7

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Multilevel")
    def test_grid_search_random_selection(self, comp_mode, benchmark):
        A = pnl.ProcessingMechanism(name='A')

        A.log.set_log_conditions(items="mod_slope")
        B = pnl.ProcessingMechanism(name='B',
                                    function=pnl.Logistic())

        comp = pnl.Composition(name='comp')
        comp.add_linear_processing_pathway([A, B])

        search_range = pnl.SampleSpec(start=15., stop=35., step=5)
        control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, A)],
                                           variable=1.0,
                                           allocation_samples=search_range,
                                           cost_options=pnl.CostFunctions.INTENSITY,
                                           intensity_cost_function=pnl.Linear(slope=0.))

        objective_mech = pnl.ObjectiveMechanism(monitor=[B])
        ocm = pnl.OptimizationControlMechanism(agent_rep=comp,
                                               state_features=[A.input_port],
                                               objective_mechanism=objective_mech,
                                               function=pnl.GridSearch(select_randomly_from_optimal_values=True),
                                               control_signals=[control_signal])

        comp.add_controller(ocm)

        inputs = {A: [[[1.0]]]}

        comp.run(inputs=inputs, num_trials=10, context='outer_comp', execution_mode=comp_mode)
        assert np.allclose(comp.results, [[[0.7310585786300049]], [[0.999999694097773]], [[0.999999694097773]], [[0.9999999979388463]], [[0.9999999979388463]], [[0.999999694097773]], [[0.9999999979388463]], [[0.999999999986112]], [[0.999999694097773]], [[0.9999999999999993]]])

        # control signal value (mod slope) is chosen randomly from all of the control signal values
        # that correspond to a net outcome of 1
        if comp_mode is pnl.ExecutionMode.Python:
            log_arr = A.log.nparray_dictionary()
            assert np.allclose([[1.], [15.], [15.], [20.], [20.], [15.], [20.], [25.], [15.], [35.]],
                               log_arr['outer_comp']['mod_slope'])

        if benchmark.enabled:
            # Disable logging for the benchmark run
            A.log.set_log_conditions(items="mod_slope", log_condition=LogCondition.OFF)
            A.log.clear_entries()
            benchmark(comp.run, inputs=inputs, num_trials=10, context='bench_outer_comp', execution_mode=comp_mode)
            assert len(A.log.get_logged_entries()) == 0

    @pytest.mark.control
    @pytest.mark.composition
    def test_input_CIM_assignment(self, comp_mode):
        input_a = pnl.ProcessingMechanism(name='oa', function=pnl.Linear(slope=1))
        input_b = pnl.ProcessingMechanism(name='ob', function=pnl.Linear(slope=1))
        output = pnl.ProcessingMechanism(name='oc')
        comp = pnl.Composition(name='ocomp',
                               controller_mode=pnl.BEFORE,
                               retain_old_simulation_data=True)
        comp.add_linear_processing_pathway([input_a, output])
        comp.add_linear_processing_pathway([input_b, output])
        comp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=comp,
                state_features=[input_a.input_port, input_b.input_port],
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                    monitor=output.output_port,
                    function=pnl.SimpleIntegrator,
                    name="Output Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[
                    pnl.ControlSignal(modulates=[(pnl.SLOPE, input_a)],
                                      intensity_cost_function=pnl.Linear(slope=1),
                                      cost_options=pnl.CostFunctions.INTENSITY,
                                      allocation_samples=[-1, 1]),
                    pnl.ControlSignal(modulates=[(pnl.SLOPE, input_b)],
                                      intensity_cost_function=pnl.Linear(slope=0),
                                      cost_options=pnl.CostFunctions.INTENSITY,
                                      allocation_samples=[-1, 1])
                ]))
        results = comp.run(inputs={input_a: [[5]], input_b: [[-2]]},
                           execution_mode=comp_mode)

        # The controller of this model uses two control signals: one that modulates the slope of input_a and one that modulates
        # the slope of input_b. Both control signals have two possible values: -1 or 1.
        #
        # In the correct case, input_a receives a control signal with value 1 and input_b receives a control signal with value
        # -1 to maximize the output of the model given their respective input values of 5 and -2.
        #
        # In the errant case, the control signals are flipped so that input_b receives a control signal with value -1 and
        # input_a receives a control signal with value 1.
        #
        # Thus, in the correct case, the output of the model is 7 ((5*1)+(-2*-1)) and in the errant case the output of the model is
        # -7 ((5*-1)+(-2*1))
        assert np.allclose(results, [[7]])


@pytest.mark.control
class TestSampleIterator:

    def test_int_step(self):
        spec = SampleSpec(step=2,
                          start=0,
                          stop=10)
        sample_iterator = SampleIterator(specification=spec)

        expected = [0, 2, 4, 6, 8, 10]

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_int_num(self):
        spec = SampleSpec(num=6,
                          start=0,
                          stop=10)
        sample_iterator = SampleIterator(specification=spec)

        expected = [0, 2, 4, 6, 8, 10]

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_neither_num_nor_step(self):
        with pytest.raises(SampleIteratorError) as error_text:
            SampleSpec(start=0,
                       stop=10)
        assert "Must specify one of 'step', 'num' or 'function'" in str(error_text.value)

    def test_float_step(self):
        # Need to decide whether stop should be exclusive
        spec = SampleSpec(step=2.79,
                          start=0.65,
                          stop=10.25)
        sample_iterator = SampleIterator(specification=spec)

        expected = [0.65, 3.44, 6.23, 9.02]

        for i in range(4):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(4):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_function(self):
        fun = pnl.NormalDist(mean=5.0)
        spec = SampleSpec(function=fun)
        sample_iterator = SampleIterator(specification=spec)

        expected = [3.4359565850611617, 4.4847029144020505, 2.4464727305984764, 5.302845918582278, 4.306822898004032]

        for i in range(5):
            assert np.allclose(next(sample_iterator), expected[i])

    def test_function_with_num(self):
        fun = pnl.NormalDist(mean=5.0)
        spec = SampleSpec(function=fun,
                          num=4)
        sample_iterator = SampleIterator(specification=spec)

        expected = [3.4359565850611617, 4.4847029144020505, 2.4464727305984764, 5.302845918582278]

        for i in range(4):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_list(self):
        sample_list = [1, 2.0, 3.456, 7.8]
        sample_iterator = SampleIterator(specification=sample_list)

        for i in range(len(sample_list)):
            assert np.allclose(next(sample_iterator), sample_list[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(len(sample_list)):
            assert np.allclose(next(sample_iterator), sample_list[i])

        assert next(sample_iterator, None) is None

        assert sample_iterator.start == 1
        assert sample_iterator.stop is None
        assert sample_iterator.num == len(sample_list)


@pytest.mark.control
class TestControlTimeScales:

    def test_time_step_before(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.BEFORE,
            controller_time_scale=pnl.TimeScale.TIME_STEP
        )
        comp.run([1], num_trials=2)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #
        #  Trial 1:
        #    (C-1) a (C-2) b
        #  Trial 2:
        #    (C-3) a (C-4) b
        #
        assert c.value == [4]
        assert c.execution_count == 4
        assert comp.results == [[2], [4]]

    def test_time_step_after(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.AFTER,
            controller_time_scale=pnl.TimeScale.TIME_STEP
        )
        comp.run([1], num_trials=2)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #
        #  Trial 1:
        #    a (C-1) b (C-2)
        #  Trial 2:
        #    a (C-3) b (C-4)
        #
        assert c.value == [4]
        assert c.execution_count == 4
        assert comp.results == [[1], [3]]

    def test_pass_before(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.BEFORE,
            controller_time_scale=pnl.TimeScale.PASS,
        )
        comp.scheduler.add_condition(
            b, pnl.AfterPass(1)
        )
        comp.run([1], num_trials=2,)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #
        #  Trial 1:
        #    (C-1) Pass 1:
        #       a
        #    (C-2) Pass 2:
        #       a
        #    (C-3) Pass 2:
        #       a   b
        #  Trial 2:
        #    (C-4) Pass 1:
        #       a
        #    (C-5) Pass 2:
        #       a
        #    (C-6) Pass 3:
        #       a   b
        assert c.value == [6]
        assert c.execution_count == 6
        assert comp.results == [[3], [6]]

    def test_pass_after(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.AFTER,
            controller_time_scale=pnl.TimeScale.PASS,
        )
        comp.scheduler.add_condition(
            b, pnl.AfterPass(1)
        )
        comp.run([1], num_trials=2)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #
        #  Trial 1:
        #    Pass 1:
        #       a
        #       (C-1)
        #    Pass 2:
        #       a
        #       (C-2)
        #    Pass 2:
        #       a   b
        #       (C-3)
        #  Trial 2:
        #    Pass 1:
        #       a
        #       (C-4)
        #    Pass 2:
        #       a
        #       (C-5)
        #    Pass 3:
        #       a   b
        #       (C-6)
        assert c.value == [6]
        assert c.execution_count == 6
        assert comp.results == [[2], [5]]

    def test_trial_before(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.BEFORE,
            controller_time_scale=pnl.TimeScale.TRIAL
        )
        comp.run([1], num_trials=2)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #
        #  (C-1) Trial 1:
        #    a  b
        #  (C-2) Trial 2:
        #    a  b
        #
        assert c.value == [2]
        assert c.execution_count == 2
        assert comp.results == [[1], [2]]

    def test_trial_after(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.AFTER,
            controller_time_scale=pnl.TimeScale.TRIAL
        )
        comp.run([1], num_trials=2)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #
        #  Trial 1:
        #    a  b
        #    (C-1)
        #  Trial 2:
        #    a  b
        #    (C-2)
        #
        assert c.value == [2]
        assert c.execution_count == 2
        assert comp.results == [[1], [1]]

    def test_run_before(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.BEFORE,
            controller_time_scale=pnl.TimeScale.RUN
        )
        comp.run([1], num_trials=2)
        comp.run([1], num_trials=2)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #  (C-1)
        #   Run 1:
        #    Trial 1:
        #      a  b
        #    Trial 2:
        #      a  b
        #  (C-2)
        #   Run 2:
        #    Trial 1:
        #      a  b
        #    Trial 2:
        #      a  b
        assert c.value == [2]
        assert c.execution_count == 2
        assert comp.results == [[1], [1], [2], [2]]

    def test_run_after(self):
        a = pnl.ProcessingMechanism()
        b = pnl.ProcessingMechanism()
        c = pnl.ControlMechanism(
            default_variable=1,
            function=pnl.SimpleIntegrator,
            control=pnl.ControlSignal(modulates=(pnl.SLOPE, b))
        )
        comp = pnl.Composition(
            pathways=[a, b],
            controller=c,
            controller_mode=pnl.AFTER,
            controller_time_scale=pnl.TimeScale.RUN
        )
        comp.run([1], num_trials=2)
        comp.run([1], num_trials=2)
        # Controller executions
        # (C-<#>) == controller execution followed by val as of the end of that execution, increments by 1
        # on each execution
        #  (C-1)
        #   Run 1:
        #    Trial 1:
        #      a  b
        #    Trial 2:
        #      a  b
        #  (C-2)
        #   Run 2:
        #    Trial 1:
        #      a  b
        #    Trial 2:
        #      a  b
        assert c.value == [2]
        assert c.execution_count == 2
        assert comp.results == [[1], [1], [1], [1]]
