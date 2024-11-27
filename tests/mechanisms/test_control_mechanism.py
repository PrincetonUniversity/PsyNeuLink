import functools
import numpy as np
import psyneulink as pnl
import pytest

import psyneulink.core.components.functions.nonstateful.transferfunctions

class TestLCControlMechanism:

    @pytest.mark.mechanism
    @pytest.mark.control_mechanism
    @pytest.mark.composition
    @pytest.mark.benchmark(group="LCControlMechanism Default")
    def test_lc_control_mechanism_as_controller(self, benchmark):
        G = 1.0
        k = 0.5
        starting_value_LC = 2.0
        user_specified_gain = 1.0

        A = pnl.TransferMechanism(function=psyneulink.core.components.functions.nonstateful.transferfunctions.Logistic(gain=user_specified_gain), name='A')
        B = pnl.TransferMechanism(function=psyneulink.core.components.functions.nonstateful.transferfunctions.Logistic(gain=user_specified_gain), name='B')
        C = pnl.Composition()
        LC = pnl.LCControlMechanism(
            modulated_mechanisms=[A, B],
            base_level_gain=G,
            scaling_factor_gain=k,
            objective_mechanism=pnl.ObjectiveMechanism(
                function=psyneulink.core.components.functions.nonstateful.transferfunctions.Linear,
                monitor=[B],
                name='LC ObjectiveMechanism'
            )
        )
        C.add_linear_processing_pathway([A,B])
        C.add_controller(LC)

        for output_port in LC.output_ports:
            output_port.parameters.value.set(output_port.value * starting_value_LC, C, override=True)

        LC.reset_stateful_function_when = pnl.Never()

        gain_created_by_LC_output_port_1 = []
        mod_gain_assigned_to_A = []
        base_gain_assigned_to_A = []
        mod_gain_assigned_to_B = []
        base_gain_assigned_to_B = []

        def report_trial(composition):
            from psyneulink import parse_context
            context = parse_context(composition)
            gain_created_by_LC_output_port_1.append(LC.output_ports[0].parameters.value.get(context))
            mod_gain_assigned_to_A.append([A.get_mod_gain(composition)])
            mod_gain_assigned_to_B.append([B.get_mod_gain(composition)])
            base_gain_assigned_to_A.append(A.function.gain.base)
            base_gain_assigned_to_B.append(B.function.gain.base)

        C._analyze_graph()
        benchmark(C.run, inputs={A: [[1.0], [1.0], [1.0], [1.0], [1.0]]},
              call_after_trial=functools.partial(report_trial, C))

        # (1) First value of gain in mechanisms A and B must be whatever we hardcoded for LC starting value
        assert mod_gain_assigned_to_A[0] == [starting_value_LC]

        # (2) _gain should always be set to user-specified value
        for i in range(5):
            assert base_gain_assigned_to_A[i] == user_specified_gain
            assert base_gain_assigned_to_B[i] == user_specified_gain

        # (3) LC output on trial n becomes gain of A and B on trial n + 1
        np.testing.assert_allclose(mod_gain_assigned_to_A[1:], gain_created_by_LC_output_port_1[0:-1])

        # (4) mechanisms A and B should always have the same gain values (b/c they are identical)
        np.testing.assert_allclose(mod_gain_assigned_to_A, mod_gain_assigned_to_B)

    @pytest.mark.mechanism
    @pytest.mark.control_mechanism
    @pytest.mark.benchmark(group="LCControlMechanism Basic")
    def test_lc_control_mech_basic(self, benchmark, mech_mode):

        LC = pnl.LCControlMechanism(
            base_level_gain=3.0,
            scaling_factor_gain=0.5,
            default_variable = 10.0
        )
        EX = pytest.helpers.get_mech_execution(LC, mech_mode)

        val = benchmark(EX, [10.0])
        expected = [[3.001397762387422]]
        # The difference in result shape is caused by shape mismatch in output port values.
        # The default shape is 1D, giving 2D overall result in compiled mode.
        # The true results are 2D per port, giving 3d overall result in Python mode.
        if mech_mode == 'Python':
            expected = [[ex] for ex in expected]
        np.testing.assert_allclose(val, expected)

    @pytest.mark.composition
    def test_lc_control_monitored_and_modulated_mechanisms_composition(self):
        """Test default configuration of LCControlMechanism with monitored and modulated mechanisms in a Composition
        Test that it implements an ObjectiveMechanism by default, that uses CombineMeans
        Test that ObjectiveMechanism can monitor Mechanisms with values of different lengths,
             and generate a scalar output.
        Test that it modulates all of the ProcessingMechanisms in the Composition but not the ObjectiveMechanism
        """

        T_1 = pnl.TransferMechanism(name='T_1', input_shapes=2)
        T_2 = pnl.TransferMechanism(name='T_2', input_shapes=3)

        C = pnl.Composition(pathways=[T_1, np.array([[1,2,3],[4,5,6]]), T_2])
        LC = pnl.LCControlMechanism(monitor_for_control=[T_1, T_2],
                                    modulated_mechanisms=C)
        C.add_node(LC)
        assert len(LC.control_signals)==1
        assert len(LC.control_signals[0].efferents)==2
        assert LC.path_afferents[0].sender.owner == LC.objective_mechanism
        assert isinstance(LC.objective_mechanism.function, pnl.CombineMeans)
        assert len(LC.objective_mechanism.input_ports[0].value) == 2
        assert len(LC.objective_mechanism.input_ports[1].value) == 3
        assert len(LC.objective_mechanism.output_ports[pnl.OUTCOME].value) == 1
        assert T_1.parameter_ports[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents
        assert T_2.parameter_ports[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents

        result = C.run(inputs={T_1:[1,2]})#, T_2:[3,4,5]
        assert LC.objective_mechanism.value == (np.mean(T_1.value) + np.mean(T_2.value))


@pytest.mark.composition
class TestControlMechanism:

    def test_control_modulation(self):
        Tx = pnl.TransferMechanism(name='Tx')
        Ty = pnl.TransferMechanism(name='Ty')
        Tz = pnl.TransferMechanism(name='Tz')
        C = pnl.ControlMechanism(
                # function=pnl.Linear,
                default_variable=[1],
                monitor_for_control=Ty,
                objective_mechanism=True,
                control_signals=pnl.ControlSignal(modulation=pnl.OVERRIDE,
                                                  modulates=(pnl.SLOPE, Tz)))
        comp=pnl.Composition(pathways=[[Tx, Tz],[Ty, C]])
        # comp.show_graph()

        assert Tz.parameter_ports[pnl.SLOPE].mod_afferents[0].sender.owner == C
        assert C.parameters.control_allocation.get() == [1]
        result = comp.run(inputs={Tx:[1,1], Ty:[4,4]})
        np.testing.assert_array_equal(comp.results, [[[4.], [4.]], [[4.], [4.]]])


    def test_identicalness_of_control_and_gating(self):
        """Tests same configuration as gating in tests/mechansims/test_gating_mechanism"""
        Input_Layer = pnl.TransferMechanism(name='Input Layer', function=pnl.Logistic, input_shapes=2)
        Hidden_Layer_1 = pnl.TransferMechanism(name='Hidden Layer_1', function=pnl.Logistic, input_shapes=5)
        Hidden_Layer_2 = pnl.TransferMechanism(name='Hidden Layer_2', function=pnl.Logistic, input_shapes=4)
        Output_Layer = pnl.TransferMechanism(name='Output Layer', function=pnl.Logistic, input_shapes=3)

        Control_Mechanism = pnl.ControlMechanism(
            input_shapes=[1], control=[Hidden_Layer_1.input_port,
                                       Hidden_Layer_2.input_port,
                                       Output_Layer.input_port])

        Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        Middle_Weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        Output_Weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)
        # This projection is specified in add_backpropagation_learning_pathway method below
        Input_Weights = pnl.MappingProjection(name='Input Weights',matrix=Input_Weights_matrix)
        # This projection is "discovered" by add_backpropagation_learning_pathway method below
        Middle_Weights = pnl.MappingProjection(name='Middle Weights',sender=Hidden_Layer_1,receiver=Hidden_Layer_2,
            matrix={
                pnl.VALUE: Middle_Weights_matrix,
                pnl.FUNCTION: pnl.AccumulatorIntegrator,
                pnl.FUNCTION_PARAMS: {
                    pnl.DEFAULT_VARIABLE: Middle_Weights_matrix,
                    pnl.INITIALIZER: Middle_Weights_matrix,
                    pnl.RATE: Middle_Weights_matrix
                },
            }
        )
        Output_Weights = pnl.MappingProjection(sender=Hidden_Layer_2,
                                           receiver=Output_Layer,
                                           matrix=Output_Weights_matrix)

        pathway = [Input_Layer, Input_Weights, Hidden_Layer_1, Hidden_Layer_2, Output_Layer]
        comp = pnl.Composition()
        backprop_pathway = comp.add_backpropagation_learning_pathway(
            pathway=pathway,
            loss_spec=None,
        )
        # c.add_linear_processing_pathway(pathway=z)
        comp.add_node(Control_Mechanism)

        np.testing.assert_allclose(np.asfarray(Control_Mechanism.parameters.control_allocation.get()), [[0], [0], [0]])

        stim_list = {
            Input_Layer: [[-1, 30]],
            Control_Mechanism: [1.0],
            backprop_pathway.target: [[0, 0, 1]]}

        comp.learn(num_trials=3, inputs=stim_list)

        expected_results =[[[0.81493513, 0.85129046, 0.88154205]],
                           [[0.81331773, 0.85008207, 0.88157851]],
                           [[0.81168332, 0.84886047, 0.88161468]]]
        np.testing.assert_allclose(comp.results, expected_results)

        stim_list[Control_Mechanism]=[0.0]
        results = comp.learn(num_trials=1, inputs=stim_list)
        expected_results = [[0.5, 0.5, 0.5]]
        np.testing.assert_allclose(results, expected_results)

        stim_list[Control_Mechanism]=[2.0]
        results = comp.learn(num_trials=1, inputs=stim_list)
        expected_results = [[0.96941429, 0.9837254 , 0.99217549]]
        np.testing.assert_allclose(results, expected_results)

    def test_control_of_all_input_ports(self, comp_mode):
        mech = pnl.ProcessingMechanism(input_ports=['A','B','C'])
        control_mech = pnl.ControlMechanism(control=mech.input_ports)
        comp = pnl.Composition()
        comp.add_nodes([(mech, pnl.NodeRole.INPUT), (control_mech, pnl.NodeRole.INPUT)])
        results = comp.run(inputs={mech:[[2],[2],[2]], control_mech:[2]}, num_trials=2, execution_mode=comp_mode)

        np.testing.assert_allclose(control_mech.parameters.control_allocation.get(), [[1], [1], [1]])
        np.testing.assert_allclose(results, [[4], [4], [4]])


    def test_control_of_all_output_ports(self, comp_mode):
        mech = pnl.ProcessingMechanism(output_ports=[{pnl.VARIABLE: (pnl.OWNER_VALUE, 0)},
                                                      {pnl.VARIABLE: (pnl.OWNER_VALUE, 0)},
                                                      {pnl.VARIABLE: (pnl.OWNER_VALUE, 0)}],)
        control_mech = pnl.ControlMechanism(control=mech.output_ports)
        comp = pnl.Composition()
        comp.add_nodes([(mech, pnl.NodeRole.INPUT), (control_mech, pnl.NodeRole.INPUT)])
        results = comp.run(inputs={mech:[[2]], control_mech:[3]}, num_trials=2, execution_mode=comp_mode)

        np.testing.assert_allclose(control_mech.parameters.control_allocation.get(), [[1], [1], [1]])
        np.testing.assert_allclose(results, [[6], [6], [6]])

    def test_control_signal_default_allocation_specification(self):

        m1 = pnl.ProcessingMechanism()
        m2 = pnl.ProcessingMechanism()
        m3 = pnl.ProcessingMechanism()

        # default_allocation *not* specified in constructor of pnl.ControlMechanism,
        #     so should be set to defaultControlAllocation (=[1])
        c1 = pnl.ControlMechanism(name='C1',
                                  default_variable=[10],
                                  control_signals=[pnl.ControlSignal(modulates=(pnl.SLOPE, m1)), # test for assignment to defaultControlAllocation
                                                   pnl.ControlSignal(default_allocation=2,       # test for scalar assignment
                                                                     modulates=(pnl.SLOPE, m2)),
                                                   pnl.ControlSignal(default_allocation=[3],     # test for array assignment
                                                                     modulates=(pnl.SLOPE, m3))])
        comp = pnl.Composition()
        comp.add_nodes([m1,m2,m3])
        comp.add_controller(c1)
        # Default controL_allocation should = defaultControlAllocation (since default_allocation arg was not specified)
        np.testing.assert_allclose(c1.defaults.control_allocation, [1])
        # Initial control_allocation should reflect the default input ([10])
        np.testing.assert_allclose(c1.parameters.control_allocation.get(), [[10], [10], [10]])
        # Initial values of the ControlSignals should reflect *their* default assignments (since not yet executed)
        # and parameters they modulate (SLOPE) should reflect *their* initial values (1)
        assert c1.control_signals[0].value == [1] # defaultControlAllocation ([1]) should be assigned,
                                                   # as no default_allocation from pnl.ControlMechanism
        assert m1.parameter_ports[pnl.SLOPE].value == [1]
        assert c1.control_signals[1].value == [2]      # default_allocation from pnl.ControlSignal (converted scalar)
        assert m2.parameter_ports[pnl.SLOPE].value == [1]
        assert c1.control_signals[2].value == [3]      # default_allocation from pnl.ControlSignal
        assert m3.parameter_ports[pnl.SLOPE].value == [1]
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        # Result should reflect:
        # 1) use of initial ControlSignal values to set parmeters
        # 2) updating of the ControlSignal values ([10]'s) from ControlMechanism's default_variable
        #     (since it does not have any other source of input control_allocation)
        np.testing.assert_allclose(result, [[2.], [6.], [12.]])
        assert c1.control_signals[0].value == [10]
        assert m1.parameter_ports[pnl.SLOPE].value == [1]
        assert c1.control_signals[1].value == [10]
        assert m2.parameter_ports[pnl.SLOPE].value == [2]
        assert c1.control_signals[2].value == [10]
        assert m3.parameter_ports[pnl.SLOPE].value == [3]
        # Results should now reflect use of updated ControlSignal values to set parmeters
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        np.testing.assert_allclose(result, [[20.], [30.], [40.]])
        assert c1.control_signals[0].value == [10]
        assert m1.parameter_ports[pnl.SLOPE].value == [10]
        assert c1.control_signals[1].value == [10]
        assert m2.parameter_ports[pnl.SLOPE].value == [10]
        assert c1.control_signals[2].value == [10]
        assert m3.parameter_ports[pnl.SLOPE].value == [10]

        # default_allocation *is* specified in constructor of pnl.ControlMechanism,
        #     so should be used unless specified in pnl.ControlSignal constructor
        c2 = pnl.ControlMechanism(
                name='C3',
                default_variable=[10],
                default_allocation=[4],
                # Test synonyms allowed for **control**: generic **modulates**, and even more generic **projections**):
                control_signals=[pnl.ControlSignal(control=(pnl.SLOPE, m1)),  # tests for assignment to default_allocation
                                 pnl.ControlSignal(default_allocation=5,  # tests for override of default_allocation
                                                   modulates=(pnl.SLOPE, m2)),
                                 pnl.ControlSignal(default_allocation=[6],  # as above same but with array
                                                   projections=(pnl.SLOPE, m3))])
        comp = pnl.Composition()
        comp.add_nodes([m1,m2,m3])
        comp.add_controller(c2)
        np.testing.assert_allclose(c2.parameters.control_allocation.get(), [[10], [10], [10]])
        assert c2.control_signals[0].value == [4]        # default_allocation from pnl.ControlMechanism assigned
        assert m1.parameter_ports[pnl.SLOPE].value == [10]  # has not yet received pnl.ControlSignal value
        assert c2.control_signals[1].value == [5]        # default_allocation from pnl.ControlSignal assigned (converted scalar)
        assert m2.parameter_ports[pnl.SLOPE].value == [10]
        assert c2.control_signals[2].value == [6]        # default_allocation from pnl.ControlSignal assigned
        assert m3.parameter_ports[pnl.SLOPE].value == [10]
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        np.testing.assert_allclose(result, [[8.], [15.], [24.]])
        assert c2.control_signals[0].value == [10]
        assert m1.parameter_ports[pnl.SLOPE].value == [4]
        assert c2.control_signals[1].value == [10]
        assert m2.parameter_ports[pnl.SLOPE].value == [5]
        assert c2.control_signals[2].value == [10]
        assert m3.parameter_ports[pnl.SLOPE].value == [6]
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        np.testing.assert_allclose(result, [[20.], [30.], [40.]])
        assert c2.control_signals[0].value == [10]
        assert m1.parameter_ports[pnl.SLOPE].value == [10]
        assert c2.control_signals[1].value == [10]
        assert m2.parameter_ports[pnl.SLOPE].value == [10]
        assert c2.control_signals[2].value == [10]
        assert m3.parameter_ports[pnl.SLOPE].value == [10]
