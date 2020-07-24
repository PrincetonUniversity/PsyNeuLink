import functools
import numpy as np
import psyneulink as pnl
import pytest

import psyneulink.core.components.functions.transferfunctions
import psyneulink.core.llvm as pnlvm

class TestLCControlMechanism:

    @pytest.mark.mechanism
    @pytest.mark.control_mechanism
    @pytest.mark.benchmark(group="LCControlMechanism Default")
    @pytest.mark.parametrize("mode", ['Python'])
    def test_lc_control_mechanism_as_controller(self, benchmark, mode):
        G = 1.0
        k = 0.5
        starting_value_LC = 2.0
        user_specified_gain = 1.0

        A = pnl.TransferMechanism(function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=user_specified_gain), name='A')
        B = pnl.TransferMechanism(function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=user_specified_gain), name='B')
        C = pnl.Composition()
        LC = pnl.LCControlMechanism(
            modulated_mechanisms=[A, B],
            base_level_gain=G,
            scaling_factor_gain=k,
            objective_mechanism=pnl.ObjectiveMechanism(
                function=psyneulink.core.components.functions.transferfunctions.Linear,
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
            base_gain_assigned_to_A.append(A.function.gain)
            base_gain_assigned_to_B.append(B.function.gain)

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
        assert np.allclose(mod_gain_assigned_to_A[1:], gain_created_by_LC_output_port_1[0:-1])

        # (4) mechanisms A and B should always have the same gain values (b/c they are identical)
        assert np.allclose(mod_gain_assigned_to_A, mod_gain_assigned_to_B)

    @pytest.mark.mechanism
    @pytest.mark.control_mechanism
    @pytest.mark.benchmark(group="LCControlMechanism Basic")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_lc_control_mech_basic(self, benchmark, mode):

        LC = pnl.LCControlMechanism(
            base_level_gain=3.0,
            scaling_factor_gain=0.5,
            default_variable = 10.0
        )
        if mode == 'Python':
            EX = LC.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(LC)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(LC)
            EX = e.cuda_execute

        val = EX([10.0])

        # LLVM returns combination of all output ports so let's do that for
        # Python as well
        if mode == 'Python':
            val = [s.value for s in LC.output_ports]

        benchmark(EX, [10.0])

        # All values are the same because LCControlMechanism assigns all of its ControlSignals to the same value
        # (the 1st item of its function's value).
        # FIX: 6/6/19 - Python returns 3d array but LLVM returns 2d array
        #               (np.allclose bizarrely passes for LLVM because all the values are the same)
        assert np.allclose(val, [[[3.00139776]], [[3.00139776]], [[3.00139776]], [[3.00139776]]])

    def test_lc_control_modulated_mechanisms_all(self):

        T_1 = pnl.TransferMechanism(name='T_1')
        T_2 = pnl.TransferMechanism(name='T_2')

        # S = pnl.System(processes=[pnl.proc(T_1, T_2, LC)])
        C = pnl.Composition(pathways=[T_1, T_2])
        LC = pnl.LCControlMechanism(monitor_for_control=[T_1, T_2],
                                    modulated_mechanisms=C)
        C.add_node(LC)

        assert len(LC.control_signals)==1
        assert len(LC.control_signals[0].efferents)==2
        assert T_1.parameter_ports[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents
        assert T_2.parameter_ports[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents

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
        result = comp.run(inputs={Tx:[1,1], Ty:[4,4]})
        assert comp.results == [[[4.], [4.]], [[4.], [4.]]]

    def test_identicalness_of_control_and_gating(self):
        """Tests same configuration as gating in tests/mechansims/test_gating_mechanism"""
        Input_Layer = pnl.TransferMechanism(name='Input Layer', function=pnl.Logistic, size=2)
        Hidden_Layer_1 = pnl.TransferMechanism(name='Hidden Layer_1', function=pnl.Logistic, size=5)
        Hidden_Layer_2 = pnl.TransferMechanism(name='Hidden Layer_2', function=pnl.Logistic, size=4)
        Output_Layer = pnl.TransferMechanism(name='Output Layer', function=pnl.Logistic, size=3)

        Control_Mechanism = pnl.ControlMechanism(size=[1], control=[Hidden_Layer_1.input_port,
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
            loss_function=None,
        )
        # c.add_linear_processing_pathway(pathway=z)
        comp.add_node(Control_Mechanism)

        stim_list = {
            Input_Layer: [[-1, 30]],
            Control_Mechanism: [1.0],
            backprop_pathway.target: [[0, 0, 1]]}

        comp.learn(num_trials=3, inputs=stim_list)

        expected_results =[[[0.81493513, 0.85129046, 0.88154205]],
                           [[0.81331773, 0.85008207, 0.88157851]],
                           [[0.81168332, 0.84886047, 0.88161468]]]
        assert np.allclose(comp.results, expected_results)

        stim_list[Control_Mechanism]=[0.0]
        results = comp.learn(num_trials=1, inputs=stim_list)
        expected_results = [[[0.5, 0.5, 0.5]]]
        assert np.allclose(results, expected_results)

        stim_list[Control_Mechanism]=[2.0]
        results = comp.learn(num_trials=1, inputs=stim_list)
        expected_results = [[0.96941429, 0.9837254 , 0.99217549]]
        assert np.allclose(results, expected_results)

    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_control_of_all_input_ports(self, mode):
        mech = pnl.ProcessingMechanism(input_ports=['A','B','C'])
        control_mech = pnl.ControlMechanism(control=mech.input_ports)
        comp = pnl.Composition()
        comp.add_nodes([(mech, pnl.NodeRole.INPUT), (control_mech, pnl.NodeRole.INPUT)])
        results = comp.run(inputs={mech:[[2],[2],[2]], control_mech:[2]}, num_trials=2, bin_execute=mode)
        np.allclose(results, [[4],[4],[4]])

    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_control_of_all_output_ports(self, mode):
        mech = pnl.ProcessingMechanism(output_ports=[{pnl.VARIABLE: (pnl.OWNER_VALUE, 0)},
                                                      {pnl.VARIABLE: (pnl.OWNER_VALUE, 0)},
                                                      {pnl.VARIABLE: (pnl.OWNER_VALUE, 0)}],)
        control_mech = pnl.ControlMechanism(control=mech.output_ports)
        comp = pnl.Composition()
        comp.add_nodes([(mech, pnl.NodeRole.INPUT), (control_mech, pnl.NodeRole.INPUT)])
        results = comp.run(inputs={mech:[[2]], control_mech:[3]}, num_trials=2, bin_execute=mode)
        np.allclose(results, [[6],[6],[6]])

    def test_control_signal_default_allocation_specification(self):

        m1 = pnl.ProcessingMechanism()
        m2 = pnl.ProcessingMechanism()
        m3 = pnl.ProcessingMechanism()

        # default_allocation not specified in constructor of pnl.ControlMechanism,
        #     so should be set to defaultControlAllocation (=[1]) if not specified in pnl.ControlSignal constructor
        c1 = pnl.ControlMechanism(
                name='C1',
                default_variable=[10],
                control_signals=[pnl.ControlSignal(modulates=(pnl.SLOPE, m1)),  # test for assignment to defaultControlAllocation
                                 pnl.ControlSignal(default_allocation=2,  # test for scalar assignment
                                                   modulates=(pnl.SLOPE, m2)),
                                 pnl.ControlSignal(default_allocation=[3],  # test for array assignment
                                                   modulates=(pnl.SLOPE, m3))])
        comp = pnl.Composition()
        comp.add_nodes([m1,m2,m3])
        comp.add_controller(c1)
        assert c1.control_signals[0].value == [10] # defaultControlAllocation should be assigned
                                                   # (as no default_allocation from pnl.ControlMechanism)
        assert m1.parameter_ports[pnl.SLOPE].value == [1]
        assert c1.control_signals[1].value == [2]      # default_allocation from pnl.ControlSignal (converted scalar)
        assert m2.parameter_ports[pnl.SLOPE].value == [1]
        assert c1.control_signals[2].value == [3]      # default_allocation from pnl.ControlSignal
        assert m3.parameter_ports[pnl.SLOPE].value == [1]
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        assert np.allclose(result, [[20.], [6.], [12.]])
        assert c1.control_signals[0].value == [10]
        assert m1.parameter_ports[pnl.SLOPE].value == [10]
        assert c1.control_signals[1].value == [10]
        assert m2.parameter_ports[pnl.SLOPE].value == [2]
        assert c1.control_signals[2].value == [10]
        assert m3.parameter_ports[pnl.SLOPE].value == [3]
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        assert np.allclose(result, [[20.], [30.], [40.]])
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
                control_signals=[pnl.ControlSignal(modulates=(pnl.SLOPE, m1)),  # tests for assignment to default_allocation
                                 pnl.ControlSignal(default_allocation=5,  # tests for override of default_allocation
                                                   modulates=(pnl.SLOPE, m2)),
                                 pnl.ControlSignal(default_allocation=[6],  # as above same but with array
                                                   modulates=(pnl.SLOPE, m3))])
        comp = pnl.Composition()
        comp.add_nodes([m1,m2,m3])
        comp.add_controller(c2)
        assert c2.control_signals[0].value == [4]        # default_allocation from pnl.ControlMechanism assigned
        assert m1.parameter_ports[pnl.SLOPE].value == [10]  # has not yet received pnl.ControlSignal value
        assert c2.control_signals[1].value == [5]        # default_allocation from pnl.ControlSignal assigned (converted scalar)
        assert m2.parameter_ports[pnl.SLOPE].value == [10]
        assert c2.control_signals[2].value == [6]        # default_allocation from pnl.ControlSignal assigned
        assert m3.parameter_ports[pnl.SLOPE].value == [10]
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        assert np.allclose(result, [[8.], [15.], [24.]])
        assert c2.control_signals[0].value == [10]
        assert m1.parameter_ports[pnl.SLOPE].value == [4]
        assert c2.control_signals[1].value == [10]
        assert m2.parameter_ports[pnl.SLOPE].value == [5]
        assert c2.control_signals[2].value == [10]
        assert m3.parameter_ports[pnl.SLOPE].value == [6]
        result = comp.run(inputs={m1:[2],m2:[3],m3:[4]})
        assert np.allclose(result, [[20.], [30.], [40.]])
        assert c2.control_signals[0].value == [10]
        assert m1.parameter_ports[pnl.SLOPE].value == [10]
        assert c2.control_signals[1].value == [10]
        assert m2.parameter_ports[pnl.SLOPE].value == [10]
        assert c2.control_signals[2].value == [10]
        assert m3.parameter_ports[pnl.SLOPE].value == [10]
