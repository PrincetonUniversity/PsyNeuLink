import numpy as np
import pytest

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.functions.selectionfunctions import max_vs_next
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.scheduling.condition import Never, WhenFinished, AtRunStart, AtTrialStart
from psyneulink.library.components.mechanisms.processing.transfer.lcamechanism import \
    LCAMechanism, MAX_VS_AVG, MAX_VS_NEXT, CONVERGENCE

class TestLCA:
    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    @pytest.mark.benchmark(group="LCAMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_LCAMechanism_length_1(self, benchmark, mode):
        T = TransferMechanism(function=Linear(slope=1.0))
        L = LCAMechanism(function=Linear(slope=2.0),
                         self_excitation=3.0,
                         leak=0.5,
                         competition=1.0,  #  competition does not matter because we only have one unit
                         time_step_size=0.1)
        C = Composition()
        C.add_linear_processing_pathway([T,L])
        L.reset_stateful_function_when = Never()
        #  - - - - - - - Equations to be executed  - - - - - - -

        # new_transfer_input =
        # previous_transfer_input
        # + (leak * previous_transfer_input_1 + self_excitation * result1 + competition * result2 + outside_input1) * dt
        # + noise

        # result = new_transfer_input*2.0

        # recurrent_matrix = [[3.0]]

        #  - - - - - - - - - - - - - -  - - - - - - - - - - - -

        C.run(inputs={T: [1.0]}, num_trials=3, bin_execute=mode)

        # - - - - - - - TRIAL 1 - - - - - - -

        # new_transfer_input = 0.0 + ( 0.5 * 0.0 + 3.0 * 0.0 + 0.0 + 1.0)*0.1 + 0.0    =    0.1
        # f(new_transfer_input) = 0.1 * 2.0 = 0.2

        # - - - - - - - TRIAL 2 - - - - - - -

        # new_transfer_input = 0.1 + ( 0.5 * 0.1 + 3.0 * 0.2 + 0.0 + 1.0)*0.1 + 0.0    =    0.265
        # f(new_transfer_input) = 0.265 * 2.0 = 0.53

        # - - - - - - - TRIAL 3 - - - - - - -

        # new_transfer_input = 0.265 + ( 0.5 * 0.265 + 3.0 * 0.53 + 0.0 + 1.0)*0.1 + 0.0    =    0.53725
        # f(new_transfer_input) = 0.53725 * 2.0 = 1.0745

        assert np.allclose(C.results, [[[0.2]], [[0.51]], [[0.9905]]])
        if benchmark.enabled:
            benchmark(C.run, inputs={T: [1.0]}, num_trials=3, bin_execute=mode)

    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    @pytest.mark.benchmark(group="LCAMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_LCAMechanism_length_2(self, benchmark, mode):
        # Note: since the LCAMechanism's threshold is not specified in this test, each execution only updates
        #       the Mechanism once.

        T = TransferMechanism(function=Linear(slope=1.0), size=2)
        L = LCAMechanism(function=Linear(slope=2.0),
                         size=2,
                         self_excitation=3.0,
                         leak=0.5,
                         competition=1.0,
                         time_step_size=0.1)

        C = Composition()
        C.add_linear_processing_pathway([T,L])
        L.reset_stateful_function_when = Never()
        #  - - - - - - - Equations to be executed  - - - - - - -

        # new_transfer_input =
        # previous_transfer_input
        # + (leak * previous_transfer_input_1 + self_excitation * result1 + competition * result2 + outside_input1) * dt
        # + noise

        # result = new_transfer_input*2.0

        # recurrent_matrix = [[3.0]]

        #  - - - - - - - - - - - - - -  - - - - - - - - - - - -

        C.run(inputs={T: [1.0, 2.0]}, num_trials=3, bin_execute=mode)

        # - - - - - - - TRIAL 1 - - - - - - -

        # new_transfer_input_1 = 0.0 + ( 0.5 * 0.0 + 3.0 * 0.0 - 1.0*0.0 + 1.0)*0.1 + 0.0    =    0.1
        # f(new_transfer_input_1) = 0.1 * 2.0 = 0.2

        # new_transfer_input_2 = 0.0 + ( 0.5 * 0.0 + 3.0 * 0.0 - 1.0*0.0 + 2.0)*0.1 + 0.0    =    0.2
        # f(new_transfer_input_2) = 0.2 * 2.0 = 0.4

        # - - - - - - - TRIAL 2 - - - - - - -

        # new_transfer_input = 0.1 + ( 0.5 * 0.1 + 3.0 * 0.2 - 1.0*0.4 + 1.0)*0.1 + 0.0    =    0.225
        # f(new_transfer_input) = 0.265 * 2.0 = 0.45

        # new_transfer_input_2 = 0.2 + ( 0.5 * 0.2 + 3.0 * 0.4 - 1.0*0.2 + 2.0)*0.1 + 0.0    =    0.51
        # f(new_transfer_input_2) = 0.1 * 2.0 = 1.02

        # - - - - - - - TRIAL 3 - - - - - - -

        # new_transfer_input = 0.225 + ( 0.5 * 0.225 + 3.0 * 0.45 - 1.0*1.02 + 1.0)*0.1 + 0.0    =    0.36925
        # f(new_transfer_input) = 0.36925 * 2.0 = 0.7385

        # new_transfer_input_2 = 0.51 + ( 0.5 * 0.51 + 3.0 * 1.02 - 1.0*0.45 + 2.0)*0.1 + 0.0    =    0.9965
        # f(new_transfer_input_2) = 0.9965 * 2.0 = 1.463

        assert np.allclose(C.results, [[[0.2, 0.4]], [[0.43, 0.98]], [[0.6705, 1.833]]])
        if benchmark.enabled:
            benchmark(C.run, inputs={T: [1.0, 2.0]}, num_trials=3, bin_execute=mode)

    def test_equivalance_of_threshold_and_when_finished_condition(self):
        # Note: This tests the equivalence of results when:
        #       execute_until_finished is True for the LCAMechanism (by default)
        #           and the call to execution loops until it reaches threshold (1st test)
        #       vs. when execute_until_finished is False and a condition is added to the scheduler
        #           that causes the LCAMechanism it to execute until it reaches threshold (2nd test).

        # loop Mechanism's call to execute
        lca_until_thresh = LCAMechanism(size=2, leak=0.5, threshold=0.7) # Note: , execute_to_threshold=True by default
        response = ProcessingMechanism(size=2)
        comp = Composition()
        comp.add_linear_processing_pathway([lca_until_thresh, response])
        result1 = comp.run(inputs={lca_until_thresh:[1,0]})

        # loop Composition's call to Mechanism
        lca_single_step = LCAMechanism(size=2, leak=0.5, threshold=0.7, execute_until_finished=False)
        comp2 = Composition()
        response2 = ProcessingMechanism(size=2)
        comp2.add_linear_processing_pathway([lca_single_step,response2])
        comp2.scheduler.add_condition(response2, WhenFinished(lca_single_step))
        result2 = comp2.run(inputs={lca_single_step:[1,0]})
        assert np.allclose(result1, result2)

    def test_LCAMechanism_matrix(self):
        matrix = [[0,-2],[-2,0]]
        lca1 = LCAMechanism(size=2, leak=0.5, competition=2)
        assert np.allclose(lca1.matrix.base, matrix)
        lca2 = LCAMechanism(size=2, leak=0.5, matrix=matrix)
        assert np.allclose(lca1.matrix.base, lca2.matrix.base)

    # Note: In the following tests, since the LCAMechanism's threshold is specified
    #       it executes until the it reaches threshold.
    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    @pytest.mark.benchmark(group="LCAMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_LCAMechanism_threshold(self, benchmark, mode):
        lca = LCAMechanism(size=2, leak=0.5, threshold=0.7)
        comp = Composition()
        comp.add_node(lca)
        result = comp.run(inputs={lca:[1,0]}, bin_execute=mode)
        assert np.allclose(result, [0.70005431, 0.29994569])
        if benchmark.enabled:
            benchmark(comp.run, inputs={lca:[1,0]}, bin_execute=mode)

    def test_LCAMechanism_threshold_with_max_vs_next(self):
        lca = LCAMechanism(size=3, leak=0.5, threshold=0.1, threshold_criterion=MAX_VS_NEXT)
        comp = Composition()
        comp.add_node(lca)
        result = comp.run(inputs={lca:[1,0.5,0]})
        assert np.allclose(result, [[0.52490032, 0.42367594, 0.32874867]])

    def test_LCAMechanism_threshold_with_max_vs_avg(self):
        lca = LCAMechanism(size=3, leak=0.5, threshold=0.1, threshold_criterion=MAX_VS_AVG)
        comp = Composition()
        comp.add_node(lca)
        result = comp.run(inputs={lca:[1,0.5,0]})
        assert np.allclose(result, [[0.51180475, 0.44161738, 0.37374946]])

    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    @pytest.mark.benchmark(group="LCAMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_LCAMechanism_threshold_with_convergence(self, benchmark, mode):
        lca = LCAMechanism(size=3, leak=0.5, threshold=0.01, threshold_criterion=CONVERGENCE)
        comp = Composition()
        comp.add_node(lca)
        result = comp.run(inputs={lca:[0,1,2]}, bin_execute=mode)
        assert np.allclose(result, [[0.19153799, 0.5, 0.80846201]])
        if mode == 'Python':
            assert lca.num_executions_before_finished == 18
        if benchmark.enabled:
            benchmark(comp.run, inputs={lca:[0,1,2]}, bin_execute=mode)

    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_equivalance_of_threshold_and_termination_specifications_just_threshold(self, mode):
        # Note: This tests the equivalence of using LCAMechanism-specific threshold arguments and
        #       generic TransferMechanism termination_<*> arguments

        lca_thresh = LCAMechanism(size=2, leak=0.5, threshold=0.7) # Note: , execute_to_threshold=True by default
        response = ProcessingMechanism(size=2)
        comp = Composition()
        comp.add_linear_processing_pathway([lca_thresh, response])
        result1 = comp.run(inputs={lca_thresh:[1,0]}, bin_execute=mode)

        lca_termination = LCAMechanism(size=2,
                                       leak=0.5,
                                       termination_threshold=0.7,
                                       termination_measure=max,
                                       termination_comparison_op='>=')
        comp2 = Composition()
        response2 = ProcessingMechanism(size=2)
        comp2.add_linear_processing_pathway([lca_termination,response2])
        result2 = comp2.run(inputs={lca_termination:[1,0]}, bin_execute=mode)
        assert np.allclose(result1, result2)

    def test_equivalance_of_threshold_and_termination_specifications_max_vs_next(self):
        # Note: This tests the equivalence of using LCAMechanism-specific threshold arguments and
        #       generic TransferMechanism termination_<*> arguments

        lca_thresh = LCAMechanism(size=3, leak=0.5, threshold=0.1, threshold_criterion=MAX_VS_NEXT)
        response = ProcessingMechanism(size=3)
        comp = Composition()
        comp.add_linear_processing_pathway([lca_thresh, response])
        result1 = comp.run(inputs={lca_thresh:[1,0.5,0]})

        lca_termination = LCAMechanism(size=3,
                                       leak=0.5,
                                       termination_threshold=0.1,
                                       termination_measure=max_vs_next,
                                       termination_comparison_op='>=')
        comp2 = Composition()
        response2 = ProcessingMechanism(size=3)
        comp2.add_linear_processing_pathway([lca_termination,response2])
        result2 = comp2.run(inputs={lca_termination:[1,0.5,0]})
        assert np.allclose(result1, result2)

    # def test_LCAMechanism_threshold_with_str(self):
    #     lca = LCAMechanism(size=2, threshold=0.7, threshold_criterion='MY_OUTPUT_PORT',
    #                      output_ports=[RESULT, 'MY_OUTPUT_PORT'])
    #     response = ProcessingMechanism(size=2)
    #     comp = Composition()
    #     comp.add_linear_processing_pathway([lca,response])
    #     comp.scheduler.add_condition(response, WhenFinished(lca))
    #     result = comp.run(inputs={lca:[1,0]})
    #     assert np.allclose(result, [[0.71463572, 0.28536428]])
    #
    # def test_LCAMechanism_threshold_with_int(self):
    #     lca = LCAMechanism(size=2, threshold=0.7, threshold_criterion=1, output_ports=[RESULT, 'MY_OUTPUT_PORT'])
    #     response = ProcessingMechanism(size=2)
    #     comp = Composition()
    #     comp.add_linear_processing_pathway([lca,response])
    #     comp.scheduler.add_condition(response, WhenFinished(lca))
    #     result = comp.run(inputs={lca:[1,0]})
    #     assert np.allclose(result, [[0.71463572, 0.28536428]])

    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_LCAMechanism_DDM_equivalent(self, mode):
        lca = LCAMechanism(size=2, leak=0., threshold=1, auto=0, hetero=0,
                           initial_value=[0, 0], execute_until_finished=False)
        comp1 = Composition()
        comp1.add_node(lca)
        result1 = comp1.run(inputs={lca:[1, -1]}, bin_execute=mode)
        assert np.allclose(result1, [[0.52497918747894, 0.47502081252106]],)


class TestLCAReset:

    def test_reset_run(self):

        L = LCAMechanism(name="L",
                         function=Linear,
                         initial_value=0.5,
                         integrator_mode=True,
                         leak=0.1,
                         competition=0,
                         self_excitation=1.0,
                         time_step_size=1.0,
                         noise=0.0)
        C = Composition(pathways=[L])

        L.reset_stateful_function_when = Never()
        assert np.allclose(L.integrator_function.previous_value, 0.5)
        assert np.allclose(L.initial_value, 0.5)
        assert np.allclose(L.integrator_function.initializer, 0.5)

        C.run(inputs={L: 1.0},
              num_trials=2,
              # reset_stateful_functions_when=AtRunStart(),
              # reset_stateful_functions_when=AtTrialStart(),
              initialize_cycle_values={L: [0.0]})

        # IntegratorFunction fn: previous_value + (rate*previous_value + new_value)*time_step_size + noise*(time_step_size**0.5)

        # Trial 1    |   variable = 1.0 + 0.0
        # integration: 0.5 + (0.1*0.5 + 1.0)*1.0 + 0.0 = 1.55
        # linear fn: 1.55*1.0 = 1.55
        # Trial 2    |   variable = 1.0 + 1.55
        # integration: 1.55 + (0.1*1.55 + 2.55)*1.0 + 0.0 = 4.255
        #  linear fn: 4.255*1.0 = 4.255
        assert np.allclose(L.integrator_function.parameters.previous_value.get(C), 3.755)

        L.integrator_function.reset(0.9, context=C)

        assert np.allclose(L.integrator_function.parameters.previous_value.get(C), 0.9)
        assert np.allclose(L.parameters.value.get(C), 3.755)

        L.reset(0.5, context=C)

        assert np.allclose(L.integrator_function.parameters.previous_value.get(C), 0.5)
        assert np.allclose(L.parameters.value.get(C), 0.5)

        C.run(inputs={L: 1.0},
              num_trials=2)
        # Trial 3    |   variable = 1.0 + 0.5
        # integration: 0.5 + (0.1*0.5 + 1.5)*1.0 + 0.0 = 2.05
        # linear fn: 2.05*1.0 = 2.05
        # Trial 4    |   variable = 1.0 + 2.05
        # integration: 2.05 + (0.1*2.05 + 3.05)*1.0 + 0.0 = 5.305
        #  linear fn: 5.305*1.0 = 5.305
        assert np.allclose(L.integrator_function.parameters.previous_value.get(C), 4.705)
        assert np.allclose(L.initial_value, 0.5)
        assert np.allclose(L.integrator_function.initializer, 0.5)

class TestClip:

    def test_clip_float(self):
        L = LCAMechanism(clip=[-2.0, 2.0],
                         function=Linear,
                         leak=0.5,
                         integrator_mode=False)
        assert np.allclose(L.execute(3.0), 2.0)
        assert np.allclose(L.execute(-3.0), -2.0)

    def test_clip_array(self):
        L = LCAMechanism(default_variable=[[0.0, 0.0, 0.0]],
                         clip=[-2.0, 2.0],
                         leak=0.5,
                         function=Linear,
                         integrator_mode=False)
        assert np.allclose(L.execute([3.0, 0.0, -3.0]), [2.0, 0.0, -2.0])

    def test_clip_2d_array(self):
        L = LCAMechanism(default_variable=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                         clip=[-2.0, 2.0],
                         leak=0.5,
                         function=Linear,
                         integrator_mode=False)
        assert np.allclose(L.execute([[-5.0, -1.0, 5.0], [5.0, -5.0, 1.0], [1.0, 5.0, 5.0]]),
                           [[-2.0, -1.0, 2.0], [2.0, -2.0, 1.0], [1.0, 2.0, 2.0]])
