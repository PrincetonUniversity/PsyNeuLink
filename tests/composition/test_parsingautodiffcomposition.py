import functools
import logging
import timeit as timeit

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.components.system import System
from psyneulink.components.process import Process
from psyneulink.components.functions.function import Linear, Logistic, ReLU, SimpleIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism, TRANSFER_OUTPUT
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.projections.projection import Projection
from psyneulink.components.states.inputstate import InputState
from psyneulink.compositions.composition import Composition, CompositionError, CNodeRole
from psyneulink.compositions.parsingautodiffcomposition import ParsingAutodiffComposition, ParsingAutodiffCompositionError
from psyneulink.compositions.pathwaycomposition import PathwayComposition
from psyneulink.compositions.systemcomposition import SystemComposition
from psyneulink.scheduling.condition import EveryNCalls
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink.scheduling.condition import EveryNPasses, AfterNCalls
from psyneulink.scheduling.time import TimeScale
from psyneulink.globals.keywords import NAME, INPUT_STATE, HARD_CLAMP, SOFT_CLAMP, NO_CLAMP, PULSE_CLAMP

logger = logging.getLogger(__name__)



# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for functions of ParsingAutodiffComposition class that are new (not in Composition)
# or override functions in Composition

# STUFF TO TEST:
# Constructor
# CIM state setup
# Pytorch Creator
# Run/Execute
# 



class TestPACConstructor:
    
    def test_no_args(self):
        comp = ParsingAutodiffComposition()
        assert isinstance(comp, ParsingAutodiffComposition)
    
    def test_two_calls_no_args(self):
        comp = ParsingAutodiffComposition()
        assert isinstance(comp, ParsingAutodiffComposition)
        
        comp_2 = ParsingAutodiffComposition()
        assert isinstance(comp, ParsingAutodiffComposition)
        assert isinstance(comp_2, ParsingAutodiffComposition)
    
    def test_target_CIM(self):
        comp = ParsingAutodiffComposition()
        assert isinstance(comp.target_CIM, CompositionInterfaceMechanism)
        assert comp.target_CIM.composition == comp
        assert comp.target_CIM_states == {}
    
    def test_model(self):
        comp = ParsingAutodiffComposition()
        assert comp.model == None
    
    def test_report_prefs(self):
        comp = ParsingAutodiffComposition()
        assert comp.input_CIM.reportOutputPref == False
        assert comp.output_CIM.reportOutputPref == False
        assert comp.target_CIM.reportOutputPref == False



class TestRun:
    
    def test_xor_training_correctness(self):
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        xor = ParsingAutodiffComposition()
        
        xor.add_c_node(xor_in)
        xor.add_c_node(xor_hid)
        xor.add_c_node(xor_out)
        
        xor.add_projection(sender=xor_in, projection=MappingProjection(), receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=MappingProjection(), receiver=xor_out)
        
        xor_inputs = np.zeros((4,2))
        xor_inputs[0] = [0, 0]
        xor_inputs[1] = [0, 1]
        xor_inputs[2] = [1, 0]
        xor_inputs[3] = [1, 1]
        
        xor_targets = np.zeros((4,1))
        xor_targets[0] = [0]
        xor_targets[1] = [1]
        xor_targets[2] = [1]
        xor_targets[3] = [0]
        
        results = xor.run(inputs={xor_in:xor_inputs}, targets={xor_out:xor_targets}, epochs=10000)
        
        for i in range(len(results[0])):
            assert np.allclose(np.round(results[0][i][0]), xor_targets[i])
    
    def test_xor_training_correctness_with_multiple_run_calls(self):
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        xor = ParsingAutodiffComposition()
        
        xor.add_c_node(xor_in)
        xor.add_c_node(xor_hid)
        xor.add_c_node(xor_out)
        
        xor.add_projection(sender=xor_in, projection=MappingProjection(), receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=MappingProjection(), receiver=xor_out)
        
        xor_inputs = np.zeros((4,2))
        xor_inputs[0] = [0, 0]
        xor_inputs[1] = [0, 1]
        xor_inputs[2] = [1, 0]
        xor_inputs[3] = [1, 1]
        
        xor_targets = np.zeros((4,1))
        xor_targets[0] = [0]
        xor_targets[1] = [1]
        xor_targets[2] = [1]
        xor_targets[3] = [0]
        
        results = None
        for i in range(10000):
            results = xor.run(inputs={xor_in:xor_inputs}, targets={xor_out:xor_targets}, epochs=1)
            
        for i in range(len(results[9999])):
            assert np.allclose(np.round(results[9999][i][0]), xor_targets[i])
    
    @pytest.mark.sarumanthewhite
    def test_xor_training_then_processing(self):
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        hid_map = MappingProjection()
        out_map = MappingProjection()
        
        xor = ParsingAutodiffComposition()
        
        xor.add_c_node(xor_in)
        xor.add_c_node(xor_hid)
        xor.add_c_node(xor_out)
        
        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        
        xor_inputs = np.zeros((4,2))
        xor_inputs[0] = [0, 0]
        xor_inputs[1] = [0, 1]
        xor_inputs[2] = [1, 0]
        xor_inputs[3] = [1, 1]
        
        xor_targets = np.zeros((4,1))
        xor_targets[0] = [0]
        xor_targets[1] = [1]
        xor_targets[2] = [1]
        xor_targets[3] = [0]
        
        results_before_proc = xor.run(inputs={xor_in:xor_inputs}, targets={xor_out:xor_targets}, epochs=10000)
        weights_before_proc, biases_before_proc = xor.get_parameters()
        
        results_proc = xor.run(inputs={xor_in:xor_inputs})
        weights_after_proc, biases_after_proc = xor.get_parameters()
        
        assert(np.shape(results_before_proc) == np.shape(results_proc))
        
        for i in range(4):
            assert np.allclose(results_before_proc[0][i][0], results_proc[1][i][0], atol=0.001)
        
        assert np.allclose(weights_before_proc[hid_map], weights_after_proc[hid_map])
        assert np.allclose(weights_before_proc[out_map], weights_after_proc[out_map])
        assert np.allclose(biases_before_proc[xor_hid], biases_after_proc[xor_hid])
        assert np.allclose(biases_before_proc[xor_out], biases_after_proc[xor_out])
    
    @pytest.mark.parametrize(
        'eps', [
            1,
            10,
            100,
            1000 # ,
            # 10000 # ,
            # 100000
        ]
    )
    def test_xor_training_runtime(self, eps):
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        xor = ParsingAutodiffComposition()
        
        xor.add_c_node(xor_in)
        xor.add_c_node(xor_hid)
        xor.add_c_node(xor_out)
        
        xor.add_projection(sender=xor_in, projection=MappingProjection(), receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=MappingProjection(), receiver=xor_out)
        
        xor_inputs = np.zeros((4,2))
        xor_inputs[0] = [0, 0]
        xor_inputs[1] = [0, 1]
        xor_inputs[2] = [1, 0]
        xor_inputs[3] = [1, 1]
        
        xor_targets = np.zeros((4,1))
        xor_targets[0] = [0]
        xor_targets[1] = [1]
        xor_targets[2] = [1]
        xor_targets[3] = [0]
        
        start = timeit.default_timer()
        result = xor.run(inputs={xor_in:xor_inputs}, targets={xor_out:xor_targets}, epochs=eps) 
        end = timeit.default_timer()
        time = end - start
        
        msg = 'completed training xor model as ParsingAutodiffComposition() for {0} epochs in {1} seconds'.format(eps, time)
        print(msg)
        logger.info(msg)
    
    @pytest.mark.parametrize(
        'eps', [
            10 # ,
            # 100
        ]
    )
    def test_xor_as_system_vs_autodiff_composition(self, eps):
        
        # SET UP SYSTEM
        
        xor_in_sys = TransferMechanism(name='xor_in',
                                       default_variable=np.zeros(3))
        
        xor_hid_sys = TransferMechanism(name='xor_hid',
                                        default_variable=np.zeros(10),
                                        function=Logistic())
        
        xor_out_sys = TransferMechanism(name='xor_out',
                                        default_variable=np.zeros(1),
                                        function=Logistic())
        
        xor_process = Process(pathway=[xor_in_sys,
                                       MappingProjection(),
                                       xor_hid_sys,
                                       MappingProjection(),
                                       xor_out_sys],
                              learning=pnl.LEARNING)
        
        xor_system = System(processes=[xor_process],
                            learning_rate=0.001)
        
        # SET UP COMPOSITION
        
        xor_in_comp = TransferMechanism(name='xor_in',
                                        default_variable=np.zeros(2))
        
        xor_hid_comp = TransferMechanism(name='xor_hid',
                                         default_variable=np.zeros(10),
                                         function=Logistic())
        
        xor_out_comp = TransferMechanism(name='xor_out',
                                         default_variable=np.zeros(1),
                                         function=Logistic())
        
        xor_comp = ParsingAutodiffComposition()
        
        xor_comp.add_c_node(xor_in_comp)
        xor_comp.add_c_node(xor_hid_comp)
        xor_comp.add_c_node(xor_out_comp)
        
        xor_comp.add_projection(sender=xor_in_comp, projection=MappingProjection(), receiver=xor_hid_comp)
        xor_comp.add_projection(sender=xor_hid_comp, projection=MappingProjection(), receiver=xor_out_comp)
        
        # SET UP INPUTS FOR SYSTEM
        
        xor_inputs_sys = np.zeros((4,3))
        xor_inputs_sys[0] = [0, 0, 1]
        xor_inputs_sys[1] = [0, 1, 1]
        xor_inputs_sys[2] = [1, 0, 1]
        xor_inputs_sys[3] = [1, 1, 1]
        
        xor_targets_sys = np.zeros((4,1))
        xor_targets_sys[0] = [0]
        xor_targets_sys[1] = [1]
        xor_targets_sys[2] = [1]
        xor_targets_sys[3] = [0]
        
        # SET UP INPUTS FOR COMPOSITION
        
        xor_inputs_comp = np.zeros((4,2))
        xor_inputs_comp[0] = [0, 0]
        xor_inputs_comp[1] = [0, 1]
        xor_inputs_comp[2] = [1, 0]
        xor_inputs_comp[3] = [1, 1]
        
        xor_targets_comp = np.zeros((4,1))
        xor_targets_comp[0] = [0]
        xor_targets_comp[1] = [1]
        xor_targets_comp[2] = [1]
        xor_targets_comp[3] = [0]
        
        # TRAIN THE SYSTEM, TIME IT
        
        start = timeit.default_timer()
        for i in range(eps):
            for j in range(4):
                
                results = xor_system.run(inputs={xor_in_sys:xor_inputs_sys},
                                         targets={xor_out_sys:xor_targets_sys})
        end = timeit.default_timer()
        sys_time = end - start
        
        msg = 'completed training xor model as System for {0} epochs in {1} seconds'.format(eps, sys_time)
        print(msg)
        logger.info(msg)
        
        # TRAIN THE AUTODIFF COMPOSITION, TIME IT
        
        start = timeit.default_timer()
        result = xor_comp.run(inputs={xor_in_comp:xor_inputs_comp}, targets={xor_out_comp:xor_targets_comp}, epochs=eps) 
        end = timeit.default_timer()
        comp_time = end - start
        
        msg = 'completed training xor model as ParsingAutodiffComposition for {0} epochs in {1} seconds'.format(eps, comp_time)
        print(msg)
        logger.info(msg)
        
        # REPORT SPEEDUP
        
        speedup = np.round((sys_time/comp_time), decimals=2)
        msg = ('training xor model as ParsingAutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)



class TestSemanticNetTraining:
    
    @pytest.mark.whatwearedoing
    def test_sem_net_training_correctness(self):
        
        # MECHANISMS FOR SEMANTIC NET:
        
        nouns_in = TransferMechanism(name="nouns_input", 
                                     default_variable=np.zeros(8))
        
        rels_in = TransferMechanism(name="rels_input", 
                                    default_variable=np.zeros(3))
        
        h1 = TransferMechanism(name="hidden_nouns",
                               default_variable=np.zeros(8),
                               function=Logistic())
        
        h2 = TransferMechanism(name="hidden_mixed",
                               default_variable=np.zeros(15),
                               function=Logistic())
        
        out_sig_I = TransferMechanism(name="sig_outs_I",
                                      default_variable=np.zeros(8),
                                      function=Logistic())
        
        out_sig_is = TransferMechanism(name="sig_outs_is",
                                       default_variable=np.zeros(12),
                                       function=Logistic())
        
        out_sig_has = TransferMechanism(name="sig_outs_has",
                                        default_variable=np.zeros(9),
                                        function=Logistic())
        
        out_sig_can = TransferMechanism(name="sig_outs_can",
                                        default_variable=np.zeros(9),
                                        function=Logistic())
        
        # COMPOSITION FOR SEMANTIC NET
        
        sem_net = ParsingAutodiffComposition()
        
        sem_net.add_c_node(nouns_in)
        sem_net.add_c_node(rels_in)
        sem_net.add_c_node(h1)
        sem_net.add_c_node(h2)
        sem_net.add_c_node(out_sig_I)
        sem_net.add_c_node(out_sig_is)
        sem_net.add_c_node(out_sig_has)
        sem_net.add_c_node(out_sig_can)
        
        sem_net.add_projection(sender=nouns_in, projection=MappingProjection(sender=nouns_in, receiver=h1), receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=MappingProjection(sender=rels_in, receiver=h2), receiver=h2)
        sem_net.add_projection(sender=h1, projection=MappingProjection(sender=h1, receiver=h2), receiver=h2)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_I), receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_is), receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_has), receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_can), receiver=out_sig_can)
        
        # INPUTS & OUTPUTS FOR SEMANTIC NET:
        
        nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
        relations = ['is', 'has', 'can']
        is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
                   'yellow']
        has_list = ['roots', 'leaves', 'bark', 'branches', 'skin', 'feathers', 'wings', 'gills', 'scales']
        can_list = ['grow', 'move', 'swim', 'fly', 'breathe', 'breathe underwater', 'breathe air', 'walk', 'photosynthesize']
        
        nouns_input = np.identity(len(nouns))
        
        rels_input = np.identity(len(relations))
        
        truth_nouns = np.identity(len(nouns))
        
        truth_is = np.zeros((len(nouns), len(is_list)))
        
        truth_is[0, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[1, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[2, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[3, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[4, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[5, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[6, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        truth_is[7, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
        
        truth_has = np.zeros((len(nouns), len(has_list)))
        
        truth_has[0, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[1, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[2, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[3, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[4, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        truth_has[7, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        
        truth_can = np.zeros((len(nouns), len(can_list)))
        
        truth_can[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[3, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[4, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[5, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[6, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        truth_can[7, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        
        # SETTING UP DICTIONARY OF INPUTS/OUTPUTS FOR SEMANTIC NET
        
        inputs_dict = {}
        inputs_dict[nouns_in] = []
        inputs_dict[rels_in] = []
        
        targets_dict = {}
        targets_dict[out_sig_I] = []
        targets_dict[out_sig_is] = []
        targets_dict[out_sig_has] = []
        targets_dict[out_sig_can] = []

        for i in range(len(nouns)):
            for j in range(len(relations)):
                inputs_dict[nouns_in].append(nouns_input[i])
                inputs_dict[rels_in].append(rels_input[j])
                targets_dict[out_sig_I].append(truth_nouns[i])
                targets_dict[out_sig_is].append(truth_is[i])
                targets_dict[out_sig_has].append(truth_has[i])
                targets_dict[out_sig_can].append(truth_can[i])
        
        # TRAIN THE MODEL
        
        result = sem_net.run(inputs=inputs_dict, targets=targets_dict, epochs=2000) # enough epochs to learn fully
        
        # CHECK CORRECTNESS
        
        for i in range(len(result[0])): # go over trial outputs in the single results entry
            for j in range(len(result[0][i])): # go over outputs for each output layer
                
                # get target for terminal node whose output state corresponds to current output
                correct_value = None
                curr_CIM_input_state = sem_net.output_CIM.input_states[j]
                for output_state in sem_net.output_CIM_states.keys():
                    if sem_net.output_CIM_states[output_state][0] == curr_CIM_input_state:
                        node = output_state.owner
                        correct_value = targets_dict[node][i]
                
                # compare model output for terminal node on current trial with target for terminal node on current trial
                assert np.allclose(np.round(result[0][i][j]), correct_value)
    
    @pytest.mark.parametrize(
        'eps', [
            1,
            10,
            100,
            1000 # ,
            # 10000
        ]
    )
    def test_sem_net_training_time(self, eps):
        
        # MECHANISMS FOR SEMANTIC NET:
        
        nouns_in = TransferMechanism(name="nouns_input", 
                                     default_variable=np.zeros(8))
        
        rels_in = TransferMechanism(name="rels_input", 
                                    default_variable=np.zeros(3))
        
        h1 = TransferMechanism(name="hidden_nouns",
                               default_variable=np.zeros(8),
                               function=Logistic())
        
        h2 = TransferMechanism(name="hidden_mixed",
                               default_variable=np.zeros(15),
                               function=Logistic())
        
        out_sig_I = TransferMechanism(name="sig_outs_I",
                                      default_variable=np.zeros(8),
                                      function=Logistic())
        
        out_sig_is = TransferMechanism(name="sig_outs_is",
                                       default_variable=np.zeros(12),
                                       function=Logistic())
        
        out_sig_has = TransferMechanism(name="sig_outs_has",
                                        default_variable=np.zeros(9),
                                        function=Logistic())
        
        out_sig_can = TransferMechanism(name="sig_outs_can",
                                        default_variable=np.zeros(9),
                                        function=Logistic())
        
        # COMPOSITION FOR SEMANTIC NET
        
        sem_net = ParsingAutodiffComposition()
        
        sem_net.add_c_node(nouns_in)
        sem_net.add_c_node(rels_in)
        sem_net.add_c_node(h1)
        sem_net.add_c_node(h2)
        sem_net.add_c_node(out_sig_I)
        sem_net.add_c_node(out_sig_is)
        sem_net.add_c_node(out_sig_has)
        sem_net.add_c_node(out_sig_can)
        
        sem_net.add_projection(sender=nouns_in, projection=MappingProjection(sender=nouns_in, receiver=h1), receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=MappingProjection(sender=rels_in, receiver=h2), receiver=h2)
        sem_net.add_projection(sender=h1, projection=MappingProjection(sender=h1, receiver=h2), receiver=h2)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_I), receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_is), receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_has), receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=MappingProjection(sender=h2, receiver=out_sig_can), receiver=out_sig_can)
        
        # INPUTS & OUTPUTS FOR SEMANTIC NET:
        
        nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
        relations = ['is', 'has', 'can']
        is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
                   'yellow']
        has_list = ['roots', 'leaves', 'bark', 'branches', 'skin', 'feathers', 'wings', 'gills', 'scales']
        can_list = ['grow', 'move', 'swim', 'fly', 'breathe', 'breathe underwater', 'breathe air', 'walk', 'photosynthesize']
        
        nouns_input = np.identity(len(nouns))
        
        rels_input = np.identity(len(relations))
        
        truth_nouns = np.identity(len(nouns))
        
        truth_is = np.zeros((len(nouns), len(is_list)))
        
        truth_is[0, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[1, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[2, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[3, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[4, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[5, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[6, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        truth_is[7, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
        
        truth_has = np.zeros((len(nouns), len(has_list)))
        
        truth_has[0, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[1, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[2, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[3, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[4, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        truth_has[7, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        
        truth_can = np.zeros((len(nouns), len(can_list)))
        
        truth_can[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[3, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[4, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[5, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[6, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        truth_can[7, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        
        # SETTING UP DICTIONARY OF INPUTS/OUTPUTS FOR SEMANTIC NET
        
        inputs_dict = {}
        inputs_dict[nouns_in] = []
        inputs_dict[rels_in] = []
        
        targets_dict = {}
        targets_dict[out_sig_I] = []
        targets_dict[out_sig_is] = []
        targets_dict[out_sig_has] = []
        targets_dict[out_sig_can] = []

        for i in range(len(nouns)):
            for j in range(len(relations)):
                inputs_dict[nouns_in].append(nouns_input[i])
                inputs_dict[rels_in].append(rels_input[j])
                targets_dict[out_sig_I].append(truth_nouns[i])
                targets_dict[out_sig_is].append(truth_is[i])
                targets_dict[out_sig_has].append(truth_has[i])
                targets_dict[out_sig_can].append(truth_can[i])
        
        # TRAIN THE MODEL WHILE MEASURING RUNTIME
        start = timeit.default_timer()
        result = sem_net.run(inputs=inputs_dict, targets=targets_dict, epochs=eps) 
        end = timeit.default_timer()
        time = end - start
        
        msg = 'completed training semantic net as ParsingAutodiffComposition() for {0} epochs in {1} seconds'.format(eps, time)
        print(msg)
        logger.info(msg)







