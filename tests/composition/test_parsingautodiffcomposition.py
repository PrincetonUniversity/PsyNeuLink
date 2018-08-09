import functools
import logging
import timeit as timeit

import numpy as np
import torch
from torch import nn


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

# TEST CLASSES:

# Constructor
# Training accuracy
# Training runtime
# Training identicality (to PsyNeuLink System)
# Training runtime (comparison to PsyNeuLink System)
# Other stuff: importing weights from PNL, training then processing

# TEST MODELS (right now - may change this later)

# XOR with bigger/variable hidden layer
# Semantic net

# NOTE: make sure mechanisms & projections for composition vs for system are always different


@pytest.mark.theshire
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



@pytest.mark.brandywinebridge
class TestMiscTrainingFunctionality:
    
    def test_param_init_from_pnl(self):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        hid_map = MappingProjection(matrix=np.random.rand(2,10))
        out_map = MappingProjection(matrix=np.random.rand(10,1))
        
        xor = ParsingAutodiffComposition(param_init_from_pnl=True)
        
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
        
        results = xor.run(inputs={xor_in:xor_inputs})
                
        assert len(xor.model.params) == 2
        assert np.allclose(hid_map.matrix, xor.model.params[0].detach().numpy())
        assert np.allclose(out_map.matrix, xor.model.params[1].detach().numpy())
    
    @pytest.mark.merryboi
    def test_get_params(self):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        hid_map = MappingProjection(matrix=np.random.rand(2,10))
        out_map = MappingProjection(matrix=np.random.rand(10,1))
        
        xor = ParsingAutodiffComposition(param_init_from_pnl=True)
        
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
        
        results = xor.run(inputs={xor_in:xor_inputs})
        
        weights_get_params = xor.get_parameters()[0]
        weights_straight_1 = xor.model.params[0]
        weights_straight_2 = xor.model.params[1]
        
        assert np.allclose(hid_map.matrix, weights_get_params[hid_map])
        assert np.allclose(weights_straight_1.detach().numpy(), weights_get_params[hid_map])
        assert np.allclose(out_map.matrix, weights_get_params[out_map])
        assert np.allclose(weights_straight_2.detach().numpy(), weights_get_params[out_map])
        
        results = xor.run(inputs={xor_in:xor_inputs},
                          targets={xor_out:xor_targets},
                          epochs=10,
                          learning_rate=1)
        
        assert np.allclose(hid_map.matrix, weights_get_params[hid_map])
        assert not np.allclose(weights_straight_1.detach().numpy(), weights_get_params[hid_map])
        assert np.allclose(out_map.matrix, weights_get_params[out_map])
        assert not np.allclose(weights_straight_2.detach().numpy(), weights_get_params[out_map])
    
    def test_training_then_processing(self):
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
        
        results_before_proc = xor.run(inputs={xor_in:xor_inputs},
                                      targets={xor_out:xor_targets},
                                      epochs=10)
        
        weights_bp, biases_bp = xor.get_parameters()
        
        results_proc = xor.run(inputs={xor_in:xor_inputs})
        
        weights_ap, biases_ap = xor.get_parameters()
        
        for i in range(4):
            assert np.allclose(results_before_proc[0][i][0], results_proc[1][i][0], atol=0.001)
        
        assert np.allclose(weights_bp[hid_map], weights_ap[hid_map])
        assert np.allclose(weights_bp[out_map], weights_ap[out_map])
        assert np.allclose(biases_bp[xor_hid], biases_ap[xor_hid])
        assert np.allclose(biases_bp[xor_out], biases_ap[xor_out])
    
    @pytest.mark.buckleburyferry
    def test_params_stay_separate_using_xor(self):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        hid_m = np.random.rand(2,10)
        out_m = np.random.rand(10,1)
        
        hid_map = MappingProjection(name='hid_map',
                                    matrix=hid_m.copy(),
                                    sender=xor_in,
                                    receiver=xor_hid)
        
        out_map = MappingProjection(name='out_map',
                                    matrix=out_m.copy(),
                                    sender=xor_hid,
                                    receiver=xor_out)
        
        xor = ParsingAutodiffComposition(param_init_from_pnl=True)
        
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
        
        result = xor.run(inputs={xor_in:xor_inputs},
                         targets={xor_out:xor_targets},
                         epochs=10,
                         learning_rate=10,
                         optimizer='sgd')
        
        weights = xor.get_parameters()[0]
        
        assert np.allclose(hid_map.matrix, hid_m)
        assert np.allclose(out_map.matrix, out_m)
        assert not np.allclose(hid_map.matrix, weights[hid_map])
        assert not np.allclose(out_map.matrix, weights[out_map])



@pytest.mark.rivendell
class TestTrainingCorrectness:
    
    @pytest.mark.elrond
    @pytest.mark.parametrize(
        'eps, calls, opt, from_pnl_or_no', [
            (2000, 'single', 'adam', True),
            (6000, 'multiple', 'adam', True),
            (2000, 'single', 'adam', False),
            (6000, 'multiple', 'adam', False)
        ]
    )
    def test_xor_training_correctness(self, eps, calls, opt, from_pnl_or_no):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        hid_map = MappingProjection(matrix=np.random.rand(2,10), sender=xor_in, receiver=xor_hid)
        out_map = MappingProjection(matrix=np.random.rand(10,1))
        
        xor = ParsingAutodiffComposition(param_init_from_pnl=from_pnl_or_no)
        
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
        
        if calls == 'single':
            results = xor.run(inputs={xor_in:xor_inputs},
                              targets={xor_out:xor_targets},
                              epochs=eps,
                              optimizer=opt,
                              learning_rate=0.1)
            
            for i in range(len(results[0])):
                assert np.allclose(np.round(results[0][i][0]), xor_targets[i])
        
        else:
            results = xor.run(inputs={xor_in:xor_inputs},
                              targets={xor_out:xor_targets},
                              epochs=1,
                              optimizer=opt)
            
            for i in range(eps-1):
                results = xor.run(inputs={xor_in:xor_inputs},
                                  targets={xor_out:xor_targets},
                                  epochs=1)
            
            for i in range(len(results[eps-1])):
                assert np.allclose(np.round(results[eps-1][i][0]), xor_targets[i])
    
    @pytest.mark.parametrize(
        'eps, opt, from_pnl_or_no', [
            (1000, 'adam', True),
            (1000, 'adam', False)
        ]
    )
    def test_semantic_net_training_correctness(self, eps, opt, from_pnl_or_no):
        
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
        
        # SET UP PROJECTIONS FOR SEMANTIC NET
        
        map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                         name="map_nouns_h1",
                                         sender=nouns_in,
                                         receiver=h1)
        
        map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                        name="map_relh2",
                                        sender=rels_in,
                                        receiver=h2)
        
        map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                                      name="map_h1_h2",
                                      sender=h1,
                                      receiver=h2)
        
        map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                                     name="map_h2_I",
                                    sender=h2,
                                    receiver=out_sig_I)
        
        map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                                      name="map_h2_is",
                                      sender=h2,
                                      receiver=out_sig_is)
        
        map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_has",
                                       sender=h2,
                                       receiver=out_sig_has)
        
        map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_can",
                                       sender=h2,
                                       receiver=out_sig_can)
                
        # COMPOSITION FOR SEMANTIC NET
        sem_net = ParsingAutodiffComposition(param_init_from_pnl=from_pnl_or_no)
        
        sem_net.add_c_node(nouns_in)
        sem_net.add_c_node(rels_in)
        sem_net.add_c_node(h1)
        sem_net.add_c_node(h2)
        sem_net.add_c_node(out_sig_I)
        sem_net.add_c_node(out_sig_is)
        sem_net.add_c_node(out_sig_has)
        sem_net.add_c_node(out_sig_can)
        
        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)
        
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
        
        hello = sem_net.run(inputs=inputs_dict)
        
        result = sem_net.run(inputs=inputs_dict,
                             targets=targets_dict,
                             epochs=eps,
                             optimizer=opt)
                
        # CHECK CORRECTNESS
        
        for i in range(len(result[1])): # go over trial outputs in the single results entry
            for j in range(len(result[1][i])): # go over outputs for each output layer
                
                # get target for terminal node whose output state corresponds to current output
                correct_value = None
                curr_CIM_input_state = sem_net.output_CIM.input_states[j]
                for output_state in sem_net.output_CIM_states.keys():
                    if sem_net.output_CIM_states[output_state][0] == curr_CIM_input_state:
                        node = output_state.owner
                        correct_value = targets_dict[node][i]
                
                # compare model output for terminal node on current trial with target for terminal node on current trial
                assert np.allclose(np.round(result[1][i][j]), correct_value)



@pytest.mark.minesofmoria
class TestTrainingTime:
    
    @pytest.mark.foolofatook
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd'),
            (10, 'sgd'),
            (100, 'sgd')
        ]
    )
    def test_and_training_time(self, eps, opt):
        
        # SET UP MECHANISMS FOR COMPOSITION
        
        and_in = TransferMechanism(name='and_in',
                                   default_variable=np.zeros(2))
        
        and_out = TransferMechanism(name='and_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        # SET UP MECHANISMS FOR SYSTEM
        
        and_in_sys = TransferMechanism(name='and_in_sys',
                                       default_variable=np.zeros(2))
        
        and_out_sys = TransferMechanism(name='and_out_sys',
                                        default_variable=np.zeros(1),
                                        function=Logistic())
        
        # SET UP PROJECTIONS FOR COMPOSITION
        
        and_map = MappingProjection(name='and_map',
                                    matrix=np.random.rand(2, 1),
                                    sender=and_in,
                                    receiver=and_out)
        
        # SET UP PROJECTIONS FOR SYSTEM
        
        and_map_sys = MappingProjection(name='and_map_sys',
                                        matrix=and_map.matrix.copy(),
                                        sender=and_in_sys,
                                        receiver=and_out_sys)
        
        # SET UP COMPOSITION
        
        and_net = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        and_net.add_c_node(and_in)
        and_net.add_c_node(and_out)
        
        and_net.add_projection(sender=and_in, projection=and_map, receiver=and_out)
        
        # SET UP INPUTS AND TARGETS
        
        and_inputs = np.zeros((4,2))
        and_inputs[0] = [0, 0]
        and_inputs[1] = [0, 1]
        and_inputs[2] = [1, 0]
        and_inputs[3] = [1, 1]
        
        and_targets = np.zeros((4,1))
        and_targets[0] = [0]
        and_targets[1] = [1]
        and_targets[2] = [1]
        and_targets[3] = [0]
        
        # TIME TRAINING FOR COMPOSITION
        
        start = timeit.default_timer()
        result = and_net.run(inputs={and_in:and_inputs},
                             targets={and_out:and_targets},
                             epochs=eps,
                             learning_rate=0.1,
                             optimizer=opt) 
        end = timeit.default_timer()
        comp_time = end - start
        
        # SET UP SYSTEM
        
        and_process = Process(pathway=[and_in_sys,
                                       and_map_sys,
                                       and_out_sys],
                              learning=pnl.LEARNING)
        
        and_sys = System(processes=[and_process],
                         learning_rate=0.1)
        
        # TIME TRAINING FOR SYSTEM
        
        start = timeit.default_timer()
        results_sys = and_sys.run(inputs={and_in_sys:and_inputs}, 
                                  targets={and_out_sys:and_targets},
                                  num_trials=(eps*and_inputs.shape[0]+1))
        end = timeit.default_timer()
        sys_time = end - start
        
        # LOG TIMES, SPEEDUP PROVIDED BY COMPOSITION OVER SYSTEM
        
        msg = 'Training XOR model as ParsingAutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
        logger.info(msg)
        
        msg = 'Training XOR model as System for {0} epochs took {1} seconds'.format(eps, sys_time)
        print(msg)
        print("\n")
        logger.info(msg)
        
        speedup = np.round((sys_time/comp_time), decimals=2)
        msg = ('Training XOR model as ParsingAutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)
    
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd'),
            (10, 'sgd'),
            (100, 'sgd')
        ]
    )
    def test_xor_training_time(self, eps, opt):
        
        # SET UP MECHANISMS FOR COMPOSITION
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        # SET UP MECHANISMS FOR SYSTEM
        
        xor_in_sys = TransferMechanism(name='xor_in_sys',
                                       default_variable=np.zeros(2))
        
        xor_hid_sys = TransferMechanism(name='xor_hid_sys',
                                        default_variable=np.zeros(10),
                                        function=Logistic())
        
        xor_out_sys = TransferMechanism(name='xor_out_sys',
                                        default_variable=np.zeros(1),
                                        function=Logistic())
        
        # SET UP PROJECTIONS FOR COMPOSITION
        
        hid_map = MappingProjection(name='hid_map',
                                    matrix=np.random.rand(2,10),
                                    sender=xor_in,
                                    receiver=xor_hid)
        
        out_map = MappingProjection(name='out_map',
                                    matrix=np.random.rand(10,1),
                                    sender=xor_hid,
                                    receiver=xor_out)
        
        # SET UP PROJECTIONS FOR SYSTEM
        
        hid_map_sys = MappingProjection(name='hid_map_sys',
                                        matrix=hid_map.matrix.copy(),
                                        sender=xor_in_sys,
                                        receiver=xor_hid_sys)
        
        out_map_sys = MappingProjection(name='out_map_sys',
                                        matrix=out_map.matrix.copy(),
                                        sender=xor_hid_sys,
                                        receiver=xor_out_sys)
        
        # SET UP COMPOSITION
        
        xor = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        xor.add_c_node(xor_in)
        xor.add_c_node(xor_hid)
        xor.add_c_node(xor_out)
        
        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        
        # SET UP INPUTS AND TARGETS
        
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
        
        # TIME TRAINING FOR COMPOSITION
        
        start = timeit.default_timer()
        result = xor.run(inputs={xor_in:xor_inputs},
                         targets={xor_out:xor_targets},
                         epochs=eps,
                         learning_rate=0.1,
                         optimizer=opt) 
        end = timeit.default_timer()
        comp_time = end - start
        
        # SET UP SYSTEM
        
        xor_process = Process(pathway=[xor_in_sys,
                                       hid_map_sys,
                                       xor_hid_sys,
                                       out_map_sys,
                                       xor_out_sys],
                              learning=pnl.LEARNING)
        
        xor_sys = System(processes=[xor_process],
                         learning_rate=0.1)
        
        # TIME TRAINING FOR SYSTEM
        
        start = timeit.default_timer()
        results_sys = xor_sys.run(inputs={xor_in_sys:xor_inputs}, 
                                  targets={xor_out_sys:xor_targets},
                                  num_trials=(eps*xor_inputs.shape[0]+1))
        end = timeit.default_timer()
        sys_time = end - start
        
        # LOG TIMES, SPEEDUP PROVIDED BY COMPOSITION OVER SYSTEM
        
        msg = 'Training XOR model as ParsingAutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
        logger.info(msg)
        
        msg = 'Training XOR model as System for {0} epochs took {1} seconds'.format(eps, sys_time)
        print(msg)
        print("\n")
        logger.info(msg)
        
        speedup = np.round((sys_time/comp_time), decimals=2)
        msg = ('Training XOR model as ParsingAutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)
    
    @pytest.mark.gimli
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd'),
            (10, 'sgd'),
            (100, 'sgd')
        ]
    )
    def test_semantic_net_training_time(self, eps, opt):
        
        # SET UP MECHANISMS FOR COMPOSITION:
        
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
        
        # SET UP MECHANISMS FOR SYSTEM
        
        nouns_in_sys = TransferMechanism(name="nouns_input_sys", 
                                         default_variable=np.zeros(8))
        
        rels_in_sys = TransferMechanism(name="rels_input_sys", 
                                        default_variable=np.zeros(3))
        
        h1_sys = TransferMechanism(name="hidden_nouns_sys",
                                   default_variable=np.zeros(8),
                                   function=Logistic())
        
        h2_sys = TransferMechanism(name="hidden_mixed_sys",
                                   default_variable=np.zeros(15),
                                   function=Logistic())
        
        out_sig_I_sys = TransferMechanism(name="sig_outs_I_sys",
                                          default_variable=np.zeros(8),
                                          function=Logistic())
        
        out_sig_is_sys = TransferMechanism(name="sig_outs_is_sys",
                                           default_variable=np.zeros(12),
                                           function=Logistic())
        
        out_sig_has_sys = TransferMechanism(name="sig_outs_has_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())
        
        out_sig_can_sys = TransferMechanism(name="sig_outs_can_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())
        
        # SET UP PROJECTIONS FOR COMPOSITION
        
        map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                         name="map_nouns_h1",
                                         sender=nouns_in,
                                         receiver=h1)
        
        map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                        name="map_rel_h2",
                                        sender=rels_in,
                                        receiver=h2)
        
        map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                                      name="map_h1_h2",
                                      sender=h1,
                                      receiver=h2)
        
        map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                                     name="map_h2_I",
                                     sender=h2,
                                     receiver=out_sig_I)
        
        map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                                      name="map_h2_is",
                                      sender=h2,
                                      receiver=out_sig_is)
        
        map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_has",
                                       sender=h2,
                                       receiver=out_sig_has)
        
        map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_can",
                                       sender=h2,
                                       receiver=out_sig_can)
        
        # SET UP PROJECTIONS FOR SYSTEM
        
        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)
        
        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)
        
        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)
        
        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)
        
        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)
        
        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)
        
        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)
        
        # COMPOSITION FOR SEMANTIC NET
        
        sem_net = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        sem_net.add_c_node(nouns_in)
        sem_net.add_c_node(rels_in)
        sem_net.add_c_node(h1)
        sem_net.add_c_node(h2)
        sem_net.add_c_node(out_sig_I)
        sem_net.add_c_node(out_sig_is)
        sem_net.add_c_node(out_sig_has)
        sem_net.add_c_node(out_sig_can)
        
        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)
        
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
        
        # SETTING UP DICTIONARIES OF INPUTS/OUTPUTS FOR SEMANTIC NET
        
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
        
        inputs_dict_sys = {}
        inputs_dict_sys[nouns_in_sys] = inputs_dict[nouns_in]
        inputs_dict_sys[rels_in_sys] = inputs_dict[rels_in]
        
        targets_dict_sys = {}
        targets_dict_sys[out_sig_I_sys] = targets_dict[out_sig_I]
        targets_dict_sys[out_sig_is_sys] = targets_dict[out_sig_is]
        targets_dict_sys[out_sig_has_sys] = targets_dict[out_sig_has]
        targets_dict_sys[out_sig_can_sys] = targets_dict[out_sig_can]        
        
        # TIME TRAINING FOR COMPOSITION
        
        start = timeit.default_timer()
        result = sem_net.run(inputs=inputs_dict,
                             targets=targets_dict,
                             epochs=eps,
                             learning_rate=0.1,
                             optimizer=opt) 
        end = timeit.default_timer()
        comp_time = end - start
        
        # SET UP SYSTEM
        
        p11 = Process(pathway=[nouns_in_sys,
                               map_nouns_h1_sys,
                               h1_sys,
                               map_h1_h2_sys,
                               h2_sys,
                               map_h2_I_sys,
                               out_sig_I_sys],
                      learning=pnl.LEARNING)

        p12 = Process(pathway=[rels_in_sys,
                               map_rels_h2_sys,
                               h2_sys,
                               map_h2_is_sys,
                               out_sig_is_sys],
                      learning=pnl.LEARNING)
        
        p21 = Process(pathway=[h2_sys,
                               map_h2_has_sys,
                               out_sig_has_sys],
                      learning=pnl.LEARNING)
        
        p22 = Process(pathway=[h2_sys, 
                               map_h2_can_sys,
                               out_sig_can_sys],
                      learning=pnl.LEARNING)
        
        sem_net_sys = System(processes=[p11,
                                        p12,
                                        p21,
                                        p22,
                                        ],
                             learning_rate=0.1)
        
        # TIME TRAINING FOR SYSTEM
        
        start = timeit.default_timer()
        results = sem_net_sys.run(inputs=inputs_dict_sys, 
                                  targets=targets_dict_sys,
                                  num_trials=(len(inputs_dict[nouns_in])*eps + 1))
        end = timeit.default_timer()
        sys_time = end - start
        
        # LOG TIMES, SPEEDUP PROVIDED BY COMPOSITION OVER SYSTEM
        
        msg = 'Training Semantic net as ParsingAutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
        logger.info(msg)
        
        msg = 'Training Semantic net as System for {0} epochs took {1} seconds'.format(eps, sys_time)
        print(msg)
        print("\n")
        logger.info(msg)
        
        speedup = np.round((sys_time/comp_time), decimals=2)
        msg = ('Training Semantic net as ParsingAutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)



@pytest.mark.lothlorien
class TestTrainingIdenticalness():
    
    @pytest.mark.xor_ident
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd') # ,
            # (10, 'sgd'),
        ]
    )
    def test_xor_training_identicalness(self, eps, opt):
        
        # SET UP MECHANISMS FOR COMPOSITION
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        # SET UP MECHANISMS FOR SYSTEM
        
        xor_in_sys = TransferMechanism(name='xor_in_sys',
                                       default_variable=np.zeros(2))
        
        xor_hid_sys = TransferMechanism(name='xor_hid_sys',
                                        default_variable=np.zeros(10),
                                        function=Logistic())
        
        xor_out_sys = TransferMechanism(name='xor_out_sys',
                                        default_variable=np.zeros(1),
                                        function=Logistic())
        
        # SET UP PROJECTIONS FOR COMPOSITION
        
        hid_map = MappingProjection(name='hid_map',
                                    matrix=np.random.rand(2,10),
                                    sender=xor_in,
                                    receiver=xor_hid)
        
        out_map = MappingProjection(name='out_map',
                                    matrix=np.random.rand(10,1),
                                    sender=xor_hid,
                                    receiver=xor_out)
        
        # SET UP PROJECTIONS FOR SYSTEM
        
        hid_map_sys = MappingProjection(name='hid_map_sys',
                                    matrix=hid_map.matrix.copy(),
                                    sender=xor_in_sys,
                                    receiver=xor_hid_sys)
        
        out_map_sys = MappingProjection(name='out_map_sys',
                                    matrix=out_map.matrix.copy(),
                                    sender=xor_hid_sys,
                                    receiver=xor_out_sys)
        
        # SET UP COMPOSITION
        
        xor = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        xor.add_c_node(xor_in)
        xor.add_c_node(xor_hid)
        xor.add_c_node(xor_out)
        
        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        
        # SET UP INPUTS AND TARGETS
        
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
        
        # TRAIN COMPOSITION
        
        result = xor.run(inputs={xor_in:xor_inputs},
                         targets={xor_out:xor_targets},
                         epochs=eps,
                         learning_rate=10,
                         optimizer=opt)
        
        comp_weights = xor.get_parameters()[0]
        
        # SET UP SYSTEM
        
        xor_process = Process(pathway=[xor_in_sys,
                                       hid_map_sys,
                                       xor_hid_sys,
                                       out_map_sys,
                                       xor_out_sys],
                              learning=pnl.LEARNING)
        
        xor_sys = System(processes=[xor_process],
                         learning_rate=10)
        
        # TRAIN SYSTEM
        
        results_sys = xor_sys.run(inputs={xor_in_sys:xor_inputs}, 
                                  targets={xor_out_sys:xor_targets},
                                  num_trials=(eps*xor_inputs.shape[0]+1))
        
        # CHECK THAT PARAMETERS FOR COMPOSITION, SYSTEM ARE SAME
        
        print("composition weights after running composition and system: ")
        print("\n")
        print(xor.model.params[0])
        print(xor.model.params[1])
        print("\n")
        print("system weights after running composition and system: ")
        print("\n")
        print(hid_map_sys.matrix)
        print(out_map_sys.matrix)
        
        assert np.allclose(comp_weights[hid_map], hid_map_sys.matrix)
        assert np.allclose(comp_weights[out_map], out_map_sys.matrix)
    
    @pytest.mark.sem_net_ident
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd') # ,
            # (10, 'sgd'),
        ]
    )
    def test_semantic_net_training_identicalness(self, eps, opt):
        
        # SET UP MECHANISMS FOR SEMANTIC NET:
        
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
        
        # SET UP MECHANISMS FOR SYSTEM
        
        nouns_in_sys = TransferMechanism(name="nouns_input_sys", 
                                         default_variable=np.zeros(8))
        
        rels_in_sys = TransferMechanism(name="rels_input_sys", 
                                        default_variable=np.zeros(3))
        
        h1_sys = TransferMechanism(name="hidden_nouns_sys",
                                   default_variable=np.zeros(8),
                                   function=Logistic())
        
        h2_sys = TransferMechanism(name="hidden_mixed_sys",
                                   default_variable=np.zeros(15),
                                   function=Logistic())
        
        out_sig_I_sys = TransferMechanism(name="sig_outs_I_sys",
                                          default_variable=np.zeros(8),
                                          function=Logistic())
        
        out_sig_is_sys = TransferMechanism(name="sig_outs_is_sys",
                                           default_variable=np.zeros(12),
                                           function=Logistic())
        
        out_sig_has_sys = TransferMechanism(name="sig_outs_has_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())
        
        out_sig_can_sys = TransferMechanism(name="sig_outs_can_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())
        
        # SET UP PROJECTIONS FOR SEMANTIC NET
        
        map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                 name="map_nouns_h1",
                                 sender=nouns_in,
                                 receiver=h1)
        
        map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                    name="map_relh2",
                                    sender=rels_in,
                                    receiver=h2)
        
        map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                                    name="map_h1_h2",
                                    sender=h1,
                                    receiver=h2)
        
        map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                                    name="map_h2_I",
                                    sender=h2,
                                    receiver=out_sig_I)
        
        map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                                    name="map_h2_is",
                                    sender=h2,
                                    receiver=out_sig_is)
        
        map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                                    name="map_h2_has",
                                    sender=h2,
                                    receiver=out_sig_has)
        
        map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                                    name="map_h2_can",
                                    sender=h2,
                                    receiver=out_sig_can)
        
        # SET UP PROJECTIONS FOR SYSTEM
        
        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)
        
        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)
        
        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)
        
        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)
        
        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)
        
        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)
        
        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)
        
        # SET UP COMPOSITION FOR SEMANTIC NET
        
        sem_net = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        sem_net.add_c_node(nouns_in)
        sem_net.add_c_node(rels_in)
        sem_net.add_c_node(h1)
        sem_net.add_c_node(h2)
        sem_net.add_c_node(out_sig_I)
        sem_net.add_c_node(out_sig_is)
        sem_net.add_c_node(out_sig_has)
        sem_net.add_c_node(out_sig_can)
        
        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)
        
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
        
        inputs_dict_sys = {}
        inputs_dict_sys[nouns_in_sys] = inputs_dict[nouns_in]
        inputs_dict_sys[rels_in_sys] = inputs_dict[rels_in]
        
        targets_dict_sys = {}
        targets_dict_sys[out_sig_I_sys] = targets_dict[out_sig_I]
        targets_dict_sys[out_sig_is_sys] = targets_dict[out_sig_is]
        targets_dict_sys[out_sig_has_sys] = targets_dict[out_sig_has]
        targets_dict_sys[out_sig_can_sys] = targets_dict[out_sig_can]        
        
        # TRAIN COMPOSITION
        
        result = sem_net.run(inputs=inputs_dict,
                             targets=targets_dict,
                             epochs=eps,
                             learning_rate=10,
                             optimizer=opt) 
        
        comp_weights = sem_net.get_parameters()[0]
        
        # SET UP SYSTEM
        
        p11 = Process(pathway=[nouns_in_sys,
                               map_nouns_h1_sys,
                               h1_sys,
                               map_h1_h2_sys,
                               h2_sys,
                               map_h2_I_sys,
                               out_sig_I_sys],
                      learning=pnl.LEARNING)

        p12 = Process(pathway=[rels_in_sys,
                               map_rels_h2_sys,
                               h2_sys,
                               map_h2_is_sys,
                               out_sig_is_sys],
                      learning=pnl.LEARNING)
        
        p21 = Process(pathway=[h2_sys,
                               map_h2_has_sys,
                               out_sig_has_sys],
                      learning=pnl.LEARNING)
        
        p22 = Process(pathway=[h2_sys, 
                               map_h2_can_sys,
                               out_sig_can_sys],
                      learning=pnl.LEARNING)
        
        sem_net_sys = System(processes=[p11,
                                        p12,
                                        p21,
                                        p22,
                                        ],
                             learning_rate=10)
        
        # TRAIN SYSTEM
        
        results = sem_net_sys.run(inputs=inputs_dict_sys, 
                                  targets=targets_dict_sys,
                                  num_trials=(len(inputs_dict_sys[nouns_in_sys])*eps + 1))
        
        # CHECK THAT PARAMETERS FOR COMPOSITION, SYSTEM ARE SAME
        
        assert np.allclose(comp_weights[map_nouns_h1], map_nouns_h1_sys.matrix)
        assert np.allclose(comp_weights[map_rels_h2], map_rels_h2_sys.matrix)
        assert np.allclose(comp_weights[map_h1_h2], map_h1_h2_sys.matrix)
        assert np.allclose(comp_weights[map_h2_I], map_h2_I_sys.matrix)
        assert np.allclose(comp_weights[map_h2_is], map_h2_is_sys.matrix)
        assert np.allclose(comp_weights[map_h2_has], map_h2_has_sys.matrix)
        assert np.allclose(comp_weights[map_h2_can], map_h2_can_sys.matrix)















'''

class TestRunTrainingSpeedCorrectness:
    
    @pytest.mark.parametrize(
        'eps', [
            1,
            10,
            100 # ,
            # 1000 # ,
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
    
    @pytest.mark.mordor
    @pytest.mark.gandalfthegrey
    @pytest.mark.parametrize(
        'eps', [
            10,
            100# ,
            # 1000
        ]
    )
    def test_xor_as_system_vs_autodiff_composition(self, eps):
        
        # SET UP SYSTEM
        
        xor_in_sys = TransferMechanism(name='xor_in',
                                       default_variable=np.zeros(2))
        
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
        
        xor_inputs_sys = np.zeros((4,2))
        xor_inputs_sys[0] = [0, 0]
        xor_inputs_sys[1] = [0, 1]
        xor_inputs_sys[2] = [1, 0]
        xor_inputs_sys[3] = [1, 1]
        
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
        results = xor_system.run(inputs={xor_in_sys:xor_inputs_sys}, 
                                         targets={xor_out_sys:xor_targets_sys},
                                         num_trials=(4*eps))
        end = timeit.default_timer()
        sys_time = end - start
        
        '''
        
'''
        start = timeit.default_timer()
        for i in range(eps):
            results = xor_system.run(inputs={xor_in_sys:xor_inputs_sys},
                                     targets={xor_out_sys:xor_targets_sys})
        end = timeit.default_timer()
        sys_time = end - start
        '''
        
'''
        msg = 'completed training xor model as System for {0} epochs in {1} seconds'.format(eps, sys_time)
        print(msg)
        logger.info(msg)
        
        # TRAIN THE AUTODIFF COMPOSITION, TIME IT
        
        start = timeit.default_timer()
        result = xor_comp.run(inputs={xor_in_comp:xor_inputs_comp}, 
                              targets={xor_out_comp:xor_targets_comp}, epochs=eps, optimizer='sgd') 
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



class TestParametersFromPNL:
    
    @pytest.mark.frodobaggins
    def test_xor_training_correctness_from_pnl(self):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        hid_map = MappingProjection(matrix=np.random.rand(2,10))
        out_map = MappingProjection(matrix=np.random.rand(10,1))
        
        xor = ParsingAutodiffComposition(param_init_from_pnl=True)
        
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
        
        results = xor.run(inputs={xor_in:xor_inputs}, targets={xor_out:xor_targets}, epochs=10000, optimizer='adam')
        
        for i in range(len(results[0])):
            assert np.allclose(np.round(results[0][i][0]), xor_targets[i])
    
    @pytest.mark.legolasandgimli
    @pytest.mark.parametrize(
        'h_size', [
            1,
            3
        ]
    )
    def test_replicating_system_training_for_xor(self, h_size):
        
        # SET UP MECHANISMS AND PROJECTIONS
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(h_size),
                                    function=Logistic())
        
        hid_map = MappingProjection(matrix=np.ones((2,10))*0.01)
        out_map = MappingProjection(matrix=np.ones((10,h_size))*0.01)
        
        # SET UP SYSTEM
        
        
        
        # SET UP COMPOSITION
        
        xor_comp = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        xor_comp.add_c_node(xor_in)
        xor_comp.add_c_node(xor_hid)
        xor_comp.add_c_node(xor_out)
        
        xor_comp.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor_comp.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        
        # SET UP INPUTS AND OUTPUTS
        
        xor_inputs = np.zeros((4,2))
        xor_inputs[0] = [0, 0]
        xor_inputs[1] = [0, 1]
        xor_inputs[2] = [1, 0]
        xor_inputs[3] = [1, 1]
        
        xor_targets = np.zeros((4,h_size))
        if h_size == 1:
            xor_targets[0] = [0]
            xor_targets[1] = [1]
            xor_targets[2] = [1]
            xor_targets[3] = [0]
        elif h_size == 3:
            xor_targets[0] = [0, 1, 0]
            xor_targets[1] = [1, 0, 0]
            xor_targets[2] = [1, 0, 0]
            xor_targets[3] = [0, 1, 0]
        else:
            xor_targets[0] = [0, 1]
            xor_targets[1] = [1, 0]
            xor_targets[2] = [1, 0]
            xor_targets[3] = [0, 1]
        
        print("\n")
        
        # TRAIN THE SYSTEM AND COMPOSITION, CHECK PARAMETERS
        
        results_comp = xor_comp.run(inputs={xor_in:xor_inputs})
        
        # weights, biases = xor_comp.get_parameters()
        print("weights of system before training: ")
        print(hid_map.matrix)
        print(out_map.matrix)
        print("\n")
        print("weights of composition before training: ")
        print(xor_comp.model.params[0])
        print(xor_comp.model.params[1])
        # print(weights[hid_map])
        # print(weights[out_map])
        print("\n")
        # assert len(biases) == 0
        # assert np.allclose(hid_map.matrix, weights[hid_map])
        # assert np.allclose(out_map.matrix, weights[out_map])
        
        results_comp = xor_comp.run(inputs={xor_in:xor_inputs},
                                    targets={xor_out:xor_targets}, epochs=1, learning_rate=10, optimizer='sgd')
        
        # weights, biases = xor_comp.get_parameters()
        print("weights of system after composition training: ")
        print(hid_map.matrix)
        print(out_map.matrix)
        print("\n")
        print("weights of composition after composition training: ")
        print(xor_comp.model.params[0])
        print(xor_comp.model.params[1])
        # print(weights[hid_map])
        # print(weights[out_map])
        print("\n")
        
        xor_process = Process(pathway=[xor_in,
                                       hid_map,
                                       xor_hid,
                                       out_map,
                                       xor_out],
                              learning=pnl.LEARNING)
        
        xor_system = System(processes=[xor_process],
                            learning_rate=10)
        
        results_sys = xor_system.run(inputs={xor_in:xor_inputs}, 
                                     targets={xor_out:xor_targets},
                                     num_trials=5)
        
        # weights, biases = xor_comp.get_parameters()
        print("weights of system after both training: ")
        print(hid_map.matrix)
        print(out_map.matrix)
        print("\n")
        print("weights of composition after both training: ")
        print(xor_comp.model.params[0])
        print(xor_comp.model.params[1])
        # print(weights[hid_map])
        # print(weights[out_map])
        print("\n")
        # assert len(biases) == 0
        # assert np.allclose(hid_map.matrix, weights[hid_map])
        # assert np.allclose(out_map.matrix, weights[out_map])
        # print(type(hid_map.matrix[0][0]))
        # print(type(weights[hid_map][0][0]))
    
    @pytest.mark.meriadocbrandybuck
    def test_system_pytorch_autodiff_comp_comparison(self):
        
        # SET UP PYTORCH XOR CLASS, MODEL OBJECT OF THIS CLASS
        
        class PT_xor(torch.nn.Module):
            
            def __init__(self):
                super(PT_xor, self).__init__()
                self.w1 = nn.Parameter(torch.ones(2,10).float())
                self.w2 = nn.Parameter(torch.ones(10,1).float())
            
            def forward(self, x):
                q = nn.Sigmoid()
                # print(x)
                # print("\n")
                x = torch.matmul(x, self.w1)
                x = q(x)
                # print(x)
                # print("\n")
                x = torch.matmul(x, self.w2)
                x = q(x)
                # print(x)
                # print("\n")
                return x
            
            def return_params(self):
                param_list = []
                param_list.append(self.w1.detach().numpy().copy())
                param_list.append(self.w2.detach().numpy().copy())
                return param_list
        
        xor_basicpt = PT_xor()
        
        # SET UP MECHANISMS AND PROJECTIONS FOR XOR SYSTEM & COMPOSITION
        
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))
        
        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())
        
        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())
        
        hid_map = MappingProjection(matrix=np.ones((2,10)))
        out_map = MappingProjection(matrix=np.ones((10,1)))
        
        # SET UP SYSTEM
        
        xor_process = Process(pathway=[xor_in,
                                       hid_map,
                                       xor_hid,
                                       out_map,
                                       xor_out],
                              learning=pnl.LEARNING)
        
        xor_system = System(processes=[xor_process],
                            learning_rate=10)
        
        # SET UP COMPOSITION
        
        xor_comp = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        xor_comp.add_c_node(xor_in)
        xor_comp.add_c_node(xor_hid)
        xor_comp.add_c_node(xor_out)
        
        xor_comp.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor_comp.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        
        # SET UP INPUTS AND OUTPUTS
        
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
        
        print("\n")
        
        # TRAIN THE SYSTEM AND COMPOSITION, CHECK PARAMETERS
        
        results_comp = xor_comp.run(inputs={xor_in:xor_inputs[0]})
        
        print("composition opt params after run on only inputs: ")
        print("\n")
        print(xor_comp.optimizer.param_groups)
        print("\n")
        
        weights, biases = xor_comp.get_parameters()
        
        print("composition opt params after get params: ")
        print("\n")
        print(xor_comp.optimizer.param_groups)
        print("\n")
        
        print("weights of system before training: ")
        print(hid_map.matrix)
        print(out_map.matrix)
        print("\n")
        print("weights of composition before training: ")
        print(weights[hid_map])
        print(weights[out_map])
        print("\n")
        print("weights of basic pytorch before training: ")
        print(xor_basicpt.w1)
        print(xor_basicpt.w2)
        print("\n")
        assert len(biases) == 0
        assert np.allclose(hid_map.matrix, weights[hid_map])
        assert np.allclose(out_map.matrix, weights[out_map])
        
        print("starting composition training: ")
        print("\n")
        
        results_comp = xor_comp.run(inputs={xor_in:xor_inputs[0]},
                                    targets={xor_out:xor_targets[0]}, epochs=1, learning_rate=10 , optimizer='sgd')
        
        print("starting basic pytorch training: ")
        print("\n")
        
        loss = nn.MSELoss(size_average=True)
        optimizer = torch.optim.SGD(xor_basicpt.parameters(), lr=10)
        '''
        
'''
        for i in range(len(xor_inputs[i])):
            inp = torch.from_numpy(xor_inputs[i].copy()).float()
            targ = torch.from_numpy(xor_targets[i].copy()).float()
            output = xor_basicpt.forward(inp)
            l = loss(output, targ)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        '''
        
'''
        for i in range(1):
            inp = torch.from_numpy(xor_inputs[0].copy()).float()
            targ = torch.from_numpy(xor_targets[0].copy()).float()
            output = xor_basicpt.forward(inp)
            l = loss(output, targ)
            optimizer.zero_grad()
            print(l)
            print("\n")
            l.backward()
            print("gradients for basic pytorch: ")
            print("\n")
            print(xor_basicpt.w1.grad)
            print("\n")
            print(xor_basicpt.w2.grad)
            print("\n")
            optimizer.step()
        
        weights, biases = xor_comp.get_parameters()
        print("weights of system after composition, basic pytorch training: ")
        print(hid_map.matrix)
        print(out_map.matrix)
        print("\n")
        print("weights of composition after composition, basic pytorch training: ")
        print(weights[hid_map])
        print(weights[out_map])
        print("\n")
        print("weights of basic pytorch after composition, basic pytorch training: ")
        print(xor_basicpt.w1)
        print(xor_basicpt.w2)
        print("\n")
        # print("PIGFARTS")
        # print(xor_inputs[0:2])
        for i in range(1):
            results_sys = xor_system.run(inputs={xor_in:xor_inputs[0:2]}, 
                                         targets={xor_out:xor_targets[0:2]})
        
        weights, biases = xor_comp.get_parameters()
        print("weights of system after both training: ")
        print(hid_map.matrix)
        print(out_map.matrix)
        print("\n")
        print("weights of composition after both training: ")
        print(weights[hid_map])
        print(weights[out_map])
        print("\n")
        print("weights of basic pytorch after composition, basic pytorch training: ")
        print(xor_basicpt.w1)
        print(xor_basicpt.w2)
        print("\n")
        assert len(biases) == 0
        assert np.allclose(hid_map.matrix, weights[hid_map])
        assert np.allclose(out_map.matrix, weights[out_map])



class TestSemanticNetTraining:
    
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
    
    @pytest.mark.ridersofrohan
    @pytest.mark.parametrize(
        'eps', [
            1 # ,
            # 10 # ,
            # 40
            # 100 # ,
            # 1000 # ,
            # 10000
        ]
    )
    def test_sem_net_as_composition_vs_as_system(self, eps):
        
        # MECHANISMS AND PROJECTIONS FOR SEMANTIC NET:
        
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
        
        map_nouns_h1 = pnl.MappingProjection(matrix=np.random.rand(8,8),
                                     name="map_nouns_h1",
                                     sender=nouns_in,
                                     receiver=h1)
        
        map_rels_h2 = pnl.MappingProjection(matrix=np.random.rand(3,15),
                                        name="map_relh2",
                                        sender=rels_in,
                                        receiver=h2)
        
        map_h1_h2 = pnl.MappingProjection(matrix=np.random.rand(8,15),
                                        name="map_h1_h2",
                                        sender=h1,
                                        receiver=h2)
        
        map_h2_I = pnl.MappingProjection(matrix=np.random.rand(15,8),
                                        name="map_h2_I",
                                        sender=h2,
                                        receiver=out_sig_I)
        
        map_h2_is = pnl.MappingProjection(matrix=np.random.rand(15,12),
                                        name="map_h2_is",
                                        sender=h2,
                                        receiver=out_sig_is)
        
        map_h2_has = pnl.MappingProjection(matrix=np.random.rand(15,9),
                                        name="map_h2_has",
                                        sender=h2,
                                        receiver=out_sig_has)
        
        map_h2_can = pnl.MappingProjection(matrix=np.random.rand(15,9),
                                        name="map_h2_can",
                                        sender=h2,
                                        receiver=out_sig_can)
        
        # COMPOSITION FOR SEMANTIC NET
        
        sem_net = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        sem_net.add_c_node(nouns_in)
        sem_net.add_c_node(rels_in)
        sem_net.add_c_node(h1)
        sem_net.add_c_node(h2)
        sem_net.add_c_node(out_sig_I)
        sem_net.add_c_node(out_sig_is)
        sem_net.add_c_node(out_sig_has)
        sem_net.add_c_node(out_sig_can)
        
        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)
        
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
        
        # TRAIN THE AUTODIFF COMPOSITION, TIME IT
        
        start = timeit.default_timer()
        result = sem_net.run(inputs=inputs_dict,
                             targets=targets_dict,
                             epochs=eps,
                             learning_rate=10,
                             optimizer='sgd')
        end = timeit.default_timer()
        comp_time = end - start
        
        msg = 'completed training semantic net as ParsingAutodiffComposition for {0} epochs in {1} seconds'.format(eps, comp_time)
        print(msg)
        logger.info(msg)
        
        print("\n")
        print(sem_net.model.params[6])
        print("\n")
        
        # FINISH CREATING THE SYSTEM (PROJECTIONS AND SYSTEM INIT)
        
        p11 = pnl.Process(pathway=[nouns_in,
                                   map_nouns_h1,
                                   h1,
                                   map_h1_h2,
                                   h2,
                                   map_h2_I,
                                   out_sig_I],
                          learning=pnl.LEARNING)

        p12 = pnl.Process(pathway=[rels_in,
                                   map_rels_h2,
                                   h2,
                                   map_h2_is,
                                   out_sig_is],
                          learning=pnl.LEARNING)
        
        p21 = pnl.Process(pathway=[h2,
                                   map_h2_has,
                                   out_sig_has],
                          learning=pnl.LEARNING)
        
        p22 = pnl.Process(pathway=[h2, 
                                   map_h2_can,
                                   out_sig_can],
                          learning=pnl.LEARNING)
        
        sem_net_sys = pnl.System(processes=[p11,
                                            p12,
                                            p21,
                                            p22,
                                            ],
                                 learning_rate=10)
        
        sem_net_sys.show_graph()
        
        # TRAIN THE SYSTEM, TIME IT
        
        start = timeit.default_timer()
        results = sem_net_sys.run(inputs=inputs_dict, 
                                  targets=targets_dict,
                                  num_trials=(len(inputs_dict[nouns_in])*eps + 1))
        # print("\n")
        # sem_net_sys._report_system_completion()
        end = timeit.default_timer()
        sys_time = end - start
        
        msg = 'completed training semantic net as System for {0} epochs in {1} seconds'.format(eps, sys_time)
        print(msg)
        logger.info(msg)
        
        # REPORT SPEEDUP
        
        speedup = np.round((sys_time/comp_time), decimals=2)
        msg = ('training semantic net as ParsingAutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)
        
        print("\n")
        print(map_h2_can.matrix)
    '''
    
'''
    @pytest.mark.parametrize(
        'eps', [
            1 # ,
            # 10,
            # 40 # ,
            # 100 # ,
            # 1000 # ,
            # 10000
        ]
    )
    '''
    
'''
    @pytest.mark.hellobuddy
    def test_sem_net_as_composition_vs_as_system_one_output(self): # , eps):
        
        # MECHANISMS AND PROJECTIONS FOR SEMANTIC NET:
        
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
        
        out = TransferMechanism(name="out",
                                default_variable=np.zeros(38),
                                function=Logistic())
        '''
        
'''
        map_nouns_h1 = pnl.MappingProjection(matrix=np.random.rand(8,8)*0.05,
                                     name="map_nouns_h1",
                                     sender=nouns_in,
                                     receiver=h1)
        
        map_rels_h2 = pnl.MappingProjection(matrix=np.random.rand(3,15)*0.05,
                                        name="map_relh2",
                                        sender=rels_in,
                                        receiver=h2)
        
        map_h1_h2 = pnl.MappingProjection(matrix=np.random.rand(8,15)*0.05,
                                        name="map_h1_h2",
                                        sender=h1,
                                        receiver=h2)
        
        map_h2_out = pnl.MappingProjection(matrix=np.random.rand(15,38)*0.05,
                                           name="map_h2_I",
                                           sender=h2,
                                           receiver=out)
        '''
        
'''
        map_nouns_h1 = pnl.MappingProjection(matrix=np.ones((8,8))*0.01,
                                     name="map_nouns_h1",
                                     sender=nouns_in,
                                     receiver=h1)
        
        map_rels_h2 = pnl.MappingProjection(matrix=np.ones((3,15))*0.01,
                                        name="map_relh2",
                                        sender=rels_in,
                                        receiver=h2)
        
        map_h1_h2 = pnl.MappingProjection(matrix=np.ones((8,15))*0.01,
                                        name="map_h1_h2",
                                        sender=h1,
                                        receiver=h2)
        
        map_h2_out = pnl.MappingProjection(matrix=np.ones((15,38))*0.01,
                                           name="map_h2_I",
                                           sender=h2,
                                           receiver=out)
        
        # COMPOSITION FOR SEMANTIC NET
        
        sem_net = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        sem_net.add_c_node(nouns_in)
        sem_net.add_c_node(rels_in)
        sem_net.add_c_node(h1)
        sem_net.add_c_node(h2)
        sem_net.add_c_node(out)
        
        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_out, receiver=out)
        
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
        targets_dict[out] = []
        
        for i in range(8):
            for j in range(3):
                inputs_dict[nouns_in].append(nouns_input[i])
                inputs_dict[rels_in].append(rels_input[j])
                targ = np.concatenate([truth_nouns[i], truth_is[i], truth_has[i], truth_can[i]])
                targets_dict[out].append(targ)
        
        # TRAIN THE AUTODIFF COMPOSITION, TIME IT
        
        start = timeit.default_timer()
        result = sem_net.run(inputs=inputs_dict,
                             targets=targets_dict,
                             epochs=1,
                             learning_rate=10,
                             optimizer='sgd')
        end = timeit.default_timer()
        comp_time = end - start
        
        print("\n")
        msg = 'completed training semantic net as ParsingAutodiffComposition for {0} epochs in {1} seconds'.format(1, comp_time)
        print(msg)
        logger.info(msg)
        
        print("\n")
        print("The weights in the composition after composition training: ")
        print("\n")
        print(sem_net.model.params[3])
        print("\n")
        
        # FINISH CREATING THE SYSTEM (PROJECTIONS AND SYSTEM INIT)
        '''
        
'''
        p11 = pnl.Process(pathway=[nouns_in,
                                   map_nouns_h1,
                                   h1,
                                   map_h1_h2,
                                   h2],
                          learning=pnl.LEARNING)

        p12 = pnl.Process(pathway=[rels_in,
                                    map_rels_h2,
                                    h2],
                          learning=pnl.LEARNING)
        
        p21 = pnl.Process(pathway=[h2, 
                                   map_h2_out,
                                   out],
                          learning=pnl.LEARNING)
        
        sem_net_sys = pnl.System(processes=[p11,
                                            p12,
                                            p21,
                                            ],
                                 learning_rate=10)
        '''
        
'''
        p11 = pnl.Process(pathway=[nouns_in,
                                   map_nouns_h1,
                                   h1,
                                   map_h1_h2,
                                   h2,
                                   map_h2_out,
                                   out],
                          learning=pnl.LEARNING)

        p12 = pnl.Process(pathway=[rels_in,
                                    map_rels_h2,
                                    h2,
                                    map_h2_out,
                                    out],
                          learning=pnl.LEARNING)
        
        sem_net_sys = pnl.System(processes=[p11,
                                            p12,
                                            ],
                                 learning_rate=10)
        
        sem_net_sys.show_graph()
        
        print("\n")
        print("The weights in the system after composition training: ")
        print("\n")
        print(map_h2_out.matrix)
        print("\n")
        
        # TRAIN THE SYSTEM, TIME IT
        
        start = timeit.default_timer()
        results = sem_net_sys.run(inputs=inputs_dict, 
                                  targets=targets_dict,
                                  num_trials=25)
                                  # num_trials=(len(inputs_dict[nouns_in])*1 + 1))
        # print("\n")
        # sem_net_sys._report_system_completion()
        end = timeit.default_timer()
        sys_time = end - start
        
        msg = 'completed training semantic net as System for {0} epochs in {1} seconds'.format(1, sys_time)
        print(msg)
        logger.info(msg)
        
        # REPORT SPEEDUP
        
        speedup = np.round((sys_time/comp_time), decimals=2)
        msg = ('training semantic net as ParsingAutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(1, speedup))
        print(msg)
        logger.info(msg)
        
        print("the weights in the system after system training: ")
        print("\n")
        print(map_h2_out.matrix)
        
        print("\n")
        pt_weight = sem_net.model.params[3].detach().numpy()
        print(pt_weight - map_h2_out.matrix)
        print("\n")
        print(type(pt_weight[0][0]))
        print(type(map_h2_out.matrix[0][0]))
        print("\n")
        print("\n")
        assert(np.allclose(pt_weight, map_h2_out.matrix))

'''







