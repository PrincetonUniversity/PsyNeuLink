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



# SETUP

comp_times = []
sys_times = []
speedups = []
h_size = 120
num_h_layers = [1,2,3,4,5,6,8,10,12,15] # [5, 4, 3, 2, 1] # [1, 2, 3] [3, 2, 1]
in_size = 2
out_size = 1
eps = 10
epoch_size = 10

# ITERATE OVER DIFFERENT HIDDEN LAYER CASES

for i in range(len(num_h_layers)):
    
    # SET UP INPUT, OUTPUT LAYERS
    
    rand_in = TransferMechanism(name='rand_in',
                                default_variable=np.zeros(2))
    
    rand_out = TransferMechanism(name='rand_out',
                                 default_variable=np.zeros(1),
                                 function=Logistic())
    
    # SET UP LIST FOR PROCESSES, SET UP HIDDEN LAYER SIZES
    
    rand_processes = []
    h_layer_size = int(h_size/num_h_layers[i])
    
    # SET UP COMPOSITION
    
    rand_comp = ParsingAutodiffComposition(param_init_from_pnl=True)
    rand_comp.add_c_node(rand_in)
    rand_comp.add_c_node(rand_out)
    
    print("For ", num_h_layers[i], " layers: ")
    print("\n")
    
    # ITERATE OVER EACH HIDDEN LAYER TO BE ADDED, CREATE COMPOSITION ALONG THE WAY
    
    for j in range(num_h_layers[i]):
        
        rand_hid = TransferMechanism(name='rand_hid',
                                     default_variable=np.zeros(h_layer_size),
                                     function=Logistic())
        
        rand_comp.add_c_node(rand_hid)
        
        in_to_hid = MappingProjection(name='in_to_hid',
                                      matrix=np.random.rand(in_size, h_layer_size),
                                      sender=rand_in,
                                      receiver=rand_hid)
        
        hid_to_out = MappingProjection(name='hid_to_out',
                                       matrix=np.random.rand(h_layer_size, out_size),
                                       sender=rand_hid,
                                       receiver=rand_out)
        
        rand_comp.add_projection(sender=rand_in, projection=in_to_hid, receiver=rand_hid)
        rand_comp.add_projection(sender=rand_hid, projection=hid_to_out, receiver=rand_out)
        
        rand_process = Process(pathway=[rand_in,
                                        in_to_hid,
                                        rand_hid,
                                        hid_to_out,
                                        rand_out],
                               learning=pnl.LEARNING)
        rand_processes.append(rand_process)
    
    # CREATE INPUTS, OUTPUTS
    
    np.random.seed(20)
    rand_inputs = np.random.randint(2, size=epoch_size*in_size).reshape(epoch_size, in_size)
    rand_targets = np.random.randint(2, size=epoch_size*out_size).reshape(epoch_size, out_size)
    
    # TIME COMPOSITION TRAINING, SAVE
    
    print("Starting composition training: ")
    print("\n")
    start = timeit.default_timer()
    results_comp = rand_comp.run(inputs={rand_in:rand_inputs},
                                 targets={rand_out:rand_targets},
                                 epochs=eps,
                                 learning_rate=0.1,
                                 optimizer='sgd')
    end = timeit.default_timer()
    comp_time = end - start
    comp_times.append(comp_time)
    print("Composition time: ", comp_time)
    print("\n")
    
    # CREATE SYSTEM
    
    rand_sys = System(processes=rand_processes,
                      learning_rate=0.1)
    
    # TIME SYSTEM TRAINING, SAVE
    
    print("Starting system training: ")
    print("\n")
    start = timeit.default_timer()
    results_comp = rand_sys.run(inputs={rand_in:rand_inputs},
                                targets={rand_out:rand_targets},
                                num_trials=(eps*epoch_size + 1))
    end = timeit.default_timer()
    sys_time = end - start
    sys_times.append(sys_time)
    print("System time: ", sys_time)
    print("\n")
    
    # COMPUTE SPEEDUP, SAVE
    
    speedup = np.round((sys_time/comp_time), decimals=2)
    speedups.append(speedup)
    print("Speedup: ", speedup)
    print("\n")

# RESULTS

print("\n")
print("\n")
print("Results: ")
print(sys_times)
print("\n")
print(comp_times)
print("\n")
print(speedups)
