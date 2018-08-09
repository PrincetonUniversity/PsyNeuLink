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

comp_times = np.zeros((5,10))
sys_times = np.zeros((5,10))
speedups = np.zeros((5,10))
in_size = 2
out_size = 1
eps = 100
epoch_size = 10


# ITERATE OVER DIFFERENT CASES, LAYER SIZES

for t in range(5):
    
    for i in range(10):
        
        h_size = 100*(1+i)
        print(h_size)
        print("\n")
        
        # SET UP MECHANISMS
        
        rand_in = TransferMechanism(name='rand_in',
                                    default_variable=np.zeros(in_size))
        # print(np.shape(rand_in.variable))
        
        rand_hid = TransferMechanism(name='rand_hid',
                                     default_variable=np.zeros(h_size))
        # print(np.shape(rand_hid.variable))
        
        rand_out = TransferMechanism(name='rand_out',
                                     default_variable=np.zeros(out_size),
                                     function=Logistic())
        print(np.shape(rand_out.variable))
        # print("\n")
        
        # SET UP PROJECTIONS
        
        hid_map = MappingProjection(name='hid_map',
                                    matrix=np.random.rand(in_size, h_size),
                                    sender=rand_in,
                                    receiver=rand_hid)
        # print(np.shape(hid_map.matrix))
        
        out_map = MappingProjection(name='out_map',
                                    matrix=np.random.rand(h_size, out_size),
                                    sender=rand_hid,
                                    receiver=rand_out)
        # print(np.shape(out_map.matrix))
        
        # SET UP COMPOSITION
        
        rand_comp = ParsingAutodiffComposition(param_init_from_pnl=True)
        
        rand_comp.add_c_node(rand_in)
        rand_comp.add_c_node(rand_hid)
        rand_comp.add_c_node(rand_out)
        
        rand_comp.add_projection(sender=rand_in, projection=hid_map, receiver=rand_hid)
        rand_comp.add_projection(sender=rand_hid, projection=out_map, receiver=rand_out)
        
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
        comp_times[t,i] = comp_time
        print("Composition time: ", comp_time)
        print("\n")
        
        # CREATE SYSTEM
        
        rand_process = Process(pathway=[rand_in,
                                        hid_map,
                                        rand_hid,
                                        out_map,
                                        rand_out],
                               learning=pnl.LEARNING)
        
        rand_sys = System(processes=rand_process,
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
        sys_times[t,i] = sys_time
        print("System time: ", sys_time)
        print("\n")
        
        # COMPUTE SPEEDUP, SAVE
        
        speedup = sys_time/comp_time
        speedups[t,i] = speedup
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
print("\n")

avg_speedups = np.round(np.mean(speedups, axis=0), decimals=2)
print("\n")
print("\n")
print(avg_speedups)
