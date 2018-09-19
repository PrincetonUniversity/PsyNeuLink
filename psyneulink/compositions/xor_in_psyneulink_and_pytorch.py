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
from psyneulink.compositions.autodiffcomposition import AutodiffComposition, AutodiffCompositionError
from psyneulink.compositions.pathwaycomposition import PathwayComposition
from psyneulink.compositions.systemcomposition import SystemComposition
from psyneulink.scheduling.condition import EveryNCalls
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink.scheduling.condition import EveryNPasses, AfterNCalls
from psyneulink.scheduling.time import TimeScale
from psyneulink.globals.keywords import NAME, INPUT_STATE, HARD_CLAMP, SOFT_CLAMP, NO_CLAMP, PULSE_CLAMP

logger = logging.getLogger(__name__)



# In this file, we create a neural network to approximate the XOR function (henceforth referred to 
# as an XOR model) in PsyNeuLink and in Pytorch, and go over some Pytorch basics. 

# The XOR function takes an input signal of 2 values and produces an output signal of a single value.
# The function is as follows:
# Input = 0, 0  -->  Target = 0
# Input = 0, 1  -->  Target = 1
# Input = 1, 0  -->  Target = 1
# Input = 1, 1  -->  Target = 0

# The model we create will have an input layer of 2 neurons (for the two input values), a hidden layer
# of 5 neurons, and an output layer of 1 neuron (for the single output). 



# XOR in PsyNeuLink -------------------------------------------------------------------------------


# Create mechanisms and projections to represent the layers and parameters:

# this mechanism takes input signals (a set of 2 values) and sends them on without changing them
# (it applies the default identity linear function to its input)
xor_in = TransferMechanism(name='xor_in',
                           default_variable=np.zeros(2))

# this mechanism takes an input signal that has been transformed by the hid_map projection (below) and 
# applies the logistic function to it.
xor_hid = TransferMechanism(name='xor_hid',
                            default_variable=np.zeros(10),
                            function=Logistic())

# this final mechanism takes the signal that has been transformed by the out_map projection (below) and 
# applies the logistic function to it to get the XOR model's output. 
xor_out = TransferMechanism(name='xor_out',
                            default_variable=np.zeros(1),
                            function=Logistic())

# this projection takes the input signal from xor_in and transforms it into a vector of 5 values for 
# the xor_hid mechanism.
# the parameters represented by the projection (the values of its matrix) start out as random
hid_map = MappingProjection(name='hid_map',
                            matrix=np.random.randn(2,10)*0.1,
                            sender=xor_in,
                            receiver=xor_hid)

# this projection takes the transformed signal from xor_hid and transforms it to get a single value
# for the xor_out mechanism.
out_map = MappingProjection(name='out_map',
                            matrix=np.random.randn(10,1)*0.1,
                            sender=xor_hid,
                            receiver=xor_out)

print(hid_map.matrix)
print("\n")
print(out_map.matrix)
print("\n")


# Put mechanisms and projections together to create a System representing the XOR model

# the order of mechanisms and projections is specified at the process level. 
xor_process = Process(pathway=[xor_in,
                               hid_map,
                               xor_hid,
                               out_map,
                               xor_out],
                      learning=pnl.LEARNING)

# the learning_rate parameter determines the size of learning updates during training for the System.
# it provides the only way for users to control learning update steps during training.  
xor_sys = System(processes=[xor_process],
                 learning_rate=0.5)

# The comparator mechanism for computing loss and the learning mechanisms/projections for doing
# backpropagation/the learning update during training are set up for the System automatically. 


# Create the inputs and targets for the XOR model:

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


# Train the System representing the XOR model by calling run. 

# setting num_trials to 4000 means performing the 4 learning steps detailed in the design doc 1000 times
# for each of the 4 XOR input-target pairs.

# The 4 learning steps are performed by the run method behind the scenes - as stated in the design doc,
# the loss measurement computed by the system's comparator mechanism defaults to MSE loss, and the 
# learning update carried out by learning mechanisms/projections defaults to that in basic stochastic
# gradient descent. 

results_sys = xor_sys.run(inputs={xor_in:xor_inputs}, 
                          targets={xor_out:xor_targets},
                          num_trials=12000)

proc_results_sys = xor_sys.run(inputs={xor_in:xor_inputs})
print(proc_results_sys[12000:])

# Print the model's output for each input after training:
# for i in range(len(xor_inputs)):
    # print("For the input: " xor_inputs[i])
    # print("The XOR System produces the output: " xor_sys)



# XOR in Pytorch ----------------------------------------------------------------------------------


# Create XOR model class that subclass neural network module

class Pytorch_XOR(torch.nn.module):
    
    
















