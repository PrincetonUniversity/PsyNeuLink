import numpy as np
import torch
from torch import nn

import psyneulink as pnl
from psyneulink.components.system import System
from psyneulink.components.process import Process
from psyneulink.components.functions.function import Logistic
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection



# In this file, we create a neural network to approximate the XOR function (henceforth referred to 
# as an XOR model) in PsyNeuLink and in Pytorch, and go over some Pytorch basics. 

# The XOR function takes an input signal of 2 values and produces an output signal of a single value.
# The function maps inputs to outputs as follows:

# 0, 0  -->  0
# 0, 1  -->  1
# 1, 0  -->  1
# 1, 1  -->  0

# The model we create will have an input layer of 2 neurons (for the two input values), a hidden layer
# of 10 neurons (10 was chosen randomely), and an output layer of 1 neuron (for the single output). 



# XOR in PsyNeuLink -------------------------------------------------------------------------------


# Create mechanisms and projections to represent the layers and parameters:

# mechanism representing the input layer
xor_in = TransferMechanism(name='xor_in',
                           default_variable=np.zeros(2))

# mechanism representing the hidden layer
xor_hid = TransferMechanism(name='xor_hid',
                            default_variable=np.zeros(10),
                            function=Logistic())

# mechanism representing the output layer
xor_out = TransferMechanism(name='xor_out',
                            default_variable=np.zeros(1),
                            function=Logistic())

# projection that takes the signal from the input layer and transforms it to get an input for 
# the hidden layer (the xor_hid mechanism)
hid_map = MappingProjection(name='hid_map',
                            matrix=np.random.randn(2,10)*0.1,
                            sender=xor_in,
                            receiver=xor_hid)

# projection that takes the signal from the hidden layer and transforms it to get an input for
# the output layer (the xor_out mechanism)
out_map = MappingProjection(name='out_map',
                            matrix=np.random.randn(10,1)*0.1,
                            sender=xor_hid,
                            receiver=xor_out)


# Put together the mechanisms and projections to get the System representing the XOR model:

# the order of mechanisms and projections is specified at the process level
xor_process = Process(pathway=[xor_in,
                               hid_map,
                               xor_hid,
                               out_map,
                               xor_out],
                      learning=pnl.LEARNING)

# the learning_rate parameter determines the size of learning updates during training for the System.
# it provides the only easy way for users to control learning during training (afaik there's no 
# way to change the actual optimizer that uses the learning rate to update parameters, or the loss)
xor_sys = System(processes=[xor_process],
                 learning_rate=0.1)

# The comparator mechanism for computing loss and the learning mechanisms/projections for doing
# backpropagation/the learning update during training are set up for the System automatically. 


# Inputs and targets for the XOR model:

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


# Train the System representing the XOR model for some number of epochs by calling run. 

# The 4 learning steps are performed by the run method behind the scenes - as stated in the design doc,
# the loss measurement computed by the system's comparator mechanism defaults to MSE loss, and the 
# learning update carried out by learning mechanisms/projections defaults to that in basic stochastic
# gradient descent. 

epochs = 100

results_sys = xor_sys.run(inputs={xor_in:xor_inputs}, 
                          targets={xor_out:xor_targets},
                          num_trials=(len(xor_inputs) * epochs + 1))

# process the inputs after training
proc_results_sys = xor_sys.run(inputs={xor_in:xor_inputs})

# see outputs for the 4 processed inputs after training
print(proc_results_sys[(len(xor_inputs) * epochs + 1):])

print("\n")
print("\n")



# XOR in Pytorch ----------------------------------------------------------------------------------


# The XOR model class - As can be seen, the class subclasses pytorch's neural net module. 
# The class defines a "blueprint" for the XOR model - to actually get a model we can do 
# processing/learning with, we create an instance of the class.

class Pytorch_XOR(torch.nn.Module):
    
    # the init method is where the parameters must be defined
    def __init__(self):
        
        # start by calling the nn module initialization
        super(Pytorch_XOR, self).__init__()
        
        # We create torch tensors and wrap them in Parameter objects to represent parameters. The
        # tensor can be thought of as a projection's matrix (the numpy array that actually represents 
        # its values), and the Parameter object as the projection. Wrapping a tensor in a Parameter 
        # object tells Pytorch to build a computation graph to track the computations (and their gradients) 
        # that the tensor is involved in, later in the forward method. 
        
        # The double() function used below makes sure the tensors representing parameters use doubles 
        # (vs floats for ex.)
        
        # this parameter object corresponds to the hid_map projection in the system above
        self.pt_hid_map = nn.Parameter(torch.randn(2,10).double()*0.1)
        
        # and this one to the out_map projection
        self.pt_out_map = nn.Parameter(torch.randn(10,1).double()*0.1)
        
        # uncomment the following to define bias parameters:
        # self.pt_hid_bias = nn.Parameter(torch.randn(10)*0.1)
        # self.pt_out_bias = nn.Parameter(torch.randn(1)*0.1)
    
    # the forward method is where forward computation is defined. Model input is an argument to the method
    def forward(self, input):
        
        # we define a sigmoid function object to apply to inputs to the hidden and output layers
        # (the sigmoid's the same as the logistic function in the system)
        logistic = nn.Sigmoid()
        
        # compute the input to the hidden layer by transforming inputs with the hidden layer weight parameters
        xor_hid_input = torch.matmul(input, self.pt_hid_map)
        
        # uncomment to add bias
        # xor_hid_input += self.pt_hid_bias
        
        # compute the hidden layer value by applying the sigmoid to the input
        xor_hid = logistic(xor_hid_input)
        
        # compute the input to the output layer the same way
        xor_out_input = torch.matmul(xor_hid, self.pt_out_map)
        
        # uncomment to add bias
        # xor_out_input += self.pt_out_bias
        
        # compute the output layer value by applying the sigmoid to the input
        xor_out = logistic(xor_out_input)
        
        # return the model output
        return xor_out


# Create an instance of the Pytorch_XOR class to use as an XOR model

xor_pt = Pytorch_XOR()


# Move the inputs and targets previously defined in numpy to pytorch. The below method of creating
# a tensor from a numpy array results in the tensor and numpy array sharing memory. 

xor_inputs_pt = torch.tensor(xor_inputs).double()
xor_targets_pt = torch.tensor(xor_targets).double()


# Set up function for training the XOR model object

def xor_pt_training(model, inputs, targets, epochs, loss_measure, optimizer):
    
    # iterate over epochs
    for epoch in range(epochs):
        
        # iterate over inputs
        for i in range(len(inputs)):
            
            # perform forward computation on input to get output
            output = model.forward(inputs[i])
            
            # calculate loss on output
            loss = loss_measure(output, targets[i])
            
            # perform backpropagation by calling loss.backward - this computes the gradient of
            # loss with respect to the tensors involved in its computation that pytorch has been
            # tracking - ie. the tensors in Parameter objects of the model. 
            loss.backward()
            
            # perform the learning update by calling the optimizer - it uses gradients computed
            # for Parameters in the computation graph to update them
            optimizer.step()
            
            # reset the gradients computed in the computation graph for the next training iteration
            optimizer.zero_grad()


# Call the training function to train the XOR model instance

# we use mean squared error for loss, like PsyNeuLink
l = nn.MSELoss()

# we use stochastic gradient descent, like PsyNeuLink, and set the learning rate to the same
# value used in the system above
optim = torch.optim.SGD(xor_pt.parameters(), lr=0.1)

# same number of epochs as system
epochs = 100

xor_pt_training(model = xor_pt,
                inputs = xor_inputs_pt,
                targets = xor_targets_pt,
                epochs = epochs,
                loss_measure = l,
                optimizer = optim)

# process inputs after training 
with torch.no_grad(): # shut off tracking computations for parameters for the time being
    proc_results1 = xor_pt.forward(xor_inputs_pt[0])
    proc_results2 = xor_pt.forward(xor_inputs_pt[1])
    proc_results3 = xor_pt.forward(xor_inputs_pt[2])
    proc_results4 = xor_pt.forward(xor_inputs_pt[3])

# print outputs
print(proc_results1)
print(proc_results2)
print(proc_results3)
print(proc_results4)



# Now, some notes that connect the Pytorch code above to the autodiff composition and pytorch model creator,
# and define key Pytorch data types/methods that arise for both classes but are not present above. 

# The PytorchModelCreator class is a subclass of the nn module - unlike the Pytorch_XOR class
# above though, its init parses the processing graph of an autodiff composition to define
# parameters and forward computation for the autodiff composition in Pytorch. 

# The autodiff_training function of the autodiff composition essentially does what the xor_pt_training
# function does above. 

# The pytorch model creator's init stores Parameter objects in a ParameterList, a data
# type that Pytorch can use to keep track of all model parameters (instead of having each 
# parameter saved individually, as is the case above). 

# When returning parameters as numpy arrays to the user, helper methods of the pytorch model creator
# use the detach() function to first create tensors representing the pytorch model's parameters 
# for which computations are not tracked. This must be done to copy parameters to numpy. 

# In the autodiff_training function of the autodiff composition, processing is done in a code block created
# by setting the "torch.no_grad" flag - this is precautionary, a way of telling pytorch not to track 
# computations on parameters and their gradients while doing processing (there is no need to).


