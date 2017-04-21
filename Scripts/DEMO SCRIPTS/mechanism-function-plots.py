from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *

import matplotlib
matplotlib.use('TkAgg')
# LINEAR

# Creating a Transfer Mechanism with a Linear function
my_Transfer_Linear = TransferMechanism(name='my_Transfer_Linear',
                       function=Linear()
                       )

# Plots Linear Function
# Any parameters specified in above mechanism (line6) will be reflected in plot
my_Transfer_Linear.plot()

# ----------------------------------------

# SOFTMAX

# Creating a Transfer Mechanism with a SoftMax function
my_Transfer_SoftMax = TransferMechanism(name='my_Transfer_SoftMax',
                       function=SoftMax()
                       )

# Plots SoftMax Function
# Any parameters specified in above mechanism (line15) will be reflected in plot
my_Transfer_SoftMax.plot()

# -----------------------------------------

# LOGISTIC

# Creating a Transfer Mechanism with a Logistic function
my_Transfer_Logistic = TransferMechanism(name='my_Transfer_Logistic',
                       function=Logistic()
                       )

# Plots Logistic Function
# Any parameters specified in above mechanism (line27) will be reflected in plot
my_Transfer_Logistic.plot()


## **** NEW FEATURES BELOW ****

# You can change the domain of any plot by passing in a list of [low x-value, high x-value] to the plot method
my_Transfer_Logistic.plot([-20,20])

# You can change the parameters that the function is plotted with by specifying them in the mechanism's function
# Creating a Transfer Mechanism with a Logistic function
my_Custom_Transfer_Logistic = TransferMechanism(name='my_Transfer_Logistic',
                       function=Logistic(gain=0.5,bias=0.5)
                       )

my_Custom_Transfer_Logistic.plot()

# -----------------------------------------

# DDM - Dynamic Plot

# Creating a DDM Mechanism
my_DDM = DDM(function=Integrator(rate=0.1, noise=0.2, integration_type="diffusion"),
             name='My_DDM',
             time_scale=TimeScale.TIME_STEP
             )

# Parameters specified in above mechanism will be reflected in plot

# Plot method takes two parameters --
# First parameter is stimulus (default =1.0)
# Second parameter is threshold (default = 10.0)
# E.g. my_DDM.plot(2.0, 11.0) plots the DDM with a stimulus of 2.0 until it crosses -11.0 or  11.0
my_DDM.plot()

#-----------------------------------------
