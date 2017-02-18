from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *

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

# -----------------------------------------

# DDM - FIXED PARAMETERS

# Creating a DDM Mechanism
my_DDM = DDM(function=DDMIntegrator(rate=0.5, noise=0.2),
             name='My_DDM',
             time_scale=TimeScale.TIME_STEP
             )

# Plots DDM dynamically over time with a fixed horizontal axis
# NOTE: parameters specified in above mechanism will be reflected in plot
my_DDM.plot()

#-----------------------------------------

# DDM - VARIABLE PARAMETERS

# Creating a DDM Mechansim
# Specifying a 'plot_threshold' (line64) causes the mechanism to plot its out put at each time step until the threshold
# Any changes to paramters (line62) will be reflected on execution and in the plot
# Plot width updates continuously
my_DDM2 = DDM(function=DDMIntegrator(rate=0.01, noise=0.2),
             name='My_DDM2',
             time_scale=TimeScale.TIME_STEP,
             plot_threshold = 10.0
             )

my_DDM2.execute()

