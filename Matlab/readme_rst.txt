## Expected Value of Control Model

1) Demo simulations

The main folder contains multiple demo simulations grouped by category:

RunSanityChecks.m
Sanity checks of the expected value of control model (reward & difficulty manipulations)

RunControlAdaptationSims.m
Sequential adaptation effects (e.g. in response to conflict and errors).

RunTaskSwitchingSims.m
Task switching effects (cued task switching & voluntary task switching). 

Within each of these scripts, the user may run the code segment of a specific simulation to obtain results.

2) Code organization

The EVC model includes the following packages

The class hierarchy of the EVC model itself is set up to allow for multiple implementations of the EVC theory (e.g. a drift-diffusion model (DDM) implementation or a prospective neural network implementation). All specific EVC simulations are implemented in separate classes and inherit parameters from parent classes in order to constrain the parameter space.

+ EVC
This package contains classes that implement the general (implementation-independent) computations of the EVC theory (see Shenhav, Botvinick & Cohen, 2013). It also includes classes representing the control signals, the task environment and learning PsyNeuLink.Components.

+ EVC.DDM
Classes in this package describe DDM-specific computations such as the retrieval of reaction times and trial outcome probabilities. It also includes classes that provide an interface between implementation-independent model components from the "EVC" package (e.g. the environment and control signals) and the DDM. 

+ Simulatons
This package includes all simulation files which parameterize the components of a single simulation (e.g. control signals, task environment, etc.). Each simulation file inherits DDM-specific parameters (e.g. default DDM parameters) from the "DDMSim" class which in turn inherits general simulation parameters (e.g. learning rates) from the "Simulation" class. The user may refer to the demo simulation files for more information with respect to how to parameterize a simulation.


Corresponding author:

Sebastian Musslick (musslick@princeton.edu)
 
