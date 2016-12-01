# EVCM
**A computational model of control allocation based on the Expected Value of Control**

*Sebastian Musslick, Amitai Shenhav, Matthew Botvinick & Jonathan Cohen*

## Introduction

While cognitive control has long been known to adjust flexibly in response to signals like errors or conflict, when and how the decision is made to adjust control remains an open question. Recently, Shenhav and colleagues (1) described a theoretical framework whereby control allocation follows from a reward optimization process, according to which the identities and intensities of potential cognitive control signals are selected so as to maximize expected reward, while at the same time discounting this reward by an intrinsic cost that attaches to increases in control allocation. This discounted expected reward quantity is referred to as the Expected Value of Control (EVC). While the form of the reward optimiza- tion policy was described, Shenhav et al. left open the question of how this optimization process might be implemented in explicit computational mechanisms, and used to make predictions concerning performance in experimental tasks. Here we describe such an implementation.

To simulate the influence of cognitive control on behavior in relevant task settings we parameterize such tasks as processes of accumulation to bound and allow control to influence the parameters of that accumulation process, resulting in attendant changes in reward rate. Control signals are specified based on an internal model of the task environment, as well as the intrinsic cost of control allocation. The latter scales both with the amount of overall control exerted and with the change in control allocation from the previous time step. After control is applied, feedback from the actual task environment is used to update the internal model, and control specification is re-optimized. The current model implementation replicates classic findings in the cognitive control literature related to sequential adaptation and task switching, and is able to generate testable predictions for future studies of voluntary rather than instructed allocation of control.

1. A. Shenhav, M. M. Botvinick, J. D. Cohen, *Neuron* **79**, 217 (2013).


## Code organization

The EVC model includes the following packages

The class hierarchy of the EVC model itself is set up to allow for multiple implementations of the EVC theory (e.g. a drift-diffusion model (DDM) implementation or a prospective neural network implementation). All specific EVC simulations are implemented in separate classes and inherit parameters from parent classes in order to constrain the parameter space.

+ **EVC** 
This package contains classes that implement the general (implementation-independent) computations of the EVC theory (see Shenhav, Botvinick & Cohen, 2013). It also includes classes representing the control signals, the task environment and learning PsyNeuLink.Components.

+ **EVC.DDM**
Classes in this package describe DDM-specific computations such as the retrieval of reaction times and trial outcome probabilities. It also includes classes that provide an interface between implementation-independent model components from the "EVC" package (e.g. the environment and control signals) and the DDM. 

+ **EVC.MSDDM**
Classes in this package implement a multi-stage drift-diffusion model (MSDDM) for the retrieval of reaction times and trial outcome probabilities. It also includes classes that provide an interface between implementation-independent model components from the "EVC" package (e.g. the environment and control signals) and the MSDDM.

+ **Simulations**
This package includes all simulation files which parameterize the components of a single simulation (e.g. control signals, task environment, etc.). Each simulation file inherits DDM-specific parameters (e.g. default DDM parameters) from the "DDMSim" class which in turn inherits general simulation parameters (e.g. learning rates) from the "Simulation" class. The user may refer to the demo simulation files for more information with respect to how to parameterize a simulation.

## Demo simulations

The main folder contains multiple demo simulations grouped by category:

**RunSanityChecks.m**
Sanity checks of the expected value of control model (reward & difficulty manipulations)

**RunControlAdaptationSims.m**
Sequential adaptation effects (e.g. in response to conflict and errors).

**RunTaskSwitchingSims.m**
Task switching effects (cued task switching & voluntary task switching). 

Within each of these scripts, the user may run the code segment of a specific simulation to obtain results.


Corresponding author:

Sebastian Musslick (musslick@princeton.edu)
