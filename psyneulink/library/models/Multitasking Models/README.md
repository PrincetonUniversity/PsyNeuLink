# Multitasking Models

This directory includes models of multitasking simulations, scripts for
running the simulations, and tools for analyzing output data.

### Directory Structure:

[Base_Models](Base_Models): contains the base neural network models used in the simulation.
nnmodel.py is a model of a neural network with configurable layer size and 
parameters.

- [Musslick_et_al_2021](Musslick_et_al_2021): contains the simulations used in Musslick et al.

- [Petri_et_al_2021](Petri_et_al_2021): contains the simulations used in Petri et al.

- [Tools](Tools): contains tools used in the simulation scripts. These tools are used in
the simulations and do not need to be run separately by the user:
    - [Environments](Tools/Environments): tools for setting up the training environment.
    - [Analysis](Tools/Analysis): tools for analyzing the simulation data.

