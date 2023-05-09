#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import GridSearch

sys.path.append(".")

from stability_flexibility_nn import make_stab_flex, generate_trial_sequence

# Let's make things reproducible
pnl_seed = 1
trial_seq_seed = 1
set_global_seed(pnl_seed)

# High-level parameters the impact performance of the test
num_trials = 10
time_step_size = 0.01
num_estimates = 10

sf_params = dict(
    gain=3.0,
    leak=3.0,
    competition=3.0,
    lca_time_step_size=time_step_size,
    non_decision_time=0.2,
    stim_hidden_wt=1.5,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=0.1,
    lca_noise=0.0,
    hidden_resp_wt=2.0,
    ddm_time_step_size=time_step_size,
)

# Make a stability flexibility composition
comp = make_stab_flex(**sf_params)

# Create a parameter estimation composition to search for parameter values
# that optimize an objective function

controlModule = comp.nodes["Task Activations [C1, C2]"]
stimulusWeighting = comp.nodes["Stimulus Input to Hidden Weighting"]
hiddenLayer = comp.nodes["Hidden Units"]
hiddenWeighting = comp.nodes["Hidden Unit to Response Weighting"]
responseLayer = comp.nodes["Response Units"]
decisionMaker = comp.nodes["DDM"]
decisionGate = comp.nodes["DECISION_GATE"]
responseGate = comp.nodes["RESPONSE_GATE"]

fit_parameters = {
    ("gain", controlModule): np.linspace(1.0, 10.0, 1000),  # Gain
    ("threshold", decisionMaker): np.linspace(0.001, 0.05, 1000),  # Threshold
}

def reward_rate(sim_data):
    """
    Objective function for PEC to optimize. This function takes in the simulation data,
    a 3D array of shape (num_trials, num_estimates, num_outcome_vars), and returns a
    scalar value that is the reward rate.
    """
    return np.mean(sim_data[:, :, 0][:] / sim_data[:, :, 1][:])


pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp,
    parameters=fit_parameters,
    outcome_variables=[
        decisionGate.output_ports[0],
        responseGate.output_ports[0],
    ],
    objective_function=reward_rate,
    optimization_function='differential_evolution',
    num_estimates=num_estimates,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)

# Generate some model inputs
taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(512, 0.5, seed=trial_seq_seed)
taskTrain = taskTrain[0:num_trials]
stimulusTrain = stimulusTrain[0:num_trials]
cueTrain = cueTrain[0:num_trials]
correctResponse = correctResponse[0:num_trials]

# CSI is in terms of time steps, we need to scale by ten because original code
# was set to run with timestep size of 0.001
cueTrain = [c / 10.0 for c in cueTrain]

taskInput = comp.nodes["Task Input"]
stimulusInput = comp.nodes["Stimulus Input"]
cueInterval = comp.nodes["Cue-Stimulus Interval"]
correctInfo = comp.nodes["Correct Response Info"]

inputs = {
    taskInput: [[np.array(taskTrain[i])] for i in range(num_trials)],
    stimulusInput: [[np.array(stimulusTrain[i])] for i in range(num_trials)],
    cueInterval: [[np.array([cueTrain[i]])] for i in range(num_trials)],
    correctInfo: [[np.array([correctResponse[i]])] for i in range(num_trials)]
}

print("Running the PEC")
ret = pec.run(inputs=inputs)
print("Optimal parameters: ", pec.optimized_parameter_values)
print("Optimal Reward Rate: ", pec.optimal_value)

print("Running the PEC again")
ret = pec.run(inputs=inputs)
print("Optimal parameters: ", pec.optimized_parameter_values)
print("Optimal Reward Rate: ", pec.optimal_value)