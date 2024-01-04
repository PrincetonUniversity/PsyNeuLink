#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd
import optuna

from psyneulink.core.globals.utilities import set_global_seed


sys.path.append(".")

from stability_flexibility import make_stab_flex, generate_trial_sequence

# Let's make things reproducible
pnl_seed = 2
trial_seq_seed = 1
set_global_seed(pnl_seed)

# High-level parameters the impact performance of the test
num_trials = 240
time_step_size = 0.01
num_estimates = 100

sf_params = dict(
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=time_step_size,
    non_decision_time=0.2,
    automaticity=0.01,
    starting_value=0.0,
    threshold=0.3,
    ddm_noise=0.1,
    lca_noise=0.0,
    scale=0.2,
    ddm_time_step_size=time_step_size,
)

# Generate some sample data to run the model on
taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(240, 0.5, seed=trial_seq_seed)
taskTrain = taskTrain[0:num_trials]
stimulusTrain = stimulusTrain[0:num_trials]
cueTrain = cueTrain[0:num_trials]
correctResponse = correctResponse[0:num_trials]

# CSI is in terms of time steps, we need to scale by ten because original code
# was set to run with timestep size of 0.001
cueTrain = [c / 10.0 for c in cueTrain]

# Make a stability flexibility composition
comp = make_stab_flex(**sf_params)

# Let's run the model with some sample data
taskLayer = comp.nodes["Task Input [I1, I2]"]
stimulusInfo = comp.nodes["Stimulus Input [S1, S2]"]
cueInterval = comp.nodes["Cue-Stimulus Interval"]
correctInfo = comp.nodes["Correct Response Info"]

inputs = {
    taskLayer: [[np.array(taskTrain[i])] for i in range(num_trials)],
    stimulusInfo: [[np.array(stimulusTrain[i])] for i in range(num_trials)],
    cueInterval: [[np.array([cueTrain[i]])] for i in range(num_trials)],
    correctInfo: [[np.array([correctResponse[i]])] for i in range(num_trials)]
}

print("Running inner composition to generate data to fit for parameter recovery test.")
comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
results = comp.results

print("Setting up PEC")

data_to_fit = pd.DataFrame(
    np.squeeze(np.array(results))[:, 1:], columns=["decision", "response_time"]
)
#data_to_fit["decision"] = data_to_fit["decision"].astype("category")

print(f"PNL Seed = {pnl_seed}")
print(f"Trial Seq Seed = {trial_seq_seed}")
print(f"task[0:5] = {taskTrain[0:5]}")
print(f"stimulus[0:5] = {stimulusTrain[0:5]}")
print(data_to_fit[0:5])


# Create a parameter estimation composition to search for parameter values
# that optimize an objective function

controlModule = comp.nodes["Task Activations [Act1, Act2]"]
congruenceWeighting = comp.nodes["Automaticity-weighted Stimulus Input [w*S1, w*S2]"]
decisionMaker = comp.nodes["DDM"]
decisionGate = comp.nodes["DECISION_GATE"]
responseGate = comp.nodes["RESPONSE_GATE"]

fit_parameters = {
    ("threshold", decisionMaker): np.linspace(0.01, 0.5, 100),  # Threshold
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
    optimization_function=pnl.PECOptimizationFunction(method=optuna.samplers.QMCSampler(),
                                                      max_iterations=50,
                                                      direction='minimize'),
    num_estimates=num_estimates,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)

print("Running the PEC")
#comp.show_graph()
ret = pec.run(inputs=inputs)
print("Optimal threshold: ", pec.optimized_parameter_values)
print("Optimal Reward Rate: ", pec.optimal_value)
print("Current threshold: ", sf_params["threshold"], ", Reward rate: ", np.mean(data_to_fit["decision"] / data_to_fit['response_time']))