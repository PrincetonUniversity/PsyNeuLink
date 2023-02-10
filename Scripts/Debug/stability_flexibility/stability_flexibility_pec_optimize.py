#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import GridSearch

sys.path.append(".")

from stability_flexibility import make_stab_flex, generate_trial_sequence

# Let's make things reproducible
seed = 0
np.random.seed(seed)
set_global_seed(seed)

# High-level parameters the impact performance of the test
num_trials = 120
time_step_size = 0.01
num_estimates = 30000

sf_params = dict(
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=time_step_size,
    non_decision_time=0.2,
    automaticity=0.01,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=0.1,
    lca_noise=0.0,
    scale=0.2,
    ddm_time_step_size=time_step_size,
)

# Generate some sample data to run the model on
taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(240, 0.5)
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

#%%

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

def objective_function(variable):
    decision_variable = variable[0]
    rt_variable = variable[1]
    # if rt_variable == 0.0:
    #     return np.array([0.0])
    rr = decision_variable / rt_variable
    return rr

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp,
    parameters=fit_parameters,
    outcome_variables=[
        decisionGate.output_ports[0],
        responseGate.output_ports[0],
    ],
    objective_function=objective_function,
    optimization_function=GridSearch(),
    num_estimates=num_estimates,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)

print("Running the PEC")
ret = pec.run(inputs=inputs)
assert True
# optimal_parameters = pec.controller.optimal_parameters

# # Print the optimized parameters.
# records = []
# for (name, mech), recovered_param in zip(fit_parameters.keys(), optimal_parameters):
#
#     if name == "slope":
#         true_param = sf_params['automaticity']
#     else:
#         true_param = sf_params[name]
#
#     records.append((name, mech.name, recovered_param))
# df = pd.DataFrame(records, columns=['Parameter', 'Component', 'Optimized Value'])
# print(df)
