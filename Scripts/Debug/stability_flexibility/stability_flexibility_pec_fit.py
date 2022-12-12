#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed
from psyneulink.core.components.functions.fitfunctions import MaxLikelihoodEstimator

sys.path.append(".")

from stability_flexibility import make_stab_flex, generate_trial_sequence

# Let's make things reproducible
seed = 0
np.random.seed(seed)
set_global_seed(seed)

# High-level parameters the impact performance of the test
num_trials = 40
time_step_size = 0.01
num_estimates = 40

sf_params = dict(
    gain=3.0,
    leak=3.0,
    competition=4.0,
    lca_time_step_size=0.01,
    non_decision_time=0.2,
    automaticity=0.15,
    starting_value=0.0,
    threshold=0.6,
    ddm_noise=0.1,
    lca_noise=0.0,
    scale=1.0,
    ddm_time_step_size=0.01,
)

# Generate some sample data to run the model on
taskTrain, stimulusTrain, cueTrain, switch = generate_trial_sequence(240, 0.5)
taskTrain = taskTrain[0:3]
stimulusTrain = stimulusTrain[0:3]
cueTrain = cueTrain[0:3]

# Make a stability flexibility composition
comp = make_stab_flex(**sf_params)

# Let's run the model with some sample data
taskLayer = comp.nodes["Task Input [I1, I2]"]
stimulusInfo = comp.nodes["Stimulus Input [S1, S2]"]
cueInterval = comp.nodes["Cue-Stimulus Interval"]
correctInfo = comp.nodes["Correct Response Info"]

inputs = {
    taskLayer: taskTrain,
    stimulusInfo: stimulusTrain,
    cueInterval: cueTrain,
    correctInfo: np.zeros_like(cueTrain),
}

comp.run(inputs)
results = comp.results

#%%

data_to_fit = pd.DataFrame(
    np.squeeze(np.array(results))[:, 1:], columns=["decision", "response_time"]
)
data_to_fit["decision"] = data_to_fit["decision"].astype("category")

# Create a parameter estimation composition to fit the data we just generated and hopefully recover the
# parameters of the composition.

controlModule = comp.nodes["Task Activations [Act1, Act2]"]
congruenceWeighting = comp.nodes["Automaticity-weighted Stimulus Input [w*S1, w*S2]"]
decisionMaker = comp.nodes["DDM"]
decisionGate = comp.nodes["DECISION_GATE"]
responseGate = comp.nodes["RESPONSE_GATE"]

fit_parameters = {
    ("gain", controlModule): np.linspace(1.0, 10.0, 1000),  # Gain
    ("slope", congruenceWeighting): np.linspace(0.0, 0.5, 1000),  # Automaticity
    ("threshold", decisionMaker): np.linspace(0.0, 1.0, 1000),  # Threshold
}

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp,
    parameters=fit_parameters,
    outcome_variables=[
        decisionGate.output_ports[0],
        responseGate.output_ports[0],
    ],
    data=data_to_fit,
    optimization_function=MaxLikelihoodEstimator(),
    num_estimates=num_estimates,
    num_trials_per_estimate=len(taskTrain),
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)

# # ll, sim_data = pec.log_likelihood(0.3, 0.6, inputs=inputs_dict)
outer_comp_inputs = [
    [
        np.array(taskTrain[i]),
        np.array(stimulusTrain[i]),
        np.array(cueTrain[i]),
        np.array(0),
    ]
    for i in range(len(cueTrain))
]

# outer_comp_inputs = pec.get_input_format(num_trials=len(cueTrain))
outer_comp_inputs = pec.get_input_format(num_trials=len(cueTrain))

# ret = pec.run(inputs={comp: outer_comp_inputs}, num_trials=len(cueTrain))
ret = pec.run(inputs=outer_comp_inputs, num_trials=len(cueTrain))

# Check that the parameters are recovered and that the log-likelihood is correct
# assert np.allclose(pec.controller.optimal_parameters, [0.3, 0.6], atol=0.1)
