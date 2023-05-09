#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed

sys.path.append(".")

from stability_flexibility_nn import make_stab_flex, generate_trial_sequence

# Let's make things reproducible
pnl_seed = 1
trial_seq_seed = 1
set_global_seed(pnl_seed)

# High-level parameters that impact duration of parameter estimation
num_trials = 512
time_step_size = 0.01
num_estimates = 100

sf_params = dict(
    gain=4.4,
    leak=3.0,
    competition=3.0,
    lca_time_step_size=time_step_size,
    non_decision_time=0.2,
    stim_hidden_wt=1.5,
    starting_value=0.0,
    threshold=0.0157,
    ddm_noise=0.1,
    lca_noise=0.0,
    hidden_resp_wt=2.0,
    ddm_time_step_size=time_step_size,
)

# Initialize composition
comp = make_stab_flex(**sf_params)
taskInput = comp.nodes["Task Input"]
stimulusInput = comp.nodes["Stimulus Input"]
cueInterval = comp.nodes["Cue-Stimulus Interval"]
correctInfo = comp.nodes["Correct Response Info"]

# Generate sample data
switchFrequency = 0.5
taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(512, switchFrequency, seed=trial_seq_seed)
taskTrain = taskTrain[0:num_trials]
stimulusTrain = stimulusTrain[0:num_trials]
cueTrain = cueTrain[0:num_trials]
correctResponse = correctResponse[0:num_trials]

# CSI is in terms of time steps, we need to scale by ten because original code
# was set to run with timestep size of 0.001
cueTrain = [c / 10.0 for c in cueTrain]

cueTrain = [90 for c in cueTrain]

inputs = {
    taskInput: [[np.array(taskTrain[i])] for i in range(num_trials)],
    stimulusInput: [[np.array(stimulusTrain[i])] for i in range(num_trials)],
    cueInterval: [[np.array([cueTrain[i]])] for i in range(num_trials)],
    correctInfo: [[np.array([correctResponse[i]])] for i in range(num_trials)]
}

comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
results = comp.results

data_to_fit = pd.DataFrame(
    np.squeeze(np.array(results))[:, 1:], columns=["decision", "response_time"]
)

print("Average RT: ", np.mean(data_to_fit["response_time"]))
print("Average ACC: ", np.mean(data_to_fit["decision"]))
print("Average RR: ", np.mean(data_to_fit["decision"] / data_to_fit["response_time"]))