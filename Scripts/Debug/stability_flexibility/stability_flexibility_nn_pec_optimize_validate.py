#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed

sys.path.append(".")

from stability_flexibility_nn import make_stab_flex, generate_trial_sequence

# Let's make things reproducible
pnl_seed = None
trial_seq_seed = None
set_global_seed(pnl_seed)

# High-level parameters that impact duration of parameter estimation
num_trials = 512
time_step_size = 0.01
num_estimates = 100

sf_params = dict(
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=time_step_size,
    non_decision_time=0.2,
    stim_hidden_wt=1.5,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=np.sqrt(0.1),
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

all_thresholds = np.linspace(0.001, 0.3, 3)
all_rr = np.array([])
all_rt = np.array([])
all_acc = np.array([])

for threshold_i in all_thresholds:

    # Update the parameters of the composition
    #comp.nodes["DDM"].function.threshold.base = threshold_i

    context = pnl.Context()

    comp.nodes["DDM"].function.parameters.threshold.set(threshold_i, context)

    # Generate sample data to
    switchFrequency = 0.5
    taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(512, switchFrequency, seed=trial_seq_seed)
    taskTrain = taskTrain[0:num_trials]
    stimulusTrain = stimulusTrain[0:num_trials]
    cueTrain = cueTrain[0:num_trials]
    correctResponse = correctResponse[0:num_trials]

    # CSI is in terms of time steps, we need to scale by ten because original code
    # was set to run with timestep size of 0.001
    cueTrain = [c / 10.0 for c in cueTrain]

    inputs = {
        taskInput: [[np.array(taskTrain[i])] for i in range(num_trials)],
        stimulusInput: [[np.array(stimulusTrain[i])] for i in range(num_trials)],
        cueInterval: [[np.array([cueTrain[i]])] for i in range(num_trials)],
        correctInfo: [[np.array([correctResponse[i]])] for i in range(num_trials)]
    }

    comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun, context=context)
    results = comp.results

    data_to_fit = pd.DataFrame(
        np.squeeze(np.array(results))[:, 1:], columns=["decision", "response_time"]
    )

    comp.reset()
    comp.results.clear()

    rr_i = np.mean(data_to_fit["decision"] / data_to_fit["response_time"])
    rt_i = np.mean(data_to_fit["response_time"])
    acc_i = np.mean(data_to_fit["decision"])

    all_rr = np.append(all_rr, rr_i)
    all_rt = np.append(all_rt, rt_i)
    all_acc = np.append(all_acc, acc_i)

print(all_thresholds)
print(all_rr)
print(all_rt)
print(all_acc)