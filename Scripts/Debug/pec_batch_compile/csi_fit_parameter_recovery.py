#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd
import random
import argparse
import optuna

parser = argparse.ArgumentParser()
parser.add_argument("--subject_id", help="The subject ID to fit model.", default=1, type=int)

args = parser.parse_args()

sys.path.append(".")

from csi_model_surrogate import make_stab_flex, generate_mixed_task_sequence

def get_node(comp, name):
    """
    Get the node from the composition with the given name. The name needs to match from the beginning, but it
    can have any numeric suffix after the name.
    """
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    return None


def make_input_dict(stab_flex_comp, taskSequence, stimulusSequence, correctResponseSequence):
    inputs = {
        get_node(stab_flex_comp, "Task Input"): [[np.array(v)] for v in taskSequence],
        get_node(stab_flex_comp, "Stimulus Input"): [[np.array(v)] for v in stimulusSequence],
        get_node(stab_flex_comp, "Correct Response"): [[np.array(v)] for v in correctResponseSequence],
        get_node(stab_flex_comp, "Cue Stimulus Interval"): [[np.array([0])] if (taskSequence[v] == taskSequence[v-1]) else [np.array([1])] for v in range(len(taskSequence))],
        get_node(stab_flex_comp, "Threshold Mechanism"): [[np.array([0])] for v in taskSequence],
    }

    return inputs



# High-level parameters that impact performance of the test
num_estimates = 10000
max_iterations = 5000
num_trials = 128

# We determine random parameter values to be recovered.
# However, we fix the iti as this recovers poorly and impairs recovery of other parameters

# Sample random values for parameters to be recovered, same for all three conditions
true_iti = random.randrange(start=0, stop=201)
true_csi_switch = random.randrange(start=0, stop=51)

# Sample random values for parameters to be recovered, different for all three conditions
true_gain = np.random.uniform(low=5, high=20, size=1)
true_threshold = np.random.uniform(low=0.08, high=0.25, size=1)
true_collapse = np.random.uniform(low=-0.003, high=0.0, size=1)
true_ndt = np.random.uniform(low=0.0, high=0.6, size=1)

sf_params = dict(
    gain=true_gain,
    leak=7.0,
    competition=3.0,
    iti=true_iti,
    csi_repeat=0,
    csi_switch=true_csi_switch,
    threshold=true_threshold,
    threshold_collapse=true_collapse,
    non_decision_time=true_ndt,
    lca_time_step_size=0.01,
    ddm_time_step_size=0.01,
    lca_noise=0.0,
    ddm_noise=0.1,
)

# Make a stability flexibility composition to simulate data for parameter recovery
comp_sim = make_stab_flex(**sf_params)

cueStimulusInterval = comp_sim.nodes["Cue Stimulus Interval"]
taskInput = comp_sim.nodes["Task Input"]
stimulusInput = comp_sim.nodes["Stimulus Input"]
correctResponse = comp_sim.nodes["Correct Response"]
controlExecution = comp_sim.nodes["Task Activations [C1, C2]"]
thresholdMechanism = comp_sim.nodes["Threshold Mechanism"]
decisionMaker = comp_sim.nodes["DDM"]
decisionGate = comp_sim.nodes["DECISION_GATE"]
responseGate = comp_sim.nodes["RESPONSE_GATE"]

# Generate trial sequence
taskSequence, stimulusSequence, correctResponseSequence, inputSequence = generate_mixed_task_sequence(num_trials, 0.5,0.5, int(args.subject_id))
inputs = make_input_dict(comp_sim, taskSequence, stimulusSequence, correctResponseSequence)

# Simulate the data
comp_sim.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
results = comp_sim.results

# Store the output
data_to_fit = pd.DataFrame(
    np.squeeze(np.array(results))[:, 1:], columns=["decision", "response_time"]
)

data_to_fit["decision"] = data_to_fit["decision"].astype("category")

# Next, create a composition for data fitting
comp_fit = make_stab_flex(**sf_params)

cueStimulusInterval = comp_fit.nodes["Cue Stimulus Interval-1"]
taskInput = comp_fit.nodes["Task Input-1"]
stimulusInput = comp_fit.nodes["Stimulus Input-1"]
correctResponse = comp_fit.nodes["Correct Response-1"]
controlExecution = comp_fit.nodes["Task Activations [C1, C2]-1"]
thresholdMechanism = comp_fit.nodes["Threshold Mechanism-1"]
decisionMaker = comp_fit.nodes["DDM-1"]
decisionGate = comp_fit.nodes["DECISION_GATE-1"]
responseGate = comp_fit.nodes["RESPONSE_GATE-1"]

# Create a parameter estimation composition to search for parameter values
# that optimize an objective function
fit_parameters = {
    ("gain", controlExecution): np.linspace(5.0, 20.0, 151),  # Control gain
    ("slope", cueStimulusInterval): np.linspace(0, 50, 51),  # Surrogate CSI on switch trials
    ("intercept", thresholdMechanism): np.linspace(0.08, 0.25, 341),  # Starting threshold DDM
    ("offset-integrator_function", thresholdMechanism): np.linspace(-0.003, 0.0, 301),  # Threshold collapse increment
    ("non_decision_time", decisionMaker): np.linspace(0.0, 0.6, 601),  # Non decision time
}

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp_fit,
    parameters=fit_parameters,
    outcome_variables=[
        decisionGate.output_ports[0],
        responseGate.output_ports[0],
    ],
    data=data_to_fit,
    optimization_function=pnl.PECOptimizationFunction(method=optuna.samplers.CmaEsSampler(restart_strategy='ipop'), max_iterations=max_iterations),
    num_estimates=num_estimates,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)

# Make the inputs for the composition
inputs = make_input_dict(comp_fit, taskSequence, stimulusSequence, correctResponseSequence)

print("Fit 1.1 Parameter Recovery, Surrogate CSI Switch Only, Fixed ITI, " + str(max_iterations) + " Iterations, Slurm Array " + str(args.subject_id))
print("Running the PEC")
ret = pec.run(inputs=inputs)
optimal_parameters = pec.optimized_parameter_values

print(optimal_parameters)

data_dict = {k: [v] for k, v in optimal_parameters.items()}
df = pd.DataFrame(data_dict)
df['log_likelihood'] = pec.optimal_value
df['subject_nr'] = args.subject_id
df['fit_type'] = "parameter_recovery_surrogate_csi_fixed_iti"
df['num_estimates'] = num_estimates
df['max_iterations'] = max_iterations
df['num_trials'] = num_trials


# Add parameters used to initialize the model, which include the true parameters
df = pd.concat([df, pd.DataFrame(sf_params, index=[0])], axis=1)

output_path = 'fits/fit_1.1_parameter_recovery_surrogate_switch_csi_fixed_iti_' + str(max_iterations) + '_iterations_sub' + str(args.subject_id) + '.csv'

df.to_csv(output_path, index=False)

print("Job Complete!")