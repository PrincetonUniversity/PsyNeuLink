#%%
import numpy as np
import psyneulink as pnl
import pandas as pd
import argparse
import optuna
from full_lca_model_lc import make_lca_model

parser = argparse.ArgumentParser()
parser.add_argument("--subject_id", help="The subject ID to fit model.", default=1, type=int)

args = parser.parse_args()

def get_node(comp, name):
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    return None

def make_input_dict(comp, taskSequence, stimulusSequence):
    inputs = {
        get_node(comp, "Task Input"): [[np.array(v)] for v in taskSequence],
        get_node(comp, "Stimulus Input"): [[np.array(v)] for v in stimulusSequence],
        get_node(comp, "Bias Mechanism"): [[np.array(0)] for v in stimulusSequence],
        get_node(comp, "w1 Mechanism"): [[np.array(0)] for v in stimulusSequence],
        get_node(comp, "w2 Mechanism"): [[np.array(0)] for v in stimulusSequence],
    }
    return inputs

# High-level parameters that impact performance of the test
num_estimates = 10000
max_iterations = 5000

lca_params = dict(
    # Control LCA,
    c_gain=10,
    c_leak=7,
    c_competition=3,
    c_bias=0,
    c_w=4,
    # Stimulus LCA
    s_bias=-0.45,
    s_gain=5,
    s_leak=8,
    s_competition=8,
    # Decision LCA
    d_bias=-0.45,
    d_gain=5,
    d_leak=8,
    d_competition=8,
    d_noise=0.0,
    # Response LCA
    r_bias=-0.45,
    r_gain=5,
    r_leak=8,
    r_competition=8,
    r_threshold=0.0,
    r_noise=0.1,
    # Time gate
    non_decision_time=0.0,
    time_step_size=0.01,
    # Weights
    w1=1.0,
    w2=1.2,
    sdr_bias=-0.45,
    lc_base_gain=5.0,
    lc_scaling=1.0,
    lc_mode=0.9,
    lc_input=0.3,
    lc_threshold=0.5,
)

# Load in the behavioral data
data_path = "flanker_data_part1.csv"
data = pd.read_csv(data_path)

data["decision"] = data["decision"].astype("category")
data["subject_nr"] = data["subject_nr"].astype("category")

data_to_fit = data[data["PrevCongruency"].notna()].reset_index(drop=True)
data_to_fit["PrevCongruency"] = data_to_fit["PrevCongruency"].astype("category")

# Create the model inputs (these fit your lca model)
stimulusSequence = data_to_fit[['S1', 'S2', 'S3', 'S4']].to_numpy()
taskSequence = data_to_fit[['T1', 'T2']].to_numpy()

likelihood_include_mask = data_to_fit['likelihood_include_mask'].to_numpy(dtype='bool')

# Must match outcome_variables order: decision then response_time
data_to_fit = data_to_fit[['decision', 'response_time', 'subject_nr', 'PrevCongruency']]

# Make a stability flexibility composition
comp = make_lca_model(**lca_params)

taskInput = comp.nodes["Task Input"]
stimulusInput = comp.nodes["Stimulus Input"]
controlExecution = comp.nodes["Control Units\n[Color, Location]"]
responseLayer = comp.nodes["Response Units\n[Left, Right]"]
decisionGate = comp.nodes["DECISION_GATE"]
timeGate = comp.nodes["RT_GATE"]
biasMechanism = comp.nodes["Bias Mechanism"]
w1Mechanism = comp.nodes["w1 Mechanism"]
w2Mechanism = comp.nodes["w2 Mechanism"]
lc_drive = comp.nodes["LC Monitor"]
lc = comp.nodes["LC"]
lc_control= comp.nodes["LC Control"]

inputs = make_input_dict(comp, taskSequence, stimulusSequence)


fit_parameters = {
    ("termination_threshold", responseLayer): np.linspace(0.25, 0.70, 451),
    ("intercept", timeGate): np.linspace(0.1, 0.3, 201),
    ("intercept", biasMechanism): np.linspace(-0.5, 0.0, 51),
    ("gain", controlExecution): np.linspace(5, 20, 151),
    ("mode", lc): np.linspace(0.1, 0.9, 9),
    ("slope", lc): np.linspace(1, 4, 41),
    ("intercept", lc): np.linspace(3, 10, 8),
}

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp,
    parameters=fit_parameters,
    depends_on={
        ("termination_threshold", responseLayer): "subject_nr",
        ("gain", controlExecution): "subject_nr",
        ("intercept", timeGate): "subject_nr",
        ("mode", lc): "PrevCongruency",
        ("slope", lc): "subject_nr",
    },
    outcome_variables=[
        decisionGate.output_ports[0],
        timeGate.output_ports[0],
    ],
    data=data_to_fit,
    likelihood_include_mask=likelihood_include_mask,
    optimization_function=pnl.PECOptimizationFunction(method=optuna.samplers.CmaEsSampler(restart_strategy='ipop'), max_iterations=max_iterations),
    num_estimates=num_estimates,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)


print("Fit Flanker Data " + str(max_iterations) + " Iterations, ALL participants")
print("Running the PEC")
ret = pec.run(inputs=inputs)
optimal_parameters = pec.optimized_parameter_values

print(optimal_parameters)

data_dict = {k: [v] for k, v in optimal_parameters.items()}
df = pd.DataFrame(data_dict)
df['log_likelihood'] = pec.optimal_value
df['subject_nr'] = "ALL"
df['fit_type'] = "Flanker"
df['num_estimates'] = num_estimates
df['max_iterations'] = max_iterations

# Add parameters used to initialize the model
df = pd.concat([df, pd.DataFrame(lca_params, index=[0])], axis=1)

output_path = 'flanker_fit_lc_prevcongruency_part1.csv'

df.to_csv(output_path, index=False)

print("Job Complete!")
