#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed

sys.path.append(".")

from stability_flexibility import make_stab_flex, generate_trial_sequence

def get_node(comp, name):
    """
    Get the node from the composition with the given name. The name needs to match from the beginning, but it
    can have any numeric suffix after the name.
    """
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    return None


def make_input_dict(stab_flex_comp, taskTrain, stimulusTrain, cueTrain, correctResponse):
    inputs = {
        get_node(stab_flex_comp, "Task Input [I1, I2]"): [[np.array(v)] for v in taskTrain],
        get_node(stab_flex_comp, "Stimulus Input [S1, S2]"): [[np.array(v)] for v in stimulusTrain],
        get_node(stab_flex_comp, "Cue-Stimulus Interval"): [[np.array(v)] for v in cueTrain],
        get_node(stab_flex_comp, "Correct Response Info"): [[np.array(v)] for v in correctResponse]
    }

    return inputs

def run_stab_flex_cond(
        taskTrain,
        stimulusTrain,
        cueTrain,
        correctResponse,
        **kwargs):
    """
    Create a stability flexibility composition and run it with the given parameters. Return the composition and the
    results as a pandas DataFrame. If any of the parameters are a list, then that parameter is assumed to be trial-wise
    and the length of the list should be the number of trials. A control mechanism will be added to the composition to
    override the parameter with the value from the input.
    """

    # Remove any parameters that are trial-wise from the kwargs, these values will be passed in as inputs
    # to the composition.
    cond_params = {name: value for name, value in kwargs.items()
                   if isinstance(value, list) or isinstance(value, np.ndarray)}

    # Remove the trial-wise parameters from the kwargs
    kwargs = {name: value for name, value in kwargs.items() if name not in cond_params}

    # Make a stability flexibility composition
    comp = make_stab_flex(**kwargs)

    inputs = make_input_dict(comp, taskTrain, stimulusTrain, cueTrain, correctResponse)

    # A dict to map keyword arg name to the corresponding mechanism in the composition
    param_map = {
        "gain": ("gain", comp.nodes["Task Activations [Act1, Act2]"]), # Gain
        "automaticity": ("slope", comp.nodes["Automaticity-weighted Stimulus Input [w*S1, w*S2]"]), # Automaticity
        "threshold": ("threshold", comp.nodes["DDM"]), # Threshold
        "non_decision_time": ("non_decision_time", comp.nodes["DECISION_GATE"]), # Non-decision time
    }
    # Go through the parameters and check if any are trial-wise, if so, add a control mechanism to override the value on
    # trial-by-trial basis with the value from the input.
    pec_mechs = {}
    for (name, value) in cond_params.items():

        if len(value) != num_trials:
            raise ValueError("Length of trial-wise parameter must be equal to the number of trials.")

        pec_mechs[name] = pnl.ControlMechanism(name=f"{name}_control",
                                               control_signals=param_map[name],
                                               modulation=pnl.OVERRIDE)
        comp.add_node(pec_mechs[name])
        inputs[pec_mechs[name]] = [[np.array([value[i]])] for i in range(num_trials)]

    comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)

    df = pd.DataFrame(
        np.squeeze(np.array(comp.results))[:, 1:], columns=["decision", "response_time"]
    )
    df["decision"] = df["decision"].astype("category")

    # Add the trial-wise parameters to the DataFrame as well.
    for name in pec_mechs.keys():
        df[name] = cond_params[name]

    assert len(comp.input_ports) > 0

    return comp, df


# Let's make things reproducible
pnl_seed = 0
set_global_seed(pnl_seed)
trial_seq_seed = 0

# High-level parameters the impact performance of the test
num_trials = 150
time_step_size = 0.01
num_estimates = 10000

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
taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(240, 0.5, seed=trial_seq_seed)
taskTrain = taskTrain[0:num_trials]
stimulusTrain = stimulusTrain[0:num_trials]
cueTrain = cueTrain[0:num_trials]
correctResponse = correctResponse[0:num_trials]

# CSI is in terms of time steps, we need to scale by ten because original code
# was set to run with timestep size of 0.001
cueTrain = [c / 10.0 for c in cueTrain]

# We will generate a dataset that comprises two different conditions. Each condition will have a different threshold.
# Randomly select which trials will be in each condition uniformly.
rng = np.random.default_rng(pnl_seed)
threshold = rng.choice([0.3, 0.7], size=num_trials, replace=True)

# Run
_, data_to_fit = run_stab_flex_cond(
    taskTrain,
    stimulusTrain,
    cueTrain,
    correctResponse,
    **{**sf_params, 'threshold': threshold}
)

# Turn our trial-wise threshold into a condition
data_to_fit['condition'] = np.where(data_to_fit['threshold'] == 0.3, 'threshold=0.3', 'threshold=0.7')
data_to_fit.drop(columns=['threshold'], inplace=True)

#%%
# Create a parameter estimation composition to fit the data we just generated and hopefully recover the
# parameters of the composition.
comp = make_stab_flex(**sf_params)

controlModule = get_node(comp, "Task Activations [Act1, Act2]")
congruenceWeighting = get_node(comp, "Automaticity-weighted Stimulus Input [w*S1, w*S2]")
decisionMaker = get_node(comp, "DDM")
decisionGate = get_node(comp, "DECISION_GATE")
responseGate = get_node(comp, "RESPONSE_GATE")

fit_parameters = {
    ("gain", controlModule): np.linspace(1.0, 10.0, 1000),  # Gain
    ("slope", congruenceWeighting): np.linspace(0.0, 0.1, 1000),  # Automaticity
    ("threshold", decisionMaker): np.linspace(0.01, 0.5, 1000),  # Threshold
    ("non_decision_time", decisionMaker): np.linspace(0.1, 0.4, 1000),  # Threshold
}

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp,
    parameters=fit_parameters,
    depends_on={("threshold", decisionMaker): 'condition'},
    outcome_variables=[
        decisionGate.output_ports[0],
        responseGate.output_ports[0],
    ],
    data=data_to_fit,
    optimization_function='differential_evolution',
    num_estimates=num_estimates,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)

inputs = make_input_dict(comp, taskTrain, stimulusTrain, cueTrain, correctResponse)

print("Running the PEC")
ret = pec.run(inputs=inputs)
optimal_parameters = pec.optimized_parameter_values

# Print the recovered parameters.
records = []
for (name, mech), recovered_param in zip(fit_parameters.keys(), optimal_parameters):

    if name == "slope":
        true_param = sf_params['automaticity']
    else:
        true_param = sf_params[name]

    percent_error = 100.0 * (abs(true_param - recovered_param) / true_param)
    records.append((name, mech.name, true_param, recovered_param, percent_error))
df = pd.DataFrame(records, columns=['Parameter', 'Component', 'Value', 'Recovered Value', 'Percent Error'])
print(df)
