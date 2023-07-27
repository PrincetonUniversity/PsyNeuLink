#%%
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed

# # Let's make things reproducible
set_global_seed(0)

# High-level parameters the impact performance of the test
num_trials = 50
time_step_size = 0.01
num_estimates = 40000

ddm_params = dict(
    starting_value=0.0,
    rate=0.3,
    noise=1.0,
    threshold=0.6,
    non_decision_time=0.15,
    time_step_size=time_step_size,
)

# Create a simple one mechanism composition containing a DDM in integrator mode.
decision = pnl.DDM(
    function=pnl.DriftDiffusionIntegrator(**ddm_params),
    output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
    name="DDM",
)

comp = pnl.Composition(pathways=decision)

# Let's generate an "experimental" dataset to fit. This is a parameter recovery test
# Lets make 10% of the trials have a positive stimulus drift rate, and the other 90%
# have a negative stimulus drift rate.
# trial_inputs = np.ones((num_trials, 1))
rng = np.random.default_rng(12345)
trial_inputs = rng.choice(
    [5.0, -5.0], size=(num_trials, 1), p=[0.10, 0.9], replace=True
)

# Make the first and last input positive for sure. This helps make sure inputs are really getting
# passed to the composition correctly during parameter fitting, and we aren't just getting a single
# trials worth of a cached input.
trial_inputs[0] = np.abs(trial_inputs[0])
trial_inputs[-1] = np.abs(trial_inputs[-1])

inputs_dict = {decision: trial_inputs}

# Store the results of this "experiment" as a numpy array. This should be a
# 2D array of shape (len(input), 2). The first column being a discrete variable
# specifying whether the upper or lower decision boundary is reached and the second column is the
# reaction time. We will put the data into a pandas DataFrame, this makes it
# easier to specify which columns in the data are categorical or not.

# Run the composition to generate some data to fit
comp.run(inputs=inputs_dict)
results = comp.results

data_to_fit = pd.DataFrame(
    np.squeeze(np.array(results)), columns=["decision", "response_time"]
)
data_to_fit["decision"] = data_to_fit["decision"].astype("category")

# Create a parameter estimation composition to fit the data we just generated and hopefully recover the
# parameters of the DDM.

fit_parameters = {
    ("rate", decision): np.linspace(-0.5, 0.5, 1000),
    ("threshold", decision): np.linspace(0.5, 1.0, 1000),
    ('non_decision_time', decision): np.linspace(0.0, 1.0, 1000),
}

#%%
pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=[comp],
    parameters=fit_parameters,
    outcome_variables=[
        decision.output_ports[pnl.DECISION_OUTCOME],
        decision.output_ports[pnl.RESPONSE_TIME],
    ],
    data=data_to_fit,
    optimization_function="differential_evolution",
    num_estimates=num_estimates,
    initial_seed=42,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)
ret = pec.run(inputs={comp: trial_inputs})
optimal_parameters = pec.optimized_parameter_values

# Check that the parameters are recovered and that the log-likelihood is correct, set the tolerance pretty high,
# things are noisy because of the low number of trials and estimates.
assert np.allclose(
    optimal_parameters,
    [ddm_params["rate"], ddm_params["threshold"], ddm_params["non_decision_time"]],
    atol=0.1,
)

records = []
for (name, mech), recovered_param in zip(fit_parameters.keys(), optimal_parameters):
    percent_error = 100.0 * (abs(ddm_params[name] - recovered_param) / ddm_params[name])
    records.append((name, mech.name, ddm_params[name], recovered_param, percent_error))
df = pd.DataFrame(records, columns=['Parameter', 'Component', 'Value', 'Recovered Value', 'Percent Error'])
print(df)
