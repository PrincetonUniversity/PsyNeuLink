import pandas as pd
import numpy as np
import psyneulink as pnl

def _run_ddm_with_params(
    starting_value,
    rate,
    noise,
    threshold,
    non_decision_time,
    time_step_size,
    trial_inputs,
):
    """Create a composition with DDM and run it with the given parameters."""

    # Create a simple one mechanism composition containing a DDM in integrator mode.
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=starting_value,
            rate=rate,
            noise=noise,
            threshold=threshold,
            non_decision_time=non_decision_time,
            time_step_size=time_step_size,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )

    comp = pnl.Composition(pathways=decision)

    # Run the composition to generate some data to fit
    comp.run(inputs={decision: trial_inputs})
    results = comp.results

    data_to_fit = pd.DataFrame(
        np.squeeze(np.array(results)), columns=["decision", "response_time"]
    )
    data_to_fit["decision"] = data_to_fit["decision"].astype("category")

    return comp, data_to_fit


# High-level parameters the impact performance of the test
num_trials = 50
time_step_size = 0.01
num_estimates = 1000

# Let's generate an "experimental" dataset to fit. This is a parameter recovery test
# Lets make 10% of the trials have a positive stimulus drift rate, and the other 90%
# have a negative stimulus drift rate.
rng = np.random.default_rng(12345)
trial_inputs = rng.choice(
    [5.0, -5.0], size=(num_trials, 1), p=[0.10, 0.9], replace=True
)

# Make the first and last input positive for sure. This helps make sure inputs are really getting
# passed to the composition correctly during parameter fitting, and we aren't just getting a single
# trials worth of a cached input.
trial_inputs[0] = np.abs(trial_inputs[0])
trial_inputs[-1] = np.abs(trial_inputs[-1])

ddm_params = dict(
    starting_value=0.0,
    rate=0.3,
    noise=1.0,
    threshold=0.6,
    non_decision_time=0.15,
    time_step_size=time_step_size,
)

# We will generate a dataset that comprises two different conditions. Each condition will have a different
# threshold.
params_cond1 = dict(
    threshold=0.7,
)

params_cond2 = dict(
    threshold=0.3,
)

comp, data_cond1 = _run_ddm_with_params(**{**ddm_params, **params_cond1}, trial_inputs=trial_inputs)
_, data_cond2 = _run_ddm_with_params(**{**ddm_params, **params_cond2}, trial_inputs=trial_inputs)

# Combine the data from the two conditions
data_cond1['condition'] = 'cond_t=0.7'
data_cond2['condition'] = 'cond_t=0.3'
data_to_fit = pd.concat([data_cond1, data_cond2])

# Add the inputs as columns to the data temporarily so we can shuffle the data and shuffle the inputs together
data_to_fit['inputs'] = np.concatenate([trial_inputs, trial_inputs])

# Shuffle the data, seed is set for reproducibility
data_to_fit = data_to_fit.sample(frac=1, random_state=42)

# Extract the shuffled inputs
trial_inputs = data_to_fit['inputs'].to_numpy().reshape(-1, 1)
data_to_fit = data_to_fit.drop(columns='inputs')

fit_parameters = {
    ("rate", comp.nodes['DDM']): np.linspace(-0.5, 0.5, 1000),
    ("non_decision_time", comp.nodes['DDM']): np.linspace(0.0, 1.0, 1000),
    ("threshold", comp.nodes['DDM']): np.linspace(0.1, 1.0, 1000),
}

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=[comp],
    parameters=fit_parameters,
    depends_on={("threshold", comp.nodes['DDM']): 'condition'},
    outcome_variables=[
        comp.nodes['DDM'].output_ports[pnl.DECISION_OUTCOME],
        comp.nodes['DDM'].output_ports[pnl.RESPONSE_TIME],
    ],
    data=data_to_fit,
    optimization_function=pnl.PECOptimizationFunction(
        method="differential_evolution",
    ),
    num_estimates=num_estimates,
    initial_seed=42,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)
pec.run(inputs={comp: trial_inputs})

records = []
params = {
    'DDM.rate': ddm_params['rate'],
    'DDM.non_decision_time': ddm_params['non_decision_time'],
    'DDM.threshold[cond_t=0.3]': 0.3,
    'DDM.threshold[cond_t=0.7]': 0.7
}
for i, (name, recovered_param) in enumerate(pec.optimized_parameter_values.items()):
    percent_error = 100.0 * (abs(params[name] - recovered_param) / params[name])
    records.append((name, params[name], recovered_param, percent_error))
df = pd.DataFrame(records, columns=['Parameter', 'Value', 'Recovered Value', 'Percent Error'])
print(df)

