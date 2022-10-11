#%%
import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed
from psyneulink.core.components.functions.fitfunctions import MaxLikelihoodEstimator

# Let's make things reproducible
seed = 0
np.random.seed(seed)
set_global_seed(seed)

# High-level parameters the impact performance of the test
num_trials = 2
time_step_size = 0.01
num_estimates = 4

ddm_params = dict(starting_value=0.0, rate=0.3, noise=1.0,
                  threshold=0.6, non_decision_time=0.15, time_step_size=time_step_size)

# Create a simple one mechanism composition containing a DDM in integrator mode.
decision = pnl.DDM(function=pnl.DriftDiffusionIntegrator(**ddm_params),
                   output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
                   name='DDM')

comp = pnl.Composition(pathways=decision)

# Let's generate an "experimental" dataset to fit. This is a parameter recovery test
# The input will be num_trials trials of the same constant stimulus drift rate of 1
# input = np.concatenate((np.repeat(-30.0, 30), np.repeat(30.0, 30)))[:, None]
trial_inputs = np.ones((num_trials, 1))
inputs_dict = {decision: trial_inputs}

# Store the results of this "experiment" as a numpy array. This should be a
# 2D array of shape (len(input), 2). The first column being a discrete variable
# specifying whether the upper or lower decision boundary is reached and the second column is the
# reaction time. We will put the data into a pandas DataFrame, this makes it
# easier to specify which columns in the data are categorical or not.

# Run the composition to generate some data to fit
comp.run(inputs=inputs_dict)
results = comp.results

data_to_fit = pd.DataFrame(np.squeeze(np.array(results)), columns=['decision', 'response_time'])
data_to_fit['decision'] = data_to_fit['decision'].astype('category')

# Create a parameter estimation composition to fit the data we just generated and hopefully recover the
# parameters of the DDM.

fit_parameters = {
    ('rate', decision): np.linspace(0.0, 0.4, 1000),
    ('threshold', decision): np.linspace(0.5, 1.0, 1000),
    # ('non_decision_time', decision): np.linspace(0.0, 1.0, 1000),
}

pec = pnl.ParameterEstimationComposition(name='pec',
                                         nodes=[comp],
                                         parameters=fit_parameters,
                                         outcome_variables=[decision.output_ports[pnl.DECISION_OUTCOME],
                                                            decision.output_ports[pnl.RESPONSE_TIME]],
                                         data=data_to_fit,
                                         optimization_function=MaxLikelihoodEstimator(),
                                         num_estimates=num_estimates,
                                         num_trials_per_estimate=len(trial_inputs),
                                         )

# pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)
ll, sim_data = pec.log_likelihood(0.3, 0.6, inputs=inputs_dict)
# ret = pec.run(inputs=inputs_dict, num_trials=len(trial_inputs))

# Check that the parameters are recovered and that the log-likelihood is correct
# assert np.allclose(pec.controller.optimal_parameters, [0.3, 0.6], atol=0.1)
