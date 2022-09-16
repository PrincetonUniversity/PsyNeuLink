import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.components.functions.fitfunctions import MaxLikelihoodEstimator


# High-level parameters the impact performance of the test
# num_estimates = 10000
# num_trials = 50
# time_step_size = 0.001
num_trials = 20
time_step_size = 0.1
num_estimates = 100

ddm_params = dict(starting_value=0.0, rate=0.3, noise=1.0,
                  threshold=0.6, non_decision_time=0.15, time_step_size=time_step_size)

# Create a simple one mechanism composition containing a DDM in integrator mode.
decision = pnl.DDM(function=pnl.DriftDiffusionIntegrator(**ddm_params),
                   output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME],
                   name='DDM')

comp = pnl.Composition(pathways=decision)

# Let's generate an "experimental" dataset to fit. This is a parameter recovery test
# The input will be 20 trials of the same constant stimulus drift rate of 1
# input = np.array([[1, 1, 0.3, 0.3, 1, 1, 0.3, 1, 0.3, 1, 0.3, 0.3, 0.3, 1, 1, 0.3, 0.3, 1, 1, 1]]).transpose()
input = np.ones((num_trials, 1))
inputs_dict = {decision: input}

# Run the composition to generate some data to fit
# comp.run(inputs=inputs_dict, num_trials=len(input))
#
# Store the results of this "experiment" as a numpy array. This should be a
# 2D array of shape (len(input), 2). The first column being a discrete variable
# specifying the upper or lower decision boundary and the second column is the
# reaction time. We will put the data into a pandas DataFrame, this makes it
# easier to specify which columns in the data are categorical or not.

# Load the results from a previous run rather than generate them like we were
# doing above.

results = pd.read_csv('ddm_exp_data.csv')

data_to_fit = pd.DataFrame(np.squeeze(np.array(results)),
                           columns=['decision', 'rt'])
data_to_fit['decision'] = pd.Categorical(data_to_fit['decision'])

# Create a parameter estimation composition to fit the data we just generated and hopefully recover the
# parameters of the DDM.

fit_parameters = {
    ('rate', decision): np.linspace(0.0, 1.0, 1000),
    ('threshold', decision): np.linspace(0.0, 1.0, 1000),
    # ('starting_value', decision): np.linspace(0.0, 0.9, 1000),
    # ('non_decision_time', decision): np.linspace(0.0, 1.0, 1000),
}

pec = pnl.ParameterEstimationComposition(name='pec',
                                         nodes=[comp],
                                         parameters=fit_parameters,
                                         outcome_variables=[decision.output_ports[pnl.DECISION_VARIABLE],
                                                            decision.output_ports[pnl.RESPONSE_TIME]],
                                         data=data_to_fit.iloc[0:num_trials, :],
                                         optimization_function=MaxLikelihoodEstimator,
                                         num_estimates=num_estimates,
                                         num_trials_per_estimate=len(input),
                                         )

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)
ret = pec.run(inputs=inputs_dict, num_trials=len(input))

# Check that the parameters are recovered and that the log-likelihood is correct
assert np.allclose(pec.controller.optimal_parameters, [0.3, 0.15], atol=0.2)
