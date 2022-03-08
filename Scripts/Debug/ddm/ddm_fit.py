#%%
import numpy as np
import pandas as pd
import psyneulink as pnl

from psyneulink.core.components.functions.fitfunctions import make_likelihood_function, \
    MaxLikelihoodEstimator

ddm_params = dict(starting_value=0.0, rate=0.3, noise=1.0,
                  threshold=0.6, non_decision_time=0.15, time_step_size=0.01)

# Create a simple one mechanism composition containing a DDM in integrator mode.
decision = pnl.DDM(function=pnl.DriftDiffusionIntegrator(**ddm_params),
                   output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME],
                   name='DDM')

comp = pnl.Composition(pathways=decision)

#%%

# Lets generate an "experimental" dataset to fit. This is a parameter recovery test
# The input will be 500 trials of the same constant stimulus drift rate of 1
input = np.ones((500, 1))
inputs_dict = {decision: input}

# Run the composition to generate some data to fit
comp.run(inputs=inputs_dict,
         num_trials=len(input),
         execution_mode=pnl.ExecutionMode.LLVMRun)

# Store the results of this "experiment" as a numpy array. This should be a
# 2D array of shape (len(input), 2). The first column being a discrete variable
# specifying the upper or lower decision boundary and the second column is the
# reaction time. We will put the data into a pandas DataFrame, this makes its
# easier to specify which columns in the data are categorical or not.
data_to_fit = pd.DataFrame(np.squeeze(np.array(comp.results)),
                           columns=['decision', 'rt'])
data_to_fit['decision'] = pd.Categorical(data_to_fit['decision'])

#%%

# Create a likelihood function from the composition itself, this is done
# using probability density approximation via kernel density estimation.
likelihood, param_map = comp.make_likelihood_function(
    fit_params=[decision.function.parameters.rate,
                decision.function.parameters.starting_value,
                decision.function.parameters.non_decision_time],
    inputs=inputs_dict,
    data_to_fit=data_to_fit,
    num_sims_per_trial=100,
    combine_trials=True)

params_to_recover = {k: ddm_params[k] for k in param_map.values()}
print(f"Parameters to recover: {params_to_recover}")
print(f"Data Neg-Log-Likelihood: {-likelihood(**params_to_recover)}")

mle = MaxLikelihoodEstimator(log_likelihood_function=likelihood,
                             fit_params_bounds={
                                'rate':              (0.0, 1.0),
                                'starting_value':    (0.0, 0.9),
                                'non_decision_time': (0.0, 1.0),
                             })

fit_results = mle.fit(display_iter=True, save_iterations=True)