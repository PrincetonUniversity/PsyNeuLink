#%%
import numpy as np
import pandas as pd
import math
import taichi as ti
import taichi_glsl as ts
import taichi_utils as tu

from ddm import Model
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPoint
from ddm.functions import display_model
from psyneulink.core.components.functions.fitfunctions import simulation_likelihood, make_likelihood_function, MaxLikelihoodEstimator


ti.init(arch=ti.gpu)

#%%
num_simulations = 1000000
rt = ti.field(ti.f32, num_simulations)
decision = ti.field(ti.i32, num_simulations)
max_time = 3.0


def ddm_pdf_analytical(drift_rate, threshold, starting_value, non_decision_time, noise=1.0, time_step_size=0.001):

    model = Model(name='Simple model',
                  drift=DriftConstant(drift=drift_rate),
                  noise=NoiseConstant(noise=noise),
                  bound=BoundConstant(B=threshold),
                  IC=ICPoint(x0=starting_value),
                  overlay=OverlayNonDecision(nondectime=non_decision_time),
                  dx=.001, dt=time_step_size, T_dur=3)
    s = model.solve()

    return model.t_domain(), s.pdf_corr(), s.pdf_err()


@ti.func
def ddm_time_step(prev_value, drift_rate, time_step_size):
    return prev_value + (tu.rand_normal() + drift_rate * time_step_size) * ti.sqrt(time_step_size)


@ti.func
def simulate_ddm(starting_value, non_decision_time, drift_rate, threshold, time_step_size):
    particle = starting_value
    t = 0
    while abs(particle) < threshold:
        particle = ddm_time_step(particle, drift_rate, time_step_size)
        t = t + 1

    rt = (non_decision_time + t * time_step_size)
    decision = 1
    if particle < -threshold:
        decision = 0

    return rt, decision


@ti.kernel
def simulate_many_ddms(starting_value: ti.f32,
                       non_decision_time: ti.f32,
                       drift_rate: ti.f32,
                       threshold: ti.f32,
                       time_step_size: ti.f32):
    for i in rt:
        rt[i], decision[i] = simulate_ddm(starting_value, non_decision_time,
                                          drift_rate, threshold, time_step_size)


@ti.func
def lca_time_step(prev_value, prev_value_f, stimulus, gamma, leak, time_step_size):
    drift = time_step_size * (stimulus - leak * prev_value + gamma @ prev_value_f)
    return prev_value + (drift + tu.rand_normal2()) * ti.sqrt(time_step_size)


@ti.func
def simulate_lca(stimulus, competition, self_excitation,
                 leak, gain, starting_value,
                 threshold, non_decision_time, time_step_size):
    gamma = ti.Matrix([[competition, self_excitation], [self_excitation, competition]], dt=ti.f32)

    pre_activation = ti.Vector([starting_value, starting_value], dt=ti.f32)
    particle = tu.relu(pre_activation)
    t = 0
    while particle.max() < threshold and t * time_step_size < max_time:
        pre_activation = lca_time_step(pre_activation, particle, stimulus, gamma, leak, time_step_size)
        particle = tu.relu(pre_activation)
        t = t + 1

    rt = (non_decision_time + t * time_step_size)

    # If the simulation exceeds the max time, we terminated early, set RT to negative to signal this
    # is a failed simulation
    if rt >= max_time:
        rt = -1

    # Figure out which threshold was crossed.
    decision = 0
    if particle[0] >= threshold:
        decision = 0

    if particle[1] >= threshold:
        decision = 1

    # If multiple dimensions crossed the threshold at the same time then this is a failure case
    # as well. With infinite precision this won't happen.
    if particle[0] >= threshold and particle[1] >= threshold:
        rt = -1

    return rt, decision


stimulus = ti.Vector.field(2, dtype=float, shape=())
stimulus[None] = [0.1, 0.2]

@ti.kernel
def simulate_many_lcas(competition: ti.f32,
                       self_excitation: ti.f32,
                       leak: ti.f32,
                       gain: ti.f32,
                       starting_value: ti.f32,
                       threshold: ti.f32,
                       non_decision_time: ti.f32,
                       time_step_size: ti.f32):
    stimulus_vec = stimulus[None]
    for i in rt:
        rt[i], decision[i] = simulate_lca(stimulus_vec, competition, self_excitation,
                                          leak, gain, starting_value,
                                          threshold, non_decision_time, time_step_size)


ddm_params = dict(starting_value=0.0, non_decision_time=0.14,
                  drift_rate=0.1, threshold=0.6, time_step_size=0.001)
lca_params = dict(competition=0.1, self_excitation=0.1,
                  leak=0.1, gain=1.0, starting_value=0.0, threshold=0.08,
                  non_decistion_time=0.3, time_step_size=0.0001)


simulate_many_ddms(*list(ddm_params.values()))
#simulate_many_lcas(*list(lca_params.values()))
rts = rt.to_numpy()
choices = decision.to_numpy()

# import time
# t0 = time.time()
# for i in range(50):
#     simulate_many_lcas(*list(lca_params.values()))
#     rts = rt.to_numpy()
#     choices = decision.to_numpy()
# print(f"Elapsed: {((time.time()-t0)/50.0)*1000} milliseconds")

ti.sync()
rts = rt.to_numpy()
choices = decision.to_numpy()
valid = rts > 0.0
rts = rts[valid]
choices = choices[valid]

# import time
# t0 = time.time()
#
# NUM_TIMES = 50
# for i in range(NUM_TIMES):
#     simulate_many_ddms(*list(ddm_params.values()))
#     rts_np = rts.to_numpy()
#
# print(f"Elapsed: { 1000*((time.time() - t0) / NUM_TIMES)} milliseconds")


#%%
import time
import matplotlib.pyplot as plt
import seaborn as sns
import boost_histogram as bh
import functools
import operator
from psyneulink.core.components.functions.fitfunctions import simulation_likelihood, make_likelihood_function, MaxLikelihoodEstimator

rt_space = np.linspace(0, 3.0, num=3000)

t0 = time.time()
# pdf0 = simulation_likelihood(np.column_stack((choices, rts)), categorical_dims=np.array([True, False]),
#                       exp_data=np.c_[np.zeros(len(rt_space)), rt_space])
# pdf1 = simulation_likelihood(np.column_stack((choices, rts)), categorical_dims=np.array([True, False]),
#                       exp_data=np.c_[np.ones(len(rt_space)), rt_space])

hist = bh.Histogram(bh.axis.Integer(0, 2), bh.axis.Regular(3000, 0.0, 3.0))
hist.fill(choices, rts)
areas = functools.reduce(operator.mul, hist.axes.widths)
density = hist.view() / hist.sum() / areas
pdf0 = density[0, :]
pdf1 = density[1, :]

print(f"Elapsed: { 1000*(time.time() - t0)} milliseconds")

df = pd.DataFrame(index=rt_space)
df[f'Correct KDE (dt={ddm_params["time_step_size"]})'] = pdf1
df[f'Error KDE (dt={ddm_params["time_step_size"]})'] = pdf0


# # Get the analytical
# t_domain, pdf_corr, pdf_err = ddm_pdf_analytical(**ddm_params)
#
# # Interpolate to common rt space
# from scipy.interpolate import interpn
#
# anal_df = pd.DataFrame(index=rt_space)
# anal_df[f"Correct Analytical"] = interpn((t_domain,), pdf_corr, rt_space,
#                                          method='linear', bounds_error=False, fill_value=1e-10)
# anal_df[f"Error Analytical"] = interpn((t_domain,), pdf_err, rt_space,
#                                        method='linear', bounds_error=False, fill_value=1e-10)
#
# df = pd.concat([anal_df, df])

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
sns.lineplot(data=df.filter(regex='Correct'), ax=axes[0])
sns.lineplot(data=df.filter(regex='Error'), ax=axes[1])
plt.show()

#%%





