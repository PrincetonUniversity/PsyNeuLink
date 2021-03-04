#%%
import numpy as np
import psyneulink as pnl

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import wfpt

from psyneulink.core.components.functions.fitfunctions import simulation_likelihood


def ddm_pdf_analytical(drift_rate, threshold, noise, starting_point, non_decision_time, time_step_size=0.01):
    from ddm import Model
    from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPoint
    from ddm.functions import display_model

    model = Model(name='Simple model',
                  drift=DriftConstant(drift=drift_rate),
                  noise=NoiseConstant(noise=noise),
                  bound=BoundConstant(B=threshold),
                  IC=ICPoint(x0=starting_point),
                  overlay=OverlayNonDecision(nondectime=non_decision_time),
                  dx=.001, dt=time_step_size, T_dur=3)
    display_model(model)
    s = model.solve()

    return model.t_domain(), s.pdf_corr(), s.pdf_err()


def ddm_pdf_simulate(drift_rate=0.75, threshold=1.0, noise=0.1, starting_point=0.0, non_decision_time=0.0,
                     time_step_size=0.01, num_samples=1000000, use_pnl=True, rt_space=None):

    if use_pnl:
        decision = pnl.DDM(function=pnl.DriftDiffusionIntegrator(starting_value=0.1234),
                           output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME],
                           name='DDM')

        comp = pnl.Composition()
        comp.add_node(decision)

        context = pnl.Context()
        decision.function.parameters.rate.set(drift_rate, context)
        decision.function.parameters.noise.set(noise, context)
        decision.function.parameters.threshold.set(threshold, context)
        decision.function.parameters.time_step_size.set(time_step_size, context)
        decision.function.parameters.starting_value.set(np.array([starting_point]), context)
        decision.function.parameters.time_step_size.set(time_step_size, context)
        decision.function.parameters.non_decision_time.set(non_decision_time, context)

        input = np.ones((1, 1))

        comp.run(inputs={decision: input},
                 num_trials=num_samples * len(input),
                 execution_mode=pnl.ExecutionMode.LLVMRun,
                 context=context)

        results = np.squeeze(np.array(comp.results))
        rts = np.array(np.vsplit(results, len(input)))

    else:
        rts = wfpt.simulate_wfpt(starting_point * np.ones(num_samples),
                                 non_decision_time * np.ones(num_samples),
                                 drift_rate * np.ones(num_samples),
                                 threshold * np.ones(num_samples),
                                 dt=time_step_size)
        rts = np.expand_dims(np.column_stack((np.sign(rts)*threshold, np.abs(rts))), axis=0)

    # Make a histogram
    # hist = bh.Histogram(bh.axis.Boolean(), bh.axis.Regular(int(3 / time_step_size), 0.0, 3.0))
    # hist.fill(rts[:, :, 0].flatten() > 0.0, rts[:, :, 1].flatten())

    if rt_space is None:
        rt_space = np.linspace(0.0, 3.0, 3000)

    df = pd.DataFrame(index=rt_space)

    df[f'Correct KDE (dt={time_step_size})'] = simulation_likelihood(rts,
                                                                    categorical_dims=np.array([True, False]),
                                                                    combine_trials=True,
                                                                    exp_data=np.c_[
                                                                        threshold * np.ones(len(rt_space)), rt_space])
    df[f'Error KDE (dt={time_step_size})'] = simulation_likelihood(rts,
                                                                  categorical_dims=np.array([True, False]),
                                                                  combine_trials=True,
                                                                  exp_data=np.c_[
                                                                      -threshold * np.ones(len(rt_space)), rt_space])

    #df[f'Correct Histogram (dt={time_step_size})'] = (hist[True, :] / hist.sum(flow=True) / time_step_size).view()
    #df[f'Error Histogram (dt={time_step_size})'] = (hist[False, :] / hist.sum(flow=True) / time_step_size).view()

    return df


def ddm_plot_check():
    ddm_params = dict(starting_point=0.1, drift_rate=0.3, noise=1.0, threshold=0.6, non_decision_time=0.8)

    # from numpy.random import rand
    # pd.DataFrame({
    #     'drift_rate': (rand() - .5) * 8,
    #     'non_decision_time': 0.2 + rand() * 0.3,
    #     'threshold': 0.5 + rand() * 1.5,
    #     'noise': 1.0
    # })

    NUM_SAMPLES = 100000

    rt_space = np.linspace(0.0, 3.0, 30000)

    # Get the analytical
    t_domain, pdf_corr, pdf_err = ddm_pdf_analytical(**ddm_params)

    # Interpolate to common rt space
    from scipy.interpolate import interpn
    anal_df = pd.DataFrame(index=rt_space)
    anal_df[f"Correct Analytical"] = interpn((t_domain,), pdf_corr, rt_space,
                                             method='linear', bounds_error=False, fill_value=1e-10)
    anal_df[f"Error Analytical"] = interpn((t_domain,), pdf_err, rt_space,
                                           method='linear', bounds_error=False, fill_value=1e-10)

    # Navarro and Fuss solution
    # p_err = np.array(
    #     [wfpt.wfpt_logp(t, 0, starting_point, non_decision_time, drift_rate, threshold, eps=1e-10) for t in
    #      model_t_domain[1:]])
    # p_corr = np.array(
    #     [wfpt.wfpt_logp(t, 1, starting_point, non_decision_time, drift_rate, threshold, eps=1e-10) for t in
    #      model_t_domain[1:]])

    df = pd.concat([
        anal_df,
        ddm_pdf_simulate(**ddm_params, time_step_size=0.01, use_pnl=True, num_samples=NUM_SAMPLES, rt_space=rt_space),
        ddm_pdf_simulate(**ddm_params, time_step_size=0.001, use_pnl=True, num_samples=NUM_SAMPLES, rt_space=rt_space),
        ddm_pdf_simulate(**ddm_params, time_step_size=0.0001, use_pnl=True, num_samples=NUM_SAMPLES, rt_space=rt_space),
    ])

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    #df = df.loc[:, ~df.columns.str.contains('Histogram')]
    sns.lineplot(data=df.filter(regex='Correct'), ax=axes[0])
    sns.lineplot(data=df.filter(regex='Error'), ax=axes[1])
    plt.show()

    plt.savefig(f"{'_'.join([f'{p}={v}' for p,v in ddm_params.items()])}.png")


ddm_plot_check()

