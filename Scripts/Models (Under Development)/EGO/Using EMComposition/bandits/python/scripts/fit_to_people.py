import pandas as pd
import lmfit

from two_step_plot import compute_stay_stats_from_df
from src.run_probabilistic import run_model_choices # two-step task
from src.run import run # revaluation task
from defaults import PARAMS as default_params

REWARD_TARGET = .5199
TRANSITION_TARGET = .4503

reval_data = {'reval_scores_reward': REWARD_TARGET,
              'reval_scores_transition': TRANSITION_TARGET}

two_step_data = {'RC':0.7,
                 'RR':0.5,
                 'UC':0.4,
                 'UR':0.6}

def two_step_loss(pred,data):
    loss =(
            ((pred['RC'] - pred['RR']) - (data['RC'] - data['RR']))**2 + # diff of diffs for rewarded
            ((pred['UC'] - pred['UR']) - (data['UC'] - data['UR']))**2 + # diff of diffs for unrewarded
            (pred['RC'] - data['RC'])**2 + # rewarded common
            (pred['UC'] - data['UC'])**2 + # unrewarded common
            (pred['RR'] - data['RR'])**2 + # rewarded rare
            (pred['UC'] - data['UC'])**2)  # unrewarded rare
    return loss

def reval_loss(pred,data):
    loss =(
        ((pred['reval_scores_reward'] - pred['reval_scores_transition']) - (data['reval_scores_reward'] - data['reval_scores_transition']))**2 + # diff of diffs
        (pred['reval_scores_reward'] - data['reval_scores_transition'])**2 + # reward revaluation
        (pred['reval_scores_transition'] - data['reval_scores_transition'])**2 # transition revaluation
    )
    return loss

def combined_loss(pred2step, data2step, predreval,datareval):
    loss =(
        two_step_loss(pred2step,data2step) +
        reval_loss(predreval,datareval)
    )
    return loss

def two_step_residual(fit_params,default_params,data,n_seeds):
    sir = fit_params['state_integration_rate'].value
    trw = fit_params['time_retrieval_weight'].value

    args = default_params.copy()

    stay_df = pd.DataFrame()

    for i in range(n_seeds):
        args.update(
            seed=i,
        )
        trial_log = run_model_choices(state_integration_rate=sir,
                                     time_retrieval_weight=trw,
                                     **args)
        trial_log_df = pd.DataFrame(trial_log)
        stay_dict = compute_stay_stats_from_df(trial_log_df)
        stay_df = pd.concat([stay_df, pd.DataFrame([stay_dict])])

    stay_means = stay_df.mean()
    return two_step_loss(stay_means,data)

def reval_residual(fit_params,data):
    sir = fit_params['state_integration_rate'].value
    trw = fit_params['time_retrieval_weight'].value
    args = default_params.copy()

    reval_results = run(state_integration_rate=sir,time_retrieval_weight=trw)
    return reval_loss(reval_results,data)

def residual(fit_params,default_params,two_step_data,reval_data,n_seeds,which_task='both'):

    if which_task=='both':
        tsres = two_step_residual(fit_params,default_params,two_step_data,n_seeds)
        rres = reval_residual(fit_params,reval_data)
        return tsres + rres
    elif which_task=='two-step':
        return two_step_residual(fit_params,default_params,two_step_data,n_seeds)
    elif which_task=='reval':
        return reval_residual(fit_params,reval_data)

which_task = 'both'

fit_params = lmfit.Parameters()
fit_params.add('state_integration_rate', value=0.5, min=0, max=1)
fit_params.add('time_retrieval_weight', value=0.5, min=0, max=1)

result = lmfit.minimize(residual,fit_params, default_params, two_step_data, reval_data, n_seeds, which_task)
result.params.pretty_print()

