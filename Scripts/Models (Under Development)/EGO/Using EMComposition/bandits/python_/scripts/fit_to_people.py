import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm

import pandas as pd
import lmfit

from two_step_plot import compute_stay_stats_from_df
from src.run_probabilistic import run_model_choices # two-step task
from src.run import run # revaluation task
from defaults import PARAMS as default_params

# Mommenejad et al, 2017
REWARD_TARGET = .5199
TRANSITION_TARGET = .4503

# Gillan et al, 2016
REWARDED_COMMON_TARGET = 0.8672
REWARDED_RARE_TARGET = 0.8219
UNREWARDED_COMMON_TARGET = 0.6976
UNREWARDED_RARE_TARGET = 0.7582

reval_data = {'reval_scores_reward': REWARD_TARGET,
              'reval_scores_transition': TRANSITION_TARGET}

two_step_data = {'RC':REWARDED_COMMON_TARGET,
                 'RR':REWARDED_RARE_TARGET,
                 'UC':UNREWARDED_COMMON_TARGET,
                 'UR':UNREWARDED_RARE_TARGET}

def two_step_loss(pred,data):
    data_rewarded_diff = (data['RC'] - data['RR'])
    pred_rewarded_diff = (pred['RC'] - pred['RR'])
    pred_unrewarded_diff = (pred['UC'] - pred['UR'])
    data_unrewarded_diff = (data['UC'] - data['UR'])

    loss =(
            ((pred_rewarded_diff - data_rewarded_diff)/data_rewarded_diff)**2 + # diff of diffs for rewarded
            ((pred_unrewarded_diff - data_unrewarded_diff)/data_unrewarded_diff)**2 + # diff of diffs for unrewarded
            ((pred['RC'] - data['RC'])/data['RC'])**2 + # rewarded common
            ((pred['UC'] - data['UC'])/data['UC'])**2 + # unrewarded common
            ((pred['RR'] - data['RR'])/data['RR'])**2 + # rewarded rare
            ((pred['UC'] - data['UC'])/data['UC'])**2)  # unrewarded rare
    return loss

def reval_loss(pred,data):
    loss =(
        ((pred['reval_scores_reward'] - pred['reval_scores_transition']) - (data['reval_scores_reward'] - data['reval_scores_transition']))**2 + # diff of diffs
        (pred['reval_scores_reward'] - data['reval_scores_transition'])**2 + # reward revaluation
        (pred['reval_scores_transition'] - data['reval_scores_transition'])**2 # transition revaluation
    )
    return loss

# def combined_loss(pred2step, data2step, predreval,datareval):
#     loss =(
#         two_step_loss(pred2step,data2step) +
#         reval_loss(predreval,datareval)
#     )
#     return loss

def _run_seed(seed, sir, trw, args):
    trial_log = run_model_choices(state_integration_rate=sir,
                                  time_retrieval_weight=trw,
                                  **{**args, 'seed': seed})
    trial_log_df = pd.DataFrame(trial_log)
    return compute_stay_stats_from_df(trial_log_df)

def two_step_residual(fit_params,data,n_seeds):
    from concurrent.futures import ProcessPoolExecutor
    sir = fit_params['state_integration_rate'].value
    trw = fit_params['time_retrieval_weight'].value

    args = {name: param.value for name, param in fit_params.items() if not param.vary}

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_run_seed, range(n_seeds), [sir]*n_seeds, [trw]*n_seeds, [args]*n_seeds))

    stay_df = pd.DataFrame(results)
    stay_means = stay_df.mean()
    return two_step_loss(stay_means,data)

def reval_residual(fit_params,data):
    sir = fit_params['state_integration_rate'].value
    trw = fit_params['time_retrieval_weight'].value

    reval_results = run(
        state_integration_rate=sir,
        time_retrieval_weight=trw,
        time_noise=fit_params['time_drift_noise'].value,
        ego_softmax_temperature=fit_params['ego_temperature'].value,
        ego_softmax_threshold=fit_params['ego_threshold'].value,
    )
    return reval_loss(reval_results,data)

def residual(fit_params,two_step_data,reval_data,n_seeds,which_task='both'):

    if which_task=='both':
        tsres = two_step_residual(fit_params,two_step_data,n_seeds)
        rres = reval_residual(fit_params,reval_data)
        return tsres + rres
    elif which_task=='two-step':
        return two_step_residual(fit_params,two_step_data,n_seeds)
    elif which_task=='reval':
        return reval_residual(fit_params,reval_data)

if __name__ == '__main__':
    which_task = 'two-step'
    n_seeds = 100

    fit_params = lmfit.Parameters()
    fit_params.add('state_integration_rate', value=0.5, min=0, max=1)
    fit_params.add('time_retrieval_weight', value=0.5, min=0, max=1)
    fit_params.add('n_base_trials', value=default_params['n_base_trials'], vary=False)
    fit_params.add('common_prob', value=default_params['common_prob'], vary=False)
    fit_params.add('sigma', value=default_params['sigma'], vary=False)
    fit_params.add('lo', value=default_params['lo'], vary=False)
    fit_params.add('hi', value=default_params['hi'], vary=False)
    fit_params.add('choice_temperature', value=default_params['choice_temperature'], min=1, max=10) #vary=False)
    fit_params.add('choice_bias', value=default_params['choice_bias'], min=0, max=1) #vary=False)
    fit_params.add('time_drift_noise', value=default_params['time_drift_noise'], vary=False)
    fit_params.add('ego_temperature', value=default_params['ego_temperature'], vary=False)
    fit_params.add('ego_threshold', value=default_params['ego_threshold'], vary=False)

    pbar = tqdm()
    def callback(params,*args,**kwargs):
        pbar.update(1)
        return(residual(params,*args,**kwargs))

    result = lmfit.minimize(callback, fit_params, method='powell', kws={'two_step_data':two_step_data, 'reval_data':reval_data,
                                                       'n_seeds': n_seeds, 'which_task':which_task})

    result.params.pretty_print()

