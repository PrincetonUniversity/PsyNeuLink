"""Audit saved adaptive fits without changing their search policy or model.

The audit rescores the final search windows on the original reference objective.
The refinement experiment replays saved ask/tell calls exactly, then continues
that same CMA-ES state. Fresh validation seeds never guide either experiment.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
import torch
import psyneulink as pnl

from dawa_adaptive_fit import pooled_scores
from dawa_fitting_budget_study import Study, total_scores
from dawa_pec_fit import LAUNCH, RT_RANGE, load_subject, save_json


def ranking_uncertainty(scores, pair_se, valid, tolerance):
    """Original policy-1 rule, retained to reproduce the historical audit."""
    order = np.flatnonzero(valid)[np.argsort(-scores[valid])]
    selected = max(1, len(scores) // 2)
    if len(order) <= selected:
        return 0., False
    top, bottom = order[:selected], order[selected:]
    gap = scores[top, None] - scores[None, bottom]
    uncertainty = float(np.maximum(0., 2 * pair_se[np.ix_(top, bottom)] - gap).max())
    return uncertainty, uncertainty > tolerance


def batches(rows):
    cursor = 0
    while cursor < len(rows):
        count = rows[cursor]['batch_size']
        group = rows[cursor:cursor + count]
        if len(group) != count:
            raise ValueError('Incomplete saved population')
        yield group
        cursor += count


class Experiment:
    def __init__(self, args):
        self.args = args
        self.manifest = json.loads((args.run / 'manifest.json').read_text())
        self.result = json.loads((args.run / 'recovery.json').read_text())
        if args.mode != 'perturb' and self.result.get('adaptive', {}).get('policy_version', 1) != 1:
            raise ValueError('Historical audit/replay supports adaptive policy 1 only')
        self.rows = [json.loads(line) for line in (args.run / 'evaluations.jsonl').read_text().splitlines()]
        self.names = list(self.manifest['bounds'])
        frame = load_subject(args.run / 'synthetic_subject.csv', self.manifest['arguments']['subject'])
        self.study = Study(frame, self.manifest['arguments']['max_steps'])
        self.seed = self.manifest['arguments']['simulation_seed']
        self.reference_n = self.manifest['arguments']['estimates']
        self.calls = self.histories = 0
        for seed in args.validation_seeds:
            if seed in {self.seed, self.manifest['arguments']['data_seed']}:
                raise ValueError('Validation must use independent seeds')
        self.started = time.perf_counter()
        # Independent plan construction must reproduce the saved reference.
        actual = self.score([list(self.result['fitted'].values())])[0]
        np.testing.assert_allclose(actual, self.result['best_training_log_likelihood'], rtol=0, atol=1e-5)

    def density(self, parameters, estimates, seed):
        self.calls += 1
        self.histories += len(parameters) * estimates
        return self.study.plan.histogram_likelihood(
            self.study.inputs, self.study.parameters(parameters), estimates, data=self.study.observed,
            categorical_dims=[0], bins=100, bin_range=[RT_RANGE], smoothing_sigma=.5,
            pseudocount=self.manifest['arguments']['pseudocount'] * estimates / self.reference_n,
            categorical_cardinalities=[2], seed=seed, invalid_candidates='nan', triton_launch_options=LAUNCH,
        )[:, 0].astype(np.float64)

    def score(self, parameters, *, seed=None):
        density = self.density(parameters, self.reference_n, self.seed if seed is None else seed)
        valid = np.isfinite(density).all(-1)
        density[~valid] = 1.
        scores = total_scores(density, self.study.mask)
        scores[~valid] = -1.e10
        return scores

    def validate(self, labeled):
        labels = list(labeled)
        records = []
        for seed in self.args.validation_seeds:
            values = self.score(list(labeled.values()), seed=seed)
            records.append({'seed': seed, **dict(zip(labels, map(float, values), strict=True))})
        return records

    def save(self, value):
        save_json(self.args.output / 'diagnosis.json', {
            'mode': self.args.mode, 'source_run': str(self.args.run), 'gpu': torch.cuda.get_device_name(),
            'observations_sha256': self.manifest['observations_sha256'],
            'reference_seed': self.seed, 'reference_estimates': self.reference_n,
            'elapsed_seconds': time.perf_counter() - self.started,
            'sampling_calls': self.calls, 'started_histories': self.histories,
            **value,
        })


def audit(experiment):
    e = experiment
    checkpoints = e.result['adaptive']['checkpoints']
    search = [r for r in e.rows if r['phase'] == 'adaptive_search']
    report = {'status': 'running', 'windows': [], 'calibration': []}
    for index in range(max(0, len(checkpoints) - e.args.last_windows), len(checkpoints)):
        checkpoint = checkpoints[index]
        previous = checkpoints[index - 1] if index else None
        first = previous['evaluations'] + 1 if previous else 1
        window = [r for r in search if first <= r['evaluation'] <= checkpoint['evaluations']]
        nominated = []
        for group in batches(window):
            order = np.argsort([-r['log_likelihood'] for r in group])[:2]
            nominated.extend(group[i] for i in order if group[i]['log_likelihood'] > -1.e10)
        incumbent = previous['parameters'] if previous else list(e.manifest['initial'].values())
        shortlist, selected = [incumbent], []
        for row in sorted(nominated, key=lambda r: -r['log_likelihood']):
            if row['parameters'] not in shortlist:
                shortlist.append(row['parameters'])
                selected.append(row['evaluation'])
            if len(shortlist) == 4:
                break
        scored = []
        for offset in range(0, len(window), 10):
            group = window[offset:offset + 10]
            scores = e.score([r['parameters'] for r in group])
            scored.extend({**r, 'reference_score': float(score),
                           'nominated': r in nominated, 'shortlisted': r['evaluation'] in selected}
                          for r, score in zip(group, scores, strict=True))
            if offset % 100 == 0:
                print(json.dumps({'audit_progress': {'checkpoint': checkpoint['evaluations'], 'rescored': offset + len(group)}}), flush=True)
        with (e.args.output / 'rescored_candidates.jsonl').open('a') as stream:
            for row in scored:
                stream.write(json.dumps(row) + '\n')
        best = max(scored, key=lambda r: r['reference_score'])
        nominated_best = max((r for r in scored if r['nominated']), key=lambda r: r['reference_score'])
        population_metrics = []
        for group in batches(scored):
            low = np.array([r['log_likelihood'] for r in group])
            high = np.array([r['reference_score'] for r in group])
            if (high <= -1.e9).any() or len(group) < 2:
                continue
            count = len(group) // 2
            chosen = np.argsort(-low)[:count]
            oracle = np.argsort(-high)[:count]
            population_metrics.append({'first_evaluation': group[0]['evaluation'], 'estimates': group[0]['estimates'],
                                       'reported_uncertainty': group[0]['ranking_uncertainty'],
                                       'score_bias_mean': float((low-high).mean()),
                                       'selected_half_reference_regret': float(high[oracle].mean()-high[chosen].mean()),
                                       'winner_reference_regret': float(high.max()-high[np.argmax(low)]),
                                       'selected_half_overlap': len(set(chosen) & set(oracle)) / count})
        labeled = {'incumbent_before': incumbent, 'original_checkpoint': checkpoint['parameters'],
                   'best_all': best['parameters'], 'best_nominated': nominated_best['parameters'],
                   'original_final': list(e.result['fitted'].values())}
        result = {'first_evaluation': first, 'last_evaluation': checkpoint['evaluations'],
                  'original_reference_score': checkpoint['reference_score'],
                  'best_reference_score': best['reference_score'], 'best_evaluation': best['evaluation'],
                  'best_was_nominated': best['nominated'], 'best_was_shortlisted': best['shortlisted'],
                  'best_nominated_reference_score': nominated_best['reference_score'],
                  'shortlisted_evaluations': selected, 'population_metrics': population_metrics,
                  'parameters': labeled, 'validation': e.validate(labeled)}
        report['windows'].append(result)
        e.save(report)
        print(json.dumps({'window_result': {k:v for k,v in result.items() if k not in ('parameters','population_metrics')}}), flush=True)

    # Repeated independent two-block measurements at one late low-budget
    # population. Compare the heuristic against a three-seed 100k reference.
    population = next(group for group in reversed(list(batches(search))) if group[0]['estimates'] == 5000 and len(group) == 10)
    candidates = [r['parameters'] for r in population]
    high = np.array([e.score(candidates, seed=seed) for seed in e.args.validation_seeds])
    target = high.mean(0)
    rng = np.random.default_rng(543210 + e.seed)
    for repetition in range(20):
        seeds = rng.integers(100000, 2**31-1, 2).tolist()
        blocks = [e.density(candidates, 2500, seed) for seed in seeds]
        scores, se, valid = pooled_scores(blocks, [2500,2500], e.study.mask)
        uncertainty, promote = ranking_uncertainty(scores,se,valid,1.)
        chosen, oracle = np.argsort(-scores)[:5], np.argsort(-target)[:5]
        remaining = np.setdiff1d(np.arange(10),chosen)
        actual_inversion = max(0., float((target[remaining,None]-target[None,chosen]).max()))
        report['calibration'].append({'seeds':seeds,'uncertainty':uncertainty,'would_promote':bool(promote),
                                       'cross_boundary_reference_inversion':actual_inversion,
                                       'winner_reference_regret':float(target.max()-target[np.argmax(scores)]),
                                       'selected_half_reference_regret':float(target[oracle].mean()-target[chosen].mean()),
                                       'mean_score_bias':float((scores-target).mean())})
    report.update(status='complete', calibration_population_first=population[0]['evaluation'],
                  calibration_reference_scores=high.tolist())
    e.save(report)


def refine(experiment):
    e = experiment
    names = e.names
    distributions = {name:FloatDistribution(*bound[:2],step=bound[2]) for name,bound in e.manifest['bounds'].items()}
    initial = dict(zip(names,e.result['adaptive']['checkpoints'][-1]['parameters'],strict=True))
    sampler_class = optuna.samplers.CmaEsSampler
    covariance = None
    if e.args.mode == 'covariance':
        sources = json.loads(e.args.covariance_source.read_text())
        source = sources.get('covariance_sources', sources)[e.args.run.name]
        covariance = np.array(source['states'][-1]['covariance_alphabetical'])

        class CovarianceSampler(optuna.samplers.CmaEsSampler):
            # Diagnostic ablation only: preserve the same center, sigma, seed,
            # and learning rates, changing only the initial covariance.
            def _init_optimizer(self, trans, direction):
                assert list(trans._search_space) == source['parameter_order']
                optimizer = super()._init_optimizer(trans, direction)
                optimizer._C = covariance.copy()
                optimizer._B = optimizer._D = None
                return optimizer

        sampler_class = CovarianceSampler
    study = optuna.create_study(direction='maximize', sampler=sampler_class(
        x0=initial,sigma0=.03,lr_adapt=True,popsize=10,seed=e.manifest['arguments']['optimizer_seed']+1))
    study.enqueue_trial(initial)
    saved = [] if covariance is not None else [r for r in e.rows if r['phase']=='refinement']
    for group in batches(saved):
        trials = [study.ask(distributions) for _ in group]
        candidates = [[trial.params[name] for name in names] for trial in trials]
        np.testing.assert_array_equal(candidates,[r['parameters'] for r in group])
        for trial,row in zip(trials,group,strict=True):
            study.tell(trial,row['log_likelihood'])
    completed = len(saved)
    best_score = (e.result['best_training_log_likelihood'] if covariance is None else
                  e.result['adaptive']['checkpoints'][-1]['reference_score'])
    best = list(e.result['fitted'].values()) if covariance is None else list(initial.values())
    report = {'status':'running','replayed_exactly':completed,'checkpoints':[],
              'original_training_score':best_score,'original_parameters':best}
    if covariance is not None:
        report['initial_covariance'] = covariance.tolist()
    labeled = {'original':best}
    started = time.perf_counter()
    while completed < e.args.refine_total:
        count = 1 if completed == 0 else min(10-(completed-1)%10,e.args.refine_total-completed)
        trials = [study.ask(distributions) for _ in range(count)]
        candidates = [[trial.params[name] for name in names] for trial in trials]
        scores = e.score(candidates)
        with (e.args.output/'continuation.jsonl').open('a') as stream:
            for trial,candidate,score in zip(trials,candidates,scores,strict=True):
                study.tell(trial,float(score))
                if score > best_score:
                    best_score,best = float(score),candidate
                stream.write(json.dumps({'evaluation':trial.number+1,'parameters':candidate,'reference_score':float(score)})+'\n')
        completed += count
        if completed % 200 in (0,1) or completed==e.args.refine_total:
            checkpoint={'evaluations':completed,'best_reference_score':best_score,'parameters':best,
                        'continuation_seconds':time.perf_counter()-started}
            report['checkpoints'].append(checkpoint)
            labeled[f'refinement_{completed}']=best
            e.save(report)
            print(json.dumps({'refinement_checkpoint':checkpoint}),flush=True)
    # Assess fixed, predeclared checkpoints only after continuation has ended.
    report.update(status='complete',validation=e.validate(labeled))
    e.save(report)


def perturb(experiment):
    """Compare coupled parameter changes with changing one coordinate alone."""
    e = experiment
    original = np.array(list(e.result['fitted'].values()))
    baseline = json.loads((e.args.comparator / 'recovery.json').read_text())
    fixed = np.array([baseline['fitted'][name] for name in e.names])
    labeled = {'adaptive': original.tolist(), 'fixed': fixed.tolist()}
    for fraction in (.25, .5, .75):
        labeled[f'joint_fraction_{fraction}'] = ((1-fraction)*original+fraction*fixed).tolist()
    for i, name in enumerate(e.names):
        candidate = original.copy()
        candidate[i] = fixed[i]
        labeled[f'change_only_{name}'] = candidate.tolist()
    if e.args.other_run:
        other = json.loads((e.args.other_run / 'recovery.json').read_text())
        labeled['other_fixed_start'] = [other['fitted'][name] for name in e.names]
    e.save({'status': 'complete', 'parameters': labeled, 'validation': e.validate(labeled),
            'note': 'Parameter perturbations, not profile likelihoods; no nuisance coordinates are refitted.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['audit','refine','perturb','covariance'],required=True)
    parser.add_argument('--last-windows',type=int,default=2)
    parser.add_argument('--refine-total',type=int,default=1200)
    parser.add_argument('--validation-seeds',type=int,nargs='+',default=[9201,9202,9203])
    parser.add_argument('--comparator',type=Path)
    parser.add_argument('--other-run',type=Path)
    parser.add_argument('--covariance-source',type=Path)
    args=parser.parse_args()
    if args.mode == 'perturb' and args.comparator is None:
        parser.error('--comparator is required for parameter perturbations')
    if args.mode == 'covariance' and args.covariance_source is None:
        parser.error('--covariance-source is required for the covariance ablation')
    args.output.mkdir(parents=True,exist_ok=False)
    pnl.set_num_threads(8);torch.set_num_threads(8)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    experiment=Experiment(args)
    {'audit': audit, 'refine': refine, 'perturb': perturb, 'covariance': refine}[args.mode](experiment)


if __name__=='__main__':
    main()
