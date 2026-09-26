"""Profile a DAWA fit without changing its proposals, random draws, or scoring.

Usage: python dawa_fit_profile.py --profile-output NEW_DIR -- [fit options]
Pass --empirical before the separator to run the empirical rather than recovery
driver. Run under nsys --trace=cuda,nvtx to associate device activity with phases.
Nested wall-time ranges overlap; use their hierarchy or the CUDA trace, not a
sum of all recorded durations. No extra synchronization is inserted per call.
"""

import argparse
from collections import defaultdict
import cProfile
from functools import wraps
import inspect
import json
from pathlib import Path
import pstats
import sys
import time

import optuna
import torch

import dawa_adaptive_fit as adaptive
import dawa_pec_fit as driver
from psyneulink.core.batched.compiler import BatchedSimulationPlan
from psyneulink.core.batched.shifted_histogram import ShiftedHistogramScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--empirical', action='store_true')
    parser.add_argument('fit_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.profile_output.mkdir(parents=True, exist_ok=False)
    fit_args = args.fit_args[1:] if args.fit_args[:1] == ['--'] else args.fit_args
    sys.argv = ['dawa_pec_fit.py', *fit_args]
    records = []
    totals = defaultdict(lambda: {'calls': 0, 'seconds': 0.})
    origin = time.perf_counter()
    fit_done = False

    def context():
        phase = 'validation_predictions' if fit_done else 'setup'
        candidates = None
        frame = inspect.currentframe().f_back
        try:
            while frame is not None:
                name = frame.f_code.co_name
                if name in ('sample_densities', 'sample_density_blocks'):
                    candidates = frame.f_locals['candidates']
                if name == 'evaluate' and isinstance(frame.f_locals.get('self'), adaptive.PopulationRacer):
                    phase = 'search'
                    break
                if name == 'reference':
                    phase = 'reference'
                if name == 'fit_adaptive':
                    values = frame.f_locals
                    if 'finalists' in values:
                        phase = 'selection'
                    elif 'refined' in values:
                        phase = 'refinement'
                    elif phase != 'reference':
                        phase = 'screening_optimizer'
                    break
                frame = frame.f_back
        finally:
            del frame
        return phase, candidates

    def instrument(function, label, capture=False):
        @wraps(function)
        def wrapped(*positional, **keywords):
            nonlocal fit_done
            phase, candidates = context()
            if label in ('race', 'fit'):
                phase = 'search' if label == 'race' else 'whole_fit'
            meta = {}
            if label in ('counts', 'count_blocks', 'histogram', 'simulation'):
                meta = {'estimates': int(positional[3]), 'candidates': len(positional[2]),
                        'seed': keywords.get('seed')}
                if label == 'count_blocks':
                    meta['seeds'] = list(keywords['seeds'])
                    meta['blocks'] = len(meta['seeds'])
            if capture and candidates is not None:
                meta['parameters'] = [list(map(float, row)) for row in candidates]
            suffix = '' if not meta else f" C={meta['candidates']} N={meta['estimates']}"
            if 'blocks' in meta:
                suffix += f" B={meta['blocks']}"
            torch.cuda.nvtx.range_push(f'dawa/{phase}/{label}{suffix}')
            start = time.perf_counter()
            try:
                return function(*positional, **keywords)
            finally:
                elapsed = time.perf_counter() - start
                torch.cuda.nvtx.range_pop()
                key = (phase, label, meta.get('estimates'), meta.get('candidates'), meta.get('blocks'))
                totals[key]['calls'] += 1
                totals[key]['seconds'] += elapsed
                if capture or label in ('fit', 'race'):
                    records.append({'phase': phase, 'operation': label, 'start': start - origin,
                                    'seconds': elapsed, **meta})
                if label == 'fit':
                    fit_done = True
        return wrapped

    BatchedSimulationPlan.discrete_output_counts = instrument(BatchedSimulationPlan.discrete_output_counts, 'counts', True)
    BatchedSimulationPlan.discrete_output_count_blocks = instrument(
        BatchedSimulationPlan.discrete_output_count_blocks, 'count_blocks', True)
    BatchedSimulationPlan.histogram_likelihood = instrument(BatchedSimulationPlan.histogram_likelihood, 'histogram', True)
    BatchedSimulationPlan.run = instrument(BatchedSimulationPlan.run, 'simulation', True)
    ShiftedHistogramScorer.densities = instrument(ShiftedHistogramScorer.densities, 'shifted_score')
    adaptive.PopulationRacer.evaluate = instrument(adaptive.PopulationRacer.evaluate, 'race')
    adaptive.pooled_scores = instrument(adaptive.pooled_scores, 'pool')
    adaptive.ranking_uncertainty = instrument(adaptive.ranking_uncertainty, 'ranking')
    driver.fit_adaptive = instrument(adaptive.fit_adaptive, 'fit')
    optuna.study.Study.ask = instrument(optuna.study.Study.ask, 'ask')
    optuna.study.Study.tell = instrument(optuna.study.Study.tell, 'tell')
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        driver.main(recovery=not args.empirical)
    finally:
        profiler.disable()
        profiler.dump_stats(str(args.profile_output / 'python.pstats'))
        stats = pstats.Stats(profiler)
        functions = [{'file': key[0], 'line': key[1], 'name': key[2], 'primitive_calls': value[0],
                      'calls': value[1], 'self_seconds': value[2], 'cumulative_seconds': value[3]}
                     for key, value in stats.stats.items()]
        report = {'fit_arguments': fit_args, 'recovery': not args.empirical,
                  'elapsed_seconds': time.perf_counter() - origin, 'calls': records,
                  'timings': [dict(phase=key[0], operation=key[1], estimates=key[2], candidates=key[3], blocks=key[4], **value)
                              for key, value in totals.items()],
                  'python_by_self': sorted(functions, key=lambda row: -row['self_seconds'])[:80],
                  'python_by_cumulative': sorted(functions, key=lambda row: -row['cumulative_seconds'])[:100],
                  'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'note': 'Nested inclusive wall times overlap; profiling overhead is included. CUDA trace separates execution from waits.'}
        (args.profile_output / 'profile.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
