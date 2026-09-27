"""Compare conditioned and marginal GPU scores on the same ordered DAWA subject.

Both paths use the original 10 ms model with noise in every LCA. They deliberately
target different objectives: this measures the cost of using observed history,
not numerical agreement between the objectives. Compilation is timed separately.
"""

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.globals.utilities import set_global_seed
from dawa_batched_simulation import SOURCE, build_model, fit_surface, node
from dawa_pec_fit import LAUNCH, load_subject, save_json, synthetic_parameters


PROPOSALS = (
    (.30, .20, -.45, 10., .90, .90, 1., 5.),
    (.40, .22, -.40, 12., .70, .80, 1.5, 5.5),
    (.50, .18, -.35, 8., .50, .70, 2., 6.),
    (.60, .25, -.30, 15., .30, .60, 2.5, 7.),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=SOURCE.parent / 'flanker_data_part1.csv')
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--trials', type=int, help='Optional prefix; omitted means the complete subject')
    parser.add_argument('--estimates', type=int, nargs='+', default=[10000, 100000])
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--likelihoods', nargs='+', choices=['marginal', 'conditioned'],
                        default=['marginal', 'conditioned'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--pseudocount', type=float, default=1., help='Per bin at 100,000 estimates')
    parser.add_argument('--smoothing-sigma', type=float, default=.5)
    parser.add_argument('--profile', type=Path, help='Save cProfile and CUDA trace for the final case')
    parser.add_argument('--execution', choices=['prepared', 'reference'], default='prepared',
                        help='Conditioned launch preparation; reference re-prepares every trial')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.estimates, *args.batch_sizes, args.repeats, args.max_steps) < 1:
        parser.error('Counts must be positive')
    if max(args.batch_sizes) > len(PROPOSALS):
        parser.error(f'At most {len(PROPOSALS)} candidates are provided')
    if args.output.exists():
        parser.error('Output already exists; choose a new report path')
    if any(not np.isfinite(x) or x < 0 for x in (args.pseudocount, args.smoothing_sigma)):
        parser.error('Pseudocount and smoothing must be finite and nonnegative')

    torch.set_num_threads(4)
    set_global_seed(args.seed)
    frame = load_subject(args.data, args.subject, trials=args.trials)
    model, inputs, outputs = build_model(trials=len(frame), c_noise=.1, s_noise=.1, d_noise=.1, r_noise=.1)
    inputs[node(model, 'Task Input')] = frame[['T1', 'T2']].to_numpy()
    inputs[node(model, 'Stimulus Input')] = frame[['S1', 'S2', 'S3', 'S4']].to_numpy()
    plan = BatchedCompositionCompiler.compile(model, backend='triton', outputs=outputs, max_steps=args.max_steps)
    fitted = {f'{mechanism.name}.{parameter}' for parameter, mechanism in fit_surface(model)}
    plan = plan.specialize_parameters({p.name: p.default for p in plan.ir.params if p.name not in fitted})
    parameters = [synthetic_parameters(model, frame, proposal) for proposal in PROPOSALS]
    observed = frame[['decision', 'response_time']].to_numpy()
    include = frame.likelihood_include_mask.to_numpy(dtype=bool)
    # All recorded trials condition history, including rows omitted from scoring.
    bins = max(100, int(np.ceil(float(observed[:, 1].max()) / .03)))
    upper = bins * .03
    shared = dict(data=observed, categorical_dims=[0], bins=bins, bin_range=[(0., upper)],
                  smoothing_sigma=args.smoothing_sigma, categorical_cardinalities=[2],
                  include_mask=include, strict_truncation=True, triton_launch_options=LAUNCH)
    report = dict(
        status='running', gpu=torch.cuda.get_device_name(), hostname=platform.node(),
        torch=torch.__version__, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        subject=args.subject, trials=len(frame), scored_trials=int(include.sum()),
        candidates=PROPOSALS, lca_dt=.01, all_lca_noise=.1, launch=LAUNCH,
        histogram=dict(bins=bins, rt_range=[0., upper], smoothing_sigma=args.smoothing_sigma,
                       pseudocount_at_100000=args.pseudocount,
                       contamination_fraction=bins * 2 * args.pseudocount / (100000 + bins * 2 * args.pseudocount)),
        note='Same model and observations; different statistical objectives. Fit-time projections assume '
             '5000 evaluations at the measured batch size and exclude optimizer/validation overhead; '
             'they are not convergence estimates. GPU allocated memory excludes other processes.',
        conditioned_execution=args.execution, cases=[],
    )
    repo = Path(__file__).resolve().parents[4]
    report['implementation_sha256'] = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in ('psyneulink/core/batched/compiler.py', 'psyneulink/core/batched/likelihood.py',
                     'psyneulink/core/batched/backend/triton/runtime.py',
                     'psyneulink/core/batched/backend/triton/conditioned.py',
                     'psyneulink/core/batched/backend/triton/emit/emitter.py',
                     'psyneulink/core/batched/backend/triton/state.py')
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)

    def evaluate(kind, rows, estimates, seed, *, diagnostics=False):
        options = dict(shared, seed=seed, pseudocount=args.pseudocount * estimates / 100000)
        if kind == 'conditioned':
            return plan.conditioned_log_likelihood(inputs, rows, estimates,
                                                   return_diagnostics=diagnostics, execution=args.execution, **options)
        return plan.log_likelihood(inputs, rows, estimates, **options)

    for estimates in args.estimates:
        for batch in args.batch_sizes:
            rows = parameters[:batch]
            for kind in args.likelihoods:
                case = dict(likelihood=kind, estimates=estimates, candidate_batch_size=batch, runs=[])
                report['cases'].append(case)
                print(json.dumps(dict(starting={k: v for k, v in case.items() if k != 'runs'})), flush=True)
                save_json(args.output, report)
                torch.cuda.synchronize()
                start = time.perf_counter()
                evaluate(kind, rows, estimates, args.seed)
                torch.cuda.synchronize()
                case['warmup_seconds'] = time.perf_counter() - start
                for repeat in range(args.repeats):
                    seed = args.seed + repeat
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    score = evaluate(kind, rows, estimates, seed)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    case['runs'].append(dict(seed=seed, seconds=elapsed,
                                             scores=np.atleast_1d(score).tolist(),
                                             peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                             incremental_peak_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20))
                    save_json(args.output, report)
                median = statistics.median(run['seconds'] for run in case['runs'])
                case.update(median_seconds=median, seconds_per_candidate=median / batch,
                            projected_5000_evaluations_hours=median / batch * 5000 / 3600)
                if kind == 'conditioned':
                    score, diagnostics = evaluate(kind, rows, estimates, args.seed, diagnostics=True)
                    case['diagnostics'] = {name: np.asarray(value).tolist() for name, value in diagnostics.items()
                                           if name in ('per_trial_densities', 'effective_sample_size',
                                                       'prior_mixture_fraction', 'zero_support')}
                    ess = np.asarray(diagnostics['effective_sample_size'])
                    case['ess_summary'] = dict(minimum=float(ess.min()), median=float(np.median(ess)),
                                               fifth_percentile=float(np.quantile(ess, .05)))
                print(json.dumps({k: v for k, v in case.items() if k != 'diagnostics'}), flush=True)
                save_json(args.output, report)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        cpu = cProfile.Profile()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as trace:
            cpu.enable()
            evaluate(kind, rows, estimates, args.seed)
            torch.cuda.synchronize()
            cpu.disable()
        cpu.dump_stats(str(args.profile.with_suffix('.pstats')))
        trace.export_chrome_trace(str(args.profile.with_suffix('.json')))
        args.profile.with_suffix('.txt').write_text(trace.key_averages().table(sort_by='self_cuda_time_total', row_limit=30))
        report['profile'] = str(args.profile)
    kernels = []
    for name, module in tuple(sys.modules.items()):
        if not name.startswith('pnl_batched_'):
            continue
        for symbol in ('pnl_batched_coevolving_graph_kernel', 'pnl_batched_stateful_graph_kernel'):
            function = getattr(module, symbol, None)
            if function is not None:
                for cache in function.device_caches.values():
                    kernels.extend(dict(registers=k.n_regs, spills=k.n_spills, metadata=str(k.metadata))
                                   for k in cache[0].values())
    report['kernels'] = kernels
    report['status'] = 'complete'
    save_json(args.output, report)


if __name__ == '__main__':
    main()
