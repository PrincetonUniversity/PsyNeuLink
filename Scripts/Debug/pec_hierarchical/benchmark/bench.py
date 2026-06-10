"""Benchmark one configuration of PEC fitting and append a JSON metrics row.

backend on an identical problem (same data, bounds, CMA-ES seed, num_estimates,
fixed CMA-ES round budget). The model is pluggable via ``--model`` (see
``models/`` -- ddm, stabflex, ...); the harness itself is model-agnostic.
Run ONE config per process for clean isolation (fresh LLVM compile, no
thread-setting/cluster leakage between runs).

Modes
  regular        standard pec.run(): serial trials, estimates over all cores
  dask-local     LocalCluster, n_workers x worker_cores on one node
  dask-jobqueue  SLURMCluster, n_workers jobs x worker_cores each (multi-node)

Timing separates a one-time warmup (LLVM compile) from the steady-state fit
loop, so throughput is not distorted by compilation.

Budget: either --n-rounds (fixed CMA-ES generations; total work scales with the
fixed optimizer population size) or --total-evals (fixed evaluation budget
shared by all configs; rounds = budget // optimizer_popsize). Keep
--optimizer-popsize fixed across a worker sweep to make every run follow the
same optimizer trajectory and differ only in evaluation concurrency.

Usage:
  python bench.py --model ddm --mode regular    --worker-cores 16 --num-estimates 4000 \
                  --optimizer-popsize 32 --total-evals 960 --reps 2 --out results.jsonl
  python bench.py --model ddm --mode dask-local    --n-workers 4 --worker-cores 4 ...
  python bench.py --model ddm --mode dask-jobqueue --n-workers 4 --worker-cores 16 ...
"""

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time

import numpy as np


def _quiet_dask():
    for name in ("distributed", "distributed.worker", "distributed.scheduler",
                 "distributed.nanny", "distributed.core", "bokeh"):
        logging.getLogger(name).setLevel(logging.WARNING)


# This file lives in pec_hierarchical/benchmark/. Make both that dir (for
# `models` + `bench_core`) and its parent (for pec_dask_mle.config) importable,
# locally and on spawned/remote Dask workers.
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.dirname(HERE)
for _p in (PKG_PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = (
    HERE + os.pathsep + PKG_PARENT + os.pathsep + os.environ.get("PYTHONPATH", "")
)

# Repo root (.../PsyNeuLink) and a home for dask-jobqueue worker logs.
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
LOG_DIR = os.path.join(REPO, "slurm_logs")

from pec_dask_mle import config  # noqa: E402  (SLURM/infra config only)
import models  # noqa: E402
import bench_core  # noqa: E402


def sampler_popsize(args):
    """Fixed CMA-ES population size used for every benchmark point."""
    return args.optimizer_popsize


def total_evals(args):
    """Likelihood evaluations implied by the fixed-round budget."""
    return args.n_rounds * sampler_popsize(args)


try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Usage sampler: mean CPU (in cores) and peak RSS over this process + children.
# Captures Dask LocalCluster workers (child processes); does NOT see remote
# jobqueue workers (separate SLURM jobs -- use sacct for those).
# ---------------------------------------------------------------------------
class UsageSampler(threading.Thread):
    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.cpu_core_samples = []
        self.peak_rss = 0
        self._enabled = psutil is not None
        # Persist Process objects across ticks -- cpu_percent(None) measures the
        # delta since the *same object's* previous call, so recreating them each
        # tick would always read 0%.
        self._procs = {}

    def _refresh(self):
        try:
            root = psutil.Process()
            live = [root] + root.children(recursive=True)
        except psutil.Error:
            return list(self._procs.values())
        seen = set()
        for p in live:
            seen.add(p.pid)
            if p.pid not in self._procs:
                self._procs[p.pid] = p
                try:
                    p.cpu_percent(None)  # prime newly-seen process
                except psutil.Error:
                    pass
        for pid in [pid for pid in self._procs if pid not in seen]:
            self._procs.pop(pid, None)
        return list(self._procs.values())

    def run(self):
        if not self._enabled:
            return
        self._refresh()  # prime
        while not self._stop.wait(self.interval):
            cpu_pct, rss = 0.0, 0
            for p in self._refresh():
                try:
                    cpu_pct += p.cpu_percent(None)
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            self.cpu_core_samples.append(cpu_pct / 100.0)  # -> cores busy
            self.peak_rss = max(self.peak_rss, rss)

    def stop(self):
        self._stop.set()

    @property
    def mean_cpu_cores(self):
        return float(np.mean(self.cpu_core_samples)) if self.cpu_core_samples else None

    @property
    def peak_rss_gb(self):
        return self.peak_rss / 1e9 if self.peak_rss else None


def _mid(provider):
    return [(lo + hi) / 2.0 for lo, hi in provider.FIT_BOUNDS.values()]


# ---------------------------------------------------------------------------
# Regular PEC baseline (CMA-ES via Optuna, pec.run())
# ---------------------------------------------------------------------------
def run_regular(args, provider, data):
    import optuna
    import psyneulink as pnl

    pnl.set_num_threads(args.worker_cores)
    comp = provider.build_comp()
    inputs = provider.make_inputs(comp, args.num_trials)
    # Fixed-round budget: popsize per generation x n_rounds == total_evals.
    optimizer = pnl.PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0, popsize=sampler_popsize(args)),
        max_iterations=total_evals(args),
    )
    pec = provider.build_pec(comp, data, args.num_estimates, optimization_function=optimizer)

    # Warmup: one likelihood eval compiles + caches the LLVM binary on this PEC.
    t0 = time.perf_counter()
    pec.log_likelihood(*_mid(provider), inputs=inputs)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pec.run(inputs=inputs)
    loop_s = time.perf_counter() - t0

    recovered = dict(zip(provider.FIT_BOUNDS, pec.optimized_parameter_values.values()))
    return compile_s, loop_s, recovered


# ---------------------------------------------------------------------------
# Dask paths (model-agnostic via bench_core + the provider)
# ---------------------------------------------------------------------------
def _warmup_workers(client, args, provider, data):
    """One eval per worker so each builds+compiles+caches its PEC before timing."""
    addrs = list(client.scheduler_info()["workers"])
    t0 = time.perf_counter()
    futs = [
        client.submit(
            bench_core.evaluate_loglik, args.model, _mid(provider),
            args.num_trials, data, args.num_estimates, args.worker_cores,
            workers=[a], pure=False,
        )
        for a in addrs
    ]
    client.gather(futs)
    return time.perf_counter() - t0


def _run_with_client(client, args, provider, data):
    compile_s = _warmup_workers(client, args, provider, data)
    t0 = time.perf_counter()
    # Fixed-round budget: one fixed-size CMA-ES population per generation. Dask
    # worker count controls only how many candidates are evaluated concurrently.
    study = bench_core.run_fit(
        client, args.model, data,
        num_trials=args.num_trials, num_estimates=args.num_estimates,
        total_evals=total_evals(args), batch_size=sampler_popsize(args),
        worker_cores=args.worker_cores, fit_bounds=provider.FIT_BOUNDS,
        verbose=False,
    )
    loop_s = time.perf_counter() - t0
    recovered = {n: study.best_params[n] for n in provider.FIT_BOUNDS}
    return compile_s, loop_s, recovered


def run_dask_local(args, provider, data):
    from dask.distributed import Client, LocalCluster

    _quiet_dask()
    cluster = LocalCluster(
        n_workers=args.n_workers, threads_per_worker=1, silence_logs=logging.WARNING
    )
    client = Client(cluster)
    try:
        client.wait_for_workers(args.n_workers, timeout=120)
        return _run_with_client(client, args, provider, data)
    finally:
        client.close()
        cluster.close()


def run_dask_jobqueue(args, provider, data):
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    _quiet_dask()
    os.makedirs(LOG_DIR, exist_ok=True)
    cluster = SLURMCluster(
        queue=config.SLURM_PARTITION,
        cores=1,
        processes=1,
        job_cpu=args.worker_cores,
        memory=config.WORKER_MEMORY,
        walltime=config.WORKER_WALLTIME,
        interface=config.SLURM_INTERFACE,
        log_directory=LOG_DIR,  # worker slurm-*.out land here, not the cwd
        # Workers must import bench_core + models (HERE) and pec_dask_mle (PKG_PARENT).
        job_script_prologue=[f"export PYTHONPATH={HERE}:{PKG_PARENT}:$PYTHONPATH"],
    )
    cluster.scale(jobs=args.n_workers)
    client = Client(cluster)
    try:
        print(f"waiting for {args.n_workers} workers...", flush=True)
        client.wait_for_workers(args.n_workers, timeout=args.worker_timeout)
        return _run_with_client(client, args, provider, data)
    finally:
        client.close()
        cluster.close()


RUNNERS = {
    "regular": run_regular,
    "dask-local": run_dask_local,
    "dask-jobqueue": run_dask_jobqueue,
}


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ddm", choices=list(models.REGISTRY))
    p.add_argument("--mode", required=True, choices=list(RUNNERS))
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--worker-cores", type=int, default=16)
    p.add_argument("--optimizer-popsize", type=int, default=config.OPTIMIZER_POPSIZE,
                   help="fixed CMA-ES population / synchronous batch size; "
                        "keep constant across worker sweeps for identical "
                        "optimizer trajectories")
    p.add_argument("--num-estimates", type=int, default=4000)
    p.add_argument("--num-trials", type=int, default=config.NUM_TRIALS,
                   help="observed data trials; more data disentangles params")
    p.add_argument("--n-rounds", type=int, default=None,
                   help=f"fixed CMA-ES generations (default {config.N_ROUNDS}); "
                        f"total work then scales with popsize")
    p.add_argument("--total-evals", type=int, default=None,
                   help="fixed evaluation budget shared across configs "
                        "(apples-to-apples scaling); rounds = budget // popsize, "
                        "rounded down to whole generations")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--worker-timeout", type=int, default=900)
    p.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = p.parse_args()

    if args.n_rounds is not None and args.total_evals is not None:
        p.error("--n-rounds and --total-evals are mutually exclusive")
    provider = models.get(args.model)
    popsize = sampler_popsize(args)
    if popsize < 2:
        p.error("--optimizer-popsize must be >= 2")
    if args.total_evals is not None:
        if args.total_evals < popsize:
            p.error(f"--total-evals must be >= --optimizer-popsize ({popsize})")
        args.n_rounds = args.total_evals // popsize
    elif args.n_rounds is None:
        args.n_rounds = config.N_ROUNDS
    if args.n_rounds < 1:
        p.error("--n-rounds must be >= 1")

    total_cores = (1 if args.mode == "regular" else args.n_workers) * args.worker_cores
    n_evals = total_evals(args)
    if args.mode != "regular" and args.n_workers > popsize:
        print(
            f"warning: n_workers ({args.n_workers}) > optimizer_popsize ({popsize}); "
            "extra workers may be idle",
            file=sys.stderr,
            flush=True,
        )

    # Same data for every mode/rep (parity).
    data = provider.make_data(args.num_trials)
    true = provider.TRUE_FIT_VALUES

    for rep in range(args.reps):
        sampler = UsageSampler()
        sampler.start()
        t_total = time.perf_counter()
        compile_s, loop_s, recovered = RUNNERS[args.mode](args, provider, data)
        total_s = time.perf_counter() - t_total
        sampler.stop()
        sampler.join(timeout=2)

        max_pct_err = max(100.0 * abs(true[k] - recovered[k]) / true[k] for k in true)
        mcc = sampler.mean_cpu_cores
        prg = sampler.peak_rss_gb
        # The local sampler only sees THIS process. For jobqueue the workers are
        # separate SLURM jobs on other nodes, so its driver-only CPU/RSS readings
        # are meaningless -- drop them (use sacct, or core_hours, for those).
        if args.mode == "dask-jobqueue":
            mcc = prg = None
        row = {
            "model": args.model,
            "mode": args.mode,
            "n_workers": args.n_workers if args.mode != "regular" else 1,
            "worker_cores": args.worker_cores,
            "total_cores": total_cores,
            "num_estimates": args.num_estimates,
            "num_trials": args.num_trials,
            "n_rounds": args.n_rounds,
            "optimizer_popsize": popsize,
            "batch_size": popsize,
            "total_evals": n_evals,
            "rep": rep,
            "compile_s": round(compile_s, 3),
            "loop_s": round(loop_s, 3),
            "total_s": round(total_s, 3),
            "evals_per_s": round(n_evals / loop_s, 3) if loop_s else None,
            "core_hours": round(total_cores * loop_s / 3600.0, 5),
            "mean_cpu_cores": round(mcc, 2) if mcc is not None else None,
            "util_pct": round(100.0 * mcc / total_cores, 1) if mcc is not None else None,
            "peak_rss_gb": round(prg, 2) if prg is not None else None,
            "max_pct_err": round(max_pct_err, 2),
            "recovered": {k: round(v, 4) for k, v in recovered.items()},
            "host": socket.gethostname(),
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[{args.model} {args.mode} {args.n_workers}x{args.worker_cores} "
            f"ne={args.num_estimates} nt={args.num_trials} pop={popsize} "
            f"rounds={args.n_rounds} "
            f"evals={n_evals} rep{rep}] loop={loop_s:.1f}s compile={compile_s:.1f}s "
            f"evals/s={row['evals_per_s']} util={row['util_pct']}% err={max_pct_err:.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
