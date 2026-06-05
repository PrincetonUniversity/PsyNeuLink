"""Benchmark one configuration of PEC fitting and append a JSON metrics row.

Compares the *regular* PEC optimizer against the Dask-distributed ask/tell
backend on the identical problem (same DDM, data, bounds, CMA-ES seed,
num_estimates, evaluation budget). Run ONE config per process for clean
isolation (fresh LLVM compile, no thread-setting/cluster leakage between runs).

Modes
  regular        standard pec.run(): serial trials, estimates over all cores
  dask-local     LocalCluster, n_workers x worker_cores on one node
  dask-jobqueue  SLURMCluster, n_workers jobs x worker_cores each (multi-node)

Timing separates a one-time warmup (LLVM compile) from the steady-state fit
loop, so the throughput numbers are not distorted by compilation.

Usage:
  python bench.py --mode regular       --worker-cores 16 --num-estimates 4000 \
                  --total-evals 240 --reps 2 --out results.jsonl
  python bench.py --mode dask-local    --n-workers 4 --worker-cores 4 ...
  python bench.py --mode dask-jobqueue --n-workers 4 --worker-cores 16 ...
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

# Make the pec_dask_mle package importable (this file lives in
# pec_hierarchical/benchmark/, the package in pec_hierarchical/).
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.dirname(HERE)
sys.path.insert(0, PKG_PARENT)
os.environ["PYTHONPATH"] = PKG_PARENT + os.pathsep + os.environ.get("PYTHONPATH", "")

from pec_dask_mle import config  # noqa: E402
from pec_dask_mle.data import make_data  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Usage sampler: mean CPU (in cores) and peak RSS over this process + children.
# Captures Dask LocalCluster workers (they are child processes); does NOT see
# remote jobqueue workers (separate SLURM jobs -- use sacct for those).
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


# ---------------------------------------------------------------------------
# Regular PEC baseline (CMA-ES via Optuna, pec.run())
# ---------------------------------------------------------------------------
def build_regular_pec(data_to_fit, num_estimates, max_iterations):
    """A standard PEC configured with a real CMA-ES optimizer for pec.run()."""
    import optuna
    import psyneulink as pnl
    from pec_dask_mle.factory import build_model

    comp, decision = build_model()
    fit_parameters = {
        (name, decision): np.linspace(lo, hi, 1000)
        for name, (lo, hi) in config.FIT_BOUNDS.items()
    }
    pec = pnl.ParameterEstimationComposition(
        name="pec_bench",
        nodes=[comp],
        parameters=fit_parameters,
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data_to_fit,
        optimization_function=pnl.PECOptimizationFunction(
            method=optuna.samplers.CmaEsSampler(seed=0), max_iterations=max_iterations
        ),
        num_estimates=num_estimates,
        initial_seed=config.INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,  # CRN, matches Dask path
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, comp


def run_regular(args, data_to_fit, trial_inputs):
    import optuna  # noqa: F401  (ensure available)
    import psyneulink as pnl

    pnl.set_num_threads(args.worker_cores)
    pec, comp = build_regular_pec(data_to_fit, args.num_estimates, args.total_evals)

    # Warmup: one likelihood eval compiles + caches the LLVM binary on this PEC.
    mid = [(lo + hi) / 2.0 for lo, hi in config.FIT_BOUNDS.values()]
    t0 = time.perf_counter()
    pec.log_likelihood(*mid, inputs={comp: trial_inputs})
    compile_s = time.perf_counter() - t0

    # Timed fit (warm).
    t0 = time.perf_counter()
    pec.run(inputs={comp: trial_inputs})
    loop_s = time.perf_counter() - t0

    recovered = dict(zip(config.FIT_BOUNDS.keys(), pec.optimized_parameter_values.values()))
    return compile_s, loop_s, recovered


# ---------------------------------------------------------------------------
# Dask paths (reuse the package's factory/worker/driver)
# ---------------------------------------------------------------------------
def _warmup_workers(client, data_to_fit, trial_inputs, worker_cores):
    """Submit one eval to each worker so every worker builds+compiles+caches its
    PEC before the timed loop. Returns wall time of the slowest warmup."""
    from pec_dask_mle.worker import evaluate_loglik

    mid = [(lo + hi) / 2.0 for lo, hi in config.FIT_BOUNDS.values()]
    addrs = list(client.scheduler_info()["workers"])
    t0 = time.perf_counter()
    futs = [
        client.submit(
            evaluate_loglik, mid, trial_inputs, data_to_fit, worker_cores,
            workers=[a], pure=False,
        )
        for a in addrs
    ]
    client.gather(futs)
    return time.perf_counter() - t0


def _run_with_client(client, args, data_to_fit, trial_inputs):
    from pec_dask_mle import driver

    # Override the package config for this benchmark point.
    config.NUM_ESTIMATES = args.num_estimates
    config.WORKER_CORES = args.worker_cores
    config.TOTAL_EVALS = args.total_evals
    config.BATCH_SIZE = args.n_workers

    compile_s = _warmup_workers(client, data_to_fit, trial_inputs, args.worker_cores)
    t0 = time.perf_counter()
    study = driver.run_fit(client, data_to_fit, trial_inputs, verbose=False)
    loop_s = time.perf_counter() - t0
    recovered = {n: study.best_params[n] for n in config.FIT_BOUNDS}
    return compile_s, loop_s, recovered


def run_dask_local(args, data_to_fit, trial_inputs):
    from dask.distributed import Client, LocalCluster

    _quiet_dask()
    cluster = LocalCluster(
        n_workers=args.n_workers, threads_per_worker=1, silence_logs=logging.WARNING
    )
    client = Client(cluster)
    try:
        client.wait_for_workers(args.n_workers, timeout=120)
        return _run_with_client(client, args, data_to_fit, trial_inputs)
    finally:
        client.close()
        cluster.close()


def run_dask_jobqueue(args, data_to_fit, trial_inputs):
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    _quiet_dask()
    cluster = SLURMCluster(
        queue=config.SLURM_PARTITION,
        cores=1,
        processes=1,
        job_cpu=args.worker_cores,
        memory=config.WORKER_MEMORY,
        walltime=config.WORKER_WALLTIME,
        interface=config.SLURM_INTERFACE,
        job_script_prologue=[f"export PYTHONPATH={PKG_PARENT}:$PYTHONPATH"],
    )
    cluster.scale(jobs=args.n_workers)
    client = Client(cluster)
    try:
        print(f"waiting for {args.n_workers} workers...", flush=True)
        client.wait_for_workers(args.n_workers, timeout=args.worker_timeout)
        return _run_with_client(client, args, data_to_fit, trial_inputs)
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
    p.add_argument("--mode", required=True, choices=list(RUNNERS))
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--worker-cores", type=int, default=16)
    p.add_argument("--num-estimates", type=int, default=4000)
    p.add_argument("--total-evals", type=int, default=240)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--worker-timeout", type=int, default=900)
    p.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = p.parse_args()

    total_cores = (1 if args.mode == "regular" else args.n_workers) * args.worker_cores

    # Same data for every mode/rep (parity).
    data_to_fit, trial_inputs = make_data()
    true = {k: config.TRUE_PARAMS[k] for k in config.FIT_BOUNDS}

    for rep in range(args.reps):
        sampler = UsageSampler()
        sampler.start()
        t_total = time.perf_counter()
        compile_s, loop_s, recovered = RUNNERS[args.mode](args, data_to_fit, trial_inputs)
        total_s = time.perf_counter() - t_total
        sampler.stop()
        sampler.join(timeout=2)

        max_pct_err = max(
            100.0 * abs(true[k] - recovered[k]) / true[k] for k in true
        )
        mcc = sampler.mean_cpu_cores
        prg = sampler.peak_rss_gb
        # The local sampler only sees THIS process. For jobqueue the workers are
        # separate SLURM jobs on other nodes, so the driver-only CPU/RSS readings
        # are meaningless -- drop them. (Use sacct of the worker jobs, or
        # core_hours below, for true multi-node usage.)
        if args.mode == "dask-jobqueue":
            mcc = prg = None
        row = {
            "mode": args.mode,
            "n_workers": args.n_workers if args.mode != "regular" else 1,
            "worker_cores": args.worker_cores,
            "total_cores": total_cores,
            "num_estimates": args.num_estimates,
            "total_evals": args.total_evals,
            "rep": rep,
            "compile_s": round(compile_s, 3),
            "loop_s": round(loop_s, 3),
            "total_s": round(total_s, 3),
            "evals_per_s": round(args.total_evals / loop_s, 3) if loop_s else None,
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
            f"[{args.mode} {args.n_workers}x{args.worker_cores} ne={args.num_estimates} "
            f"rep{rep}] loop={loop_s:.1f}s compile={compile_s:.1f}s "
            f"evals/s={row['evals_per_s']} util={row['util_pct']}% err={max_pct_err:.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
