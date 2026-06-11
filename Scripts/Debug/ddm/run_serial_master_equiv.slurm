#!/bin/bash
# Serial PEC correctness check for the DDM benchmark settings.
#
# This is intentionally standalone/master-compatible: it uses the regular
# ParameterEstimationComposition + pec.run() path only, with no Dask benchmark
# imports.  The defaults mirror benchmark/run_multinode.slurm:
#   MODEL=ddm, NE=8000, NT=250, POP=32, TE=960
#
#SBATCH --job-name=pec_serial_master
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ap9344@princeton.edu

set -euo pipefail

REPO=${REPO:-/scratch/gpfs/JDC/ap9344/PsyNeuLink}
PY=${PY:-$REPO/.venv/bin/python3}
OUT=${OUT:-$REPO/Scripts/Debug/ddm/results_serial_master_equiv.jsonl}

export CORES=${CORES:-16}
export NE=${NE:-8000}
export NT=${NT:-250}
export POP=${POP:-32}
export TE=${TE:-960}
export OUT

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/pnl-matplotlib-${SLURM_JOB_ID:-$$}}
mkdir -p "$MPLCONFIGDIR"

cd "$REPO"
rm -f "$OUT"

"$PY" - <<'PY'
import json
import os
import socket
import threading
import time

import numpy as np
import optuna
import pandas as pd
import psyneulink as pnl
from psyneulink.core.globals.utilities import set_global_seed


CORES = int(os.environ["CORES"])
NE = int(os.environ["NE"])
NT = int(os.environ["NT"])
POP = int(os.environ["POP"])
TE = int(os.environ["TE"])
OUT = os.environ["OUT"]

TRUE_PARAMS = dict(
    starting_value=0.0,
    rate=0.3,
    noise=1.0,
    threshold=0.6,
    non_decision_time=0.15,
    time_step_size=0.01,
)
FIT_BOUNDS = {
    "rate": (-0.5, 0.5),
    "threshold": (0.5, 1.0),
    "non_decision_time": (0.0, 1.0),
}


try:
    import psutil
except ImportError:
    psutil = None


class UsageSampler(threading.Thread):
    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.cpu_core_samples = []
        self.peak_rss = 0
        self._procs = {}

    def _refresh(self):
        try:
            root = psutil.Process()
            live = [root] + root.children(recursive=True)
        except psutil.Error:
            return list(self._procs.values())
        seen = set()
        for proc in live:
            seen.add(proc.pid)
            if proc.pid not in self._procs:
                self._procs[proc.pid] = proc
                try:
                    proc.cpu_percent(None)
                except psutil.Error:
                    pass
        for pid in [pid for pid in self._procs if pid not in seen]:
            self._procs.pop(pid, None)
        return list(self._procs.values())

    def run(self):
        if psutil is None:
            return
        self._refresh()
        while not self._stop_event.wait(self.interval):
            cpu_pct = 0.0
            rss = 0
            for proc in self._refresh():
                try:
                    cpu_pct += proc.cpu_percent(None)
                    rss += proc.memory_info().rss
                except psutil.Error:
                    pass
            self.cpu_core_samples.append(cpu_pct / 100.0)
            self.peak_rss = max(self.peak_rss, rss)

    def stop(self):
        self._stop_event.set()

    @property
    def mean_cpu_cores(self):
        return float(np.mean(self.cpu_core_samples)) if self.cpu_core_samples else None

    @property
    def peak_rss_gb(self):
        return self.peak_rss / 1e9 if self.peak_rss else None


def build_comp():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(**TRUE_PARAMS),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision)


def make_inputs(comp, num_trials):
    decision = comp.nodes[0]
    rng = np.random.default_rng(12345)
    trial_inputs = rng.choice([5.0, -5.0], size=(num_trials, 1), p=[0.1, 0.9])
    return {decision: trial_inputs}


def make_data(num_trials):
    set_global_seed(0)
    comp = build_comp()
    comp.run(inputs=make_inputs(comp, num_trials))
    data = pd.DataFrame(
        np.squeeze(np.array(comp.results)), columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")
    return data


def build_pec(comp, data, num_estimates, optimization_function):
    decision = comp.nodes[0]
    fit_parameters = {
        (name, decision): np.linspace(lo, hi, 1000)
        for name, (lo, hi) in FIT_BOUNDS.items()
    }
    pec = pnl.ParameterEstimationComposition(
        name="pec_ddm_serial_master_equiv",
        nodes=[comp],
        parameters=fit_parameters,
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=optimization_function,
        num_estimates=num_estimates,
        initial_seed=42,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec


def midpoint_values():
    return [(lo + hi) / 2.0 for lo, hi in FIT_BOUNDS.values()]


def fnum(value, width, prec):
    return ("-" if value is None else f"{value:.{prec}f}").rjust(width)


def print_summary(row):
    label = f"{row['model']}/{row['mode']} {row['n_workers']}x{row['worker_cores']}"
    hdr = (f"{'config':<26}{'cores':>6}{'ne':>7}{'rounds':>8}{'pop':>7}"
           f"{'evals':>7}{'loop_s':>9}{'evals/s':>9}{'speedup':>9}"
           f"{'util%':>7}{'rss_gb':>8}{'err%':>7}{'core_h':>9}")
    print("\n==== SUMMARY ====", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    print(
        f"{label:<26}{row['total_cores']:>6}{row['num_estimates']:>7}"
        f"{row['n_rounds']:>8}{row['optimizer_popsize']:>7}"
        f"{row['total_evals']:>7}"
        f"{fnum(row['loop_s'], 9, 1)}{fnum(row['evals_per_s'], 9, 1)}"
        f"{'-':>9}{fnum(row['util_pct'], 7, 0)}"
        f"{fnum(row['peak_rss_gb'], 8, 2)}{fnum(row['max_pct_err'], 7, 1)}"
        f"{fnum(row['core_hours'], 9, 4)}",
        flush=True,
    )


def main():
    if TE < POP:
        raise SystemExit(f"TE ({TE}) must be >= POP ({POP})")
    if TE % POP:
        print(f"warning: TE={TE} is not divisible by POP={POP}; "
              f"Optuna will still run {TE} serial trials.", flush=True)

    pnl.set_num_threads(CORES)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = make_data(NT)
    comp = build_comp()
    inputs = make_inputs(comp, NT)
    optimizer = pnl.PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0, popsize=POP),
        max_iterations=TE,
    )
    pec = build_pec(comp, data, NE, optimizer)

    sampler = UsageSampler()
    sampler.start()
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    pec.log_likelihood(*midpoint_values(), inputs=inputs)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pec.run(inputs=inputs)
    loop_s = time.perf_counter() - t0
    total_s = time.perf_counter() - total_t0

    sampler.stop()
    sampler.join(timeout=2)

    recovered = dict(zip(FIT_BOUNDS, pec.optimized_parameter_values.values()))
    max_pct_err = max(
        100.0 * abs(TRUE_PARAMS[name] - recovered[name]) / TRUE_PARAMS[name]
        for name in FIT_BOUNDS
    )
    mean_cpu_cores = sampler.mean_cpu_cores
    peak_rss_gb = sampler.peak_rss_gb
    row = {
        "model": "ddm",
        "mode": "regular",
        "n_workers": 1,
        "worker_cores": CORES,
        "total_cores": CORES,
        "num_estimates": NE,
        "num_trials": NT,
        "n_rounds": TE // POP,
        "optimizer_popsize": POP,
        "batch_size": POP,
        "total_evals": TE,
        "rep": 0,
        "compile_s": round(compile_s, 3),
        "loop_s": round(loop_s, 3),
        "total_s": round(total_s, 3),
        "evals_per_s": round(TE / loop_s, 3) if loop_s else None,
        "core_hours": round(CORES * loop_s / 3600.0, 5),
        "mean_cpu_cores": round(mean_cpu_cores, 2) if mean_cpu_cores is not None else None,
        "util_pct": (
            round(100.0 * mean_cpu_cores / CORES, 1)
            if mean_cpu_cores is not None else None
        ),
        "peak_rss_gb": round(peak_rss_gb, 2) if peak_rss_gb is not None else None,
        "max_pct_err": round(max_pct_err, 2),
        "recovered": {k: round(v, 4) for k, v in recovered.items()},
        "host": socket.gethostname(),
    }
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")

    print("==== JSONL ROW ====", flush=True)
    print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    print_summary(row)


if __name__ == "__main__":
    main()
PY
