"""Plot the per-sampler worker x core study (sweep_samplers.sh output).

Reads results/samplers.d/*.jsonl (popsize == NW per config) and writes:
  plots/samplers.csv
  plots/sampler_quality_vs_workers.png    -- max_pct_err vs NW(=popsize), per sampler
  plots/sampler_throughput_vs_workers.png -- evals/s vs NW, per sampler (WC=1 line)
  plots/sampler_throughput_scaling.png    -- evals/s vs total_cores, per sampler
  plots/sampler_cost_vs_workers.png       -- core_hours vs NW, per sampler

The headline is quality_vs_workers: with popsize pinned to NW, growing the worker
count grows the batch, so this curve shows whether "popsize = num_workers" costs
solution quality for each sampler (CMA-ES/NSGA degrade; random/qmc are flat; TPE
is tolerant; GP holds quality). throughput_vs_workers shows the flip side: GP's
serial refit caps its evals/s while the others scale with workers.

Tolerant of partial data -- plots whatever configs have landed so far.
"""
import glob
import json
import os
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(HERE, "results", "samplers.d")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

rows = []
for p in sorted(glob.glob(os.path.join(RESDIR, "*.jsonl"))):
    for line in open(p):
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    raise SystemExit(f"no results in {RESDIR} -- run slurm/sweep_samplers.sh first")

# Stable per-sampler colour + display order (only those actually present).
ORDER = ["cmaes", "tpe", "tpe_noliar", "random", "qmc", "gp", "nsga2"]
samplers = [s for s in ORDER if any(r.get("sampler") == s for r in rows)]
samplers += sorted({r.get("sampler") for r in rows} - set(samplers))
cmap = plt.get_cmap("tab10")
color = {s: cmap(i % 10) for i, s in enumerate(samplers)}

# ---- CSV ---------------------------------------------------------------------
cols = ["sampler", "n_workers", "worker_cores", "total_cores", "optimizer_popsize",
        "num_estimates", "num_trials", "total_evals", "n_rounds",
        "loop_s", "evals_per_s", "core_hours", "max_pct_err"]
with open(os.path.join(OUT, "samplers.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for r in sorted(rows, key=lambda r: (r.get("sampler", ""), r["n_workers"], r["worker_cores"])):
        f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def by_sampler_nw(metric, reduce=median):
    """{sampler: sorted [(NW, reduced metric over WC)]} -- quality/cost are
    WC-independent (trajectory depends only on NW), so reduce across WC."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is not None:
            acc[r["sampler"]][r["n_workers"]].append(v)
    return {s: sorted((nw, reduce(vs)) for nw, vs in d.items()) for s, d in acc.items()}


def line_plot(data, ylabel, title, fname, logx=True, logy=False):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for s in samplers:
        pts = data.get(s)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=color[s], label=s)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log", base=2)
    ax.set_xlabel("n_workers  (= popsize = batch per round)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="sampler")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=120)
    plt.close(fig)


# ---- FIG 1 (headline): quality vs workers (popsize=NW) -----------------------
line_plot(
    by_sampler_nw("max_pct_err"),
    "max recovery error (%)  — lower is better",
    "Does popsize = num_workers cost quality?  (DDM, popsize pinned to NW)\n"
    "median over cores/worker; growing NW = bigger batch, fewer optimizer updates",
    "sampler_quality_vs_workers.png",
)

# ---- FIG 2: throughput vs workers at WC=1 (concurrency only) -----------------
wc1 = defaultdict(list)
for r in rows:
    if r["worker_cores"] == 1 and r.get("evals_per_s") is not None:
        wc1[r["sampler"]].append((r["n_workers"], r["evals_per_s"]))
line_plot(
    {s: sorted(v) for s, v in wc1.items()},
    "throughput (evals/s)",
    "Throughput vs workers (1 core/worker): GP's serial refit caps it,\n"
    "the others scale with concurrency  (DDM, popsize=NW)",
    "sampler_throughput_vs_workers.png", logy=True,
)

# ---- FIG 3: throughput vs total cores, per sampler ---------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for s in samplers:
    pts = sorted((r["total_cores"], r["evals_per_s"]) for r in rows
                 if r["sampler"] == s and r.get("evals_per_s") is not None)
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o", color=color[s], label=s, alpha=0.8)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("total cores (n_workers x cores/worker)")
ax.set_ylabel("throughput (evals/s)")
ax.set_title("Throughput vs total cores across the grid, per sampler  (DDM, popsize=NW)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(title="sampler")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sampler_throughput_scaling.png"), dpi=120)
plt.close(fig)

# ---- FIG 4: cost (core-hours) vs workers -------------------------------------
line_plot(
    by_sampler_nw("core_hours"),
    "core-hours for the fit  — lower is cheaper",
    "Cost of the popsize=NW regime per sampler  (DDM)\n"
    "median over cores/worker; GP burns cores idling on the serial GP fit",
    "sampler_cost_vs_workers.png", logy=True,
)

print(f"plotted {len(rows)} configs across {len(samplers)} samplers: {', '.join(samplers)}")
print("wrote:", ", ".join(sorted(f for f in os.listdir(OUT) if f.startswith("sampler"))))
