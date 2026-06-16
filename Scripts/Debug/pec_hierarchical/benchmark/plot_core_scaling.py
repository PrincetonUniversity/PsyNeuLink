"""Consolidate the CMA-ES core-scaling study results into one CSV + a set of plots.

This is the historical CMA-ES study (fixed popsize=32, swept worker x core). The
per-sampler study is plotted separately by plot_samplers.py.

Sources (all under results/):
  results/core_grid.d/*.jsonl     dask-srun, full 23-config core-scaling grid
  results/bell.d/*.jsonl          over-subscribed bell-curve completion runs
  results/singlenode.jsonl        regular + dask-local baselines (single node)
  results/stabflex.jsonl          stabflex regular vs dask-srun (heavy model)

Writes:  plots/consolidated.csv  and  plots/fig*.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)


def load(paths, source):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                r["source"] = source
                rows.append(r)
    return rows


grid = load(sorted(glob.glob(os.path.join(RES, "core_grid.d/*.jsonl"))), "grid_srun")
bell = load(sorted(glob.glob(os.path.join(RES, "bell.d/*.jsonl"))), "bell_srun")
single = load([os.path.join(RES, "singlenode.jsonl")], "singlenode")
stab = load([os.path.join(RES, "stabflex.jsonl")], "stabflex")
allrows = grid + bell + single + stab
POPSIZE = 32

# ---- consolidated CSV --------------------------------------------------------
cols = ["source", "model", "mode", "n_workers", "worker_cores", "total_cores",
        "num_estimates", "num_trials", "optimizer_popsize", "total_evals",
        "compile_s", "loop_s", "total_s", "evals_per_s", "core_hours", "max_pct_err"]
with open(os.path.join(OUT, "consolidated.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for r in allrows:
        f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
print(f"wrote consolidated.csv ({len(allrows)} rows)")


def cfg(r):
    return f"{r['n_workers']}x{r['worker_cores']}"


# ---- FIG: grid heatmap of throughput (evals/s) -------------------------------
NW = [1, 2, 4, 8, 16, 32]
WC = [1, 2, 4, 8]
gd = {(r["n_workers"], r["worker_cores"]): r for r in grid}
H = np.full((len(WC), len(NW)), np.nan)
for i, wc in enumerate(WC):
    for j, nw in enumerate(NW):
        if (nw, wc) in gd:
            H[i, j] = gd[(nw, wc)]["evals_per_s"]
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(H, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(len(NW))); ax.set_xticklabels(NW)
ax.set_yticks(range(len(WC))); ax.set_yticklabels(WC)
ax.set_xlabel("n_workers (concurrency)")
ax.set_ylabel("cores per worker (LLVM threads)")
ax.set_title("DDM PEC throughput (evals/s) across the core-scaling grid\n"
             "diagonal = equal total cores; brighter = faster")
for i in range(len(WC)):
    for j in range(len(NW)):
        if not np.isnan(H[i, j]):
            ax.text(j, i, f"{H[i, j]:.1f}", ha="center", va="center",
                    color="white" if H[i, j] < H[np.isfinite(H)].max() * 0.6 else "black", fontsize=8)
fig.colorbar(im, label="evals/s")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_grid_heatmap.png"), dpi=120)
plt.close(fig)

# ---- FIG 3: throughput vs total cores, lines by cores/worker -----------------
fig, ax = plt.subplots(figsize=(9, 5.5))
for wc in WC:
    pts = sorted([(r["total_cores"], r["evals_per_s"]) for r in grid if r["worker_cores"] == wc])
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "o-", label=f"{wc} core(s)/worker")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("total cores (n_workers x cores/worker)")
ax.set_ylabel("throughput (evals/s)")
ax.set_title("DDM PEC scaling: at equal cores, more WORKERS beats fatter workers\n"
             "(LLVM threading saturates fast; concurrency scales better)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_throughput_scaling.png"), dpi=120)
plt.close(fig)

# ---- FIG 4: parallel efficiency + cost (core-hours) --------------------------
base = gd[(1, 1)]["loop_s"]  # 1x1 = serial-equivalent baseline
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
# speedup vs ideal, 1-core-worker line (pure concurrency)
for wc in WC:
    pts = sorted([(r["total_cores"], base / r["loop_s"]) for r in grid if r["worker_cores"] == wc])
    xs, ys = zip(*pts)
    a1.plot(xs, ys, "o-", label=f"{wc} core(s)/worker")
ideal = sorted({r["total_cores"] for r in grid})
a1.plot(ideal, ideal, "k--", alpha=0.5, label="ideal linear")
a1.set_xscale("log", base=2); a1.set_yscale("log", base=2)
a1.set_xlabel("total cores"); a1.set_ylabel("speedup vs 1x1 serial")
a1.set_title("Parallel speedup vs ideal")
a1.grid(True, which="both", alpha=0.3); a1.legend()
# cost: core-hours by config (lower = cheaper for same work)
gsort = sorted(grid, key=lambda r: r["core_hours"])
a2.barh([cfg(r) for r in gsort], [r["core_hours"] for r in gsort],
        color=["#4a7" if r["worker_cores"] == 1 else "#88c" if r["worker_cores"] == 2
               else "#c94" if r["worker_cores"] == 4 else "#c44" for r in gsort])
a2.set_xlabel("core-hours for the same 960-eval fit (lower = cheaper)")
a2.set_title("Cost efficiency (color = cores/worker: green=1 .. red=8)")
a2.tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_efficiency.png"), dpi=120)
plt.close(fig)

# ---- FIG 5: stabflex serial vs dask-srun (heavy model uplift) ----------------
if len(stab) >= 2:
    sd = {r["mode"]: r for r in stab}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    modes = ["regular", "dask-srun"]
    present = [m for m in modes if m in sd]
    a1.bar(present, [sd[m]["loop_s"] for m in present], color=["#c44", "#4a7"])
    a1.set_ylabel("fit loop time (s)")
    a1.set_title(f"stabflex loop time\n(dask-srun {sd['dask-srun']['n_workers']}x"
                 f"{sd['dask-srun']['worker_cores']} = "
                 f"{sd['regular']['loop_s']/sd['dask-srun']['loop_s']:.1f}x faster)")
    # recovered params vs truth
    names = list(sd["regular"]["recovered"])
    truth = {"gain": 3.0, "slope": 0.01, "threshold": 0.1, "non_decision_time": 0.2}
    xx = np.arange(len(names))
    a2.plot(xx, [sd["regular"]["recovered"][n] for n in names], "s-", label="regular")
    a2.plot(xx, [sd["dask-srun"]["recovered"][n] for n in names], "o--", label="dask-srun")
    a2.plot(xx, [truth[n] for n in names], "k*", markersize=12, label="truth")
    a2.set_xticks(xx); a2.set_xticklabels(names, rotation=20)
    a2.set_title("recovered params: serial vs parallel vs truth")
    a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_stabflex.png"), dpi=120)
    plt.close(fig)

# ---- FIG 6: ideal cores/worker bell curve (grid + over-subscribed bell runs) -
# At a fixed core budget, throughput vs cores/worker: peak at WC = cores/popsize
# (== exactly POPSIZE workers). Left of peak = over-subscribed (n_workers>popsize
# -> idle workers waste cores); right = under-concurrency + thread over-provision.
merged = {(r["n_workers"], r["worker_cores"]): r for r in grid}
merged.update({(r["n_workers"], r["worker_cores"]): r for r in bell})
fig, ax = plt.subplots(figsize=(9.5, 5.5))
for tc in (64, 128):
    pts = sorted([(r["worker_cores"], r["evals_per_s"], r["n_workers"])
                  for r in merged.values() if r["total_cores"] == tc])
    if not pts:
        continue
    wcs, es, nws = zip(*pts)
    line, = ax.plot(wcs, es, "o-", label=f"{tc} cores total")
    peak_wc = tc // POPSIZE
    for wc, e, nw in pts:
        tag = " (peak)" if wc == peak_wc else (" idle>popsize" if nw > POPSIZE else "")
        ax.annotate(f"{nw}w{tag}", (wc, e), fontsize=7,
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    color=line.get_color())
    ax.scatter([peak_wc], [dict((w, e) for w, e, _ in pts)[peak_wc]],
               s=180, facecolors="none", edgecolors=line.get_color(), linewidths=2, zorder=5)
ax.set_xscale("log", base=2)
ax.set_xlabel("cores per worker (WC)   —   n_workers = total_cores / WC")
ax.set_ylabel("throughput (evals/s)")
ax.set_title("Ideal cores-per-worker (DDM, popsize=32): peak at WC = total_cores / popsize\n"
             "left of peak = n_workers > popsize -> idle workers waste cores")
ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_bell_curve.png"), dpi=120)
plt.close(fig)

print("wrote plots:", ", ".join(sorted(os.listdir(OUT))))
