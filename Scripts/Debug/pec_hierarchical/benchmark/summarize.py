"""Summarize benchmark results: median per config, speedup vs the regular baseline.

Usage: python summarize.py [results.jsonl ...]
"""

import json
import os
import statistics
import sys
from collections import defaultdict


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def load(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def main():
    paths = sys.argv[1:] or [os.path.join(os.path.dirname(__file__), "results.jsonl")]
    rows = load(paths)
    if not rows:
        print("no results found in:", paths)
        return

    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["n_workers"], r["worker_cores"], r["num_estimates"])].append(r)

    agg = []
    for (mode, nw, wc, ne), rs in groups.items():
        agg.append({
            "label": f"{mode} {nw}x{wc}",
            "mode": mode, "ne": ne, "cores": rs[0]["total_cores"],
            "loop_s": med([r["loop_s"] for r in rs]),
            "compile_s": med([r["compile_s"] for r in rs]),
            "evals_per_s": med([r["evals_per_s"] for r in rs]),
            "util_pct": med([r["util_pct"] for r in rs]),
            "rss_gb": med([r["peak_rss_gb"] for r in rs]),
            "err": med([r["max_pct_err"] for r in rs]),
            "core_hours": med([r["core_hours"] for r in rs]),
        })

    # baseline loop time per num_estimates = the regular run
    base = {a["ne"]: a["loop_s"] for a in agg if a["mode"] == "regular"}

    def fnum(v, width, prec):
        return ("-" if v is None else f"{v:.{prec}f}").rjust(width)

    hdr = (f"{'config':<22}{'cores':>6}{'ne':>6}{'loop_s':>9}{'evals/s':>9}"
           f"{'speedup':>9}{'util%':>7}{'rss_gb':>8}{'err%':>7}{'core_h':>9}")
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(agg, key=lambda x: (x["ne"], x["cores"], x["label"])):
        b = base.get(a["ne"])
        sp = (b / a["loop_s"]) if (b and a["loop_s"]) else None
        print(
            f"{a['label']:<22}{a['cores']:>6}{a['ne']:>6}"
            f"{fnum(a['loop_s'], 9, 1)}{fnum(a['evals_per_s'], 9, 1)}"
            f"{fnum(sp, 9, 2)}{fnum(a['util_pct'], 7, 0)}"
            f"{fnum(a['rss_gb'], 8, 2)}{fnum(a['err'], 7, 1)}{fnum(a['core_hours'], 9, 4)}"
        )
    print("\nspeedup = regular loop_s / config loop_s (per num_estimates).")
    print("util%   = mean busy cores / allocated cores; '-' for multi-node "
          "(jobqueue) since remote workers are not locally sampled.")


if __name__ == "__main__":
    main()
