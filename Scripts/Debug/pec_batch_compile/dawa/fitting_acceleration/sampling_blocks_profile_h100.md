# H100 profile after sampling-block batching and compiler reuse

**GPU simulation now occupies 91% of fitting time.** Adaptive search and
100k refinement contribute almost equally. NDT scoring, source generation,
and device transfers are small costs after the latest changes.

This is a full replay of the [fastest validated configuration](sampling_blocks_h100.md)
on an otherwise unused della-rse H100 NVL, followed by two Nsight Compute
kernel replays. It uses the same synthetic subject/start: 760 trials, 720
scored, 10 ms LCA steps, noise SD 0.1 on all four LCAs, seven dynamic search
coordinates, and profiled NDT. LC retains ten 20 ms integration steps per pass.
Search has 1,000 proposals with adaptive 5k–100k budgets; refinement has 300
proposals at 100k. No simulation or fitting policy changed for this profile.

All **1,300 evaluation records match exactly** after excluding elapsed-time
fields, including proposals, scores, budgets, seeds, and NDT selections.
Final parameters, independent validation, predictions, and invalid-proposal
rejections also match. This checks replay of the latest adaptive fit; the
earlier comparison with the original fixed-100k optimizer remains a separate
fit-quality comparison.

## Full fitting breakdown

| Work | GPU simulation time | Share of fitting wall time |
| --- | ---: | ---: |
| Adaptive search | 90.37 s | 43.2% |
| 100k refinement | 86.52 s | 41.4% |
| Reference checks, screening, final selection | 14.23 s | 6.8% |
| Host work, compilation, gaps, and other GPU activity | — | 8.6% |
| Total fitting | **209.20 s wall / 191.11 s simulation** | **100%** |

The trace contains 348 simulation launches representing 839 independent
sampling blocks and 60,070,000 candidate histories. Search's initial four
1,250-estimate blocks take 23.76 s; subsequent promotions take 66.61 s.
Optimizing only the initial small-budget launches would miss most search time.

Nsight Systems plus cProfile increase runtime: this run takes **261.19 s
overall**, versus **238.13 s overall / 200.80 s fitting** in the uninstrumented
benchmark. Use the latter for throughput estimates. The profiled run spends
39.58 s in setup and 12.41 s after fitting. Setup includes model construction,
synthetic-data checks, and cold compilation; post-fit work includes validation
and predictions. GPU kernels outside fitting total only 0.15 s before fitting
and 4.10 s afterward.

The previous, pre-batching profile spent 231.78 s in simulation and 133.48 s
without GPU activity during its 365.43 s fit. The new trace spends 191.11 s
in simulation and 17.96 s without GPU activity. Instrumentation especially
inflates repeated Python/compiler work, so these differences explain the
bottleneck shift; the uninstrumented benchmarks establish the actual speedup.

## Host work and memory

Explicit inclusive fitting timers measure:

| Operation | Time |
| --- | ---: |
| Optimizer proposal generation | 4.06 s |
| Optimizer updates | 0.42 s |
| Pooling densities and selecting NDT | 0.91 s |
| Shifted-histogram density scoring | 0.32 s |
| Ranking uncertainty | 0.04 s |

Across the whole run, cProfile records 0.83 s for cached source preparation,
including eight actual source emissions totaling 0.45 s. The previous profile
emitted source 850 times. Parameter normalization takes 3.94 s across 357
calls, and input/parameter tensor preparation takes 0.65/0.98 s. These are
inclusive timings and must not be added to enclosing call durations.

All GPU copies **inside fitting** total just **0.056 s** for 748 MB across
44,830 copies; other GPU kernels total 0.057 s. Large `.cpu()` and CUDA copy
API durations mostly reflect waiting for simulation to finish. They do not
indicate equivalent time spent transferring data or independent CPU work.

Peak Torch allocation is **275 MB** (296 MB reserved), including up to 243 MB
for four simultaneous count buffers. CUDA context and profiler allocations
are additional. The count buffers are larger after batching, but neither
copy bandwidth nor global-memory bandwidth is the measured bottleneck.

## Inside the simulation kernel

Nsight Compute replays use ten actual refinement proposals, the complete
subject, and the real fitter's sampling callback. Each case warms up once and
checks exact density equality on the measured replay.

| Metric | 4 × 1,250 estimates | 1 × 20,000 estimates |
| --- | ---: | ---: |
| Kernel duration | 0.210 s | 0.591 s |
| Registers per thread | 128 | 128 |
| Achieved occupancy | 16.6% | 22.3% |
| Scheduler issue activity, active cycles | 67.5% | 73.5% |
| Eligible warps per scheduler per active cycle | 1.11 | 1.58 |
| DRAM throughput, fraction of peak | 0.0027% | 0.0007% |
| Register spilling | None recorded | None recorded |

Register allocation limits residency to 16 one-warp blocks per SM: a 25%
theoretical occupancy ceiling for this launch. Low occupancy does not imply
a proportional speedup is available, but reducing simultaneously live state
could allow the GPU to hide instruction dependencies better. The earlier
single 1,250-estimate replay reached only 4.7% occupancy and 35.6% scheduler
issue activity; the four-block launch uses the device substantially better.
That historical replay used a different setup helper, so this is descriptive
context rather than a controlled hardware-counter ablation.

Source-attributed instruction samples locate the remaining hot regions:

| Source category | 4 × 1,250 | 1 × 20,000 |
| --- | ---: | ---: |
| Gaussian RNG: Philox, uniform conversion, normal transform | 38.4% | 39.2% |
| Logistic transfer calculations | 16.2% | 15.5% |
| LC Euler integration | 12.8% | 12.6% |
| Count accumulation, exact-support lookup, diagnostics | 8.9% | 9.2% |
| Reductions | 1.9% | 1.6% |
| Other simulation and control | 21.7% | 21.8% |

These are **instruction sample shares, not exclusive wall-time percentages**
or additive speedup predictions. The 20k capture flagged sample-buffer
overflow, although reported dropped bytes were zero; treat its shares as
approximate. The 4 × 1,250 capture has no overflow flags and gives the same
broad picture. No new 100k hardware-counter capture was taken; its complete
refinement cost is measured directly in the full-fit CUDA trace above.

## Next optimization priorities

1. **Simulation arithmetic and register pressure.** Inspect RNG instruction
   scheduling, live state, repeated parameter expressions, LC updates, and
   nonlinear transfers. Improvements here affect both search and refinement.
   Reducing RNG rounds, changing integration, or approximating transfer
   functions would require separate numerical and fit-quality validation.
2. **Exact-support indexing.** Binary lookup and count diagnostics are a
   visible secondary cost. A compiler specialization could use a cheaper
   index when finite output support permits it, while retaining exact FP32
   bin membership and diagnostics.
3. **Amortize setup across subjects where possible.** This may help study
   throughput, but the measured setup includes recovery-specific generation
   checks and cold compilation; it is not all removable per-subject overhead.

Further changes limited to Python scoring or data transfers have little room
to help. Refinement-budget reductions could save substantial time, but would
change the fitting strategy and need new independent quality comparisons.
The current uninstrumented 238 s overall time still needs about 73 s removed
to reach 10× relative to the original 1,652 s run. Eliminating all remaining
host gaps within fitting would not close that gap.

## Reproduction and artifacts

Use the same arguments as the linked sampling-block benchmark, replacing its
entry point with the profiling wrapper:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output /tmp/dawa-profile-trace \
  python Scripts/Debug/pec_batch_compile/dawa/dawa_fit_profile.py \
  --profile-output /tmp/dawa-profile-report -- \
  --fit-strategy adaptive --profile-ndt --estimates 100000 --evaluations 5000 \
  --adaptive-search-evaluations 1000 --adaptive-refine-evaluations 300 \
  --start 1 --optimizer-seed 202 --simulation-seed 37 --optimizer-storage memory \
  --validation-estimates 100000 --validation-seeds 9401 9402 9403 9404 9405 \
  --output /tmp/dawa-profile-fit
```

The wrapper now labels simultaneous block counts in its NVTX ranges and
timing summaries. It adds no synchronization. Phase totals use explicit NVTX
ranges and GPU activity timestamps, rather than summing overlapping Python
calls. Some enclosing cProfile cumulative counters were incomplete; the
completed child-function timings above are supporting evidence only.

[Machine-readable measurements](sampling_blocks_profile_h100.json) include
source hashes, replay checks, phase/shape timings, hardware metrics, source
samples, and limitations. Raw traces, analysis/replay scripts, fit outputs,
and monitored GPU process logs are in
`/scratch/gpfs/CSES/dmturner/dawa-benchmarks/profile-blocks-20260926`.
The frozen model/compiler source is the previous benchmark's
`blocks-20260926/source-final`. Monitoring recorded no competing GPU process.
