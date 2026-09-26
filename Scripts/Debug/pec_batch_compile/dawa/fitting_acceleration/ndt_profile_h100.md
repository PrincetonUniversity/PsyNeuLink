# Where the fastest accurate DAWA fit spends its time

The strongest next optimization is **batching the independent small sampling
blocks within each adaptive race**. A monitored H100 experiment ran four
1,250-estimate blocks **1.91× faster** concurrently, with exactly identical
densities. Removing repeated source generation and reducing first-use
compilation are additional opportunities. NDT scoring and CMA-ES updates are
already small costs.

This profiles the faster successful start from [the NDT experiment](ndt_h100.md):
1,000 search proposals, 300 refinement proposals, seven dynamic coordinates,
NDT profiling, 10 ms LCA steps, and adaptive sampling up to 100k estimates.
The complete subject has 760 trials, 720 scored, and retains all trial history.
Its previous uninstrumented runtime was **312.09 s fitting / 349.19 s overall**.

The instrumented repeat reproduced **all 1,300 proposals, scores, sampling
budgets, seeds, NDT selections, final parameters, and validation scores exactly**.
Profiling increased runtime to 365.46 s fitting / 417.75 s overall. Those longer
times diagnose costs; they are not a new throughput benchmark. The GPU was
otherwise idle throughout the full fit and the final concurrency experiment.
Measurements and commands are in [ndt_profile_h100.json](ndt_profile_h100.json).

## Full-fit trace

Nsight Systems measured **231.78 s in the simulation kernel**, across 839
sampling calls and 60.07 million candidate histories. Other GPU kernels took
0.059 s; actual device transfers took 0.111 s. The union of GPU work occupied
231.95 s of the 365.43-second instrumented fitting interval.

| Sampling phase | Calls | GPU simulation | Inclusive sampling-call wall time |
| --- | ---: | ---: | ---: |
| Adaptive search | 780 | 132.39 s | 228.60 s |
| Screening | 21 | 3.48 s | 8.65 s |
| Reference checks | 5 | 3.62 s | 11.46 s |
| Refinement | 30 | 85.41 s | 95.13 s |
| Final selection | 3 | 6.87 s | 10.78 s |

The wall-time column includes simulation, preparation, source emission,
validation, compilation, and waiting. It excludes the subsequent shifted
histogram calculation and optimizer work. Its excess over GPU time cannot all
be interpreted as removable overhead, especially under the Python profiler.

Across the entire fit:

- Shifted NDT scoring took **0.409 s** on the host, including its GPU work.
- Pooling blocks took **1.032 s**; the ranking-uncertainty calculation took
  **0.042 s**.
- Optuna ask/tell took **4.51 s** combined.
- Python attributed 235.92 s to Tensor `.cpu()` calls, primarily waiting for
  simulation. Actual transfers consumed only 0.111 s on the GPU. Optimizing
  transfer bandwidth would not remove those waits.
- Peak Torch allocation was **92.4 MB**, with 159.4 MB reserved. CUDA context
  and profiler allocations are additional; these are not total device-memory
  figures. The ten-candidate count buffer itself is 60.8 MB.

## Small blocks underuse the GPU

Search used 101 races. Their total budgets ranged from 5k to 80k, but those
budgets were assembled from smaller independent blocks:

| Estimates per block | Search calls | GPU simulation time |
| --- | ---: | ---: |
| 1,250 | 404 | 55.62 s |
| 2,500 | 160 | 23.30 s |
| 5,000 | 130 | 22.51 s |
| 10,000 | 72 | 22.73 s |
| 20,000 | 14 | 8.22 s |

Nsight Compute profiled an actual ten-candidate refinement population at two
block sizes. Both kernels allocate 128 registers per thread, limiting theoretical
occupancy to 25%. Measured occupancy was **4.65% at 1,250 estimates**, versus
**22.41% at 20,000**. The small launch supplies too few warps to hide instruction
latency. Scheduler issue utilization rose from 35.56% to 74.62% between those
sizes. DRAM throughput remained below 0.002% of peak in both cases.

A separate experiment evaluated identical independently seeded blocks either
sequentially or with a thread pool and separate CUDA streams. It used five
alternating-order warm timing repeats, the same ten actual candidates, and
checked exact density equality between modes:

| Independent blocks | Sequential median | Concurrent median | Speedup |
| --- | ---: | ---: | ---: |
| 4 × 1,250 estimates | 0.515 s | 0.269 s | **1.91×** |
| 2 × 2,500 estimates | 0.298 s | 0.206 s | **1.44×** |
| 2 × 10,000 estimates | 0.662 s | 0.625 s | **1.06×** |

This is an execution experiment, not a changed adaptive policy or another
complete fit. The same seeds, estimates, histories, smoothing, and pseudocounts
produce the same densities. The benefit decreases as each individual block
already fills the GPU.

A general compiler implementation should accept a batch of independent
estimate blocks, retain a seed and count output for each, and expose the block
axis to the GPU. Initial four-block races and two-block promotions can then
launch together without changing the optimizer's statistical decisions. This
would also avoid Python thread coordination and repeated per-block preparation.
Its memory cost grows with the number of simultaneously retained count buffers.
It does not require splitting a subject's sequential history.

## Repeated compiler work and first-use compilation

The CPU profile identified **850 source emissions**, including fitting and
other stages. `TritonGraphEmitter.emit()` revalidates the kernel IR, walks the
operations, transforms normal-draw templates, and regenerates source on each
call. Caching the compiled binary does not avoid this work.

The heavily instrumented totals were 50.16 s in emission, including 35.01 s in
IR validation. A separate uninstrumented 30-repeat probe gave:

| Operation | Median per call |
| --- | ---: |
| Kernel-IR validation | 10.00 ms |
| Source emission, including validation | 15.32 ms |
| Normalizing a ten-candidate parameter batch | 5.33 ms |

The emission measurement extrapolates to **about 12.9 s over 839 sampling
calls**, considerably less than its Python-profiled total. A prepared objective
could also reuse parameter structure and device inputs, while still validating
changing parameter values. There were 104,875 small device copies in the fit;
their aggregate GPU time was tiny, but preparing and submitting them has a
separate host cost.

Caching must preserve the compiler's correctness checks. The current emitter
explicitly revalidates because nested KernelIR mappings can be mutated after
construction. Freeze a validated executable snapshot, or provide reliable
invalidation, before reusing emitted code. Do not simply suppress validation
on the existing mutable representation.

Cold compilation is a separate cost. Sixteen first calls had 3.67–4.19 s of
non-kernel time each, **61.19 s combined in the instrumented trace**. These
include JIT and other first-use work, not just source generation. Variants
appeared for different estimate and candidate counts, including the nine-member
tail populations and eight finalists. `total_lanes` and `num_estimates` are
currently constexpr arguments. Reducing unnecessary specialization by candidate
count is worth testing; retain specialization where it helps execution. Reusing
the existing persistent compilation cache also amortizes this cost across fits.

## What remains inside the simulation kernel

At the larger 20k workload, source instruction sampling attributed:

| Source category | Share of sampled instruction positions |
| --- | ---: |
| Philox, uniform conversion, Gaussian transform | 32.30% |
| LC Euler integration | 14.37% |
| Logistic calculations | 16.87% |
| Count lookup and diagnostics | 10.84% |
| Block reductions | 1.70% |
| Other model arithmetic and scheduling | 23.92% |

These are source-sample shares, **not exclusive wall-time percentages or
additive speedup predictions**. The classifier groups lines by helper function
and source text; compiler line attribution has limits. Full hot-line samples
and the classification results are in the JSON.

The current fast sine/cosine transform has already removed much of the earlier
Gaussian bottleneck. The largest remaining Gaussian source line is its
logarithm/square root. Further RNG work should examine unused Philox outputs
and register lifetime, with an explicit RNG version if draw addressing changes.
LC and LCA arithmetic remain substantive. None of this profiling changes
nonlinearities, clocks, noise, or trial-state retention.

## Recommended order

1. **Batch independent small estimate blocks in the compiler.** This has the
   strongest measured opportunity while preserving the existing estimator and
   adaptive decisions. Retest a complete fit for exact replay after implementation.
2. **Prepare and cache validated executable code.** Remove repeated emission
   and invariant preparation, preserve mutation detection, and investigate the
   extra JIT variants caused by candidate-batch sizes. Measure both cold-cache
   and warm-cache fits.
3. **Then revisit RNG and arithmetic within the kernel.** These dominate once
   enough work is available. NDT scoring, smoothing, copy bandwidth, and CMA-ES
   bookkeeping are lower priorities on this profile.

The 1.91× microbenchmark is not a 1.91× whole-fit prediction. Refinement alone
uses 85 s of GPU simulation and already runs large batches. The next changes
could save tens of seconds; this profile does not establish another order of
magnitude in fitting speed.

## Reproduce and inspect

The new [profiling wrapper](../dawa_fit_profile.py) adds timers, NVTX ranges,
and a Python profile around the existing driver:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output /tmp/dawa-fit-trace \
  python Scripts/Debug/pec_batch_compile/dawa/dawa_fit_profile.py \
  --profile-output /tmp/dawa-fit-profile -- \
  --fit-strategy adaptive --profile-ndt --estimates 100000 --evaluations 5000 \
  --adaptive-search-evaluations 1000 --adaptive-refine-evaluations 300 \
  --start 1 --optimizer-seed 202 --simulation-seed 37 \
  --optimizer-storage memory --validation-estimates 100000 \
  --validation-seeds 9401 9402 9403 9404 9405 --output /tmp/dawa-profiled-fit
```

It writes `profile.json` and `python.pstats`; nested wall timings overlap.
The wrapper passed a local 16-trial recovery smoke run and the exact full-fit
replay above. The production compiler, fitting policy, and model are unchanged
by this profiling work.

Raw Nsight reports, SQLite trace, profiles, replay scripts, and monitoring logs
are under `/scratch/gpfs/CSES/dmturner/dawa-benchmarks/profile-ndt-20260926` on
della-rse. Tools were Nsight Systems 2025.3.1, Nsight Compute 2026.2.1, and CUDA
toolkit module 13.0. Full-fit and final concurrency monitoring observed only our
process on GPU1. An initial concurrency run on GPU0 was excluded after other
processes appeared. A 100k Nsight Compute replay was stopped because counter
collection was slow; the hardware comparison uses 1,250 and 20k. The full-fit
Nsight Systems trace still includes all original 100k refinement/selection calls.
