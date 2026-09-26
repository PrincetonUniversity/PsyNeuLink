# Batched sampling blocks and compiler reuse on the H100

The fastest previously validated NDT-profiled fit now takes **3.97
minutes overall**, down from **5.82 minutes**: **1.47× faster**.
All 1,300 proposals, likelihood scores, sampling budgets, block seeds, NDT
selections, final parameters, validation scores, predictions, and invalid
proposal rejections are exactly unchanged. The changes improve execution;
they do not reduce the simulated sample count or change the fitting policy.

This is one complete synthetic subject/start on an otherwise unused H100 NVL
on della-rse. It has 760 trials (720 scored), 10 ms LCA steps, and noise SD 0.1
on all four LCAs. LC integration remains ten 20 ms steps per model pass.
Search uses four initial blocks totaling 5,000 estimates, adaptive promotions
up to 100,000, 1,000 search proposals, and 300 refinement proposals at 100,000.
Final selection pools three fresh 100,000-estimate blocks; five separate seeds
validate the chosen full parameter vector at 100,000 each.

## Full fits

| Implementation | Fitting | Overall, including setup and validation |
| --- | ---: | ---: |
| Previous NDT implementation | 312.09 s | 349.19 s |
| Batched blocks + source cache + runtime candidate count | 219.00 s | 257.84 s |
| Final: also reuse kernels across estimate budgets | **200.80 s** | **238.13 s** |

Each full fit starts with separate empty caches, so these times include JIT
compilation. The previous run is the uninstrumented `ndt-20260926/short1`
benchmark, not the slower Nsight/Python-profiled replay. The warm microbenchmark
below separately reruns the old and new implementations on the same H100.
An earlier runtime-budget variant took 200.44 s fitting / 237.27 s overall,
also with exact fit/validation equality. The final run includes an overflow-safe
lane decoder for the full supported int32 estimate range.

The fit still samples **60,070,000 candidate histories in 839 independent
blocks**. The compiler now launches **348 simulation kernels**, combining only
blocks that the existing adaptive race already intended to request. Mean
fresh-seed fitted log likelihood remains **254.76855469**.
This preserves the previous fit's quality; it does not resolve the earlier LC
parameter-recovery concerns or establish performance across subjects/starts.

## What changed

1. `BatchedSimulationPlan.discrete_output_count_blocks` adds a GPU launch
   dimension for independently seeded, equal-sized sampling blocks. Each block
   keeps its own counts and diagnostics. Candidate/subject/estimate RNG
   addresses, common-random-number sharing, and complete retained-state trial
   histories match separate calls. This is a general finite-support compiler
   reduction, not a DAWA-specific dynamics implementation.
2. Histogram and exact-support reductions reuse validated emitted source.
   A bounded process-local cache compares complete serialized IR contents and
   the immutable op-spec sidecar identity. Nested dictionary/array changes
   force validation/emission again. Nonserializable extension metadata takes
   the existing uncached path.
3. Total lane count and estimate count are runtime kernel arguments. Changing
   an adaptive budget or tail population no longer needs another binary just
   because of those counts. Specializations for model structure and parameter
   layout remain.

The adaptive driver groups equal-size blocks while preserving their original
seed order and pooling weights. Reference checks, high-budget refinement, and
final selection retain their existing policy. No approximations to Gaussian
draws, LC integration, nonlinear transfer functions, or histogram scoring were
introduced.

## Warm sampling calls

These replay ten actual refinement candidates on the same complete subject.
Each row uses seven alternating-order repeats after warmup, including parameter
preparation, simulation, diagnostics, and NDT density evaluation. Equality is
checked for every block. The old source runs separately on the same monitored
GPU; serial/batched new-source modes alternate within a process.

| Blocks × estimates per block | Previous serial | New serial | New batched | Previous / new batched |
| --- | ---: | ---: | ---: | ---: |
| 4 × 1,250 | 0.511 s | 0.461 s | 0.217 s | 2.36× |
| 2 × 2,500 | 0.298 s | 0.271 s | 0.172 s | 1.73× |
| 2 × 5,000 | 0.370 s | 0.341 s | 0.317 s | 1.17× |
| 2 × 10,000 | 0.659 s | 0.632 s | 0.603 s | 1.09× |
| 2 × 20,000 | 1.214 s | 1.201 s | 1.167 s | 1.04× |
| 1 × 100,000 | 2.848 s | 2.863 s | 2.864 s | 0.99× |

The four small blocks benefit most from filling the GPU and amortizing host
preparation. Large calls already occupy the device: the 100k workload remains
about 2.85–2.9 s, with roughly percent-level variation between these variants.
These changes are not a uniform speedup of every simulated integration step.
The earlier profile's RNG, LC integration, and nonlinear transfer costs remain
important targets, especially during high-budget refinement.
The [updated full-fit profile](sampling_blocks_profile_h100.md) measures these
remaining costs: simulation occupies 91% of fitting time, with adaptive search
and 100k refinement contributing almost equally.

A separate 30-repeat host probe measured source preparation at
**15.42 ms uncached** versus
**0.95 ms cached**, including the content check.

Four simultaneous count buffers use **243.3 MB** instead of **60.8 MB** for a
single ten-candidate block. Storage still depends on candidates, trials, support
size, and simultaneous blocks, not the estimate count. The returned block views
share that allocation; retaining one view retains the allocation. The kernel-call
benchmark's peak Torch allocation for four blocks is about 275 MB; CUDA context
and compiler allocations are additional.

## Run it

Block batching is enabled by default with `--fit-strategy adaptive --profile-ndt`.
The ordinary fixed-budget fitter remains the default when those options are
absent. Reproduce this recovery configuration from the repository root:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_pec_recovery.py \
  --fit-strategy adaptive --profile-ndt --estimates 100000 --evaluations 5000 \
  --adaptive-search-evaluations 1000 --adaptive-refine-evaluations 300 \
  --start 1 --optimizer-seed 202 --simulation-seed 37 --optimizer-storage memory \
  --validation-estimates 100000 --validation-seeds 9401 9402 9403 9404 9405 \
  --output /tmp/dawa-blocks-recovery
```

Add `--no-batch-sampling-blocks` to compare separate calls while keeping source
caching and runtime sizes. It also reduces count-buffer memory. For warm timings:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_sampling_block_benchmark.py \
  --run /tmp/dawa-blocks-recovery --output /tmp/dawa-block-timings.json
```

The benchmark also supports `--serial-only` for the earlier source snapshot.
Raw artifacts, launch scripts, monitored process logs, and isolated source
snapshots are under `/scratch/gpfs/CSES/dmturner/dawa-benchmarks/blocks-20260926`.
[Measurements and source hashes](sampling_blocks_h100.json) preserve the full
comparison without checking in the large fit logs.

## Verification

The local reduction/cache suite passed 52 checks (47 inactive interpreter
variants skipped). Three adaptive driver integration cases passed, covering
recovery with/without NDT profiling and real-data fitting. An explicitly enabled
CPU interpreter block test also passed. Tests compare against materialized
histories with two subjects, both scheduler modes, both CRN policies, repeated
and high-bit 64-bit seeds, and partial GPU blocks. They also exercise invalid
candidates, unsupported/nonfinite outputs, invalid seeds, unequal adaptive
block sizes, and cache invalidation after nested IR mutations. Ruff and
`git diff --check` passed. The H100 replay checks all 1,300 optimizer records,
not just the final parameter vector.
