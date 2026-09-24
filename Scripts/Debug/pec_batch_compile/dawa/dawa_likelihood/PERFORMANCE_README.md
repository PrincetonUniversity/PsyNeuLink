# Continuous DAWA CPU and GPU performance

The direct solver now supports parallel native CPU execution and a fused Triton
GPU implementation, both including all seven parameter gradients. The
[recorded comparison](study_results/performance_parallel_gpu.json) uses the same
nonlinear equations, float64 precision, 129×129 grid, and numerical integration
settings for every direct backend.

For the cross-model question of why sampling wins here while direct evaluation
wins for CSI, see the [matched CSI/DAWA benchmark](../../CSI_DAWA_PERFORMANCE.md).
It controls trial counts and horizon and checks spatial and sampling refinement.

## Current comparison

Intel i7-9700K under WSL and RTX 2080 Ti. Each backend ran in a separate process,
without overlapping tests or benchmark jobs. Times are medians of three calls
after one warmup per case, including deterministic paths and observation scoring.
GPU times are synchronized. Compilation and CUDA graph construction are excluded
from these warmed measurements; the JSON also records warmup evaluation times.

| Backend, two fresh-trial conditions | Likelihood | Likelihood + all gradients | Gradient speedup over CPU 1 |
| --- | ---: | ---: | ---: |
| CPU, 1 thread | 3.792 s | 11.005 s | 1.00× |
| CPU, 2 threads | 4.065 s | 11.536 s | 0.95× |
| CPU, 4 threads | 2.218 s | 6.938 s | 1.59× |
| CPU, 8 threads | 2.595 s | 11.843 s | 0.93× |
| GPU, CUDA graphs | 0.411 s | 0.961 s | 11.45× |
| GPU, individual launches | 0.977 s | 3.818 s | 2.88× |

The fastest CPU gradient case in this audit is **CPU, 4 threads**. Threading
is an explicit option; one thread remains the default because small grids can
be dominated by coordination overhead.

The fused GPU backend is **9.23× faster for
forward likelihoods** and **11.45× faster
for likelihoods plus gradients** than the current one-thread CPU backend.

The separate continuous GPU sampler takes **0.0551 s**
for 100,000 estimates **per condition** (200,000 total), or
**0.0622 s** with empirical choice/RT histogram
scoring. The GPU direct forward calculation remains about
**6.6×** slower
than sampling plus scoring here, while supplying differentiable probabilities.
The sampler provides no parameter gradients. Different numerical approximations
and response-state precision prevent interpreting this as an equal-accuracy
comparison or as an optimizer/full-fit speedup.

## Implementation and scope

CPU `cpu_threads` controls OpenMP teams inside the coefficient and density
kernels. Rate calculations parallelize over coefficient times; forward and
backward density calculations partition grid cells within a persistent team.
Each source cell owns its outgoing rate derivatives. Boundary flux reductions
and time-step barriers preserve the absorbing scheme. The original one-thread
kernel remains available. This setting is independent of PyTorch's thread pool.

GPU `flux_backend="triton"` uses fused float64 rate, density, and adjoint kernels.
The backward pass differentiates the same SSP-RK2 updates and all three absorbing
categories as the CPU implementation. It replays checkpoint blocks instead of
retaining every density state for a full trial. Scalar interpolation arithmetic
and nonlinear coefficient calculations are explicitly float64; no mixed
precision, linearized activation, or relaxed integration tolerance is used.

CUDA graphs group launches over a checkpoint block. Cached buffers are keyed by
device, stream, tensor shapes, substep count, and time step; outputs are copied
before reuse, and the cache holds at most eight graphs. A new numerical topology
can require graph construction, so fixed-candidate warmed timings do not include
all costs encountered during optimization. Set `gpu_graphs=False` (CLI
`--no-gpu-graphs`) to use individual launches.

Generated deterministic ODE paths and their adjoints remain on CPU. The GPU
receives the small coefficient paths through differentiable transfers. Density
arrays, rate calculations, and their adjoints stay on the GPU. Peak GPU tensor
allocation in this benchmark was approximately
**1.56 GiB**; the JSON also reports reserved memory.

Both optimized backends support **first-order derivatives only**. The Torch
reference backend remains available. CPU extensions require a C++ compiler,
Ninja, and OpenMP; the GPU backend requires CUDA and Triton.

## Trial history and validation

Parallelism is within a trial's grid, not across dependent simulated trials.
The existing sequential likelihood still carries the control LCA's state,
advances excluded warm-up trials, and differentiates history using observed
RT minus candidate nondecision time. Stimulus, decision, response, and LC
integrator states reset as before. No history is removed to obtain these speedups.

The timing workload deliberately uses two fresh-state conditions and scores
4,000 saved observations. It is not a sequential subject fit. Each solve extends
to 1.401 seconds with 1 ms coefficient intervals, 0.5 ms maximum ODE steps,
CFL 0.8, five density substeps per interval, 16-interval blocks, and retained
rates. The observation bins and censoring match `recovery_two.json`.

Across the timing cases, maximum absolute differences from the one-thread CPU
reference were **2.22e-16** for bin probabilities and **1.91e-14** for parameter
gradients. The benchmark asserts parity with `--compare`.

The DAWA suites report **47 passed, 4 skipped**, covering:

* Parallel CPU and GPU forward/adjoint agreement for arbitrary transition rates,
  final survivor mass, both choice exits, and lower-domain escape.
* Nonlinear rate values and all coefficient derivatives.
* Reused CUDA graph buffers with changing inputs, eager versus graph execution,
  and first-order/dtype contracts.
* Sequential likelihoods and all parameter gradients, retained and recomputed
  coefficients, and excluded-trial control/NDT carryover. A finite-difference
  check verifies that an excluded trial's NDT affects later likelihoods.
* The existing analytic first-passage, conservation, and numerical-gradient checks.

A separate CLI check scored 12 real-data trials (nine excluded warm-ups and
three included observations) with a 65×65 grid. CPU with four threads and GPU
gave identical reported log likelihoods, with maximum absolute gradient
difference `6.82e-13`. This checks sequential command integration, not performance.

## Reproduction

Run from the repository root. Full reports default to `/tmp`; only the compact
comparison is kept with the source.

```bash
# One-thread reference; also times 100,000 GPU samples per condition.
PATH="$PWD/.venv/bin:$PATH" PYTHONWARNINGS=ignore .venv/bin/python \
  Scripts/Debug/pec_batch_compile/dawa/dawa_continuous_benchmark.py \
  --cpu-threads 1 --repeats 3 --output /tmp/dawa_cpu1.json

# Parallel CPU, with probability and gradient checks against that reference.
PATH="$PWD/.venv/bin:$PATH" PYTHONWARNINGS=ignore .venv/bin/python \
  Scripts/Debug/pec_batch_compile/dawa/dawa_continuous_benchmark.py \
  --cpu-threads 4 --skip-sampling --compare /tmp/dawa_cpu1.json \
  --output /tmp/dawa_cpu4.json

# Fused float64 GPU likelihood and all seven gradients.
PATH="$PWD/.venv/bin:$PATH" PYTHONWARNINGS=ignore .venv/bin/python \
  Scripts/Debug/pec_batch_compile/dawa/dawa_continuous_benchmark.py \
  --flux-backend triton --compare /tmp/dawa_cpu1.json \
  --output /tmp/dawa_gpu.json
```

Add `--no-gpu-graphs` to measure individual launches, `--recompute-rates` for the
lower-memory mode, or `--parameters` followed by seven values to change the fixed
candidate. `--compare` permits backend/thread/graph differences but requires the
same numerical settings, parameters, and observations.

For sequential scoring or fitting, `dawa_continuous_likelihood.py score|fit`
accepts `--device cpu --ode-backend generated --flux-backend native --cpu-threads 4`,
or `--device cuda --ode-backend generated --flux-backend triton`. Add
`--retain-rates` when memory permits. The existing `--threads` option controls
PyTorch's pool, separately from native `--cpu-threads`.

## Earlier serial CPU optimization

The [earlier audit](study_results/performance_cpu_optimization.json) reduced
one-thread forward time from 12.44 to 3.84 seconds and value-plus-gradient time
from 30.12 to 11.31 seconds. It removed repeated tensor access, reused rates
between stability scans and propagation, improved stable Bernoulli evaluation,
and enabled independent gathers. The current comparison starts from that
optimized implementation and adds threading and the GPU backend.
