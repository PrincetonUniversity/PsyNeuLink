# PEC batch compiler experiments

Model-specific code lives in [csi/](csi/README.md) and [dawa/](dawa/README.md).
Shared compiler benchmarks and design notes stay in this directory. The compiler
implementation itself lives in [psyneulink/core/batched](../../../psyneulink/core/batched/).

```text
pec_batch_compile/
  csi/
    csi_model_surrogate.py
    csi_triton_vs_llvm.py
    csi_batched_parameter_recovery.py
    audit_generated_csi_compatibility.py
    csi_fit/                 # fitting drivers, source model, and cluster launchers
      direct_likelihood/
      handoff/
  dawa/
    dawa_pec_fit.py           # single-subject GPU fits
    dawa_pec_recovery.py      # synthetic data and parameter recovery
    dawa_batched_simulation.py
    dawa_llvm_benchmark.py
    dawa_continuous_*.py
    dawa_direct_likelihood.py
    dawa_lca_model/           # source model and original fitting scripts
    dawa_likelihood/          # solvers, validation, and selected study results
  benchmark_*.py             # shared compiler/numerical benchmarks
  gpu_batch_compile_benchmark.py
  validate_likelihood_compile_gpu.py
```

The model directories retain their existing script and module names. Run the
documented commands from the repository root; include the `csi/` or `dawa/`
segment in script paths. Local data, fit outputs, and caches move with their
model directories and remain ignored by Git.

| Shared entry point | Purpose |
| --- | --- |
| [gpu_batch_compile_benchmark.py](gpu_batch_compile_benchmark.py) | General batch compiler workloads |
| [benchmark_continuous_dynamics.py](benchmark_continuous_dynamics.py) | Generated ODE integration and gradients, using a CSI fixture |
| [benchmark_first_passage_backend.py](benchmark_first_passage_backend.py) | Generic first-passage solver performance |
| [benchmark_csi_dawa_first_passage.py](benchmark_csi_dawa_first_passage.py) | Matched continuous CSI/DAWA sampling and direct costs; [results](CSI_DAWA_PERFORMANCE.md) |
| [benchmark_likelihood_specializations.py](benchmark_likelihood_specializations.py) | Generated and specialized likelihood routes, using CSI data |
| [benchmark_hierarchical_fit.py](benchmark_hierarchical_fit.py) | Complete hierarchical DDM fits with Triton and threaded LLVM |
| [validate_likelihood_compile_gpu.py](validate_likelihood_compile_gpu.py) | Conditional GPU sampling and scoring checks |

Start with the [likelihood compiler usage guide](LIKELIHOOD_COMPILE_USAGE.md)
and [batch compiler development notes](BATCH_COMPILE_WIP.md). The
[continuous equation compilation notes](CONTINUOUS_EQUATION_COMPILATION.md)
and [numerical backend notes](NUMERICAL_LIKELIHOOD_BACKEND.md) describe the
shared integration and first-passage machinery.

Deferred ideas for parameter gradients through Monte Carlo simulations,
continuous RT kernels, and reusable compiler support are recorded in
[sampling gradient notes](SAMPLING_GRADIENT_NOTES.md).

The simulation compiler supports explicit fixed parameter inputs:

```python
plan = BatchedCompositionCompiler.compile(model, backend="triton")
specialized = plan.specialize_parameters({"accumulator.leak": 0.3})
# Equivalent: compile(model, backend="triton", fixed_parameters={...})
```

Use canonical component-qualified names or registered aliases. The new plan
uses those scalar values when they are omitted from a parameter row and rejects
conflicting scalar, candidate-vector or trial-varying overrides. Comparisons
use FP32 execution precision. The source model and original plan remain usable;
constants are part of the generated source/cache key. Base parameter constants
do not freeze controller modulation. PEC offers
`batched_specialize_fixed_parameters=True` to specialize non-fitted defaults.

The generated Triton backend also accepts
`normal_rng="philox4x_fast_v1"` in its launch options. It uses the same uniform
streams as `philox4x_v1` with a faster bounded-angle Gaussian transform; rounding
and stopping steps can differ. It requires CUDA compilation. The interpreter
and handwritten CSI oracle reject this mode explicitly. General defaults remain
unspecialized with `philox4x_v1`; the DAWA subject benchmark opts into both
optimizations. See [DAWA compiler notes](dawa/COMPILER_NOTES.md#fixed-parameters-and-faster-gaussian-conversion).

## Hierarchical GPU versus LLVM benchmark

Generate one synthetic dataset, then run each backend in a separate process:

```bash
BENCH=Scripts/Debug/pec_batch_compile/benchmark_hierarchical_fit.py
RESULTS=Scripts/Debug/pec_batch_compile/benchmark_results/hierarchical_ddm
.venv/bin/python "$BENCH" --backend generate --data "$RESULTS/data.csv"
.venv/bin/python "$BENCH" --backend triton --data "$RESULTS/data.csv" --output "$RESULTS/triton.json" --hessian-batch-size 1
.venv/bin/python "$BENCH" --backend llvm --data "$RESULTS/data.csv" --output "$RESULTS/llvm.json" --hessian-batch-size 1
```

The defaults fit drift rate and threshold for six participants with 60 trials
each, using 2,000 simulations per trial and eight CPU threads. Both backends run
the same hierarchical Laplace EM solver with full posterior curvature. The
commands above select individual Hessian evaluations to reproduce the original
benchmark; the current default batches up to 16 parameter vectors per call. The
benchmark records group and participant convergence, likelihood evaluations,
parameter recovery, model construction, first scoring calls, and warm fit time.
The first scoring calls include compilation or compiler-cache lookup; the
benchmark does not purge Triton's disk cache.

Both backends use the same smoothed histogram likelihood. The LLVM participant
uses the PEC objective hook to replace the usual KDE with the CPU histogram
scorer. This compares equivalent density-estimation settings; it is not a
timing of LLVM's default KDE workflow. RNG algorithms and native simulation
precision differ, so fitted parameters and the number of optimizer evaluations
can differ. Check convergence before comparing complete-fit timings, and compare
likelihood-call counts as well as wall time. Run the backends sequentially to
avoid resource contention.

Raw data, truth, result JSON, EM history, and participant estimates are written
under the ignored `benchmark_results/` directory.

### Local measurement, 2026-10-06

Measured at `a6f013548e`, on an RTX 2080 Ti and an Intel i7-9700K using all eight
CPU cores. Two fresh-process runs per backend used the defaults above, the same
360-trial dataset, and matched histogram settings: 160 bins over 0–8 seconds,
smoothing sigma 1 bin, and pseudocount 0.1. Data generation used LLVM separately
from the timed fits. Likelihood calls are scalar requests from the hierarchical
driver; each GPU call parallelizes the underlying simulations.

| Measurement | Triton GPU | LLVM CPU | GPU speedup |
| --- | ---: | ---: | ---: |
| Warm complete fit, median | 29.68 s | 356.41 s | 12.0× |
| Model setup + first scoring + fit, median | 42.48 s | 380.75 s | 9.0× |
| Fixed-parameter likelihood call, median of 14 | 32.20 ms | 167.01 ms | 5.2× |
| EM iterations | 2 | 3 | — |
| Likelihood evaluations, including final E-step | 1,228 | 2,054 | — |

Warm fit times ranged from 29.37–29.99 seconds on GPU and 356.17–356.65 seconds
on CPU. Both group fits and all six participant fits converged. Repeated runs
reproduced each backend's fitted parameters, posterior covariance matrices,
objective, and evaluation counts exactly. Full-fit speedup includes the GPU
objective requiring fewer evaluations; it is not solely a per-call speedup.

Drift-rate recovery RMSE was 0.154 (GPU) and 0.158 (CPU); threshold RMSE was
0.0778 and 0.0738. Group variances in unconstrained space differed: respectively
`[0.332, 0.0985]` and `[0.211, 0.0626]`. Similar point-estimate recovery therefore
does not establish equivalent posterior uncertainty. Particle-budget and
curvature sensitivity need checking before treating those uncertainties as
interchangeable. These measurements concern this DDM workload and hardware;
they do not measure the history-conditioned DAWA model.

### 25,000 estimates: measured GPU fit and projected LLVM runtime

Reuse the same dataset and solver settings, record the GPU optimization trace,
then time LLVM at a sample of those parameter vectors without running LLVM EM:

```bash
BENCH=Scripts/Debug/pec_batch_compile/benchmark_hierarchical_fit.py
RESULTS=Scripts/Debug/pec_batch_compile/benchmark_results/hierarchical_ddm
LARGE="$RESULTS/estimates_25000"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
.venv/bin/python "$BENCH" --backend triton --data "$RESULTS/data.csv" \
  --estimates 25000 --pseudocount 1.25 --hessian-batch-size 1 --output "$LARGE/triton.json" \
  --trace-output "$LARGE/triton.trace.csv"
.venv/bin/python "$BENCH" --backend llvm --data "$RESULTS/data.csv" \
  --estimates 25000 --pseudocount 1.25 --output "$LARGE/llvm_probes.json" \
  --score-only --probe-trace "$LARGE/triton.trace.csv" \
  --probe-points 5 --score-repeats 2
.venv/bin/python "$BENCH" --backend triton --data "$RESULTS/data.csv" \
  --estimates 25000 --pseudocount 1.25 --hessian-batch-size 1 --output "$LARGE/triton_repeat.json" \
  --trace-output "$LARGE/triton_repeat.trace.csv"
```

The pseudocount increases from 0.1 to 1.25 to preserve its strength relative to
the number of simulations; histogram bins and smoothing sigma stay unchanged.
These runs used the same revision, hardware, six participants, and 360 trials
as the 2,000-estimate benchmark above.

| Measurement | GPU, measured | LLVM, projected for GPU call counts | LLVM, projected for earlier LLVM call counts |
| --- | ---: | ---: | ---: |
| Warm complete fit | 35.05 s | 27.26 min | 44.03 min |
| Model setup + first scoring + fit | 49.70 s | 27.80 min | 44.57 min |
| Likelihood evaluations | 1,278 | 1,278 assumed | 2,054 assumed |

The two GPU fits took 35.42 and 34.69 seconds, converged in three EM iterations,
and reproduced their fitted parameters, posterior covariance, objective, and
evaluation counts exactly. All six participant fits converged. Drift-rate and
threshold recovery RMSE were 0.165 and 0.0616. Group variances in unconstrained
space were `[0.257, 0.0498]`; changing the simulation budget still changes the
inferred uncertainty, so these timings do not establish uncertainty convergence.

LLVM measurements comprise five parameter vectors per participant, each timed
twice: 60 calls at 25,000 estimates. Vectors are the midpoints of five equal
chronological strata in each participant's GPU evaluation trace, selected
without reference to scores or runtime. Participant mean call times ranged
from 1.12 to 1.45 seconds. The projection sums each participant's mean call time
times its assumed evaluation count, then adds measured solver overhead (less
than 0.3 seconds). Setup uses the measured LLVM construction and first-call
times at 25,000 estimates, plus the earlier LLVM group-construction time.

The first projection assumes the GPU workload; the second assumes the earlier
2,000-estimate LLVM fit's per-participant evaluation counts. They imply roughly
47–75× warm-fit speedups, but are workload scenarios, not confidence bounds.
LLVM's actual optimization path, convergence, and fitted parameters at 25,000
estimates were not measured. As above, the comparison uses matched histograms,
not LLVM's default KDE. Setup-plus-fit timings exclude process startup, imports,
and extra benchmark repeatability checks and scoring probes.

The local `estimates_25000/summary.json` records both projections, their formulas,
assumptions, per-participant timing weights, and source/data/harness hashes.

### Batching hierarchical Hessian evaluations

Hierarchical fits accept `hierarchical_options={"hessian_batch_size": 16, ...}`.
This limits the parameter vectors per finite-difference batch; `1` selects the
scalar path. The default is 16. GPU participants use `log_likelihood_batch`,
while LLVM and neural participants keep their scalar likelihoods. Both local
fits and distributed worker tasks support the setting. Nelder–Mead remains
scalar, and the simulation budget and random seed are shared across probes.

For a full Hessian with two parameters, all eight probes fit in one batch. The
benchmark records parameter evaluations separately from scalar and batch API
calls, and times the Hessian work separately from the whole fit:

```bash
BENCH=Scripts/Debug/pec_batch_compile/benchmark_hierarchical_fit.py
RESULTS=Scripts/Debug/pec_batch_compile/benchmark_results/hierarchical_ddm
BATCHES="$RESULTS/estimates_25000/hessian_batches"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
.venv/bin/python "$BENCH" --backend triton --data "$RESULTS/data.csv" \
  --estimates 25000 --pseudocount 1.25 --hessian-batch-size 1 \
  --output "$BATCHES/scalar.json" --trace-output "$BATCHES/scalar.trace.csv"
.venv/bin/python "$BENCH" --backend triton --data "$RESULTS/data.csv" \
  --estimates 25000 --pseudocount 1.25 --hessian-batch-size 16 \
  --output "$BATCHES/batch.json" --trace-output "$BATCHES/batch.trace.csv"
```

First batch calls are warmed before the fit and charged to setup, like the first
scalar calls. Each result includes a hash and saved patch of the uncommitted
library changes used for that run. Trace times for rows in a batch are the
batch's elapsed time divided by its row count.

The comparison also exposed a layout-dependent floating-point reduction in the
histogram scorer: identical per-trial densities could produce slightly different
log scores when NumPy summed a strided trial axis. The scorer now makes that
axis contiguous before summing, including after trial masking. Regression tests
check exact agreement across array layouts, candidate order, and chunk sizes.

On the same RTX 2080 Ti, six participants, 360 trials, and 25,000 estimates,
three fresh-process runs per mode gave the following medians. Runs were
sequential, ordered scalar / batch / batch / scalar / scalar / batch:

| Measurement | Scalar Hessian | Batched Hessian | Speedup |
| --- | ---: | ---: | ---: |
| Warm complete fit | 34.52 s | 30.81 s | 1.12× |
| Model setup + first scoring + fit | 48.03 s | 45.60 s | 1.05× |
| All Hessian calculations within the fit | 5.15 s | 2.18 s | 2.36× |
| Scalar likelihood API calls | 1,278 | 1,086 | — |
| Batched likelihood API calls | 0 | 24, with 8 rows each | — |
| Total parameter vectors evaluated | 1,278 | 1,278 | — |

Warm fits ranged from 34.44–34.62 seconds scalar and 30.67–31.01 seconds batched.
All runs converged in three EM iterations, including every participant, and
matched every traced parameter vector and likelihood score exactly. Fitted
parameters, covariance matrices, objectives, and evaluation counts also matched
the fit before this implementation.

For this two-parameter workload, batching saves about 11% of warm fit time and
5% including setup. Hessians account for only 15% of the original fit, limiting
the total benefit. Of the approximately 3.7-second warm-fit reduction, 3.0
seconds were measured in Hessian calculations; the remainder was outside that
phase. Higher-dimensional Hessians have more probes, but their speedup has not
been benchmarked here. LLVM timing was not repeated for this comparison.

Raw measurements, traces, source patches, and the aggregate `summary.json` are
saved locally in `estimates_25000/hessian_batches/` under the ignored results
directory.
