# LLVM / GPU conditional-likelihood validation

The LLVM histogram observation model now uses the GPU's FP32 observations,
predictions, bin edges and smoothing tables. Exact interior edges belong to
the lower bin. The upper domain bound is expanded by one part per million,
and bin volume and contamination use the resulting rounded histogram cells.
Likelihood accumulation remains FP64. Gaussian kernels retain their existing
behavior; select `kernel="histogram"` when comparing with the GPU filter.

CPU edge construction reproduces CUDA linspace's two-endpoint FP32 arithmetic,
including fused rounding when a tiny endpoint breaks a rounding tie. The
[PyTorch implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/RangeFactories.cu)
defines that construction. This does not require CUDA or Torch at runtime.
Unrepresentable edges/bin volumes and overlapping category tolerance regions
are rejected explicitly. Histogram categorical matching uses absolute
tolerance 1e-6, so declared categories must be separated by more than 2e-6.

## Observation regressions

`tests/composition/pec/particlefilter_gpu_reference.json` contains five captures
from the unmodified GPU implementation at `8722b247c9`. Its inputs and options
are retained alongside exact edge bits, densities, normalized particle weights
and contamination responsibilities. CPU-only regression tests cover exact
edges and their FP32 neighbors, out-of-domain particles, multiple continuous
outputs, categorical tolerance, data-derived domains, and unsmoothed kernels.
Additional tests check normalization, zero support, tiny contamination, and
an independent Kalman-filter likelihood. Optional CUDA tests compare the edge
constructor with live Torch, including domains spanning very different scales.

Validation passed: 75 tests with each of `--fp-precision=fp32` and
`--fp-precision=fp64`, plus live verification of all five GPU captures:

```bash
.venv/bin/python -m pytest \
  tests/composition/pec/test_particlefilter.py \
  tests/composition/pec/test_pec_conditioned_likelihood.py \
  -o addopts='' --fp-precision=fp64 -q
```

Verify the fixture against the isolated GPU checkout:

```bash
PYTHONPATH="$pnl_gpu_source" .venv/bin/python \
  Scripts/Debug/llvm_conditional_likelihood/gpu_observation_reference.py
```

To regenerate for review, add `--output /tmp/particlefilter_gpu_reference.json`
and `--source-revision <GPU-commit>`. Regeneration reads the cases from the
existing fixture and writes to the explicitly supplied path.

## Benchmark protocol

`benchmark.py` measures the public PEC `log_likelihood` API separately in each
checkout. It requires the standalone DAWA model source, not the feature
branch's batched simulation driver. Both backends use the same source and
construction seed, input sequence, observations, parameters, and contamination
probability. The output records source hashes, implementation hashes, initial
model outputs, every timing and independent-seed score, and per-trial diagnostics.

The workload has 16 synthetic trials, Gaussian SD 0.1 in all four LCA layers,
10 ms integration, response threshold 0.3, and the source's recurrent schedule.
One threshold parameter is fitted. All observations are assimilated and scored.
These are realistic particle budgets on a short sequence, not complete
760-trial subject fits or parameter-recovery results.

Observation settings are 100 RT bins over [0, 3], smoothing sigma 0.5 bins, and
two choice categories. GPU pseudocount is `N / 100000`; the corresponding LLVM
`contamination_probability` is always `200 / 100200`. This keeps the likelihood
target fixed when changing the particle budget. LLVM options are:

```python
likelihood_options = {
    "kernel": "histogram",
    "bins": 100,
    "bin_range": [(0.0, 3.0)],
    "smoothing_sigma": 0.5,
    "categorical_values": [[0.0, 1.0]],
    "contamination_probability": 200 / 100200,
}
```

The GPU uses prepared execution, fixed-parameter specialization, 32 particles
per block, one warp, Philox fast Gaussian conversion, and strict truncation
checks at 4,000 passes. LLVM uses four CPU workers. Model construction and the
first likelihood call are excluded from warm timings. Each median uses three
score-only calls with the same seed; GPU calls are synchronized. Timed CPU/GPU
runs are sequential. Independent-seed runs measure score variability separately;
their diagnostics are not included in the timing medians. GPU diagnostic scores
are checked against its public API score for the warmup seed.

The backends use different random streams and simulation precisions, so exact
score replay across backends is not expected. Compare repeated-run means and
Monte Carlo uncertainty. A confidence interval containing zero does not prove
equivalence; longer sequences, other candidates and observation models still
need validation.

## Measurements

Measured on 2026-10-02 on an Intel Core i7-9700K with four LLVM workers and an
NVIDIA GeForce RTX 2080 Ti under WSL2, using Python 3.13.3. GPU simulation uses FP32;
LLVM simulation uses FP64. Both histogram observation models use FP32.

| Particles | LLVM median | GPU median | GPU speedup |
| ---: | ---: | ---: | ---: |
| 10,000 | 8.844 s | 0.0852 s | 103.8x |
| 100,000 | 104.919 s | 0.0840 s | 1,248.8x |

These are warmed times for one 16-trial likelihood evaluation. At this short
sequence length, increasing GPU particles tenfold did not measurably increase
the median of three runs. This does not imply constant cost at larger budgets
or on longer sequences. LLVM's model runs natively, while Python coordinates
filtering and copies retained particle state during resampling. These timings
do not isolate those costs or estimate the benefit of moving filtering to LLVM.

GPU measurements use `8722b247c9`. LLVM measurements use `f5c98527df` at 10,000
particles and `45f83cbac0` at 100,000 particles. The latter adds extreme-scale
rounding and validation fixes; it preserves the [0, 3] histogram used here.
The reports confirm identical model-source hashes, initial outputs, inputs,
observations, parameters and contamination probabilities across backends.

Independent-seed log-likelihood summaries:

| Particles | Backend | Runs | Mean | Sample SD |
| ---: | :--- | ---: | ---: | ---: |
| 10,000 | LLVM | 12 | 3.47060 | 0.15007 |
| 10,000 | GPU | 12 | 3.55181 | 0.14152 |
| 100,000 | LLVM | 6 | 3.56182 | 0.03209 |
| 100,000 | GPU | 12 | 3.57282 | 0.07469 |

The LLVM-minus-GPU mean difference is -0.08121 at 10,000 particles, with a
Welch 95% confidence interval of [-0.20472, 0.04231]. At 100,000 particles the
difference is -0.01100, with interval [-0.06453, 0.04252]. These intervals use
independent samples, not paired seeds. They are consistent with matching
likelihoods on this workload but do not establish equivalence. In particular,
the six-run LLVM standard deviation at 100,000 particles is a noisy estimate.

## Reproduction

From the repository root, extract an isolated GPU package and the shared model.
This leaves the current branch unchanged:

```bash
pnl_gpu_revision=8722b247c9b891d9cd6827ee4295a53b996114b6
pnl_gpu_source=$(mktemp -d)
git archive "$pnl_gpu_revision" psyneulink | tar -x -C "$pnl_gpu_source"
git show "${pnl_gpu_revision}:Scripts/Debug/pec_batch_compile/dawa/dawa_lca_model/full_lca_model_lc.py" > /tmp/dawa_source.py
```

Run each command separately, repeating with `--particles 10000` and `100000`.
Use 12 independent runs for both backends at 10,000 particles; at 100,000,
the measurements above use 12 GPU runs and six LLVM runs.
Use fresh output paths; the driver refuses to overwrite an earlier report.

```bash
PYTHONPATH="$pnl_gpu_source" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python Scripts/Debug/llvm_conditional_likelihood/benchmark.py \
  --backend gpu --particles 100000 --model-source /tmp/dawa_source.py \
  --source-revision "$pnl_gpu_revision" \
  --replicates 12 --output /tmp/gpu-100000.json

PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python Scripts/Debug/llvm_conditional_likelihood/benchmark.py \
  --backend llvm --particles 100000 --model-source /tmp/dawa_source.py \
  --replicates 6 --output /tmp/llvm-100000.json
```

Use `--threshold` to change the candidate, `--trials` for sequence length, and
`--precision fp32` to evaluate LLVM with FP32 simulation. Histogram observation
precision stays FP32 in either case. `--save-clouds` requests an extra LLVM
evaluation and saves its predictive outcomes for shared-outcome comparisons.
Generated JSON, logs and arrays belong outside the source tree.
