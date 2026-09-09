# Generic conditional likelihood execution: local GPU validation

Validated on 2026-09-09 in WSL, using the NVIDIA GeForce RTX 2080 Ti and
PyTorch 2.13.0+cu130. No cluster job was submitted. Generated JSON output was
not added to version control.

## What was validated

The generic compiler now constructs these stages from the supported composition
and observation declarations:

1. Guarded count inversion and sequential observed-history reconstruction.
2. Candidate/trial-parallel deterministic boundary trajectories.
3. Candidate/trial/estimate stochastic primitive execution using registered
   source step, readout, state, parameter, and RNG implementations.
4. Checked scalar observation gates, followed by explicit empirical count-domain
   matching and reduction.

The first stage's count inversion remains CPU-side. Generated scheduler/path/
sampling/readout kernels execute on the GPU; matching/reduction uses GPU Torch.
Public traces still materialize on the host and cross-stage transfers remain.
This is not yet a fully device-resident, fused, chunked, or PEC-routed objective.

The comparison oracle restores the same canonical observed-history state and
held controls independently for every trial/estimate lane, then executes the
full coupled source scheduler. Its observation outputs come from actual source
gates, not the newly derived readout expressions. This tests the execution split
and observation translation. It shares registered primitive implementations,
so it is not an independent proof of the primitive mathematics.

## Larger stochastic check

Configuration: 8 trials, 3 parameter candidates, 4,097 estimates, seeds 17 and 29,
and both common-random-number policies. The non-power-of-two estimate count
exercises partially populated GPU blocks. Candidates vary nondecision time and
LCA gain. The model has deterministic LCA evolution, DDM noise 0.15, positive ITI,
and alternating cue timing. Both LCA and DDM use the selected timestep;
physical onset durations and threshold-collapse rate are held consistent between
the two discretizations. Caps are 128 steps at 10 ms and 1,280 steps at 1 ms.

Across both timesteps, **786,624 sample lanes** were compared:

- Zero active-count mismatches.
- Zero response mismatches.
- Zero empirical-score hit-count mismatches.
- RT comparison passed a 1 microsecond maximum absolute-error guard.
- No truncation or zero-hit factors in this particular run.

These are matched executions, not 786,624 independent observations: CRN shares
random draws across candidates. Comparing the two timesteps to one another,
or comparing either to the continuous direct likelihood, is not the purpose
of this check. Each specialization is compared to its own source discretization.

## Illustrative warm end-to-end times

Medians of three warm runs for the workload above, in seconds:

| Operation | 10 ms | 1 ms |
|---|---:|---:|
| Generated observation sampling | 0.162 | 0.270 |
| Full coupled conditional reference | 0.159 | 0.270 |
| Generated empirical mass score | 0.194 | 0.274 |

These include Python checks, endpoint reconstruction, history/path preparation,
transfers, GPU execution, and output validation. They are not isolated kernel
times, full-subject fit estimates, or comparisons to the direct CPU solver or
handwritten CSI fitting backend. The coupled oracle currently shares path
preparation even though it only needs canonical starts, so it is not an optimized
baseline. Concurrent CPU regression tests and a display-attached GPU also limit
the precision of this small benchmark.

There is **no demonstrated end-to-end speedup at this batch size**. Profile and
cache validated lowering/source, retain intermediate tensors on-device, then
measure larger workloads and chunking before claiming a production speedup.

## Additional acceptance coverage

- Interpreter and GPU comparisons for deterministic/noisy models, parameter
  candidates, fixed RNG policies, and ordinary forward simulation.
- Renamed components, altered affine gates and projection weights, reversed
  observation columns, positive onset timing, and trial-varying parameters.
- Forged-witness rejection, explicit truncation, zero-hit `-inf` log factors,
  unscored-but-conditioning RTs, density refusal, and rejection of an unchecked
  additional scored event time.
- Existing GPU acceptance tests against a deterministic transition oracle,
  LLVM execution, and fresh PNL Python execution passed. This does not modify
  or endorse the old LLVM PEC likelihood calculation.

## Reproduce

Run the two execution modes in separate fresh processes:

```bash
env -u TRITON_INTERPRET .venv/bin/python \
  Scripts/Debug/pec_batch_compile/validate_likelihood_compile_gpu.py

env -u TRITON_INTERPRET .venv/bin/python -m pytest \
  tests/composition/pec/test_batched_conditional_gpu.py \
  tests/composition/pec/test_batched_observed_sampling.py \
  tests/composition/pec/test_batched_observation_metamorphic.py \
  --require-batched-backend=triton_gpu -o addopts='' -q

TRITON_INTERPRET=1 .venv/bin/python -m pytest \
  tests/composition/pec/test_batched_observed_sampling.py \
  tests/composition/pec/test_batched_observation_metamorphic.py \
  --require-batched-backend=triton_interpreter -o addopts='' -q
```

The script prints its JSON report to stdout. Its default configuration includes
both timesteps; `--trials`, `--candidates`, `--estimates`, `--seeds`, `--repeats`,
and `--buffer-mib` configure bounded local experiments.

## Scientific scope of the score

`compile_empirical_mass()` is an explicit empirical target in the checked
event-count domain, with exact FP32 matching of other scored fields. Count
compatibility is a numerical inversion guard, not proof of exact floating-point
readout support. It supplies no hidden smoothing, floor, measurement model,
gradient, or MCMC guarantee. Zero hits remain zero; truncated samples raise.

This validates a compiler route for the supported exact-discrete subset. It
does not establish that this estimator is suitable for recorded participant RTs,
replace the existing smoothed CSI objective, or deliver arbitrary-composition
likelihood compilation. Those require further estimator/recording semantics
and explicit PEC integration.
