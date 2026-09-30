# Source-model output audit, 2026-09-30

The tested GPU simulations agree with the bundled Dawa composition executed
in native Python, including noise in every LCA, nonlinear dynamics, recurrent
scheduling, and state carried between trials. This checks the current source
builder with the previously agreed scheduling correction and noise settings.
It does not establish equality with the preserved LLVM **likelihood**, which
still uses unconditioned histories.

The current fitting branch (`0fdfaa52be`) also retains the known LLVM
reset/modulation discrepancy. A separate reference run on
`fix/llvm-reset-sampled-modulation` (`926e2a9b46`) resolves the deterministic
output difference and agrees statistically with the GPU in the noisy check.
That fix was not merged into the fitting branch during this audit. No production
compiler, model, or fitting behavior was changed.

Full measurements and provenance are in
[source_parity_20260930.json](fitting_acceleration/source_parity_20260930.json).
The audit ran locally on the RTX 2080 Ti.

## Native Python with identical noise draws

The new [source parity regression](../../../../tests/composition/pec/test_dawa_source_parity.py)
feeds the GPU's Gaussian draws to the source composition's NormalDist functions.
All other native operations use PNL's scheduler, integrators, Logistic functions,
controllers, resets, and stopping rule. The draw table is generated independently
of network dynamics; separate Gaussian tests check its moments, tails, and
stream independence. This isolates dynamics agreement from differing RNGs.

Two seeds each run six trials with SD 0.1 in every LCA. Tasks alternate, and
LC mode, bias, both weights, NDT, control gain, and response threshold vary
between trials. This exercises retained control activity, held controller
outputs, and parameters sampled for subsequent resets. The GPU uses fixed
parameter specialization and the fitting driver's fast Gaussian/launch settings.

- Choices and response stopping-execution counts match exactly on all 12 trials.
- Maximum RT difference is **2.003e-7 seconds**, within float32 rounding.
- Maximum trial-end LCA activity / LC gain difference is **4.055e-7**.
- Native noise-call counts match each mechanism's executions and width.

The test checks stopping counts independently of reported RT rounding. It
allows at most two FP32 ULPs for RT and `atol=rtol=2e-6` for activities.
This is a targeted regression, not exhaustive coverage of the parameter space.

## LLVM forward simulations

With noise disabled and the first four retained subject-1 inputs, the RTs are:

| Reference | Trial 1 | Trial 2 | Trial 3 | Trial 4 |
| --- | ---: | ---: | ---: | ---: |
| Native Python | 0.96 | 1.26 | 0.89 | 1.25 |
| GPU | 0.96 | 1.26 | 0.89 | 1.25 |
| LLVM in the fitting branch | **0.97** | 1.26 | 0.89 | 1.25 |
| LLVM on the reset-fix branch | 0.96 | 1.26 | 0.89 | 1.25 |

All choices match. Python/GPU trial-end activities differ by at most 2.084e-7.
The [earlier causal diagnosis](dawa_benchmark_results.md#first-trial-llvm-rt-discrepancy-reset-diagnosis-2026-09-25)
explains why LLVM's first reset samples initial controller outputs instead of
the parameters last sampled by the LCA. Differences can propagate through
retained state even when later RTs agree.

An eight-trial, all-noisy comparison with 10,000 estimates reproduces the
current-branch discrepancy: first-trial mean RT is 941.739 ms in LLVM and
930.389 ms on GPU, a difference of 4.35 combined standard errors.

On the reset-fix branch, a larger check uses **100,000 estimates per trial**.
First-trial mean RT is 932.527 ms in LLVM versus 931.780 ms on GPU (0.90 combined
SE). Across all eight trials, mean RT and choice-rate differences are within
**1.81 combined SE**. Pooled 10th/50th/90th/99th RT percentiles match to FP32
rounding. These are pointwise statistical checks, not exact seeded replay or
a proof of distributional equivalence; sampled maxima differ.

Both noisy comparisons use the same construction seed, input sequence,
parameters, 10 ms LCA steps, LC clock, and independent component streams.
LLVM's benchmark explicitly supplies separate seed offsets; the independent
seed fix also exists on a separate branch and is not integrated here. The
reset-fix reference uses its own clean PNL checkout; its audit harness omits
unused GPU-compiler imports only. No LLVM reset method is monkeypatched.

## Likelihood and regression checks

LLVM was used only to return forward samples. Its objective was a dummy zero;
LLVM likelihood values were never treated as a conditioned-likelihood oracle.
The production filter was checked separately against the exact finite-state
HMM reference, independent NumPy observation calculations, and noisy state
transport/resampling tests. These checks do not remove finite-particle error
or observation-model approximation; see [likelihood accuracy](CONDITIONED_ACCURACY.md).

Final focused test runs: **96 passed, 29 skipped**, with 18 interpreter cases
deselected in one run. Totals include configured style checks. Ruff passed.
The new regression can be rerun with:

```bash
python -m pytest -q -n 0 --disable-warnings \
  tests/composition/pec/test_dawa_source_parity.py
```

For handoff, use the tested GPU workflow. Integrating the existing LLVM reset
and independent-seed fixes is separate from implementing observed-history
likelihood conditioning on LLVM; the reset fix does not add that conditioning.
