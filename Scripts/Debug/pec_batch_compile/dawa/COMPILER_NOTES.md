# DAWA compiler notes

For running fits and parameter recovery, start with the [fitting guide](README.md).
This page records implementation details and simulation checks. Timings and
the history of performance work are in the [benchmark results](dawa_benchmark_results.md).

## Supported model components

The ordinary PEC batch compiler supports the source model's nonlinear Logistic
LCAs and Gaussian noise in all four layers. There is no DAWA-specific kernel or
graph recognizer. The shared adapters cover:

- Scheduled Logistic LCAs of width 1–32, finite dense recurrent matrices,
  scalar numeric or `NormalDist` noise, maximum-activity termination, and
  decision index/time/step and energy outputs.
- Scalar Euler FitzHugh–Nagumo integration in a Linear TransferMechanism,
  with zero initializers and a fixed number of internal executions per pass.
- Elementwise ObjectiveMechanisms and scalar OVERRIDE controllers that can
  fan out to multiple registered parameters in a dynamic schedule.

Scheduled LCAs use `execute_until_finished=False`. Gaussian LCAs support
`AtTrialStart` and `Never` resets. Custom recurrent matrices are frozen;
scalar competition/self-excitation overrides on those matrices are rejected.

## Recurrent scheduling

The [shared model builder](dawa_lca_model/full_lca_model_lc.py) and
[simulation driver](dawa_batched_simulation.py) default to recurrent scheduling.
All four LCAs and the four weighted processing mechanisms use `Always()` so
they keep advancing in the graph's execution order until response threshold.
Bias and weight controllers use `AtPass(0)` and hold their values throughout
the trial; output gates use `WhenFinished(responseLayer)`.

Originally, processing nodes inherited `EveryNCalls` dependencies on the
once-per-trial controllers. The downstream LCAs could execute only once,
stalling trials that needed further integration. The builder fixes this for
native Python, LLVM, and the original fitting scripts. The driver's
`--schedule source` option uses the supplied builder's own conditions; the
bundled builder already contains the fix.

## Noise in all four LCAs

`c_noise`, `s_noise`, `d_noise`, and `r_noise` are zero-mean Gaussian standard
deviations. Each integration step scales its draw by `sqrt(dt)`. Control
retains state across trials; the other three LCAs reset at trial start.
The simulation driver defaults to response noise only, while the full-noise
recovery and subject benchmark drivers explicitly set all four SDs to 0.1.

PNL samples the persistent control LCA's initial recurrent RESULT port during
construction. Its value can differ from the separately initialized mechanism
value. Native PEC copies this constructed activity into each estimate. The
batch compiler freezes the same RESULT default via
`StateDecl.initial_attribute`; the integrated state starts at zero. Runtime
gain/noise changes affect subsequent steps without resampling the constructed
activity. Set and record the model-construction seed when comparing runs.

Random streams are independent by accumulator, estimate, and trial, with common
random numbers across parameter candidates by default. Seeded GPU runs replay
on the GPU; they do not reproduce NumPy/LLVM draws.

Both held controller outputs and the values last sampled by each target are
represented explicitly. Trial resets use the sampled values. Run each
subject's complete ordered sequence in one call: `initial_states` does not
yet restore controller values, so resuming these networks is rejected.
Compilation starts from construction defaults, not a live composition's state.

## Simulation checks

Run from the repository root in the configured Python environment:

```bash
python Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py \
  --c-noise .1 --s-noise .1 --d-noise .1 --r-noise .1 \
  --trials 4 --estimates 256 --max-steps 2000 --pec-smoke \
  --output /tmp/dawa_samples.npz

python Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py \
  --deterministic --reference python \
  --trials 4 --estimates 3 --max-steps 500
```

Samples have axes `[candidate, subject, trial, estimate, outcome]`, ending in
decision and RT. `--pec-smoke` evaluates two candidates, including conditional
parameters; it does not run an optimizer. Component tests cover initial
activity, reset semantics, parameter changes, noise moments, cross-trial
covariance, stream independence, replay, and full-network Python parity.

The [first-trial LLVM reset investigation](dawa_benchmark_results.md#first-trial-llvm-rt-discrepancy-reset-diagnosis-2026-09-25)
documents a mismatch between native LLVM and Python in the version benchmarked
on September 25. Python is the reference for that reset behavior. Aggregate
LLVM/GPU timing comparisons do not establish exact backend equivalence.

## Fixed parameters and faster Gaussian conversion

The optimized PEC configuration used for recovery is:

```python
optimization_function = pnl.PECOptimizationFunction(
    method="differential_evolution",  # replaced with a CMA-ES study by the recovery driver
    batched_backend="triton",
    batched_max_steps=2000,
    batched_strict_truncation=True,
    batched_specialize_fixed_parameters=True,
    batched_fused_likelihood=True,
    batched_bins=100,
    batched_bin_range=[(0., 3.)],
    batched_smoothing_sigma=.5,
    batched_pseudocount=1.,
    batched_categorical_cardinalities=[2],
    batched_triton_launch_options={
        "block_size": 32,
        "num_warps": 1,
        "trial_schedule": "independent",
        "normal_rng": "philox4x_fast_v1",
    },
)
```

Specialization compiles non-fitted parameter defaults as constants. Fitted
parameters, conditional LC modes, and controller modulation remain dynamic.
Conflicting overrides of fixed parameters raise an error; compile a new plan
to change a fixed value.

Fused histogram scoring retains counts instead of all simulated outcomes.
Smoothing sigma is in RT-bin units: 0.5 bins is 15 ms for 100 bins over 0–3 s.
Pseudocounts are per joint choice/RT cell. Smoothing uses neighboring RT bins
with boundary renormalization and does not mix choices or change simulations.

`trial_schedule="independent"` lets each estimate begin its next trial when
ready, preserving its own ordered history and controller state. Conditional
parameters and histogram writes use that estimate's trial index. The general
compiler default remains `"synchronized"`; relative performance depends on
the model.

`philox4x_fast_v1` uses grouped Philox draws and a faster CUDA Gaussian
conversion. `philox4x_v1` and `legacy` remain available for reproducing older
runs. Rounding differences can change stopping steps, so record the RNG mode
as well as the seed. The fast mode requires CUDA compilation.

## Benchmark and research entry points

| Entry point | Purpose |
| --- | --- |
| [dawa_pec_fit.py](dawa_pec_fit.py) / [dawa_pec_recovery.py](dawa_pec_recovery.py) | Shared single-subject GPU optimization, using recorded or synthetic observations |
| [dawa_pec_fit_benchmark.py](dawa_pec_fit_benchmark.py) | Time four fixed proposals on a subject's data; no optimization |
| [dawa_llvm_benchmark.py](dawa_llvm_benchmark.py) | Compare materialized GPU and native LLVM samples |
| [Benchmark results](dawa_benchmark_results.md) | Measured workloads, profiling, validation, and reproduction commands |
| [Direct likelihood](dawa_likelihood/README.md) | Earlier response-noise likelihood prototype |
| [Continuous likelihood](dawa_likelihood/CONTINUOUS_README.md) | Continuous model extension and independent validation sampler |
| [Gradient ideas](../SAMPLING_GRADIENT_NOTES.md) | Deferred gradients through full-noise simulation |

The direct-likelihood prototypes do not implement the all-four-noisy-LCA
fitting workflow. Their continuous GPU sampler validates a different model
extension; it is separate from the source-model PEC sampler used for recovery.
