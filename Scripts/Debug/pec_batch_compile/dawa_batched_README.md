The DAWA LC/LCA network can now use the ordinary batched compiler and PEC
simulation objective on Triton. `dawa_batched_simulation.py` loads the local
`dawa_lca_model/full_lca_model_lc.py`. The
[source model, original fitting scripts, and Slurm examples](dawa_lca_model/README.md)
are tracked; subject data and generated outputs remain ignored.
All three `flanker_fit_lc_part*.py` scripts use this same model builder.

The compiler additions cover scheduled Logistic LCAs of width 1–32, finite dense
recurrent matrices, scalar numeric or `NormalDist` noise, maximum-activity
termination, standard decision index/time/step and energy outputs, scalar Euler
FitzHugh–Nagumo integration in a Linear TransferMechanism, and elementwise
ObjectiveMechanisms. Scalar OVERRIDE controllers can fan out to multiple
registered parameters in a dynamic schedule. Both held control values and the
values last sampled by each target are represented explicitly; trial resets use
the latter. These are component and scheduler features, with no DAWA-specific
kernel or compiler graph recognizer.

Recurrent scheduling is now the default in both the shared model builder and
the driver. All four LCAs and the four weighted processing mechanisms use
`Always()`, so they keep advancing through the graph's existing execution order
until the response reaches threshold. Bias/weight controllers still use
`AtPass(0)` and hold their values throughout the trial; output gates still use
`WhenFinished(responseLayer)`. Native Python, LLVM, and the fitting scripts
inherit the fix directly from the shared builder.

Previously, the processing nodes inherited `EveryNCalls` dependencies on the
once-per-trial controllers. The stimulus, decision, and response LCAs could
therefore execute only once, stalling trials that needed further integration.
The driver also applies the fix to older/custom builders by default. Its
optional `--schedule source` setting uses the loaded builder's own conditions;
the bundled builder already contains the fix, so this option also runs
recurrently with the bundled model.

Run a stochastic simulation and evaluate two candidates through PEC:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa_batched_simulation.py \
  --trials 4 --estimates 64 --max-steps 500 \
  --pec-smoke --output /tmp/dawa_samples.npz
```

The sample array has axes `[candidate, subject, trial, estimate, outcome]`, with
decision and response time as its final two columns. PEC smoke testing includes
all seven original fitting parameters and conditional parameters for subject
and previous congruency, matching the supplied fit scripts (12 coordinates for
the example's two subjects). It evaluates the existing histogram simulation
objective; it does not run an optimizer or infer an analytic likelihood.

Check deterministic results against native LLVM (or use `--reference python`):

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa_batched_simulation.py \
  --deterministic --reference llvm \
  --trials 4 --estimates 3 --max-steps 500
```

The local RTX 2080 Ti audit matched LLVM choices and response times, with maximum
absolute error below `3e-8`. A separate Python comparison of all four LCA layers
and LC activity across four trials also passed. A warm
stochastic run of two candidates × four trials × 64 estimates took about 30 ms
on this machine; this small workload measurement includes host preparation and
result transfer and is not a fitting-scale throughput claim.

To use these features directly:

```python
from psyneulink.core.batched import BatchedCompositionCompiler

plan = BatchedCompositionCompiler.compile(model, backend="triton", max_steps=2000)
samples = plan.run(inputs, parameter_sets, num_estimates=1000,
                   seed=29, strict_truncation=True)
```

Existing PEC workflows select
`PECOptimizationFunction(..., batched_backend="triton", batched_max_steps=2000)`.
The shared builder supports repeated integration at positive fitting thresholds
by default. Keep strict truncation checks enabled during simulation audits.

The new LCA configurations require `execute_until_finished=False`; the existing
CSI run-to-completion path is retained. Gaussian LCA noise requires
`AtTrialStart` resets, and the new FHN adapter supports scalar Euler integration
with zero initializers and fixed per-pass internal execution counts. Custom LCA
matrices are frozen; scalar competition/self-excitation overrides on those
matrices are rejected. GPU random streams reproduce seeded GPU runs, not
NumPy/LLVM draws. Networks with scalar OVERRIDE controls must run each subject's
complete sequence in one call: `initial_states` does not yet restore held and
sampled control values, so resuming these networks is rejected explicitly.

For fitting-scale performance against PEC's threaded LLVM simulation path,
see [the benchmark results](dawa_benchmark_results.md) and
[reusable benchmark driver](dawa_llvm_benchmark.py). Measurements include
1,000 and 10,000 estimates over both 64-trial slices and a full 760-trial subject,
with setup and likelihood scoring separated from simulation timing.

A [differentiable direct-likelihood prototype](dawa_likelihood/README.md) is also
available. It propagates the joint response-state distribution and supports
gradients through all seven fitting parameters in its RT observation model.
The documentation distinguishes direct stopping-step scoring from the
empirical-RT mode's fixed history approximation and records validation results.

The separate [continuous-time DAWA likelihood](dawa_likelihood/CONTINUOUS_README.md)
uses coupled ODE dynamics and a two-dimensional absorbing Fokker–Planck solver.
It scores RT intervals without added measurement noise, differentiates all seven
parameters, and updates history using candidate-dependent decision durations.
Its instantaneous gain modulation defines a continuous extension of the source
model; validation uses an independent continuous SDE sampler rather than expecting
parity with the original 10 ms scheduler.

A [source-convergence and synthetic-recovery study](dawa_likelihood/STUDY_README.md)
now checks that connection explicitly, using 100,000 estimates per case and
joint refinement of the original LCA and LC time steps at a fixed clock ratio.
