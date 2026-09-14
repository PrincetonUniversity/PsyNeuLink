# Common likelihood planning: analytic providers

The compiler now has an opt-in method-selection entry point alongside the
existing generated CSI sampling APIs. The first direct provider recognizes
scalar affine-Gaussian graphs automatically from frozen primitive contracts.
This is analytic distribution propagation, not equation discovery or a new CSI
solver. Existing PEC and CSI fitting defaults are unchanged.

A second provider now supports the explicit continuous first-passage target
for a single reset fixed-bound DDM. See [Wiener likelihood provider](WIENER_LIKELIHOOD_PROVIDER.md)
for its process contract, example, numerical methods, and GPU refinement checks.

## Example

```python
import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler, ObservationField, ObservationSpec,
)

noise = pnl.ProcessingMechanism(
    function=pnl.NormalDist(mean=0.3, standard_deviation=0.7),
)
readout = pnl.ProcessingMechanism(
    function=pnl.Linear(slope=-2.0, intercept=0.5),
)
model = pnl.Composition(pathways=[noise, readout])
observations = ObservationSpec((
    ObservationField(readout.output_port, measure="lebesgue"),
))
plan = BatchedCompositionCompiler.compile_likelihood(
    model, observations, method="auto", process="ideal_real",
)
print(plan.explain())
result = plan.value_and_grad(
    inputs={noise: [[0.0], [0.0], [0.0]]},
    data=[[0.1], [-0.2], [0.7]],
    parameter_sets=[{}, {f"{noise.name}.mean": 0.4}],
)
print(result.log_likelihood)    # [candidate]
print(result.log_factors)       # [candidate, trial]
print(result.gradient)          # [candidate, parameter]
print(result.parameter_names)   # canonical gradient column ordering
```

NormalDist ignores its input; the zero inputs specify the trials, not its mean.
To add a known trial-varying predictor, project a separate deterministic Linear
node into the readout alongside the Gaussian node. Parameter mappings accept
existing canonical names/aliases. Parameters are scalar per candidate;
trial-varying parameter wrappers are explicitly rejected in this milestone.

`score` only evaluates the objective; it does not optimize or simulate.
`value_and_grad` additionally differentiates with Torch autodiff. Both run in
float64 on CPU and preserve sub-FP32 changes in inputs and proposed parameters.
Calls describe one subject; no hierarchical fitting interface is provided.

For a differentiable tensor interface, use
`plan.evaluator.log_prob(inputs, data, parameters)`. Parameters must be a CPU
float64 tensor of shape `[parameter]` or `[candidate, parameter]`, in
`plan.evaluator.parameter_names` order. The returned `[candidate, trial]` tensor
retains its parameter autograd graph. Priors, parameter transforms, and an MCMC
driver remain the caller's responsibility. `include_mask` on score/value_and_grad
selects summed trial factors but does not permit missing/nonfinite observations.

## Process target is separate from method and device

| Selection | Meaning in this milestone |
| --- | --- |
| `process="source"` (default) | Preserve the finite-step simulation process; an explicit sampling estimator is required. |
| `process="ideal_real"` | Interpret admitted primitive laws and affine arithmetic over real numbers with independent ideal Gaussian draws. |
| `process="continuous_time"` | Registered single reset fixed-bound DDM first-passage density; unsupported compositions are rejected. |
| `method="auto"` | Select an available provider without changing the requested process or inventing an estimator. |
| `method="analytic"` | Require the registered analytic tier. |
| `method="sampling"` | Require source sampling with an explicit estimator. |
| `method="numerical"` | Reserved deterministic numerical-solver tier; currently rejected. |

An ideal Gaussian density is not the probability mass of a finite-precision
PRNG output. The description records the ideal-real interpretation, float64
evaluation, and constants already frozen by source lowering (projection
matrices currently enter that graph as FP32). It does not claim bitwise
simulator equivalence or recovery of pre-rounding matrix entries.

Likewise, a continuous DDM first-passage law must not be substituted for a
discrete endpoint-tested simulator merely because `method="auto"` was chosen.
The Wiener provider requires this explicit continuous target.

`backend="auto"` chooses `torch_cpu` for analytic and `triton_cpu` for sampling.
Explicit analytic GPU requests are rejected, not moved silently to CPU.
Existing GPU source sampling uses `backend="triton"`. Analytic compilation uses
structural lowering/source preflight but does not require an available Triton
runtime or CUDA device.

The description includes method, process, backend, evaluator, requested
reference measures, gradient support, reason, assumptions, and approximations.
Sampling descriptions also retain the estimator configuration. The histogram
surrogate label matters: observation reference-measure declarations do not
make a smoothed histogram an exact source likelihood.

## Existing sampling through the common interface

For an already supported CSI composition and ObservationSpec:

```python
from psyneulink.core.batched import HistogramEstimatorSpec

plan = BatchedCompositionCompiler.compile_likelihood(
    model, observations,
    process="source", method="sampling", backend="triton",
    estimator=HistogramEstimatorSpec(
        categorical_dims=(0,), bins=100,
        smoothing_sigma=0.5, pseudocount=0.1,
        categorical_cardinalities=(2,),
    ),
    max_steps=1200,
)
result = plan.score(inputs, data, parameter_sets, num_estimates=100_000, seed=17)
```

Here the variables denote a supported CSI case, not the Gaussian example
above. The facade calls the existing checked history/trajectory/observation
sampler and fused scorer. It does not add sampling eligibility for arbitrary
independent-output graphs. `estimator="empirical_mass"` selects the existing
counting-measure estimator; its endpoint and observation restrictions remain.
Results retain estimator-specific fields plus common `log_factors` and
`log_likelihood` fields. Neither sampling estimator has a registered gradient.
Approximate point-history timing remains labeled; it is not promoted to exact
conditioning. There is no fallback to a handwritten CSI kernel.

## Analytic compilation

1. Capture the existing immutable graph, kernel, bindings, and contract snapshot.
2. Reuse dependency analysis to establish independent trials.
3. Admit a scalar static single-pass graph without state, held controls,
   stopping events, or undeclared effects. Accept Always and checked
   (All)EveryNCalls(1); projected producers must publish in earlier consideration
   sets. Trial termination must require all nodes.
4. Translate Gaussian/affine rules into an immutable `GaussianWitness` program
   with parameter indices and projection coefficients.
5. Propagate a mean and coefficients of independent standard Gaussian draws.
   For `Y = m + sum_j a_j Z_j`, variance is `sum_j a_j**2`; evaluate its normal
   log density and differentiate through the propagation and scoring.

Tracking each primitive's coefficient preserves reconverging-path covariance:
coefficients are added **before squaring**. Treating the two paths in `2*Z - Z`
as independent would incorrectly yield `5 Var(Z)` rather than `Var(Z)`.
Multiple independent Gaussian roots are supported for one scalar observation.
Correlated multi-output densities are rejected, not multiplied as marginals.

Primitive laws are trusted declarations bound to the implementation snapshot.
Re-derivation checks the translated witness, not the truth of those declarations;
it is not a Lean certificate. Live-model/global-registry changes do not alter a
compiled plan. Runtime proposals must be supplied explicitly.

Zero observed variance is a singular law, not a Gaussian density. It raises an
error; no variance jitter, pseudocount, or score floor is introduced. Negative
standard deviations and nonfinite numerical values are rejected. Nonlinear
functions, vector outputs, partial/multiple observations, recording noise,
rounding/censoring, and unresolved initial states need additional rules.

## Reading order

- `psyneulink/core/batched/likelihood_planning.py`: common facade and selection.
- `psyneulink/core/batched/components/normal.py`: forward draw and law contract.
- `psyneulink/core/batched/likelihood_ir.py`: `GaussianReadout` declaration.
- `psyneulink/core/batched/gaussian_likelihood.py`: admission, translated program,
  coefficient propagation, CPU scoring and autodiff.
- `tests/composition/pec/test_batched_gaussian_likelihood.py`: formula, quadrature,
  Jacobian, covariance, finite-difference/gradcheck, snapshot and forward tests.
- `tests/composition/pec/test_batched_likelihood_planning.py`: existing CSI
  histogram/empirical-mass equivalence through the common interface.

NormalDist uses an opt-in exact mechanism/function pair registration:
`register_batched_op(..., function_specific=True)`. It does not shadow other
ProcessingMechanism functions. Instance overrides retain precedence and
existing class registrations retain their behavior.

## Validation and next milestones

```bash
env -u TRITON_INTERPRET .venv/bin/pytest -n 0 -q \
  tests/composition/pec/test_batched_gaussian_likelihood.py \
  tests/composition/pec/test_batched_likelihood_planning.py
```

Analytic tests need no GPU. The local RTX 2080 Ti check uses 50,000 draws for
each of two trials, testing moments, interval probabilities, and cross-trial
correlation. Original-PNL Python forward execution is checked independently.
An additional 50,000-draw GPU check covers two independent Gaussian sources
and reconverging shared-noise paths. It also guards against emitting an empty
final-state block for multiple stochastic mechanisms without retained state.
Both supplement deterministic density/gradient oracles; Monte Carlo agreement
alone is not proof. The initial broader regression run passed 114 checks with
31 skips for backend selection/style caching, including existing PEC, zero-step
history, registry snapshots, graph identity, and transfer semantics.
The additional GPU/reconvergence checks passed (4 selected checks), as did both
CSI facade comparisons in a separate required CPU-interpreter run.

Reset-per-trial DDM support is now implemented in the narrow slice described
above. The CSI PDE and adjoint have since been extracted into a shared
[numerical backend](NUMERICAL_LIKELIHOOD_BACKEND.md), selected by mathematical
requirements rather than whole-model recognition. The existing CSI driver uses
that shared implementation. An [explicit continuous equation IR and generated
C++ phase backend](CONTINUOUS_EQUATION_COMPILATION.md) now supply deterministic
equations, clock gradients and dependency slicing. Composition-level numerical
admission is still not registered: next come frozen PNL primitive bindings,
clock/reset contracts and checked observed-history lowering, retaining compiled
subject scans rather than Python per-step callbacks.
General equation synthesis, particle inference, formal proofs, numerical DDM/CSI
providers, automatic observation-model inference, and PEC optimizer wiring are
not implemented by this milestone.
