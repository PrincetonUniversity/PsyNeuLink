# Registered continuous DDM likelihood

The common compiler now recognizes a single reset-per-trial DDM with a
registered `WienerProcessReadout` and selects an analytic first-passage density
when explicitly asked for `process="continuous_time"`. This is the second
direct-provider milestone, after scalar affine-Gaussian propagation. It does
not integrate the full CSI solver or change PEC/CSI fitting defaults.

## Example

```python
import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler, ObservationField, ObservationSpec,
)

ddm = pnl.DDM(
    function=pnl.DriftDiffusionIntegrator(
        rate=1.2, noise=0.5, threshold=0.4, initializer=0.06,
        non_decision_time=0.2, time_step_size=0.01,
    ),
    output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
)
model = pnl.Composition(pathways=ddm)
observations = ObservationSpec((
    ObservationField(ddm.output_ports[0], "counting"),
    ObservationField(ddm.output_ports[1], "lebesgue", role="event_time"),
))
plan = BatchedCompositionCompiler.compile_likelihood(
    model, observations, process="continuous_time", method="auto",
)
result = plan.value_and_grad(
    inputs={ddm: [0.2, -0.1, 0.35]},
    data=[[1, 0.5], [0, 0.7], [1, 1.0]],
    parameter_sets=[{}, {"rate": 0.9}],
)
print(plan.explain())
print(result.log_likelihood)          # [candidate]
print(result.log_factors)             # [candidate, trial]
print(result.gradient)                # [candidate, parameter]
print(result.parameter_names)         # full source parameter ordering
print(plan.evaluator.active_parameter_names)
```

The bare DDM graph uses unqualified source parameter names, including
`starting_value` for the integrator initializer. This is a candidate-batched,
single-subject scorer: it does not perform a fit, create a prior, or launch a
simulation. CPU float64 Torch evaluation supplies autodiff gradients. Use
`plan.evaluator.log_prob(inputs, data, parameters)` for a differentiable tensor
interface, with CPU float64 parameters in canonical `[P]` or `[C,P]` order.
Candidates have scalar parameters; external drift inputs may vary across trials.

## The target process

Within a trial the admitted law is

```
dX(t) = (rate * supplied_input) dt + noise dW(t)
X(0) = starting_value
T = first exit from (-threshold, threshold)
choice = 0 for lower exit, 1 for upper exit
RT = non_decision_time + T
```

`noise` is the diffusion standard deviation, not its variance. Bounds are
fixed and symmetric, but the starting state need not be centered. Each trial
starts afresh. The score is the **joint choice/RT density** with counting
measure for choice and Lebesgue measure for RT; it is not normalized separately
within each choice.

The continuous target differs from endpoint-tested simulation at any finite
step size. Source `time_step_size` is validated as positive but does not enter
this density. Its gradient is zero. The source `offset` is an increment per
execution, not automatically a drift per unit time; it and `threshold_collapse`
must remain exactly zero. Their gradient columns are zero placeholders for
domain-fixed parameters, not valid directions in which to optimize. Use
`active_parameter_names` to select rate, noise, threshold, starting value, and
nondecision time; even these can be statistically confounded (for example,
jointly rescaling evidence-space quantities preserves the law). This milestone
does not solve parameter identifiability.

`max_steps` is rejected for this target: it has no simulation step cap or RT
horizon. Censoring would need an observation operator, not silent tail removal.
Changing `process="source"` still requires an explicit sampling estimator.
`method="numerical"` remains reserved for future PDE providers; an infinite
series representation of a known analytic law is selected as `"analytic"`.

## Admission and failure behavior

Recognition uses the primitive contract in the frozen compiler snapshot, not
the composition name. The source compiler establishes supported output ports,
reset and execution semantics. The provider further requires a single scalar
primitive, static execute-to-completion scheduling, no projections or held
controls, and both directly bound output observations. Other compositions fail
closed: even a deterministic preprocessing projection is outside this first
DDM slice. Extending to affine input subgraphs is separate work.

Both observations must be complete, exact supplied values and scored. Reversed
ObservationSpec field order is supported. Recording noise, rounding, censoring,
missing fields, approximate endpoint-history timing, collapsing bounds, and
multi-mechanism state/history are not inferred. Positive threshold/noise and a
strictly interior initial state are checked per proposal. Zero-noise or
boundary-starting cases need singular-law treatment and are rejected.

RT minus nondecision time at or below zero gives `-inf` log density. There is
no floor, clipping, or variance jitter. A gradient request with such an included
trial raises because the log-likelihood gradient is undefined. An explicit
boolean `include_mask` may select which trial factors contribute to the total;
excluded out-of-support factors do not contaminate included gradients. All
observations must still be finite. Numerical overflow/range failures raise,
rather than turning into apparently valid zero likelihoods.

## Numerical method and validation

The evaluator uses the complementary short-time image and long-time sine
series for the Wiener first-passage density described by
[Navarro and Fuss (2009)](https://papers.djnavarro.net/2009_firstpassage.pdf).
Its pairing, log scaling, and fixed truncations are our evaluation scheme, not
the paper's adaptive error-selection algorithm.

With width `L=2*threshold`, dimensionless time is `tau=T*noise**2/L**2`.
For `tau <= 0.2` the implementation uses eight reflected-image pairs (plus the
leading image where required); otherwise it uses sixteen eigenmodes. It
scales out leading exponentials and uses `expm1`/`sinc` forms near boundaries.
For short times, the drift and leading-image exponent are combined before
evaluation, avoiding cancellation near a strong-drift peak. At the centered
starting point, a one-sided algebraic representation retains the correct
starting-point derivative; averaging the derivatives of a `min` reflection
would incorrectly erase part of it.

These are numerically checked truncations, not a formal error certificate.
Validation includes:

- Unpaired independent image/spectral calculations at 90-digit precision:
  420 combinations spanning short/long times, both choices, positive/zero/
  negative drift, and starting points within `1e-10` of a boundary.
- Density and gradient agreement across the series switch; strong-drift peaks.
- Integrated total probability, upper/lower choice probabilities, and mean
  first-passage time against separate closed-form identities.
- Finite differences for all five active parameters and Torch gradcheck for
  both centered and displaced initial states.
- Interval-mass agreement with the existing CSI finite-volume PDE solver in
  its fixed-boundary case (0.2% relative tolerance).
- Runtime support/domain guards, frozen registration/witness tests, and
  independence from GPU/Triton runtime availability.

The first broader run passed 108 checks with 13 skips, covering the Gaussian
provider, common sampling routes, likelihood diagnosis, PEC, and zero-history
behavior. The focused Wiener suite was also run with a required local GPU.
The subsequent endpoint/history/Wiener regression run passed 81 checks with
11 skips, including the strengthened primitive-contract registration tests.

## Local GPU refinement experiment

The endpoint-tested simulator was run with 100,000 estimates at each timestep
on the RTX 2080 Ti: drift 0.25, noise 0.5, bounds +/-0.4, centered start,
nondecision time 0.2 s, seed 219. This is a toy fixed-boundary DDM, **not CSI**.

| Quantity | Continuous target | GPU 10 ms | GPU 1 ms |
| --- | ---: | ---: | ---: |
| P(upper choice and recorded RT <= 0.6 s) | 0.297865 | 0.263260 | 0.284660 |
| P(upper choice) | 0.689974 | 0.703420 | 0.696000 |
| Mean decision time, seconds | 0.607918 | 0.696881 | 0.636320 |

The finer simulation is closer but is not equivalent to the continuous law.
In this case the remaining mean decision-time difference is about 28 ms.
The comparison checks convergence behavior, not paired random draws or exact
likelihood equality. No sampled paths were truncated at the test's 16,000-step
cap. No cluster jobs or large result files were needed.

Reproduce the focused tests and printed refinement summary:

```bash
env -u TRITON_INTERPRET .venv/bin/pytest -n 0 -q -s \
  tests/composition/pec/test_batched_wiener_likelihood.py \
  --require-batched-backend triton_gpu
```

## Reading order and next step

- `psyneulink/core/batched/likelihood_ir.py`: `WienerProcessReadout` contract.
- `psyneulink/core/batched/components/ddm.py`: contract attached to the existing
  DDM primitive without changing its sampling recurrence.
- `psyneulink/core/batched/ddm_likelihood.py`: checked admission and scoring.
- `psyneulink/core/batched/wiener.py`: independent first-passage evaluator.
- `psyneulink/core/batched/analytic_runtime.py`: shared float64 parameter/input
  preparation used by both direct providers.
- `tests/composition/pec/test_batched_wiener_likelihood.py`: validation oracles.

Next is the checked adapter for the existing continuous CSI numerical solver,
with explicit model/observation/parameter compatibility checks. An adapter is
integration, not automatic derivation of the CSI PDE. More general drift and
boundary generation should follow from separating its reusable solver pieces.
