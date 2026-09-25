# Gradients for compiled Monte Carlo likelihoods

Status: deferred design ideas, recorded 2026-09-24. No sampling-gradient
implementation or performance result is claimed here. The user intends to
revisit this work later; this note does not initiate implementation.

## Motivation and current state

CSI's direct solver supplies deterministic derivatives of its numerical
likelihood and has enabled efficient gradient-based searches. The
[CSI fitting results](csi/csi_fit/direct_likelihood/README.md) include single-start
pilots requiring 46–99 evaluations; these do not establish global convergence.
DAWA's eventual model includes noise in all four LCAs, making a direct solve
over the full stochastic state much less attractive. The question is whether
the fast compiled sampler can supply useful Monte Carlo gradient estimates
without finite differences or linearizing the LCA dynamics.

The ordinary compiled simulation/histogram fitting path currently has no
registered parameter gradients. Existing analytic/numerical likelihood
providers and the earlier DAWA direct prototype have separate gradient support.
Simply exposing the simulator's outputs as Torch tensors would not differentiate
the generated Triton kernel, threshold decisions, or histogram counts.

The latest optimized H100 NVL comparison used 760 trials (720 scored), 100,000
estimates per trial per proposal, noise SD 0.1 in all LCAs, smoothing sigma 0.5
bins, and pseudocount 1:

| Execution | 10 ms LCA step, seconds/evaluation | 1 ms LCA step, seconds/evaluation |
| --- | ---: | ---: |
| Serial proposals | 0.3008 | 2.6284 |
| Batches of four, amortized | 0.2697 | 2.4834 |

The LC timestep was scaled proportionally (20 to 2 ms), retaining ten internal
LC updates per scheduler pass; step caps retained a 20-second decision horizon.
See the [matched timestep benchmark](dawa/dawa_benchmark_results.md#h100-timestep-refinement-10-ms-versus-1-ms-2026-09-24).
At 1 ms, 5,000 proposals extrapolate to approximately 3.45–3.65 hours, excluding
setup and optimizer overhead. As an illustration only, 500 value-and-gradient
evaluations costing three serial scores each would take about 66 minutes.
Neither the gradient cost nor the required evaluation count has been measured.

Success means better independently verified fit quality per GPU-minute, not
merely the ability to return a derivative. Gradients cannot resolve weak LC
parameter identifiability by themselves.

## What is being differentiated?

Distinguish smooth state sensitivities, derivatives of response probabilities,
and derivatives of a finite-sample log-likelihood estimate. They are not
interchangeable. With fixed random draws, integer stopping steps, choices, and
RT-bin assignments are usually locally unchanged even when their underlying
probabilities depend on a parameter. Ordinary pathwise autodiff can therefore
return zero or incomplete gradients through the fitting objective.

Record the numerical process, observation model, estimator, and history target
explicitly. The current full-sequence benchmark propagates each estimate's
retained state but scores trial-marginal histograms. Differentiating that score
does not turn it into a joint likelihood conditioned on the observed sequence.
A fully observation-conditioned latent-history likelihood may require
filtering and separate treatment of resampling. Keep this distinction aligned
with the [likelihood planning interfaces](LIKELIHOOD_METHOD_PLANNING.md) and
[history-likelihood work](AUTOMATIC_HISTORY_LIKELIHOOD_PLAN.md).

Probability-gradient estimators can be unbiased under appropriate conditions.
Forming a log-likelihood gradient by dividing an estimated probability gradient
by an estimated probability generally introduces finite-sample bias. Floors,
pseudocounts, kernel normalization, and any truncation also belong to the stated
objective and its derivative.

## Candidate methods

### Pathwise sensitivities

Hold the base Gaussian draws fixed and differentiate the discrete integration
updates, nonlinear transfer functions, projections, and LC dynamics. This
preserves the nonlinear equations and supplies derivatives for smooth outputs.
Philox and its integer counters are not themselves differentiated.

Hard stopping, winner selection, and sample binning need additional rules.
Replacing them with soft decisions is an explicitly approximate alternative;
a straight-through derivative is not the exact gradient of the original
response probabilities. Treat smoothing the stopping rule separately from
smoothing the RT observation density.

### Likelihood-ratio / score-function estimators

Accumulate derivatives of conditional transition log densities along a sampled
trajectory, then weight them by its observation contribution. Gaussian
transition laws are available even when the final choice/RT law is intractable.
The relevant density is that of the model's stochastic transitions, not the
parameter-independent base normal draws or PRNG seed.

This can handle discontinuous outcomes for admitted parameterizations, but a
naive transition-score sum is insufficient for parameter-dependent stopping
boundaries or observation events. Derive the necessary boundary contributions
or a valid change of variables. Deterministic intermediate states, nonlinear
state transformations, and parameter-dependent initialization need consistent
treatment as well. A transition-density rule must state its support and
nondegeneracy assumptions; zero noise cannot silently use an inverse-variance
Gaussian score formula.

The main practical risk is variance over long trajectories and trial history.
Explore valid baselines/control variates and conditional averaging. Resetting
score accumulators at every trial is not automatically correct when retained
state affects later observations. Measure uncertainty across independent seeds.

### Conditional Monte Carlo for crossings

Given the current state, analytically average over part of the next noise draw
to obtain crossing/survival probabilities. Differentiate these probabilities
while sampling the remaining dynamics. Correct conditioning, conditional
sampling, and weighting can preserve the discretized probability target while
reducing discontinuities and variance; this need not linearize the LCA.

This is a promising research direction, not an established DAWA estimator.
Competing response units, simultaneous crossings, nonlinear readouts, scheduler
order, and latent state carried into future trials must all be handled. A local
crossing probability alone is not enough if replacing the event changes the
distribution of subsequent state. Begin with explicitly supported event
families rather than arbitrary Boolean stopping conditions.

## Continuous RT kernels

For samples `(c_i, t_i)`, a Gaussian kernel estimate of the joint choice/RT
density at observed `(c, t)` is

\[
\widehat p(c,t\mid\theta)
=\frac{1}{N}\sum_{i=1}^N
  \mathbf{1}[c_i=c]\,\frac{1}{h}
  \phi\!\left(\frac{t-t_i}{h}\right).
\]

Here `h` is a bandwidth in physical time and `phi` is the standard normal
density. Normalize by all `N` samples to retain the simulated choice
probability; normalizing only by matching choices gives a conditional RT
density instead. Keep categories separate.

Unlike smoothing bin counts after integer assignments, this uses each sample's
actual RT. It changes smoothly when nondecision time shifts the sample. With
integer stopping times and hard bins, the exact binned objective is piecewise
constant in nondecision time between bin crossings; post-binning Gaussian
smoothing does not remove this problem.

A continuous kernel does not fix discrete stopping-time or choice gradients.
Its bandwidth creates a bias/variance tradeoff and must be reported and tested.
At finite bandwidth it is a smoothed density estimate, not the exact likelihood
of the integer-step process. Interpreting it as a measurement-noise model is a
separate scientific choice, not an automatic justification for smoothing.
Handle support, tails, and any background/pseudocount density explicitly.

## General compiler design

Build a reusable capability for supported components; DAWA should be a client,
without model-name recognition. Relevant existing integration points are:

- [Component specifications](../../../psyneulink/core/batched/specs.py): parameter
  bindings, state and trial-state declarations, random streams, step/readout
  emitters, and immutable implementation snapshots.
- [Kernel IR](../../../psyneulink/core/batched/kernel_ir.py): scheduling,
  state updates, resets, and modulation across a composition.
- [Likelihood contracts](../../../psyneulink/core/batched/likelihood_ir.py) and
  the existing likelihood planner: declared effects, observation semantics,
  capability diagnostics, and probability targets.
- [Symbolic equation support](../../../psyneulink/core/batched/continuous_ir.py):
  reusable algebra/derivative machinery. Its continuous declarations do not
  establish equivalence to a source scheduler or discrete update.

Proposed extensions, not existing API promises:

1. **Smooth derivative rules.** Register derivatives of supported discrete
   updates and readouts, initially Linear, Logistic, projections, LCA, and FHN.
   Some implementations emit opaque Triton source; their current metadata is
   insufficient for automatic differentiation. Start with explicit rules or
   shared differentiable equations. Preserve the actual source update order.
2. **Probability and event rules.** Describe transition laws, reparameterization
   maps, transition scores, support restrictions, and supported crossing or
   conditional-sampling operations. RNG ownership alone is not such a rule.
3. **Composition.** Propagate state sensitivities and score information through
   inputs, recurrence, held/sampled controller values, resets, and trial history.
   Respect frozen construction-time state as well as parameter-dependent
   initialization. Sum contributions into shared and conditional PEC fitting
   coordinates. Specialize only non-fitted inputs, as in ordinary simulation.
4. **Observation and reduction.** Generate kernel contributions and derivative
   reductions in the fused scorer, including normalization and regularization.
   Avoid materializing every trajectory. Preserve masked trials' state effects.
5. **Inspectable gradient plans.** Report parameter ordering, supported
   derivatives, method, probability target, assumptions, approximations,
   truncation policy, seed/RNG mode, and uncertainty diagnostics. Unsupported
   parameter paths should give specific diagnostics, not silent zero gradients.

The common `value_and_grad`/`explain` conventions are a useful interface model,
but sampling-gradient estimators must remain distinguishable from deterministic
analytic/numerical derivatives. Align cache keys and frozen registrations with
the selected derivative rules, parameter directions, and estimator configuration.
Do not claim gradients of finite-precision PRNG bit-pattern probabilities from
an ideal-Gaussian transition contract.

For the eight DAWA fitting coordinates, start by investigating forward
sensitivities that carry derivatives alongside the current state. This avoids
a tape of every step for every estimate. Extra register use may hurt occupancy
or cause spills; consider dependency pruning and small groups of parameter
directions. Reverse mode with replay/checkpointing can be reconsidered for larger
parameter sets. Actual overhead needs GPU measurement; no multiplier is promised.

## Suggested restart sequence and acceptance criteria

1. Establish a small CPU/reference estimator on a Gaussian model with known
   probability gradients. Add a stopped DDM and a nonlinear LCA to test event
   handling. Finite differences may be a validation oracle, not the production
   gradient estimator; compare matching discretizations and observation targets.
2. Test a single trial of full-noise DAWA, including threshold and LC parameters
   from the start. Success only for nondecision time is insufficient evidence
   for gradient-based fitting of the model.
3. Add short sequences with retained control state, varying trial conditions,
   and masked scoring. Check state/gradient reset semantics before full subjects.
4. Measure gradient uncertainty across independent seeds and estimate counts,
   both far from and near a fitted solution. Check useful descent directions,
   probability normalization, bandwidth sensitivity, timestep refinement, and
   boundary cases. Do not infer gradient quality from likelihood precision alone.
5. Generate the admitted estimators through the component interfaces and compare
   CPU/reference and GPU behavior. Profile registers, spills, memory, and total
   value-and-gradient cost on the same H100 workloads used for ordinary scoring.
6. Compare optimizers with matched starts, bounds, observation objectives, and
   wall-clock budgets. Count line-search evaluations and all simulation work.
   Independently rescore solutions with fresh, larger simulation budgets; check
   that an apparent benefit is not fitting one fixed seed or a smoothing artifact.

Proceed to broader compiler coverage only if the prototype gives reproducible
search improvements per GPU-minute. If raw trajectory scores are too noisy,
prioritize conditional averaging/variance reduction before larger Monte Carlo
budgets or more general automatic differentiation. Keep any softened stopping
surrogate explicitly separate from the original probability target.

## References

- [Schulman et al., Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254):
  composition of pathwise and score-function estimators.
- [Mohamed et al., Monte Carlo Gradient Estimation in Machine Learning](https://arxiv.org/abs/1906.10652):
  estimator assumptions and bias/variance tradeoffs.
- [Glasserman and Staum, Conditioning on One-Step Survival](https://business.columbia.edu/faculty/research/conditioning-one-step-survival-barrier-option-simulations):
  conditional simulation for barrier events; a methodological lead, not a ready
  implementation for DAWA's recurrent multi-accumulator model.
- [Lew et al., ADEV](https://arxiv.org/abs/2212.06386):
  compositional differentiation of probabilistic-program expectations under
  explicit rules; not an automatic solution for arbitrary Triton code/events.
