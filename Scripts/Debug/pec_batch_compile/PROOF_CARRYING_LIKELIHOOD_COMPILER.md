# Proof-Carrying Likelihood Compilation for PsyNeuLink

## Status

This document records a long-term research direction. It is not a description
of a currently implemented PsyNeuLink feature or a commitment to a public API.
The immediate purpose is to preserve the idea, define it precisely enough to
evaluate, and identify a path from the present batched compiler and CSI direct
likelihood work to a general research prototype.

The central goal is not a compiler specifically for decision models, drift
diffusion models, choices, or reaction times. It is a likelihood compiler for
arbitrary data-generating PsyNeuLink Compositions. Decision models are one
important test family, and CSI is a particularly useful benchmark, but neither
defines the scope of the proposed system.

## Executive summary

The proposed system would take an executable PsyNeuLink Composition, a
parameter declaration, and an observation specification, and attempt to
produce the strongest likelihood artifact it can justify. Possible results
include:

1. a certified closed-form likelihood;
2. a certified equation characterizing the likelihood;
3. a deterministic numerical likelihood with a certified error bound;
4. an exact or partially marginalized state-space calculation;
5. a certified unbiased Monte Carlo likelihood estimator;
6. an explicitly labeled approximate likelihood; or
7. a simulator-only result with a diagnosis of why stronger compilation failed.

Large language models would be used for the creative parts of likelihood
derivation: proposing factorizations, latent-state representations, changes of
variables, marginalizations, governing equations, boundary conditions, and
closed-form solutions. They would not be trusted to decide correctness.
Deterministic compiler analyses, proof checking, numerical validation, and
comparison with the original simulator would accept, reject, or refine their
proposals.

The ideal result is a **proof-carrying likelihood**: generated likelihood code
accompanied by a machine-checkable statement connecting it to the formal
semantics of the source Composition.

## The central idea

Treat a PsyNeuLink Composition as an executable specification of a probability
law rather than merely as code that produces samples. For parameters
`theta`, initial conditions `s0`, and an observation map `observe`, execution
induces an observation law

```text
Composition + theta + s0 + observation specification
    -> distribution over recorded data
```

The likelihood compiler attempts to transform this law into an efficiently
evaluable representation while preserving its meaning.

At the mathematical level, a Composition should denote a probability kernel

\[
K_C(\theta, s_0; d\omega),
\]

over execution traces or terminal states. The experiment's observation map
`O` pushes this kernel forward to a law over recorded data:

\[
P^Y_{\theta,s_0} = O_\# K_C(\theta,s_0).
\]

When this observation law is absolutely continuous with respect to an
appropriate reference measure `nu`, its likelihood is a Radon--Nikodym
derivative:

\[
L(\theta;y)
=
\frac{dP^Y_{\theta,s_0}}{d\nu}(y).
\]

This definition covers far more than an ordinary continuous probability
density. Choosing and proving the appropriate reference measure is itself part
of likelihood compilation.

## Why the observation specification is essential

A Composition alone does not completely determine an experimental likelihood.
The compiler must also know what constitutes the recorded data. For example,
an experiment could record:

- an exact continuous Mechanism value;
- a value rounded to a fixed precision;
- a discrete category obtained by thresholding a continuous state;
- a noisy measurement of a latent value;
- an event time or first-passage event;
- a censored event time;
- a fixed-length time series;
- a variable-length sequence of events;
- only selected trials or selected output ports;
- missing values under a stated missingness mechanism; or
- a timeout or nontermination outcome.

Each produces a different likelihood even when the underlying Composition is
unchanged. The compiler therefore needs an explicit `ObservationSpec` rather
than inferring all observation semantics from whichever values happen to be
returned by a Python call.

A future observation specification should describe at least:

- observed ports, states, events, and timestamps;
- discrete, continuous, structured, or mixed observation type;
- recording precision, binning, rounding, and censoring;
- measurement noise and missingness rules;
- trial and sequence boundaries;
- the conditioning relationship between inputs and outputs; and
- the reference measure when supplied by the user.

## Likelihoods need not be ordinary densities

Arbitrary Compositions can induce observation laws containing:

- probability masses over discrete values;
- continuous densities;
- mixtures of masses and densities;
- deterministic Dirac measures;
- singular distributions supported on lower-dimensional manifolds;
- variable-dimensional values or execution traces; and
- subprobability mass when termination is not guaranteed.

A deterministic transformation observed without measurement noise generally
does not have a density with respect to Lebesgue measure. A choice and reaction
time can use counting measure times Lebesgue measure. Rounded continuous data
are discrete even if the underlying latent value is continuous. A timeout may
introduce an atom in addition to a continuous event-time density.

For this reason, the compiler's primary object should be a measure or
probability kernel. A scalar `log_likelihood(theta, data)` is one possible
lowered interface, not the foundational semantics.

## What “arbitrary Composition” can realistically mean

No system can guarantee a closed-form likelihood for every arbitrary program.
Symbolic integration, termination, program equivalence, and many other
semantic properties are undecidable for sufficiently expressive programs.
User-defined Python functions can also hide state changes, external effects,
or random-number generation from static analysis.

The appropriate goal is therefore a sound but incomplete compiler:

- Every well-specified Composition can be submitted for analysis.
- The compiler never silently claims an exact likelihood it has not justified.
- It returns the strongest certified or empirically qualified artifact found.
- Unsupported or opaque constructs produce explicit proof obligations or a
  fallback, rather than an unsound assumption.
- The operator and theorem libraries can grow over time without changing the
  source-model abstraction.

“Arbitrary” means general admission and honest diagnosis, not guaranteed
symbolic success.

## Spectrum of possible output artifacts

### 1. Certified closed form

The best result is an explicit likelihood expression, along with any necessary
support conditions and a proof that it denotes the observation law. The
compiler can generate values, log values, gradients, and possibly higher-order
derivatives from this expression.

Examples include standard distributions, invertible deterministic
transformations with Jacobians, finite-state marginalizations, and special
stochastic processes with known transition or first-passage laws.

### 2. Certified equation

The compiler may derive a differential, integral, recurrence, or fixed-point
equation whose unique solution is the desired likelihood. Candidate equation
families include:

- forward and backward Kolmogorov equations;
- Fokker--Planck equations;
- finite-state master equations;
- first-passage boundary-value problems;
- Volterra and renewal equations;
- Chapman--Kolmogorov recurrences;
- hidden Markov forward recurrences;
- dynamic programs over finite execution structure; and
- deterministic change-of-variables equations.

This is already a major success even when no closed form is found. It replaces
an opaque simulator with a precise mathematical characterization that can be
solved by several independent numerical methods.

### 3. Certified numerical likelihood

A numerical solver can lower a certified equation into an executable
likelihood. The strongest form would return a value and a rigorous bound:

\[
|\widehat L(\theta;y)-L(\theta;y)| \leq \epsilon.
\]

The certificate must address truncation, discretization, boundary treatment,
and floating-point error. A proof that a PDE characterizes the likelihood does
not by itself prove that a C++, Torch, or Triton program solves that PDE
accurately.

Weaker initial versions can prove consistency of the numerical scheme and use
convergence studies and independent solvers for empirical error control. They
must distinguish those guarantees from a formal runtime error certificate.

### 4. Exact structured inference

Some stateful models admit exact recursive inference without a single closed
form for the full data likelihood. Examples include finite-state hidden Markov
models and linear-Gaussian state-space models. A compiler can generate the
appropriate forward algorithm, Kalman filter, variable elimination, or other
registered exact operator.

### 5. Partially marginalized or Rao--Blackwellized inference

When only part of the latent state is analytically tractable, the compiler can
integrate that portion exactly and sample the remainder. The artifact should
identify which variables are represented symbolically, deterministically, or
by particles, and why the partition is valid.

### 6. Certified unbiased likelihood estimator

For general stochastic persistent state, a particle filter or related method
may provide an unbiased estimate of the marginal likelihood under stated
conditions. The proof obligation changes from pointwise equality to something
like

\[
\mathbb{E}[\widehat L(\theta;y)] = L(\theta;y).
\]

This result is valuable for pseudo-marginal MCMC even though individual
evaluations remain noisy.

### 7. Explicit approximation or simulation only

Synthetic likelihoods, neural density estimators, approximate Bayesian
computation, discretized observation models, and other approximations may be
useful when stronger results are unavailable. They must be labeled with their
assumptions and validation evidence. The final fallback is the original
simulator.

## Model structure the deterministic compiler should analyze

The LLM should not rediscover facts that an ordinary compiler can establish
more reliably. A deterministic front end should extract:

- parameter, subject, trial, time, and simulation-lane dependencies;
- persistent and reset state;
- deterministic and stochastic operations;
- sources, ownership, and addressing of randomness;
- scheduler conditions and termination conditions;
- control and data dependencies;
- observation dependencies;
- state updated at event boundaries or continuously during execution;
- feedback from stochastic stopping times into other state;
- conditional independence and exchangeability;
- static and dynamic execution topology; and
- opaque operations requiring contracts.

The current PEC batched compiler's graph and axis-dependency analyses are a
useful starting point, but likelihood compilation needs a richer probabilistic
and observational intermediate representation.

## A proposed likelihood intermediate representation

The compiler should lower the Composition into a small, typed, explicitly
probabilistic language. A sketch of its concepts is:

```text
Model
  parameters
  initial_state
  inputs
  state_variables
  random_variables
  transitions
  scheduler
  termination
  observations

StateVariable
  domain
  persistence
  reset_policy
  update_equation
  time_semantics

RandomVariable
  distribution_or_process
  parameters
  rng_identity
  independence_contract

Observation
  source
  transformation
  recording_model
  support
  reference_measure

LikelihoodPlan
  factorization
  eliminated_variables
  conditioned_variables
  equations
  operator_bindings
  approximations
  proof_obligations
  lowering_strategy
```

The language should be restrictive enough to formalize and validate, while
allowing opaque primitives with declared semantic contracts. It should avoid
making raw Python or generated C++ part of the trusted mathematical core.

## Operator, equation, and theorem registries

General compilation requires reusable knowledge. Each supported primitive or
model family should advertise capabilities such as:

```text
Simulator semantics
Transition kernel
Closed-form density or mass function
Moment-generating or characteristic function
Forward/backward equation
First-passage equation
Exact filtering operator
Reparameterization
Gradient and sensitivity rules
Support and reference measure
Formal theorem identifier
Numerical implementations and validity domains
```

For example, a diffusion Mechanism might expose simulator semantics and a
general Fokker--Planck construction. A particular constant-coefficient case
might additionally expose a closed form. A linear-Gaussian subgraph might
expose a Kalman update. A generic stochastic Mechanism might expose only a
simulation kernel.

The compiler can compose these local capabilities. It does not need an LLM to
rederive the normal density or Kalman filter on every run.

## The role of the LLM

The LLM is most valuable where the derivation requires mathematical creativity
rather than routine graph analysis. Its jobs can include:

- selecting a useful latent-state representation;
- finding sufficient state or a Markov augmentation;
- recognizing known stochastic-process families;
- proposing conditional-independence factorizations;
- selecting variables to integrate out;
- deriving changes of variables and Jacobians;
- deriving forward, backward, master, renewal, and first-passage equations;
- stating initial, boundary, interface, and normalization conditions;
- proposing nondimensionalizations and transformations of moving domains;
- guessing closed-form solutions;
- retrieving relevant identities and theorem templates;
- proposing exact/sampled partitions for hybrid inference;
- generating proof sketches and formal lemmas;
- interpreting failed proof obligations;
- designing limiting-case and metamorphic tests; and
- revising a derivation in response to counterexamples.

The LLM should generate a typed likelihood plan, equations, and proof terms or
proof obligations—not an unrestricted native implementation that is accepted
because it compiles.

## Synthesis and counterexample loop

A candidate synthesis loop could be:

```text
1. Freeze the source Composition and ObservationSpec.
2. Lower them to the likelihood IR.
3. Run deterministic dependency and measure analyses.
4. Retrieve applicable operators, equations, examples, and theorems.
5. Ask the LLM for one or more derivation plans.
6. Type-check the plans and reject semantic inconsistencies.
7. Attempt symbolic simplification and proof construction.
8. Generate executable candidate likelihoods or equation solvers.
9. Compare each candidate with the simulator and independent oracles.
10. Return counterexamples and failed obligations to the synthesizer.
11. Repeat until a certified artifact is found or the search budget expires.
12. Emit the artifact, assumptions, certificate, tests, and diagnostics.
```

Generating several genuinely different derivations is preferable to asking one
model to restate the same argument. Agreement between a forward equation, a
backward equation, and simulation is more informative than superficial
self-consistency.

## Proof-carrying likelihoods

### The trusted statement

The most important theorem should relate the generated result to the formal
denotation of the source model. A schematic statement is:

```lean
theorem compiled_likelihood_correct
    (theta : Parameters)
    (h_theta : Admissible theta) :
    observationLaw composition observationSpec theta =
      withDensity referenceMeasure (likelihood theta)
```

Equivalent formulations may use equality of integrals over every measurable
set:

\[
P^Y_\theta(A)
=
\int_A L(\theta;y)\,\nu(dy)
\quad\text{for every measurable }A.
\]

This prevents a common failure mode in generated mathematics: proving useful
properties of a candidate expression without proving that the expression is
actually the observation law of the source program.

### Equation certificates

For an equation-derived likelihood, the proof can be factored into:

1. execution of the source probabilistic program induces a stated process;
2. the process's observation law satisfies the generated equation and
   conditions;
3. the equation and conditions admit a unique admissible solution; and
4. the compiled likelihood is that solution.

The uniqueness step is crucial. Merely showing that a proposed function
satisfies a differential equation is not enough when other solutions satisfy
the same incomplete conditions.

### Closed-form certificates

For a novel closed form, the proof should establish:

- the expression is defined over the declared parameter and observation
  domain;
- it has the required measurability, support, and nonnegativity;
- it satisfies the governing equation and all boundary/initial conditions;
- it has the required normalization or subprobability mass;
- uniqueness identifies it with the observation law; and
- any parameter differentiation is valid under the required regularity
  assumptions.

### Factorization certificates

When the compiler parallelizes trials or subgraphs, it should prove the
conditional independence or deterministic conditioning that permits the
factorization. This is particularly important for persistent-state models,
where an apparently harmless trialwise product can define the wrong
stochastic process.

### Estimator certificates

A sampling implementation needs theorem statements appropriate to its role,
including unbiasedness or consistency, treatment of resampling, support of
proposal kernels, and the relationship between the estimator and the desired
marginal likelihood.

## What Lean can and cannot certify initially

Lean and mathlib already contain substantial measure theory, probability
density, probability-kernel, disintegration, and Radon--Nikodym infrastructure.
That makes the semantic target plausible.

The difficult engineering work would include:

- defining a formal denotation for the likelihood IR;
- connecting PsyNeuLink scheduler behavior to that denotation;
- formalizing the required stochastic processes;
- building reusable theorems for program transformations;
- proving existence and uniqueness results for generated equations;
- representing first-passage and stopped-process semantics; and
- connecting real-number mathematical algorithms to floating-point kernels.

The first prototype need not formally verify a production PDE solver. A useful
progression is:

1. prove the symbolic factorization or governing equation;
2. validate a conventional solver extensively against simulation;
3. prove convergence of the numerical scheme;
4. add interval or residual-based runtime error bounds; and
5. eventually verify the generated implementation or its checker.

A Lean certificate proves the formal statement presented to Lean. It does not
prove that the formalized model faithfully describes the user's intended
experiment. Keeping the translation from PsyNeuLink into the formal IR small,
deterministic, inspectable, and tested is therefore as important as checking
the final proof.

## Simulator-based verification

The executable source model provides a valuable differential-testing oracle.
Every candidate should be tested against samples from the same frozen semantic
configuration used to generate its formal model.

### What simulation can establish

Simulation can strongly falsify an incorrect candidate and quantify agreement
over scientifically relevant parameter regions. It is especially effective
for:

- low-dimensional marginals;
- transition laws and short conditional segments;
- moments, quantiles, event probabilities, and correlations;
- known symmetries and invariants;
- parameter-dependent changes in distributions; and
- convergence comparisons between discrete and continuous semantics.

### What simulation cannot establish

Finite simulation cannot prove equality of distributions. It can miss errors
in rare events, tails, isolated parameter regimes, or high-dimensional
dependencies. Moreover, an accurate sampled density estimate may be
impractical for high-dimensional observations.

The system should therefore avoid assuming that it can always compare two
pointwise likelihood values. Depending on the observation space it can use:

- exact probability comparisons on discrete finite spaces;
- histogram or kernel comparisons in low dimensions;
- probability integral transform tests;
- energy distance or kernel two-sample statistics;
- classifier-based two-sample tests;
- characteristic functions;
- marginal and conditional moment tests;
- posterior predictive checks;
- simulation-based calibration;
- likelihood-ratio identities; and
- local transition-kernel tests instead of full-trajectory density estimation.

### Required test design

Tests should cover:

- interior and boundary parameter values;
- limiting cases that remove interactions or noise;
- short and long execution horizons;
- common and rare outcomes;
- reset and persistent-state configurations;
- alternative scheduler paths;
- multiple random seeds and simulation resolutions; and
- transformations expected to leave the observation law unchanged.

An adversarial parameter search can actively look for locations where the
candidate and simulator disagree.

## Numerical verification and the implementation gap

There are three distinct objects that must not be conflated:

1. the source Composition's mathematical semantics;
2. a derived exact expression or characterizing equation; and
3. a finite-precision implementation of that expression or equation.

Proof of `1 = 2` does not imply `2 = 3`. The implementation may contain
discretization, truncation, iterative-solver, underflow, or code-generation
errors.

Possible bridges include:

- interval arithmetic;
- outward-rounded special-function implementations;
- residual bounds combined with stability estimates;
- validated ODE/PDE integration;
- exact rational checks on reduced test cases;
- independently generated CPU and GPU implementations;
- refinement and Richardson-style convergence studies; and
- proof-producing numerical kernels or small verified result checkers.

The artifact manifest should state exactly which layer each guarantee covers.

## CSI as a benchmark, not the definition

CSI is a valuable early benchmark because it contains several difficult
features in a compact model:

- persistent deterministic LCA state;
- continuous within-trial evolution;
- a stochastic DDM stopping time;
- scheduler coupling through stopping duration;
- time-varying DDM coefficients derived from the LCA;
- first-passage observations;
- moving boundaries; and
- parameter-dependent timing.

Conditioning on observed reaction times makes the LCA history deterministic,
allowing a sequential history scan followed by conditionally independent
first-passage calculations. The direct implementation then solves the
resulting time-dependent first-passage problem numerically.

An LLM-driven benchmark can deliberately hide the human direct-likelihood
implementation and provide only:

- the executable PNL model;
- its formal scheduler and state semantics;
- the observation specification;
- the primitive operator/theorem library; and
- access to the simulator as a test oracle.

The experiment would measure whether the system rediscovers the correct
factorization, derives the correct equation and conditions, produces an
equivalent solver, and identifies the assumptions under which the
transformation is valid.

CSI should be followed by non-decision examples so that the architecture does
not accidentally encode choice/RT assumptions. Candidate families include
learning models, control systems, recurrent neural mechanisms, evidence-free
time-series models, finite-state task models, and noisy deterministic dynamical
systems.

## Compiler architecture

### Front end: semantic freezing

The front end resolves PsyNeuLink objects, scheduler rules, parameters,
initialization, reset behavior, and observation declarations into an immutable
model snapshot. Dynamic Python behavior outside the accepted subset is either
captured through a contract or rejected for certified compilation.

### Deterministic analysis

The analysis pass establishes state persistence, randomness, dependency,
termination, observation type, and possible factorization. It also identifies
unknown facts and proof obligations.

### Knowledge retrieval

The system retrieves only operators, equations, formal theorems, numerical
methods, and validated examples whose applicability conditions can be stated
in the likelihood IR. Literature retrieval may inform synthesis, but a citation
is not a proof that a formula applies to the current program.

### LLM synthesis

One or more LLM agents propose typed derivations. They may invoke symbolic
algebra, automatic differentiation, equation solvers, theorem provers, and
simulation experiments. Generated plans retain provenance for assumptions and
transformations.

### Verification

Plans undergo type and dimensional checks, semantic dependency validation,
symbolic identities, proof checking, normalization checks, and adversarial
simulation comparison.

### Planning and lowering

The compiler selects the strongest verified plan under user constraints on
accuracy, runtime, gradients, and target hardware. A backend-neutral
Likelihood Plan IR lowers to CPU, GPU, autodiff, optimization, HMC, or
pseudo-marginal interfaces as appropriate.

### Artifact manifest

Every compiled likelihood should report:

- frozen source-model identity;
- observation specification;
- parameter domain and assumptions;
- selected reference measure;
- likelihood factorization;
- exact and approximate transformations;
- formal statements proved and axioms used;
- numerical solver and tolerances;
- validation experiments and failures;
- unsupported source behavior;
- gradient semantics; and
- reproducible build information.

## Example diagnostic outcomes

The public interface should explain decisions rather than merely returning a
function. Illustrative results are:

```text
Result: certified closed-form likelihood
Observation law: mixed discrete/continuous
Reference measure: counting(choice) x Lebesgue(time)
Persistent state: none
Parallel dimensions: subject x trial x parameter
Proof: accepted
```

```text
Result: certified equation + validated numerical solver
Persistent state: deterministic after conditioning on observations
History phase: sequential
Emission phase: conditionally parallel
Equation: moving-boundary first-passage Fokker--Planck problem
Symbolic proof: accepted
Numerical certificate: empirical refinement only
```

```text
Result: particle likelihood estimator
Persistent state: stochastic and observation-coupled
Exact substructure: linear-Gaussian block marginalized
Remaining state: particle representation
Unbiasedness proof: accepted under listed proposal-support assumptions
```

```text
Result: simulator only
Reason: opaque user function performs undeclared external randomness
Required action: provide a semantic contract or registered stochastic operator
```

## Relationship to existing research

Several research systems provide important parts of this vision:

- **Hakaru** performs semantics-preserving transformations such as
  disintegration and symbolic simplification over a measure-oriented
  probabilistic language.
- **Delayed sampling** and **Birch** dynamically retain analytically tractable
  random variables and automatically Rao--Blackwellize portions of sequential
  Monte Carlo programs.
- **ProbZelus semi-symbolic inference** combines exact symbolic state with
  sampling in streaming probabilistic programs.
- **Siren** represents hybrid particle-filter inference plans and statically
  checks whether requested symbolic/sampled partitions can be implemented.
- **Augur** extracts conditional independence from Bayesian networks to
  generate data-parallel GPU inference.
- **RootPPL/CorePPL** compiles probabilistic programs into parallel sequential
  Monte Carlo implementations.
- **Infer.NET** compiles factor graphs into scheduled calls to registered
  factor-specific message operators.

The proposed contribution combines ideas from these areas but changes the
source domain and synthesis ambition. The source is an operational,
scheduler-driven scientific model; the system attempts open-ended equation or
closed-form discovery; the simulator supplies counterexamples; and the final
artifact may carry a machine-checked equivalence certificate.

Relevant starting references:

- Narayanan et al., *Probabilistic Inference by Program Transformation in
  Hakaru*, 2016:
  <https://www.cas.mcmaster.ca/~carette/publications/hakaru_system.pdf>
- Murray et al., *Delayed Sampling and Automatic Rao--Blackwellization of
  Probabilistic Programs*, 2018:
  <https://proceedings.mlr.press/v84/murray18a.html>
- Baudart et al., *Reactive Probabilistic Programming*, 2020:
  <https://arxiv.org/abs/1908.07563>
- Atkinson et al., *Semi-Symbolic Inference for Efficient Streaming
  Probabilistic Programming*, 2022:
  <https://arxiv.org/abs/2209.07490>
- Cheng et al., *Inference Plans for Hybrid Particle Filtering*, 2024:
  <https://arxiv.org/abs/2408.11283>
- Lundén et al., *Compiling Universal Probabilistic Programming Languages with
  Efficient Parallel Sequential Monte Carlo Inference*, 2021:
  <https://arxiv.org/abs/2112.00364>
- Tristan et al., *Augur: Data-Parallel Probabilistic Modeling*, 2014:
  <https://papers.neurips.cc/paper_files/paper/2014/file/2d6e6b9675fb31f6c5250b7ea73fc37d-Paper.pdf>
- Lean mathlib probability-density and probability-kernel documentation:
  <https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Density.html>
  and
  <https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Kernel/RadonNikodym.html>

This area should not be confused with work commonly called *inference
compilation*, in which a neural network is trained to amortize approximate
posterior inference. Here, the intended compiler derives or characterizes the
likelihood defined by an executable model.

## Trust model and principal risks

### LLM hallucination

An LLM can produce plausible but incorrect factorizations, signs, Jacobians,
boundary conditions, support constraints, and proofs. All LLM output is
untrusted until checked.

### Proving the wrong statement

A formally valid proof is irrelevant if the formalized source model or
observation specification differs from the executed Composition. Translation
validation and an inspectable semantic snapshot are required.

### Likelihood hacking

A generated expression can score observed data highly while failing to
normalize or otherwise not representing a probability law. Normalization,
support, and equivalence to the observation law are mandatory proof
obligations. Restricting generation to a safe typed IR reduces this risk.

### Numerical mismatch

The exact derivation may be correct while the finite numerical implementation
is unstable or biased. Mathematical and implementation certificates must be
reported separately.

### Semantic ambiguity

Scheduler timing, initialization, reset policies, recording precision, and
random-stream semantics can change the data-generating law. They must not be
filled in silently by the synthesizer.

### Missing density

The requested scalar density may not exist relative to the assumed reference
measure. The correct outcome may be a mixed or singular measure rather than a
failure of mathematical ingenuity.

### Computational intractability

An exact characterization can still be too expensive for inference. The
compiler needs a cost model and a principled route to hybrid or approximate
plans.

### Scientific adequacy

A perfectly compiled likelihood proves fidelity to the programmed model, not
that the model is identifiable, empirically adequate, or scientifically
meaningful. Those remain statistical and scientific questions.

## Evaluation program

The research prototype should be judged on several independent axes.

### Semantic correctness

- Does the artifact represent the exact observation law under its stated
  assumptions?
- Are state, timing, reset, and conditioning semantics preserved?
- Does it fail closed on unsupported behavior?

### Derivation capability

- How often does it identify a useful factorization?
- How often does it derive the correct governing equation?
- Can it rediscover held-out known closed forms?
- Can it produce valid novel closed forms on problems without a supplied
  solution?

### Proof capability

- What fraction of correct candidates receive machine-checked certificates?
- Where are missing library theorems the bottleneck?
- How often does formalization expose an incorrect informal derivation?

### Empirical agreement

- Does the candidate pass adversarial simulator comparisons?
- Are tail probabilities and rare scheduler paths preserved?
- Does agreement improve under numerical refinement?

### Statistical utility

- Are gradients accurate and useful?
- Does the likelihood support optimization, HMC, or pseudo-marginal MCMC?
- Does parameter recovery achieve calibrated uncertainty?

### Performance

- What compilation and proof-search cost is amortized over repeated inference?
- How does runtime compare with simulation-based likelihoods?
- Does structural specialization expose useful CPU or GPU parallelism?

### Generality

- What fraction of representative non-decision PNL models reach each artifact
  tier?
- Which unsupported constructs account for most failures?

## Proposed staged research roadmap

### Phase 0: define the semantic contract

- Specify parameters, input data, latent state, outputs, observations, and
  randomness independently.
- Define what it means for two likelihood artifacts to be equivalent.
- Define guarantees and vocabulary for exact, certified numerical, unbiased,
  consistent, and approximate results.
- Select a small `CorePNL` subset with explicit scheduler semantics.

### Phase 1: likelihood IR and deterministic diagnosis

- Lower supported Compositions and `ObservationSpec` values to the likelihood
  IR.
- Analyze state persistence, reset behavior, stochasticity, observation
  dependencies, reference-measure candidates, and conditional independence.
- Produce diagnostics without generating new likelihood mathematics.
- Cover discrete, continuous, mixed, and deterministic toy models.

### Phase 2: registered exact operators

- Introduce the operator/equation registry.
- Compose known probability masses, densities, transformations, HMM recurrences,
  and linear-Gaussian filters.
- Generate CPU likelihoods and gradients from trusted primitives.
- Establish simulator differential tests for every operator.

### Phase 3: LLM likelihood-plan synthesis

- Define the restricted output grammar for derivation plans.
- Supply compiler facts and retrieved theorem/operator metadata as context.
- Build the derive, execute, test, counterexample, and revise loop.
- Benchmark rediscovery of known factorizations while hiding their
  implementations.

### Phase 4: equation synthesis

- Add templates for master, Fokker--Planck, backward, renewal, and first-passage
  equations.
- Validate generated equations on models with known solutions.
- Generate independent solvers and convergence tests.
- Use CSI as a held-out, stateful first-passage benchmark.

### Phase 5: proof-carrying symbolic artifacts

- Formalize the likelihood IR's measure/kernel semantics in Lean.
- Prove reusable compiler transformations and operator contracts.
- Have the LLM generate proof obligations and candidate Lean proofs.
- Initially certify factorizations and closed forms for small models.
- Expand toward equation characterization and uniqueness results.

### Phase 6: numerical certification

- Connect certified equations to validated numerical methods.
- Add interval, residual, stability, and floating-point error analysis.
- Emit values with error bounds where practical.
- Maintain ordinary high-performance implementations checked against slower
  certified references.

### Phase 7: hybrid and general stochastic compilation

- Generate exact/sampled state partitions.
- Add Kalman, finite-state, Rao--Blackwellized, and generic particle plans.
- Prove estimator properties under explicit assumptions.
- Lower plans to the batched CPU/GPU compiler.

### Phase 8: broad PsyNeuLink evaluation

- Assemble a corpus spanning decision, learning, control, recurrent,
  time-series, and event-driven models.
- Hold out mathematical solutions during synthesis.
- Measure coverage by artifact tier, proof success, empirical correctness,
  inference utility, and performance.

## A focused first research prototype

The first end-to-end prototype should remain smaller than the ultimate vision:

1. Define a likelihood IR for deterministic state, standard random draws,
   continuous ODE state, one-dimensional diffusions, scheduler events, and
   explicit observations.
2. Encode several toy models plus CSI in that IR.
3. Give the LLM the IR and an equation/operator library, but withhold the human
   likelihood derivations.
4. Require it to return a factorization, governing equation, conditions,
   numerical plan, and validation plan.
5. Reject plans that conflict with compiler-derived dependencies.
6. Compare generated predictions with large simulator experiments and known
   references.
7. Formalize the factorization and one or two tractable operator cases in Lean.
8. Record precisely which CSI proof steps remain outside the formal library.

This would answer the first important feasibility question: can an LLM, when
constrained by executable semantics and aggressive verification, reconstruct
non-obvious likelihood mathematics rather than merely generate plausible
prose?

## Open research questions

- What is the smallest formal IR that preserves PsyNeuLink's scientifically
  meaningful execution semantics?
- How should scheduler semantics be expressed as probability kernels?
- How should user-defined Mechanisms state and discharge semantic contracts?
- Can a dominating reference measure be synthesized compositionally?
- Which transformations can be proven once as compiler passes rather than for
  every generated model?
- Can the system reliably discover useful state augmentations?
- Can simulator counterexamples guide mathematical synthesis efficiently?
- How should equivalence be tested for high-dimensional or structured output
  laws?
- How much of first-passage theory and PDE uniqueness must be added to Lean?
- Can numerical solvers emit practical error certificates without destroying
  performance?
- How should the compiler trade exactness, variance, memory, and parallelism?
- Can proof and validation artifacts be reused across nearby model variants?
- How should scientific users inspect and approve assumptions introduced by
  synthesis?

## Potential research contribution

A concise statement of the proposed contribution is:

> A proof-carrying likelihood compiler for executable scientific models that
> combines deterministic program analysis, LLM-assisted mathematical
> synthesis, formal probabilistic semantics, simulator-driven counterexample
> generation, and heterogeneous exact, numerical, and sampling backends.

The most novel and risky component is open-ended equation and closed-form
derivation from a scheduled data-generating program. The simulator makes this
more tractable than unconstrained mathematical generation because it supplies
an executable semantic oracle. Formal certificates make it scientifically
defensible because empirical agreement alone cannot prove equality.

The long-term success criterion is not that every Composition yields a closed
form. It is that the compiler can automatically discover surprising exact
structure when it exists, prove what it has discovered, exploit partial
structure when a complete derivation is impossible, and state its remaining
uncertainty without silently changing the model.
