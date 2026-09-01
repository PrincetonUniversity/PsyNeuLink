# Direct-likelihood approaches for the CSI model

## Decision and prototype status

The current prototype targets the scientifically intended **continuous-time**
model rather than preserving the old 10 ms scheduler and Euler discretization.
It is implemented in `direct_likelihood/`, with the command-line entry point
`csi_direct_likelihood.py`. It leaves PsyNeuLink core code unchanged.

The implemented deterministic-LCA prototype includes all 13 fitted CSI
parameters:

- three condition-specific LCA gains;
- one continuous switch-CSI duration, with repeat CSI fixed to zero;
- three condition-specific initial boundaries;
- three condition-specific boundary-collapse rates in units per second; and
- three condition-specific nondecision times.

The two-unit LCA is integrated with RK4 through the 1 s ITI, CSI, and decision
period. During the decision period its logistic output defines a time-varying
DDM drift. A conservative Chang--Cooper finite-volume solver computes the DDM
first-passage density on a fixed transformed spatial interval, using
Crank--Nicolson with Rannacher startup steps and a differentiable batched
tridiagonal solve. The scored probability is choice-specific boundary flux
integrated over the 1 ms response-time recording interval.

The participant likelihood remains sequential. Every row advances the
persistent LCA state, while `likelihood_include_mask` controls only whether the
row contributes to the likelihood sum. An included row with a physically
invalid inferred decision time has zero probability. A masked invalid row
advances the LCA by zero decision time and is reported as a diagnostic.

Scoring now separates this sequential dependency from the parallel work. A
first pass advances the LCA and records every included trial's state at decision
onset. It then constructs decision-period drift paths in batches and solves the
DDM PDE in duration-sorted buckets, reducing padding and kernel-launch overhead.
The benchmark command reports these three phases independently. On the initial
subject-1 CPU benchmark with a debugging mesh (5 ms DDM cells, 33 spatial
points, and a 20 ms LCA maximum step), the refactor reduced an isolated score
from roughly 26 seconds to roughly 4.2 seconds without changing any probability
or latent endpoint. These numbers are machine- and mesh-specific rather than a
general performance claim.

For fitting, the original reverse-mode graph was prohibitive: on subject 1 at
the default 1 ms/65-point/10 ms mesh, one objective plus all 13 gradients took
about 86.9 seconds. Exact compact Torch adjoints first reduced that to 27.6
seconds. The CPU path now goes further by fusing the complete sequential LCA
scan, batched drift integrations, DDM forward time loop, and their analytic
reverse passes in a small native extension. OpenMP parallelizes independent
drift and DDM lanes while leaving the history-dependent subject scan serial. On
eight cores, a warm objective-gradient evaluation takes about 0.310 seconds and
a score takes about 0.078 seconds. One/two/four/eight-thread gradient medians
were 1.478/0.798/0.471/0.310 seconds, making the 8-thread path approximately
89-fold faster than the compact Torch path and 280-fold faster than the original
graph.

The native subject-1 log likelihood differs from the Torch/PCR result by less
than `1.8e-11`, and all 13 gradients differ by less than `5e-11`; the difference
comes from Thomas-versus-PCR floating-point solve order. Unit tests also compare
the native implicit PDE adjoint directly with the Torch VJP. The extension is
built once and cached when Ninja and a C++ compiler are available. The Torch
path remains the fallback and correctness oracle. A process containing one
warmup and two measured objective-gradient evaluations peaked at about 0.97 GB
resident memory. These timings are specific to the development CPU.

The fitter now uses the valid model default as its first start and optimizes in
normalized bound coordinates. Full-resolution, single-start pilots for subjects
1, 4, and 7 took approximately 102, 47, and 126 seconds for 99, 57, and 46
objective-gradient evaluations. Their fitted log likelihoods improved over the
defaults by 353.46, 576.29, and 62.67, respectively. Fresh forward-only scores
reproduced every fitted likelihood exactly and reported no invalid or
zero-probability included rows.

Valid random starts now enforce RT and moving-boundary constraints and are
screened by an actual positive-density score. The fitter scores a larger pool
of valid candidates and optimizes only the best requested starts. A three-start
subject-1 run found the same best basin from the default and one random start,
while another random start exposed a poorer local basin. Strict gradient
restarts and Powell did not materially improve the best solution. A multiscale
coordinate poll found no bounded axis move improving log likelihood by `1e-7`,
despite a nonzero smooth gradient at RT-bin kinks.

At the fitted points, default-grid log likelihoods for subjects 1, 4, and 7
differed from a 0.5 ms/129-point/5 ms LCA reference by -0.293, -0.156, and
-0.077. The recovery generator now uses a Brownian-bridge correction for paths
that cross and return between Euler endpoints. In two prescribed DDM cases,
including time-varying drift and a collapsing boundary, 100,000 simulated paths
at 0.5 ms agreed with a 0.25 ms/257-point PDE reference within 0.20 percentage
points and two Monte Carlo standard errors. The uncorrected endpoint detector
retained errors as large as 0.62 percentage points at that mesh. Per-trial
independent random streams also make all samples invariant to the maximum-time
safety horizon.

A corrected five-seed by two-truth recovery matrix used 485 included rows per
fit. After replacing two unmistakable local-search failures with expanded or
coarse-to-fine searches, every fit exceeded the finite-sample likelihood at its
generating vector. Mean normalized bound-range RMSE was 0.0597 for the interior
truth and 0.1084 for a deliberately contrasting truth. The contrasting regime
was weakest for gains and non-decision times. One failure was repaired by
screening 64 candidates and optimizing four starts; another needed a
1,000-evaluation coarse CMA-ES basin search followed by default-mesh L-BFGS-B.

Warm-started 0.5 ms/129-point/5 ms polishing changed the empirical subject-1
and interior-recovery vectors by only 0.0010 and 0.0015 normalized RMSE. It
improved fine-mesh likelihood by 0.0027 and 0.0019; the latter recovery's
parameter RMSE changed from 0.0615 to 0.0607. This supports default-mesh search
followed by selective fine-mesh final polishing. More truth regimes,
participant sequences, and PNL comparisons remain necessary before
all-participant fitting.

The successful coarse-to-fine fallback is now a single `staged-fit` workflow:
coarse CMA-ES basin discovery, default-mesh exact-gradient fitting, and optional
fine-mesh polishing. Its combined JSON exposes the final vector at top level
and retains per-stage meshes, results, and wall times.

A controlled comparison for subjects 1, 4, and 7 evaluated the rounded
direct-fit and legacy PNL-fit vectors under both objectives. Continuous CSI was
rounded to the nearest legacy 10 ms execution because the PNL scheduler
requires an integer count. The fine direct likelihood preferred the direct
vectors by 36.87, 21.01, and 12.46 log units. Three paired PNL histogram scores
with 10,000 simulations per seed preferred the PNL vectors by 4.64 and 23.41
units for subjects 1 and 4. For subject 7, the direct vector was preferred by
6.47 units on average, but the paired SD was 11.48. Absolute likelihood values
were not compared across estimators because the PDE RT-bin probability and
legacy histogram objective use different numerical scales.

The old PNL implementation remains a discretization/refinement oracle. The
validation command compares the direct solver with the analytic fixed-boundary
Wiener density and compares the continuous LCA endpoint with the actual PNL
Euler recurrence at successively smaller time steps.

## Motivation

The current CSI fitting workflow estimates a likelihood by simulating many
replications of every observed trial and constructing a smoothed histogram of
the simulated choice and response-time outcomes. GPU batching makes this
practical, but the resulting objective retains Monte Carlo error and requires
independent high-resolution rescoring.

The CSI model has more structure than a general likelihood-free model. Its
control process is a two-unit leaky competing accumulator (LCA), and its only
currently stochastic component is a drift diffusion model (DDM). This creates
opportunities to replace some or all of the simulation with a direct numerical
likelihood.

The implementation remains an exploratory research prototype; it is not yet a
validated replacement for the existing GPU likelihood workflow.

## Deterministic LCA and a direct DDM likelihood

With the current CSI configuration, the LCA trajectory is deterministic for a
fixed parameter vector and an observed task/RT sequence. The LCA continues to
evolve while the stochastic DDM is running, so the observed stopping time
determines the persistent LCA state passed to the next trial. There is no
dynamic DDM-to-LCA projection, but it would still be incorrect to replace each
trial by one fixed drift value. Instead, the direct solver computes the full
within-trial drift path and conditions the next LCA state on the observed RT.

The joint log likelihood then factorizes conditionally on that deterministic
trajectory:

\[
\log L(\theta)
=
\sum_i \log \int_{B_i}
j_{c_i}\left(
t-t_{0,i}-CSI_i\mid v_i(s;\theta),a_i(s;\theta),\sigma
\right)\,dt,
\]

where \(B_i\) is the RT recording interval, \(v_i(s;\theta)\) is the
time-varying LCA-derived drift, \(a_i(s)\) is the decision boundary,
\(t_{0,i}\) is nondecision time, and \(j_{c_i}\) is absorbed probability flux
through the observed boundary.

For fixed boundaries, the standard Wiener first-passage density can be
evaluated with a Navarro--Fuss-style series calculation. This would remove the
need for simulation counts, histogram bins, smoothing, pseudocounts, and
simulation-seed rescoring during fitting.

The fitted CSI model currently permits a linearly collapsing boundary. The
fixed-boundary formula does not directly apply in that case. Plausible direct
methods include a one-dimensional Fokker--Planck solver, a Volterra integral
equation for the two moving boundaries, or another numerical first-passage
solver. The absorbed probability flux at the upper and lower boundaries gives
the choice-specific response-time densities.

The existing simulator uses a discrete Euler update with a 0.01 time step,
whereas the conventional first-passage formulations describe a continuous-time
diffusion. The prototype explicitly chooses:

1. a continuous LCA ODE;
2. a continuous diffusion with a continuously collapsing boundary; and
3. PNL step-size refinement, rather than equality at 10 ms, as the compatibility
   criterion.

For response times recorded at finite precision, the likelihood should ideally
integrate density over the measurement interval rather than evaluate only the
density at a point.

The two objectives also differ statistically. Each legacy Monte Carlo lane
simulates the complete sequence with its own preceding RT history. The
trialwise histogram therefore averages a trial over simulated histories and
then sums the trial log densities. The direct likelihood instead propagates
the deterministic LCA through the participant's observed preceding RTs and
evaluates a sequential conditional likelihood. These coincide only when the
persistent state is effectively independent of history. A one-second ITI makes
the practical difference fairly small in the pilot ablation below, but the
objectives are not algebraically identical.

## Differentiation and gradient-based inference

The fixed-grid numerical PDE operations are implemented with differentiable
Torch operations. Automatic differentiation through the ordinary graph remains
the correctness oracle. The fitting path uses mathematically equivalent custom
vector-Jacobian products: it stores each PDE density/LCA state needed by the
reverse pass and applies the implicit transpose-tridiagonal derivative. The
native CPU path evaluates those recurrences in fused loops, while the Torch
path replays compiled local VJPs. Both preserve the exact discrete derivative.
This supports the prototype's bounded L-BFGS optimizer and leaves open later
Bayesian inference with HMC or NUTS.

A moving-boundary PDE is most naturally made differentiable by transforming the
physical interval \([-a(t),a(t)]\) onto a fixed spatial grid. The boundary
parameters then enter smoothly through the PDE coefficients. The native PDE
adjoint stores the forward density history, solves the transposed implicit
system, and analytically differentiates the Chang--Cooper coefficients and RT
overlap. Its independent Torch counterpart replays a compiled one-cell VJP.

In the prototype, CSI duration is expressed in seconds and is continuous. The
number of integration cells changes discretely at mesh boundaries, but the
duration and the final partial interval remain differentiable within a fixed
mesh. This deliberately differs from the legacy integer execution count.

The RT-bin likelihood is an exact integral of the solver's piecewise-constant
flux approximation. It is continuous but has a numerical derivative kink when
an interval endpoint lies exactly on a PDE time-cell boundary. The solver uses
a deterministic zero subgradient at that exact tie; otherwise autograd agrees
with centered finite differences for the tested NDT and boundary parameters.
Finite-difference validation should avoid exact cell-edge ties or explicitly
test one-sided derivatives.

Gradient information improves local optimization but does not establish that
the objective is unimodal. Multiple starts, parameter-recovery studies, and
posterior diagnostics would remain necessary.

## Genuine stochastic LCA dynamics

A scalar numeric LCA noise value in the current batched implementation is a
deterministic per-step term. The discussion here concerns genuine random process
noise in the two LCA accumulators.

With genuine LCA noise, there is no longer one deterministic drift trajectory.
Because LCA state persists between trials, simply averaging a separate drift
distribution for each trial is also incorrect. The observations provide
information about the latent LCA state, and that state distribution must be
updated and carried forward.

The full likelihood becomes a nonlinear state-space integral:

\[
p(y_{1:T}\mid\theta)
=
\int p(z_0)
\prod_t p(z_t\mid z_{t-1},\theta)
p(y_t\mid z_t,\theta)\,dz_{0:T},
\]

where \(z_t\) is the persistent two-dimensional LCA state and the observation
likelihood is the direct DDM first-passage density induced by that state.

### Deterministic density-filter formulation

The main non-sampling approach is to propagate a two-dimensional probability
density over the LCA state. For each trial:

1. **Predict:** propagate the previous filtered density through the noisy ITI
   and CSI dynamics,

   \[
   p_t^-(z)=\int K_t(z\mid z',\theta)p_{t-1}(z')\,dz'.
   \]

2. **Score:** evaluate the DDM likelihood of the observed choice and response
   time for each possible LCA state,

   \[
   \ell_t(z)=p(y_t\mid v(z),\theta).
   \]

3. **Update:** condition the state density on the observation,

   \[
   p_t(z)=\frac{\ell_t(z)p_t^-(z)}
   {\int \ell_t(z)p_t^-(z)\,dz}.
   \]

4. Add the log of the normalizing denominator to the total log likelihood and
   carry \(p_t(z)\) into the next trial.

This preserves cross-trial state statistically without sampling LCA paths. A
finite-volume Fokker--Planck method, semi-Lagrangian method, spectral method, or
discretized transfer operator could implement the prediction step. Because the
ITI and CSI contain a small number of recurring input regimes, it may be
possible to reuse or exponentiate their transition operators rather than
explicitly replaying every LCA step.

The approach is plausible because the stochastic state is only two-dimensional.
It would replace 100,000 sampled trajectories with a fixed two-dimensional
density grid and yield a deterministic, potentially differentiable likelihood.
It will not scale gracefully to models with many noisy state dimensions.

A one-dimensional approximation based on the difference between the two LCA
accumulators may be possible, but it is not exact: the sum and difference modes
remain coupled by the logistic nonlinearity.

### Other stochastic-LCA options

- Gaussian, Gaussian-sum, or unscented filtering would be cheaper but may fail
  if competitive LCA dynamics produce a skewed or multimodal state density.
- A particle filter could retain the direct DDM likelihood and sample only LCA
  paths. This is a useful Rao--Blackwellized method, but it retains a stochastic
  likelihood and may not work with ordinary HMC.
- Treating all LCA noise innovations as latent variables permits a
  reparameterized HMC formulation in principle, but creates a very
  high-dimensional posterior when noise is injected at every integration step.
- Because the LCA continues changing during the decision process, a genuine
  noisy-LCA extension would require a joint LCA--DDM density during decisions;
  its numerical state would have two LCA dimensions plus the DDM dimension.

There is unlikely to be a closed-form likelihood for the complete nonlinear,
noisy, history-dependent CSI model. The two-dimensional sequential density
filter is the most credible deterministic numerical alternative to full
simulation.

## Suggested validation sequence

Implemented checks and remaining scientific validation are:

1. The fixed-boundary PDE flux agrees with the analytic Wiener density and
   conserves probability mass.
2. The Torch Euler mirror agrees with the actual PNL LCA recurrence to floating
   point precision. PNL endpoints converge linearly toward the RK4 continuous
   endpoint as its time step is refined.
3. The moving-boundary solver reports mass error, negative-density, and invalid
   boundary diagnostics for every parameter evaluation.
4. Autograd through the sequential LCA and PDE is exercised by unit tests;
   threshold and off-cell-edge NDT derivatives agree with centered finite
   differences. Selected gradients should additionally be checked over a broad
   parameter grid, including gain, CSI, collapse, and all condition indices.
5. The CLI exposes both multi-start L-BFGS-B and CMA-ES so their fitted optima
   can be compared.
6. Fitted-point grid convergence and one seeded 13-parameter recovery are now
   exercised. Three high-resolution original-PNL paired simulation comparisons
   and a numerical-semantic ladder are now exercised. Broader repeated recovery,
   explicit marginal-history comparison, and simulation-based calibration
   remain required before scientific use.
7. A genuine noisy-LCA model remains a separate future phase requiring the
   two-dimensional sequential filter described above (or a three-dimensional
   joint LCA--DDM density during decisions).

The critical feasibility test is whether a converged two-dimensional filter is
both faster and more accurate than the GPU Monte Carlo likelihood over the
parameter ranges used by CSI.

## Direct versus legacy disagreement diagnostic

`semantic-ladder` scores the same direct-fit and PNL-fit vectors while changing
one convention at a time:

1. round the direct CSI to a 10 ms scheduler count;
2. replace the 1 ms recording interval with the legacy estimator's 100 fixed
   empirical-range RT bins;
3. reset LCA state each trial as a history-sensitivity ablation;
4. replace RK4 with the exact PNL Euler LCA recurrence; and
5. use a 10 ms temporal mesh; and
6. replace continuous first-passage flux with a deterministic Gaussian Euler
   random walk that absorbs only at 10 ms step endpoints.

The 10 ms PDE diagnostic uses fully implicit time stepping. Crank--Nicolson on
that unusually coarse grid produced a negative early-time flux and one zero
likelihood; fully implicit stepping prevents that numerical ringing. This rung
is still a continuous diffusion PDE, not the legacy discrete endpoint-crossing
random walk.

For the three pilot subjects, positive numbers below mean the direct objective
prefers the direct-fit vector. The last column is the independently paired PNL
simulator ranking (10,000 simulations at each of three seeds).

| Subject | Continuous, exact CSI | 10 ms continuous PDE | Endpoint DDM | Endpoint + reset ablation | PNL simulator |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | +36.87 | +18.11 | +2.73 | -2.12 | -4.64 |
| 4 | +21.36 | +11.83 | -17.73 | -18.79 | -23.41 |
| 7 | +12.46 | +10.44 | +1.90 | +1.25 | +6.47 |

CSI rounding changed the ranking by 0.00, 0.35, and 0.00 log units. Euler
versus RK4 changed it by only 0.08--0.13 on the fine grid and 0.87--1.71 on the
10 ms grid. Resetting history changed it by 0.23--4.04. The broad RT bins moved
subject 1 by 5.76 but the other two by less than one. The temporal mesh was the
largest continuous-PDE numerical factor, moving the ranking by 1.98--14.71
after matching RT bins, but no continuous rung reversed the direct preference.

An independent first-passage generator check makes the remaining leading
explanation concrete. At a 10 ms step, detecting threshold crossings only at
Euler endpoints shifted cumulative first-passage probabilities by as much as
2.68 percentage points for a fixed boundary and 2.08 points for a collapsing
boundary relative to the continuous PDE. A Brownian-bridge crossing correction
reduced the corresponding maximum errors below 0.18 points in the same
50,000-path check. The legacy PNL DDM uses endpoint crossing, so its stochastic
process is detectably different from the continuous diffusion targeted here.

The deterministic endpoint solver performs that decisive comparison. It uses a
fixed evidence grid, applies each Gaussian Euler transition with an FFT heat
kernel, and removes mass beyond the collapsing boundary only after a 10 ms
step. A one-step test matches the exact Gaussian tail probabilities. In a
200,000-path, 100-step fixed-boundary check, the finest deterministic upper and
lower crossing probabilities were 0.29632 and 0.13929, versus Monte Carlo
values 0.29529 and 0.13933. The upper difference is approximately one Monte
Carlo standard error.

The result shows that DDM process semantics account for most of the ranking
disagreement. Switching from the stable 10 ms continuous PDE to endpoint
crossing moved the three ranking gaps by -15.35, -29.52, and -8.52 log units.
Subject 4 then agreed with the PNL preference, subject 7 retained the same
preference, and subject 1 was only +2.73 rather than +18.11. Evidence-grid
refinement from 511 to 1023 points changed those gaps by just 0.031, 0.036, and
0.024; finest-grid mass errors were below `4e-15`.

After dividing bin mass by the same RT-bin width as the PNL histogram, the
finest endpoint scores were `(269.09, 266.36)`, `(284.74, 302.47)`, and
`(-245.55, -247.45)` for the direct and PNL vectors. The corresponding paired
PNL simulator means were `(261.65, 266.29)`, `(276.88, 300.29)`, and
`(-248.07, -254.54)`. Thus even the absolute objective scales are now close,
although they are not expected to be identical because one is deterministic
and history-conditioned while the other is a finite-simulation composite
likelihood.

Resetting LCA state at every trial is not an implementation of the PNL
marginal-history objective, but it gives a useful sensitivity bound. With the
endpoint DDM it changed the gaps to -2.12, -18.79, and +1.25, matching the PNL
preference sign for all three subjects. The remaining exact experiment is
therefore explicit averaging over simulated prior RT histories. The continuous
model remains the mathematically preferred target; the endpoint solver is a
compatibility diagnostic and currently has no fitting adjoint.
