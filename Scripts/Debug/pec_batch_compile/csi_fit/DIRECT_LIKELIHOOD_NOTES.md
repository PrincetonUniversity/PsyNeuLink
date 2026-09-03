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
4. Autograd through the sequential LCA and PDE is exercised by unit tests. At
   an off-grid six-trial point spanning every condition, all 13 fitted-parameter
   derivatives agree with centered finite differences and remain nonzero. The
   compact Torch and fused native adjoints then agree with the ordinary-autograd
   oracle. Additional random points and mesh resolutions would still strengthen
   coverage before gradient-based posterior sampling.
5. The CLI exposes both multi-start L-BFGS-B and CMA-ES so their fitted optima
   can be compared.
6. Fitted-point grid convergence and one seeded 13-parameter recovery are now
   exercised. Three high-resolution original-PNL paired simulation comparisons
   and a numerical-semantic ladder are now exercised. A conditioned simulator
   likelihood is now implemented for the explicit marginal-history comparison;
   broader repeated recovery and simulation-based calibration remain required
   before scientific use.
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
preference sign for all three subjects.

## Conditioned simulator-likelihood prototype

The PEC simulator paths now have an opt-in CSI correction through
`PECOptimizationFunction(conditioned_likelihood=True)`. The CSI fitting script
exposes it as `--condition-observed-history` and enables it by default. Both
the co-evolving batched compiler and threaded LLVM PEC follow the same
sequential Monte Carlo recursion:

1. begin a trial with particles representing persistent composition state
   conditioned on the earlier observed trials;
2. run the complete coupled CSI scheduler for one trial, so the LCA still
   evolves on every scheduler pass while the DDM is active;
3. compute each simulated terminal state's contribution to the observed
   choice/RT histogram cell (including optional Gaussian bin smoothing);
4. add the log of the mean observation density;
5. resample the complete terminal state using those contributions; and
6. launch the next observed trial from the resampled states.

The batched kernel can now accept and return retained state with shape
`[parameter, subject, estimate, state]`. Split launches preserve the original
full-sequence Philox trial/lane indexing; a stochastic two-trial CSI test gives
bit-identical outcomes when an unsplit launch is replaced by two state-carrying
launches without resampling. LLVM's evaluation function now has a retained-state
variant that gives each estimate its own composition state and composition data,
writes both back after one trial, and resumes the resampled structures on the
next call. Resampling copies ancestral model state while retaining each
destination lane's advanced LLVM RNG state, so duplicate offspring do not clone
or restart their future random streams. Tests execute the real CSI graph through
both implementations and verify deterministic replay of the conditioned score.

This corrects the particular statistical bug under the histogram observation
model: simulated prior histories are no longer left unconditional when the
next trial is scored. It does not change the within-trial stochastic process.
The legacy simulator still uses 10 ms Euler endpoint crossing, whereas the
scientific direct likelihood still targets continuous first passage.

Current prototype limits are deliberate. A positive histogram pseudocount has
no particle ancestry and is rejected; the batched API currently accepts one
contiguous subject sequence per call; and LLVM state transport is CPU-threaded,
not PTX. A general PEC solution should define an observation-kernel/resampler
interface, missing-observation behavior, subject batching, effective-parameter
state, and the ownership of every retained state component before exposing this
behavior for arbitrary compositions.

The GPU prototype now treats the data-anchored RT histogram as genuinely
finite: simulated RTs below its first edge or above its last edge contribute no
mass, rather than being folded into an edge bin and divided by that bin's finite
width. On the subject-1 14-observation diagnostic, five 10,000-particle scores
had mean 5.8405 (SD 0.2825), versus 5.9550 from the deterministic 10 ms endpoint
solver. The difference is 0.41 Monte Carlo standard deviations. This comparison
uses the endpoint solver as a compatibility oracle; it does not change the
continuous first-passage scientific target.

GPU seed, trial offset, sequence length, and LCA step cap are runtime scalars
with Triton's automatic integer-value and alignment specialization disabled.
Numerical and truncation reductions are queued across the sequential particle
filter and synchronized once at the end; resampling and state transport remain
on the GPU. On the local RTX 2080 Ti benchmark, a 561-trial subject with 10,000
particles fell from 21.25 to 11.63 seconds of scoring time (1.83x), while a warm
14-observation score fell from roughly 0.34 to 0.25--0.27 seconds. Different
seeds and sequence lengths reuse the same compiled kernel. A warm full-subject
11-candidate CMA-ES population took 17.23 seconds versus 10.85 seconds for one
candidate, giving about 6.9x the candidate throughput of sequential evaluation.
No LLVM changes were made in this GPU accuracy/performance pass.

## Deterministic CSI observed-history specialization

CSI admits a cheaper corrected likelihood when LCA noise is exactly zero. For
fixed parameters and an observed RT endpoint, its persistent LCA state is then
deterministic: the LCA follows the participant's observed duration, while the
DDM simulations are needed only to estimate the current trial's choice/RT
density. There is no reason to copy that identical LCA trajectory into 10,000
particles or to resample it.

The CUDA specialization implements two stages:

1. one sequential LCA lane per parameter candidate follows the observed trial
   history and records the within-trial drift path presented to the DDM; and
2. one parallel launch simulates every candidate/trial/DDM estimate from those
   cached drift paths and evaluates the existing histogram likelihood.

The history pass preserves the scheduler's overlap exactly: the first DDM
update uses the same pass that makes a positive LCA execution count finished.
In particular, with a positive ITI and zero CSI, that DDM update precedes task
onset by one pass, matching the historical model rather than imposing a cleaner
phase boundary.

The observed decision duration is mapped to the legacy endpoint process as
`ceil(max(RT - nondecision - CSI, 0) / 0.01)` scheduler steps. Thus this mode
conditions on the recorded RT endpoint. It is not the broader interpretation
in which the whole observed histogram bin represents latent endpoint-state
uncertainty; use the particle filter for that interpretation or whenever the
persistent process itself is stochastic. The specialization rejects nonzero
LCA noise and any graph outside the authenticated CSI structure rather than
silently changing semantics.

The fitting script selects this path with
`--backend triton --deterministic-observed-history`. The default corrected mode
remains the particle filter so that choosing endpoint conditioning is explicit.
On the local RTX 2080 Ti, a warm 561-trial, 10,000-estimate, 11-candidate
representative batch took 1.24 seconds. The same workload previously took
17.23 seconds with particle filtering and 0.756 seconds with the old
unconditional-history likelihood. This provisional result recovers most of the
old throughput while retaining the corrected observed history.

The deterministic specialization now has a bounded-memory likelihood path. A
fit only consumes the number of simulations that match each trial's observed
choice and RT bin, so the DDM kernel reduces one estimate tile directly into
that count instead of materializing every simulated choice and RT. It launches
only likelihood-included trials, stops a tile once every trajectory has crossed,
and stops an uncrossed trajectory once its monotonically increasing RT has
passed the last bin that can contribute to the observation. The latter is exact
observation-window censoring, not an approximation. Strict truncation checks
disable it and retain the full `max_steps` execution. Gaussian histogram
smoothing through radius eight (including the fitting values 0.5 and 1.0) is
also accumulated in the kernel; raw outcomes remain available through the
debug path, and unusually wide smoothing kernels use the general materialized
fallback.

This removes the poor 100,000-estimate scaling. Before fusion, a warm
11-candidate by 100,000-estimate subject-1 batch took 40.60 seconds and reported
21.87 GiB of peak CUDA allocation on the 11 GiB RTX 2080 Ti. After fusion, five
runs with a separate compilation warmup gave a 0.510-second warm median without
smoothing and a 0.576-second median with smoothing sigma 0.5 and pseudocount
0.1. Peak allocation was 28.8 MiB. The unsmoothed score changed by only
`3.1e-5` log units, attributable to floating-point reduction order, and focused
tests compare fused and materialized results at smoothing 0, 0.5, and 1.0.

For a same-machine throughput reference, the primary continuous direct
likelihood (1 ms DDM mesh, 65 spatial points, native CPU paths) scored the same
561-row subject in a 0.0688-second warm median over five runs. Eleven sequential
direct scores would therefore take about 0.756 seconds, versus 0.510 seconds for
the 11-candidate, 100,000-estimate GPU population. The GPU sampling population
has about 1.48x the candidate throughput in this comparison, although one GPU
candidate still has higher latency and finite Monte Carlo error. The direct
likelihood remains the scientific standard: it is deterministic, continuous
first passage, and differentiable; these timings only compare the computational
cost of evaluating a population with the legacy 10 ms sampling process.

### Local likelihood-surface comparison after GPU fusion

`csi_likelihood_surface_comparison.py` evaluates one reproducible candidate
surface under the primary continuous likelihood, a continuous likelihood with
the GPU histogram's fixed RT bins, and the optimized GPU endpoint sampler. It
compares rankings rather than raw likelihood values because the direct and GPU
methods implement different stochastic processes and likelihood conventions.

The subject-1 pilot used 121 candidates centered on the direct-fit vector,
100,000 simulations per candidate, and five GPU seeds. Without smoothing, the
Spearman correlation between the mean GPU score and the primary direct score
was 0.113; matching the direct solver to 100 fixed RT bins increased it only to
0.177. The direct-fit vector ranked 46th under the GPU mean, the GPU-preferred
candidate ranked 101st under the direct likelihood, and their top-ten sets did
not overlap. In contrast, the median rank correlation between independent GPU
seeds was 0.993, showing that this disagreement was not dominated by sampling
noise.

With smoothing sigma 0.5 and pseudocount 0.1, GPU seed-to-seed correlations
were 0.993--0.998 and the maximum across-seed score SD fell below 0.95 log
units. GPU-versus-direct rank correlations remained 0.026 for the primary
continuous likelihood and 0.093 for the bin-matched continuous likelihood.
The same GPU candidate remained preferred. A deterministic 10 ms endpoint
spot check also preferred the leading GPU candidates over the direct-fit
vector. Together these diagnostics attribute the local ranking disagreement
primarily to endpoint-process semantics, not to the fused likelihood reduction,
Monte Carlo instability, or RT-bin width.

This pilot does not require cluster hardware. On the RTX 2080 Ti, all eleven
100,000-simulation batches for one seed took about 5.8 seconds unsmoothed and
6.8--7.2 seconds with smoothing after compilation; all 121 direct scores took
about ten seconds per direct variant. GB300/Grace resources become useful for
a multi-subject replication or substantially larger surfaces, where subjects,
GPU seeds, and candidate batches can be distributed independently.

### One-millisecond GPU refinement

The GPU comparison drivers can also refine the sampled endpoint process while
holding physical model parameters fixed. A valid 1 ms refinement changes both
the LCA and DDM steps, converts the one-second ITI from 100 to 1,000 executions,
converts the fitted 11-step CSI to 110 executions, divides each per-step
boundary-collapse increment by ten, and raises the 12-second execution cap from
1,200 to 12,000 steps. Changing only the DDM step would make the lockstep LCA
evolve ten times too quickly relative to response time.

For subject 1, 100,000-sample RT-density overlays at the direct-fit vector show
the refined endpoint distribution nearly superimposed on the continuous
direct curve. In common 10 ms display bins, changing the sampled process from
10 ms to 1 ms reduced integrated absolute density error by 57% for
NoInstruction, 56% for RealRare, and 66% for RealFrequent on the three
representative trials. These are visualization bins only; the 1 ms simulation
still performs endpoint checks every millisecond.

The full 121-candidate, five-seed, 100,000-estimate surface check showed the
same convergence. GPU-versus-primary-direct Spearman correlation rose from
0.113 at 10 ms to 0.838 at 1 ms; correlation with the bin-matched continuous
likelihood rose from 0.177 to 0.878. The direct-fit vector improved from GPU
rank 46 to rank 12. The highest GPU candidates were separated from it by less
than 0.5 log units while their across-seed SDs were roughly 0.5--0.8, so this
pilot cannot reliably distinguish the local optimum within that near-tied set.

Temporal refinement preserves bounded memory but costs arithmetic. One
121-candidate GPU seed took 56--58 seconds at 1 ms versus about 5.8 seconds at
10 ms on the RTX 2080 Ti, almost exactly the expected tenfold increase. The
GB300 system is therefore unnecessary for an individual diagnostic but useful
for additional seeds, multiple subjects, or a full 1 ms fitting study.
