# Direct-likelihood approaches for the CSI model

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

These ideas are exploratory and are being tabled while the existing GPU kernel
is profiled and optimized.

## Deterministic LCA and a direct DDM likelihood

With the current CSI configuration, the LCA trajectory is deterministic for a
fixed parameter vector and observed task sequence. It carries state between
trials, but the DDM does not feed back into that state. The LCA can therefore be
executed once to obtain a trial-specific drift rate, rather than once for every
Monte Carlo estimate.

The joint log likelihood then factorizes conditionally on that deterministic
trajectory:

\[
\log L(\theta)
=
\sum_i \log f_{c_i}\left(
RT_i-t_{0,i}\mid v_i(\theta),a_i(t;\theta),\sigma
\right),
\]

where \(v_i(\theta)\) is the LCA-derived drift for trial \(i\), \(a_i(t)\)
is its decision boundary, \(t_{0,i}\) is nondecision time, and \(f_{c_i}\)
is the first-passage density through the observed boundary.

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
diffusion. We would need to choose explicitly between:

1. treating the scientifically intended model as a continuous diffusion; or
2. implementing a discrete transition-density solver that more closely
   reproduces the existing update.

For response times recorded at finite precision, the likelihood should ideally
integrate density over the measurement interval rather than evaluate only the
density at a point.

## Differentiation and gradient-based inference

Navarro--Fuss series terms and fixed-grid numerical PDE operations can be
implemented with differentiable array operations. Automatic differentiation
could then propagate through the deterministic LCA trajectory and the
first-passage likelihood. This would support constrained gradient optimizers
such as L-BFGS and Bayesian inference with HMC or NUTS.

A moving-boundary PDE is most naturally made differentiable by transforming the
physical interval \([-a(t),a(t)]\) onto a fixed spatial grid. The boundary
parameters then enter smoothly through the PDE coefficients. Reverse-mode
differentiation through all time steps is the simplest prototype; checkpointing,
an adjoint, or a custom vector-Jacobian product could reduce its memory cost.

The current surrogate CSI duration is effectively an integer execution count.
It is therefore discrete and does not have a useful ordinary gradient. It could
be enumerated or updated with a discrete sampler while continuous parameters use
gradients. Alternatively, a continuous-time LCA formulation could make CSI
duration differentiable, but that would be a change to the mathematical model.

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
- If the LCA continued changing during the decision process, rather than
  providing a fixed trial drift, a joint LCA--DDM density would be required and
  the numerical state dimension would increase further.

There is unlikely to be a closed-form likelihood for the complete nonlinear,
noisy, history-dependent CSI model. The two-dimensional sequential density
filter is the most credible deterministic numerical alternative to full
simulation.

## Suggested validation sequence

If this direction is revisited:

1. Implement and test a differentiable fixed-boundary Wiener likelihood.
2. Compare it against high-resolution DDM simulations over a parameter grid.
3. Fit the deterministic, fixed-boundary CSI model with both CMA-ES and L-BFGS.
4. Implement and grid-refine a moving-boundary first-passage solver.
5. Compare its choice probabilities, response-time densities, and gradients
   against high-resolution simulation and finite differences.
6. Implement a standalone two-dimensional noisy-LCA density propagator and
   compare its transition densities and moments against large Monte Carlo runs.
7. Add sequential DDM observation updates and verify likelihood convergence as
   the LCA grid and time discretization are refined.
8. Perform parameter recovery and simulation-based calibration before using the
   solver for hierarchical HMC.

The critical feasibility test is whether a converged two-dimensional filter is
both faster and more accurate than the GPU Monte Carlo likelihood over the
parameter ranges used by CSI.
