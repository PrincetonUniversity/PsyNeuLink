# Hierarchical PEC Plan After LLVM `log_likelihood` Fix

## Summary
Build hierarchical parameter estimation on top of the now-working LLVM-backed `ParameterEstimationComposition.log_likelihood()`. The first hierarchical version will use empirical-Bayes Laplace EM with complete simulation-backed likelihoods, not neural likelihood approximators and not Python execution mode.

The model is:

- each subject has latent parameter vector `theta_s`;
- `theta_s` is represented in an unconstrained transformed space `z_s`;
- group means are modeled as `mu_s = X_s beta`, where `X_s` is an intercept plus optional subject-level group covariates;
- group random-effect covariance is `covariance="diagonal"` in v1, meaning independent group variance per expanded fit parameter.

## Public API Changes
- Add prior/MAP foundation:
  - `PECParameterPrior`, initially supporting normal priors in transformed parameter space.
  - `pec.log_posterior(*params, inputs=..., priors=...)`.
  - `pec.fit_map(inputs=..., priors=..., optimization_function=...)`.
- Add hierarchical fitting:
  ```python
  result = pec.fit_hierarchical(
      inputs=inputs,
      subject_id="subject",
      group_covariates=["age", "group"],
      covariance="diagonal",
      max_em_iterations=25,
      em_tol=1e-3,
      damping=0.5,
      variance_floor=1e-4,
      hessian_step=1e-3,
      common_random_numbers=True,
  )
  ```
- Add `HierarchicalPECResults` with:
  - `group_parameters`
  - `subject_parameters`
  - `subject_posteriors`
  - `covariance`
  - `em_history`
  - `fit_param_names`
  - `transform_metadata`

## Key Implementation Changes
- Reuse the fixed LLVM `log_likelihood` path; hierarchical fitting must raise `ParameterEstimationCompositionError` if `comp_execution_mode` is not `"LLVM"`.
- Add a simulation-data likelihood helper so hierarchical code can compute likelihoods against subject-specific data slices without mutating global PEC data.
- Split data into subject-level views:
  - outcome data per subject,
  - input trials per subject,
  - optional `depends_on` masks per subject,
  - one subject-level covariate row per subject.
- Preserve existing `depends_on` semantics:
  - trial-level conditional parameters still expand into names like `DDM.threshold[a]`;
  - each expanded parameter becomes one dimension of the subject random-effect vector.
- Use automatic parameter transforms:
  - bounded parameters use a logit transform from bounds to unconstrained space;
  - reported estimates are converted back to natural parameter units.
- Implement Laplace EM:
  - E-step: for each subject, maximize `log_likelihood_s(theta_s) + log N(z_s | X_s beta, Sigma)`.
  - Hessian: finite-difference diagonal Hessian of the negative log posterior in transformed space, using common random numbers by default.
  - Posterior approximation: `q_s(z) ≈ N(z_hat_s, V_s)`.
  - M-step: update `beta` by least squares on posterior means and update diagonal variances from posterior residual moments.
  - Apply variance floors and damping every M-step to prevent variance collapse and noisy EM oscillation.
- Add an internal subject-task abstraction so the E-step can later be distributed; v1 runner is local serial/process-safe, with no new distributed dependency.

## Tests And Smoke Scripts
- Add LLVM-only unit tests for:
  - independent MAP combines prior and likelihood;
  - hierarchical two-subject DDM returns finite group and subject estimates;
  - group covariates build the expected design matrix and reject within-subject-varying covariates;
  - `depends_on` expanded parameters work inside hierarchical fitting;
  - input dictionaries are not mutated;
  - Python mode raises a clear error;
  - variance floors prevent zero group variance;
  - finite-difference Hessian fallback handles bad curvature.
- Add `Scripts/Debug/pec_hierarchical/hierarchical_em_smoke.py`:
  - small synthetic DDM dataset with subject IDs;
  - one scalar random effect;
  - one group covariate example;
  - prints EM history, group variance, subject MAPs, and posterior SDs.
- Keep the existing `log_likelihood_smoke.py` as the lower-level likelihood diagnostic.

## Reference Points
- Julia EM implementation: https://github.com/ndawlab/em
  - README describes the target family: hierarchical decision-making model fits using EM, with group-level distributions, optional group covariates, and per-subject parameters.
  - `example.jl` is the closest API reference: `X` is a subject-level design matrix, `betas` has shape predictors x parameters, `sigma` is a group variance/covariance, and subject parameters follow `x_ij ~ Normal(X beta_j, Sigma)`.
  - `src/emcore.jl` is the algorithm reference: E-step optimizes each subject with a Gaussian group prior, computes per-subject inverse Hessians, and M-step updates `betas` and `sigma`; it supports diagonal vs full covariance and uses threaded per-subject E-steps.
  - PEC adaptation: keep the design-matrix and EM result shape, but replace the analytic likelihood callback with LLVM-backed simulated `log_likelihood`; use finite-difference Hessians because the simulated likelihood is not analytically differentiable.
- HDDM classic hierarchical model: https://hddm.readthedocs.io/en/latest/
  - Use as the cognitive-modeling API reference for subject/group hierarchical DDM estimates and the user-facing vocabulary around subject parameters, group distributions, posterior predictives, and convergence diagnostics.
  - HDDM's `depends_on` interface supports condition-specific parameters; PEC should preserve its existing `depends_on` semantics while adding a separate subject/group hierarchy above the expanded parameters.
  - HDDMRegression/Patsy-style formulas are useful prior art for covariates, but PEC v1 should use explicit `group_covariates=[...]` rather than adding a formula parser.
- HDDM LAN extension: https://hddm.readthedocs.io/en/latest/lan_tutorial.html and https://hddm.readthedocs.io/en/latest/lan_to_hddm_end_to_end.html
  - Use as contrast, not v1 implementation: HDDMnn/LAN trains neural likelihood approximators from simulator-generated data, while PEC v1 should keep a complete simulation-backed likelihood evaluated directly through LLVM.
  - Revisit LAN/MNLE only after the EM/MAP path is stable and performance bottlenecks are measured.

## Assumptions And Defaults
- First implementation is empirical Bayes, not full Bayesian sampling.
- First covariance structure is only `"diagonal"`; `"full"` should raise a clear not-implemented error.
- Group covariates are subject-level, not trial-level; they must be constant within subject.
- Continuous group covariates are centered/scaled internally; categorical covariates use treatment coding with stable category order.
- All likelihood evaluations use LLVM simulation. Python mode is not supported for MAP or hierarchical fitting.
- Work continues on branch `feat/pec-hierachical`, with one-off development scripts under `Scripts/Debug/pec_hierarchical/`.
