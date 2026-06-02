# Run-Centric Hierarchical PEC API Plan

## Summary
Keep `ParameterEstimationComposition.run(inputs=...)` as the only fitting entrypoint. MAP and hierarchical fitting are configured on the PEC at construction time, then executed through `run`, preserving the current PEC model as a `Composition` that performs fitting or optimization.

Build on the now-working LLVM-backed `ParameterEstimationComposition.log_likelihood()` path. The first hierarchical version uses empirical-Bayes Laplace EM with complete simulation-backed likelihoods, not neural likelihood approximators and not Python execution mode.

The hierarchical model is:

- each subject has latent parameter vector `theta_s`;
- `theta_s` is represented in unconstrained transformed space `z_s`;
- group means are modeled as `mu_s = X_s beta`, where `X_s` is an intercept plus optional subject-level group covariates;
- group random-effect covariance is `covariance="diagonal"` in v1, meaning independent group variance per expanded fit parameter.

## Public API Changes
- Add constructor configuration:
  ```python
  pec = pnl.ParameterEstimationComposition(
      ...,
      data=data_to_fit,
      outcome_columns=["decision", "response_time"],
      fit_method="hierarchical",
      subject_id="subject",
      group_covariates=["age", "group"],
      priors=priors,
      covariance="diagonal",
      max_em_iterations=25,
      em_tol=1e-3,
      damping=0.5,
      variance_floor=1e-4,
      hessian_step=1e-3,
      common_random_numbers=True,
      optimization_function=pnl.PECOptimizationFunction(...),
  )

  pec.run(inputs=inputs)
  result = pec.fit_results
  ```
- `fit_method=None` preserves current behavior.
- `fit_method="map"` requires `data`; `priors` are optional.
- `fit_method="hierarchical"` requires DataFrame `data`, `outcome_columns`, and `subject_id`.
- Do not add public `fit_map` or `fit_hierarchical` methods in v1.
- Add `PECParameterPrior`, initially supporting normal priors in transformed parameter space.
- Priors are keyed by expanded fit parameter names, e.g. `DDM.threshold` and `DDM.threshold[a]`.
- Keep scalar diagnostics:
  ```python
  pec.log_likelihood(*params, inputs=inputs)
  pec.log_posterior(*params, inputs=inputs, priors=None)
  pec.log_likelihood_subject(*params, subject="S01", inputs=inputs, subject_id=None)
  ```

## Input And Data Model
- Hierarchical data is one stacked trial table across subjects.
- `pec.run(inputs=...)` receives one stacked input sequence aligned row-by-row with `data`.
- Subjects may have different input values and different trial counts.
- All subjects share the same model and input schema in v1.
- Internally, subject tasks slice `data`, `inputs`, `depends_on` masks, and likelihood masks using the same subject mask.
- Group covariates are subject-level, not trial-level; they must be constant within each subject.
- Continuous group covariates are centered/scaled internally.
- Categorical group covariates use treatment coding with stable category order.

## Key Behavior
- `run()` return shape stays compatible with current PEC behavior; rich fit outputs live on `pec.fit_results`.
- MAP updates `optimized_parameter_values` with natural MAP estimates and `optimal_value` with log posterior.
- Hierarchical updates `optimized_parameter_values` with population intercept/group-mean natural estimates and `optimal_value` with final EM objective.
- `fit_results` is a DataFrame-based result object for MAP or hierarchical runs.
- `MAPPECResults` should include natural and transformed parameter estimates plus posterior/objective metadata.
- `HierarchicalPECResults` should include:
  - `group_parameters`
  - `subject_parameters`
  - `subject_posteriors`
  - `covariance`
  - `em_history`
  - `fit_param_names`
  - `transform_metadata`
  - covariate encoding metadata

## Implementation Changes
- Store both original DataFrame metadata and processed likelihood data.
- `outcome_columns` selects likelihood columns; extra metadata columns require `outcome_columns`.
- Use expanded `fit_param_names` for priors, outputs, and hierarchical dimensions.
- Transform each expanded parameter using PEC search bounds as hard support:
  ```text
  z = logit((theta - lower) / (upper - lower))
  theta = lower + sigmoid(z) * (upper - lower)
  ```
- Put priors and hierarchical Gaussian random effects on transformed `z`; evaluate likelihoods in natural parameter units.
- Reuse `PECOptimizationFunction` for MAP and hierarchical subject E-steps.
- Reuse the fixed LLVM `log_likelihood` path; MAP and hierarchical fitting must raise `ParameterEstimationCompositionError` if `comp_execution_mode` is not `"LLVM"`.
- Add a simulation-data likelihood helper so hierarchical code can compute likelihoods against explicit subject-specific data slices without mutating global PEC data.
- Split data into subject-level views:
  - outcome data per subject;
  - input trials per subject;
  - optional `depends_on` masks per subject;
  - likelihood include masks per subject;
  - one group-covariate row per subject.
- Preserve existing `depends_on` semantics:
  - trial-level conditional parameters still expand into names like `DDM.threshold[a]`;
  - each expanded parameter becomes one dimension of the subject random-effect vector.
- Implement hierarchical EM with diagonal covariance only; `covariance="full"` raises not implemented.
- E-step: for each subject, maximize `log_likelihood_s(theta_s) + log N(z_s | X_s beta, Sigma)`.
- Hessian: compute finite-difference diagonal Hessian of the negative log posterior in transformed space, using common random numbers by default.
- Posterior approximation: `q_s(z) ~= N(z_hat_s, V_s)`.
- M-step: update `beta` by least squares on posterior means and update diagonal variances from posterior residual moments.
- Apply variance floors and damping every M-step to prevent variance collapse and noisy EM oscillation.
- Add an internal subject-task abstraction so the E-step can later be distributed; v1 runner is local serial/process-safe, with no new distributed dependency.

## Test Plan
- Existing PEC tests pass unchanged.
- Constructor tests verify `outcome_columns`, metadata handling, and current API preservation.
- MAP tests verify posterior composition, expanded-name priors, state updates, and LLVM-only behavior.
- Subject likelihood tests verify aligned slicing and no mutation.
- Hierarchical tests cover two-subject DDM, subject-specific inputs, group covariates, `depends_on`, variance floors, Hessian fallback, and full-covariance rejection.
- Add `Scripts/Debug/pec_hierarchical/hierarchical_em_smoke.py`:
  - small synthetic DDM dataset with subject IDs;
  - one scalar random effect;
  - one group covariate example;
  - prints EM history, group variance, subject MAPs, and posterior SDs.
- Keep `Scripts/Debug/pec_hierarchical/log_likelihood_smoke.py` as the lower-level likelihood diagnostic.

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

## Assumptions
- All MAP and hierarchical likelihood evaluations require LLVM mode.
- Search ranges are hard support for transforms.
- v1 is empirical Bayes with diagonal covariance.
- Hierarchical metadata lives in PEC `data`, enabled by explicit `outcome_columns`.
- Work continues on branch `feat/pec-hierachical`, with one-off development scripts under `Scripts/Debug/pec_hierarchical/`.
