# Run-Centric Hierarchical PEC API Plan

## Summary
Keep `ParameterEstimationComposition.run(inputs=...)` as the only fitting entrypoint. MAP and hierarchical fitting are configured on the PEC at construction time, then executed through `run`, preserving the current PEC model as a `Composition` that performs fitting or optimization.

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
      optimization_function=pnl.PECOptimizationFunction(...),
  )

  pec.run(inputs=inputs)
  result = pec.fit_results
  ```
- `fit_method=None` preserves current behavior.
- `fit_method="map"` requires `data`; `priors` are optional.
- `fit_method="hierarchical"` requires DataFrame `data`, `outcome_columns`, and `subject_id`.
- Do not add public `fit_map` or `fit_hierarchical` methods in v1.
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

## Key Behavior
- `run()` return shape stays compatible with current PEC behavior; rich fit outputs live on `pec.fit_results`.
- MAP updates `optimized_parameter_values` with natural MAP estimates and `optimal_value` with log posterior.
- Hierarchical updates `optimized_parameter_values` with population intercept/group-mean natural estimates and `optimal_value` with final EM objective.
- `fit_results` is a DataFrame-based result object for MAP or hierarchical runs.

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
- Implement hierarchical EM with diagonal covariance only; `covariance="full"` raises not implemented.

## Test Plan
- Existing PEC tests pass unchanged.
- Constructor tests verify `outcome_columns`, metadata handling, and current API preservation.
- MAP tests verify posterior composition, expanded-name priors, state updates, and LLVM-only behavior.
- Subject likelihood tests verify aligned slicing and no mutation.
- Hierarchical tests cover two-subject DDM, subject-specific inputs, group covariates, `depends_on`, variance floors, Hessian fallback, and full-covariance rejection.
- Add `Scripts/Debug/pec_hierarchical/hierarchical_em_smoke.py`.

## Assumptions
- All MAP and hierarchical likelihood evaluations require LLVM mode.
- Search ranges are hard support for transforms.
- v1 is empirical Bayes with diagonal covariance.
- Hierarchical metadata lives in PEC `data`, enabled by explicit `outcome_columns`.
