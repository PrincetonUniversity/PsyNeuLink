# Continuous CSI direct likelihood

This research-local package evaluates the complete deterministic-LCA CSI model
as a continuous-time sequential likelihood. It does not modify PsyNeuLink's
public parameter-estimation implementation.

From the repository root, run the numerical and PNL refinement checks with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py validate
```

Score the default 13-parameter vector for an actual `subject_nr` with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    score --subject 1
```

Use `--parameters fit.json` to score either a JSON `parameter_vector` or a
mapping keyed by the names printed by `score`. Per-trial probabilities and
diagnostics can be written with `--trial-output` and `--output`.

Fit with bounded multi-start L-BFGS-B, Optuna CMA-ES, or both:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    fit --subject 1 --optimizer both --output direct_fit_1.json
```

L-BFGS-B uses exact custom adjoints for the moving-boundary PDE and both LCA
integration paths by default. It optimizes all parameters in normalized
`[0, 1]` bound coordinates so gains, times, thresholds, and collapse rates have
comparable numerical scales. The first start is the model's valid default
vector; requested additional starts are constrained so RT shifts and collapsing
boundaries remain feasible, then scored and rejected unless every included row
has a finite positive density. By default, fitting screens 16 valid random
candidates (32 in recovery experiments) and only optimizes the highest-scoring
ones needed to fill the requested start count. The best result receives a
strict restart and a multiscale coordinate poll that remains meaningful at
RT-bin derivative kinks. Saved vectors can be supplied with repeated
`--initial-parameters` arguments.

For a standardized basin-search and mesh-refinement pipeline, use:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    staged-fit --subject 1 --output staged_fit_subject_1.json
```

The defaults run 1,000 coarse CMA-ES evaluations at 2 ms/33 points/20 ms,
warm-start a two-start exact-gradient fit at the ordinary model mesh, and
finish with a one-start 0.5 ms/129-point/5 ms polish. The output retains every
stage result and timing while exposing the final `parameter_vector` at the top
level, so it can be reused anywhere a normal fit JSON is accepted.

When Ninja and an OpenMP-capable C++ compiler are available, the CPU CLI builds
a research-local extension from `csi_kernels.cpp` on first use and fuses the
complete subject LCA scan, batched drift paths, DDM time loop, and their reverse
passes.
Install the small build dependency, if needed, with `uv pip install ninja`.
The compiled extension is cached outside the repository; subsequent processes
reuse it. `--no-native-lca-scan --no-native-ddm-forward` selects the Torch
correctness path explicitly.

On the 8-core development CPU, subject 1 at the default 1 ms/65-point/10 ms
mesh takes about 0.310 seconds per warm objective plus all 13 gradients. A score
alone takes about 0.078 seconds. The native OpenMP loops parallelize independent
drift and DDM lanes; the history-dependent 8 ms subject LCA scan remains serial.
One/two/four/eight-thread objective-gradient medians were
1.478/0.798/0.471/0.310 seconds. The 8-thread result is approximately 89 times
faster than the previous 27.6-second custom-Torch implementation and 280 times
faster than the original 86.9-second graph. The native log likelihood
matched the Torch path within `1.8e-11`, and every parameter gradient matched
within `5e-11`. A benchmark process with one warmup and two measured gradient
evaluations peaked at about 0.97 GB resident memory. These are machine-specific
prototype measurements.

The gradient tests use an off-grid, six-trial sequence spanning all conditions.
All 13 ordinary-autograd derivatives are checked against centered finite
differences, then the compact Torch and native adjoints are checked against the
ordinary-autograd result. Exact RT-cell boundaries are excluded from those
comparisons because they are legitimate piecewise-smooth numerical points
rather than differentiable ones.

Three full-resolution, single-start pilot fits gave:

| Subject | log L(default) | log L(fit) | Iterations | Evaluations | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | -3420.87 | -3067.41 | 69 | 99 | ~102 s |
| 4 | -3537.64 | -2961.35 | 41 | 57 | ~47 s |
| 7 | -3594.73 | -3532.06 | 36 | 46 | ~126 s |

All independently rescored fits had no invalid or zero-probability included
rows. A later three-start subject-1 run completed 323 exact evaluations in
about 126 seconds. The default and one random start reached the same basin
within 0.001 log-likelihood unit; a second random start exposed a poorer basin.
Strict and Powell polishing did not materially improve it. Although the smooth
projected-gradient test remains nonzero at mesh kinks, a normalized
`1e-3,1e-4,1e-5` coordinate poll found no move improving log likelihood by
`1e-7`. Subjects 1 and 7 still place parameters on bounds. More recovery seeds
and participants remain necessary. The comparison with approximately
three-minute, 5,000-candidate Triton simulator runs on a GB300 is not
apples-to-apples.

Use `--gradient-method parallel-finite-difference` as a diagnostic alternative.
It evaluates deterministic perturbations in worker processes, but the default
mesh is memory-bandwidth limited and it was not faster on the development
machine. `gradient-benchmark` times one such numerical gradient explicitly:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    gradient-benchmark --subject 1 --workers 7
```

The defaults use a 1 ms DDM mesh, 65 spatial grid points, float64, and an RK4
LCA maximum step of 10 ms. Use `--ddm-time-step`, `--ddm-spatial-points`, and
`--lca-max-step` for convergence studies. Ordinary library evaluations retain
the checkpointed Torch graph as a correctness oracle. The fitting CLI enables
the exact adjoints and available native CPU kernels automatically. Coarser
meshes remain useful for debugging and warm starts, not final scores.

Time the serial LCA, batched drift construction, and DDM solve separately with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    benchmark --subject 1 --repeats 3
```

Add `--with-gradient` to time one objective plus all 13 exact gradients. The
warmup evaluation loads or builds the native extension before measured runs.

Drift paths and PDE lanes are sorted into batches of 256 trials by duration.
Use `--ddm-bucket-size 0` for one fully padded batch, or benchmark smaller
buckets when using a GPU.

The sequential dependency still requires one serial pass over trials to obtain
the LCA state at each decision onset. The native CPU path performs that complete
scan and its analytic RK4 adjoint in one extension call. It constructs the
within-decision drift paths and solves each duration-bucketed PDE similarly.
The Torch implementation remains available as the independent numerical and
autograd oracle. The native kernels are CPU-only; the current CUDA route is not
fused and is not expected to outperform the CPU path.

Run a subject-level convergence comparison, listing meshes from coarse to fine,
with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    grid-refinement --subject 1 \
    --level 0.004,33,0.02 \
    --level 0.002,65,0.01 \
    --level 0.001,129,0.005
```

The final level is treated as the reference. The report includes total and
per-trial log-probability differences, LCA-state differences, mass diagnostics,
and component timings. At the three fitted pilot points, the default 1 ms/65
grid differed from a 0.5 ms/129 reference by 0.293, 0.156, and 0.077 total log
likelihood units for subjects 1, 4, and 7. Mean absolute per-trial differences
were 0.00157, 0.00049, and 0.00079.

Run one seeded full-model recovery without starting at the truth with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    recover --subject 1 --simulation-seed 17 --starts 2 \
    --simulated-output recovery_data.csv --output recovery.json
```

The generator retains the selected subject's ordered task/stimulus sequence,
uses the same continuous deterministic LCA, and samples a first-passage DDM path
at 0.5 ms. A Brownian-bridge test detects paths that cross and return between
Euler endpoints; `--no-bridge-correction` is available only for diagnostic
comparisons. The independent fitting likelihood remains the PDE solver.

Validate the synthetic first-passage generator separately from the sequential
LCA with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    generator-validation --paths 100000 \
    --output generator_pde_validation.json
```

This compares fixed and time-varying drift cases, including a collapsing
boundary, at 2, 1, and 0.5 ms against a 0.25 ms/257-point PDE reference. With
100,000 paths, the uncorrected endpoint detector retained 0.6--1.1 percentage
point maximum CDF errors. Brownian-bridge correction reduced those maxima to
0.10--0.30 points, no more than about three Monte Carlo standard errors across
all cases and meshes. The earlier one-seed recovery result predates this
correction and should not be treated as a calibration result.

A corrected 5-seed by 2-truth recovery matrix used 485 included observations
per fit. After repairing two obvious local-search failures (their fits were
more than 100 log-likelihood units below the known truth), every fitted vector
beat the generating vector on its finite sample. Mean normalized bound-range
RMSE was 0.0597 for the interior truth and 0.1084 for a deliberately contrasting
truth; corresponding medians were 0.0615 and 0.1082. The largest contrasting
regime errors were in condition-specific gains and non-decision times. A
64-candidate/four-start search repaired one failed basin. The other required a
1,000-evaluation 2 ms/33-point/20 ms coarse CMA-ES search followed by a
default-mesh L-BFGS-B polish. This supports an adaptive coarse global-search
fallback rather than unlimited blind local starts.

Warm-started fine-mesh polishing at 0.5 ms/129 points/5 ms changed the empirical
subject-1 and interior-recovery parameters by only 0.0010 and 0.0015 normalized
RMSE. Fine-grid likelihood improved by 0.0027 and 0.0019 from the respective
warm starts; recovery RMSE changed from 0.0615 to 0.0607. The default mesh is
therefore suitable for search in these checks, with the fine mesh reserved for
final polishing and reporting.

## Comparison with the legacy PNL simulator

Convert a direct fit to the one-row legacy CSV expected by the original
`ParameterEstimationComposition` driver with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    export-pnl-parameters --subject 1 --parameters staged_fit_subject_1.json \
    --output staged_fit_subject_1_pnl.csv
```

Because the PNL scheduler accepts only integer CSI execution counts, export
rounds continuous CSI to the nearest 10 ms count and records both the original
count and rounding error. After rescoring the direct and legacy-fitted vectors
with identical simulator seeds, `compare-pnl` evaluates both vectors under the
direct likelihood and aggregates paired simulator differences. Absolute direct
and histogram-likelihood numbers are deliberately not compared across methods.

The first three-subject check used three paired seeds with 10,000 simulations
per vector and seed:

| Subject | Direct objective: direct - PNL | PNL objective: direct - PNL | PNL paired SD |
| --- | ---: | ---: | ---: |
| 1 | +36.87 | -4.64 | 3.14 |
| 4 | +21.01 | -23.41 | 0.64 |
| 7 | +12.46 | +6.47 | 11.48 |

The direct objective preferred its fitted vector for all three participants.
The legacy simulator preferred its own vector for subjects 1 and 4, while the
direct vector had a higher mean simulator score for subject 7 with substantial
seed variability. Both implementations therefore run and rank candidates
coherently, but their optima remain detectably different under continuous
versus legacy 10 ms semantics.

Factor that disagreement into controlled numerical changes with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    semantic-ladder --subject 1 \
    --direct-parameters direct_fit_subject_1_fine_mesh.json \
    --pnl-parameters legacy_fit_subject_1.csv \
    --ddm-time-step 0.0005 --ddm-spatial-points 129 \
    --lca-max-step 0.005 --output semantic_ladder_subject_1.json
```

The ladder independently tests CSI rounding, PNL-sized fixed RT bins, a
reset-history ablation, Euler versus RK4 LCA updates, and a stable 10 ms PDE
mesh. Across subjects 1, 4, and 7, CSI rounding and LCA integration choice were
small effects. Fixed RT bins and especially the coarse temporal mesh reduced
some gaps, but every direct rung still preferred the direct vector. The closest
rung gave direct-minus-PNL rankings of +18.11, +11.83, and +10.44, versus PNL
simulator rankings of -4.64, -23.41, and +6.47.

The unresolved leading difference is the DDM process itself. The PNL simulator
checks a 10 ms Euler random walk for threshold crossing only at step endpoints;
the direct solver computes continuous first-passage flux. In a 50,000-path
diagnostic, 10 ms endpoint detection differed from a fine continuous PDE by up
to 2.68 percentage points in cumulative crossing probability (2.08 points in a
collapsing-boundary case).

That exact bridge is now implemented as a deterministic FFT transition solver.
It advances the Gaussian 10 ms Euler random walk and absorbs probability only
at step endpoints. At a converged 1023-point evidence grid it changes the
direct-minus-PNL vector rankings to +2.73, -17.73, and +1.90. Subject 4 now
agrees with the legacy preference, subject 7 remains consistent, and most of
subject 1's gap disappears. A reset-history sensitivity ablation gives -2.12,
-18.79, and +1.25, matching all three PNL preference signs; it is not a
substitute for the remaining explicit marginal-history calculation.

Check evidence-grid convergence independently with:

```bash
uv run python Scripts/Debug/pec_batch_compile/csi_fit/csi_direct_likelihood.py \
    endpoint-grid-refinement --subject 1 \
    --direct-parameters direct_fit_subject_1_fine_mesh.json \
    --pnl-parameters legacy_fit_subject_1.csv \
    --spatial-points 255 --spatial-points 511 --spatial-points 1023 \
    --output endpoint_grid_subject_1.json
```

The 511-point ranking gaps were within 0.036 log units of 1023 points for all
three pilots. This endpoint solver is a compatibility experiment without a
custom fitting adjoint; continuous first-passage remains the primary model.

The likelihood integrates the piecewise-constant numerical boundary flux over
each recorded RT interval. It is continuous as an interval endpoint crosses a
DDM time-cell boundary, but the numerical derivative has a kink there. The
implementation uses a deterministic zero subgradient for an endpoint exactly
on a cell edge. Gradient checks should therefore use observations or perturbed
parameters whose interval endpoints do not lie exactly on mesh boundaries.

Legacy conversions treat CSI values as 10 ms execution counts and threshold
collapse as an increment per 10 ms. Internally both are represented in seconds:
`csi_duration = legacy_csi * 0.01` and
`collapse_rate = legacy_collapse / 0.01`.
