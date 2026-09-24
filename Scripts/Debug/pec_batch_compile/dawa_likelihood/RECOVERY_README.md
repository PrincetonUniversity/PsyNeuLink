# Refined continuous DAWA recovery

This audit follows the initial [source-convergence and recovery study](STUDY_README.md).
It separates convergence of the fitting algorithm, spatial discretization error,
and weak parameter identification. The model uses instantaneous gain modulation
and an LC clock ratio of 20 throughout.

## Results

**All three unrestricted starts converge to the same solution at 129×129.**
They finish after 43, 39, and 20 accepted iterations, with maximum projected
gradients below `3.2e-7`. The mean training log likelihood is approximately
`-3.5723857413`. Thus the earlier iteration limit and starting point do not
explain the remaining LC parameter differences in this experiment.

| Parameter | Generating value | Converged estimate |
| --- | ---: | ---: |
| Threshold | 0.28000 | 0.28797 |
| Nondecision time (s) | 0.17000 | 0.17650 |
| Bias | -0.43000 | -0.42203 |
| Control gain | 12.000 | 12.340 |
| LC mode | 0.600 | 0.880 |
| LC scaling | 1.500 | **4.000, upper bound** |
| LC base gain | 5.400 | 5.120 |

At 257×257, the unrestricted fit differs from the generating model by at most
**0.00805 in joint choice/RT CDF probability**. Changing the fitted model's grid
from 129×129 to 257×257 changes that CDF by at most **0.00240**, and changes its
total training log likelihood by approximately **0.276**. These are checks of
predictions and scores at fixed parameters; they do not prove that the
257×257 parameter optimum is identical.

The conditional comparisons optimize the other six parameters:

| Constraint | Fitted LC mode | Fitted LC scaling | Training log-likelihood loss, 129×129 | Loss at 257×257, no refit |
| --- | ---: | ---: | ---: | ---: |
| None | 0.880 | 4.0 | 0 | 0 |
| Mode fixed at 0.6, high-scaling start | 0.600 | 4.0 | 0.933 | 0.943 |
| Scaling fixed at 1.5 | 0.900 | 1.5 | 0.930 | 0.914 |
| Mode fixed at 0.6, generating-value start | 0.600 | 1.0 | 2.607 | 2.396 |
| Mode fixed at 0.3 | 0.300 | 1.0 | 2.859 | 2.618 |

These losses are **total log-likelihood differences across 4,000 training
observations**, not per-trial differences. In particular, fixing scaling at
its generating value still pushes mode to its upper bound. Fixing scaling
alone does not recover the generating mode in this sample. The two converged
mode-0.6 fits also illustrate the need to check conditional fits from more than
one start. These are attained conditional likelihoods, without a claim that
every conditional global optimum has been found.

Held-out scores do not give a clear preference among these fits: their differences
from the unrestricted fit are within 1.4 paired standard errors. This describes
sampling variation in the fixed held-out split, not parameter uncertainty or
numerical-error confidence intervals.

The [LC trajectory comparison](study_results/refined_recovery/summary_trajectories.png)
shows the interpretation more directly: large changes in the early LC transient
and gain path can coexist with very similar response distributions. The
unrestricted fit produces a much larger LC fast-state pulse than the generating
model. These choice/RT observations do not tightly constrain that latent
trajectory in this design.
The displayed trajectories use a 0.25 ms maximum ODE step; the CDFs use the
257×257 verification probabilities.

* [Machine-readable comparison and held-out paired standard errors](study_results/refined_recovery/summary.json)
* [Numerical settings, simulation provenance, and synthetic observation counts](study_results/refined_recovery/provenance.json)

Complete optimizer traces and checkpoints remain local. The reproduction commands
below write them to the chosen output paths, and the report driver generates the
summary and LC-trajectory figures. Only the numerical summaries and LC-trajectory
PNG are retained here; duplicate PDFs and the additional summary figure are
generated artifacts.

This supports moving next to sequential recovery with explicit LC constraints
and the original condition-dependent modes, then relaxing gain parameters one
at a time. The synthetic generating constants used here are diagnostic
constraints, not proposed fixed values for empirical participants. One parameter
vector and one simulated data set do not establish general recovery or global
non-identifiability.

## Design

The data are the same independent continuous-SDE samples as the initial study:
2,000 training and 10,000 held-out observations for each of the color-incongruent
and location-incongruent conditions. Each observation starts with fresh states.
RTs use 25 ms bins, an initial 0–225 ms bin, and right censoring at 1.5 seconds.
The independent SDE generator uses 0.25 ms steps and Brownian-bridge crossing
corrections. The generating parameter vector is
`(0.28, 0.17, -0.43, 12, 0.6, 1.5, 5.4)` in the usual seven-parameter order.

The fitting grid is 129×129 finite-volume cells, with 1 ms coefficient intervals
and additional CFL-controlled integration steps. Completed fits are evaluated
again at 257×257. Verification scores hold the fitted parameters fixed; they do
not represent optimization on the verification grid.

Three starting points include the two original prescribed starts and the earlier
65×65 fit. Every fit uses all seven parameters unless a fixed parameter is
explicitly recorded. Fixed-mode comparisons re-optimize the other six parameters.
These sparse comparisons are samples of a profile likelihood, not confidence
intervals or proof of global non-identifiability.

All 28 numerical regression tests pass. They cover the original discrete and
continuous likelihoods, native coefficient/flux adjoints, gradients through trial
history, constrained quadratic optimality, interrupted multi-start recovery,
and paired held-out statistics. An independent adaptive DOP853 integration
checks the fitted deterministic paths. For the unrestricted estimate, the
maximum state discrepancy at the fitting ODE setting is below `4.8e-7` and falls
below `3.0e-8` with the finer ODE step; see
[the ODE checks](study_results/refined_recovery/ode_reference.json).

## Optimization and reproducibility

The optimizer uses BFGS in coordinates scaled by the parameter bounds. A small
active-set solve minimizes each quadratic approximation subject to the box and
step limits; feasible Armijo backtracking accepts actual likelihood improvements.
This avoids rescaling useful directions because an LC parameter points outside
its bound. Its stopping test is a maximum projected
gradient of `1e-6` for the mean negative training log likelihood. Iteration limits
and unsuccessful line searches are reported as unsuccessful termination.
The report saves every evaluation, accepted loss, gradient, and inverse-Hessian
restart state. Saving uses an atomic file replacement.

An optional expected-information matrix, estimated by finite differences of
65×65 probabilities at the current parameter vector, initializes the optimizer's
curvature estimate. Small eigenvalues are bounded below to avoid unstable steps.
This preconditioner affects search directions only. All accepted losses and
gradients use the 129×129 objective. It uses the fitting start, not knowledge of
the generating parameters, except in the explicitly truth-started fixed-mode run.

The native backend now also evaluates the finite-volume coefficients and their
first-order derivatives. Neighboring cells share a face calculation. Coefficient
and flux derivatives are checked against Torch, including moving boundaries,
gain, bias, initial mass, and each absorbing edge. `--retain-rates` trades RAM for
less recomputation; it does not change values or gradients.

From the repository root, after generating the cached samples with
`dawa_continuous_study.py`:

```bash
source .venv/bin/activate
python Scripts/Debug/pec_batch_compile/dawa_continuous_recovery.py \
  --points 129 --verify-points 257 --iterations 250 --starts 2 \
  --precondition-points 65 --retain-rates \
  --output /tmp/dawa_refined.json

# Continue saved accepted steps after an interruption or iteration limit.
python Scripts/Debug/pec_batch_compile/dawa_continuous_recovery.py \
  --points 129 --verify-points 257 --iterations 250 --resume --retain-rates \
  --output /tmp/dawa_refined.json

# Each --initial contains threshold, NDT, bias, control gain, LC mode,
# LC scaling, and LC base gain. Repeat it for additional starting points.
python Scripts/Debug/pec_batch_compile/dawa_continuous_recovery.py \
  --points 129 --verify-points 257 --iterations 250 \
  --precondition-points 65 --retain-rates --fix lc_mode=0.6 \
  --initial .28 .17 -.43 12 .6 1.5 5.4 \
  --output /tmp/dawa_mode06.json

python Scripts/Debug/pec_batch_compile/dawa_continuous_recovery_report.py \
  /tmp/dawa_refined.json /tmp/dawa_mode06.json \
  --output /tmp/dawa_refined_summary.json --trajectories
```

The comparison report ranks starts by training likelihood, preserves failed
convergence diagnostics, and computes held-out log-score differences against
the generating parameters. Its paired standard error accounts for the fixed
sample count within each condition. It does not quantify spatial or SDE
discretization error.

These fresh-trial results do not yet establish recovery for the original
condition-dependent, sequential fitting design. Warm-up trials, persistent
control state, and condition-specific LC mode need a separate recovery study.
