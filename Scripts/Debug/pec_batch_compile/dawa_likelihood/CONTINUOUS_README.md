# Continuous-time DAWA likelihood

`continuous_sequence_likelihood` implements a continuous-time extension of the
DAWA fitting model. A coupled deterministic ODE supplies the response drive and
LC gain. A two-dimensional absorbing Fokker–Planck equation then gives the joint
choice/response-time distribution. Gradients pass through both systems and
through history reconstructed from each parameter candidate.

The [source-convergence and recovery study](STUDY_README.md) now compares the
actual scheduled simulator against this formulation as both source time steps
shrink, and includes LC startup diagnostics, synthetic recovery, and optional
native CPU backends with checked gradients.

This is a research-local implementation alongside the existing
[10 ms discrete likelihood](README.md). The continuous API and driver are
separate; PEC's public fitting backend still uses its existing objective.

## Defined continuous model

The control, stimulus, and decision layers retain the fitting model's leak,
competition, logistic activations, and projection weights. Let `C`, `S`, and `D`
be their pre-logistic states, and `(v,w)` the two FitzHugh–Nagumo LC states:

* Control activity is `sigmoid(control_gain * C)`.
* Stimulus and decision activity is `sigmoid(g(t) * (state + bias))`.
* LC gain is `g(t) = lc_base_gain + lc_scaling * w(t)`.
* LC receives `0.3 * sum(decision_activity)` and modulates S, D, and R
  instantaneously, including the feedback from D to LC.

Fixed constants follow `dawa_batched_simulation.DEFAULTS`, including control
leak 7, control competition 3, control projection weight 4, and stimulus weights
1 and 1.2. Custom settings in a source model are not imported automatically.

The source runs ten LC Euler steps of 0.02 internal units per 0.01-second LCA
pass. The continuous ODE preserves that **20:1 LC/LCA clock ratio**. It multiplies
both LC derivatives by 20, while retaining the source time constants of 0.05
and 5.0. `--lc-clock-ratio` makes this assumption explicit. A ratio of 1 defines
a different model; it is not a numerical accuracy setting.

For the two response integrator states, the SDE is

```text
dX_i = [I_i(t) - 8 X_i - 8 sigmoid(g(t) (X_j + bias))] dt + 0.1 dW_i
I_0  = 4 sum(control_activity) + decision_activity_0 - decision_activity_1
I_1  = 4 sum(control_activity) - decision_activity_0 + decision_activity_1
```

`W_0` and `W_1` are independent Brownian motions. The first response activity
to reach `threshold` determines choice and decision time. For positive gain,
the corresponding upper boundary in each integrator coordinate is

```text
a(t) = logit(threshold) / g(t) - bias
```

The fitting configuration has response noise only. Adding decision or stimulus
noise would require a higher-dimensional or approximate likelihood.

At trial onset S, D, R, and both LC states reset to zero; C carries across trials.
Activity is evaluated immediately from these states and the current parameters,
so gain begins at `lc_base_gain`. This defines the continuous model rather than
preserving the old scheduler's first-pass allocation of 1, delayed gain
publication, or previously sampled gains at resets. Results therefore need not
match the legacy 10 ms simulator. The independent continuous SDE sampler is the
appropriate simulation reference.

## First passage, observations, and gradients

The solver propagates the joint survivor probability in `(X_0,X_1)`. Coordinates
`z = (x - lower) / (a(t) - lower)` fix the moving domain to `[0,1]^2`. The
transformed drift includes the boundary velocity; omitting it would solve the
wrong first-passage problem when LC gain changes.

Exponentially fitted finite-volume fluxes define a positive Markov generator.
SSP-RK2 time integration advances survivor mass and absorbing flux together.
Automatic CFL substeps maintain nonnegative mass. Upper-edge flux records the
choice-specific first-passage probabilities. There is no endpoint winner test
or time-step atom in the underlying model. Numerical output bins approximate
the continuous first-passage density. The lower truncation edges absorb and
report escaped mass as `lower_loss`; the solver never renormalizes that loss.

For a recorded RT `r`, resolution `delta`, and candidate nondecision time `ndt`,
the likelihood integrates choice-specific flux over

```text
[r - ndt - delta/2, r - ndt + delta/2]
```

The default recording resolution is 1 ms. No Gaussian RT measurement noise is
added. The reported likelihood is a **choice and RT-bin probability**, not a
density value. Changing the observation resolution changes these probabilities.
A conservative, nonnegative linear reconstruction of flux within each numerical
time cell avoids artificially flat NDT derivatives for narrower recording bins.

As in the CSI likelihood, control history advances for `r - ndt` on every trial,
including excluded warm-up observations. Consequently NDT affects both the
current RT interval and subsequent control states, with gradients through both.
This uses the recorded bin center to reconstruct history; it does not integrate
over uncertainty in the exact decision time within each rounding bin. Stimulus,
decision, response, and LC states reset, so no unobserved stochastic response
state must be carried into the next trial.

PyTorch autograd differentiates all seven parameters in this order:

```text
threshold, non_decision_time, bias, control_gain,
lc_mode, lc_scaling, lc_base_gain
```

A `[7]` vector shares parameters across trials. A `[trial,7]` tensor supports
condition-specific parameters and differentiable ties created by indexing.
Checkpointing reduces the memory needed for backward evaluation. Derivatives
are of the numerical likelihood; grid deposition, RT-bin interpolation, and
integer integration counts make it piecewise smooth. Refinement is necessary
to establish useful accuracy of both values and gradients.

## Running it

From the repository root:

```bash
# Adaptive ODE oracle, analytic moving-boundary race, spatial/time refinement,
# finite differences, CPU/GPU gradients, and 100,000 continuous SDE simulations.
PYTHONWARNINGS=ignore .venv/bin/python \
  Scripts/Debug/pec_batch_compile/dawa_continuous_likelihood.py validate \
  --device cpu --estimates 100000 --output /tmp/dawa_continuous_validation.json

# Twelve retained rows of subject 1: nine excluded warm-up rows, three scored.
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa_continuous_likelihood.py score \
  --device cpu --trials 12 --points 65 --output /tmp/dawa_continuous_score.json

# Small bounded gradient-ascent smoke test, with backtracking.
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa_continuous_likelihood.py fit \
  --device cpu --trials 12 --points 65 --iterations 1 \
  --output /tmp/dawa_continuous_fit.json
```

`score` and `fit` use the local, ignored `flanker_data_part1.csv` by default;
`--data` and `--subject` select other data. The CLI fits one shared vector. It
does not yet reproduce the original scripts' full subject/condition fitting
design. `validate` runs fixed oracle/refinement cases and records each case's
settings; it requires CUDA for Monte Carlo and device comparisons even with
CPU density calculations. The SDE sampler uses independent random trajectories
with Brownian-bridge crossing corrections and checks two time steps.

Library example, with `Scripts/Debug/pec_batch_compile` on `PYTHONPATH`:

```python
import torch
from dawa_likelihood import (
    DEFAULT_PARAMETERS, ContinuousConfig, continuous_sequence_likelihood,
)

p = torch.tensor(DEFAULT_PARAMETERS, dtype=torch.float64, requires_grad=True)
result = continuous_sequence_likelihood(
    p, tasks=[[1., 0.]], stimuli=[[0., 1., 0., 1.]],
    choices=[1], response_times=[.95], resolution=.001,
    config=ContinuousConfig(points=65, time_step=.001, ode_step=.0005),
)
gradient, = torch.autograd.grad(result.log_likelihood, p)
```

## Accuracy and scope

The PDE time step defaults to 1 ms; the deterministic RK4 maximum step is
0.5 ms. The PDE may substep further to meet its stability condition. These are
numerical approximation settings for continuous dynamics, not the model's
stopping-time resolution. Spatial refinement increases the number of cells
per response-state axis, separately from time refinement.

The [recorded audit](continuous_validation_results.json) includes the analytic
independent Brownian race with a moving boundary, independent adaptive ODE
integration, DAWA spatial/time refinement, all seven finite differences,
checkpoint equivalence, and CPU/GPU agreement. The continuous and existing
discrete regression suites pass together. Probability conservation and
nonnegativity hold to float64 precision in the audited cases.

For the default fresh-trial DAWA condition, the 257×257 CDF differs from 100,000
independent RTX 2080 Ti simulations by at most `0.00232`. The PDE uses 1 ms
coefficient intervals with 11 stability substeps in that case; simulations use
0.5 ms and 0.25 ms Euler/bridge steps. The 129×129 analytic moving-boundary race
has maximum CDF error `0.000399`. All seven gradients match centered finite
differences to approximately `1e-9` after scaling by `max(1, abs(gradient))`.

The 65×65 default is a quick exploratory setting. Spatial refinement has a much
larger effect than reducing the 1 ms time step in the audited DAWA case. Check
129×129 and 257×257 grids before interpreting a fit; a converged CDF alone does
not guarantee converged likelihood tails or gradients. Check lower-domain loss
and move the lower boundary farther down when necessary. The supported parameter
region requires positive gain and an upper boundary above the reset state
throughout the scored interval. Proposals outside it are rejected. Zero
observation probabilities are not floored.

Against the 257×257 reference, the maximum CDF differences at 33×33, 65×65,
and 129×129 are `0.0530`, `0.0136`, and `0.00281`. At fixed 65×65, reducing the
coefficient interval from 1 ms to 0.5 ms changes the CDF by only `2.3e-7`.
The coupled ODE's state error against adaptive DOP853 falls from `7.4e-7` to
`4.3e-8` to `2.6e-9` for maximum RK4 steps of 1, 0.5, and 0.25 ms.

In the two-trial gradient refinement case, the largest scaled gradient difference
from the 129×129 reference falls from `0.262` at 33×33 to `0.0676` at 65×65.
Halving the time step at 129×129 changes the gradient by `4.1e-5` on that scale.
These remaining spatial differences are material for inference; passing the
finite-difference check establishes differentiation correctness at a given grid,
not convergence of that grid to the continuous likelihood.

The [one-iteration real-data smoke test](continuous_fit_smoke.json) improved
log likelihood from `-22.14352` to `-18.76807` for three included observations
after nine warm-up rows. Each value-and-gradient evaluation took 42–46 seconds
on one CPU thread at 65×65 while other audits were running. These are audit
timings rather than a controlled throughput benchmark. This demonstrates working
optimization, not parameter
recovery or a converged subject fit. The small deterministic ODE runs on CPU even
when the density is on CUDA; differentiable device copies preserve gradients.
Further performance optimization and batched density evaluation remain future work.

The later [backend audit](STUDY_README.md#optional-faster-backends) adds optional
generated ODE and native flux execution. Select `--device cpu --ode-backend
generated --flux-backend native` after activating the virtual environment.
The earlier timings above describe the original Torch implementation.
