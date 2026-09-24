# DAWA direct likelihood prototype

For continuous-time choice/RT fitting, see the new
[continuous model and 2D first-passage solver](CONTINUOUS_README.md). It supports
all seven gradients, scores recorded RT intervals without added Gaussian noise,
and reconstructs history using each candidate's nondecision time. The remainder
of this page documents the original discrete 10 ms prototype and its observation
modes.

This research-local implementation evaluates the DAWA fitting configuration
with deterministic control, stimulus, decision, and LC dynamics and Gaussian
noise in the two response accumulators. It follows the corrected shared model
builder and uses the CSI direct-likelihood package as a guide for explicit
numerical settings, observed-history replay, gradient checks, and optimization
in normalized parameter coordinates. It does not change PEC's public fitting
backend.

The implementation has two observation modes:

* `step_sequence_likelihood` evaluates the joint probability of choices and
  **observed integer stopping steps**, preserving the complete deterministic
  history between trials. This directly targets the simulator's 10 ms endpoint
  threshold checks, subject to numerical quadrature and domain truncation.
  Nondecision time has zero derivative in this mode because it is not part of
  the stopping-step observation.
* `sequence_likelihood` evaluates choice and continuous RT observations with
  an explicit Gaussian measurement error around `step * 0.01 + ndt`, integrated
  over the RT rounding bin. Defaults are 10 ms measurement SD and 1 ms rounding
  resolution. The measurement error is an added observation-model assumption,
  not noise already present in the original DAWA source.

The RT mode requires previous latent stopping steps as a separate input. They
are known for simulator-generated data. For empirical data the CLI freezes
`round((RT - reference_ndt) / 0.01)`, using a reference NDT of 0.2 seconds by
default. **This is a plug-in history approximation, not a fully marginalized
sequential likelihood for noisy RT observations.** Optimizer proposals never
silently re-round or detach their own parameter-dependent history. The fitted
NDT gradient affects the RT observation model; history steps remain fixed.
Marginalizing uncertain past stopping steps remains future work.

## Numerical method

`model.py` replays the deterministic network using differentiable PyTorch
operations. It preserves the initial controller allocation of 1, ten internal
Euler FHN updates per pass, persistent task control, and trial resets using
the last sampled gains. Stimulus and decision layers use the held gain before
the current LC update; the response layer uses the newly published gain.
These details are verified against native Python on every pass across trials.

`solver.py` maintains a joint survivor distribution in the two pre-logistic
response states. For positive LC gain `g`, activity threshold `theta`, and bias
`b`, the survivor region has upper edge `logit(theta) / g - b` on both axes.
Each 10 ms transition has independent Gaussian innovations conditional on the
previous joint state; its means include leak and nonlinear cross-inhibition.

A tensor product Gauss-Legendre quadrature approximates the survivor integral.
The destination quadratures are normalized to their analytic in-domain
Gaussian probabilities. This ensures positivity and conservation, but does
not eliminate discretization error: inspect `quadrature_defect` and refine the
grid. Probability below the finite lower bound is reported as `lower_loss`,
never silently returned to the survivor distribution.

Exit probability is integrated separately, including the event that both units
cross on the same step. The winner is the larger final activity, matching the
native argmax rule. The solver returns the full `[step, choice]` probability
array, remaining survival, domain loss, and conservation diagnostics. It never
renormalizes the finite time horizon or floors zero observation probabilities.

## Gradients and fitting

Autograd differentiates the quadrature, moving boundary, competition, LC
trajectory, and carried deterministic history. Activation checkpointing
recomputes blocks of response-density steps during backward evaluation to
reduce memory. These are derivatives of the implemented numerical likelihood
conditional on its supplied history; they are not derivatives of a Monte Carlo
histogram or an estimator using simulated samples. No custom adjoint is needed
for this first prototype.

All seven shared fitting parameters are supported, in this order:

```text
threshold, non_decision_time, bias, control_gain,
lc_mode, lc_scaling, lc_base_gain
```

The library accepts a `[7]` vector or a `[trial, 7]` tensor. Constructing the
latter by indexing shared/condition-specific leaf parameters supports tied
parameters and conditional fits without breaking gradients. The CLI currently
fits one shared seven-parameter vector for one subject. Excluded observations
still advance history; their expensive response-density solves are skipped.

From the repository root:

```bash
# Native replay, 100k GPU simulations, spatial refinement, seven finite
# difference checks, and checkpointed/uncheckpointed gradient agreement.
PYTHONWARNINGS=ignore .venv/bin/python \
  Scripts/Debug/pec_batch_compile/dawa_direct_likelihood.py validate \
  --device cuda --points 97 --output /tmp/dawa_direct_validation.json

# The first nine retained rows of subject 1 are excluded warm-up trials.
# Twelve retained rows therefore score three observations and retain history.
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa_direct_likelihood.py score \
  --trials 12 --points 97 --max-steps 120 --output /tmp/dawa_direct_score.json

# Bounded gradient ascent with backtracking, in unit parameter coordinates.
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa_direct_likelihood.py fit \
  --trials 12 --points 97 --max-steps 120 --iterations 3 \
  --output /tmp/dawa_direct_fit.json
```

For `score` and `fit`, use `--device cpu` when CUDA is unavailable. The `validate`
command's Monte Carlo stage requires the existing Triton GPU simulator even
when density calculations use the CPU. The CSV remains local and ignored by
Git. A CSV `decision_steps` column supplies known history instead of the CLI's
RT-derived approximation. Use `--measurement-sd`, `--resolution`, and
`--history-reference-ndt` to make the observation/history assumptions explicit.
The fit command is a small optimizer smoke test, not a production multi-start
fit or parameter-recovery study.

Library example (add `Scripts/Debug/pec_batch_compile` to `PYTHONPATH`):

```python
import torch
from dawa_likelihood import DEFAULT_PARAMETERS, sequence_likelihood

p = torch.tensor(DEFAULT_PARAMETERS, dtype=torch.float64,
                 device="cuda", requires_grad=True)
result = sequence_likelihood(
    p, tasks=[[1., 0.]], stimuli=[[0., 1., 0., 1.]],
    choices=[1], response_times=[.95], history_steps=[75], max_steps=150,
)
gradient, = torch.autograd.grad(result.log_likelihood, p)
```

For this single-trial example the supplied stopping step affects only the state
that would be carried into a subsequent trial; the RT likelihood sums all
possible current stopping steps.

## Validation and limits

The recorded [validation results](validation_results.json) cover the local RTX
2080 Ti, float64 calculations, the default parameter vector, and a perturbed
vector for gradient and native replay checks. Native replay agrees within
`4e-16`. Against 100,000 fresh-trial Triton samples the largest joint choice/RT
CDF discrepancy is about `0.00207`. Increasing the grid from 97 to 129 points
per axis changes that CDF by less than `5e-8`; the 65-point grid is measurably
less accurate. All seven gradient coordinates agree with centered finite
differences to better than `2e-8` after scaling by `max(1, abs(gradient))`.
Refining the gradient calculation from 97 to 129 spatial points and from 24
to 48 winner-quadrature points changes the scaled gradient by less than `1.3e-6`.
Nine CPU regression tests cover analytic first-step probabilities and
derivatives, simultaneous crossings, domain loss, all seven finite differences,
checkpoint equivalence, excluded-trial history, native resets, exact stopping
step scoring, and the optimizer's handling of invalid proposals.

A three-iteration real-data smoke test, retaining twelve rows of subject 1
and scoring three after warm-up, increased the RT-mode log likelihood from
`-22.64185` to `-18.59340`. This tests optimization of the documented plug-in
objective, not parameter recovery or a converged subject fit.

Grid size must be checked again for other parameter regions, particularly
higher response bounds or lower gains. Conservation alone cannot establish
accuracy because transition quadratures are normalized. A finite horizon also
leaves survivor mass; extend it when relevant to the observed RT bins. Changing
decision noise from zero, altering fixed model constants, or changing source
scheduling falls outside this validated configuration.

This is a correctness and gradient prototype. The 97-point response solver
took about 0.46 seconds for one 160-step trial in the validation run; an end-to-end
two-trial value-and-gradient evaluation took about 6.6 seconds. These are audit
timings, not a warmed throughput benchmark. The implementation is currently
slower than the highly parallel batched simulator. Batching density solves and
optimizing deterministic replay are future performance work.
