# DAWA source-to-continuous convergence and recovery

The tested source models approach the continuous formulation when the LCA and
LC time steps shrink together at a fixed 20:1 clock ratio. This supports using
instantaneous modulation as the continuous limit of the source schedule. The
original 10 ms model has noticeable finite-step effects, especially its initial
LC pulse. The study establishes numerical behavior of these configurations;
it does not establish which LC timescale or transient was scientifically intended.

## Source convergence

The stochastic comparison uses the actual shared PNL model and ordinary Triton
batch compiler, not a reimplementation of the scheduled simulator. Each of two
parameter vectors is tested on color/location tasks with congruent/incongruent
stimuli. Each condition is an independent fresh trial, with **100,000 estimates**
per condition, parameter vector, and time step. The horizon is two seconds of
decision time. Censored samples remain in the probability denominator.

The source LCA steps are 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625, and 0.078125 ms.
LC still performs ten internal updates per outer pass, with internal step equal
to twice the LCA step. Thus all eight runs preserve `10 * LC_dt / LCA_dt = 20`.
The continuous reference uses 0.25 ms Euler SDE steps with Brownian-bridge crossing
corrections, and a maximum deterministic RK4 step of 0.125 ms. Source and
reference random streams are independent.

Across the eight parameter/condition cases, maximum joint choice/time CDF
differences decrease from **0.0291–0.0436 at 10 ms** to **0.00678–0.00992 at
0.078125 ms**. These finest differences still contain endpoint-crossing bias
and Monte Carlo variation; they are not evidence of exact agreement at that step.
The error plot's shaded band is an approximate sampling scale, not a simultaneous
confidence certificate across every condition and time point.

* [Joint CDF error versus source step](study_results/distribution_error_convergence.png)
* [Numerical results](study_results/convergence.json)

The study driver also generates response-time CDF overlays and `curves.npz`
in its output directory; these additional artifacts are not tracked.

Two selected conditions also compare the continuous SDE reference with the
absorbing PDE. At 257×257 and 1 ms coefficient intervals, maximum CDF differences
are **0.00232** and **0.00388**. Probability conservation errors are below `1e-15`,
and lower-domain loss is negligible. The PDE uses additional stability substeps;
its 1 ms coefficient interval is not an endpoint-only crossing rule.

## LC trajectories and startup pulse

The deterministic comparison forces two 0.4-second trials, with a task switch
and the source model's trial resets between them. It compares all ten internal
states at fixed 10 ms physical-time locations. Replay is checked against native
PNL Python at three source time steps, including across resets, to within `1e-12`.
Fixed sampling locations avoid confusing the first-pass gain-allocation jump,
whose duration shrinks with the outer step, with a persistent state discrepancy.

RMS state error falls from `0.0651` to `0.00162` for the default parameter vector,
and from `0.0776` to `0.000319` for the perturbed vector, when the outer step shrinks
from 10 ms to 0.3125 ms. The gain paths and LC states converge across the reset.

The initial fast-LC pulse is particularly sensitive to the outer update:

| Configuration, default parameters | Peak LC fast state |
| --- | ---: |
| Original 10 ms outer step, ten LC updates | 1.036 |
| Same outer step, 100 finer LC updates, same elapsed LC time | 1.029 |
| Same outer step, initial held gain changed from 1 to 5 | 0.942 |
| Outer step reduced to 2.5 ms, ratio still 20 | 0.300 |
| Continuous model | 0.323 |

The pulse remains large when only LC's internal integration is refined. Changing
the initial gain alone also leaves a large pulse. The joint refinement points
to the coarse outer network/coupling update as the main source of the difference.
LC's fast time-constant coefficient corresponds to `0.05 / 20 = 0.0025` seconds
on the decision-time axis, while the original outer update is 0.01 seconds.

The study driver generates `trajectory_convergence.png` and
`startup_ablation.png` in its output directory. Their numerical comparisons
are retained in [the study report](study_results/convergence.json).

These comparisons do not change the source model, its default schedule, or its
initial allocations. They identify a modeling choice that matters when comparing
fits from the original scheduled model with fits from its continuous extension.

## Synthetic recovery and identification

The [refined recovery audit](RECOVERY_README.md) extends the preliminary fits
below with 129×129 optimization, multiple starts, fixed-LC-mode comparisons,
257×257 verification, and explicit convergence diagnostics.

Recovery data come from independent continuous SDE simulations of the perturbed
parameter vector. The test uses color-incongruent and location-incongruent fresh
trials, with **2,000 training and 10,000 held-out observations per condition**.
The objective groups RTs into 25 ms bins (with an initial 0–225 ms bin) and retains
responses after 1.5 seconds as a right-censored category. No Gaussian RT jitter
is added. This is a multinomial choice/RT-bin likelihood, with gradients through
all fitted parameters.

The [threshold/NDT recovery](study_results/recovery_two.json) holds the other
five parameters at their generating values. It recovers threshold approximately
`0.28044` from `0.28` and NDT approximately `0.16755 s` from `0.17 s`. On the
129×129 verification grid, the fitted and generating joint CDFs differ by about
`0.00274`.

The recovery driver generates the corresponding threshold/NDT recovery figure.

The [seven-parameter experiment](study_results/recovery_all.json) evaluates
predictive recovery separately from recovery of individual parameters. The
local information calculation finds two much weaker parameter combinations.
In coordinates scaled by the fitting ranges, the weighted-sensitivity condition
number is about **5,800**, with the weakest direction dominated by LC scaling
and LC mode. The sensitivity is full rank locally, but those two directions are
poorly constrained by this data set. This is a result for these fresh-trial,
coarsened RT observations; it is not a global non-identifiability proof.

The seven-parameter run completed 25 accepted iterations and reached its
iteration limit. At 129×129 its predicted joint CDF differs from the generating
CDF by at most `0.0148`. LC mode is approximately `0.898` versus the generating
`0.6`, and scaling is `1.83` versus `1.5`. These are preliminary predictive
recovery results, not a converged seven-parameter recovery claim.

The recovery driver also generates the seven-parameter recovery figure.

Both fits use the 65×65 exploratory grid and are checked again at 129×129. A full
sequential parameter-recovery study, finer-grid optimization, multiple generating
vectors, and repeated data sets remain necessary before claiming reliable
recovery of all seven parameters for the empirical fitting design.

## Optional faster backends

The ODE path can now use the compiler's existing general generated RK4 phase and
adjoint machinery. The model-specific part declares the equations. Integration,
symbolic equation derivatives, and their native execution are shared machinery.
A half-cell prefix aligns the phase backend's midpoint readouts with the endpoints
needed by the PDE. This changes the RK4 partition slightly; values and gradients
are checked against the Torch implementation.

The optional native CPU flux loop implements the same conservative SSP-RK2
scheme as the Torch solver, including all three absorbing categories. Its reverse
pass differentiates arbitrary supplied transition rates; it contains no DAWA
equations or parameter-specific derivatives. Checkpointed blocks retain the
existing memory strategy. Regression tests compare the native and Torch state,
rate, exit-flux, and sequential likelihood gradients.

The refined audit also adds a fused native implementation of the sigmoid-LCA
finite-volume coefficients and their derivatives. That model-specific coefficient
calculation is separate from the generic absorbing time loop. The original
timings below predate this additional implementation.

These backends support **CPU float64 and first-order derivatives**. The Torch
backends remain the defaults. The native backends require a C++ compiler and
Ninja on `PATH`; activating the repository virtual environment supplies Ninja
on this machine. [Warmed timings and parity](study_results/profile.json) measure
one trial's likelihood and gradient, not full-subject fitting throughput.

Median warmed times over three runs were **32.85 s** for Torch ODE/Torch flux,
**17.12 s** for generated ODE/Torch flux, and **4.79 s** for generated ODE/native
flux. The largest gradient difference from the reference was `5.1e-9`. Other
audits were active during parts of the run, so these timings describe the local
single-trial audit. The native flux loop itself agrees with the Torch loop and
its rate/state/absorbing-flux derivatives to roughly `1e-15` on the small random
operator test.

## Current direct and sampling timings

A later [short warmed timing check](study_results/performance.json) uses the
unrestricted refined-recovery estimate, both incongruent conditions, and a
1.401-second decision horizon. The native direct solver uses a 129×129 grid,
1 ms coefficient intervals with stability substeps, float64, and one CPU thread.
It scores the 4,000 synthetic training observations from the recovery audit.

| Workload, two conditions | Median warm time |
| --- | ---: |
| Direct likelihood | 12.59 s |
| Direct likelihood and all seven parameter gradients | 30.64 s |
| Continuous GPU sampling, 100,000 estimates per condition | 0.0726 s |

Direct timings use two measured calls after warm-up; sampling uses three.
The sampler shares generated deterministic trajectories across its response
simulations, uses float32 response states and 0.25 ms steps on the RTX 2080 Ti,
and includes trajectory generation, transfers, and returned samples. It excludes
density estimation, likelihood scoring, and parameter gradients. Compilation is
excluded from all three timings. This compares the continuous-model implementations
at one parameter vector, with different numerical approximations and precision;
it does not establish equal-accuracy likelihood or full-fit throughput. These
sampling timings are not measurements of the original scheduled PEC simulator.

The optimized native direct backend remains CPU-only. The Torch reference can
also run the density calculation on CUDA; a dedicated fused GPU density solver
has not been implemented.

## Reproduce

From the repository root:

```bash
# Actual source simulations, continuous references, PDE checks, and plots.
PYTHONWARNINGS=ignore .venv/bin/python \
  Scripts/Debug/pec_batch_compile/dawa_continuous_study.py \
  --output /tmp/dawa_continuous_study

# Native backends need the environment's Ninja executable on PATH.
source .venv/bin/activate

python Scripts/Debug/pec_batch_compile/dawa_continuous_recovery.py \
  --study /tmp/dawa_continuous_study --free threshold non_decision_time \
  --iterations 15 --output /tmp/dawa_recovery_two.json

python Scripts/Debug/pec_batch_compile/dawa_continuous_recovery.py \
  --study /tmp/dawa_continuous_study --iterations 25 --information \
  --output /tmp/dawa_recovery_all.json

python Scripts/Debug/pec_batch_compile/dawa_continuous_profile.py \
  --output /tmp/dawa_profile.json

# Use the optional faster backends for ordinary sequential scoring as well.
python Scripts/Debug/pec_batch_compile/dawa_continuous_likelihood.py score \
  --device cpu --ode-backend generated --flux-backend native --trials 12
```

The study directory caches raw synthetic samples for resuming stages. The
repository retains selected numerical reports, the source-convergence figure,
and the refined LC-trajectory comparison. Raw samples, curve arrays, optimizer
checkpoints, duplicate PDFs, and other generated figures remain local; the
directory's `.gitignore` explicitly lists the retained artifacts. Changing
parameters or conditions requires a new output directory. Recovery uses projected BFGS with feasible
Armijo backtracking, and records iteration-limit and line-search outcomes rather
than treating a rejected proposal as convergence.
