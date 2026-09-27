# Observation-conditioned GPU fitting

The DAWA fit and recovery drivers now default to `--likelihood conditioned`.
Each simulated particle carries its own latent state, and each observed
choice/RT updates the distribution of states entering the next trial. The
previous `--likelihood marginal` objective simulated complete stateful histories
but did not update them using earlier observations.

This uses the original composition's 10 ms LCA dynamics, with Gaussian noise in
all four LCA layers. It does not use the continuous direct-likelihood prototype.

## State and observation semantics

The ordinary generated simulator executes one trial per particle. The filter
then weights and systematically resamples complete terminal states. State
transport includes held controller outputs and the values last sampled by
their targets, because next-trial resets can use those sampled values before a
controller publishes again. DAWA's flat state buffer has 47 floats: 27 mechanism
states, 10 held control values, and 10 sampled values. The transport is a generic
compiler feature.

Conditioning uses a histogram observation kernel, rather than exact continuous
choice/RT densities. The default RT bin width is 30 ms and Gaussian smoothing
has standard deviation 0.5 bins. For in-range simulated outcomes, the Gaussian
kernel is normalized over possible **observed** bins, including at domain edges.
Simulated outcomes outside the finite domain retain implicit overflow mass;
they are not folded into an edge bin or removed by renormalizing the particles.

A symmetric pseudocount is interpreted as uniform observation contamination.
With `N` particles, `K` joint choice/RT bins, and pseudocount `alpha` per bin, the
contamination probability is `K*alpha/(N + K*alpha)`. Each particle receives its
ordinary observation weight plus `alpha/N`. This supplies a defined ancestry
distribution even when no simulated response matches the observation: the
filter then retains the predictive state distribution. At 100,000 particles,
100 RT bins, two choices, and alpha=1, contamination probability is about 0.2%.
Rescoring at another particle budget scales alpha to keep that probability fixed.

All retained observations condition state, including trials excluded from the
log score. The mask selects which conditional log densities to sum; it does not
mean an observation is missing. The result is the full sequence likelihood under
this observation model only when all terms are included. Masked RTs above 3 s
expand the domain in 30 ms increments. Missing observations are not supported.

The exact noisy-model likelihood is still approximated by finite particles and
the declared observation kernel. Effective sample size and posterior
contamination responsibility are available through
`conditioned_log_likelihood(..., return_diagnostics=True)`.

## Execution and validation

The prepared CUDA path validates and emits the kernel once per likelihood call,
uploads the sequence's inputs and parameters once, and reuses output, diagnostic,
and state buffers between trials. It executes the ordinary simulation kernel;
there is no DAWA-specific dynamics implementation. `execution="reference"`
retains the slower one-trial `plan.run()` loop for differential tests.

Tests cover exact split-versus-continuous seeded simulation, changing controller
values and trial parameters, both trial schedules, and prepared-versus-reference
filtering. Changing an earlier unscored observed RT changes subsequent filtered
predictions while leaving subsequent marginal predictions unchanged. Independent
finite-state tests compare the filtering recursion against an analytically
known posterior and sequence likelihood. Observation-kernel tests check boundary
normalization, overflow mass, contamination ancestry, and zero-support errors.

Candidate batches share systematic resampling offsets when common random
numbers are requested. Evaluating a candidate alone or in a batch therefore
preserves its resampling stream. The cumulative raw weights and systematic
positions use FP64: FP32 scan/reduction rounding changed particle ancestry
when the candidate batch size changed. Model simulation remains FP32; the
reported log scores can still differ at the level of FP32 reduction rounding.
Future simulation noise remains indexed by
destination particle and original trial number, so duplicate offspring do not
duplicate their future noise.

## RTX 2080 Ti measurements, 2026-09-27

The complete recorded subject 1 has 760 trials, of which 720 contribute to the
score. These are medians of three warm evaluations with independent seeds,
using the original 10 ms model, noise SD 0.1 in all four LCAs, 30 ms RT bins,
0.5-bin smoothing, and a constant approximately 0.2% contamination probability.
Both methods use the same parameter proposals and observations but evaluate
different objectives. Timings include simulation, observation updates,
resampling, and returning scores to the CPU. Compilation is excluded.

| Particles | Candidates per batch | Previous marginal, s/batch | Conditioned, s/batch | Conditioned, s/candidate | Conditioned peak allocation, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 0.169 | 1.626 | 1.626 | 6.35 |
| 10,000 | 4 | 0.357 | 1.481 | 0.370 | 23.29 |
| 100,000 | 1 | 0.913 | 1.938 | 1.938 | 57.23 |
| 100,000 | 4 | 3.011 | 5.594 | 1.399 | 227.39 |

At 100,000 particles the conditioning cost is **2.12x** for one candidate or
**1.86x** for batches of four. At the latter rate, 5,000 parameter evaluations
would take approximately **1.94 hours**, before optimizer and validation
overhead. These four proposals do not establish an average over an actual
optimization run. Peak allocation is Torch tensor memory in the benchmark
process, excluding CUDA context, allocator reservation, and other processes.
The compiled simulation kernels used 127–141 registers and reported no spills.

Preparation mattered substantially: a 64-trial, 10,000-particle evaluation
took 2.87 s with the initial per-trial preparation loop and 0.146 s with the
prepared runner. Profiling identified repeated source validation/emission and
thousands of small parameter uploads. The prepared path removes that repetition.
Small particle budgets still show substantial per-trial launch overhead; the
10,000-particle case is not a tenfold speedup over 100,000 particles.

This establishes execution correctness and timing, not particle convergence
or fitting accuracy. At 100,000 particles, the first proposal's median ESS was
2,645 and minimum ESS was 56; across the four proposals the minimum was about
5. Low-support observations can therefore still produce noisy filtering and
scores. Particle-budget and independent-seed checks remain necessary before
drawing conclusions from a fitted subject. The existing recovery studies used
the marginal objective and do not validate this conditioned objective.

The [compact results](fitting_acceleration/conditioned_2080ti.json) include all
timings and scores, ESS and contamination summaries, model/data/implementation
hashes, and warmup times. Large profiler traces are not included in the repo.

## Reproduction

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa/dawa_conditioned_benchmark.py \
  --subject 1 --estimates 10000 100000 --batch-sizes 1 4 --repeats 3 \
  --output /tmp/dawa-conditioned-benchmark.json
```

The benchmark measures both objectives on the same ordered subject, parameter
proposals, model clocks, smoothing width, and contamination fraction. Warmup
and compilation are reported separately. Its fit-time projection assumes 5,000
parameter evaluations at the measured batch size; it is not a convergence claim.
Use `--execution reference` to measure the original per-trial setup overhead.

Cached NDT profiling and the existing adaptive pooling of per-trial density
blocks require `--likelihood marginal`. They cannot be reused unchanged when
the observations and NDT affect subsequent particle histories.
