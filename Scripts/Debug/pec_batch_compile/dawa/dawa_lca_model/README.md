# DAWA LC/LCA source model

For the current single-subject GPU fitting and recovery commands, use the
[fitting guide](../README.md). The scripts described below preserve the
original partition-wide LLVM workflow.

`full_lca_model_lc.py` builds the original scheduled PsyNeuLink network: control,
stimulus, decision, and response LCAs, with a FitzHugh–Nagumo LC mechanism
modulating stimulus, decision, and response gain. The builder uses the recurrent
schedule described in the [batch compiler notes](../COMPILER_NOTES.md#recurrent-scheduling).
Bias and weight controllers execute once per trial; the processing layers keep
integrating until the response reaches threshold.

Both `make_lca_model` and `run_lca_model` accept `c_noise`, `s_noise`, `d_noise`,
and `r_noise` (zero-mean Gaussian standard deviations). Setting all four to 0.1
enables noise throughout the LCA network without changing reset policies:
control carries state across trials, while the other three LCAs reset. Existing
defaults are unchanged. The batch compiler supports noisy persistent control;
see the [noise support notes](../COMPILER_NOTES.md#noise-in-all-four-lcas).

`flanker_fit_lc_part1.py`, `flanker_fit_lc_part2.py`, and
`flanker_fit_lc_part3.py` preserve the original PEC fitting setup for the three
data partitions. They use 10,000 estimates, up to 5,000 optimization iterations,
and LLVM execution. LC mode depends on previous congruency; several other
parameters depend on subject. These scripts fit all subjects in their partition;
their existing `--subject_id` argument does not filter the data.

Run each script from this directory with its corresponding
`flanker_data_partN.csv` available locally. Behavioral data, generated fit CSVs,
and logs are intentionally untracked. The CSV schema and filtering are shown
in the scripts. The supplied `.slurm` files are examples for the original
cluster's modules, environment, and resource settings; adapt those settings
before submitting elsewhere. Full empirical fits are not part of the regression
suite.

The [batched simulation driver](../dawa_batched_simulation.py) and
[LLVM/GPU benchmark](../dawa_llvm_benchmark.py) use this same model builder.
The separate [direct-likelihood prototypes](../dawa_likelihood/README.md)
include a discrete approximation and a continuous-time extension. The continuous
GPU sampler is an independent validation implementation for the latter; it does
not replace this source model or these fitting scripts.
