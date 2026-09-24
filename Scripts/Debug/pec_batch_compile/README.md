# PEC batch compiler experiments

Model-specific code lives in [csi/](csi/README.md) and [dawa/](dawa/README.md).
Shared compiler benchmarks and design notes stay in this directory. The compiler
implementation itself lives in [psyneulink/core/batched](../../../psyneulink/core/batched/).

```text
pec_batch_compile/
  csi/
    csi_model_surrogate.py
    csi_triton_vs_llvm.py
    csi_batched_parameter_recovery.py
    audit_generated_csi_compatibility.py
    csi_fit/                 # fitting drivers, source model, and cluster launchers
      direct_likelihood/
      handoff/
  dawa/
    dawa_batched_simulation.py
    dawa_llvm_benchmark.py
    dawa_continuous_*.py
    dawa_direct_likelihood.py
    dawa_lca_model/           # source model and original fitting scripts
    dawa_likelihood/          # solvers, validation, and selected study results
  benchmark_*.py             # shared compiler/numerical benchmarks
  gpu_batch_compile_benchmark.py
  validate_likelihood_compile_gpu.py
```

The model directories retain their existing script and module names. Run the
documented commands from the repository root; include the `csi/` or `dawa/`
segment in script paths. Local data, fit outputs, and caches move with their
model directories and remain ignored by Git.

| Shared entry point | Purpose |
| --- | --- |
| [gpu_batch_compile_benchmark.py](gpu_batch_compile_benchmark.py) | General batch compiler workloads |
| [benchmark_continuous_dynamics.py](benchmark_continuous_dynamics.py) | Generated ODE integration and gradients, using a CSI fixture |
| [benchmark_first_passage_backend.py](benchmark_first_passage_backend.py) | Generic first-passage solver performance |
| [benchmark_likelihood_specializations.py](benchmark_likelihood_specializations.py) | Generated and specialized likelihood routes, using CSI data |
| [validate_likelihood_compile_gpu.py](validate_likelihood_compile_gpu.py) | Conditional GPU sampling and scoring checks |

Start with the [likelihood compiler usage guide](LIKELIHOOD_COMPILE_USAGE.md)
and [batch compiler development notes](BATCH_COMPILE_WIP.md). The
[continuous equation compilation notes](CONTINUOUS_EQUATION_COMPILATION.md)
and [numerical backend notes](NUMERICAL_LIKELIHOOD_BACKEND.md) describe the
shared integration and first-passage machinery.
