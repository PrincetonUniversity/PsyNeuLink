# CSI model experiments

This directory groups the CSI model, simulation benchmarks, recovery drivers,
and fitting workflows. Shared compiler benchmarks and design notes are in the
[parent directory](../README.md).

| Entry point | Purpose |
| --- | --- |
| [csi_model_surrogate.py](csi_model_surrogate.py) | CSI construction used by compiler tests and benchmarks |
| [csi_triton_vs_llvm.py](csi_triton_vs_llvm.py) | Batched GPU versus LLVM simulation benchmark |
| [csi_batched_parameter_recovery.py](csi_batched_parameter_recovery.py) | GPU simulation-based parameter recovery |
| [audit_generated_csi_compatibility.py](audit_generated_csi_compatibility.py) | Generated history and likelihood compatibility audit |
| [csi_fit/](csi_fit/README.md) | Empirical fitting, local and cluster launchers, and setup instructions |
| [csi_fit/direct_likelihood/](csi_fit/direct_likelihood/README.md) | Continuous and discrete direct likelihoods and recovery tools |

The original fitting model is
[expectation_model_study2_study3.py](<csi_fit/data fitting/expectation_model_study2_study3.py>).
Local behavioral data and generated fit outputs remain inside `csi_fit/` and
are not tracked. The existing filenames and command-line options are retained.

The [matched CSI/DAWA comparison](../CSI_DAWA_PERFORMANCE.md) measures direct
first-passage solvers and continuous GPU sampling with the same trial counts,
sample count, and decision horizon, including numerical refinement.

For example, from the repository root:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/csi/csi_triton_vs_llvm.py --help
.venv/bin/python Scripts/Debug/pec_batch_compile/csi/csi_fit/csi_direct_likelihood.py --help
```
