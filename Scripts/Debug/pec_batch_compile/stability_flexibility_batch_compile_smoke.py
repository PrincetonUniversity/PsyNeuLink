import sys
from pathlib import Path

import numpy as np

from psyneulink.core.batched import BatchedCompositionCompiler


HELPER_DIR = Path(__file__).resolve().parents[3] / "tests" / "composition" / "pec"
sys.path.insert(0, str(HELPER_DIR))

from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex  # noqa: E402


def _default_backend():
    # triton_cpu (interpret) and triton (compiled GPU) cannot coexist in one
    # process, so pick one by CUDA availability.
    try:
        import torch

        return "triton" if torch.cuda.is_available() else "triton_cpu"
    except ImportError:
        return "triton_cpu"


comp = make_stab_flex(
    lca_time_step_size=0.01,
    ddm_time_step_size=0.01,
    threshold=0.05,
    ddm_noise=0.0,
    lca_noise=0.0,
)
task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=1)
inputs = make_input_dict(comp, task[:2], stimulus[:2], cue[:2], correct[:2])
params = [
    {
        "DDM.threshold": 0.05,
        "DDM.noise": 0.0,
        "Task Activations [Act1, Act2].noise": 0.0,
    }
]

for backend in (_default_backend(),):
    report = BatchedCompositionCompiler.diagnose(comp, backend=backend)
    print(f"{backend} supported={report.is_supported} available={report.backend_available}")
    if not report.is_supported or not report.backend_available:
        print(report.unsupported_reasons)
        continue
    try:
        result = BatchedCompositionCompiler.compile(
            comp,
            backend=backend,
            max_steps=256,
        ).run(
            inputs=inputs,
            parameter_sets=params,
            num_estimates=1,
            seed=1,
        )
    except RuntimeError as error:
        print(f"{backend} skipped: {error}")
        continue
    print(backend, result.values.shape, np.squeeze(result.values))
