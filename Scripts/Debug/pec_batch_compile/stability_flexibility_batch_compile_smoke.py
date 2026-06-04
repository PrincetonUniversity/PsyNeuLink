import sys
from pathlib import Path

import numpy as np

from psyneulink.core.batched import BatchedCompositionCompiler


HELPER_DIR = Path(__file__).resolve().parents[3] / "tests" / "composition" / "pec"
sys.path.insert(0, str(HELPER_DIR))

from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex  # noqa: E402


comp = make_stab_flex(
    lca_time_step_size=0.01,
    ddm_time_step_size=0.01,
    threshold=0.05,
    ddm_noise=0.0,
    lca_noise=0.0,
)
task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=1)
inputs = make_input_dict(comp, task[:2], stimulus[:2], cue[:2], correct[:2])
params = [{"threshold": 0.05, "ddm_noise": 0.0, "lca_noise": 0.0}]

for backend in ("reference", "triton"):
    report = BatchedCompositionCompiler.diagnose(comp, backend=backend)
    print(f"{backend} supported={report.is_supported} available={report.backend_available}")
    if not report.is_supported or not report.backend_available:
        print(report.unsupported_reasons)
        continue
    try:
        result = BatchedCompositionCompiler.compile(
            comp,
            backend=backend,
            max_steps=256 if backend == "triton" else None,
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
