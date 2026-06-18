import numpy as np
import psyneulink as pnl

from psyneulink.core.batched import BatchedCompositionCompiler


def _default_backend():
    # triton_cpu (interpret) and triton (compiled GPU) cannot coexist in one
    # process, so pick one by CUDA availability.
    try:
        import torch

        return "triton" if torch.cuda.is_available() else "triton_cpu"
    except ImportError:
        return "triton_cpu"


decision = pnl.DDM(
    function=pnl.DriftDiffusionIntegrator(
        starting_value=0.0,
        rate=1.0,
        noise=0.0,
        threshold=0.05,
        non_decision_time=0.0,
        time_step_size=0.01,
    ),
    output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
    name="DDM",
)
comp = pnl.Composition(pathways=decision)
inputs = {decision: np.array([[1.0], [-1.0]], dtype=float)}
params = [{"rate": 1.0, "threshold": 0.05, "noise": 0.0, "time_step_size": 0.01}]

for backend in (_default_backend(),):
    report = BatchedCompositionCompiler.diagnose(comp, backend=backend)
    print(f"{backend} supported={report.is_supported} available={report.backend_available}")
    if not report.is_supported or not report.backend_available:
        continue
    try:
        result = BatchedCompositionCompiler.compile(
            comp,
            backend=backend,
            max_steps=64,
        ).run(
            inputs=inputs,
            parameter_sets=params,
            num_estimates=2,
            seed=1,
        )
    except RuntimeError as error:
        print(f"{backend} skipped: {error}")
        continue
    print(backend, result.values.shape, result.values[0, 0, :, :, :])
