"""Compare the noisy compiled network with the source composition in Python.

The only substituted native operation is NormalDist's random draw. Supplying
the GPU's Gaussian draws lets the original PNL scheduler, nonlinear functions,
resets, controllers, and stopping rule serve as a pathwise dynamics oracle.
The Gaussian generator itself has separate moment/stream tests.
"""

import importlib.util
import json
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.batched.backend.triton.cache import load_triton_kernel_module
from psyneulink.core.batched.backend.triton.runtime import _import_torch_triton
from psyneulink.core.globals.utilities import set_global_seed

from test_batched_vector_normal import _DrawEmitter


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
    pytest.mark.triton,
    pytest.mark.triton_gpu,
]

_DRIVER = (
    Path(__file__).resolve().parents[3]
    / "Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py"
)
_LAYERS = (
    "Control Units\n[Color, Location]",
    "Stimulus Units\n[Red, Blue, Left, Right]",
    "Decision Units\n[Left, Right]",
    "Response Units\n[Left, Right]",
    "LC",
)


def source_parity(seed, *, trials=6):
    """Return measured errors after native Python replays a noisy GPU path."""
    spec = importlib.util.spec_from_file_location("dawa_source_parity_model", _DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    set_global_seed(29)
    composition, inputs, outputs = driver.build_model(
        trials=trials,
        schedule="source",
        c_noise=0.1,
        s_noise=0.1,
        d_noise=0.1,
        r_noise=0.1,
    )
    layers = [driver.node(composition, name) for name in _LAYERS]
    # Alternate tasks as well as congruency to exercise retained control state.
    inputs[driver.node(composition, "Task Input")] = np.resize(
        [[1.0, 0.0], [0.0, 1.0]], (trials, 2)
    )
    ports = (*outputs, *(layer.output_port for layer in layers))
    plan = BatchedCompositionCompiler.compile(
        composition, backend="triton", outputs=ports, max_steps=1000
    )
    varying = {
        ("LC", "mode"): [0.2, 0.8, 0.3, 0.7, 0.1, 0.9],
        ("Bias Mechanism", "intercept"): [-0.45, -0.4, -0.48, -0.43, -0.35, -0.46],
        ("w1 Mechanism", "intercept"): [1.0, 1.3, 0.8, 1.1, 1.2, 0.9],
        ("w2 Mechanism", "intercept"): [1.2, 0.9, 1.1, 0.8, 1.0, 1.3],
        ("RT_GATE", "intercept"): [0.2, 0.1, 0.3, 0.15, 0.25, 0.12],
        (_LAYERS[0], "gain"): [10.0, 12.0, 8.0, 14.0, 11.0, 9.0],
        (_LAYERS[3], "termination_threshold"): [0.3, 0.35, 0.4, 0.3, 0.45, 0.35],
    }
    parameters = {
        f"{driver.node(composition, name).name}.{parameter}": BatchedTrialParameter(
            np.resize(values, trials)
        )
        for (name, parameter), values in varying.items()
    }
    # Use the production fitting launch configuration, including specialization.
    plan = plan.specialize_parameters(
        {p.name: p.default for p in plan.ir.params if p.name not in parameters}
    )
    actual = plan.run(
        inputs,
        [parameters],
        1,
        seed=seed,
        strict_truncation=True,
        triton_launch_options={
            "block_size": 32,
            "num_warps": 1,
            "trial_schedule": "independent",
            "normal_rng": "philox4x_fast_v1",
        },
    ).values[0, 0, :, 0]

    # Draws are indexed by (estimate * sequence_length + trial, execution,
    # component slot). Generate them independently of all network dynamics.
    torch, triton = _import_torch_triton(False)
    widths = tuple(stream.width for stream in plan.kernel_ir.rng_streams)
    draws = torch.empty((trials, 1000, sum(widths)), device="cuda")
    legacy = torch.empty_like(draws)
    module = load_triton_kernel_module(
        _DrawEmitter(widths, "philox4x_fast_v1").source(),
        "dawa_native_noise_replay",
        "test",
    )
    module.draws[(triton.cdiv(trials * 1000, 128),)](
        draws, legacy, trials, 1000, seed, 128
    )
    draws = draws.cpu().numpy()
    trial = -1
    counters = {}
    slot = 0
    for stream in plan.kernel_ir.rng_streams:
        layer = composition.nodes[stream.node]
        noise = layer.parameters.noise.get()
        counters[layer.name] = 0

        def replay(
            self,
            variable=None,
            context=None,
            params=None,
            *,
            layer_name=layer.name,
            width=stream.width,
            start=slot,
        ):
            assert 0 <= trial < trials
            count = counters[layer_name]
            counters[layer_name] += 1
            z = float(draws[trial, count // width, start + count % width])
            mean = self._get_current_parameter_value("mean", context)
            sd = self._get_current_parameter_value("standard_deviation", context)
            return self.convert_output_type(mean + sd * z)

        noise._function = MethodType(replay, noise)
        slot += stream.width

    expected = []
    executions = []

    def before_trial():
        nonlocal trial
        trial += 1
        for name in counters:
            counters[name] = 0
        for (name, parameter), values in varying.items():
            owner = driver.node(composition, name)
            if parameter == "mode":
                target = owner.integrator_function
            elif parameter == "termination_threshold":
                target = owner
            else:
                target = owner.function
            getattr(target.parameters, parameter).set(
                values[trial % len(values)], context=composition
            )

    def after_trial():
        expected.append(
            np.concatenate(
                [
                    np.asarray(port.parameters.value.get(composition)).ravel()
                    for port in ports
                ]
            )
        )
        counts = [
            int(layer.parameters.num_executions.get(composition).trial)
            for layer in layers[:4]
        ]
        for stream in plan.kernel_ir.rng_streams:
            count = (
                composition.nodes[stream.node]
                .parameters.num_executions.get(composition)
                .trial
            )
            assert counters[stream.node] == count * stream.width
        executions.append(counts)

    composition.run(
        inputs,
        execution_mode=pnl.ExecutionMode.Python,
        call_before_trial=before_trial,
        call_after_trial=after_trial,
        termination_processing={
            pnl.TimeScale.TRIAL: pnl.Any(pnl.AllHaveRun(), pnl.AfterNPasses(1000))
        },
    )
    expected = np.asarray(expected)
    assert max(map(max, executions)) < 1000
    np.testing.assert_array_equal(actual[:, 0], expected[:, 0])
    # Check stopping executions exactly, separately from FP32 clock/readout
    # rounding. A fixed absolute RT tolerance is inappropriate as RT grows.
    ndt = np.resize(varying[("RT_GATE", "intercept")], trials)
    gpu_executions = np.rint((actual[:, 1].astype(float) - ndt) / 0.01).astype(int)
    np.testing.assert_array_equal(gpu_executions, np.asarray(executions)[:, -1])
    np.testing.assert_array_max_ulp(
        actual[:, 1], expected[:, 1].astype(np.float32), maxulp=2
    )
    np.testing.assert_allclose(actual[:, 2:], expected[:, 2:], rtol=2e-6, atol=2e-6)
    return {
        "seed": seed,
        "trials": trials,
        "max_output_error": float(np.max(np.abs(actual[:, :2] - expected[:, :2]))),
        "max_activity_error": float(np.max(np.abs(actual[:, 2:] - expected[:, 2:]))),
        "python_response_executions": np.asarray(executions)[:, -1].tolist(),
        "gpu_choices_rt": actual[:, :2].tolist(),
        "python_choices_rt": expected[:, :2].tolist(),
    }


@pytest.mark.parametrize("seed", [17, 29])
def test_all_noisy_layers_match_native_python_with_common_draws(seed, tmp_path):
    report = source_parity(seed)
    (tmp_path / "source_parity.json").write_text(json.dumps(report, indent=2) + "\n")
