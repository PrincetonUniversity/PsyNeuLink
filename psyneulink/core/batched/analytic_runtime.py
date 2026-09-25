"""Shared float64 parameter and scalar-input preparation for direct providers."""

import numpy as np

from psyneulink.core.batched.prep import _extract_bound_input, normalize_parameter_sets


def parameter_tensor(ir, parameter_sets, *, requires_grad=False):
    import torch

    rows = normalize_parameter_sets(parameter_sets, ir)
    if not rows:
        raise ValueError("At least one parameter candidate is required.")
    if any(not np.isscalar(v) for row in rows for v in row.values()):
        raise ValueError("Analytic parameters must be scalar per candidate; trial-varying parameters are not supported yet.")
    return torch.tensor([[row[p.name] for p in ir.params] for row in rows],
                        dtype=torch.float64, requires_grad=requires_grad)


def validate_parameter_tensor(ir, parameters):
    import torch

    p = parameter_tensor(ir, None) if parameters is None else parameters
    if not isinstance(p, torch.Tensor) or p.device.type != "cpu" or p.dtype != torch.float64:
        raise ValueError("parameters must be a CPU float64 tensor.")
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != len(ir.params) or not len(p) or not torch.isfinite(p).all():
        raise ValueError("parameters must be finite with shape [candidate, parameter].")
    for i, spec in enumerate(ir.params):
        value = p[:, i]
        if spec.constant_value is not None:
            constant = torch.tensor(spec.constant_value, dtype=torch.float32)
            if torch.any(value.to(torch.float32).view(torch.int32) != constant.view(torch.int32)):
                raise ValueError(f"Parameter {spec.name} is specialized; compile a new plan to change its value.")
        if not spec.runtime_mutable and torch.any(value != spec.default):
            raise ValueError(f"Parameter {spec.name} is fixed by the source graph.")
        if spec.minimum is not None and torch.any(value < spec.minimum if spec.minimum_inclusive else value <= spec.minimum):
            raise ValueError(f"Parameter {spec.name} violates its lower bound.")
        if spec.maximum is not None and torch.any(value > spec.maximum if spec.maximum_inclusive else value >= spec.maximum):
            raise ValueError(f"Parameter {spec.name} violates its upper bound.")
    return p


def scalar_inputs(ir, bindings, inputs, num_trials):
    import torch

    external = {}
    for spec in ir.graph.inputs:
        raw = _extract_bound_input(inputs, spec, bindings)
        values = np.asarray(raw, dtype=np.float64)
        if (spec.width != 1 or values.ndim < 1 or values.shape[0] != num_trials
                or values.size != num_trials or not np.isfinite(values).all()):
            raise ValueError("Every scalar input needs one finite value per observed trial.")
        external[spec.component_id] = torch.as_tensor(values.reshape(1, -1))
    return external
