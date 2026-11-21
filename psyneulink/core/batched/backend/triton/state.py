"""Flat retained-state ABI shared by generated kernels and their launcher."""


def model_value_layout(params, states):
    """Pack uniform FP32 parameter values followed by retained-state initializer inputs."""
    values = [p.default if p.constant_value is None else p.constant_value for p in params]
    offsets = {}
    for state in states:
        offsets[state.state_id] = len(values)
        initializer = state.function_initializer
        values.extend(state.initial_value if initializer is None else initializer.input_value)
    return values, offsets


def retained_control_layout(graph):
    """Return control-value offsets following the mechanism-state prefix.

    Held modulation and the last value sampled by a target are distinct state:
    a trial-start reset may use the sampled value before the controller executes
    again. Preserve both across split launches and particle resampling.
    """
    offset = sum(state.width for state in graph.states)
    sampled = {
        modulation.effective_parameter_id
        for modulation in graph.modulations
        if graph.node(modulation.controller).attrs.get("scalar_override_control")
    }
    layout = {}
    for parameter in graph.effective_parameters:
        parameter_id = parameter.effective_parameter_id
        layout[f"effective:{parameter_id}"] = offset
        offset += parameter.width
        if parameter_id in sampled:
            layout[f"sampled:{parameter_id}"] = offset
            offset += 1
    return layout, offset
