"""Flat retained-state ABI shared by generated kernels and their launcher."""


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
