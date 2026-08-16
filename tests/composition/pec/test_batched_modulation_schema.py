"""Object-free identity schema for typed batched parameter modulation."""

from dataclasses import FrozenInstanceError, replace

import pytest

from psyneulink.core.batched import (
    BatchedAbsorbedProjectionSpec,
    BatchedEffectiveParameterSpec,
    BatchedModulationSpec,
    BatchedParameterBindingSpec,
    BatchedPortSpec,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _absorbed_projection(*, kind="MappingProjection"):
    return BatchedAbsorbedProjectionSpec(
        projection_id=0,
        name="edge",
        kind=kind,
        sender="source",
        sender_component_id=0,
        sender_port="RESULT",
        sender_port_id=0,
        receiver="target",
        receiver_component_id=1,
        receiver_port="InputPort-0",
        receiver_port_id=1,
        initial_value=(1.0,) if kind == "ControlProjection" else (),
    )


def _modulation():
    binding = BatchedParameterBindingSpec(
        argument="slope",
        parameter="controller.slope",
        parameter_id=0,
    )
    return BatchedModulationSpec(
        modulation_id=0,
        controller="controller",
        controller_component_id=1,
        controller_input_port="InputPort-0",
        controller_input_port_id=1,
        control_signal_port="ControlSignal-0",
        control_signal_port_id=2,
        source="source",
        source_component_id=0,
        source_port="RESULT",
        source_port_id=0,
        target="target",
        target_component_id=2,
        target_parameter="termination_threshold",
        target_parameter_port_id=3,
        effective_parameter_id=0,
        monitor_projection_id=0,
        control_projection_id=1,
        controller_function_spec_key="example.Linear",
        controller_param_bindings=(binding,),
    )


def test_typed_modulation_schema_is_object_free_and_immutable():
    port = BatchedPortSpec(
        port_id=3,
        name="termination_threshold",
        owner="target",
        owner_component_id=2,
        kind="ParameterPort",
        width=1,
    )
    effective = BatchedEffectiveParameterSpec(
        effective_parameter_id=0,
        target="target",
        target_component_id=2,
        target_parameter="termination_threshold",
        target_parameter_port_id=3,
        base_value=(9.0,),
        initial_modulation_value=(1.0,),
    )
    monitor = _absorbed_projection()
    control = _absorbed_projection(kind="ControlProjection")
    modulation = _modulation()

    assert port.kind == "ParameterPort"
    assert effective.storage == "lane_persistent"
    assert monitor.initial_value == ()
    assert control.initial_value == (1.0,)
    assert modulation.controller_param_bindings[0].parameter_id == 0
    with pytest.raises(FrozenInstanceError):
        modulation.mode = "MULTIPLICATIVE"


@pytest.mark.parametrize(
    ("factory", "changes", "message"),
    [
        (_absorbed_projection, {"projection_id": -1}, "identities"),
        (
            _absorbed_projection,
            {"kind": "ControlProjection", "initial_value": ()},
            "initial value",
        ),
        (_modulation, {"mode": "PREFIX_OVERRIDE"}, "scalar float32 OVERRIDE"),
        (_modulation, {"controller_component_id": True}, "identities"),
        (
            _modulation,
            {"controller_param_bindings": ()},
            "implementation and parameter bindings",
        ),
    ],
)
def test_typed_modulation_schema_rejects_malformed_identity(
    factory,
    changes,
    message,
):
    with pytest.raises(ValueError, match=message):
        replace(factory(), **changes)
