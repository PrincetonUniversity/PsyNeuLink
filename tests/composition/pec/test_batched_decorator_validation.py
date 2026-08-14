import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    batched_node_op,
    batched_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched import specs as batched_specs
from psyneulink.core.batched.backend.triton.api import (
    TritonOpError,
    pnl_triton_op,
)


pytestmark = pytest.mark.batched


def _remove_function_spec(function_class):
    registered = batched_specs._FUNCTION_SPECS.pop(function_class, None)
    if registered is not None:
        batched_specs._SPECS_BY_KEY.pop(registered.key, None)


def test_batched_op_allows_tl_and_declared_helper_without_triton():
    class ValidatedLinear(pnl.Linear):
        pass

    try:
        @pnl_triton_op(name="_validation_helper")
        def validation_helper(x):
            return tl.exp(x)

        @batched_op(ValidatedLinear, helpers=(validation_helper,))
        def validated_linear(x, slope, intercept):
            return slope * _validation_helper(x) + intercept + tl.abs(x)

        registered = batched_specs.function_spec_for(ValidatedLinear())
        template = registered.triton_template
        assert template.dependencies == (validation_helper,)
        assert "_validation_helper(x)" in template.source
        assert "tl.abs(x)" in template.source
    finally:
        _remove_function_spec(ValidatedLinear)


def test_batched_node_op_allows_tl_without_triton():
    node_name = "Decorator Validation Node"
    try:
        @batched_node_op(node_name)
        def validated_node(x0):
            return tl.sqrt(x0)

        assert validated_node.__name__ == "validated_node"
        assert node_name in batched_specs._INSTANCE_SPECS
    finally:
        unregister_batched_instance_op(node_name)


def test_batched_node_op_rejects_unbound_typo_at_decoration_time():
    node_name = "Decorator Typo Node"

    with pytest.raises(TritonOpError, match=r"unsupported free variables: tll"):
        @batched_node_op(node_name)
        def typo_node(x0):
            return tll.sqrt(x0)

    assert node_name not in batched_specs._INSTANCE_SPECS


def test_batched_op_rejects_undeclared_helper_at_decoration_time():
    class InvalidLinear(pnl.Linear):
        pass

    with pytest.raises(
        TritonOpError,
        match=r"unsupported free variables: _missing_validation_helper",
    ):
        @batched_op(InvalidLinear)
        def invalid_linear(x):
            return _missing_validation_helper(x)

    assert batched_specs.function_spec_for(InvalidLinear()) is None
