from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class TritonOpError(RuntimeError):
    """Raised when a component-owned Triton helper cannot be emitted."""


@dataclass(frozen=True)
class TritonOpTemplate:
    """Inspectable Triton helper source captured from a Python function.

    The template stores only source and signature metadata. It intentionally
    does not import Triton or hold target-specific runtime objects until the
    generated kernel module is compiled.
    """

    name: str
    arg_names: tuple[str, ...]
    source: str
    constexpr: tuple[str, ...] = ()


@dataclass(frozen=True)
class TritonOpCall:
    """A bound call to a component-owned Triton helper."""

    template: TritonOpTemplate
    outputs: tuple[str, ...]
    args: tuple[str, ...]


class TritonEmitContext:
    """Binding context passed to private `_gen_triton_*` component hooks."""

    def __init__(self, emitter):
        self._emitter = emitter

    def helper_name(self, template: TritonOpTemplate) -> str:
        return self._emitter.register_template(template)

    def emit_call(self, call: TritonOpCall) -> None:
        helper_name = self.helper_name(call.template)
        arg_expr = ", ".join(call.args)
        if len(call.outputs) == 0:
            self.line(f"{helper_name}({arg_expr})")
        elif len(call.outputs) == 1:
            self.line(f"{call.outputs[0]} = {helper_name}({arg_expr})")
        else:
            self.line(f"{', '.join(call.outputs)} = {helper_name}({arg_expr})")

    def line(self, text: str = "") -> None:
        self._emitter.builder.line(text)

    def param(self, node_spec, local_name: str) -> str:
        return self._emitter.param_vars[node_spec.params[local_name]]

    def state(self, state_name: str, index: int) -> str:
        return self._emitter.state_vars[(state_name, index)]

    def raw_input_value(self, node_name: str, component_idx: int = 0) -> str:
        return self._emitter.raw_input_value(node_name, component_idx)

    def ddm_random_base(self, node_name: str) -> str:
        return self._emitter.ddm_random_base(node_name)

    def lca_stream_index(self, node_name: str) -> int:
        return self._emitter.lca_stream_index[node_name]

    def emit_trial_random_base_if_needed(self) -> None:
        self._emitter.emit_trial_random_base_if_needed()

    @property
    def block_width(self) -> str:
        return "BLOCK"

    @property
    def seed(self) -> str:
        return "SEED"

    @property
    def max_steps(self) -> str:
        return "MAX_STEPS"

    @property
    def lca_max_steps(self) -> str:
        return "LCA_MAX_STEPS"

    def float_literal(self, value: float) -> str:
        return repr(float(value))

    def zero_vector(self) -> str:
        return "tl.zeros((BLOCK,), dtype=tl.float32)"


def pnl_triton_op(
    function=None,
    *,
    name: str | None = None,
    constexpr: Iterable[str] = (),
):
    """Capture a small helper function as generated `@triton.jit` source.

    The decorated function body may refer to `tl` because the generated kernel
    source imports `triton.language as tl`. Other globals and closures are
    rejected so the emitted helper remains inspectable and self-contained.
    """

    def decorate(func):
        return _template_from_function(func, name=name, constexpr=tuple(constexpr))

    if function is None:
        return decorate
    return decorate(function)


def _template_from_function(func, *, name: str | None, constexpr: tuple[str, ...]) -> TritonOpTemplate:
    if not inspect.isfunction(func):
        raise TritonOpError("@pnl_triton_op can only decorate Python functions.")
    if func.__closure__:
        raise TritonOpError(f"Triton op helper '{func.__name__}' cannot close over values.")

    closure_vars = inspect.getclosurevars(func)
    unsupported_globals = sorted(
        global_name
        for global_name in closure_vars.globals
        if global_name != "tl"
    )
    if closure_vars.nonlocals or unsupported_globals:
        details = ", ".join(
            sorted(closure_vars.nonlocals) + unsupported_globals
        )
        raise TritonOpError(
            f"Triton op helper '{func.__name__}' uses unsupported free variables: {details}."
        )

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as error:
        raise TritonOpError(
            f"Triton op helper '{func.__name__}' must have inspectable Python source."
        ) from error

    module = ast.parse(source)
    function_def = _single_function_def(module, func.__name__)
    arg_names = tuple(arg.arg for arg in function_def.args.args)
    missing_constexpr = tuple(arg for arg in constexpr if arg not in arg_names)
    if missing_constexpr:
        raise TritonOpError(
            f"Triton op helper '{func.__name__}' constexpr args are not in the "
            f"signature: {', '.join(missing_constexpr)}."
        )

    function_def.decorator_list = [
        ast.Attribute(value=ast.Name(id="triton", ctx=ast.Load()), attr="jit", ctx=ast.Load())
    ]
    function_def.name = name or func.__name__
    _annotate_constexpr_args(function_def, constexpr)
    ast.fix_missing_locations(module)

    return TritonOpTemplate(
        name=function_def.name,
        arg_names=arg_names,
        source=ast.unparse(module),
        constexpr=constexpr,
    )


def _single_function_def(module: ast.Module, function_name: str) -> ast.FunctionDef:
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1:
        raise TritonOpError(
            f"Triton op helper '{function_name}' must contain exactly one function definition."
        )
    return function_defs[0]


def _annotate_constexpr_args(function_def: ast.FunctionDef, constexpr: Sequence[str]) -> None:
    constexpr_args = set(constexpr)
    for arg in function_def.args.args:
        if arg.arg in constexpr_args:
            arg.annotation = ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="constexpr",
                ctx=ast.Load(),
            )
