from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


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
    # Other helper templates this one calls; the emitter emits them (transitively)
    # ahead of this template so the `@triton.jit` device functions are defined
    # before their callers.
    dependencies: tuple[TritonOpTemplate, ...] = ()


@dataclass(frozen=True)
class TritonOpCall:
    """A bound call to a component-owned Triton helper."""

    template: TritonOpTemplate
    outputs: tuple[str, ...]
    args: tuple[str, ...]


class TritonEmitContext:
    """Binding context passed to custom batched op `triton_emit` callables."""

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

    def component_symbol(self, node_spec) -> str:
        """Lowering-local symbol prefix for component-owned temporary values."""

        return self._emitter.component_symbol(node_spec)

    def raw_input_value(self, node_name: str, component_idx: int = 0) -> str:
        return self._emitter.raw_input_value(node_name, component_idx)

    def rng_base(self, node_name: str) -> str:
        return self._emitter.rng_base(node_name)

    def rng_stream_offset(self, node_name: str, component_idx: int = 0) -> int:
        """Absolute Philox offset of one of a node's RNG streams.

        Add it to `rng_base`; the step index goes in the low bits.  Each stream
        owns a full 2**32 counter space, so this does not depend on any cap.
        """

        return self._emitter.rng_stream_offset(node_name, component_idx)

    def normal_draw(self, node_name: str, step: str) -> str:
        """Emit or reference one lane-local standard-normal draw.

        Dynamic scheduled regions may cache a two-draw Philox result across
        component executions.  Other regions retain the direct one-draw
        lowering.  Component adapters use this API so the RNG optimization is
        owned by the backend rather than embedded in a particular model.
        """

        return self._emitter.normal_draw(node_name, step)

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

    def sampled_effective_parameter(
        self,
        node_spec,
        target_parameter: str,
    ) -> str | None:
        """A typed effective value sampled by the active dynamic member."""

        return self._emitter.sampled_effective_parameter(
            node_spec,
            target_parameter,
        )

    def float_literal(self, value: float) -> str:
        return repr(float(value))

    def zero_vector(self) -> str:
        return "tl.zeros((BLOCK,), dtype=tl.float32)"


_ALLOWED_TEMPLATE_GLOBALS = frozenset({"tl"})


def pnl_triton_op(
    function=None,
    *,
    name: str | None = None,
    constexpr: Iterable[str] = (),
    helpers: Iterable[TritonOpTemplate] = (),
):
    """Capture a small helper function as generated `@triton.jit` source.

    The decorated function body may refer to `tl` because the generated kernel
    source imports `triton.language as tl`. Other globals and closures are
    rejected so the emitted helper remains inspectable and self-contained —
    except other helper templates passed via ``helpers``, which the body may
    call by their Python name (the emitter emits those device functions ahead of
    this one). This lets a shared recurrence live in one ``@triton.jit`` helper
    called by both a run-to-completion loop and a single-step path. Undefined
    names other than ``tl`` and those declared helpers are rejected immediately.
    """

    def decorate(func):
        return _template_from_function(
            func, name=name, constexpr=tuple(constexpr), helpers=tuple(helpers)
        )

    if function is None:
        return decorate
    return decorate(function)


def _template_from_function(
    func, *, name: str | None, constexpr: tuple[str, ...], helpers: tuple = ()
) -> TritonOpTemplate:
    if not inspect.isfunction(func):
        raise TritonOpError("@pnl_triton_op can only decorate Python functions.")
    if func.__closure__:
        raise TritonOpError(f"Triton op helper '{func.__name__}' cannot close over values.")

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as error:
        raise TritonOpError(
            f"Triton op helper '{func.__name__}' must have inspectable Python source."
        ) from error

    module = ast.parse(source)
    function_def = _single_function_def(module, func.__name__)

    allowed_globals = _ALLOWED_TEMPLATE_GLOBALS | {helper.name for helper in helpers}
    closure_vars = inspect.getclosurevars(func)
    unsupported_globals = sorted(
        global_name
        for global_name in closure_vars.globals
        if global_name not in allowed_globals
    )
    # Python 3.13 reports attribute names (e.g. ``exp`` in ``tl.exp``) in
    # ``unbound`` as well as the actual unresolved root (``tl``). Intersect
    # with AST Name loads to distinguish valid attributes from misspelled
    # globals or undeclared helper calls.
    loaded_names = {
        node.id
        for node in ast.walk(function_def)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    unsupported_unbound = sorted(
        name
        for name in closure_vars.unbound & loaded_names
        if name not in allowed_globals
    )
    if closure_vars.nonlocals or unsupported_globals or unsupported_unbound:
        details = ", ".join(
            sorted(closure_vars.nonlocals)
            + unsupported_globals
            + unsupported_unbound
        )
        raise TritonOpError(
            f"Triton op helper '{func.__name__}' uses unsupported free variables: {details}."
        )

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
        dependencies=tuple(helpers),
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
