"""Declarative batched op specs and the researcher-facing registration API.

A *batched op spec* declares, once, everything the batched compiler needs to
lower a PsyNeuLink component: which parameters its kernel reads, which
lane-local state it owns, which RNG streams it draws from, and the Triton
kernel body (which runs compiled on GPU and interpreted on CPU — there is no
separate numpy implementation).  Specs are registered per exact component
class; subclasses are intentionally **not** inherited because PsyNeuLink
subclasses routinely change semantics (for example ``LCAMechanism`` is a
``TransferMechanism`` subclass).

The primary registration surface is the :func:`batched_op` decorator, which
introspects the decorated body's signature and binds argument names against
the component's PNL ``Parameters`` metadata:

    from psyneulink.core.batched.specs import batched_op

    @batched_op(SoftReLU)
    def soft_relu(x, gain, bias):
        return tl.log(1.0 + tl.exp(gain * (x - bias))) / gain  # tl: triton.language

Reserved argument names carry roles instead of binding to parameters:

- ``x``: the node's (combined) input.
- ``rng``: a ``numpy.random.Generator`` (CPU bodies only).
- ``seed`` / ``rng_base``: the RNG seed and per-lane stream offset (Triton
  bodies only); a CPU ``rng`` argument corresponds to the Triton pair.
- ``max_steps``: the bounded-loop step cap (``tl.constexpr`` on Triton).
- ``lane_mask``: which lanes of the block are in range.  A body that runs a
  bounded loop uses it to exit as soon as every in-range lane has finished,
  instead of always running to ``max_steps``; out-of-range lanes carry default
  parameters and never finish, so they must be excluded from that test.

Every other argument name must resolve to a ``Parameter`` on the component
class (for mechanisms, the required function class is searched first).  Names
that PNL metadata cannot express - aliases, fallbacks, cross-component
lookups - are declared explicitly through the ``bind=`` mapping with
:func:`param`, or by constructing and registering a spec dataclass directly.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from psyneulink.core.batched.backend.triton.api import (
    TritonOpCall,
    TritonOpTemplate,
    pnl_triton_op,
)


class BatchedOpSpecError(ValueError):
    """Raised when a batched op spec cannot be built or registered."""


# Reserved body-argument names and the roles they bind to.
INPUT_ARG = "x"
SEED_ARG = "seed"
RNG_BASE_ARG = "rng_base"
MAX_STEPS_ARG = "max_steps"
LANE_MASK_ARG = "lane_mask"

_TRITON_RESERVED = {INPUT_ARG, SEED_ARG, RNG_BASE_ARG, MAX_STEPS_ARG, LANE_MASK_ARG}


@dataclass(frozen=True)
class ArgBinding:
    """How one body argument is supplied at execution/emission time."""

    role: str  # "input" | "param" | "seed" | "rng_base" | "max_steps" | "lane_mask"
    name: str = ""


@dataclass(frozen=True)
class ParamBinding:
    """Binds one kernel argument to a PNL parameter value.

    ``resolve`` reads the live value from a node at lowering time.  ``scope``
    selects whether the parameter lives on the mechanism or on its primary
    function; ``get`` overrides resolution entirely for irregular bindings.
    """

    arg: str
    pnl_name: str = ""
    fallbacks: tuple[str, ...] = ()
    default: float = 0.0
    scope: str = "function"  # "function" | "mechanism"
    get: Callable[[Any], float] | None = None

    def resolve(self, component) -> float:
        if self.get is not None:
            return float(self.get(component))
        owner = component
        if self.scope == "function":
            # Navigate to the owned function component when given a mechanism;
            # a Function's own `.function` attribute is a method, not a component.
            function = getattr(component, "function", None)
            if function is not None and hasattr(function, "parameters"):
                owner = function
        for name in (self.pnl_name or self.arg,) + self.fallbacks:
            value = component_param(owner, name)
            if value is not None:
                return value
        return float(self.default)


def param(
    pnl_name: str = "",
    *,
    fallback: str | tuple[str, ...] = (),
    default: float = 0.0,
    scope: str = "function",
    get: Callable[[Any], float] | None = None,
) -> ParamBinding:
    """Explicit ``bind=`` override for one body argument."""

    fallbacks = (fallback,) if isinstance(fallback, str) else tuple(fallback)
    return ParamBinding(
        arg="",
        pnl_name=pnl_name,
        fallbacks=fallbacks,
        default=default,
        scope=scope,
        get=get,
    )


@dataclass(frozen=True)
class StateDecl:
    """A lane-local state slot owned by a mechanism op.

    ``width=None`` resolves to the node's primary output width at lowering.
    Declared states persist across trials, which selects the stateful lane
    layout for the containing graph. ``initialize_with_function`` means the
    registered primary elementwise function is applied to ``initial`` using
    the lane's effective parameters; this represents initialized recurrent
    sender values without embedding that function's formula in a backend.
    """

    name: str
    width: int | None = None
    initial: float = 0.0
    initialize_with_function: bool = False


@dataclass(frozen=True)
class RngDecl:
    """A lane-local random stream owned by a mechanism op.

    ``step_extent`` names the kernel step cap that bounds how many draws the
    stream consumes (currently ``"MAX_STEPS"`` or ``"LCA_MAX_STEPS"``).  It no
    longer determines *where* the stream lives: every stream is allocated a
    fixed ``RNG_STREAM_STRIDE`` of Philox counter space, so draws do not shift
    when a cap changes.  It is checked against that stride at launch.
    ``width=None`` resolves to the node's primary output width.
    """

    name: str
    step_extent: str = "MAX_STEPS"
    width: int | None = 1


@dataclass(frozen=True)
class OutputDecl:
    port: str
    width: int = 1


@dataclass(frozen=True)
class ElementwiseFunctionSpec:
    """A stateless elementwise PNL ``Function`` (for example ``Linear``).

    ``body`` is the kernel body ``body(x, *params)``, written against
    ``triton.language`` (``tl``), captured as Triton source and run compiled on
    GPU / interpreted on CPU.
    """

    function_class: type
    params: tuple[ParamBinding, ...]
    body: Callable
    triton_template: TritonOpTemplate | None = None
    key: str = ""


@dataclass(frozen=True)
class PassthroughMechanismSpec:
    """A mechanism that just combines its inputs and applies its function.

    Nodes whose mechanism class has a passthrough spec are supported whenever
    their function class has an :class:`ElementwiseFunctionSpec`.
    """

    mechanism_class: type
    key: str = ""


@dataclass(frozen=True)
class DenseProjectionSpec:
    """A projection lowered to a dense matrix multiply."""

    projection_class: type
    triton_emit: Callable | None = None
    key: str = ""


@dataclass(frozen=True)
class MechanismOpSpec:
    """A mechanism with its own kernel body (integrators, accumulators, ...).

    The op has a single implementation: the Triton kernel body, supplied either
    declaratively as ``triton_template`` (auto-bound from its signature) or via
    the ``triton_emit`` escape hatch for irregular emission.  That kernel runs
    compiled on the GPU and interpreted on the CPU, so there is no separate CPU
    body to maintain.
    """

    mechanism_class: type
    function_class: type | None = None
    display_name: str = ""
    params: tuple[ParamBinding, ...] = ()
    states: tuple[StateDecl, ...] = ()
    # Per-trial state for the co-evolution step form: reset at the start of each
    # trial (unlike ``states``, which persist across trials).  E.g. a DDM's
    # accumulated value / step count / finished flag.
    trial_states: tuple[StateDecl, ...] = ()
    rng: tuple[RngDecl, ...] = ()
    outputs: tuple[OutputDecl, ...] | None = None
    triton_template: TritonOpTemplate | None = None
    triton_bindings: tuple[ArgBinding, ...] = ()
    supports: Callable | None = None
    extract_attrs: Callable | None = None
    triton_emit: Callable | None = None
    # Co-evolution: emit ONE integration step instead of running to completion,
    # so coupled stateful mechanisms can step together in a fused per-step loop.
    # step_emit(ctx, node, inputs, outputs, step_var) updates lane state in place
    # and returns the step's outputs.  finished_output names a per-lane 0/1 flag
    # (set when that lane has terminated) for an op that terminates the loop.
    step_emit: Callable | None = None
    # readout_emit(ctx, node, output_vars) turns the terminator's final trial
    # state into its modeled outputs after the step loop (e.g. DDM decision/RT).
    readout_emit: Callable | None = None
    finished_output: str = ""
    single_node_model_kind: str | None = None
    param_alias_prefixes: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    key: str = ""

    @property
    def persistent_state(self) -> bool:
        return bool(self.states)

    @property
    def has_triton(self) -> bool:
        return self.triton_template is not None or self.triton_emit is not None

    @property
    def can_step(self) -> bool:
        return self.step_emit is not None

    @property
    def is_terminator(self) -> bool:
        return bool(self.finished_output)

    @property
    def label(self) -> str:
        return self.display_name or self.mechanism_class.__name__


_FUNCTION_SPECS: dict[type, ElementwiseFunctionSpec] = {}
_MECHANISM_SPECS: dict[type, MechanismOpSpec] = {}
_PASSTHROUGH_SPECS: dict[type, PassthroughMechanismSpec] = {}
_PROJECTION_SPECS: dict[type, DenseProjectionSpec] = {}
# Instance-level ops, keyed by node *name* (not class).  A node whose class
# already has a class-level spec (e.g. a ProcessingMechanism wrapping a
# UserDefinedFunction) can be given its own op here; the name is the stable,
# researcher-controlled handle (object identity is unusable because PEC rebuilds
# the model each simulation).
_INSTANCE_SPECS: dict[str, MechanismOpSpec] = {}
_SPECS_BY_KEY: dict[str, Any] = {}

_BUILTINS_REGISTERED = False


@dataclass(frozen=True)
class BatchedOpSpecSnapshot:
    """Immutable op implementations resolved for one compiled graph.

    Registration remains process-global so the decorator APIs stay convenient,
    but compilation captures the exact frozen spec objects its graph references.
    Emission can therefore never observe a later replacement or removal from
    the global registry.
    """

    specs_by_key: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "specs_by_key",
            MappingProxyType(dict(self.specs_by_key)),
        )

    def lookup_spec(self, key: str):
        try:
            return self.specs_by_key[key]
        except KeyError as error:
            raise BatchedOpSpecError(
                f"Compiled batched plan has no op spec for key '{key}'."
            ) from error


def snapshot_batched_op_specs(keys) -> BatchedOpSpecSnapshot:
    """Resolve ``keys`` once and return an immutable per-plan registry view."""

    ensure_builtin_specs()
    registry = _SPECS_BY_KEY.copy()
    resolved = {}
    for key in keys:
        try:
            resolved[key] = registry[key]
        except KeyError as error:
            raise BatchedOpSpecError(
                f"No registered batched op spec for key '{key}'."
            ) from error
    return BatchedOpSpecSnapshot(resolved)


def spec_key(component_class: type) -> str:
    return f"{component_class.__module__}.{component_class.__qualname__}"


def register_batched_op(spec):
    """Register a batched op spec for its exact component class."""

    if isinstance(spec, ElementwiseFunctionSpec):
        target, table = spec.function_class, _FUNCTION_SPECS
    elif isinstance(spec, MechanismOpSpec):
        target, table = spec.mechanism_class, _MECHANISM_SPECS
    elif isinstance(spec, PassthroughMechanismSpec):
        target, table = spec.mechanism_class, _PASSTHROUGH_SPECS
    elif isinstance(spec, DenseProjectionSpec):
        target, table = spec.projection_class, _PROJECTION_SPECS
    else:
        raise BatchedOpSpecError(f"Unknown batched op spec type '{type(spec).__name__}'.")

    spec = replace(spec, key=spec_key(target))
    table[target] = spec
    _SPECS_BY_KEY[spec.key] = spec
    return spec


def register_batched_instance_op(node_name: str, spec: MechanismOpSpec) -> MechanismOpSpec:
    """Register a :class:`MechanismOpSpec` for a single node, keyed by its name.

    Unlike :func:`register_batched_op` (keyed by exact component class), this
    binds the op to one specific node instance, so a node whose class already
    has a class-level spec (e.g. a ``ProcessingMechanism`` wrapping a UDF) can be
    given its own kernel.  The spec is kept out of ``_MECHANISM_SPECS`` so it
    never affects other nodes of the same class.
    """

    spec = replace(spec, key=f"instance:{node_name}")
    _INSTANCE_SPECS[node_name] = spec
    _SPECS_BY_KEY[spec.key] = spec
    return spec


def unregister_batched_instance_op(node_name: str) -> None:
    """Remove an instance-level op (and its key); a no-op if not registered."""

    removed = _INSTANCE_SPECS.pop(node_name, None)
    if removed is not None:
        _SPECS_BY_KEY.pop(removed.key, None)


def function_spec_for(function) -> ElementwiseFunctionSpec | None:
    return _FUNCTION_SPECS.get(type(function))


def mechanism_spec_for(node) -> MechanismOpSpec | None:
    # An instance-level op (keyed by node name) takes precedence over the
    # node's class-level spec, so a single node can override its class default.
    # Rebuilding a model in one process suffixes duplicate names ("Foo" ->
    # "Foo-1"), so match the unsuffixed name too (as parameter aliases do).
    name = getattr(node, "name", None)
    if name is not None:
        instance_spec = _INSTANCE_SPECS.get(name) or _INSTANCE_SPECS.get(_unsuffixed_name(name))
        if instance_spec is not None:
            return instance_spec
    return _MECHANISM_SPECS.get(type(node))


def _unsuffixed_name(name: str) -> str:
    return re.sub(r"-\d+$", "", name)


def passthrough_spec_for(node) -> PassthroughMechanismSpec | None:
    return _PASSTHROUGH_SPECS.get(type(node))


def projection_spec_for(projection) -> DenseProjectionSpec | None:
    return _PROJECTION_SPECS.get(type(projection))


def lookup_spec(key: str):
    try:
        return _SPECS_BY_KEY[key]
    except KeyError as error:
        raise BatchedOpSpecError(f"No registered batched op spec for key '{key}'.") from error


def registered_specs() -> tuple:
    return tuple(_SPECS_BY_KEY.values())


def ensure_builtin_specs() -> None:
    """Register the built-in component specs (idempotent)."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    import psyneulink.core.batched.components  # noqa: F401  (registers on import)

    _BUILTINS_REGISTERED = True


def batched_op(
    component_class: type,
    *,
    function: type | None = None,
    outputs=None,
    bind: Mapping[str, ParamBinding] | None = None,
    constexpr: tuple[str, ...] = (),
    display_name: str | None = None,
    single_node_model_kind: str | None = None,
    param_alias_prefixes: tuple[str, ...] = (),
    diagnostics: tuple[str, ...] = (),
    step_emit: Callable | None = None,
    readout_emit: Callable | None = None,
    trial_states: tuple[StateDecl, ...] = (),
    finished_output: str = "",
    helpers: tuple = (),
):
    """Register a batched op for ``component_class`` from its kernel body.

    The decorated function is the op's single kernel body — written against
    ``triton.language`` (``tl``), captured as inspectable Triton source, and run
    compiled on the GPU and interpreted on the CPU.  Its
    signature is introspected: reserved names (``x``, ``seed``, ``rng_base``,
    ``max_steps``) bind to execution roles, and every other argument must
    resolve to a ``Parameter`` on the component.  For mechanism ops,
    ``function`` names the required function class and is searched first; when a
    name exists on both the function and the mechanism, the function parameter
    wins.

    Stateful mechanisms (lane-local state, custom RNG/termination) that the
    declarative form cannot express register a :class:`MechanismOpSpec` with a
    ``triton_emit`` callable directly via :func:`register_batched_op`.

    ``diagnostics`` names trailing return values the body yields *after* its
    graph outputs (e.g. a bounded integrator returning a ``"truncated"`` flag
    when it hit ``max_steps`` without reaching threshold).  They are not graph
    outputs; the compiler routes them to a separate per-lane diagnostic buffer
    (a ``StoreFlag`` op) so the runtime can surface truncation without
    perturbing the modelled outputs.
    """

    def decorate(body):
        if not inspect.isfunction(body):
            raise BatchedOpSpecError("@batched_op can only decorate Python functions.")
        if _is_pnl_function_class(component_class):
            if constexpr:
                raise BatchedOpSpecError("Elementwise batched ops do not take constexpr arguments.")
            _register_function_op(
                component_class,
                body,
                bind or {},
                helpers=tuple(helpers),
            )
        else:
            _register_mechanism_op(
                component_class,
                body,
                function_class=function,
                outputs=outputs,
                bind=bind or {},
                constexpr=tuple(constexpr),
                display_name=display_name,
                single_node_model_kind=single_node_model_kind,
                param_alias_prefixes=param_alias_prefixes,
                diagnostics=tuple(diagnostics),
                step_emit=step_emit,
                readout_emit=readout_emit,
                trial_states=tuple(trial_states),
                finished_output=finished_output,
                helpers=tuple(helpers),
            )
        return body

    return decorate


def batched_node_op(
    node_name: str,
    *,
    outputs=None,
    constexpr: tuple[str, ...] = (),
):
    """Register a batched op for a single node, supplying a whole-input-vector body.

    Unlike :func:`batched_op` (keyed by component/function class and applied
    element-wise), this binds the op to the node named ``node_name`` and gives
    the body its node's **entire combined input vector** — one positional
    argument per input component — so it can compute reductions a class-level
    elementwise op cannot.  This is how a researcher maps a model-specific
    function (for example a ``UserDefinedFunction`` drift rate) onto the batched
    compiler without touching PNL core classes::

        from psyneulink.core.batched import batched_node_op  # body uses tl

        @batched_node_op("Drift Rate Value")
        def drift_rate(x0, x1, x2, x3, x4, x5, x6):
            # arbitrary tl arithmetic over the 7 input components -> one scalar
            ...

    The body is written against ``triton.language`` (``tl``), captured as
    inspectable Triton source (closures/globals are rejected), and run compiled
    on GPU / interpreted on CPU.  Every positional argument is an input
    component; the count must equal the node's combined input width.  ``outputs``
    defaults to the node's own output width (the body returns that many values).

    (Binding extra ``tl`` arguments to node ``Parameters`` or RNG streams is not
    supported yet — input components only.)
    """

    def decorate(body):
        if not inspect.isfunction(body):
            raise BatchedOpSpecError("@batched_node_op can only decorate Python functions.")
        arg_names = _signature_args(body)
        reserved = [name for name in arg_names if name in _TRITON_RESERVED]
        if reserved:
            raise BatchedOpSpecError(
                f"Instance batched op for '{node_name}' takes only input components; "
                f"reserved arguments are not allowed: {', '.join(reserved)}."
            )
        arity = len(arg_names)
        template = pnl_triton_op(
            name=f"_pnl_triton_instance_{_safe_ident(node_name)}",
            constexpr=tuple(constexpr),
        )(body)

        def _instance_triton_emit(ctx, node_spec, inputs, output_vars):
            if len(inputs) != arity:
                raise BatchedOpSpecError(
                    f"Instance batched op for '{node_name}' expects {arity} input "
                    f"component(s) but node '{node_spec.name}' has {len(inputs)}."
                )
            ctx.emit_call(
                TritonOpCall(
                    template=template,
                    outputs=tuple(output_vars),
                    args=tuple(inputs),
                )
            )
            return tuple(output_vars)

        register_batched_instance_op(
            node_name,
            MechanismOpSpec(
                mechanism_class=None,
                function_class=None,
                display_name=node_name,
                outputs=_normalize_outputs(outputs),
                triton_emit=_instance_triton_emit,
            ),
        )
        return body

    return decorate


def _safe_ident(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def _register_function_op(function_class, body, bind, *, helpers=()):
    arg_names = _signature_args(body)
    if not arg_names or arg_names[0] != INPUT_ARG:
        raise BatchedOpSpecError(
            f"Elementwise batched op for '{function_class.__name__}' must take "
            f"'{INPUT_ARG}' as its first argument."
        )
    reserved = [name for name in arg_names[1:] if name in _TRITON_RESERVED]
    if reserved:
        raise BatchedOpSpecError(
            f"Elementwise batched op for '{function_class.__name__}' supports only "
            f"(x, parameters...); reserved arguments are not allowed: {', '.join(reserved)}."
        )

    params = tuple(
        _bind_parameter_arg(name, function_class, None, bind, scope="function")
        for name in arg_names[1:]
    )
    template = pnl_triton_op(
        name=f"_pnl_triton_{function_class.__name__.lower()}",
        helpers=helpers,
    )(body)
    register_batched_op(
        ElementwiseFunctionSpec(
            function_class=function_class,
            params=params,
            body=body,
            triton_template=template,
        )
    )


def _register_mechanism_op(
    mechanism_class,
    body,
    *,
    function_class,
    outputs,
    bind,
    constexpr,
    display_name,
    single_node_model_kind,
    param_alias_prefixes,
    diagnostics=(),
    step_emit=None,
    readout_emit=None,
    trial_states=(),
    finished_output="",
    helpers=(),
):
    template = pnl_triton_op(
        name=f"_pnl_triton_{mechanism_class.__name__.lower()}",
        constexpr=constexpr,
        helpers=helpers,
    )(body)
    triton_bindings, params = _bind_mechanism_args(
        template.arg_names, mechanism_class, function_class, bind
    )
    uses_rng = any(binding.role == "seed" for binding in triton_bindings)
    rng = (RngDecl(name="rng", step_extent="MAX_STEPS", width=1),) if uses_rng else ()

    register_batched_op(
        MechanismOpSpec(
            mechanism_class=mechanism_class,
            function_class=function_class,
            display_name=display_name or mechanism_class.__name__,
            params=params,
            rng=rng,
            outputs=_normalize_outputs(outputs),
            triton_template=template,
            triton_bindings=triton_bindings,
            single_node_model_kind=single_node_model_kind,
            param_alias_prefixes=tuple(param_alias_prefixes),
            diagnostics=tuple(diagnostics),
            step_emit=step_emit,
            readout_emit=readout_emit,
            trial_states=tuple(trial_states),
            finished_output=finished_output,
        )
    )


def _bind_mechanism_args(arg_names, mechanism_class, function_class, bind):
    bindings: list[ArgBinding] = []
    params: list[ParamBinding] = []
    for name in arg_names:
        if name == INPUT_ARG:
            bindings.append(ArgBinding(role="input"))
        elif name == MAX_STEPS_ARG:
            bindings.append(ArgBinding(role="max_steps"))
        elif name == SEED_ARG:
            bindings.append(ArgBinding(role="seed"))
        elif name == RNG_BASE_ARG:
            bindings.append(ArgBinding(role="rng_base"))
        elif name == LANE_MASK_ARG:
            bindings.append(ArgBinding(role="lane_mask"))
        elif name in _TRITON_RESERVED:
            raise BatchedOpSpecError(
                f"Batched op for '{mechanism_class.__name__}': reserved argument "
                f"'{name}' is not valid here."
            )
        else:
            binding = _bind_parameter_arg(name, function_class, mechanism_class, bind, scope=None)
            bindings.append(ArgBinding(role="param", name=name))
            params.append(binding)
    return tuple(bindings), tuple(params)


def _bind_parameter_arg(name, function_class, mechanism_class, bind, *, scope):
    override = bind.get(name)
    if override is not None:
        if not isinstance(override, ParamBinding):
            raise BatchedOpSpecError(
                f"bind['{name}'] must be a ParamBinding (use specs.param(...))."
            )
        return replace(override, arg=name)

    on_function = function_class is not None and _class_has_parameter(function_class, name)
    on_mechanism = mechanism_class is not None and _class_has_parameter(mechanism_class, name)
    if scope == "function" or on_function:
        owner_class, owner_scope, found = function_class, "function", on_function
    else:
        owner_class, owner_scope, found = mechanism_class, "mechanism", on_mechanism

    if not found:
        available = sorted(
            set(_class_parameter_names(function_class)) | set(_class_parameter_names(mechanism_class))
        )
        target = (mechanism_class or function_class).__name__
        raise BatchedOpSpecError(
            f"Batched op argument '{name}' does not match a Parameter on "
            f"'{target}'"
            + (f" or its function '{function_class.__name__}'" if mechanism_class and function_class else "")
            + f". Available parameters: {', '.join(available) or '<none>'}. "
            "Use bind={...} with specs.param(...) for irregular bindings."
        )

    return ParamBinding(
        arg=name,
        pnl_name=name,
        default=_class_default(owner_class, name),
        scope=owner_scope,
    )


def component_param(component, name: str) -> float | None:
    """Read a scalar parameter value from a live component, or ``None``.

    Resolution order matches the live-value semantics the batched compiler
    has always used: ``parameters.<name>.get(None)``, then
    ``defaults.<name>``, then a plain attribute.
    """

    if component is None:
        return None

    parameters = getattr(component, "parameters", None)
    if parameters is not None and hasattr(parameters, name):
        parameter = getattr(parameters, name)
        for getter in ("get", "_get"):
            if hasattr(parameter, getter):
                try:
                    return _as_float(getattr(parameter, getter)(None))
                except Exception:
                    pass

    defaults = getattr(component, "defaults", None)
    if defaults is not None and hasattr(defaults, name):
        try:
            return _as_float(getattr(defaults, name))
        except Exception:
            pass

    if hasattr(component, name):
        try:
            return _as_float(getattr(component, name))
        except Exception:
            pass

    return None


def resolve_component_param(component, name: str, default: float) -> float:
    value = component_param(component, name)
    return float(default) if value is None else value


def _class_has_parameter(component_class, name: str) -> bool:
    try:
        return hasattr(component_class.parameters, name)
    except Exception:
        return False


def _class_parameter_names(component_class) -> tuple[str, ...]:
    if component_class is None:
        return ()
    try:
        return tuple(parameter.name for parameter in component_class.parameters)
    except Exception:
        return ()


def _class_default(component_class, name: str, fallback: float = 0.0) -> float:
    for source_name in ("class_defaults", "defaults"):
        source = getattr(component_class, source_name, None)
        if source is not None and hasattr(source, name):
            try:
                return _as_float(getattr(source, name))
            except Exception:
                pass
    try:
        return _as_float(getattr(component_class.parameters, name).default_value)
    except Exception:
        return float(fallback)


def _normalize_outputs(outputs) -> tuple[OutputDecl, ...] | None:
    if outputs is None:
        return None
    normalized = []
    for output in outputs:
        if isinstance(output, OutputDecl):
            normalized.append(output)
        else:
            port, width = output
            normalized.append(OutputDecl(port=str(port), width=int(width)))
    return tuple(normalized)


def _signature_args(body) -> tuple[str, ...]:
    signature = inspect.signature(body)
    for parameter in signature.parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise BatchedOpSpecError(
                f"Batched op body '{body.__name__}' must use plain positional "
                f"arguments; got '{parameter}'."
            )
    return tuple(signature.parameters)


def _is_pnl_function_class(component_class) -> bool:
    from psyneulink.core.components.functions.function import Function_Base

    return isinstance(component_class, type) and issubclass(component_class, Function_Base)


def _as_float(value) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) == 0:
        return 0.0
    return float(array[0])
