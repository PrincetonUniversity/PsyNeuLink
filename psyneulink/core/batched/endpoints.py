"""Checked event-readout derivation and bounded endpoint reconstruction.

This pass resolves a primitive's active-step counter, not the complete scheduler
history. It does not authorize trial splitting or compute likelihood values.
"""

from dataclasses import dataclass, field

import numpy as np

from psyneulink.core.batched.ir import FP32_EXACT_INTEGER_LIMIT
from psyneulink.core.batched.kernel_ir import validate_kernel_ir
from psyneulink.core.batched.likelihood_ir import (
    EndpointExpression as Expr,
    EndpointWitness,
    ScalarReadoutWitness,
    LikelihoodDiagnostic,
)
from psyneulink.core.batched.observation import ResolvedObservationField
from psyneulink.core.batched.prep import normalize_parameter_sets, prepare_inputs, prepare_parameter_values


class EndpointReconstructionError(ValueError):
    """An observation does not determine one admissible event count."""

    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _reject(code, detail):
    raise EndpointReconstructionError(code, detail)


def derive_endpoint_witness(kernel, observation):
    """Derive a scalar readout from frozen primitive contracts and projections.

    The accepted slice has one event counter, scalar affine operations, known
    inputs, and parameter values. Unknown functions and time-varying held
    parameters are rejected rather than assigned their defaults.
    """
    if observation.role != "event_time" or not observation.condition_history:
        _reject("endpoint.not_conditioning_event", "Endpoint reconstruction requires a conditioning event field.")
    if observation.recording != "exact" or observation.availability != "complete":
        _reject("endpoint.recording_unsupported", "Only complete, exact event observations currently have a reconstruction rule.")
    if observation.history_timing not in ("exact", "ceil_fp32_8ulp"):
        _reject("endpoint.history_policy", "Unknown endpoint history timing policy.")
    value = derive_scalar_readout(kernel, observation)
    node = next(node for node in kernel.graph.nodes if node.component_id == value.clock_component_id)
    readout = kernel.op_specs.lookup_spec(value.clock_spec_key).likelihood_contract.event_readout
    step_id = next(parameter.parameter_id for parameter in kernel.params if parameter.name == node.params[readout.step_parameter])
    return EndpointWitness(observation, value.clock_component_id, value.clock_spec_key,
                           readout.counter_state, readout.minimum_count, step_id, value.expression,
                           value.component_ids, value.projection_ids, value.parameter_ids,
                           guarantee=("registered_readout_with_runtime_roundoff_guard" if observation.history_timing == "exact"
                                      else "declared_affine_ceiling_history_projection"))


def derive_scalar_readout(kernel, observation, *, primitive_ports=()):
    """Shared scalar affine/publication checker with optional primitive leaves.

    The caller must authenticate primitive_ports against its sampler witness.
    Without those leaves this derives the registered active-count readout.
    """
    validate_kernel_ir(kernel)
    graph = kernel.graph
    outputs = {item.port_id: item for item in graph.outputs}
    if observation.port_id not in outputs:
        _reject("endpoint.output_missing", "The observation is not a frozen graph output.")
    output = outputs[observation.port_id]
    if observation.component_id != output.component_id or observation.width != output.width:
        _reject("endpoint.output_identity", "Observed component/port identity does not match the graph.")
    nodes = {node.component_id: node for node in graph.nodes}
    ports = {port.port_id: port for port in graph.ports}
    parameters = {parameter.name: parameter for parameter in kernel.params}
    schedule = {item.component_id: item for item in graph.scheduler}
    used_nodes, used_projections, used_params = set(), set(), set()
    clocks = {}
    visiting = set()
    memo = {}
    dynamic = graph.fusion_kind == "coevolving_graph"
    if not dynamic and graph.metadata.get("schedule_kind") not in (None, "static_graph"):
        _reject("endpoint.schedule_unsupported", "Event readout replay for this schedule tier is not implemented.")

    def parameter(node, argument):
        try:
            spec = parameters[node.params[argument]]
        except KeyError:
            _reject("endpoint.parameter_missing", "A readout contract parameter has no frozen binding.")
        if any(
            item.target_component_id == node.component_id and item.target_parameter == argument
            for item in (*graph.effective_parameters, *graph.folded_affine_controls)
        ):
            _reject("endpoint.modulated_parameter", "A readout parameter is scheduler-modulated; its held value must be reconstructed.")
        used_params.add(spec.parameter_id)
        return Expr("parameter", identity=spec.parameter_id)

    def binary(kind, left, right):
        return Expr(kind, (left, right))

    def clock_ids(expr):
        if expr.kind == "count":
            return {expr.identity}
        if expr.kind == "sample":
            return {ports[expr.identity].owner_component_id}
        return set().union(*(clock_ids(arg) for arg in expr.arguments))

    def visit(port_id):
        if port_id in memo:
            return memo[port_id]
        if port_id in visiting:
            _reject("endpoint.cyclic_readout", "A recurrent readout requires state reconstruction.")
        visiting.add(port_id)
        port = ports[port_id]
        node = nodes[port.owner_component_id]
        used_nodes.add(node.component_id)
        if port_id not in node.output_port_ids or port.width != 1:
            _reject("endpoint.scalar_output_required", "Event derivation currently requires scalar output ports.")
        key = node.attrs.get("spec_key")
        spec = kernel.op_specs.lookup_spec(key) if key else None
        contract = None if spec is None else spec.likelihood_contract
        if contract is None:
            _reject("endpoint.contract_missing", "A readout dependency has no registered semantic contract.")
        readout = contract.event_readout
        if port_id in primitive_ports:
            if readout is None or port.name not in {output.port for output in spec.outputs}:
                _reject("endpoint.primitive_leaf_invalid", "A sampled leaf must be a declared output of an event primitive.")
            clocks[node.component_id] = (key, readout, None)
            result = Expr("sample", identity=port_id)
        elif readout is not None and port.name == readout.output_port:
            step = parameter(node, readout.step_parameter)
            offset = parameter(node, readout.offset_parameter)
            clocks[node.component_id] = (key, readout, step.identity)
            result = binary("add", offset, binary("multiply", Expr("count", identity=node.component_id), step))
        elif contract.value_rule == "affine":
            if (
                node.input_width != 1 or node.output_width != 1 or node.combine != "sum"
                or any(name in node.attrs for name in ("clip", "integrator_pre", "noise"))
                or any(state.component_id == node.component_id for state in graph.states)
            ):
                _reject("endpoint.affine_effects_unsupported", "The affine readout has unsupported width, combination, or state effects.")
            terms = []
            if any(item.component_id == node.component_id for item in graph.inputs):
                terms.append(Expr("input", identity=node.component_id))
            for projection in graph.projections:
                if projection.receiver_component_id != node.component_id:
                    continue
                projection_spec = kernel.op_specs.lookup_spec(projection.spec_key)
                projection_contract = projection_spec.likelihood_contract
                if projection_contract is None or projection_contract.value_rule != "dense_projection":
                    _reject("endpoint.projection_rule_missing", "An event projection has no registered dense linear rule.")
                matrix = np.asarray(projection.matrix)
                if matrix.shape != (1, 1) or not np.all(np.isfinite(matrix)):
                    _reject("endpoint.projection_shape", "Event derivation currently requires finite scalar projections.")
                used_projections.add(projection.projection_id)
                coefficient = float(matrix[0, 0])
                if coefficient != 0.0:
                    terms.append(binary("multiply", visit(projection.sender_port_id), Expr("constant", value=coefficient)))
            if not terms:
                _reject("endpoint.input_missing", "No supplied input or projection establishes the readout value.")
            combined = terms[0]
            for term in terms[1:]:
                combined = binary("add", combined, term)
            result = binary("add", binary("multiply", parameter(node, "scale"), binary(
                "add", binary("multiply", combined, parameter(node, "slope")), parameter(node, "intercept"),
            )), parameter(node, "offset"))
            if dynamic:
                predicate = schedule[node.component_id]
                event_ids = clock_ids(result)
                if event_ids:
                    if (
                        predicate.condition_type != "WhenFinished"
                        or set(predicate.dependency_component_ids) != event_ids
                    ):
                        _reject("endpoint.publication_unproven", "An event-dependent readout must execute after its event is finished.")
                elif not (
                    predicate.condition_type == "AtPass" and predicate.attrs.get("pass_index") == 0
                ):
                    _reject("endpoint.prelude_unproven", "Known readout inputs currently require an AtPass(0) prelude.")
                for projection in graph.projections:
                    if projection.receiver_component_id == node.component_id and np.any(projection.matrix):
                        producer = schedule[projection.sender_component_id]
                        if producer.consideration_set_id >= predicate.consideration_set_id:
                            _reject("endpoint.publication_unproven", "Readout dependencies must publish in an earlier consideration set.")
        else:
            _reject("endpoint.readout_rule_missing", "The observed value is not a registered event readout or supported affine transform.")
        visiting.remove(port_id)
        memo[port_id] = result
        return result

    expression = visit(observation.port_id)
    if len(clocks) != 1:
        _reject("endpoint.clock_not_unique", "The observed readout must depend on exactly one event counter.")
    clock_id, (key, readout, step_id) = next(iter(clocks.items()))
    if dynamic and observation.port_id not in primitive_ports:
        termination_ids = {
            component for item in graph.termination
            if item.condition_type == "AllHaveRun" for component in item.dependency_component_ids
        }
        if observation.component_id not in termination_ids or not any(
            item.component_id in termination_ids and item.condition_type == "WhenFinished"
            and item.dependency_component_ids == (clock_id,) for item in graph.scheduler
        ):
            _reject("endpoint.termination_unproven", "Trial termination must require publication after the observed event finishes.")
    return ScalarReadoutWitness(
        observation, clock_id, key, expression,
        tuple(sorted(used_nodes)), tuple(sorted(used_projections)), tuple(sorted(used_params)),
    )


def validate_endpoint_witness(kernel, witness):
    """Translation validation: replay the derivation against the source snapshot.

    A caller-supplied expression/clock/guard cannot authorize reconstruction by
    itself. This checker derives the expected translation from registered rules.
    It is not a formal theorem checker or independent numerical oracle.
    """
    if type(witness) is not EndpointWitness or witness != derive_endpoint_witness(kernel, witness.observation):
        _reject("endpoint.witness_mismatch", "Endpoint witness does not match the frozen source readout.")


def discover_endpoints(kernel, observations):
    witnesses, diagnostics = [], []
    for observation in observations:
        if observation.role != "event_time" or not observation.condition_history:
            continue
        try:
            witnesses.append(derive_endpoint_witness(kernel, observation))
        except EndpointReconstructionError as error:
            diagnostics.append(LikelihoodDiagnostic(error.code, str(error), (observation.component_id,)))
    return tuple(witnesses), tuple(diagnostics)


def _enclose(lower, upper):
    """Outward fp32 rounding, also enclosing an unrounded intermediate for FMA.

    The envelope permits contraction of the declared add/multiply expression;
    it does not cover arbitrary fast-math reassociation. Subnormal arithmetic
    is rejected because backend flush-to-zero behavior is not specified here.
    """
    for value in (lower, upper):
        if np.any((np.abs(value) > 0) & (np.abs(value) < np.finfo(np.float32).tiny)):
            _reject("endpoint.arithmetic_domain", "Subnormal endpoint arithmetic is not supported.")
    with np.errstate(over="ignore", invalid="ignore"):
        lo = np.nextafter(np.asarray(lower, dtype=np.float32), np.float32(-np.inf)).astype(np.float64)
        hi = np.nextafter(np.asarray(upper, dtype=np.float32), np.float32(np.inf)).astype(np.float64)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        _reject("endpoint.arithmetic_domain", "Endpoint arithmetic exceeds the finite fp32 domain.")
    # Preserve exact zero to avoid creating subnormal uncertainty at every
    # identity projection or zero offset.
    exact_zero = (np.asarray(lower) == 0) & (np.asarray(upper) == 0)
    return np.where(exact_zero, 0.0, lo), np.where(exact_zero, 0.0, hi)


def _evaluate_enclosure(expr, parameters, inputs, counts):
    if expr.kind == "count":
        return counts, counts
    if expr.kind == "parameter":
        return parameters[expr.identity], parameters[expr.identity]
    if expr.kind == "input":
        return inputs[expr.identity], inputs[expr.identity]
    if expr.kind == "constant":
        value = float(np.float32(expr.value))
        return value, value
    left = _evaluate_enclosure(expr.arguments[0], parameters, inputs, counts)
    right = _evaluate_enclosure(expr.arguments[1], parameters, inputs, counts)
    if expr.kind == "add":
        return _enclose(left[0] + right[0], left[1] + right[1])
    if expr.kind == "multiply":
        products = np.broadcast_arrays(*(a * b for a in left for b in right))
        return _enclose(np.minimum.reduce(products), np.maximum.reduce(products))
    _reject("endpoint.expression_invalid", "Unknown endpoint arithmetic operation.")


def _evaluate_count_interval(expr, parameters, inputs, first, last):
    """Enclose every point enclosure in an integer interval.

    This is interval evaluation of the original expression, not reassociation
    into an affine formula. The additional flag certifies that *all* intermediate
    arithmetic throughout the range avoids the reference evaluator's rejected
    subnormal domain. A range crossing zero is conservatively uncertified even
    when its discrete counts might skip the problematic values.
    """
    if expr.kind == "count":
        return first, last, True
    if expr.kind in ("parameter", "input", "constant"):
        lower, upper = _evaluate_enclosure(expr, parameters, inputs, first)
        return lower, upper, True
    left = _evaluate_count_interval(expr.arguments[0], parameters, inputs, first, last)
    right = _evaluate_count_interval(expr.arguments[1], parameters, inputs, first, last)
    if expr.kind == "add":
        lower, upper = left[0] + right[0], left[1] + right[1]
    elif expr.kind == "multiply":
        products = np.broadcast_arrays(*(a * b for a in left[:2] for b in right[:2]))
        lower, upper = np.minimum.reduce(products), np.maximum.reduce(products)
    else:
        _reject("endpoint.expression_invalid", "Unknown endpoint arithmetic operation.")
    tiny = np.finfo(np.float32).tiny
    domain = (lower >= tiny) | (upper <= -tiny) | ((lower == 0) & (upper == 0))
    lower, upper = _enclose(lower, upper)
    return lower, upper, domain & left[2] & right[2]


def _invert_count_intervals(expression, parameters, inputs, observed, minimum, maximum):
    """Batched checked inversion; 0 means fallback, -1 no match, -2 ambiguous.

    Each surviving interval is split until its original point enclosure can
    be tested. Disjoint rejected intervals cover all other counts, so accepting
    a singleton also establishes uniqueness. Work is bounded to avoid interval
    dependency/cancellation causing exponential growth. No caller-supplied
    inverse or model-specific name is trusted by this helper.
    """
    size = len(observed)
    result = np.zeros(size, dtype=np.int64)
    if minimum > maximum:
        result.fill(-1)
        return result
    first = np.full(size, minimum, dtype=np.float64)
    last = np.full(size, maximum, dtype=np.float64)
    try:
        lower, upper, certified = _evaluate_count_interval(expression, parameters, inputs, first, last)
    except EndpointReconstructionError as error:
        if error.code != "endpoint.arithmetic_domain":
            raise
        # Preserve reference rejection, including errors at counts far away
        # from the observation. Do not hide them by pruning that count range.
        return result
    certified = np.broadcast_to(certified, (size,))
    result[certified] = -1
    indices = np.flatnonzero(certified & (observed >= lower) & (observed <= upper))
    first, last = first[indices], last[indices]
    hits = np.zeros(size, dtype=np.int64)
    work = size
    while len(indices):
        # An inspection oracle is preferable to unbounded interval subdivision
        # for degenerate readouts. These caps are independent of the count cap.
        if len(indices) > 8 * size or work > 64 * size:
            result[np.unique(indices)] = 0
            break
        lower, upper, _ = _evaluate_count_interval(
            expression, {key: value[indices] for key, value in parameters.items()},
            {key: value[indices] for key, value in inputs.items()}, first, last,
        )
        possible = (observed[indices] >= lower) & (observed[indices] <= upper)
        indices, first, last = indices[possible], first[possible], last[possible]
        leaves = first == last
        if np.any(leaves):
            leaf_ids = indices[leaves]
            # Verify leaves with the unchanged exhaustive evaluator, rather
            # than relying on agreement between two interval implementations.
            lo, hi = _evaluate_enclosure(
                expression, {key: value[leaf_ids] for key, value in parameters.items()},
                {key: value[leaf_ids] for key, value in inputs.items()}, first[leaves],
            )
            accepted = (observed[leaf_ids] >= lo) & (observed[leaf_ids] <= hi)
            accepted_ids = leaf_ids[accepted]
            result[accepted_ids] = first[leaves][accepted].astype(np.int64)
            np.add.at(hits, accepted_ids, 1)
            result[hits > 1] = -2
        split = ~leaves & (hits[indices] < 2)
        indices, first, last = indices[split], first[split], last[split]
        middle = np.floor((first + last) / 2)
        indices = np.concatenate((indices, indices))
        first, last = np.concatenate((first, middle + 1)), np.concatenate((middle, last))
        work += len(indices)
    return result


def _exhaustive_count(expression, parameters, inputs, observed, minimum, maximum, location):
    """Original enumeration kept as the fallback and numerical test oracle."""
    matches = []
    for start in range(minimum, maximum + 1, 256):
        counts = np.arange(start, min(start + 256, maximum + 1), dtype=np.float64)
        lower, upper = _evaluate_enclosure(expression, parameters, inputs, counts)
        matches.extend(counts[(observed >= lower) & (observed <= upper)].astype(np.int64).tolist())
        if len(matches) > 1:
            _reject("endpoint.count_ambiguous", f"Multiple event counts are compatible with {location}.")
    if not matches:
        _reject("endpoint.count_incompatible", f"No event count within the step cap is compatible with {location}.")
    return matches[0]


def _affine_timing_coefficients(expr, parameters, inputs):
    """FP64 offset/slope of the checked affine readout, with FP32 leaves.

    This reassociation defines an explicit compatibility policy; it must not
    replace the operation-preserving exact endpoint enclosure evaluator.
    """
    if expr.kind == "count":
        return 0., 1., True
    if expr.kind in ("parameter", "input", "constant"):
        value, _ = _evaluate_enclosure(expr, parameters, inputs, 0.)
        return value, 0., False
    left = _affine_timing_coefficients(expr.arguments[0], parameters, inputs)
    right = _affine_timing_coefficients(expr.arguments[1], parameters, inputs)
    if expr.kind == "add":
        return left[0] + right[0], left[1] + right[1], left[2] or right[2]
    if expr.kind == "multiply" and not (left[2] and right[2]):
        return left[0] * right[0], left[1] * right[0] + left[0] * right[1], left[2] or right[2]
    _reject("endpoint.ceiling_nonaffine", "Ceiling history timing requires an affine count readout.")


def _ceil_history_counts(expression, parameters, inputs, observed, minimum, maximum, *, allow_zero=False):
    """Declared count projection, not exact conditioning or a recording model."""
    offset, slope, _ = _affine_timing_coefficients(expression, parameters, inputs)
    if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(slope)) or not np.all(np.asarray(slope) > 0):
        _reject("endpoint.ceiling_direction", "Ceiling history timing requires a finite strictly increasing affine readout.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = np.maximum((observed - offset) / slope, 0.)
        nearest = np.rint(ratio)
        reconstructed = offset + nearest * slope
        fp32 = np.asarray(reconstructed, dtype=np.float32)
        spacing = np.abs(np.spacing(fp32).astype(np.float64))
        snap = np.abs(observed - reconstructed) <= 8. * spacing
        projected = np.where(snap, nearest, np.ceil(ratio))
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(fp32)):
        _reject("endpoint.arithmetic_domain", "Ceiling projection exceeds the finite arithmetic domain.")
    if np.any((projected < minimum) & ((projected != 0) if allow_zero else True)):
        _reject("endpoint.projected_count_below_minimum", "Ceiling timing produced a zero-step history; this replay tier requires at least one event step.")
    if np.any(projected > maximum):
        _reject("endpoint.projected_count_above_cap", "Ceiling timing exceeds the source event step cap.")
    # The chosen event's original expression must still have a supported
    # finite readout; projected observations themselves need not lie on it.
    _evaluate_enclosure(expression, parameters, inputs, projected)
    return projected.astype(np.int64)


@dataclass(frozen=True)
class ObservedEndpointPlan:
    """Batched CPU reconstruction with explicit per-candidate runtime guards.

    Under default exact timing, success establishes a unique count inside the
    bounded roundoff enclosure. Declared ceiling timing instead selects an
    approximate point history and is labeled accordingly in the witness.
    It does not establish nonzero likelihood, observation-law equivalence, a
    complete scheduler history, or a formal floating-point certificate.
    """

    witnesses: tuple[EndpointWitness, ...]
    simulation_plan: object = field(repr=False, compare=False)
    observations: tuple[ResolvedObservationField, ...]

    @property
    def column_count(self):
        return sum(observation.width for observation in self.observations)

    def reconstruct(self, inputs, data, parameter_sets=None, *, method="auto"):
        """Return integer counts shaped [candidate, trial, event] for one subject.

        Default exact timing rejects zero or multiple compatible counts in the
        roundoff envelope. ``auto`` uses checked interval inversion with
        exhaustive fallback; ``exhaustive`` selects enumeration for validation.
        Explicit ``ceil_fp32_8ulp`` timing instead projects an affine readout
        to a nonnegative count (auto only); zero requires the separate checked
        replay extension and is not exact conditioning or a forward event.
        """
        if method not in ("auto", "exhaustive"):
            raise ValueError("Endpoint reconstruction method must be 'auto' or 'exhaustive'.")
        plan = self.simulation_plan
        column, seen = 0, set()
        for observation in self.observations:
            if observation.column_start != column or observation.port_id in seen:
                _reject("endpoint.observation_layout", "Observation columns/ports do not form a unique contiguous layout.")
            column += observation.width
            seen.add(observation.port_id)
        if tuple(witness.observation for witness in self.witnesses) != tuple(
            observation for observation in self.observations
            if observation.role == "event_time" and observation.condition_history
        ):
            _reject("endpoint.witness_mismatch", "Endpoint witnesses do not match the declared observation fields.")
        for witness in self.witnesses:
            validate_endpoint_witness(plan.kernel_ir, witness)
        if not self.witnesses:
            _reject("endpoint.no_events", "No endpoint witnesses were supplied.")
        if not 1 <= plan.ir.max_steps <= FP32_EXACT_INTEGER_LIMIT:
            _reject("endpoint.count_domain", "Event counts must fit the exact fp32 integer domain.")
        observations = np.asarray(data, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != self.column_count or not len(observations):
            _reject("endpoint.data_shape", "Data must have one row per trial and the declared observation columns.")
        rows = normalize_parameter_sets(parameter_sets, plan.ir)
        if not rows:
            _reject("endpoint.empty_candidates", "At least one parameter candidate is required.")
        prepared = prepare_inputs(plan.ir, inputs, parameter_sets=rows, component_bindings=plan.component_bindings)
        first_input = next(iter(prepared.values()))
        if first_input.shape[:2] != (1, len(observations)):
            _reject("endpoint.data_shape", "Endpoint reconstruction currently accepts one contiguous subject with matching trial inputs.")
        buffers, _ = prepare_parameter_values(plan.ir, rows, num_subjects=1, num_trials=len(observations))
        shape = (len(rows), len(observations))
        parameters = {
            spec.parameter_id: np.broadcast_to(buffer[:, None] if buffer.ndim == 1 else buffer[:, 0, :], shape).astype(np.float64).ravel()
            for spec, buffer in zip(plan.ir.params, buffers)
        }
        trial_inputs = {
            item.component_id: np.broadcast_to(np.asarray(prepared[item.node][0]).reshape(-1), shape).astype(np.float64).ravel()
            for item in plan.ir.graph.inputs if item.width == 1
        }
        result = np.empty((*shape, len(self.witnesses)), dtype=np.int64)
        for event, witness in enumerate(self.witnesses):
            observed = np.broadcast_to(observations[:, witness.observation.column_start], shape).ravel()

            def location(index):
                return f"candidate {index // shape[1]}, trial {index % shape[1]}, event {event}"

            def reject_invalid(valid, code, detail):
                bad = np.flatnonzero(~valid)
                if len(bad):
                    _reject(code, f"{detail} at {location(bad[0])}.")

            reject_invalid(np.isfinite(observed), "endpoint.observation_nonfinite", "Missing or nonfinite event")
            relevant = {key: parameters[key] for key in witness.parameter_ids}
            for value in relevant.values():
                reject_invalid(np.isfinite(value), "endpoint.parameter_nonfinite", "Nonfinite readout parameter")
            for value in trial_inputs.values():
                reject_invalid(np.isfinite(value), "endpoint.input_nonfinite", "Nonfinite readout input")
            reject_invalid(parameters[witness.step_parameter_id] > 0, "endpoint.step_nonpositive", "Event step size must be positive")
            if witness.observation.history_timing == "ceil_fp32_8ulp":
                if method != "auto":
                    raise ValueError("Exhaustive exact inversion does not implement ceiling history projection.")
                result[..., event] = _ceil_history_counts(
                    witness.expression, relevant, trial_inputs, observed, witness.minimum_count, plan.ir.max_steps, allow_zero=True,
                ).reshape(shape)
                continue
            counts = (np.zeros(len(observed), dtype=np.int64) if method == "exhaustive" else
                      _invert_count_intervals(witness.expression, relevant, trial_inputs, observed,
                                              witness.minimum_count, plan.ir.max_steps))
            # Reference fallback also supplies the established error messages
            # for no match/ambiguity, without weakening any arithmetic guards.
            for index in np.flatnonzero(counts <= 0):
                counts[index] = _exhaustive_count(
                    witness.expression, {key: float(value[index]) for key, value in relevant.items()},
                    {key: float(value[index]) for key, value in trial_inputs.items()}, observed[index],
                    witness.minimum_count, plan.ir.max_steps, location(index),
                )
            result[..., event] = counts.reshape(shape)
        result.flags.writeable = False
        return result


def compile_observed_endpoints(simulation_plan, observations):
    from psyneulink.core.batched.observation import resolve_observations

    resolved = resolve_observations(
        observations, simulation_plan.ir.graph, simulation_plan.component_bindings,
    )
    witnesses, diagnostics = discover_endpoints(simulation_plan.kernel_ir, resolved)
    if diagnostics:
        first = diagnostics[0]
        _reject(first.code, first.detail)
    if not witnesses:
        _reject("endpoint.no_events", "No reconstructible conditioning events were declared.")
    return ObservedEndpointPlan(witnesses, simulation_plan, resolved)
