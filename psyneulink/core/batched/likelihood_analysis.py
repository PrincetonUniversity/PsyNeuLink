"""Conservative likelihood diagnosis over an existing simulation snapshot.

No dependencies are removed from simulation IR. In particular, stochastic
termination remains a dependency of repeatedly executed deterministic state.
Candidate classification is not permission to split or execute a likelihood.
"""

from collections import deque

from psyneulink.core.batched.dependency import analyze_axis_dependencies
from psyneulink.core.batched.kernel_ir import validate_kernel_ir
from psyneulink.core.batched.likelihood_ir import (
    LikelihoodCapabilityReport,
    LikelihoodDiagnostic,
    PrimitiveLikelihoodEvidence,
)
from psyneulink.core.batched.observation import ObservationSpec, resolve_observations


def single_pass_value_schedule(kernel, *, prefix):
    """Authenticate the existing single-pass schedule, independently of effects."""
    from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError

    def reject(code, detail):
        raise LikelihoodPlanningError(prefix + "." + code, detail)

    validate_kernel_ir(kernel)
    graph = kernel.graph
    if graph.metadata.get("schedule_kind") != "static_graph":
        reject("schedule", "Value propagation requires the checked single-pass static schedule.")
    ids = {n.component_id for n in graph.nodes}
    if not any(t.condition_type == "AllHaveRun" and set(t.dependency_component_ids) == ids for t in graph.termination):
        reject("termination", "Trial termination must require all admitted nodes to run.")
    schedule = {s.component_id: s for s in graph.scheduler}
    if set(schedule) != ids or any(s.condition_type != "Always" and not (
            s.condition_type in ("EveryNCalls", "AllEveryNCalls") and s.attrs.get("calls") == 1) for s in graph.scheduler):
        reject("schedule", "Only Always and checked (All)EveryNCalls(1) static predicates are admitted.")
    return schedule


def stateless_value_schedule(kernel, *, prefix, allow_random=False):
    """Shared static/stateless admission; laws and edge publication are separate."""
    from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError

    schedule = single_pass_value_schedule(kernel, prefix=prefix)
    graph = kernel.graph
    if (graph.states or graph.effective_parameters or graph.modulations or graph.folded_affine_controls
            or graph.absorbed_projections or graph.finished_values or (graph.rng_streams and not allow_random)):
        raise LikelihoodPlanningError(prefix + ".state", "Retained state, noise, controls, and stopping events require another interpretation.")
    return schedule


def _random_dependency_path(roots, targets, edges):
    """A stable shortest dependency witness, including scheduler/control edges."""
    queue = deque((root, ()) for root in sorted(roots))
    seen = set(roots)
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge.producer_component_id, []).append(edge)
    while queue:
        node, path = queue.popleft()
        if node in targets:
            return node, path
        for edge in adjacency.get(node, ()):
            target = edge.consumer_component_id
            if target not in seen:
                seen.add(target)
                queue.append((target, (*path, edge)))
    return None


def analyze_likelihood(simulation, ir, kernel, bindings, observations):
    """Diagnose structure independently of backend availability.

    Uses only the already frozen primitive snapshot, not the global registry.
    Event readouts can be derived with runtime uniqueness guards. Complete
    scheduler reconstruction and likelihood code generation remain separate.
    """
    if type(observations) is not ObservationSpec:
        raise TypeError("observations must be an ObservationSpec.")
    common = dict(
        simulation=simulation,
        input_policy=observations.input_policy,
        initial_state=observations.initial_state,
        sequence_policy=observations.sequence_policy,
    )
    if ir is None or kernel is None:
        return LikelihoodCapabilityReport(
            **common, observations=(), history_kind="unknown",
            factorization_status="blocked",
            diagnostics=(LikelihoodDiagnostic(
                "likelihood.simulation_ir_unavailable",
                "A supported simulation IR is required; inspect simulation diagnostics.",
            ),),
        )
    validate_kernel_ir(kernel)
    graph = kernel.graph
    if graph is not ir.graph:
        raise ValueError("Likelihood analysis requires the simulation plan's exact graph.")
    resolved = resolve_observations(observations, graph, bindings)
    axis = analyze_axis_dependencies(graph, ir.params)
    contracts = []
    diagnostics = []
    obligations = []
    for key, implementation in sorted(kernel.op_specs.specs_by_key.items()):
        contract = implementation.likelihood_contract
        if contract is None:
            diagnostics.append(LikelihoodDiagnostic(
                "likelihood.effects_undeclared",
                f"Registered implementation {key!r} has no likelihood effect contract.",
            ))
        else:
            contracts.append(PrimitiveLikelihoodEvidence(
                key, contract.version, contract.randomness,
            ))

    reset_ids = {
        state_id for reset in graph.resets
        if reset.condition_type == "AtTrialStart"
        for state_id in reset.state_ids
    }
    retained = tuple(state for state in graph.states if state.state_id not in reset_ids)
    # Held modulation and folded controls survive independently of graph.states.
    held_ids = tuple(sorted({
        item.effective_parameter_id
        for item in (*graph.effective_parameters, *graph.folded_affine_controls)
    }))
    targets = {state.component_id for state in retained}
    # Follow the producing controller, not the stochastic mechanism consuming
    # the held parameter. A DDM using a deterministic threshold does not make
    # the threshold random by dataflow; its stopping duration may do so.
    targets.update(item.controller_component_id for item in graph.modulations)
    targets.update(item.controller_component_id for item in graph.folded_affine_controls)
    held_projections = tuple(
        projection for projection in graph.absorbed_projections
        if projection.kind == "ControlProjection"
    )
    targets.update(item.sender_component_id for item in held_projections)
    common.update(
        observations=resolved,
        retained_state_ids=tuple(sorted(state.state_id for state in retained)),
        held_parameter_ids=held_ids,
        held_projection_ids=tuple(sorted(item.projection_id for item in held_projections)),
        stochastic_component_ids=axis.stochastic_root_component_ids,
        contracts=tuple(contracts),
    )
    if observations.initial_state == "latent":
        diagnostics.append(LikelihoodDiagnostic(
            "likelihood.initial_state_unresolved",
            "Latent initial conditions need a distribution and marginalization rule.",
        ))
    if diagnostics:
        return LikelihoodCapabilityReport(
            **common, history_kind="unknown", factorization_status="blocked",
            diagnostics=tuple(diagnostics),
        )

    from psyneulink.core.batched.endpoints import discover_endpoints

    endpoints, endpoint_diagnostics = discover_endpoints(kernel, resolved)
    common["endpoint_witnesses"] = endpoints
    obligations.extend(endpoint_diagnostics)
    if endpoints:
        obligations.append(LikelihoodDiagnostic(
            "likelihood.endpoint_runtime_guard_required",
            "Registered readout expressions were checked. Each candidate and trial "
            "must still pass its declared timing policy's count-domain and arithmetic guards.",
            tuple(sorted({endpoint.clock_component_id for endpoint in endpoints})),
        ))

    # Factorization and the existence/evaluation of a requested density are
    # separate questions. Do not infer a Lebesgue density from a float output.
    if any(field.measure == "lebesgue" for field in resolved):
        obligations.append(LikelihoodDiagnostic(
            "observation.density_not_established",
            "Absolute continuity of the recorded law has not been established. "
            "Discrete simulator event times cannot silently become continuous densities.",
        ))
    if any(field.recording != "exact" for field in resolved):
        obligations.append(LikelihoodDiagnostic(
            "observation.recording_operator_required",
            "Recording noise, rounding, or censoring needs an explicit observation operator.",
        ))
    if any(field.history_timing != "exact" for field in resolved):
        obligations.append(LikelihoodDiagnostic(
            "observation.projected_history",
            "A declared ceiling policy selects a point history; this is not exact conditioning or marginalization over recorded-time uncertainty.",
        ))
    if any(not field.score for field in resolved):
        obligations.append(LikelihoodDiagnostic(
            "observation.selected_score",
            "Unscored conditioning fields define selected conditional factors, "
            "not the full joint likelihood of all supplied fields.",
        ))

    if not targets:
        return LikelihoodCapabilityReport(
            **common, history_kind="independent_trials", factorization_status="eligible",
            candidate_strategy="independent_trials", obligations=tuple(obligations),
        )

    # Only explore removal of trial-termination control edges. The candidate
    # STILL needs a checked endpoint/scheduler reconstruction pass. Ordinary
    # scheduler, data, and modulation edges remain and can block the candidate.
    nonterminal_edges = tuple(
        edge for edge in axis.edges if edge.kind != "schedule_termination_control"
    )
    residual = _random_dependency_path(
        axis.stochastic_root_component_ids, targets, nonterminal_edges,
    )
    if residual is not None:
        target, path = residual
        return LikelihoodCapabilityReport(
            **common, history_kind="unresolved_stochastic_history",
            factorization_status="blocked", obligations=tuple(obligations),
            diagnostics=(LikelihoodDiagnostic(
                "likelihood.stochastic_history_unresolved",
                "Dependency analysis cannot exclude random influence on retained "
                "state or held controls beyond trial duration. No reconstruction "
                "rule currently resolves this dependency.",
                (target,), path,
            ),),
        )

    timing = _random_dependency_path(axis.stochastic_root_component_ids, targets, axis.edges)
    if timing is not None:
        event_fields = tuple(field for field in resolved if field.role == "event_time")
        usable_events = tuple(
            field for field in event_fields
            if field.condition_history and field.availability == "complete"
            and field.recording == "exact"
        )
        if not usable_events:
            code = (
                "observation.endpoint_ambiguous" if event_fields
                else "observation.endpoint_required"
            )
            return LikelihoodCapabilityReport(
                **common, history_kind="unresolved_stochastic_history",
                factorization_status="blocked", obligations=tuple(obligations),
                diagnostics=(LikelihoodDiagnostic(
                    code,
                    "Stochastic duration changes retained history. Supply a complete "
                    "conditioning event observation; rounded, noisy, censored, or "
                    "missing events need an endpoint reconstruction/inference rule.",
                    (timing[0],), timing[1],
                ),),
            )
        covered_clocks = {endpoint.clock_component_id for endpoint in endpoints}
        # All stochastic roots are required conservatively. A future liveness
        # rule can omit clocks proven irrelevant to the reconstructed history.
        if not set(axis.stochastic_root_component_ids) <= covered_clocks:
            obligations.append(LikelihoodDiagnostic(
                "likelihood.endpoint_reconstruction_required",
                "Bind all relevant stochastic clocks to observed readouts; an "
                "event_time label alone does not resolve an execution endpoint.",
                (timing[0],), timing[1],
            ))
    obligations.extend((
        LikelihoodDiagnostic(
            "likelihood.history_reconstruction_required",
            "Construct and validate history replay, including initialization, "
            "held controls, and all scheduler effects on the terminating pass.",
            tuple(sorted(targets)),
        ),
        LikelihoodDiagnostic(
            "likelihood.trajectory_partition_required",
            "Validate within-trial deterministic boundary paths and all stochastic "
            "region inputs before parallelizing trial evaluations.",
        ),
    ))
    return LikelihoodCapabilityReport(
        **common, history_kind="deterministic_history_candidate",
        factorization_status="candidate", candidate_strategy="deterministic_history",
        obligations=tuple(obligations),
    )
