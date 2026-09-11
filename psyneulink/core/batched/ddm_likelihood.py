"""Registered continuous first-passage likelihood for one reset DDM primitive.

This provider deliberately does not interpret arbitrary schedules, collapsing
bounds, per-step offsets, or CSI history as the constant-coefficient Wiener law.
"""

from dataclasses import dataclass

import numpy as np

from psyneulink.core.batched.analytic_runtime import parameter_tensor, scalar_inputs, validate_parameter_tensor
from psyneulink.core.batched.kernel_ir import validate_kernel_ir
from psyneulink.core.batched.likelihood_analysis import analyze_likelihood
from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.registry import analyze_composition
from psyneulink.core.batched.wiener import wiener_log_density


def _reject(code, detail):
    raise LikelihoodPlanningError("wiener." + code, detail)


@dataclass(frozen=True)
class WienerWitness:
    component_id: int
    spec_key: str
    parameters: tuple[tuple[str, int], ...]
    choice_column: int
    time_column: int
    guarantee: str = "registered_continuous_wiener_interpretation_with_checked_single_trial_reset"


def derive_wiener_witness(kernel, observations):
    validate_kernel_ir(kernel)
    graph = kernel.graph
    if (len(graph.nodes) != 1 or graph.states or graph.projections or graph.absorbed_projections
            or graph.effective_parameters or graph.modulations or graph.folded_affine_controls
            or graph.metadata.get("schedule_kind") != "static_graph"):
        _reject("structure", "This Wiener tier requires one reset scalar primitive, without projections, retained state, controls or dynamic schedules.")
    node = graph.nodes[0]
    if (node.input_width != 1 or len(graph.inputs) != 1
            or len(graph.scheduler) != 1 or graph.scheduler[0].condition_type != "Always"
            or graph.execution_order != (node.name,)
            or not any(t.condition_type == "AllHaveRun" and t.dependency_component_ids == (node.component_id,) for t in graph.termination)):
        _reject("schedule", "The Wiener primitive must execute to completion exactly once per trial.")
    spec = kernel.op_specs.lookup_spec(node.attrs["spec_key"])
    contract = spec.likelihood_contract
    if contract is None or contract.wiener_readout is None:
        _reject("rule_missing", "This primitive has no registered continuous Wiener interpretation.")
    law, event = contract.wiener_readout, contract.event_readout
    if (event.execution_rule != "one_step_until_finished" or spec.states
            or len(spec.rng) != 1 or len(graph.rng_streams) != 1):
        _reject("reset", "A single reset diffusion event and RNG stream must be established by the frozen contract.")
    by_name = {p.name: p.port_id for p in graph.ports if p.port_id in node.output_port_ids}
    expected = ((law.choice_port, "counting", "value"), (event.output_port, "lebesgue", "event_time"))
    if len(observations) != 2:
        _reject("observation", "The Wiener density requires joint scalar choice and RT observations.")
    columns = []
    for port_name, measure, role in expected:
        matches = [f for f in observations if f.port_id == by_name.get(port_name) and f.component_id == node.component_id]
        if len(matches) != 1:
            _reject("observation", "Choice/RT must bind directly to the registered primitive output ports.")
        f = matches[0]
        if (f.width != 1 or f.measure != measure or f.role != role or not f.score
                or f.recording != "exact" or f.availability != "complete" or f.history_timing != "exact"):
            _reject("observation", "Joint Wiener scoring requires complete exact choice (counting) and RT (Lebesgue); projected history and recording operators are not inferred.")
        columns.append(f.column_start)
    arguments = dict(
        rate=law.rate_parameter, noise=law.noise_parameter, boundary=law.threshold_parameter,
        start=law.starting_value_parameter, offset=law.offset_parameter, collapse=law.collapse_parameter,
        ndt=event.offset_parameter, source_dt=event.step_parameter,
    )
    indices = {p.name: i for i, p in enumerate(kernel.params)}
    try:
        parameters = tuple((role, indices[node.params[name]]) for role, name in arguments.items())
    except KeyError:
        _reject("parameters", "The continuous interpretation has an unbound parameter.")
    for role, index in parameters:
        if role in ("offset", "collapse") and kernel.params[index].default != 0:
            _reject("fixed_bounds", "The fixed-bound continuous interpretation requires zero source offset and collapse; nonzero per-step effects are not silently converted to rates.")
    return WienerWitness(node.component_id, spec.key, parameters, *columns)


@dataclass(frozen=True)
class WienerLikelihoodResult:
    log_factors: np.ndarray
    log_likelihood: np.ndarray
    parameter_names: tuple[str, ...]
    gradient: np.ndarray | None = None
    backend: str = "torch_cpu"
    target: str = "continuous_wiener_joint_choice_rt_density"


@dataclass(frozen=True)
class WienerLikelihoodPlan:
    ir: object
    kernel: object
    bindings: object
    observations: tuple
    witness: WienerWitness

    @property
    def parameter_names(self):
        return tuple(p.name for p in self.ir.params)

    @property
    def active_parameter_names(self):
        return tuple(self.parameter_names[i] for role, i in self.witness.parameters if role not in ("offset", "collapse", "source_dt"))

    def log_prob(self, inputs, data, parameters=None):
        """Differentiable [candidate, trial] joint log density; one subject.

        The source dt is validated but does not enter the continuous law.
        Offset/collapse are restricted to zero, not continuous fitting axes.
        Nonpositive decision times have zero density and return -inf.
        """
        import torch

        if self.witness != derive_wiener_witness(self.kernel, self.observations):
            _reject("witness", "Wiener witness does not match the frozen source contract.")
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 2 or not len(data) or not np.isfinite(data).all():
            raise ValueError("data must be a finite [trial, 2] choice/RT array in ObservationSpec order.")
        p = validate_parameter_tensor(self.ir, parameters)
        args = {role: p[:, index, None] for role, index in self.witness.parameters}
        if torch.any(args["offset"] != 0) or torch.any(args["collapse"] != 0):
            _reject("fixed_bounds", "Runtime proposals must keep source offset and collapse exactly zero.")
        if torch.any(args["ndt"] < 0) or torch.any(args["source_dt"] <= 0):
            raise ValueError("Nondecision time must be nonnegative and source dt positive.")
        external = scalar_inputs(self.ir, self.bindings, inputs, len(data))[self.witness.component_id]
        choice = torch.as_tensor(data[:, self.witness.choice_column].copy())[None, :]
        rt = torch.as_tensor(data[:, self.witness.time_column].copy())[None, :]
        return wiener_log_density(rt - args["ndt"], args["rate"] * external,
                                   args["boundary"], args["noise"], args["start"], choice)

    def _evaluate(self, inputs, data, parameter_sets, include_mask, gradient):
        import torch

        with torch.enable_grad() if gradient else torch.no_grad():
            p = parameter_tensor(self.ir, parameter_sets, requires_grad=gradient)
            factors = self.log_prob(inputs, data, p)
            mask = np.ones(factors.shape[1], dtype=bool) if include_mask is None else np.asarray(include_mask)
            if mask.dtype != np.bool_ or mask.shape != (factors.shape[1],):
                raise ValueError("include_mask must be a boolean vector with one entry per trial.")
            total = factors[:, mask].sum(-1)
            if gradient and not torch.isfinite(total).all():
                _reject("gradient_support", "A scored observation has zero density; its log-likelihood gradient is undefined.")
            grad = torch.autograd.grad(total.sum(), p)[0] if gradient else None
            if grad is not None and not torch.isfinite(grad).all():
                _reject("numeric_range", "Wiener gradient exceeded the supported float64 range.")
        return WienerLikelihoodResult(factors.detach().numpy(), total.detach().numpy(), self.parameter_names,
                                       None if grad is None else grad.detach().numpy())

    def score(self, inputs, data, parameter_sets=None, *, include_mask=None):
        return self._evaluate(inputs, data, parameter_sets, include_mask, False)

    def value_and_grad(self, inputs, data, parameter_sets=None, *, include_mask=None):
        return self._evaluate(inputs, data, parameter_sets, include_mask, True)


def compile_wiener_likelihood(composition, observations, *, ignored_control_nodes=()):
    report, ir, bindings, kernel = analyze_composition(
        composition, outputs=observations.output_ports, ignored_control_nodes=ignored_control_nodes,
    )
    if ir is None or kernel is None:
        _reject("source_ir", "A supported reset source graph is required: " + "; ".join(report.unsupported_reasons))
    diagnosis = analyze_likelihood(report, ir, kernel, bindings, observations)
    if diagnosis.factorization_status != "eligible" or diagnosis.history_kind != "independent_trials":
        _reject("factorization", "Independent trials were not established from source effects and resets.")
    witness = derive_wiener_witness(kernel, diagnosis.observations)
    return WienerLikelihoodPlan(ir, kernel, bindings, diagnosis.observations, witness)
