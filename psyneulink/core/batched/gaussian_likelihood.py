"""Checked scalar affine-Gaussian likelihood compilation and Torch evaluation.

The compiler propagates coefficients of independent primitive draws, not just
variances. Reconverging paths therefore retain their covariance. This is an
ideal-real interpretation of registered rules, not a density for rounded floats.
"""

from dataclasses import dataclass
import math

import numpy as np

from psyneulink.core.batched.likelihood_analysis import analyze_likelihood, stateless_value_schedule
from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.analytic_runtime import parameter_tensor, scalar_inputs, validate_parameter_tensor
from psyneulink.core.batched.registry import analyze_composition


@dataclass(frozen=True)
class GaussianOperation:
    component_id: int
    kind: str
    parameters: tuple[int, ...]  # canonical parameter column indices
    external_input: bool
    projections: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class GaussianWitness:
    operations: tuple[GaussianOperation, ...]
    output_component_id: int
    guarantee: str = "checked_affine_propagation_of_registered_ideal_gaussian_draws"


def _reject(code, detail):
    raise LikelihoodPlanningError("gaussian." + code, detail)


def derive_gaussian_witness(kernel, observations):
    """Reconstruct the admitted program from immutable source contracts."""
    schedule = stateless_value_schedule(kernel, prefix="gaussian", allow_random=True)
    graph = kernel.graph
    if len(observations) != 1:
        _reject("observation", "This tier requires exactly one scalar observed field, not independent marginal scores for correlated outputs.")
    obs = observations[0]
    if (obs.width != 1 or obs.measure != "lebesgue" or obs.role != "value"
            or obs.recording != "exact" or obs.availability != "complete" or not obs.score):
        _reject("observation", "This tier requires a complete, exact, scored scalar value with Lebesgue measure.")
    for spec in kernel.op_specs.specs_by_key.values():
        if spec.likelihood_contract is None:
            _reject("contract", "All frozen operations require an explicit effect contract.")
    param_indices = {p.name: i for i, p in enumerate(kernel.params)}
    input_ids = {i.component_id for i in graph.inputs}
    operations = []
    available = set()
    random_ids = set()
    for name in graph.execution_order:
        node = graph.node(name)
        if (node.input_width != 1 or node.output_width != 1 or len(node.output_port_ids) != 1
                or len(node.input_port_ids) != 1 or node.combine != "sum"
                or any(key in node.attrs for key in ("clip", "noise", "integrator_pre"))):
            _reject("effects", "Only scalar, sum-combined nodes without clipping, integration, or extra noise are admitted.")
        spec = kernel.op_specs.lookup_spec(node.attrs["spec_key"])
        contract = spec.likelihood_contract
        gaussian = contract.gaussian_readout
        if gaussian is not None:
            if (getattr(spec, "states", ()) or getattr(spec, "trial_states", ())
                    or len(spec.rng) != 1 or spec.rng[0].width != 1):
                _reject("primitive", "Gaussian primitives must draw one fresh scalar without modeled state.")
            port = next(p for p in graph.ports if p.port_id == node.output_port_ids[0])
            if gaussian.output_port is not None and gaussian.output_port != port.name:
                _reject("primitive", "The Gaussian law is not bound to this output port.")
            arguments = (gaussian.mean_parameter, gaussian.standard_deviation_parameter)
            kind = "normal"
            random_ids.add(node.component_id)
        elif contract.value_rule == "affine" and contract.randomness == "none":
            arguments = ("slope", "intercept", "scale", "offset")
            kind = "affine"
        else:
            _reject("rule_missing", "A node is neither a registered scalar Gaussian draw nor an affine operation.")
        projections = []
        for projection in graph.projections:
            if projection.receiver_component_id != node.component_id:
                continue
            pc = kernel.op_specs.lookup_spec(projection.spec_key).likelihood_contract
            matrix = np.asarray(projection.matrix)
            if pc.value_rule != "dense_projection" or matrix.shape != (1, 1) or not np.isfinite(matrix).all():
                _reject("projection", "Only registered finite scalar dense projections are admitted.")
            if (projection.sender_component_id not in available
                    or schedule[projection.sender_component_id].consideration_set_id >= schedule[node.component_id].consideration_set_id):
                _reject("publication", "A producer must publish in an earlier consideration set.")
            projections.append((projection.sender_component_id, float(matrix[0, 0])))
        try:
            params = tuple(param_indices[node.params[arg]] for arg in arguments)
        except KeyError:
            _reject("parameters", "A mathematical rule has an unbound parameter.")
        operations.append(GaussianOperation(node.component_id, kind, params,
                                            node.component_id in input_ids, tuple(projections)))
        available.add(node.component_id)
    if random_ids != {r.component_id for r in graph.rng_streams}:
        _reject("randomness", "Every stochastic source must have a registered Gaussian law.")
    if not random_ids:
        _reject("singular", "A deterministic output has no Lebesgue density without an observation-noise model.")
    if not any(o.port_id == obs.port_id and o.component_id == obs.component_id for o in graph.outputs):
        _reject("observation", "Observation identity is not bound to the frozen graph output.")
    return GaussianWitness(tuple(operations), obs.component_id)


@dataclass(frozen=True)
class GaussianLikelihoodResult:
    log_factors: np.ndarray  # [candidate, trial]
    log_likelihood: np.ndarray  # [candidate]
    parameter_names: tuple[str, ...]
    gradient: np.ndarray | None = None  # [candidate, parameter], full log likelihood
    backend: str = "torch_cpu"
    target: str = "ideal_real_gaussian_density"


@dataclass(frozen=True)
class GaussianLikelihoodPlan:
    ir: object
    kernel: object
    bindings: object
    observations: tuple
    witness: GaussianWitness

    @property
    def parameter_names(self):
        return tuple(p.name for p in self.ir.params)

    def _parameters(self, parameter_sets, *, requires_grad=False):
        return parameter_tensor(self.ir, parameter_sets, requires_grad=requires_grad)

    def log_prob(self, inputs, data, parameters=None):
        """Differentiable [candidate, trial] log densities, CPU float64.

        Tensor parameters use canonical parameter_names order and shape [P] or
        [C,P]. No detach, FP32 parameter packing, sampling, or fitting occurs.
        One subject per call; all trials are independent in this admitted tier.
        """
        import torch

        if self.witness != derive_gaussian_witness(self.kernel, self.observations):
            _reject("witness", "The Gaussian program does not match its frozen source contracts.")
        y = torch.as_tensor(np.asarray(data, dtype=np.float64))
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1 or not len(y) or not torch.isfinite(y).all():
            raise ValueError("data must be finite with shape [trial] or [trial, 1].")
        p = validate_parameter_tensor(self.ir, parameters)
        external = scalar_inputs(self.ir, self.bindings, inputs, len(y))
        zero = torch.zeros((len(p), len(y)), dtype=p.dtype)
        values = {}
        for op in self.witness.operations:
            params = [p[:, i, None] for i in op.parameters]
            if op.kind == "normal":
                mean, sd = params
                if torch.any(sd < 0):
                    _reject("scale", "Gaussian standard deviations must be nonnegative.")
                values[op.component_id] = (mean + zero, {op.component_id: sd + zero})
                continue
            combined = external[op.component_id] + zero if op.external_input else zero
            coefficients = {}
            for source, weight in op.projections:
                mean, noise = values[source]
                combined = combined + weight * mean
                for root, coefficient in noise.items():
                    coefficients[root] = coefficients.get(root, zero) + weight * coefficient
            slope, intercept, scale, offset = params
            gain = scale * slope
            values[op.component_id] = (scale * (combined * slope + intercept) + offset,
                                      {root: gain * coef for root, coef in coefficients.items()})
        mean, coefficients = values[self.witness.output_component_id]
        variance = sum((coef.square() for coef in coefficients.values()), zero)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            _reject("nonfinite", "Gaussian propagation overflowed for a proposed parameter candidate.")
        if torch.any(variance <= 0):
            _reject("singular", "An observed variance is zero: this proposal has no Lebesgue density; no floor or jitter was added.")
        factors = -.5 * (math.log(2 * math.pi) + variance.log() + (y[None, :] - mean).square() / variance)
        if not torch.isfinite(factors).all():
            _reject("nonfinite", "Gaussian log-density evaluation overflowed; no score floor was added.")
        return factors

    def _evaluate(self, inputs, data, parameter_sets, include_mask, gradient):
        import torch

        with torch.enable_grad() if gradient else torch.no_grad():
            p = self._parameters(parameter_sets, requires_grad=gradient)
            factors = self.log_prob(inputs, data, p)
            if include_mask is None:
                total = factors.sum(-1)
            else:
                mask = np.asarray(include_mask)
                if mask.dtype != np.bool_ or mask.shape != (factors.shape[1],):
                    raise ValueError("include_mask must be a boolean vector with one entry per trial.")
                total = factors[:, mask].sum(-1)
            grad = torch.autograd.grad(total.sum(), p)[0].detach().numpy() if gradient else None
        return GaussianLikelihoodResult(factors.detach().numpy(), total.detach().numpy(), self.parameter_names, grad)

    def score(self, inputs, data, parameter_sets=None, *, include_mask=None):
        return self._evaluate(inputs, data, parameter_sets, include_mask, False)

    def value_and_grad(self, inputs, data, parameter_sets=None, *, include_mask=None):
        return self._evaluate(inputs, data, parameter_sets, include_mask, True)


def compile_gaussian_likelihood(composition, observations, *, ignored_control_nodes=()):
    # Structural lowering/preflight does not require an available Triton runtime.
    report, ir, bindings, kernel = analyze_composition(
        composition, outputs=observations.output_ports, ignored_control_nodes=ignored_control_nodes,
    )
    if ir is None or kernel is None:
        _reject("source_ir", "A supported frozen graph is required: " + "; ".join(report.unsupported_reasons))
    diagnosis = analyze_likelihood(report, ir, kernel, bindings, observations)
    if diagnosis.factorization_status != "eligible" or diagnosis.history_kind != "independent_trials":
        _reject("factorization", "Independent trials were not established: " + "; ".join(d.detail for d in diagnosis.diagnostics))
    witness = derive_gaussian_witness(kernel, diagnosis.observations)
    return GaussianLikelihoodPlan(ir, kernel, bindings, diagnosis.observations, witness)
