"""Method selection with explicit process targets, independent of device choice.

No continuous limit or sampling estimator is inferred from a request for an
analytic likelihood. The continuous-time target admits a registered fixed-bound
Wiener series. Reusable coefficient-level PDE kernels live in ``numerical``;
Composition-level numerical admission still requires continuous graph lowering.
"""

from dataclasses import asdict, dataclass

from psyneulink.core.batched.likelihood_ir import LikelihoodDiagnostic


class LikelihoodPlanningError(ValueError):
    """A requested method/target has no checked implementation."""

    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code
        self.diagnostics = (LikelihoodDiagnostic(code, detail),)


@dataclass(frozen=True)
class HistogramEstimatorSpec:
    """Explicit estimator configuration, not an observation recording model."""

    categorical_dims: tuple[int, ...]
    bins: int = 100
    bin_range: tuple | None = None
    smoothing_sigma: float = 0.
    pseudocount: float = 0.
    categorical_cardinalities: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "categorical_dims", tuple(self.categorical_dims))
        if self.bin_range is not None:
            object.__setattr__(self, "bin_range", tuple(tuple(bounds) for bounds in self.bin_range))
        if self.categorical_cardinalities is not None:
            object.__setattr__(self, "categorical_cardinalities", tuple(self.categorical_cardinalities))


@dataclass(frozen=True)
class LikelihoodPlanDescription:
    method: str
    process: str
    backend: str
    evaluator: str
    measure: tuple[str, ...]
    gradient: str
    reason: str
    assumptions: tuple[str, ...]
    approximations: tuple[str, ...]
    estimator: HistogramEstimatorSpec | str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CompiledLikelihoodPlan:
    """Common scoring facade; evaluator-specific inspection remains available.

    ``score`` always exposes log_factors and log_likelihood in its result.
    Additional sampling budgets and diagnostics remain evaluator-specific.
    Analytic value_and_grad uses canonical parameter_names order, not fitting
    coordinates. Existing PEC defaults are not changed by this opt-in API.
    """

    description: LikelihoodPlanDescription
    evaluator: object

    def explain(self):
        return self.description.to_dict()

    def score(self, inputs, data, parameter_sets=None, **kwargs):
        return self.evaluator.score(inputs, data, parameter_sets, **kwargs)

    def value_and_grad(self, inputs, data, parameter_sets=None, **kwargs):
        if self.description.gradient == "unavailable":
            raise LikelihoodPlanningError("likelihood.gradient_unavailable", "This sampling estimator has no registered gradient.")
        return self.evaluator.value_and_grad(inputs, data, parameter_sets, **kwargs)


def compile_likelihood(composition, observations, *, method="auto", process="source",
                       backend="auto", estimator=None, max_steps=None,
                       ignored_control_nodes=()):
    """Compile a supported target without silently changing its probability law.

    source: existing finite-step simulation semantics; requires an explicitly
        selected sampling estimator in this milestone.
    ideal_real: registered real-arithmetic distribution rules, preserving the
        admitted trial schedule. Currently scalar Gaussian/affine DAGs only.
    continuous_time: registered fixed-bound reset Wiener primitive, with ideal
        first-passage timing rather than source endpoint testing.

    backend auto resolves to torch_cpu for analytic or triton_cpu for sampling.
    Explicit incompatible backend choices fail instead of being ignored.
    """
    from psyneulink.core.batched.compiler import BatchedCompositionCompiler
    from psyneulink.core.batched.observation import ObservationSpec

    if type(observations) is not ObservationSpec:
        raise TypeError("observations must be an ObservationSpec.")
    if method not in ("auto", "analytic", "numerical", "sampling"):
        raise ValueError("Unknown likelihood method.")
    if process not in ("source", "ideal_real", "continuous_time"):
        raise ValueError("Unknown likelihood process target.")
    if method == "numerical":
        raise LikelihoodPlanningError("likelihood.numerical_not_registered", "No numerical PDE likelihood provider is registered yet; the source process was not substituted.")
    if process == "continuous_time":
        if method == "sampling" or estimator is not None:
            raise LikelihoodPlanningError("likelihood.target_estimator_mismatch", "Continuous first-passage scoring does not accept a source sampling estimator.")
        if backend not in ("auto", "torch_cpu"):
            raise LikelihoodPlanningError("likelihood.backend_unsupported", "The Wiener series evaluator currently supports torch_cpu only.")
        if max_steps is not None:
            raise LikelihoodPlanningError("wiener.horizon", "The continuous Wiener law has no simulation step cap; omit max_steps. Censoring requires an explicit observation rule.")
        from psyneulink.core.batched.ddm_likelihood import compile_wiener_likelihood

        evaluator = compile_wiener_likelihood(composition, observations, ignored_control_nodes=ignored_control_nodes)
        description = LikelihoodPlanDescription(
            "analytic", process, "torch_cpu", "fixed_bound_wiener", tuple(f.measure for f in observations.fields),
            "torch_autodiff", "A registered reset diffusion primitive admits constant-drift, fixed-bound continuous first passage.",
            ("Trusted continuous interpretation, not equivalence to finite-step endpoint testing.",
             "Joint choice/RT density; conditioned trial inputs; positive noise and bounds; interior starting state.",
             "Offset and collapse are restricted to zero. Source dt is inactive in this target."),
            ("Float64 evaluation using eight image pairs for scaled time <=0.2 and sixteen eigenmodes otherwise.",
             "Fixed series truncations validated numerically; no formal floating-point or gradient error certificate.",
             "No RT binning, time mesh, likelihood floor, or conditional-on-choice renormalization."),
        )
        return CompiledLikelihoodPlan(description, evaluator)
    if process == "ideal_real":
        if method == "sampling" or estimator is not None:
            raise LikelihoodPlanningError("likelihood.target_estimator_mismatch", "The ideal-real analytic tier does not accept a source sampling estimator.")
        if backend not in ("auto", "torch_cpu"):
            raise LikelihoodPlanningError("likelihood.backend_unsupported", "The analytic Gaussian evaluator currently supports torch_cpu only.")
        from psyneulink.core.batched.gaussian_likelihood import compile_gaussian_likelihood

        evaluator = compile_gaussian_likelihood(composition, observations, ignored_control_nodes=ignored_control_nodes)
        description = LikelihoodPlanDescription(
            "analytic", process, "torch_cpu", "scalar_affine_gaussian", ("lebesgue",),
            "torch_autodiff", "The frozen graph has registered Gaussian draws and scalar affine propagation, with independent trials.",
            ("Trusted primitive distribution/effect contracts, not a formal proof.",
             "One complete scalar observation per trial; conditioned inputs; no retained state or controls.",
             "Positive observed variance is checked for every candidate and trial."),
            ("Ideal independent Gaussian randomness and real arithmetic, not finite-PRNG output masses.",
             "Float64 evaluation; constants use the already frozen graph's values (including FP32 projection matrices)."),
        )
        return CompiledLikelihoodPlan(description, evaluator)
    if method == "analytic":
        raise LikelihoodPlanningError("likelihood.process_mismatch", "A Gaussian density requires explicit process='ideal_real'; it is not the mass of a source simulator float.")
    if estimator is None:
        raise LikelihoodPlanningError("likelihood.estimator_required", "Source likelihood compilation requires an explicit HistogramEstimatorSpec or estimator='empirical_mass'; no estimator is chosen automatically.")
    if backend == "auto":
        backend = "triton_cpu"
    if backend not in ("triton", "triton_cpu"):
        raise LikelihoodPlanningError("likelihood.backend_unsupported", "Source sampling requires triton or triton_cpu.")
    options = dict(backend=backend, max_steps=max_steps, ignored_control_nodes=ignored_control_nodes)
    if type(estimator) is HistogramEstimatorSpec:
        evaluator = BatchedCompositionCompiler.compile_histogram_score(composition, observations, **options, **asdict(estimator))
        evaluator._validate()
        name = "histogram_surrogate"
        approximations = ("Finite Monte Carlo sample; explicit histogram binning, smoothing, pseudocount and existing score floor conventions.",)
    elif type(estimator) is str and estimator == "empirical_mass":
        evaluator = BatchedCompositionCompiler.compile_empirical_mass(composition, observations, **options)
        name = "empirical_mass"
        approximations = ("Finite Monte Carlo estimate of counting-measure mass.",)
    else:
        raise ValueError("estimator must be a HistogramEstimatorSpec or 'empirical_mass'.")
    if any(f.history_timing != "exact" for f in observations.fields):
        approximations += ("Explicit projected point history; not exact conditioning on uncertain recorded times.",)
    description = LikelihoodPlanDescription(
        "sampling", process, backend, name, tuple(f.measure for f in observations.fields),
        "unavailable", "Explicit estimator lowered through the existing checked history/trajectory/sampling compiler.",
        ("Trusted primitive contracts and checked scheduler/history transformation.",), approximations, estimator,
    )
    return CompiledLikelihoodPlan(description, evaluator)
