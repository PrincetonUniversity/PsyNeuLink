"""Hierarchical participant likelihoods use the configured batched compiler."""

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
    pytest.mark.usefixtures("set_threads_to_one"),
]


def test_hierarchical_fit_uses_batched_participant_likelihoods(batched_backend, monkeypatch):
    def reject_legacy_simulation(*args, **kwargs):
        pytest.fail("A batched participant fell back to legacy PEC simulation")

    monkeypatch.setattr(pnl.PECOptimizationFunction, "_run_simulations", reject_legacy_simulation)
    participants = []

    def participant_factory(data, subject_index=None):
        seed = 100 + subject_index
        decision = pnl.DDM(
            function=pnl.DriftDiffusionIntegrator(
                rate=0.7, noise=0.5, threshold=0.2, time_step_size=0.05,
                non_decision_time=0.0, seed=seed,
            ),
            output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        )
        model = pnl.Composition(pathways=decision)
        participant = pnl.ParameterEstimationComposition(
            model=model,
            parameters={("rate", decision): np.linspace(0.1, 1.5, 101)},
            outcome_variables=list(decision.output_ports),
            data=data,
            optimization_function=pnl.PECOptimizationFunction(
                method="differential_evolution", max_iterations=1,
                batched_backend=batched_backend, batched_seed=seed,
                batched_max_steps=256, batched_strict_truncation=True,
                batched_bins=20, batched_bin_range=[(0.0, 2.0)],
                batched_smoothing_sigma=1.0, batched_pseudocount=0.1,
                batched_categorical_cardinalities=[2],
            ),
            num_estimates=32,
            initial_seed=seed,
            same_seed_for_all_parameter_combinations=True,
        )
        inputs = {decision: np.ones((len(data), 1))}
        participants.append((participant, inputs))
        return participant, inputs

    data = pd.DataFrame({
        "subject": ["first"] * 4 + ["second"] * 4,
        "decision": pd.Categorical([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        "response_time": [0.10, 0.20, 0.15, 0.25, 0.15, 0.25, 0.20, 0.30],
    })
    group = pnl.ParameterEstimationComposition(
        data=data,
        fit_method="hierarchical",
        hierarchical_options={
            "subject_id": "subject", "max_iterations": 1,
            "hessian_step": 0.2, "estep_options": {"maxiter": 2},
        },
        distributed_options={"pec_factory": participant_factory},
    )
    result = group.run()

    # This deliberately short fit checks execution and routing, not convergence.
    assert tuple(result.subject_labels) == ("first", "second")
    assert np.isfinite(result.objective)
    assert np.isfinite(result.posterior_variance).all()
    assert len(participants) == 2
    for participant, inputs in participants:
        plan = participant.controller.function._batched_plan
        assert plan.backend == batched_backend
        first = participant.log_likelihood(0.7, inputs=inputs)
        assert participant.log_likelihood(0.7, inputs=inputs) == first
        assert participant.controller.function._batched_plan is plan
