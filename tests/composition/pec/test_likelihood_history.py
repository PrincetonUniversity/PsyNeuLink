import pytest

import psyneulink as pnl
from psyneulink.core.compositions.likelihoodhistory import analyze_likelihood_history


@pytest.mark.parametrize("reset,expected", [
    (pnl.Never(), True), (pnl.AtTrial(0), True), (pnl.AtTrialStart(), False),
    (pnl.Always(), False), (pnl.AtPass(0), False), (pnl.AtPass(1), True),
    (pnl.All(pnl.AtTrialStart(), pnl.AtTrial(0)), True),
    (pnl.Any(pnl.Never(), pnl.AtTrialStart()), False),
])
def test_persistent_function_detection(reset, expected):
    node = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator, reset_stateful_function_when=reset)
    model = pnl.Composition(pathways=[node])
    report = analyze_likelihood_history(model, [node.output_port])
    assert report.requires_conditioning is expected
    if expected:
        assert node.name in " ".join(report.reasons)


def test_trial_resetting_ddm_remains_independent():
    node = pnl.DDM(function=pnl.DriftDiffusionIntegrator())
    model = pnl.Composition(pathways=[node])
    assert not analyze_likelihood_history(model, [node.output_port]).requires_conditioning


def test_unused_integrator_and_inactive_integrator_do_not_select_filter():
    unused = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator)
    output = pnl.TransferMechanism(integrator_mode=False)
    model = pnl.Composition(nodes=[unused, output])
    assert not analyze_likelihood_history(model, [output.output_port]).requires_conditioning


def test_nested_persistent_ancestor_is_relevant():
    node = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator)
    inner = pnl.Composition(pathways=[node])
    output = pnl.ProcessingMechanism()
    model = pnl.Composition(pathways=[inner, output])
    assert analyze_likelihood_history(model, [output.output_port]).requires_conditioning


def test_stateless_feedback_outputs_are_still_history():
    a, b = pnl.TransferMechanism(), pnl.TransferMechanism()
    model = pnl.Composition(pathways=[a, b])
    model.add_projection(sender=b, receiver=a, feedback=True)
    assert analyze_likelihood_history(model, [b.output_port]).requires_conditioning


def test_rng_alone_does_not_select_filter():
    node = pnl.ProcessingMechanism(function=pnl.NormalDist())
    model = pnl.Composition(pathways=[node])
    assert not analyze_likelihood_history(model, [node.output_port]).requires_conditioning


@pytest.mark.parametrize("nested", [False, True])
def test_conditional_execution_can_hold_a_stateless_random_output(nested):
    node = pnl.ProcessingMechanism(function=pnl.NormalDist())
    gated = pnl.Composition(pathways=[node]) if nested else node
    output = pnl.TransferMechanism()
    model = pnl.Composition(pathways=[gated, output])
    model.scheduler.add_condition(gated, pnl.AtTrial(0))
    report = analyze_likelihood_history(model, [output.output_port])
    assert report.requires_conditioning
    assert any("conditional execution" in reason for reason in report.reasons)


def test_stateful_output_port_is_not_covered_by_mechanism_reset():
    node = pnl.TransferMechanism(output_ports=[{pnl.FUNCTION: pnl.SimpleIntegrator}],
                                 reset_stateful_function_when=pnl.AtTrialStart())
    model = pnl.Composition(pathways=[node])
    report = analyze_likelihood_history(model, [node.output_port])
    assert report.requires_conditioning
    assert any("stateful port" in reason for reason in report.reasons)


def test_hidden_stopping_state_is_relevant_without_an_output_projection():
    hidden = pnl.DDM(function=pnl.DriftDiffusionIntegrator(), reset_stateful_function_when=pnl.Never())
    output = pnl.TransferMechanism()
    model = pnl.Composition(nodes=[hidden, output], termination_processing={pnl.TimeScale.TRIAL: pnl.WhenFinished(hidden)})
    assert analyze_likelihood_history(model, [output.output_port]).requires_conditioning
