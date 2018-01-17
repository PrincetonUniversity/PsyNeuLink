from psyneulink.components.functions.function import Linear, SimpleIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.composition import Composition
from psyneulink.scheduling.scheduler import Scheduler

import numpy as np
import pytest

@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
def test_run_composition(benchmark):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: [1.0]}, scheduler_processing=sched)
    assert 25 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
def test_run_llvm_wrapper_composition(benchmark):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: [1.0]}, scheduler_processing=sched, bin_execute=True)
    assert 25 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
def test_run_composition_default(benchmark):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, scheduler_processing=sched)
    assert 25 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
def test_run_llvm_wrapper_composition_default(benchmark):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, scheduler_processing=sched, bin_execute=True)
    assert 25 == output[0][0]

VECTOR_LENGTH=4

@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition Vector")
def test_run_composition_vector(benchmark):
    var = [1.0 for x in range(VECTOR_LENGTH)];
    comp = Composition()
    A = IntegratorMechanism(default_variable=var, function=Linear(slope=5.0))
    B = TransferMechanism(default_variable=var, function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: var}, scheduler_processing=sched)
    assert np.allclose([25.0 for x in range(VECTOR_LENGTH)], output[0])


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition Vector")
def test_run_llvm_wrapper_composition_vector(benchmark):
    var = [1.0 for x in range(VECTOR_LENGTH)];
    comp = Composition()
    A = IntegratorMechanism(default_variable=var, function=Linear(slope=5.0))
    B = TransferMechanism(default_variable=var, function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: [1.0]}, scheduler_processing=sched, bin_execute=True)
    assert np.allclose([25.0 for x in range(VECTOR_LENGTH)], output[0])
