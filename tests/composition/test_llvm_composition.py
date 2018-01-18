from psyneulink.components.functions.function import Linear, SimpleIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.composition import Composition
from psyneulink.scheduling.scheduler import Scheduler

from itertools import product
import numpy as np
import pytest

@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
@pytest.mark.parametrize("llvm", ['Python', 'LLVM'])
def test_run_composition(benchmark, llvm):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: [[1.0]]}, scheduler_processing=sched, bin_execute=(llvm == 'LLVM'))
    assert 25 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
@pytest.mark.parametrize("llvm", ['Python', 'LLVM'])
def test_run_composition_default(benchmark, llvm):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, scheduler_processing=sched, bin_execute=(llvm == 'LLVM'))
    assert 25 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition Pathway 5")
@pytest.mark.parametrize("llvm", ['Python', 'LLVM'])
def test_LPP(benchmark, llvm):

    var = 1.0
    comp = Composition()
    A = TransferMechanism(default_variable=var, name="A", function=Linear(slope=2.0))   # 1 x 2 = 2
    B = TransferMechanism(default_variable=var, name="B", function=Linear(slope=2.0))   # 2 x 2 = 4
    C = TransferMechanism(default_variable=var, name="C", function=Linear(slope=2.0))   # 4 x 2 = 8
    D = TransferMechanism(default_variable=var, name="D", function=Linear(slope=2.0))   # 8 x 2 = 16
    E = TransferMechanism(default_variable=var, name="E", function=Linear(slope=2.0))  # 16 x 2 = 32
    comp.add_linear_processing_pathway([A, B, C, D, E])
    comp._analyze_graph()
    inputs_dict = {A: [var]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.execute, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(llvm=='LLVM'))
    assert 32 == output[0][0]

@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition Vector")
@pytest.mark.parametrize("llvm, vector_length", product(('Python', 'LLVM'), [2**x for x in range(1)]))
def test_run_composition_vector(benchmark, llvm, vector_length):
    var = [1.0 for x in range(vector_length)];
    comp = Composition()
    A = IntegratorMechanism(default_variable=var, function=Linear(slope=5.0))
    B = TransferMechanism(default_variable=var, function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: [var]}, scheduler_processing=sched, bin_execute=(llvm=='LLVM'))
    assert np.allclose([25.0 for x in range(vector_length)], output[0])
