from psyneulink.components.functions.function import Linear, SimpleIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.compositions.composition import Composition
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


@pytest.mark.skip
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


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_5_mechanisms_2_origins_1_terminal(benchmark, mode):
    # A ----> C --
    #              ==> E
    # B ----> D --

    # 5 x 1 = 5 ----> 5 x 5 = 25 --
    #                                25 + 25 = 50  ==> 50 * 5 = 250
    # 5 x 1 = 5 ----> 5 x 5 = 25 --

    comp = Composition()
    A = TransferMechanism(name="A", function=Linear(slope=1.0))
    B = TransferMechanism(name="B", function=Linear(slope=1.0))
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
    comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp._analyze_graph()
    inputs_dict = {A: [5.0],
                   B: [5.0]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert 250 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_terminal(benchmark, mode):
    # C --
    #              ==> E
    # D --

    # 5 x 5 = 25 --
    #                25 + 25 = 50  ==> 50 * 5 = 250
    # 5 x 5 = 25 --

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp._analyze_graph()
    inputs_dict = {C: [5.0],
                   D: [5.0]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert 250 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_terminal_mimo(benchmark, mode):
    # C --
    #              ==> E
    # D --

    # [5, 6] x 5 = [25, 30] --
    #                            30 + 40 = 70  ==> 70 * 5 = 350
    # [7, 8] x 5 = [35, 40] --
    # FIXME: what happened to the first states???

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp._analyze_graph()
    inputs_dict = {C: [5.0, 6.0],
                   D: [7.0, 8.0]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert 350 == output[0][0]
