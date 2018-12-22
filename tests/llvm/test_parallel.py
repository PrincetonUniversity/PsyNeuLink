import pytest
import psyneulink.core.llvm as pnlvm

import numpy as np
import psyneulink.core.components.functions.function as Function
import psyneulink.core.components.functions.objectivefunctions as Functions
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.scheduling.scheduler import Scheduler
import psyneulink.core.globals.keywords as kw

SIZE=10
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = [np.random.rand(SIZE) + Function.EPSILON, np.random.rand(SIZE) + Function.EPSILON]
v1 = test_var[0]
v2 = test_var[1]
expected = np.linalg.norm(v1 - v2)

@pytest.mark.cuda
@pytest.mark.llvm
@pytest.mark.parallel
@pytest.mark.function
@pytest.mark.distance_function
@pytest.mark.benchmark(group="DistanceFunction parallel")
@pytest.mark.parametrize("executions", [1,5,100])
def test_ptx_cuda_parallel(benchmark, executions):
    f = Functions.Distance(default_variable=test_var, metric=kw.EUCLIDEAN)
    e = pnlvm.execution.FuncExecution(f, [None for _ in range(executions)])
    res = benchmark(e.cuda_execute, [test_var for _ in range(executions)])
    assert np.allclose(res, [expected for _ in range(executions)])
    assert executions == 1 or len(res) == executions

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.mechanism
@pytest.mark.parallel
@pytest.mark.transfer_mechanism
@pytest.mark.benchmark(group="TransferMechanism parallel")
@pytest.mark.parametrize("executions", [1,5,100])
def test_transfer_mech_parallel(benchmark, executions):
    variable = [0 for _ in range(SIZE)]
    T = TransferMechanism(
        name='T',
        default_variable=variable,
        integration_rate=1.0,
        noise=-2.0,
        integrator_mode=True
    )
    var = [[10.0 for _ in range(SIZE)] for _ in range(executions)]
    expected = [[8.0 for i in range(SIZE)]]
    e = pnlvm.execution.MechExecution(T, [None for _ in range(executions)])
    res = benchmark(e.cuda_execute, var)
    if executions > 1:
        expected = [expected for _ in range(executions)]

    assert np.allclose(res, expected)
    assert len(res) == executions


@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.parallel
@pytest.mark.nested
@pytest.mark.composition
@pytest.mark.benchmark(group="TransferMechanism nested composition parallel")
@pytest.mark.parametrize("executions", [1,5,100])
def test_nested_transfer_mechanism_composition_parallel(benchmark, executions):

    # mechanisms
    A = ProcessingMechanism(name="A",
                            function=AdaptiveIntegrator(rate=0.1))
    B = ProcessingMechanism(name="B",
                            function=Logistic)

    inner_comp = Composition(name="inner_comp")
    inner_comp.add_linear_processing_pathway([A, B])
    inner_comp._analyze_graph()
    sched = Scheduler(composition=inner_comp)

    outer_comp = Composition(name="outer_comp")
    outer_comp.add_c_node(inner_comp)

    outer_comp._analyze_graph()
    sched = Scheduler(composition=outer_comp)

    e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
    # The input dict should assign inputs origin nodes (inner_comp in this case)
    var = {inner_comp: [[1.0]]}
    expected = [[0.52497918747894]]
    if executions > 1:
        var = [var for _ in range(executions)]
    e.cuda_execute(var)
    res = e.extract_node_output(outer_comp.output_CIM)
    benchmark(e.cuda_execute, var)
    assert np.allclose(res, [expected for _ in range(executions)])
    assert len(res) == executions

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.parallel
@pytest.mark.composition
@pytest.mark.benchmark(group="TransferMechanism nested composition parallel run")
@pytest.mark.parametrize("executions", [1,5,100])
def test_nested_transfer_mechanism_composition_run_parallel(benchmark, executions):

    # mechanisms
    A = ProcessingMechanism(name="A",
                            function=AdaptiveIntegrator(rate=0.1))
    B = ProcessingMechanism(name="B",
                            function=Logistic)

    inner_comp = Composition(name="inner_comp")
    inner_comp.add_linear_processing_pathway([A, B])
    inner_comp._analyze_graph()
    sched = Scheduler(composition=inner_comp)

    outer_comp = Composition(name="outer_comp")
    outer_comp.add_c_node(inner_comp)

    outer_comp._analyze_graph()
    sched = Scheduler(composition=outer_comp)

    e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
    # The input dict should assign inputs origin nodes (inner_comp in this case)
    var = {inner_comp: [[[2.0]]]}
    expected = [[[0.549833997312478]]]
    if executions > 1:
        var = [var for _ in range(executions)]
    res = e.cuda_run(var, 1, 1)
    benchmark(e.cuda_run, var, 1, 1)
    assert np.allclose(res, [expected for _ in range(executions)])
    assert len(res) == executions or executions == 1

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.parallel
@pytest.mark.composition
@pytest.mark.benchmark(group="TransferMechanism nested composition parallel run multi")
@pytest.mark.parametrize("executions", [1,5,100])
def test_nested_transfer_mechanism_composition_run_multi_parallel(benchmark, executions):

    # mechanisms
    A = ProcessingMechanism(name="A",
                            function=AdaptiveIntegrator(rate=0.1))
    B = ProcessingMechanism(name="B",
                            function=Logistic)

    inner_comp = Composition(name="inner_comp")
    inner_comp.add_linear_processing_pathway([A, B])
    inner_comp._analyze_graph()
    sched = Scheduler(composition=inner_comp)

    outer_comp = Composition(name="outer_comp")
    outer_comp.add_c_node(inner_comp)

    outer_comp._analyze_graph()
    sched = Scheduler(composition=outer_comp)

    e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
    # The input dict should assign inputs origin nodes (inner_comp in this case)
    var = {inner_comp: [[[2.0]], [[3.0]]]}
    expected = [[[0.549833997312478]], [[0.617747874769249]], [[0.6529428177055896]], [[0.7044959416252289]]]
    if executions > 1:
        var = [var for _ in range(executions)]
    res = e.cuda_run(var, 4, 2)
    benchmark(e.cuda_run, var, 4, 2)
    assert np.allclose(res, [expected for _ in range(executions)])
    assert len(res) == executions or executions == 1
