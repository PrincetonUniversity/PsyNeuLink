import pytest
import psyneulink.core.llvm as pnlvm

import numpy as np
import psyneulink.core.components.functions.function as Function
import psyneulink.core.components.functions.nonstateful.objectivefunctions as Functions
from psyneulink.core.components.functions.stateful.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition
import psyneulink.core.globals.keywords as kw

SIZE=10
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = [np.random.rand(SIZE) + Function.EPSILON, np.random.rand(SIZE) + Function.EPSILON]
v1 = test_var[0]
v2 = test_var[1]
expected = np.linalg.norm(v1 - v2)

@pytest.mark.multirun
@pytest.mark.function
@pytest.mark.distance_function
@pytest.mark.benchmark
@pytest.mark.parametrize("executions", [1, 10, 100])
def test_function(benchmark, executions, func_mode):
    f = Functions.Distance(default_variable=test_var, metric=kw.EUCLIDEAN)
    benchmark.group = "DistanceFunction multirun {}".format(executions)
    var = [test_var for _ in range(executions)] if executions > 1 else test_var
    if func_mode == 'Python':
        e = f.function if executions == 1 else lambda x: [f.function(xi) for xi in x]
    elif func_mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f, [None for _ in range(executions)]).execute
    elif func_mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f, [None for _ in range(executions)]).cuda_execute

    res = benchmark(e, var)
    np.testing.assert_allclose(res, [expected for _ in range(executions)])

@pytest.mark.multirun
@pytest.mark.mechanism
@pytest.mark.transfer_mechanism
@pytest.mark.benchmark
@pytest.mark.parametrize("executions", [1, 10, 100])
def test_mechanism(benchmark, executions, mech_mode):
    benchmark.group = "TransferMechanism multirun {}".format(executions)
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
    if mech_mode == 'Python':
        e = T.execute if executions == 1 else lambda x : [T.execute(xi) for xi in x]
    elif mech_mode == 'LLVM':
        e = pnlvm.execution.MechExecution(T, [None for _ in range(executions)]).execute
    elif mech_mode == 'PTX':
        e = pnlvm.execution.MechExecution(T, [None for _ in range(executions)]).cuda_execute

    if executions > 1:
        expected = [expected for _ in range(executions)]

    res = benchmark(e, var)
    np.testing.assert_allclose(res, expected)


@pytest.mark.multirun
@pytest.mark.nested
@pytest.mark.composition
@pytest.mark.benchmark
@pytest.mark.parametrize("executions", [1, 10, 100])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_nested_composition_execution(benchmark, executions, mode):
    benchmark.group = "Nested Composition execution multirun {}".format(executions)

    # mechanisms
    A = ProcessingMechanism(name="A",
                            function=AdaptiveIntegrator(rate=0.1))
    B = ProcessingMechanism(name="B",
                            function=Logistic)

    inner_comp = Composition(name="inner_comp")
    inner_comp.add_linear_processing_pathway([A, B])
    inner_comp._analyze_graph()

    outer_comp = Composition(name="outer_comp")
    outer_comp.add_node(inner_comp)

    outer_comp._analyze_graph()

    # The input dict should assign inputs origin nodes (inner_comp in this case)
    var = {inner_comp: [[1.0]]}
    expected = [[0.52497918747894]]
    if executions > 1:
        var = [var for _ in range(executions)]

    if mode == 'Python':
        e = outer_comp.execute if executions == 1 else lambda x : [outer_comp.execute(x[i], context=i) for i in range(executions)]
        res = e(var)
        benchmark(e, var)
    elif mode == 'LLVM':
        e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
        e.execute(var)
        res = e.extract_node_output(outer_comp.output_CIM)
        benchmark(e.execute, var)
    elif mode == 'PTX':
        e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
        e.cuda_execute(var)
        res = e.extract_node_output(outer_comp.output_CIM)
        benchmark(e.cuda_execute, var)
    else:
        assert False, "Unknown mode: {}".format(mode)

    expected = [expected for _ in range(executions)] if executions > 1 else expected
    np.testing.assert_allclose(res, expected)


@pytest.mark.multirun
@pytest.mark.nested
@pytest.mark.composition
@pytest.mark.benchmark
@pytest.mark.parametrize("executions", [1, 10, 100])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_nested_composition_run(benchmark, executions, mode):
    benchmark.group = "Nested Composition multirun {}".format(executions)

    # mechanisms
    A = ProcessingMechanism(name="A",
                            function=AdaptiveIntegrator(rate=0.1))
    B = ProcessingMechanism(name="B",
                            function=Logistic)

    inner_comp = Composition(name="inner_comp")
    inner_comp.add_linear_processing_pathway([A, B])
    inner_comp._analyze_graph()

    outer_comp = Composition(name="outer_comp")
    outer_comp.add_node(inner_comp)

    outer_comp._analyze_graph()

    # The input dict should assign inputs origin nodes (inner_comp in this case)
    var = {inner_comp: [[[2.0]]]}
    expected = [[[0.549833997312478]]]
    if executions > 1:
        var = [var for _ in range(executions)]
    if mode == 'Python':
        e = outer_comp.run if executions == 1 else lambda x: [outer_comp.run(x[i], context=i) for i in range(executions)]
        res = e(var)

        # Composition.run returns only the result of the last trail,
        # unlike results for all trials reported by CompExecution.run below
        expected = expected[0]

        benchmark(e, var)
    elif mode == 'LLVM':
        e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
        res = e.run(var, 1, 1)
        benchmark(e.run, var, 1, 1)
    elif mode == 'PTX':
        e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
        res = e.cuda_run(var, 1, 1)
        benchmark(e.cuda_run, var, 1, 1)
    else:
        assert False, "Unknown mode: {}".format(mode)

    expected = [expected for _ in range(executions)] if executions > 1 else expected
    np.testing.assert_allclose(res, expected)


@pytest.mark.multirun
@pytest.mark.nested
@pytest.mark.composition
@pytest.mark.benchmark
@pytest.mark.parametrize("executions", [1, 10, 100])
@pytest.mark.parametrize("mode", [
    'Python',
    pytest.param('LLVM', marks=pytest.mark.llvm),
    pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])
])
def test_nested_composition_run_trials_inputs(benchmark, executions, mode):
    benchmark.group = "Nested Composition mutliple trials/inputs multirun {}".format(executions)

    # mechanisms
    A = ProcessingMechanism(name="A",
                            function=AdaptiveIntegrator(rate=0.1))
    B = ProcessingMechanism(name="B",
                            function=Logistic)

    inner_comp = Composition(name="inner_comp")
    inner_comp.add_linear_processing_pathway([A, B])
    inner_comp._analyze_graph()

    outer_comp = Composition(name="outer_comp")
    outer_comp.add_node(inner_comp)

    outer_comp._analyze_graph()

    # The input dict should assign inputs origin nodes (inner_comp in this case)
    var = {inner_comp: [[[2.0]], [[3.0]]]}
    expected = [[[0.549833997312478]], [[0.617747874769249]], [[0.6529428177055896]], [[0.7044959416252289]]]
    if executions > 1:
        var = [var for _ in range(executions)]
    if mode == 'Python':
        def f(v, num_trials, copy_results=False):
            results = []
            for i in range(executions):
                outer_comp.run(v[i], context=i, num_trials=num_trials)
                if copy_results: # copy the results immediately, otherwise it's empty
                    results.append(outer_comp.results.copy())
            return results[0] if len(results) == 1 else results

        res = f(var, 4, True) if executions > 1 else f([var], 4, True)
        benchmark(f if executions > 1 else outer_comp.run, var, num_trials=4)
    elif mode == 'LLVM':
        e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
        res = e.run(var, 4, 2)
        benchmark(e.run, var, 4, 2)
    elif mode == 'PTX':
        e = pnlvm.execution.CompExecution(outer_comp, [None for _ in range(executions)])
        res = e.cuda_run(var, 4, 2)
        benchmark(e.cuda_run, var, 4, 2)
    else:
        assert False, "Unknown mode: {}".format(mode)

    expected = [expected for _ in range(executions)] if executions > 1 else expected
    np.testing.assert_allclose(res, expected)
