import composition
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
import time

test_constructor = False
test_add_mechanism = False
test_add_projection = False
test_analyze_graph = True

# Unit tests for each function of the Composition class #######################
# Unit tests for Composition.Composition()
if test_constructor:
    print("Constructor Tests:")

    print("Test 1: No args")
    comp = composition.Composition()
    assert isinstance(comp, composition.Composition)
    print("passed")

    print("Test 2: Second call no args")
    comp_2 = composition.Composition()
    assert isinstance(comp, composition.Composition)
    print("passed")

    print("Test 3: Timing no args")
    count = 10000
    start = time.time()
    for i in range(count):
        comp = composition.Composition()
    end = time.time()
    print("passed in " + str(end-start) + " seconds for " + str(count) + " calls")

# Unit tests for Composition.add_mechanism
if test_add_mechanism:
    print("\n" + "add_mechanism tests:")
    comp = composition.Composition()

    print("Test 1: Basic Test")
    comp.add_mechanism(mechanism())
    print("passed")

    print("Test 2: Second call")
    comp.add_mechanism(mechanism())
    print("passed")

    print("Test 3: Adding same mechanism twice")
    mech = mechanism()
    comp.add_mechanism(mech)
    comp.add_mechanism(mech)
    print("passed")

    print("Test 4: Timing and Stress Test")
    count = 100
    mech_list = []
    for i in range(count):
        mech_list.append(mechanism())
    start = time.time()
    for i in range(count):
        comp.add_mechanism(mech_list[i])
    end = time.time()
    print("passed in " + str(end-start) + " seconds for " + str(count) + " calls")

# Unit tests for Composition.add_projection
if test_add_projection:
    print("\n" + "add_projection tests:")

    print("Test 1: Basic Test")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    print("passed")

    print("Test 2: Second call")
    comp.add_projection(A, MappingProjection(), B)
    print("passed")

    print("Test 3: Adding same projection twice")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    proj = MappingProjection()
    comp.add_projection(A, proj, B)
    comp.add_projection(A, proj, B)
    print("passed")

    print("Test 4: Timing and Stress Test")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    count = 1000
    proj_list = []
    for i in range(count):
        proj_list.append(MappingProjection())
    start = time.time()
    for i in range(count):
        comp.add_projection(A, proj_list[i], B)
    end = time.time()
    print("passed in " + str(end-start) + " seconds for " + str(count) + " calls")

# Unit tests for Composition.analyze_graph
if test_analyze_graph:
    print("\n" + "analyze_graph tests:")

    print("Test 1: Empty Call")
    comp = composition.Composition()
    comp.analyze_graph()
    print("passed")

    print("Test 2: Singleton")
    comp = composition.Composition()
    A = mechanism()
    comp.add_mechanism(A)
    comp.analyze_graph()
    assert A in comp.graph.mechanisms
    assert A in comp.origin_mechanisms
    assert A in comp.terminal_mechanisms
    print("passed")

    print("Test 3: Two independent")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.analyze_graph()
    assert A in comp.origin_mechanisms
    assert B in comp.origin_mechanisms
    assert A in comp.terminal_mechanisms
    assert B in comp.terminal_mechanisms
    print("passed")

    print("Test 4: Two in a row")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    assert A in comp.origin_mechanisms
    assert B not in comp.origin_mechanisms
    assert A not in comp.terminal_mechanisms
    assert B in comp.terminal_mechanisms
    print("passed")

    print("Test 5: Two recursive (A)<->(B)")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.add_projection(B, MappingProjection(), A)
    comp.analyze_graph()
    assert A not in comp.origin_mechanisms
    assert B not in comp.origin_mechanisms
    assert A not in comp.terminal_mechanisms
    assert B not in comp.terminal_mechanisms
    assert A in comp.cycle_mechanisms
    assert B in comp.recurrent_init_mechanisms
    print("passed")

    print("Test 6: Two origins pointing to recursive pair (A)->(B)<->(C)<-(D)")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    C = mechanism()
    D = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_projection(A, MappingProjection(), B)
    comp.add_projection(C, MappingProjection(), B)
    comp.add_projection(B, MappingProjection(), C)
    comp.add_projection(D, MappingProjection(), C)
    comp.analyze_graph()
    assert A in comp.origin_mechanisms
    assert D in comp.origin_mechanisms
    assert B in comp.cycle_mechanisms
    assert C in comp.recurrent_init_mechanisms
    print("passed")