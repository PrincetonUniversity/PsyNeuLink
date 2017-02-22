import composition
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
import time

test_constructor = False
test_add_mechanism = False
test_add_projection = False
test_analyze_graph = False
test_validate_feed_dict = False

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


if test_validate_feed_dict:
    print("\n" + "validate_feed_dict tests:")

    print("Test 1: Origin & Terminal Mechanisms w/ Mapping Projection")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [ [0] ]}
    feed_dict_terminal = {B: [[0] ]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")

    print("Test 2: Empty Feed Dicts")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {}
    feed_dict_terminal = {}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")


    print("Test 3: Origin & Terminal Mechanisms w/ Swapped Feed Dicts")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {B: [ [0] ]}
    feed_dict_terminal = {A: [[0] ]}
    try:
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    except ValueError:
        print("passed (1/2)")
    try:
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    except ValueError:
        print("passed (2/2)")

    print("Test 4: Multiple Origin Mechanisms")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    C = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_projection(A, MappingProjection(), C)
    comp.add_projection(B, MappingProjection(), C)
    comp.analyze_graph()
    feed_dict_origin = {A: [ [0] ], B: [ [0] ] }
    feed_dict_terminal = {C: [[0] ]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")

    print("Test 5: Multiple Origin Mechanisms, Only 1 in Feed Dict")
    comp = composition.Composition()
    A = mechanism()
    B = mechanism()
    C = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_projection(A, MappingProjection(), C)
    comp.add_projection(B, MappingProjection(), C)
    comp.analyze_graph()
    feed_dict_origin = {B: [ [0] ] }
    feed_dict_terminal = {C: [[0] ]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")


    print("Test 6: Input State Length 3")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0,1,2])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [ [0,1,2] ] }
    feed_dict_terminal = {B: [[0] ]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")

    print("Test 7: Input State Length 3; Length 2 in Feed Dict")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0,1,2])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [ [0,1] ] }
    feed_dict_terminal = {B: [[0] ]}
    try:
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    except ValueError:
        print("passed")

    print("Test 8: Input State Length 2; Length 3 in Feed Dict")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0,1])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [ [0,1,2] ] }
    feed_dict_terminal = {B: [[0] ]}
    try:
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    except ValueError:
        print("passed")

    print("Test 9: Feed Dict Includes Mechanisms of the Correct & Incorrect Types")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [ [0] ], B: [ [0] ]}
    try:
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    except ValueError:
        print("passed")

    print("Test 10: Input State Length 3, 1 Set of Extra Brackets")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0,1,2])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [ [ [0,1,2] ] ] }
    feed_dict_terminal = {B: [[0] ]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")


    print("Test 11: Input State Length 3, 1 Set of Missing Brackets")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0,1,2])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A:  [0,1,2] }
    feed_dict_terminal = {B: [[0] ]}
    try:
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    except TypeError:
        print("passed")


    print("Test 12: Empty Feed Dict For Empty Type")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [[0]]}
    feed_dict_monitored = {}
    comp.validate_feed_dict(feed_dict_monitored, comp.monitored_mechanisms, "monitored")
    print("passed")

    print("Test 13: Mechanism in Feed Dict For Empty Type")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [[0]]}
    feed_dict_monitored = {B: [[0]]}
    try:
        comp.validate_feed_dict(feed_dict_monitored, comp.monitored_mechanisms, "monitored")
    except ValueError:
        print("passed")

    print("Test 14: One Mechanism")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[0])
    comp.add_mechanism(A)
    comp.analyze_graph()
    feed_dict_origin = {A: [[0]]}
    feed_dict_terminal = {A: [[0]]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    print("passed (1/2)")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed (2/2)")


    print("Test 15: Multiple Time Steps")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[[0,1,2]])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [[0,1,2], [0,1,2]] }
    feed_dict_terminal = {B: [[0]]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")

    print("Test 16: Multiple Time Steps")
    comp = composition.Composition()
    A = TransferMechanism(default_input_value=[[0,1,2]])
    B = mechanism()
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(), B)
    comp.analyze_graph()
    feed_dict_origin = {A: [[[0,1,2]], [[0,1,2]]] }
    feed_dict_terminal = {B: [[0]]}
    comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
    comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
    print("passed")