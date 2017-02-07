import composition as composition
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
import time

# Unit tests for each function of the Composition class #######################
# Unit tests for Composition.Composition()
print("Constructor Tests:")
print("Test 1: No args")
comp = composition.Composition()
print("passed")
print("Test 2: Second call no args")
comp_2 = composition.Composition()
print("passed")
print("Test 3: Timing no args")
count = 10000
start = time.time()
for i in range(count):
    comp = composition.Composition()
end = time.time()
print("passed in " + str(end-start) + " seconds for " + str(count) + " calls")

# Unit tests for Composition.add_mechanism
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
count = 10000
proj_list = []
for i in range(count):
    proj_list.append(MappingProjection())
start = time.time()
for i in range(count):
    comp.add_projection(A, proj_list[i], B)
end = time.time()
print("passed in " + str(end-start) + " seconds for " + str(count) + " calls")
