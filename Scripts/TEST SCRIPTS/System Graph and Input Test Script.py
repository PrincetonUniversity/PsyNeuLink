from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism import TransferMechanism

# from PsyNeuLink.Globals.Run import run, construct_inputs
# from PsyNeuLink.Components.Process import MappingProjection

# INPUT SEQUENCES FOR TESTING:
# FACTORS (# levels to test):
#   Unequal inputs (2)
#   Singleton inputs (2)
#   Level of array embedding (4)
#   Number of trials (2)
#   Number of phases (2)
#   Trial list vs. dict format (2)
#  = 128 combinations!

# INPUTS OUT OF ORDER:
# inputs=[[0], [2,2]]top

# EQUAL INPUT LENGTHS:
# inputs=[[2,2],[0,0]]
# inputs=[[[2,2],[0,0]]]
# inputs=[[[[2,2],[0,0]]]]
# inputs=[[[2,2],[0,0]],[[2,2],[0,0]]]
# inputs=[[[[2,2],[0,0]],[[2,2],[0,0]]]]
# inputs=[[[[2,2],[0,0]]],[[[2,2],[0,0]]]]
# inputs=[[[2,2,2],[0,0,0]],[[2,2,2],[0,0,0]]]

# UNEQUAL INPUT LENGTHS:
# inputs=[[2,2],0]
# inputs=[[2,2],[0]]
# inputs=[[[2,2],0],[[2,2],0]]
# inputs=[[[2,2],[0]],[[2,2],[0]]]
# inputs=[[[[2,2],[0]]],[[[2,2],[0]]]]

# STIMULUS DICT:
# inputs={a:[2,2], c:[0]})
# inputs={a:[[2,2]], c:[[0]]})


# FEEDBACK CONNECTIONS:
# fb1 = MappingProjection(sender=c, receiver=b, name='fb1')
# fb2 = MappingProjection(sender=d, receiver=e, name = 'fb2')
# fb3 = MappingProjection(sender=e, receiver=a, name = 'fb3')

show_graph = False

print ('*****************************************************************************')

# A) BRANCH -----------------------------------------------------------------------------

a = TransferMechanism(name='test a',default_variable=[0,0])
b = TransferMechanism(name='b')
c = TransferMechanism(name='c')
d = TransferMechanism(name='d')

p1 = process(pathway=[a, b, c], name='p1')
p2 = process(pathway=[a, b, d], name='p2')

s = system(processes=[p1, p2],
           name='Branch System',
           initial_values={a:[1,1]})

s.show()
if show_graph:
    s.show_graph(direction='LR')

# inputs=[2,2]
inputs={a:[2,2]}
s.run(inputs)

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])


print ('*****************************************************************************')

# B) BYPASS -----------------------------------------------------------------------------

a = TransferMechanism(name='a',default_variable=[0,0])
b = TransferMechanism(name='b',default_variable=[0,0])
c = TransferMechanism(name='c')
d = TransferMechanism(name='d')

p1 = process(pathway=[a, b, c, d], name='p1')
p2 = process(pathway=[a, b, d], name='p2')

s = system(processes=[p1, p2],
           name='Bypass System',
           initial_values={a:[1,1]})

# inputs=[[[2,2],[0,0]],[[2,2],[0,0]]]
inputs={a:[[2,2],[0,0]]}
s.run(inputs=inputs)

s.show()
if show_graph:
    s.show_graph(direction='LR')

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])

print ('*****************************************************************************')

# C) CHAIN -----------------------------------------------------------------------------

a = TransferMechanism(name='a',default_variable=[0,0,0])
b = TransferMechanism(name='b')
c = TransferMechanism(name='c')
d = TransferMechanism(name='d')
e = TransferMechanism(name='e')

p1 = process(pathway=[a, b, c], name='p1')
p2 = process(pathway=[c, d, e], name='p2')

s = system(processes=[p1, p2],
           name='Chain System',
           initial_values={a:[1,1,1]})

# inputs=[[[2,2,2],[0,0,0]]]
inputs={a:[[2,2,2],[0,0,0]]}
s.run(inputs=inputs)

s.show()
if show_graph:
    s.show_graph(direction='LR')

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])
print ('E: ',e.systems[s])


print ('*****************************************************************************')

# D) CONVERGENT -----------------------------------------------------------------------------

a = TransferMechanism(name='a',default_variable=[0,0])
b = TransferMechanism(name='b')
c = TransferMechanism(name='c')
c = TransferMechanism(name='c',default_variable=[0])
d = TransferMechanism(name='d')
e = TransferMechanism(name='e')

p1 = process(pathway=[a, b, e], name='p1')
p2 = process(pathway=[c, d, e], name='p2')

s = system(processes=[p1, p2],
           name='Convergent System',
           initial_values={a:[1,1]})

# inputs=[[2,2],0]
# inputs=[[[[[2,2],[0]]]]]
inputs={a:[2,2],
        c:[0]}
s.run(inputs=inputs)

s.show()
if show_graph:
    s.show_graph(direction='LR')

inputs={a:[[2,2]], c:[[0]]}
s.run(inputs=inputs)

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])
print ('E: ',e.systems[s])


print ('*****************************************************************************')

# E) CYCLIC INCLUDING ORIGIN IN CYCLE (ONE PROCESS) ------------------------------------

a = TransferMechanism(name='a',default_variable=[0,0])
b = TransferMechanism(name='b',default_variable=[0,0])

p1 = process(pathway=[a, b, a], name='p1')

s = system(processes=[p1],
           name='Cyclic System with one Process',
           initial_values={a:[1,1]})

s.show()
if show_graph:
    s.show_graph(direction='LR')

# inputs=[[1,1]]
inputs={a:[1,1]}
s.run(inputs=inputs)

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])


print ('*****************************************************************************')

# F) CYCLIC INCLUDING ORIGIN IN CYCLE (TWO PROCESSES) -----------------------------------

a = TransferMechanism(name='a',default_variable=[0,0])
b = TransferMechanism(name='b',default_variable=[0,0])
c = TransferMechanism(name='c',default_variable=[0,0])

p1 = process(pathway=[a, b, a], name='p1')
p2 = process(pathway=[a, c, a], name='p2')

s = system(processes=[p1, p2],
           name='Cyclic System with two Processes',
           initial_values={a:[1,1]})

s.show()
if show_graph:
    s.show_graph(direction='LR')

# inputs=[[1,1]]
inputs={a:[1,1]}
s.run(inputs=inputs)

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])


# G) CYCLIC WITH TWO PROCESSES AND AN EXTENDED LOOP ------------------------------------

a = TransferMechanism(name='a',default_variable=[0,0])
b = TransferMechanism(name='b')
c = TransferMechanism(name='c')
d = TransferMechanism(name='d')
e = TransferMechanism(name='e',default_variable=[0])
f = TransferMechanism(name='f')

p1 = process(pathway=[a, b, c, d], name='p1')
p2 = process(pathway=[e, c, f, b, d], name='p2')

s = system(processes=[p1, p2],
           name='Cyclic System with Extended Loop',
           initial_values={a:[1,1]})

s.show()
if show_graph:
    s.show_graph(direction='LR')

inputs={a:[2,2], e:[0]}
s.run(inputs=inputs)

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])
print ('E: ',e.systems[s])
print ('F: ',f.systems[s])

# print("\nGRAPH:")
# for receiver, senders in s.graph.items():
#     print(receiver[0].name, ":")
#     for sender in senders:
#         print('\t', sender[0].name, ":")
#
# print("\nEXECUTION GRAPH:")
# for receiver, senders in s.execution_graph.items():
#     print(receiver[0].name, ":")
#     for sender in senders:
#         print('\t', sender[0].name, ":")
#



# # H) CYCLIC WITH MECHANISM THAT IS BOTH INTIALIZE AND TERMINAL ------------------------------------
#
# a = TransferMechanism(name='a',default_variable=[0,0])
# b = TransferMechanism(name='b')
# c = TransferMechanism(name='c')
# d = TransferMechanism(name='d')
# e = TransferMechanism(name='e',default_variable=[0])
#
# p1 = process(pathway=[a, b, c, b], name='p1')
# p2 = process(pathway=[d, c], name='p2')
#
# s = system(processes=[p1, p2],
#            name='Cyclic System with Terminal = Initialize Mechanism',
#            initial_values={a:[1,1]})
#
# s.show()
#
# inputs={a:[2,2], d:[0]}
# s.run(inputs=inputs)
#
# print ('A: ',a.systems[s])
# print ('B: ',b.systems[s])
# print ('C: ',c.systems[s])
# print ('D: ',d.systems[s])

# s.show_graph()