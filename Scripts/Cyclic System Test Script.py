from PsyNeuLink.Functions.System import system
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Process import Mapping


# fb1 = Mapping(sender=c, receiver=b, name='fb1')
# fb2 = Mapping(sender=d, receiver=e, name = 'fb2')
#
# fb3 = Mapping(sender=e, receiver=a, name = 'fb3')

# p1 = process(configuration=[a, b, c, d], name='p1')


print ('*****************************************************************************')

# BRANCH -----------------------------------------------------------------------------

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b',default_input_value=[0,0])
c = Transfer(name='c')
d = Transfer(name='d')

p1 = process(configuration=[a, b, c], name='p1')
p2 = process(configuration=[a, b, d], name='p2')

s = system(processes=[p1, p2],
           name='Branch System',
           initial_values={a:[1,1]})

s.inspect()

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])


print ('*****************************************************************************')

# BYPASS -----------------------------------------------------------------------------

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b',default_input_value=[0,0])
c = Transfer(name='c')
d = Transfer(name='d')

p1 = process(configuration=[a, b, c, d], name='p1')
p2 = process(configuration=[a, b, d], name='p2')

s = system(processes=[p1, p2],
           name='Bypass System',
           initial_values={a:[1,1]})

s.inspect()

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])


print ('*****************************************************************************')

# CHAIN -----------------------------------------------------------------------------

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b',default_input_value=[0,0])
c = Transfer(name='c')
d = Transfer(name='d')
e = Transfer(name='e')

p1 = process(configuration=[a, b, c], name='p1')
p2 = process(configuration=[c, d, e], name='p2')

s = system(processes=[p1, p2],
           name='Chain System',
           initial_values={a:[1,1]})

s.inspect()

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])
print ('E: ',e.systems[s])


print ('*****************************************************************************')

# CONVERGENT -----------------------------------------------------------------------------

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b')
c = Transfer(name='c')
# c = Transfer(name='c',default_input_value=[0,0])
d = Transfer(name='d')
e = Transfer(name='e')

p1 = process(configuration=[a, b, e], name='p1')
p2 = process(configuration=[c, d, e], name='p2')

s = system(processes=[p1, p2],
           name='Chain System',
           initial_values={a:[1,1]})

s.inspect()

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])
print ('E: ',e.systems[s])

# inputs=s.construct_input(inputs={a:[[2,2]], c:[[0]]})
inputs=s.construct_input(inputs={a:[2,2], c:[0]})

# #                                |--------LIST---------------------|
# #                                     |-------TRIAL -----------|
# #                                        |-------PHASE -----|
# #                                            MECH1  MECH2
# inputs=s.construct_input(inputs=[    [   [  [2,2],  [0]     ]   ]  ])

s.execute(inputs=inputs)
# s.execute(inputs=[[0,0],[0]])


print ('*****************************************************************************')

# CYCLIC INCLUDING ORIGIN IN CYCLE (ONE PROCESS) ------------------------------------

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b',default_input_value=[0,0])

p1 = process(configuration=[a, b, a], name='p1')

s = system(processes=[p1],
           name='Cyclic System with one Process',
           initial_values={a:[1,1]})

s.inspect()

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])


print ('*****************************************************************************')

# CYCLIC INCLUDING ORIGIN IN CYCLE (TWO PROCESSES) -----------------------------------

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b',default_input_value=[0,0])
c = Transfer(name='c',default_input_value=[0,0])

p1 = process(configuration=[a, b, a], name='p1')
p2 = process(configuration=[a, c, a], name='p2')

s = system(processes=[p1, p2],
           name='Cyclic System with one Process',
           initial_values={a:[1,1]})

s.inspect()

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])


# CYCLIC WITH TWO PROCESSES AND AN EXTENDED LOOP ------------------------------------

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b',default_input_value=[0,0])
c = Transfer(name='c')
d = Transfer(name='d')
e = Transfer(name='e')
f = Transfer(name='f')

p1 = process(configuration=[a, b, c, d], name='p1')
p2 = process(configuration=[e, c, f, b, d], name='p2')

s = system(processes=[p1, p2],
           name='Cyclic System with Extended Loop',
           initial_values={a:[1,1]})

s.inspect()

print ('A: ',a.systems[s])
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
print ('D: ',d.systems[s])
print ('E: ',e.systems[s])
print ('F: ',f.systems[s])
