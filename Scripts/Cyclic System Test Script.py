from PsyNeuLink.Functions.System import system
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Process import Mapping

a = Transfer(name='a',default_input_value=[0,0])
b = Transfer(name='b',default_input_value=[0,0])
c = Transfer(name='c')
d = Transfer(name='d')
e = Transfer(name='e')
f = Transfer(name='f')

# fb1 = Mapping(sender=c, receiver=b, name='fb1')
# fb2 = Mapping(sender=d, receiver=e, name = 'fb2')
#
# fb3 = Mapping(sender=e, receiver=a, name = 'fb3')

# p1 = process(configuration=[a, b, c, d], name='p1')

# @@@@@@@@@@@@@@@@@@@@@@@@@@
# a -> origin
# b -> internal
# c -> internal
# d -> initialize
# e -> terminal

# # CYCLIC:
# p2 = process(configuration=[e, c, b, d], name='p2')
# p1e = process(configuration=[a, b, c, d], name='p1e')
# p2 = process(configuration=[e, c, b, d], name='p2')
# p2 = process(configuration=[e, c, f, b, d], name='p2')

# CYCLIC INCLUDING ORIGIN IN CYCLE:
# p1e = process(default_input_value=[[0,0]], configuration=[a, b], name='p1e')
# p1e = process(configuration=[a, b], name='p1e')
p1e = process(configuration=[a, b, a], name='p1e')
p2 = process(configuration=[a, c, a], name='p2')
# p2 = process(configuration=[e, f], name='p2')


# # BRANCH:
# p1e = process(configuration=[a, b, c], name='p1e')
# p2 = process(configuration=[a, b, d], name='p2')

# # # BYPASS (OUT OF PROCESS BRANCH):
# p1e = process(configuration=[a, b, c, d], name='p1e')
# p2 = process(configuration=[a, b, d], name='p2')


# # CHAIN:
# p1e = process(configuration=[a, b, c], name='p1e')
# p2 = process(configuration=[c, d, e], name='p2')

# CONVERGENT:
# p1e = process(configuration=[a, b, e], name='p1e')
# p2 = process(configuration=[c, d, e], name='p2')


# WORKS (treats e as an origin):
# s = system(default_input_value=[[0,0]], processes=[p1e], name='system')
# s = system(processes=[p1e], name='system')

s = system(processes=[p1e, p2],
           name='system',
           initial_values={a:[1,1]})

# s = system(processes=[p2 ,p1e], name='system')
        # Senders to b:
        # 	a
        # 	e
        # Senders to c:
        # 	b
        # a is origin
        # e is origin
        # Senders to d:
        # 	c
        #

# HAS CYCLE (does NOT treat e as an origin):
# a = system(processes=[p2, p1e], name='system')
        # Senders to e:
        # 	c
        # a is origin
        # Senders to d:
        # 	c
        # Senders to b:
        # 	e
        # 	a
        # Senders to c:
        # 	b

# SEQUENTIAL PROCESSES:
# p_1 = process(configuration=[e, a, b, c, b], name='p_1')
# p_2 = process(configuration=[b, d, e], name='p_2')
# p_1 = process(configuration=[a, b], name='p_1')
# p_2 = process(configuration=[b, a], name='p_2')
# a = system(processes=[p_1, p_2], name='seq_system')




# p4 = process(configuration=[a, b, c], name='p4')
# p5= process(configuration=[c, d, e], name='p5')
# a = system(processes=[p1, p2, p3], name='system')


# a = system(processes=[p4, p5], name='system')

# s.inspect()
s.inspect()

print ('A: ',a.systems[s])
# print ('A: ',a.processes)
print ('B: ',b.systems[s])
print ('C: ',c.systems[s])
# print ('D: ',d.systems[s])
# print ('E: ',e.systems[s])

for projection in e.inputState.receivesFromProjections:
    print("Projection name: {}; sender: {};  receiver: {}".
          format(projection.name, projection.sender.owner.name, projection.receiver.owner.name))

s.execute(inputs=[[0,0]])
# s.execute()
