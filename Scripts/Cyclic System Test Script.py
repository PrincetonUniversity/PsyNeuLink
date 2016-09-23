from PsyNeuLink.Functions.System import system
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Process import Mapping

a = Transfer(name='a')
b = Transfer(name='b')
c = Transfer(name='c')
d = Transfer(name='d')
e = Transfer(name='e')

# fb1 = Mapping(sender=c, receiver=b, name='fb1')
# fb2 = Mapping(sender=d, receiver=e, name = 'fb2')

# fb3 = Mapping(sender=e, receiver=a, name = 'fb3')

# p1 = process(configuration=[a, b, c, d], name='p1')

p1e = process(configuration=[a, b, c, e], name='p1e')
p2 = process(configuration=[e, b, c, d], name='p2')

# p4 = process(configuration=[a, b, c], name='p4')
# p5= process(configuration=[c, d, e], name='p5')
# a = system(processes=[p1, p2, p3], name='system')

# WORKS:
a = system(processes=[p1e, p2], name='system')
# HAS CYCLE:
# a = system(processes=[p2, p1e], name='system')

# a = system(processes=[p4, p5], name='system')

a.inspect()

for projection in e.inputState.receivesFromProjections:
    print("Projection name: {}; sender: {};  receiver: {}".
          format(projection.name, projection.sender.owner.name, projection.receiver.owner.name))

a.execute()
