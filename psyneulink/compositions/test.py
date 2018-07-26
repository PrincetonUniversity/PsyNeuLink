# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:48:49 2018

@author: karan
"""

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.functions.function import Linear, Logistic, ReLU, SoftMax
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.compositions.composition import Composition
from psyneulink.compositions.parsingautodiffcomposition import ParsingAutodiffComposition
from psyneulink.compositions.pytorchcreator import PytorchCreator
from toposort import toposort


import numpy as np

mech1 = TransferMechanism(name='mech1',
                          default_variable=np.zeros(10),
                          )

mech2 = TransferMechanism(name='mech2',
                          default_variable=np.zeros(10),
                          )

mech3 = TransferMechanism(name='mech3',
                          default_variable=np.zeros(10),
                          )

big_mech = TransferMechanism(name='big_mech',
                             default_variable=np.zeros(15),
                             function=Logistic()
                             )

proj1 = MappingProjection(matrix=np.random.rand(10,15),
                          name='proj1',
                          sender=mech1,
                          receiver=big_mech
                          )

proj2 = MappingProjection(matrix=np.random.rand(10,15),
                          name='proj2',
                          sender=mech2,
                          receiver=big_mech
                          )

proj3 = MappingProjection(matrix=np.random.rand(10,15),
                          name='proj3',
                          sender=mech3,
                          receiver=big_mech
                          )

'''
print(big_mech)
print(big_mech.variable)
print(np.shape(big_mech.variable))
print(big_mech.input_states)
print(big_mech.external_input_states)
print(big_mech.external_input_values)
print(np.shape(big_mech.external_input_values))
print("hello")
print(big_mech.external_input_states[0].variable)
print(big_mech.external_input_states[0].value)
print(big_mech.input_states[0])
print(big_mech.input_states[0].variable)
print(np.shape(big_mech.input_states[0].variable))
print(big_mech.input_states[0].value)
print(np.shape(big_mech.input_states[0].value))
print("\n")
print("\n")
print("\n")
'''
'''
compy = Composition()

print(compy._graph_processing)
print("\n")
print("\n")

compy.add_c_node(mech1)
compy.add_c_node(mech2)
compy.add_c_node(mech3)
compy.add_c_node(big_mech)
compy.add_projection(mech1, proj1, big_mech)
compy.add_projection(mech2, proj2, big_mech)
compy.add_projection(mech3, proj3, big_mech)
'''

'''
for i in range(len(compy._graph_processing.vertices)):
    vertex = compy._graph_processing.vertices[i]
    print(vertex)
    print(vertex.component)
    print(vertex.parents)
    print(vertex.children)
    print("\n")

print("\n")
print("\n")
print("\n")

for i in range(len(compy.graph.vertices)):
    print(compy.graph.vertices[i])

print("\n")
print("\n")
print("\n")
'''
'''
a = 10
b = 3
if not (a > 3 and b > 13):
    print("butts")
'''
'''
inputs_dict = {}
inputs_dict[mech1] = np.ones(10)
inputs_dict[mech2] = np.ones(10)
inputs_dict[mech3] = np.ones(10)

compy.input_CIM.reportOutputPref = True
compy.output_CIM.reportOutputPref = True
output = compy.run(inputs=inputs_dict)
print("\n")
print(output)

print("\n")
print("\n")

# check if mechanism can have vector bias
'''
'''
n = np.ones(10)

test_mech = TransferMechanism(name='test_mech',
                             default_variable=np.zeros(10),
                             noise=n,
                             function=Logistic(gain=1, bias=n)
                             )

print(test_mech.execute(np.ones(10)))


hoity = ParsingAutodiffComposition()
hoity.run()
'''

'''
print("\n")
print("\n")

mecho = TransferMechanism()
mecho2 = TransferMechanism()
print(mecho.variable)
print(mecho.input_states)
print(mecho.output_states)
print("\n")

compoid = ParsingAutodiffComposition()
compoid.add_c_node(mecho)
compoid.add_c_node(mecho2)
# compoid.add_projection(mecho2, MappingProjection(sender=mecho2, receiver=mecho), mecho)
print(compoid._graph_processing.vertices)
print("\n")
compoid._update_processing_graph()
print(compoid._graph_processing.vertices)
compoid.model = PytorchCreator(compoid._graph_processing)
print(compoid.model.parameters)
a = compoid.model.get_weights_for_projections()
print(len(a))
result = compoid.run(inputs={mecho:np.ones(1), mecho2:np.ones(1)})
print(result)
print("\n")
print(compoid.graph.vertices)
print("\n")
result = compoid.run(inputs={mecho2:np.ones(1)}, targets={mecho:np.zeros(1)}, epochs=1)
print("\n")
print(compoid._graph_processing.vertices)
print("\n")
print(result)
'''

# testing toposort
nouns_in = TransferMechanism(name="nouns_input", 
                             default_variable=np.zeros(8))
        
rels_in = TransferMechanism(name="rels_input", 
                            default_variable=np.zeros(3))
        
h1 = TransferMechanism(name="hidden_nouns",
                       default_variable=np.zeros(8),
                       function=Logistic())
        
h2 = TransferMechanism(name="hidden_mixed",
                       default_variable=np.zeros(15),
                       function=Logistic())
        
out_sig_I = TransferMechanism(name="sig_outs_I",
                              default_variable=np.zeros(8),
                              function=Logistic())
        
out_sig_is = TransferMechanism(name="sig_outs_is",
                               default_variable=np.zeros(12),
                               function=Logistic())
        
out_sig_has = TransferMechanism(name="sig_outs_has",
                                default_variable=np.zeros(9),
                                function=Logistic())
        
out_sig_can = TransferMechanism(name="sig_outs_can",
                                default_variable=np.zeros(9),
                                function=Logistic())
        
# COMPOSITION FOR SEMANTIC NET
        
sem_net = Composition()
        
sem_net.add_c_node(nouns_in)
sem_net.add_c_node(rels_in)
sem_net.add_c_node(h1)
sem_net.add_c_node(h2)
sem_net.add_c_node(out_sig_I)
sem_net.add_c_node(out_sig_is)
sem_net.add_c_node(out_sig_has)
sem_net.add_c_node(out_sig_can)
    
sem_net.add_projection(nouns_in, MappingProjection(sender=nouns_in, receiver=h1), h1)
sem_net.add_projection(rels_in, MappingProjection(sender=rels_in, receiver=h2), h2)
sem_net.add_projection(h1, MappingProjection(sender=h1, receiver=h2), h2)
sem_net.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_I), out_sig_I)
sem_net.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_is), out_sig_is)
sem_net.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_has), out_sig_has)
sem_net.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_can), out_sig_can)
# sem_net.add_projection(h1, MappingProjection(sender=h1, receiver=nouns_in), nouns_in)

sem_net._update_processing_graph()

trainable_params = [vert.component for vert in sem_net.graph.vertices if isinstance(vert.component, MappingProjection)]
print(trainable_params)

print("\n")
print("\n")
print("\n")

dependencies = {}
for vert in sem_net._graph_processing.vertices:
    dependencies[vert.component] = set()
    for parent in sem_net._graph_processing.get_parents_from_component(vert.component):
        dependencies[vert.component].add(parent.component)

print(dependencies)
print("\n")
topoed = list(toposort(dependencies))
print(topoed)
print("\n")
print(type(topoed))
print("\n")

for i in range(len(topoed)):
    print(type(topoed[i]))
    topoed[i] = list(topoed[i])
    print(type(topoed[i]))
    for j in range(len(topoed[i])):
        print(topoed[i][j])
    print("\n")
    
print("\n")
print("\n")
print("\n")

dependencies = {}
for vert in sem_net._graph_processing.vertices:
    dependencies[vert.component] = set()
    for parent in sem_net._graph_processing.get_parents_from_component(vert.component):
        dependencies[vert.component].add(parent.component)

print(dependencies)
print("\n")
topoed = list(toposort(dependencies))
print(topoed)
print("\n")
print(type(topoed))
print("\n")

for i in range(len(topoed)):
    print(type(topoed[i]))
    topoed[i] = list(topoed[i])
    print(type(topoed[i]))
    for j in range(len(topoed[i])):
        print(topoed[i][j])
    print("\n")

print("\n")
print("\n")
print("\n")

dependencies = {}
for vert in sem_net._graph_processing.vertices:
    dependencies[vert.component] = set()
    for parent in sem_net._graph_processing.get_parents_from_component(vert.component):
        dependencies[vert.component].add(parent.component)

print(dependencies)
print("\n")
topoed = list(toposort(dependencies))
print(topoed)
print("\n")
print(type(topoed))
print("\n")

for i in range(len(topoed)):
    print(type(topoed[i]))
    topoed[i] = list(topoed[i])
    print(type(topoed[i]))
    for j in range(len(topoed[i])):
        print(topoed[i][j])
    print("\n")










                          