from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.Projections.LearningSignal import LearningSignal
from PsyNeuLink.Functions.Utilities.Utility import Logistic, random_matrix
from PsyNeuLink.Globals.Run import run

from PsyNeuLink.Functions.Utilities.Utility import *


#region PROCESS EXAMPLES ********************************************************************

# Specification of mechanisms in a pathway
mechanism_1 = Transfer()
mechanism_2 = DDM()
some_params = {PARAMETER_STATE_PARAMS:{FUNCTION_PARAMS:{THRESHOLD:2,NOISE:0.1}}}
my_process = process(pathway=[mechanism_1, Transfer, (mechanism_2, some_params, 0)])
print(my_process.execute())

# Default projection specification
mechanism_1 = Transfer()
mechanism_2 = Transfer()
mechanism_3 = DDM()
my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
print(my_process.execute())

# Inline projection specification using an existing projection
mechanism_1 = Transfer()
mechanism_2 = Transfer()
mechanism_3 = DDM()
projection_A = Mapping()
my_process = process(pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])
print(my_process.execute())

mechanism_1 = Transfer()
mechanism_2 = Transfer()
mechanism_3 = DDM()
# Inline projection specification using a keyword
my_process = process(pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2, mechanism_3])
print(my_process.execute())

# Stand-alone projection specification
mechanism_1 = Transfer()
mechanism_2 = Transfer()
mechanism_3 = DDM()
projection_A = Mapping(sender=mechanism_1, receiver=mechanism_2)
my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
print(my_process.execute())

# Process that implements learning
mechanism_1 = Transfer(function=Logistic)
mechanism_2 = Transfer(function=Logistic)
mechanism_3 = Transfer(function=Logistic)
# my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3],
#                      learning=LEARNING_SIGNAL)
my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3],
                     learning=LEARNING_SIGNAL,
                     target=[0])
print(my_process.execute())

#endregion


#region SYSTEM EXAMPLES ********************************************************************
#
#endregion