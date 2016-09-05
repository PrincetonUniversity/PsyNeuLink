

from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.Comparator import Comparator
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

# from Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
# sample_mech = Transfer(name='Sample',
#                        params={kwFunction:kwLogistic},
#                        default_input_value = [0,0])
#
# target_mech = Transfer(name='Target',
#                        params={kwFunction:kwLogistic},
#                        default_input_value = [0,0])

import numpy as np

# my_comparator = Comparator(default_input_value=[[0,0], [0,1]],
#                                  name='My Comparator')
#
# my_comparator.execute(variable=np.array([[0,0], [0,1]]))
#
# my_process = Process_Base(default_input_value=[[0,0], [0,1]],
#                           params={kwConfiguration:[my_comparator]},
#                           # prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}
#                           )
#
# my_process.execute([[-1, 30],[1, 15]])


my_comparator = Comparator(default_sample_and_target=[[0], [0]],
                                 name='My Comparator')


my_process = Process_Base(default_input_value=[[0],[1]],
                 params={kwConfiguration:[my_comparator]},
                 # prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}
                          )
my_process.execute(input=np.array([[0], [1]]))


