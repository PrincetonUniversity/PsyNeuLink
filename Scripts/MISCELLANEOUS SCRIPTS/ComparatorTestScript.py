

from PsyNeuLink.Components.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

# from Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
# sample_mech = TransferMechanism(name='Sample',
#                        params={FUNCTION:LOGISTIC_FUNCTION},
#                        default_input_value = [0,0])
#
# target_mech = TransferMechanism(name='Target',
#                        params={FUNCTION:LOGISTIC_FUNCTION},
#                        default_input_value = [0,0])

import numpy as np

# my_comparator = ComparatorMechanism(default_input_value=[[0,0], [0,1]],
#                                  name='My ComparatorMechanism')
#
# my_comparator.execute(variable=np.array([[0,0], [0,1]]))
#
# my_process = Process_Base(default_input_value=[[0,0], [0,1]],
#                           params={PATHWAY:[my_comparator]},
#                           # prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}
#                           )
#
# my_process.execute([[-1, 30],[1, 15]])



my_process = Process_Base(default_input_value=[[0],[1]],
                 params={PATHWAY:[my_comparator]},
                 # prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}
                          )
my_process.execute(input=np.array([[0], [1]]))


