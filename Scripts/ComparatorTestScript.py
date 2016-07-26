

from Functions.Mechanisms.MonitoringMechanisms.LinearComparator import LinearComparator
from Functions.Process import Process_Base
from Globals.Keywords import *

# from Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
# sample_mech = Transfer(name='Sample',
#                        params={kwExecuteMethod:kwLogistic},
#                        default_input_value = [0,0])
#
# target_mech = Transfer(name='Target',
#                        params={kwExecuteMethod:kwLogistic},
#                        default_input_value = [0,0])


my_comparator = LinearComparator(name='My Comparator')

my_comparator.execute(variable=[[0,0], [0,1]])

#
# my_process = Process_Base(default_input_value=[[0,0], [0,1]],
#                  params={kwConfiguration:[my_comparator]},
#                  # prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}
#                           )
#
# my_process.execute([-1, 30])
