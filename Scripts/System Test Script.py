from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Functions.System import System_Base
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer

# DDM_prefs = FunctionPreferenceSet(
#                 prefs = {
#                     kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
#                     kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})
#
# my_DDM = DDM(name='my_DDM')
# my_DDM.prefs = DDM_prefs
# my_DDM.prefs.level = PreferenceLevel.SYSTEM
#
# my_DDM.prefs.show()
#
# #region MAIN SCRIPT
# myMechanism = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:(1.0, CONTROL_SIGNAL),
#                                                  THRESHOLD:(10.0, CONTROL_SIGNAL)},
#                           kwDDM_AnalyticSolution:kwBogaczEtAl},
#                   prefs = DDM_prefs,
#                   name='My_DDM'
#                   )
#
# myMechanism_2 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:2.0,
#                                                    THRESHOLD:20.0},
#                             kwDDM_AnalyticSolution:kwBogaczEtAl},
#                     prefs = DDM_prefs,
#                     name='My_DDM_2'
#                     )
#
# myMechanism_3 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:3.0,
#                                                    THRESHOLD:30.0},
#                             kwDDM_AnalyticSolution:kwBogaczEtAl},
#                     prefs = DDM_prefs,
#                     name='My_DDM_3'
#                     )
#
# process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
#                                       verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))
#
# process_prefs.show()

Layer_1 = Transfer(default_input_value=[0,0], name='Layer 1')
Layer_2 = Transfer(default_input_value=[0,0], name='Layer 2')
Layer_3 = Transfer(default_input_value=[0,0], name='Layer 3')


myProcess_1 = Process_Base(default_input_value=[0, 0],
                           params={PATHWAY:[(Layer_1, 0),
                                                    IDENTITY_MATRIX,
                                                    (Layer_3, 0)]})

myProcess_2 = Process_Base(default_input_value=[0, 0],
                           params={PATHWAY:[(Layer_2, 0),
                                                    FULL_CONNECTIVITY_MATRIX,
                                                    (Layer_3, 0)]})

mySystem = System_Base(params={kwProcesses:[(myProcess_1,0), (myProcess_2,0)]})

myProcess_1.reportOutputPref = True
myProcess_2.reportOutputPref = True
mySystem.reportOutputPref = True

mySystem.execute([[0,0], [1,1]])

def print_inputs():
    print('INPUTS :',mySystem.inputs)

stimuli = [[[[0,0], [1,1]]], [[[2,2], [3,3]]]]
my_results = mySystem.run(inputs=stimuli, num_trials=2, call_before=print_inputs)

print(my_results)

