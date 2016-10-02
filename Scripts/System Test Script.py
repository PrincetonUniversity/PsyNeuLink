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
# my_DDM.prefs.inspect()
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
# process_prefs.inspect()

Layer_1 = Transfer(name='a')
Layer_2 = Transfer(name='b')
Layer_3 = Transfer(name='c')


myProcess_1 = Process_Base(default_input_value=[30],
                           params={CONFIGURATION:[(myMechanism_2, 0),
                                                    IDENTITY_MATRIX,
                                                    (myMechanism, 0)]},
                           prefs = process_prefs)

myProcess_2 = Process_Base(default_input_value=[10],
                           params={CONFIGURATION:[(myMechanism_3, 0),
                                                    FULL_CONNECTIVITY_MATRIX,
                                                    (myMechanism, 0)]},
                           prefs = process_prefs)

mySystem = System_Base(params={kwProcesses:[(myProcess_1,0), (myProcess_2,0)]})

mySystem.execute([[1], [2]])

