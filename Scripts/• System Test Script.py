from Functions.System import System_Base
from Functions.Process import Process_Base
from Functions.Mechanisms.DDM import *
from Globals.Keywords import *
from Functions.Utility import UtilityRegistry
from Functions.MechanismStates.MechanismState import MechanismStateRegistry

DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

my_DDM = DDM(name='my_DDM')
my_DDM.prefs = DDM_prefs
my_DDM.prefs.level = PreferenceLevel.SYSTEM

my_DDM.prefs.inspect()

#region MAIN SCRIPT
myMechanism = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:(1.0, kwControlSignal),
                                                 kwDDM_Threshold:(10.0, kwControlSignal)},
                          kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                  prefs = DDM_prefs,
                  name='My_DDM'
                  )

myMechanism_2 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:2.0,
                                                   kwDDM_Threshold:20.0},
                            kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                    prefs = DDM_prefs,
                    name='My_DDM_2'
                    )

myMechanism_3 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:3.0,
                                                   kwDDM_Threshold:30.0},
                            kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                    prefs = DDM_prefs,
                    name='My_DDM_3'
                    )

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

process_prefs.inspect()

myProcess_1 = Process_Base(default_input_value=[30],
                           params={kwConfiguration:[(myMechanism_2, 1),
                                                    kwIdentityMatrix,
                                                    (myMechanism, 1)]},
                           prefs = process_prefs)

myProcess_2 = Process_Base(default_input_value=[10],
                           params={kwConfiguration:[(myMechanism_3, 1),
                                                    kwFullConnectivityMatrix,
                                                    (myMechanism, 1)]},
                           prefs = process_prefs)

mySystem = System_Base(params={kwProcesses:[(myProcess_1,0), (myProcess_2,0)]})

mySystem.execute([[1], [2]])

