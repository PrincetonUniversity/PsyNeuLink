from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

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

z = Process_Base(default_input_value=[[30], [10]],
                 params={kwConfiguration:[myMechanism,
                                          (kwIdentityMatrix, 1),
                                          myMechanism_2,
                                          (kwFullConnectivityMatrix, 1),
                                          # (kwIdentityMatrix, 1),
                                          myMechanism_3]},
                 prefs = process_prefs)

z.execute([[30], [10]])

myMechanism.log.print_entries(ALL_ENTRIES, kwTime, kwValue)
