from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
# from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Globals.Keywords import *

DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

# my_DDM = DDM(name='my_DDM')

my_DDM = DDM(drift_rate=(1.0, kwControlSignal),
                  threshold=(10.0, kwControlSignal),
                  starting_point=0.0,
                  prefs = DDM_prefs,

                  name='My_DDM'
                  )


my_DDM.prefs = DDM_prefs
my_DDM.prefs.level = PreferenceLevel.SYSTEM

my_DDM.prefs.inspect()

#region MAIN SCRIPT
# myMechanism = DDM(params={kwFunctionParams:{kwDDM_DriftRate:(1.0, kwControlSignal),
#                                                  kwDDM_Threshold:(10.0, kwControlSignal)},
#                           kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
#                   prefs = DDM_prefs,
#                   name='My_DDM'
#                   )

myMechanism = DDM(drift_rate=(1.0, kwControlSignal),
                  threshold=(10.0, kwControlSignal),
                  starting_point=0.0,
                  prefs = DDM_prefs,

                  name='My_DDM'
                  )

myMechanism_2 = DDM(drift_rate=2.0,
                    threshold=20.0,
                    analytic_solution=kwDDM_BogaczEtAl,
                    prefs = DDM_prefs,
                    name='My_DDM_2'
                    )

myMechanism_3 = DDM(drift_rate=3.0,
                    threshold=30.0,
                    analytic_solution=kwDDM_BogaczEtAl,
                    prefs = DDM_prefs,
                    name='My_DDM_3'
                    )

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

process_prefs.inspect()

z = process(default_input_value=[[30], [10]],
# z = Process_Base(default_input_value=[[30], [10]],
                 params={kwConfiguration:[myMechanism,
                                          (kwIdentityMatrix, 1),
                                          myMechanism_2,
                                          (kwFullConnectivityMatrix, 1),
                                          # (kwIdentityMatrix, 1),
                                          myMechanism_3]},
                 prefs = process_prefs)

z.execute([[30], [10]])

myMechanism.log.print_entries(ALL_ENTRIES, kwTime, kwValue)
