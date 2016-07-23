from Functions.Mechanisms.ProcessingMechanisms.LinearMechanism import LinearMechanism
from Functions.Mechanisms.ProcessingMechanisms.SigmoidLayer import SigmoidLayer

from Functions.Mechanisms.ProcessingMechanisms.DDM import *
from Functions.Process import Process_Base
from Globals.Keywords import *

# x = Process_Base()
# x.execute(10.0)


my_Sigmoid = SigmoidLayer(name='my_Sigmoid', default_input_value = [0,0])
my_Linear = LinearMechanism(name='my_Linear', default_input_value = [0,0])

# myMechanism = xxx(params={kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
#                   prefs = {kpReportOutputPref: PreferenceLevel.SYSTEM},
#                   name='My_DDM'
#                   )
#
#region ADDITION MECHANISMS
# # DDM.classPreferences.reportOutputPref = PreferenceEntry(False, PreferenceLevel.INSTANCE)
#
# # my_Mechanism_2 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:2.0,
# #                                                         kwDDM_Threshold:1.0},
# #                              # kwDDM_AnalyticSolution:kwDDM_NavarroAndFuss  # Note: this requires matlab engine be installed
# #                              kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
# #                      # prefs=DDM_prefs
# #                      # prefs = {kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
# #                      #          kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
# #                      #          }
# #                      prefs = {kpReportOutputPref: True,
# #                               kpVerbosePref: False
# #                               },
# #                      name='My_DDM'
# #                      )
# #
# # my_Mechanism_3 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:2.0,
# #                                                         kwDDM_Threshold:1.0},
# #                              # kwDDM_AnalyticSolution:kwDDM_NavarroAndFuss  # Note: this requires matlab engine be installed
# #                              kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
# #                      # prefs=DDM_prefs
# #                      # prefs = {kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
# #                      #          kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
# #                      #          }
# #                      prefs = {kpReportOutputPref: True,
# #                               kpVerbosePref: False
# #                               },
# #                      name='My_DDM'
# #                      )
#
# # process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
# #                                        verbose_pref=PreferenceEntry(True,PreferenceLevel.SYSTEM))
# from Functions.Utility import LinearCombination
# y = Process_Base(params={kwConfiguration:[(myMechanism,
#                                            {
#                                                # kwMechanismInputStateParams:{},
#                                                kwMechanismParameterStateParams:
#                                                    {kwParamModulationOperation: ModulationOperation.MULTIPLY, # B
#                                                     kwDDM_DriftRate:(30.0,
#                                                                      ModulationOperation.MULTIPLY), # C
#                                                     kwDDM_Threshold:20.0,
#                                                     kwExecuteMethodParams:
#                                                        {LinearCombination.kwOffset: 100}, # A
#                                                     # kwProjectionParams:
#                                                     #     {Linear.kwIntercept: 1},
#                                                     },
#                                            }),
#                                           (myMechanism,
#                                            {
#                                                # kwMechanismInputStateParams:{},
#                                                kwMechanismParameterStateParams:
#                                                    {kwParamModulationOperation: ModulationOperation.MULTIPLY, # B
#                                                     kwDDM_DriftRate:(30.0,
#                                                                      ModulationOperation.MULTIPLY), # C
#                                                     kwDDM_Threshold:20.0,
#                                                     kwExecuteMethodParams:
#                                                        {LinearCombination.kwOffset: 100}, # A
#                                                     # kwProjectionParams:
#                                                     #     {Linear.kwIntercept: 1},
#                                                     },
#                                            }),
#                                           myMechanism]},
#
#                  # prefs=process_prefs)
#                  # prefs = {kpReportOutputPref: PreferenceLevel.INSTANCE,
#                  #          kpVerbosePref: PreferenceLevel.INSTANCE},
#                  prefs = {kpReportOutputPref: True,
#                           kpVerbosePref: False,
#                           kpParamValidationPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)},
#                  name='My_Process')
#
# y.execute(1.0)
# # y.execute(1.0)
# # y.execute(1.0)
#endregion

z = Process_Base(default_input_value=[0, 0],
                 params={kwConfiguration:[my_Sigmoid]},
                 prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

# z = Process_Base(params={kwConfiguration:[myMechanism, myMechanism]})
# z = Process_Base(params={kwConfiguration:[DDM, DDM, DDM]})
# z = Process_Base(params={kwConfiguration:[mechanism()]})
z.execute([30, 30])
# #
