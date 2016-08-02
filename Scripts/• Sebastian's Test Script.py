from Functions.Mechanisms.ProcessingMechanisms.DDM import *
from Functions.Mechanisms.ProcessingMechanisms.DDM import kwDDM_DriftRate
from Functions.Process import Process_Base
from Globals.Keywords import *

# NOTE: DDM_prefs is now a "free-standing" PreferenceSet, so that it can be referred to by other objects
#  1) DDM_1 and DDM_2 refer to it, but DDM_3 does not
#  2) The next declaration (now uncommented) assigns a different set of prefs to the DDM class;  notice
#      that this affects DDM_3 but not DDM_1 or DDM_2 (which have been assigned to DDM_prefs)
DDM_prefs = FunctionPreferenceSet(
                # owner=DDM,
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    # kpLogPref: PreferenceEntry(LogLevel.OFF,PreferenceLevel.CATEGORY),
                    kpLogPref: PreferenceEntry(LogLevel.OFF,PreferenceLevel.CATEGORY),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpExecuteMethodRuntimeParamsPref: PreferenceEntry(ModulationOperation.OVERRIDE,PreferenceLevel.INSTANCE)})

DDM.classPreferences = DDM_prefs
DDM.classPreferences.reportOutputPref = PreferenceEntry(False, PreferenceLevel.INSTANCE)

#region MAIN SCRIPT
myMechanism_1 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:(2.0, kwControlSignal),
                                                   kwDDM_Threshold:(10.0, kwControlSignal)},
                            #                       kwParamModulationOperation: ModulationOperation.OVERRIDE},
                            # {kwParameterStateParams: {kwParamModulationOperation: ModulationOperation.OVERRIDE}},
                            # kwParamModulationOperation: ModulationOperation.OVERRIDE,
                            # kwDDM_AnalyticSolution:kwDDM_NavarroAndFuss  # Note: this requires matlab engine be installed
                            kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                    prefs=DDM_prefs,
                    # prefs = {kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE),
                    #          kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
                    #          },
                    # prefs = {kpReportOutputPref: True,
                    #          kpVerbosePref: False
                    #          },
                    # prefs = {kpReportOutputPref: False,
                    #          kpVerbosePref: False
                    #          },
                    name='My_DDM_1'
                    )

# DDM_prefs = FunctionPreferenceSet(
#                 owner=myMechanism_1,
#                 prefs = {
#                     kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#                     kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
#                     kpExecuteMethodRuntimeParamsPref: PreferenceEntry(ModulationOperation.OVERRIDE,PreferenceLevel.CATEGORY)})

myMechanism_2 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:0.3,
                                                        kwDDM_Threshold:1.0},
                             # kwDDM_AnalyticSolution:kwDDM_NavarroAndFuss  # Note: this requires matlab engine be installed
                             kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                     prefs=DDM_prefs,
                     # prefs = {kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                     #          kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
                     #          },
                     # prefs = {kpReportOutputPref: False,
                     #          kpVerbosePref: False
                     #          },
                     name='My_DDM_2'
                     )


# myMechanism_3 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:-0.5,
#                                                         kwDDM_Threshold:2.0},
#                              # kwDDM_AnalyticSolution:kwDDM_NavarroAndFuss  # Note: this requires matlab engine be installed
#                              kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
#                      # prefs=DDM_prefs
#                      prefs = {kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE),
#                               kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
#                               },
#                      # prefs = {kpReportOutputPref: False,
#                      #          kpVerbosePref: False
#                      #          },
#                      name='My_DDM_3'
#                      )

# QUESTION
myMechanism_3 = DDM(params={kwExecuteMethodParams:{kwKwDDM_StartingPoint:2.0, kwDDM_Threshold:2.0}, # -> LOGS DRIFT RATE, BUT NOT BIAS OR THRESHOLD
# myMechanism_3 = DDM(params={kwExecuteMethodParams:{kwKwDDM_StartingPoint:2.0, kwDDM_Threshold:2.0}, # -> LOGS DRIFT RATE ONLY
# myMechanism_3 = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate: 2.0}, # -> LOGS BIAS AND THRESHOLD BUT NOT DRIFT RATE
                             # kwDDM_AnalyticSolution:kwDDM_NavarroAndFuss  # Note: this requires matlab engine be installed
                             kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                     # prefs=DDM_prefs,
                     prefs = {kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE),
                              kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
                              },
                     # prefs = {kpReportOutputPref: False,
                     #          kpVerbosePref: False
                     #          },
                     name='My_DDM_3'
                     )



# process_prefs = FunctionPreferenceSet(prefs={kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
#                                              kpVerbosePref: PreferenceEntry(True,PreferenceLevel.SYSTEM)})
# from Functions.Utility import LinearCombination
# y = Process_Base(params={kwConfiguration:[(myMechanism,
#                                            {
#                                                # kwInputStateParams:{},
#                                                kwParameterStateParams:
#                                                    {kwParamModulationOperation: ModulationOperation.MULTIPLY, # B
#                                                     kwDDM_DriftRate:(30.0,
#                                                                      ModulationOperation.MULTIPLY), # C
#                                                     kwDDM_Threshold:20.0,   # Execute method param for Mechanism execute method
#                                                     kwExecuteMethodParams:  # Execute method params for parameter states execute method
#                                                        {LinearCombination.kwOffset: 100}, # A
#                                                     # kwProjectionParams:
#                                                     #     {Linear.kwIntercept: 1},
#                                                     },
#                                            }),
#                                           (myMechanism,
#                                            {
#                                                # kwInputStateParams:{},
#                                                kwParameterStateParams:
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

#myMapping = Mapping(sender = myMechanism_1.outputState, receiver= myMechanism_3.inputState)

#myMapping = Mapping(sender = myMechanism_3.outputState, receiver= myMechanism_2.inputState)

# NOTE: THIS ASSIGNS THE CONTROL SIGNAL TO THE *INPUT* OF MY_DDM_3;
#       SINCE IT PASSES ITS ALLOCATION (1) UNMODIFIED, THAT GETS ADDED TO THE INPUT OF MY_DDM 3
#       IF YOU WANT TO EXPLICITLY HAVE IT CONTROL A PARAMETER OF MY_DDM_3, IT'S RECEIVER NEEDS
#       TO BE THE CORRESPONDING PARAMETER_STATE
# control = ControlSignal(receiver=myMechanism_3.inputState,
#                         prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)})

# control = ControlSignal(receiver=myMechanism_3.executeMethodParameterStates[kwDDM_DriftRate],
#                         prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)})
#
# control.set_intensity(100)

z = Process_Base(params={kwConfiguration:[myMechanism_1, myMechanism_2, myMechanism_3]},
                 prefs={kpReportOutputPref: True,
                        kpVerbosePref: False})
                 # prefs={kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                 #                             kpVerbosePref: PreferenceEntry(True,PreferenceLevel.SYSTEM)})

# z = Process_Base(params={kwConfiguration:[DDM, DDM, DDM]})
# z = Process_Base(params={kwConfiguration:[mechanism()]})
z.execute(1)
# #

# 1) How to specify modulation operation when instantiating mechanism?
# 2) How are drift rates getting calculated in mechanism 2?
# 3) why is threshold 4.0 in 3rd mechanism?
# CLARIFIED) Order of mechanisms in configuration matters? Just to check. Maybe should give warning if order violates mappings?
# CLARIFIED) What is allocation_source in ControlSignal?
# 6) ControlSignal.set_intensity() doesn't seem to work?
