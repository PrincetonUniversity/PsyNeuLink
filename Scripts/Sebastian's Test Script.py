from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import DRIFT_RATE
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

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
                    kpFunctionRuntimeParamsPref: PreferenceEntry(ModulationOperation.OVERRIDE,PreferenceLevel.INSTANCE)})

DDM.classPreferences = DDM_prefs
DDM.classPreferences.reportOutputPref = PreferenceEntry(False, PreferenceLevel.INSTANCE)

#region MAIN SCRIPT
myMechanism_1 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:(2.0, CONTROL_SIGNAL),
                                                   THRESHOLD:(10.0, CONTROL_SIGNAL)},
                            #                       kwParamModulationOperation: ModulationOperation.OVERRIDE},
                            # {kwParameterStateParams: {kwParamModulationOperation: ModulationOperation.OVERRIDE}},
                            # kwParamModulationOperation: ModulationOperation.OVERRIDE,
                            # kwDDM_AnalyticSolution:NAVARRO_AND_FUSS  # Note: this requires matlab engine be installed
                            kwDDM_AnalyticSolution:BOGACZ_ET_AL},
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
#                     kpFunctionRuntimeParamsPref: PreferenceEntry(ModulationOperation.OVERRIDE,PreferenceLevel.CATEGORY)})

myMechanism_2 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:0.3,
                                                        THRESHOLD:1.0},
                             # kwDDM_AnalyticSolution:NAVARRO_AND_FUSS  # Note: this requires matlab engine be installed
                             kwDDM_AnalyticSolution:BOGACZ_ET_AL},
                     prefs=DDM_prefs,
                     # prefs = {kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                     #          kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
                     #          },
                     # prefs = {kpReportOutputPref: False,
                     #          kpVerbosePref: False
                     #          },
                     name='My_DDM_2'
                     )


# myMechanism_3 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:-0.5,
#                                                         THRESHOLD:2.0},
#                              # kwDDM_AnalyticSolution:NAVARRO_AND_FUSS  # Note: this requires matlab engine be installed
#                              kwDDM_AnalyticSolution:BOGACZ_ET_AL},
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
myMechanism_3 = DDM(params={FUNCTION_PARAMS:{kwKwDDM_StartingPoint:2.0, THRESHOLD:2.0}, # -> LOGS DRIFT RATE, BUT NOT BIAS OR THRESHOLD
# myMechanism_3 = DDM(params={FUNCTION_PARAMS:{kwKwDDM_StartingPoint:2.0, THRESHOLD:2.0}, # -> LOGS DRIFT RATE ONLY
# myMechanism_3 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE: 2.0}, # -> LOGS BIAS AND THRESHOLD BUT NOT DRIFT RATE
                             # kwDDM_AnalyticSolution:NAVARRO_AND_FUSS  # Note: this requires matlab engine be installed
                             kwDDM_AnalyticSolution:BOGACZ_ET_AL},
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
# y = Process_Base(params={CONFIGURATION:[(myMechanism,
#                                            {
#                                                # kwInputStateParams:{},
#                                                kwParameterStateParams:
#                                                    {kwParamModulationOperation: ModulationOperation.MULTIPLY, # B
#                                                     DRIFT_RATE:(30.0,
#                                                                      ModulationOperation.MULTIPLY), # C
#                                                     THRESHOLD:20.0,   # Execute method param for Mechanism execute method
#                                                     FUNCTION_PARAMS:  # Execute method params for parameter states execute method
#                                                        {LinearCombination.OFFSET: 100}, # A
#                                                     # kwProjectionParams:
#                                                     #     {Linear.INTERCEPT: 1},
#                                                     },
#                                            }),
#                                           (myMechanism,
#                                            {
#                                                # kwInputStateParams:{},
#                                                kwParameterStateParams:
#                                                    {kwParamModulationOperation: ModulationOperation.MULTIPLY, # B
#                                                     DRIFT_RATE:(30.0,
#                                                                      ModulationOperation.MULTIPLY), # C
#                                                     THRESHOLD:20.0,
#                                                     FUNCTION_PARAMS:
#                                                        {LinearCombination.OFFSET: 100}, # A
#                                                     # kwProjectionParams:
#                                                     #     {Linear.INTERCEPT: 1},
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

# control = ControlSignal(receiver=myMechanism_3.parameterStates[DRIFT_RATE],
#                         prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)})
#
# control.set_intensity(100)

z = Process_Base(params={CONFIGURATION:[myMechanism_1, myMechanism_2, myMechanism_3]},
                 prefs={kpReportOutputPref: True,
                        kpVerbosePref: False})
                 # prefs={kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                 #                             kpVerbosePref: PreferenceEntry(True,PreferenceLevel.SYSTEM)})

# z = Process_Base(params={CONFIGURATION:[DDM, DDM, DDM]})
# z = Process_Base(params={CONFIGURATION:[mechanism()]})
z.execute(1)
# #

# 1) How to specify modulation operation when instantiating mechanism?
# 2) How are drift rates getting calculated in mechanism 2?
# 3) why is threshold 4.0 in 3rd mechanism?
# CLARIFIED) Order of mechanisms in configuration matters? Just to check. Maybe should give warning if order violates mappings?
# CLARIFIED) What is allocation_source in ControlSignal?
# 6) ControlSignal.set_intensity() doesn't seem to work?
