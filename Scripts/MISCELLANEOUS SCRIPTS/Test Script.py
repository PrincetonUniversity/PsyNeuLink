from PsyNeuLink.Components.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

# x = Process_Base()
# x.execute(10.0)


# DDM_prefs = ComponentPreferenceSet(
#                 owner=DDM,
#                 verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
#                 reportOutput_pref=PreferenceEntry(True,PreferenceLevel.SYSTEM),
#                 functionRuntimeParams_pref=PreferenceEntry(Modulation.OVERRIDE,PreferenceLevel.CATEGORY),
#                 name='Reassigned'

DDM_prefs = ComponentPreferenceSet(
# FIX: AttributeError: 'ComponentPreferenceSet' object has no attribute '_verbose_pref'
#                 owner=DDM,
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(False,PreferenceLevel.SYSTEM),
                    kpRuntimeParamModulationPref: PreferenceEntry(Modulation.OVERRIDE,PreferenceLevel.CATEGORY)})


DDM_prefs.verbosePref = PreferenceEntry(True,PreferenceLevel.INSTANCE)
# FIX: HAD TO DISABLE PreferenceSet.validate_log
# DDM_prefs.logPref = LogEntry.TIME_STAMP
# DDM_prefs.logPref = (LogEntry.INPUT_VALUE | LogEntry.OUTPUT_VALUE)

# DDM_prefs.reportOutputPref = 'Hello'

DDM.classPreferences = DDM_prefs

DDM_prefs2 = ComponentPreferenceSet(
                owner=DDM,
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.SYSTEM),
                    kpRuntimeParamModulationPref: PreferenceEntry(Modulation.OVERRIDE,PreferenceLevel.INSTANCE)})

my_DDM = DDM(name='my_DDM')
my_DDM.prefs = DDM_prefs
my_DDM.prefs.level = PreferenceLevel.SYSTEM

# my_DDM.prefs.verbosePref = PreferenceLevel.SYSTEM
# from Components.Projections.ControlProjection import LogEntry
# my_DDM.prefs.logPref = LogEntry.TIME_STAMP

# FIX: SHOULDN'T BE ABLE TO ASSIGN enum TO PREF THAT DOESN'T REQUIRE ONE:
# my_DDM.prefs.verbosePref = LogEntry.TIME_STAMP

my_DDM.prefs.show()

#region MAIN SCRIPT
myMechanism = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:(2.0, CONTROL_PROJECTION),
                                                 THRESHOLD:(10.0, CONTROL_PROJECTION),
                                                 kwKwDDM_StartingPoint:(0.5, CONTROL_PROJECTION)},
# myMechanism = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:2.0,
#                                                  THRESHOLD:10.0},
                          # kwDDM_AnalyticSolution:kwNavarrosAndFuss  # Note: this requires matlab engine be installed
                          kwDDM_AnalyticSolution:kwBogaczEtAl},
                  # prefs=DDM_prefs,
                  # prefs = {kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.CATEGORY),
                  #          kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
                  #          },
                  prefs = {kpReportOutputPref: PreferenceLevel.SYSTEM,
                           # kpVerbosePref: PreferenceLevel.INSTANCE
                           },
                  # prefs = {kpReportOutputPref: False,
                  #          kpVerbosePref: False
                  #          },
                  name='My_DDM'
                  )


#region ADDITIONAL MECHANISMS
# # DDM.classPreferences.reportOutputPref = PreferenceEntry(False, PreferenceLevel.INSTANCE)
#
# # my_Mechanism_2 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:2.0,
# #                                                         THRESHOLD:1.0},
# #                              # kwDDM_AnalyticSolution:kwNavarrosAndFuss  # Note: this requires matlab engine be installed
# #                              kwDDM_AnalyticSolution:kwBogaczEtAl},
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
# # my_Mechanism_3 = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:2.0,
# #                                                         THRESHOLD:1.0},
# #                              # kwDDM_AnalyticSolution:kwNavarrosAndFuss  # Note: this requires matlab engine be installed
# #                              kwDDM_AnalyticSolution:kwBogaczEtAl},
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
# # process_prefs = ComponentPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
# #                                        verbose_pref=PreferenceEntry(True,PreferenceLevel.SYSTEM))
# from Components.Function import LinearCombination
# y = Process_Base(params={PATHWAY:[(myMechanism,
#                                            {
#                                                # INPUT_STATE_PARAMS:{},
#                                                PARAMETER_STATE_PARAMS:
#                                                    {PARAMETER_MODULATION_OPERATION: Modulation.MULTIPLY, # B
#                                                     DRIFT_RATE:(30.0,
#                                                                      Modulation.MULTIPLY), # C
#                                                     THRESHOLD:20.0,
#                                                     FUNCTION_PARAMS:
#                                                        {LinearCombination.OFFSET: 100}, # A
#                                                     # PROJECTION_PARAMS:
#                                                     #     {Linear.INTERCEPT: 1},
#                                                     },
#                                            }),
#                                           (myMechanism,
#                                            {
#                                                # INPUT_STATE_PARAMS:{},
#                                                PARAMETER_STATE_PARAMS:
#                                                    {PARAMETER_MODULATION_OPERATION: Modulation.MULTIPLY, # B
#                                                     DRIFT_RATE:(30.0,
#                                                                      Modulation.MULTIPLY), # C
#                                                     THRESHOLD:20.0,
#                                                     FUNCTION_PARAMS:
#                                                        {LinearCombination.OFFSET: 100}, # A
#                                                     # PROJECTION_PARAMS:
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

# z = Process_Base(params={PATHWAY:[myMechanism]})
z = Process_Base(params={PATHWAY:[myMechanism, myMechanism]})
# z = Process_Base(params={PATHWAY:[DDM, DDM, DDM]})
# z = Process_Base(params={PATHWAY:[mechanism()]})
z.execute(30)
# #

# myMechanism.log.print_entries(ALL_ENTRIES, kwTime, kwValue)
myMechanism.log.print_entries(ALL_ENTRIES, kwTime, kwValue)
