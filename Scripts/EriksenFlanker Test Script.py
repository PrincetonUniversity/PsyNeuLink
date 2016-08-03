from PsyNeuLink.Functions.Mechanisms.AdaptiveIntegrator import *
from PsyNeuLink.Functions.Mechanisms.SigmoidLayer import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Functions.System import System_Base
from PsyNeuLink.Globals.Keywords import *

#region Preferences
DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

#endregion

#region Mechanisms
Target = SigmoidLayer(name='Target', params={kwExecuteMethodParams:{kwSigmoidLayer_Gain:(1.0, kwControlSignal)}})
Distractor = SigmoidLayer(name='Distractor')


Decision = DDM(params={kwExecuteMethodParams:{kwDDM_Threshold:(10.0, kwControlSignal)},
                          kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                  prefs = DDM_prefs,
                  name='Decision'
                  )
#endregion

#region Projections

#endregion

#region Processes
TargetProcess = Process_Base(default_input_value=[1],
                                    params={kwConfiguration:[(Target, 1),
                                                             kwIdentityMatrix,
                                                             (Decision, 1)]},
                                    prefs = process_prefs,
                                    name = 'TargetProcess')

DistractorProcess = Process_Base(default_input_value=[1],
                                    params={kwConfiguration:[(Distractor, 1),
                                                             kwIdentityMatrix,
                                                             (Decision, 1)]},
                                    prefs = process_prefs,
                                    name = 'DistractorProcess')


#region System
mySystem = System_Base(params={kwProcesses:[TargetProcess, DistractorProcess]})
#endregion

#region Run
mySystem.execute([[1], [1]])
#endregion
