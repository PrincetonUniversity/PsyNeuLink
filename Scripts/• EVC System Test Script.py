from Functions.System import System_Base
from Functions.Process import Process_Base
from Functions.Mechanisms.DDM import *
from Functions.Mechanisms.SigmoidLayer import *
from Functions.Mechanisms.AdaptiveIntegrator import *
from Functions.Mechanisms.EVCMechanism import *
from Globals.Keywords import *
from Functions.Utility import UtilityRegistry
from Functions.MechanismStates.MechanismState import MechanismStateRegistry


# Preferences
DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))


# Mechanisms

#region MAIN SCRIPT
Input = SigmoidLayer()
Reward = SigmoidLayer()
StimulusPrediction = AdaptiveIntegratorMechanism()
RewardPrediction = AdaptiveIntegratorMechanism()
Decision = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:(1.0, kwControlSignal),
                                                 kwDDM_Threshold:(10.0, kwControlSignal)},
                          kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                  prefs = DDM_prefs,
                  name='My_DDM'
                  )
EVC = EVCMechanism()

TaskExecutionProcess = Process_Base(default_input_value=[0],
                           params={kwConfiguration:[(Input, 1),
                                                    kwIdentityMatrix,
                                                    (Decision, 1)]}, # WILL THIS GET TWO inputStates IN EVC?
                           prefs = process_prefs)

RewardProcess = Process_Base(default_input_value=[0],
                           params={kwConfiguration:[(Reward, 2),
                                                    kwIdentityMatrix,
                                                    (RewardPrediction, 3)]},
                           prefs = process_prefs)

StimulusPredictionProcess = Process_Base(default_input_value=[0],
                           params={kwConfiguration:[(Input, 1),
                                                    kwIdentityMatrix
                                                    (StimulusPrediction, 3),
                                                    kwIdentityMatrix,
                                                    (Decision, 3)]}, # WILL THIS GET TWO inputStates IN EVC?
                           prefs = process_prefs)


mySystem = System_Base(params={kwProcesses:[TaskExecutionProcess, RewardProcess, StimulusPredictionProcess]})

mySystem.execute([[1]])

