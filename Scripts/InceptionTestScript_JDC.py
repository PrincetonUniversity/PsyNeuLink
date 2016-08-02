from Functions.Process import Process_Base
from Functions.Mechanisms.DDM import *
from Globals.Keywords import *
from Functions.Utility import UtilityRegistry
from Functions.States.State import StateRegistry
from Globals.Preferences.FunctionPreferenceSet import FunctionPreferenceSet
import csv

# does not incorporate scaling factor for all words weighted by memory score
# does not incorporate perceptual boost scaling factor for target and probe words that were visually percieved
# does not incorporate confidence of classifier.

DDM_prefs = FunctionPreferenceSet(
    prefs={
        kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE),
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

my_DDM = DDM(name='my_DDM')
my_DDM.prefs = DDM_prefs
my_DDM.prefs.level = PreferenceLevel.SYSTEM
my_DDM.prefs.inspect()

filepath = "SumDuringDelayTargetContextPerSubjectPerTrial.csv"
targetEvidence = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=0)

DDMEstimates = np.zeros(targetEvidence.shape)
myMechanism = DDM(params={kwExecuteMethodParams: {kwDDM_Threshold: (2.0),  # starting off with hardcoded defaults
                                                  kwKwDDM_StartingPoint: (0.0),
                                                  kwDDM_Noise: (0.5),
                                                  kwDDM_T0: (0.2),
                                                  },
                          kwDDM_AnalyticSolution: kwDDM_BogaczEtAl},
                  # matlab integration doesn't work, so sticking with BogaczEtAl
                  prefs=DDM_prefs,
                  name='My_DDM'
                  )
for row in range(targetEvidence.shape[0]): #for each subject
    for entry in range(targetEvidence.shape[1]): #for each trial
        driftRateForTrial = targetEvidence[row, entry] #get drift rate from subject data
        myMechanism.execute([driftRateForTrial])
        DDMEstimates[row, entry] = myMechanism.outputStates[kwDDM_RT_Mean].value #Not sure if this is the best way to get RTs

myMechanism.log.print_entries(ALL_ENTRIES, kwTime, kwValue)

# then correlate the DDM RTs with real subjects' RTs