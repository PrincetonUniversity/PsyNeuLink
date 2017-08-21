from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import System_Base
from PsyNeuLink.Components.Functions.Function import Exponential, Linear
from PsyNeuLink.Globals.Keywords import *

if MPI_IMPLEMENTATION:
    import time
    from mpi4py import MPI
    Comm = MPI.COMM_WORLD
    Comm.Barrier()
    startTime = time.time()
    Comm.Barrier()

#region Preferences
DDM_prefs = ComponentPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = ComponentPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))
#endregion

#region Mechanisms
Input = LinearMechanism(name='Input')
Reward = LinearMechanism(name='Reward')
Decision = DDM(
               # drift_rate=(2.0, CONTROL_PROJECTION),
               # drift_rate=(2.0, ControlProjection),
               # drift_rate=(2.0, ControlProjection()),
               # drift_rate=(2.0, ControlProjection(function=Linear)),
               drift_rate=(2.0, ControlProjection(function=Linear(slope=2, intercept=10),
                                              # allocation_samples=np.arange(.1, 1.01, .1))),
                                              allocation_samples=[0, .1, .5, 1.0])),
               # drift_rate=(2.0, ControlProjection(function=Exponential)),
               # drift_rate=(2.0, ControlProjection(function=Exponential(rate=2, scale=10))),
               # threshold=(5.0, CONTROL_PROJECTION),
               # threshold=(5.0, ControlProjection()),
               # threshold=(5.0, ControlProjection(function=Exponential)),
               # threshold=(5.0, ControlProjection(function=Exponential(slope=2, intercept=10))),
               threshold=(5.0, ControlProjection(function=Exponential(rate=2, scale=10))),
               # threshold=(5.0, ControlProjection(function=Exponential)),
               # threshold=(5.0, CONTROL_PROJECTION),
               analytic_solution=kwBogaczEtAl,
               prefs = DDM_prefs,
               name='Decision'
               )
#endregion


#region Processes
TaskExecutionProcess = process(default_variable=[0],
                               pathway=[(Input, 0), IDENTITY_MATRIX, (Decision, 0)],
                               prefs = process_prefs,
                               name = 'TaskExecutionProcess')

RewardProcess = process(default_variable=[0],
                        pathway=[(Reward, 1)],
                        prefs = process_prefs,
                        name = 'RewardProcess')
#endregion

#region System
mySystem = System_Base(processes=[TaskExecutionProcess, RewardProcess],
                       monitor_for_control=[Reward, ERROR_RATE, (RESPONSE_TIME, -1, 1)],
                       name='Test System')
#endregion

#region Show
mySystem.show()
mySystem.controller.show()
#endregion

#region Run

for i in range(2):
    # Present stimulus:
    CentralClock.trial = i
    CentralClock.time_step = 0
    mySystem.execute([[0.5],[0]])
    print ('\nTRIAL: {}; Time Step: {}\n{}\n{}'.format(CentralClock.trial, CentralClock.time_step,
                                                     mySystem.terminal_mechanisms.outputStateNames,
                                                     mySystem.terminal_mechanisms.outputStateValues))

    # Present feedback:
    CentralClock.time_step = 1
    mySystem.execute([[0],[1]])
    print ('\nTRIAL: {}; Time Step: {}\n{}\n{}'.format(CentralClock.trial, CentralClock.time_step,
                                                     mySystem.terminal_mechanisms.outputStateNames,
                                                     mySystem.terminal_mechanisms.outputStateValues))

#endregion

if MPI_IMPLEMENTATION:
    Comm.Barrier()
    endTime = time.time()
    Comm.Barrier()

    print("\nRuntime: ", endTime-startTime)

print('DONE')
