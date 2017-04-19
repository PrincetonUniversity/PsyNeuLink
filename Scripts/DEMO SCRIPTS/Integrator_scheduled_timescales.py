from PsyNeuLink.Components.System import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Functions.Function import Linear, Integrator
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import AfterNCalls, Any, AtPass, EveryNCalls

process_prefs = {
    REPORT_OUTPUT_PREF: True,
    VERBOSE_PREF: False
}

A = TransferMechanism(
    name='A',
    default_input_value = [0],
    function=Linear(slope=2.0),
    prefs={REPORT_OUTPUT_PREF: PreferenceEntry(True,PreferenceLevel.INSTANCE)}
)

B = IntegratorMechanism(
    name='B',
    default_input_value = [0],
    function=Integrator(
        rate=.5,
        integration_type=SIMPLE
    )
)

C = IntegratorMechanism(
    name='C',
    default_input_value = [0],
    function=Integrator(
        rate=.5,
        integration_type=SIMPLE
    )
)

D = TransferMechanism(
    name='D',
    default_input_value = [0],
    function=Linear(slope=1.0),
    prefs={REPORT_OUTPUT_PREF: PreferenceEntry(True,PreferenceLevel.INSTANCE)}
)

p = process(
    default_input_value = [0],
    pathway = [A, B, D],
    name = 'p'
)

q = process(
    default_input_value = [0],
    pathway = [A, C, D],
    name = 'q',
    prefs=process_prefs
)

s = system(
    processes=[p, q],
    name = 's'
)

stim_list = {A: [[1]]}

s.scheduler = Scheduler(system=s)
s.scheduler.add_condition(A, Any(AtPass(0), EveryNCalls(D, 1)))
s.scheduler.add_condition(C, EveryNCalls(B, 5))
s.scheduler.add_condition(D, EveryNCalls(C, 1))

term_conds = {TimeScale.TRIAL: AfterNCalls(D, 1)}

results = s.run(
    inputs=stim_list,
    termination_conditions=term_conds
)
print(results)