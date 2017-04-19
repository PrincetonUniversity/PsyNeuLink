from PsyNeuLink.Components.System import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Functions.Function import Linear, Integrator
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import AfterNCalls, WhenFinished
from PsyNeuLink.Globals.Keywords import DIFFUSION
from PsyNeuLink.Globals.TimeScale import TimeScale

process_prefs = {
    REPORT_OUTPUT_PREF: True,
    VERBOSE_PREF: False
}

o = TransferMechanism(
    name='origin',
    default_input_value = [0],
    function=Linear(slope=.5),
)

ddm = DDM(
    function=Integrator(
        integration_type = DIFFUSION,
        noise=0.5
    ),
    name='ddm',
    time_scale=TimeScale.TIME_STEP,
    thresh=10
)

term = TransferMechanism(
    name='terminal',
    default_input_value = [0],
    function=Linear(slope=2.0),
)

p = process(
    default_input_value = [0],
    pathway = [o, ddm, term],
    name = 'p',
    prefs=process_prefs
)

s = system(
    processes=[p],
    name='s',
)

stim_list = {o: [[1]]}

s.scheduler = Scheduler(system=s)
s.scheduler.add_condition(term, WhenFinished(ddm))

term_conds = {TimeScale.TRIAL: AfterNCalls(term, 1)}

results = s.run(
    inputs=stim_list,
    termination_conditions=term_conds
)
print(results)
