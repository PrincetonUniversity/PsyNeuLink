import logging

from PsyNeuLink.Components.System import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Functions.Function import Linear, Integrator
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import Any, AfterNCalls, AtPass, WhenFinished
from PsyNeuLink.Globals.Keywords import DIFFUSION
from PsyNeuLink.Globals.TimeScale import TimeScale

logger = logging.getLogger(__name__)

o = TransferMechanism(
    name='origin',
    default_input_value = [0],
    function=Linear(slope=.5),
    prefs={REPORT_OUTPUT_PREF: PreferenceEntry(True,PreferenceLevel.INSTANCE)}
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
    prefs={REPORT_OUTPUT_PREF: PreferenceEntry(True,PreferenceLevel.INSTANCE)}
)

p = process(
    default_input_value = [0],
    pathway = [o, ddm, term],
    name = 'p',
)

# origin → DDM → terminal
s = system(
    processes=[p],
    name='s',
)

stim_list = {o: [[1]]}

s.scheduler_processing = Scheduler(system=s)
s.scheduler_processing.add_condition(o, AtPass(0))
# ddm has default condition of Always - run at every chance
s.scheduler_processing.add_condition(term, Any(WhenFinished(ddm), AfterNCalls(ddm, 10)))

term_conds = {TimeScale.TRIAL: AfterNCalls(term, 1)}

s.show_graph()
results = s.run(
    inputs=stim_list,
    termination_processing=term_conds
)
logger.info('System result: {0}'.format(results))
