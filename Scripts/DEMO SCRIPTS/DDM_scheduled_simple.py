from PsyNeuLink.Components.Functions.Function import Integrator, Linear
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.System import *
from PsyNeuLink.Globals.Keywords import DIFFUSION
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms import DDM
from PsyNeuLink.Scheduling.Condition import AfterNCalls, AtPass, WhenFinished
from PsyNeuLink.Scheduling.Scheduler import Scheduler
from PsyNeuLink.Scheduling.TimeScale import TimeScale

logger = logging.getLogger(__name__)

o = TransferMechanism(
    name='origin',
    default_variable = [0],
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
    default_variable = [0],
    function=Linear(slope=2.0),
    prefs={REPORT_OUTPUT_PREF: PreferenceEntry(True,PreferenceLevel.INSTANCE)}
)

p = process(
    default_variable = [0],
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
s.scheduler_processing.add_condition(term, WhenFinished(ddm))

term_conds = {TimeScale.TRIAL: AfterNCalls(term, 1)}

s.show_graph()
results = s.run(
    inputs=stim_list,
    termination_processing=term_conds
)
logger.info('System result: {0}'.format(results))
