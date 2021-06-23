"""
This module provides utilities used to schedule the execution of psyneulink components

https://princetonuniversity.github.io/PsyNeuLink/Scheduling.html
"""
import graph_scheduler

# timescale mappings to pnl versions
# must be done before importing condition module, which relies on these
# mappings already being present
_time_scale_mappings = {
    "TIME_STEP": graph_scheduler.time.TimeScale.CONSIDERATION_SET_EXECUTION,
    "TRIAL": graph_scheduler.time.TimeScale.ENVIRONMENT_STATE_UPDATE,
    "RUN": graph_scheduler.time.TimeScale.ENVIRONMENT_SEQUENCE,
}
for our_ts, scheduler_ts in _time_scale_mappings.items():
    graph_scheduler.set_time_scale_alias(our_ts, scheduler_ts)


from . import condition  # noqa: E402
from . import scheduler  # noqa: E402
from . import time  # noqa: E402

from .condition import *  # noqa: E402
from .scheduler import *  # noqa: E402
from .time import *  # noqa: E402

__all__ = list(condition.__all__)
__all__.extend(scheduler.__all__)
__all__.extend(time.__all__)

del graph_scheduler
