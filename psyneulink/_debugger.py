# -*- coding: utf-8 -*-
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
# **************************************  _debugger  *********************************************
"""Internal debugger hook for stepping through PsyNeuLink execution.

This module is **internal** (note the leading underscore). The surface here may
change without notice until it is promoted to a public ``psyneulink.debugger``
module after at least one external consumer has stabilized on it.

Design
------
Replaces the inert ``assert 'DEBUGGING BREAK POINT: ...'`` literals that were
sprinkled through the codebase with a pluggable hook. When no listener is
registered, ``step()`` is a single ``is not None`` comparison and returns
immediately — zero behavioral change and no measurable performance cost.

A listener is a callable ``(category, locals_provider) -> None``. The locals
provider is a zero-argument callable that returns a dict of relevant state
captured at the call site; building that dict is deferred so listeners that
ignore a category (or only inspect the category enum) pay nothing.

Categories map to scheduler-level execution boundaries plus a handful of
non-scheduler events (parameter setting, init completion, pytorch step). They
are stable identifiers; new categories may be added but existing ones are not
renamed or repurposed.

Usage from a consumer (e.g. PsyNeuView's stepping worker)
---------------------------------------------------------
::

    from psyneulink import _debugger

    def listener(category, locals_provider):
        if category is _debugger.BreakpointCategory.NODE_EXECUTION:
            state = locals_provider()           # only built when consumed
            archive.snapshot(state)

    _debugger.set_listener(listener)
    composition.run(inputs)
    _debugger.set_listener(None)                # tear down when done

Locals schema per category (informal — listeners key-check what they need)
--------------------------------------------------------------------------
BEGINNING_OF_RUN        scheduler, context, num_trials
END_OF_RUN              scheduler, context, results
BEGINNING_OF_TRIAL      trial_num, scheduler, context
END_OF_TRIAL            trial_num, scheduler, context, outputs
EXECUTION_SET           execution_set, scheduler, context
END_OF_EXECUTION_SET    execution_set, scheduler, context, outputs
INPUTS_TO_NODE          node, execution_set, scheduler, context
NODE_EXECUTION          node, execution_set, scheduler, context,
                        output_port_element_names  (fires post-execute)
PARAMETER_SETTING       parameter, owner, value, context
EXCEPTION               exception, scheduler, context, trial_num
END_OF_INIT             component
PYTORCH_STEP            wrapper, composition

Categories may grow new keys over time without breaking existing listeners.
"""

from enum import Enum
from typing import Callable, Optional


class BreakpointCategory(Enum):
    """Identifier for a breakpoint emission site."""
    BEGINNING_OF_RUN     = "beginning_of_run"
    END_OF_RUN           = "end_of_run"
    BEGINNING_OF_TRIAL   = "beginning_of_trial"
    END_OF_TRIAL         = "end_of_trial"
    EXECUTION_SET        = "execution_set"
    END_OF_EXECUTION_SET = "end_of_execution_set"
    INPUTS_TO_NODE       = "inputs_to_node"
    NODE_EXECUTION       = "node_execution"
    PARAMETER_SETTING    = "parameter_setting"
    EXCEPTION            = "exception"
    END_OF_INIT          = "end_of_init"
    PYTORCH_STEP         = "pytorch_step"


Listener = Callable[["BreakpointCategory", Callable[[], dict]], None]

_listener: Optional[Listener] = None


def set_listener(listener: Optional[Listener]) -> None:
    """Register the single global listener, or pass ``None`` to clear.

    Single-listener by design; consumers that need to fan out can wrap multiple
    callables themselves. Assignment is atomic in CPython so no lock is needed
    for the common case (set once at startup, clear at teardown).
    """
    global _listener
    _listener = listener


def get_listener() -> Optional[Listener]:
    """Return the currently registered listener (or ``None``)."""
    return _listener


def step(category: BreakpointCategory, locals_provider: Callable[[], dict]) -> None:
    """Emit a breakpoint event. No-op when no listener is attached.

    The hot path is a single ``is not None`` check; the locals_provider is only
    invoked if a listener is registered (and then only if the listener chooses
    to call it).
    """
    if _listener is not None:
        _listener(category, locals_provider)
