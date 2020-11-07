# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *************************************************  Context ***********************************************************

"""
.. _Context_Overview:

Overview
--------
The Context class is used to pass information about execution and state.  It is generally
created at runtime, and updated under various operating conditions. Its primary
attribute is `flags <Context.flags>` - a binary vector, the individual flags of which are specified using the
`ContextFlags` enum.  The `flags <Context.flags>` attribute is divided functionally into the two fields:

  * `execution_phase <Context.execution_phase>` - phase of execution of the Component;

  * `source <Context.source>` - source of a call to a method belonging to or operating on the Component.

Each field can be addressed using the corresponding property of the class; only one source
flag may be set, but in some cases multiple execution_phase flags may be set
(although see individual property documentation for exceptions).

Context and Logging
-------------------

The `flags <Context.flags>` attribute is used by `Log` to identify conditions for logging (see).  Accordingly, the
`LogCondition`\\(s) used to specify such conditions in the `set_log_conditions <Log.set_log_conditions>` method of Log
are a subset of (and are aliased to) the flags in `ContextFlags`.

.. _Context_Additional_Attributes:

Additional Attributes
---------------------

In addition to `flags <Context.flags>`, `execution_phase <Context.execution_phase>`, and
`source <Context.source>`, Context has four other attributes that record information
relevant to the operating state of the Component:

    `owner <Context.owner>`
      the Component to which the Context belongs (assigned to its `context <Component.context>` attribute;
    `flags_string <Context.flags_string>`
      a string containing the names of the flags currently set in each of the fields of the `flags <Context.flags>`
      attribute;
    `composition <Context.composition>`
      the `Composition <Composition>` in which the Component is currently being executed;
    `execution_id <Context.execution_id>`
      the `execution_id` assigned to the Component by the Composition in which it is currently being executed;
    `execution_time <Context.execution_time>`
      the current time of the scheduler running the Composition within which the Component is currently being executed;
    `string <Context.string>`
      contains message(s) relevant to a method of the Component currently invoked or that is referencing the Component.
      In general, this contains a copy of the **context** argument passed to method of the Component or one that
      references it, but it is possible that future uses will involve other messages.

    .. _Context_String_Note:

    .. note::
       The `string <Context.string>` attribute of Context is not the same as, nor does it usually contain the same
       information as the string returned by the `flags_string <Context.flags_string>` method of Context.

COMMENT:
    IMPLEMENTATION NOTE: Use of ContextFlags in **context** argument of methods for context message-passing
        ContextFlags is also used for passing context messages to methods (in the **context** argument).

        Among other things, this is used to determine the source of call of a constructor (until someone
            proposes/implements a better method!).  This is used in several ways, for example:
            a) to determine whether an InputPort or OutputPort is being added as part of the construction process
              (e.g., for LearningMechanism) or by the user from the command line (see Mechanism.add_ports)

COMMENT

.. _Context_Class_Reference:

Class Reference
---------------

"""

import enum
import functools
import inspect
import warnings

from collections import defaultdict, namedtuple
from queue import Queue

import time as py_time  # "time" is declared below
import typecheck as tc

from psyneulink.core.globals.keywords import CONTEXT, CONTROL, EXECUTING, EXECUTION_PHASE, FLAGS, INITIALIZATION_STATUS, INITIALIZING, LEARNING, SEPARATOR_BAR, SOURCE, VALIDATE
from psyneulink.core.globals.utilities import get_deepcopy_with_shared


__all__ = [
    'Context',
    'ContextFlags',
    '_get_context',
    'INITIALIZATION_STATUS_FLAGS',
    'handle_external_context',
]

STATUS = 'status'

time = namedtuple('time', 'run trial pass_ time_step')

class ContextError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ContextFlags(enum.IntFlag):
    """Used to identify the initialization and execution status of a `Component <Component>`.

    Used when a Component's `value <Component.value>` or one of its attributes is being accessed.
    Also used to specify the context in which a value of the Component or its attribute is `logged <Log_Conditions>`..

    COMMENT:
        Used to by **context** argument of all methods to specify type of caller.
    COMMENT
    """

    UNSET = 0

    # initialization_status flags:
    DEFERRED_INIT = enum.auto()
    """Set if flagged for deferred initialization."""
    INITIALIZING = enum.auto()
    """Set during initialization of the Component."""
    VALIDATING = enum.auto()
    """Set during validation of the value of a Component or its attribute."""
    INITIALIZED = enum.auto()
    """Set after completion of initialization of the Component."""
    RESET = enum.auto()
    """Set on stateful Components when they are re-initialized."""
    UNINITIALIZED = enum.auto()
    """Default value set before initialization"""
    INITIALIZATION_MASK = DEFERRED_INIT | INITIALIZING | VALIDATING | INITIALIZED | RESET | UNINITIALIZED

    # execution_phase flags:
    PREPARING = enum.auto()
    """Set while `Composition is preparing to `execute <Composition_Execution>`."""
    PROCESSING = enum.auto()
    """Set while `Composition is `executing <Composition_Execution>` `ProcessingMechanisms <ProcessingMechanism>`."""
    LEARNING = enum.auto()
    """Set while `Composition is `executing <Composition_Execution>` `LearningMechanisms <LearningMechanism>`."""
    CONTROL = enum.auto()
    """Set while Composition's `controller <Composition.controller>` or its `ObjectiveMechanism` is executing."""
    IDLE = enum.auto()
    """Identifies condition in which no flags in the `execution_phase <Context.execution_phase>` are set.
    """
    EXECUTING = PROCESSING | LEARNING | CONTROL
    EXECUTION_PHASE_MASK = IDLE | PREPARING | EXECUTING

    # source (source-of-call) flags:
    COMMAND_LINE = enum.auto()
    """Direct call by user (either interactively from the command line, or in a script)."""
    CONSTRUCTOR = enum.auto()
    """Call from Component's constructor method."""
    METHOD = enum.auto()
    """Call by method of the Component other than its constructor."""
    COMPOSITION = enum.auto()
    """Call by a/the Composition to which the Component belongs."""

    NONE = enum.auto()

    """Call by a/the Composition to which the Component belongs."""
    SOURCE_MASK = COMMAND_LINE | CONSTRUCTOR | METHOD | COMPOSITION | NONE

    # runmode flags:
    DEFAULT_MODE = enum.auto()
    """Default mode"""
    LEARNING_MODE = enum.auto()
    """Set during `compositon.learn`"""
    SIMULATION_MODE = enum.auto()
    """Set during simulation by Composition.controller"""

    RUN_MODE_MASK = LEARNING_MODE | DEFAULT_MODE | SIMULATION_MODE

    ALL_FLAGS = INITIALIZATION_MASK | EXECUTION_PHASE_MASK | SOURCE_MASK | RUN_MODE_MASK

    @classmethod
    @tc.typecheck
    def _get_context_string(cls, condition_flags,
                            fields:tc.any(tc.enum(EXECUTION_PHASE,
                                                  SOURCE), set, list)={EXECUTION_PHASE,
                                                                       SOURCE},
                            string:tc.optional(str)=None):
        """Return string with the names of flags that are set in **condition_flags**

        If **fields** is specified, then only the names of the flag(s) in the specified field(s) are returned.
        The fields argument must be the name of a field (*EXECUTION_PHASE* or *SOURCE*)
        or a set or list of them.

        If **string** is specified, the string returned is prepended by **string**.
        """

        if string:
            string += ": "
        else:
            string = ""

        if isinstance(fields, str):
            fields = {fields}

        flagged_items = []
        # If OFF or ALL_FLAGS, just return that
        if condition_flags == ContextFlags.ALL_FLAGS:
            return ContextFlags.ALL_FLAGS.name
        if condition_flags == ContextFlags.UNSET:
            return ContextFlags.UNSET.name
        # Otherwise, append each flag's name to the string
        # for c in (EXECUTION_PHASE_FLAGS | SOURCE_FLAGS):
        #     if c & condition_flags:
        #        flagged_items.append(c.name)
        if EXECUTION_PHASE in fields:
            for c in EXECUTION_PHASE_FLAGS:
                if not condition_flags & ContextFlags.EXECUTION_PHASE_MASK:
                    flagged_items.append(ContextFlags.IDLE.name)
                    break
                if c & condition_flags:
                    flagged_items.append(c.name)
        if SOURCE in fields:
            for c in SOURCE_FLAGS:
                if not condition_flags & ContextFlags.SOURCE_MASK:
                    flagged_items.append(ContextFlags.NONE.name)
                    break
                if c & condition_flags:
                    flagged_items.append(c.name)
        string += ", ".join(flagged_items)
        return string

INITIALIZATION_STATUS_FLAGS = {ContextFlags.DEFERRED_INIT,
                               ContextFlags.INITIALIZING,
                               ContextFlags.VALIDATING,
                               ContextFlags.INITIALIZED,
                               ContextFlags.RESET,
                               ContextFlags.UNINITIALIZED}

EXECUTION_PHASE_FLAGS = {ContextFlags.PREPARING,
                         ContextFlags.PROCESSING,
                         ContextFlags.LEARNING,
                         ContextFlags.CONTROL,
                         ContextFlags.IDLE
                         }

SOURCE_FLAGS = {ContextFlags.COMMAND_LINE,
                ContextFlags.CONSTRUCTOR,
                ContextFlags.METHOD,
                ContextFlags.COMPOSITION,
                ContextFlags.NONE}

RUN_MODE_FLAGS = {
    ContextFlags.LEARNING_MODE,
    ContextFlags.DEFAULT_MODE,
    ContextFlags.SIMULATION_MODE,
}


class Context():
    """Used to indicate the state of initialization and phase of execution of a Component, as well as the source of
    call of a method;  also used to specify and identify `conditions <Log_Conditions>` for `logging <Log>`.


    Attributes
    ----------

    owner : Component
        Component to which the Context belongs.

    flags : binary vector
        represents the current operating context of the `owner <Context.owner>`; contains two fields
        `execution_phase <Context.execution_phase>`,
        and `source <Context.source>` (described below).

    flags_string : str
        contains the names of the flags currently set in each of the fields of the `flags <Context.flags>` attribute;
        note that this is *not* the same as the `string <Context.string>` attribute (see `note <Context_String_Note>`).

    execution_phase : field of flags attribute
        indicates the phase of execution of the Component;
        one or more of the following flags can be set:

            * `PREPARING <ContextFlags.PREPARING>`
            * `PROCESSING <ContextFlags.PROCESSING>`
            * `LEARNING <ContextFlags.LEARNING>`
            * `CONTROL <ContextFlags.CONTROL>`
            * `IDLE <ContextFlags.IDLE>`

        If `IDLE` is set, the Component is not being executed at the current time, and `flags_string
        <Context.flags_string>` will include *IDLE* in the string.  In some circumstances all of the
        `execution_phase <Context.execution_phase>` flags may be set (other than *IDLE* and *PREPARING*),
        in which case `flags_string <Context.flags_string>` will include *EXECUTING* in the string.

    source : field of the flags attribute
        indicates the source of a call to a method belonging to or referencing the Component;
        one of the following flags is always set:

            * `CONSTRUCTOR <ContextFlags.CONSTRUCTOR>`
            * `COMMAND_LINE <ContextFlags.COMMAND_LINE>`
            * `COMPOSITION <ContextFlags.COMPOSITION>`

    composition : Composition
      the `Composition <Composition>` in which the `owner <Context.owner>` is currently being executed.

    execution_id : str
      the execution_id assigned to the Component by the Composition in which it is currently being executed.

    execution_time : TimeScale
      current time of the `Scheduler` running the Composition within which the Component is currently being executed.

    string : str
      contains message(s) relevant to a method of the Component currently invoked or that is referencing the Component.
      In general, this contains a copy of the **context** argument passed to method of the Component or one that
      references it, but it is possible that future uses will involve other messages.  Note that this is *not* the
      same as the `flags_string <Context.flags_string>` attribute (see `note <Context_String_Note>`).

    rpc_pipeline : Queue
      queue to populate with messages for external environment in cases where execution was triggered via RPC call
      (e.g. through PsyNeuLinkView).

    """

    __name__ = 'Context'
    _deepcopy_shared_keys = {'owner', 'composition', '_composition'}

    def __init__(self,
                 owner=None,
                 composition=None,
                 flags=None,
                 execution_phase=ContextFlags.IDLE,
                 source=ContextFlags.NONE,
                 runmode=ContextFlags.DEFAULT_MODE,
                 execution_id=NotImplemented,
                 string:str='',
                 time=None,
                 rpc_pipeline:Queue=None):

        self.owner = owner
        self.composition = composition
        self._execution_phase = execution_phase
        self._source = source
        self._runmode = runmode

        if flags:
            if (execution_phase and not (flags & ContextFlags.EXECUTION_PHASE_MASK & execution_phase)):
                raise ContextError("Conflict in assignment to flags ({}) and execution_phase ({}) arguments "
                                   "of Context for {}".
                                   format(ContextFlags._get_context_string(flags & ContextFlags.EXECUTION_PHASE_MASK),
                                          ContextFlags._get_context_string(flags, EXECUTION_PHASE), self.owner.name))
            if not (flags & ContextFlags.SOURCE_MASK & source):
                raise ContextError("Conflict in assignment to flags ({}) and source ({}) arguments of Context for {}".
                                   format(ContextFlags._get_context_string(flags & ContextFlags.SOURCE_MASK),
                                          ContextFlags._get_context_string(flags, SOURCE),
                                          self.owner.name))
        if execution_id is NotImplemented:
            subsecond_res = 10 ** 6
            cur_time = py_time.time()
            subsec = int((cur_time * subsecond_res) % subsecond_res)
            time_format = f'%Y-%m-%d %H:%M:%S.{subsec} %Z%z'
            execution_id = py_time.strftime(time_format, py_time.localtime(cur_time))
        else:
            try:
                execution_id = execution_id.default_execution_id
            except AttributeError:
                pass

        self.execution_id = execution_id
        self.execution_time = None
        self.string = string
        self.rpc_pipeline = rpc_pipeline

    __deepcopy__ = get_deepcopy_with_shared(_deepcopy_shared_keys)

    @property
    def composition(self):
        try:
            return self._composition
        except AttributeError:
            self._composition = None

    @composition.setter
    def composition(self, composition):
        # from psyneulink.core.compositions.composition import Composition
        # if isinstance(composition, Composition):
        if (
            composition is None
            or composition.__class__.__name__ in {'Composition', 'AutodiffComposition'}
        ):
            self._composition = composition
        else:
            raise ContextError("Assignment to context.composition for {self.owner.name} ({composition}) "
                               "must be a Composition (or \'None\').")

    @property
    def flags(self):
        return self.execution_phase | self.source

    @flags.setter
    def flags(self, flags: ContextFlags):
        if isinstance(flags, (ContextFlags, int)):
            self.execution_phase = flags & ContextFlags.EXECUTION_PHASE_MASK
            self.source = flags & ContextFlags.SOURCE_MASK
        else:
            raise ContextError("\'{}\'{} argument in call to {} must be a {} or an int".
                               format(FLAGS, flags, self.__name__, ContextFlags.__name__))

    @property
    def flags_string(self):
        return ContextFlags._get_context_string(self.flags)

    @property
    def execution_phase(self):
        return self._execution_phase

    @execution_phase.setter
    def execution_phase(self, flag):
        """Check that flag is a valid execution_phase flag assignment"""
        if not flag:
            self._execution_phase = ContextFlags.IDLE
        elif flag not in EXECUTION_PHASE_FLAGS:
            raise ContextError(
                f"Attempt to assign more than one non-SIMULATION flag ({str(flag)}) to execution_phase"
            )
        elif (flag & ~ContextFlags.EXECUTION_PHASE_MASK):
            raise ContextError("Attempt to assign a flag ({}) to execution_phase "
                               "that is not an execution phase flag".
                               format(str(flag)))
        else:
            self._execution_phase = flag

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, flag):
        """Check that a flag is one and only one source flag"""
        if flag in SOURCE_FLAGS:
            self._source = flag
        elif not flag:
            self._source = ContextFlags.NONE
        elif not flag & ContextFlags.SOURCE_MASK:
            raise ContextError("Attempt to assign a flag ({}) to source that is not a source flag".
                               format(str(flag)))
        else:
            raise ContextError("Attempt to assign more than one flag ({}) to source".
                               format(str(flag)))

    @property
    def runmode(self):
        return self._runmode

    @runmode.setter
    def runmode(self, flag):
        """Check that a flag is one and only one run mode flag"""
        if (
            flag in RUN_MODE_FLAGS
            or (flag & ~ContextFlags.SIMULATION_MODE) in RUN_MODE_FLAGS
        ):
            self._runmode = flag
        elif not flag:
            self._runmode = ContextFlags.DEFAULT_MODE
        elif not flag & ContextFlags.RUN_MODE_MASK:
            raise ContextError("Attempt to assign a flag ({}) to run mode that is not a run mode flag".
                               format(str(flag)))
        else:
            raise ContextError("Attempt to assign more than one non-SIMULATION flag ({}) to run mode".
                               format(str(flag)))

    @property
    def execution_time(self):
        try:
            return self._execution_time
        except AttributeError:
            return None

    @execution_time.setter
    def execution_time(self, time):
        self._execution_time = time

    def update_execution_time(self):
        if self.execution & ContextFlags.EXECUTING:
            self.execution_time = _get_time(self.owner, self.most_recent_context.flags)
        else:
            raise ContextError("PROGRAM ERROR: attempt to call update_execution_time for {} "
                               "when 'EXECUTING' was not in its context".format(self.owner.name))

    def add_to_string(self, string):
        if self.string is None:
            self.string = string
        else:
            self.string = '{0} {1} {2}'.format(self.string, SEPARATOR_BAR, string)

    def _change_flags(self, *flags, operation=lambda attr, blank_flag, *flags: NotImplemented):
        # split by flag type to avoid extra costly binary operations on enum flags
        if all([flag in EXECUTION_PHASE_FLAGS for flag in flags]):
            self.execution_phase = operation(self.execution_phase, ContextFlags.IDLE, *flags)
        elif all([flag in SOURCE_FLAGS for flag in flags]):
            self.source = operation(self.source, ContextFlags.NONE, *flags)
        elif all([flag in RUN_MODE_FLAGS for flag in flags]):
            self.runmode = operation(self.runmode, ContextFlags.DEFAULT_MODE, *flags)
        else:
            raise ContextError(f'Flags must all correspond to one of: execution_phase, source, run mode')

    def add_flag(self, flag: ContextFlags):
        def add(attr, blank_flag, flag):
            return (attr & ~blank_flag) | flag

        self._change_flags(flag, operation=add)

    def remove_flag(self, flag: ContextFlags):
        def remove(attr, blank_flag, flag):
            if attr & flag:
                res = (attr | flag) ^ flag
                if res is ContextFlags.UNSET:
                    res = blank_flag
                return res
            else:
                return attr

        self._change_flags(flag, operation=remove)

    def replace_flag(self, old: ContextFlags, new: ContextFlags):
        def replace(attr, blank_flag, old, new):
            return (attr & ~old) | new

        self._change_flags(old, new, operation=replace)

@tc.typecheck
def _get_context(context:tc.any(ContextFlags, Context, str)):
    """Set flags based on a string of ContextFlags keywords
    If context is already a ContextFlags mask, return that
    Otherwise, return mask with flags set corresponding to keywords in context
    """
    # FIX: 3/23/18 UPDATE WITH NEW FLAGS
    if isinstance(context, ContextFlags):
        return context
    if isinstance(context, Context):
        context = context.string
    context_flag = ContextFlags.UNSET
    if VALIDATE in context:
        context_flag |= ContextFlags.VALIDATING
    if EXECUTING in context:
        context_flag |= ContextFlags.EXECUTING
    if CONTROL in context:
        context_flag |= ContextFlags.CONTROL
    if LEARNING in context:
        context_flag |= ContextFlags.LEARNING
    # if context == ContextFlags.TRIAL.name: # cxt-test
    #     context_flag |= ContextFlags.TRIAL
    # if context == ContextFlags.RUN.name:
    #     context_flag |= ContextFlags.RUN
    if context == ContextFlags.COMMAND_LINE.name:
        context_flag |= ContextFlags.COMMAND_LINE
    return context_flag


def _get_time(component, context):
    """Get time from Scheduler of Composition in which Component is being executed.

    Returns tuple with (run, trial, time_step) if being executed during Processing or Learning
    Otherwise, returns (None, None, None)

    """

    from psyneulink.core.globals.context import time
    from psyneulink.core.components.shellclasses import Mechanism, Projection, Port

    no_time = time(None, None, None, None)

    # Get mechanism to which Component being logged belongs
    if isinstance(component, Mechanism):
        ref_mech = component
    elif isinstance(component, Port):
        if isinstance(component.owner, Mechanism):
            ref_mech = component.owner
        elif isinstance(component.owner, Projection):
            ref_mech = component.owner.receiver.owner
        else:
            raise ContextError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                               format(component.__class__.__name__,
                                      Mechanism.__name__, Port.__name__, Projection.__name__))
    elif isinstance(component, Projection):
        ref_mech = component.receiver.owner
    else:
        raise ContextError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                           format(component.__class__.__name__,
                                  Mechanism.__name__, Port.__name__, Projection.__name__))

    # Get Composition in which it is being (or was last) executed (if any):

    composition = context.composition
    if composition is None:
        # If called from COMMAND_LINE, get context for last time value was assigned:
        composition = component.most_recent_context.composition

    if composition and hasattr(composition, 'scheduler'):
        execution_flags = context.execution_phase
        try:
            if execution_flags & (ContextFlags.PROCESSING | ContextFlags.LEARNING | ContextFlags.IDLE):
                t = composition.scheduler.get_clock(context).time
                t = time(t.run, t.trial, t.pass_, t.time_step)
            elif execution_flags & ContextFlags.CONTROL:
                t = composition.scheduler.get_clock(context).time
                t = time(t.run, t.trial, t.pass_, t.time_step)
            else:
                t = None
        except KeyError:
            t = None

    else:
        if component.verbosePref:
            offender = "\'{}\'".format(component.name)
            if ref_mech is not component:
                offender += " [{} of {}]".format(component.__class__.__name__, ref_mech.name)
            warnings.warn("Attempt to log {offender} which is not in a Composition; "
                          "logging is currently supported only when executing Components within a Composition.")
        t = None

    return t or no_time


_handle_external_context_arg_cache = defaultdict(dict)


def handle_external_context(
    source=ContextFlags.COMMAND_LINE,
    execution_phase=ContextFlags.IDLE,
    execution_id=None,
    fallback_most_recent=False,
    fallback_default=False,
    **context_kwargs
):
    """
        Arguments
        ---------

        source
            default ContextFlags to be used for source field when Context is not specified

        execution_phase
            default ContextFlags to be used for execution_phase field when
            Context is not specified

        context_kwargs
            additional keyword arguments to be given to Context.__init__ when
            Context is not specified

        Returns
        -------

        a decorator that ensures a Context argument is passed in to the decorated method

    """
    def decorator(func):
        assert not fallback_most_recent or not fallback_default

        # try to detect the position of the 'context' argument in function's
        # signature, to handle non-keyword specification in calls
        try:
            context_arg_index = _handle_external_context_arg_cache[func][CONTEXT]
        except KeyError:
            # this is true when there is a variable positional argument
            # (like *args). don't try to infer context position in this case,
            # because it can vary. I don't see a good way to get around this
            # restriction in general
            if len([
                sig_param for name, sig_param in inspect.signature(func).parameters.items()
                if sig_param.kind is sig_param.VAR_POSITIONAL
            ]):
                context_arg_index = None
            else:
                try:
                    context_arg_index = list(inspect.signature(func).parameters.keys()).index(CONTEXT)
                except ValueError:
                    context_arg_index = None

            _handle_external_context_arg_cache[func][CONTEXT] = context_arg_index

        @functools.wraps(func)
        def wrapper(*args, context=None, **kwargs):
            eid = execution_id

            if context is not None and not isinstance(context, Context):
                try:
                    eid = context.default_execution_id
                except AttributeError:
                    eid = context
                context = None
            else:
                try:
                    if args[context_arg_index] is not None:
                        if isinstance(args[context_arg_index], Context):
                            context = args[context_arg_index]
                        else:
                            try:
                                eid = args[context_arg_index].default_execution_id
                            except AttributeError:
                                eid = args[context_arg_index]
                            context = None
                except (TypeError, IndexError):
                    pass

            if context is None:
                if eid is None:
                    # assume first positional arg when fallback_most_recent or fallback_default
                    # true is the object that has the relevant context

                    if fallback_most_recent:
                        eid = args[0].most_recent_context.execution_id
                    if fallback_default:
                        eid = args[0].default_execution_id

                context = Context(
                    execution_id=eid,
                    source=source,
                    execution_phase=execution_phase,
                    **context_kwargs
                )
                if context_arg_index is not None:
                    try:
                        args = list(args)
                        args[context_arg_index] = context
                    except IndexError:
                        pass

            try:
                return func(*args, context=context, **kwargs)
            except TypeError as e:
                # context parameter may be passed as a positional arg
                if (
                    f"{func.__name__}() got multiple values for argument"
                    not in str(e)
                ):
                    raise e

            return func(*args, **kwargs)

        return wrapper
    return decorator
