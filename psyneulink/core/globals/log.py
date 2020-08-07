# -*- coding: utf-8 -*-
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **************************************************  Log **************************************************************

"""

Overview
--------

A Log object is used to record the `value <Component.value>` of PsyNeuLink Components during their "life cycle" (i.e.,
when they are created, validated, and/or executed).  Every Component has a Log object, assigned to its `log
<Component.log>` attribute when the Component is created, that can be used to record its value and/or that of other
Components that belong to it.  These are stored in `entries <Log.entries>` of the Log, that contain a sequential list
of the recorded values, along with the time and context of the recording.  The conditions under which values are
recorded is specified by the `logPref <Component.logPref>` property of a Component.  While these can be set directly,
they are most easily specified using the Log's `set_log_conditions <Log.set_log_conditions>` method, together with its
`loggable_items <Log.loggable_items>` and `logged_items <Log.logged_items>` attributes that identify and track the
items to be logged, respectively. Entries can also be made by the user programmatically, using the `log_values
<Log.log_values>` method. Logging can be useful not only for observing the behavior of a Component in a model, but also
in debugging the model during construction. The entries of a Log can be displayed in a "human readable" table using
its `print_entries <Log.print_entries>` method, and returned in CSV and numpy array formats using its and `nparray
<Log.nparray>`, `nparray_dictionary <Log.nparray_dictionary>` and `csv <Log.csv>`  methods.

.. _Log_Creation:

Creating Logs and Entries
-------------------------

A log object is automatically created for and assigned to a Component's `log <Component.log>` attribute when the
Component is created.  An entry is automatically created and added to the Log's `entries <Log.entries>` attribute
when its `value <Component.value>` or that of a Component that belongs to it is recorded in the Log.

.. _Log_Structure:

Structure
---------

A Log is composed of `entries <Log.entries>`, each of which is a dictionary that maintains a record of the logged
values of a Component.  The key for each entry is a string that is the name of the Component, and its value is a list
of `LogEntry` tuples recording its values.  Each `LogEntry` tuple has three items:
    * *time* -- the `RUN`, `TRIAL <TimeScale.TRIAL>`, `PASS`, and `TIME_STEP` in which the value of the item was
      recorded;
    * *context* -- a string indicating the context in which the value was recorded;
    * *value* -- the value of the item.
The time is recorded only if the Component is executed within a `System`;  otherwise, the time field is `None`.

A Log has several attributes and methods that make it easy to manage how and when it values are recorded, and
to access its `entries <Log.entries>`:

    * `loggable_items <Log.loggable_items>` -- a dictionary with the items that can be logged in a Component's `log
      <Component.log>`;  the key for each entry is the name of a Component,  and the value is the currently assigned
      `condition(s) <Log_Conditions>` under which it will be logged.
    ..
    * `set_log_conditions <Log.set_log_conditions>` -- used to assign the `condition <Log_Conditions>` for logging one
      or more Components. Components can be specified by their names, a reference to the Component object,
      in a tuple that specifies the `condition(s) <Log_Conditions>` for logging that Component, or in a list with a
      condition to be assigned to multiple items at once.
    ..
    * `log_values <Log.log_values>` -- used to the `value <Component.value>` of one or more Components in the Log
      programmatically ("manually").  Components can be specified by their names or references to the objects.
    ..
    * `logged_items <Log.logged_items>` -- a dictionary with the items that currently have entries in a Component's
      `log <Component.log>`; the key for each entry is the name of a Component, and the `condition(s)
      <Log_Conditions>` under which it is being logged.
    ..
    * `print_entries <Log.print_entries>` -- this prints a formatted list of the `entries <Log.entries>` in the Log.
    ..
    * `nparray <Log.csv>` -- returns a 2d np.array with the `entries <Log.entries>` in the Log.
    ..
    * `nparray_dictionary <Log.nparray_dictionary>` -- returns a dictionary of np.arrays with the `entries <Log.entries>` in the Log.
    ..
    * `csv <Log.csv>` -- returns a CSV-formatted string with the `entries <Log.entries>` in the Log.

.. _Log_Loggable_Items:

*Loggable Items*
~~~~~~~~~~~~~~~~

Although every Component is assigned its own Log, that records the `value <Component.value>` of that Component,
the Logs for `Mechanisms <Mechanism>` and `MappingProjections <MappingProjection>` also  provide access to and control
the Logs of their `Ports <Port>`.  Specifically the Logs of these Components contain the following information:

* **Mechanisms**

  * *value* -- the `value <Mechanism_Base.value>` of the Mechanism.

  * *InputPorts* -- the `value <InputPort.value>` of any `InputPort` (listed in the Mechanism's `input_ports
    <Mechanism_Base.input_ports>` attribute).

  * *ParameterPorts* -- the `value <ParameterPort.value>` of `ParameterPort` (listed in the Mechanism's
    `parameter_ports <Mechanism_Base.parameter_ports>` attribute);  this includes all of the `user configurable
    <Component_User_Params>` parameters of the Mechanism and its `function <Mechanism_Base.function>`.

  * *OutputPorts* -- the `value <OutputPort.value>` of any `OutputPort` (listed in the Mechanism's `output_ports
    <Mechanism_Base.output_ports>` attribute).
..
* **Projections**

  * *value* -- the `value <Projection_Base.value>` of the Projection.

  * *matrix* -- the value of the `matrix <MappingProjection.matrix>` parameter (for `MappingProjections
    <MappingProjection>` only).

.. _Log_Conditions:

*Logging Conditions*
~~~~~~~~~~~~~~~~~~~~

Configuring a Component to be logged is done using a condition, that specifies a `LogCondition` under which its
`value <Component.value>` should be entered in its Log.  These can be specified in the `set_log_conditions
<Log.set_log_conditions>` method of a Log, or directly by specifying a `LogCondition` for the value a Component's
`logPref  <Compnent.logPref>` item of its `prefs <Component.prefs>` attribute.  The former is easier, and allows
multiple Components to be specied at once, while the latter affords more control over the specification (see
`Preferences`).  `LogConditions <LogCondition>` are treated as binary "flags", and can be combined to permit logging
under more than one condition using bitwise operators on the `LogConditions <LogCondition>`.  For convenience, they
can also be referred to by their names, and combined by specifying a list.  For example, all of the following specify
that the `value <Mechanism_Base.value>` of ``my_mech`` be logged both during execution and learning::

    >>> import psyneulink as pnl
    >>> my_mech = pnl.TransferMechanism()
    >>> my_mech.set_log_conditions('value', pnl.LogCondition.EXECUTION | pnl.LogCondition.LEARNING)
    >>> my_mech.set_log_conditions('value', pnl.LogCondition.EXECUTION + pnl.LogCondition.LEARNING)
    >>> my_mech.set_log_conditions('value', [pnl.EXECUTION, pnl.LEARNING])


.. note::
   Currently, the `VALIDATION` `LogCondition` is not implemented.

.. note::
   Using `LogCondition.INITIALIZATION` to log the `value <Component.value>` of a Component during its initialization
   requires that it be assigned in the **prefs** argument of the Component's constructor.  For example::

   COMMENT:
   FIX: THIS EXAMPLE CAN'T CURRENTLY BE EXECUTED AS IT PERMANENTLY SETS THE LogPref FOR ALL TransferMechanism
   COMMENT
    >>> my_mech = pnl.TransferMechanism(
    ...        prefs={pnl.LOG_PREF: pnl.PreferenceEntry(pnl.LogCondition.INITIALIZATION, pnl.PreferenceLevel.INSTANCE)})

.. hint::
   `LogCondition.TRIAL` logs the `value <Component.value>` of a Component at the end of a `TRIAL <TimeScale.TRIAL>`.
   To log its `value <Component.value>` at the start of a `TRIAL <TimeScale.TRIAL>`, use its `log_values
   <Component.log_values>` method in the **call_before_trial** argument of the System's `run <System.run>` method.

.. _Log_Execution:

Execution
---------

The value of a Component is recorded to a Log when the condition assigned to its `logPref <Component.logPref>` is met.
This is specified as a `LogCondition` or a boolean combination of them (see `Log_Conditions`).  The default LogCondition
is `OFF`.

.. _Log_Examples:

Examples
--------

The following example creates a Composition with two `TransferMechanisms <TransferMechanism>`, one that projects to
another, and logs the `noise <TransferMechanism.noise>` and *RESULT* `OutputPort` of the first and the
`MappingProjection` from the first to the second::

    # Create a Process with two TransferMechanisms, and get a reference for the Projection created between them:
    >>> my_mech_A = pnl.TransferMechanism(name='mech_A', size=2)
    >>> my_mech_B = pnl.TransferMechanism(name='mech_B', size=3)
    >>> my_composition = pnl.Composition(pathways=[my_mech_A, my_mech_B])
    >>> proj_A_to_B = my_mech_B.path_afferents[0]

    COMMENT:
    FIX: THESE EXAMPLES CAN'T BE EXECUTED AS THEY RETURN DICT ENTRIES IN UNRELIABLE ORDERS
    COMMENT
    # Show the loggable items (and current condition assignments) for each Mechanism and the Projection between them:
    >> my_mech_A.loggable_items
    {'InputPort-0': 'OFF', 'slope': 'OFF', 'RESULT': 'OFF', 'integration_rate': 'OFF', 'intercept': 'OFF', 'noise': 'OFF'}
    >> my_mech_B.loggable_items
    {'InputPort-0': 'OFF', 'slope': 'OFF', 'RESULT': 'OFF', 'intercept': 'OFF', 'noise': 'OFF', 'integration_rate': 'OFF'}
    >> proj_A_to_B.loggable_items
    {'value': 'OFF', 'matrix': 'OFF'}

    # Assign the noise parameter and RESULT OutputPort of my_mech_A, and the matrix of the Projection, to be logged
    >>> my_mech_A.set_log_conditions([pnl.NOISE, pnl.RESULT])
    >>> proj_A_to_B.set_log_conditions(pnl.MATRIX)

Note that since no `condition <Log_Conditions>` was specified, the default (LogCondition.EXECUTION) is used.
Executing the Process generates entries in the Logs, that can then be displayed in several ways::

    COMMENT:
        disable this test due to inconsistent whitespacing across machines
    COMMENT
    # Execute the System twice (to generate some values in the logs):
    >> my_system.execute()
    [array([ 0.,  0.,  0.])]
    >> my_system.execute()
    [array([ 0.,  0.,  0.])]

    COMMENT:
    FIX: THESE EXAMPLES CAN'T BE EXECUTED AS THEY RETURN DICT ENTRIES IN UNRELIABLE ORDERS
    COMMENT
    # List the items of each Mechanism and the Projection that were actually logged:
    >> my_mech_A.logged_items
    {'RESULT': 'EXECUTION', 'noise': 'EXECUTION'}
    >> my_mech_B.logged_items
    {}
    >> proj_A_to_B.logged_items
    {'matrix': 'EXECUTION'}

Notice that entries dictionary of the Log for ``my_mech_B`` is empty, since no items were specified to be logged for
it.  The results of the two other logs can be printed to the console using the `print_entries <Log.print_entries>`
method of a Log::

    COMMENT:
    FIX: THIS EXAMPLE CAN'T BE EXECUTED AS IT REQUIRES INSERTION OF "<BLANKLINE>"'S THAT CAN'T BE SUPPRESSED IN HTML
    COMMENT
    # Print the Log for ``my_mech_A``:
    >> my_mech_A.log.print_entries()

    Log for mech_A:

    Logged Item:   Time       Context                                                                   Value

    'RESULT'      0:0:0     " EXECUTING  System System-0| Mechanism: mech_A [in processes: ['Pro..."   [ 0.  0.]
    'RESULT'      0:1:0     " EXECUTING  System System-0| Mechanism: mech_A [in processes: ['Pro..."   [ 0.  0.]


    'noise'        0:0:0     " EXECUTING  System System-0| Mechanism: mech_A [in processes: ['Pro..."   [ 0.]
    'noise'        0:1:0     " EXECUTING  System System-0| Mechanism: mech_A [in processes: ['Pro..."   [ 0.]


They can also be exported in numpy array and CSV formats.  The following shows the CSV-formatted output of the Logs
for ``my_mech_A`` and  ``proj_A_to_B``, using different formatting options::

    COMMENT:
    FIX: THESE EXAMPLES CAN'T BE EXECUTED AS THEY RETURN FORMATS ON JENKINS THAT DON'T MATCH THOSE ON LOCAL MACHINE(S)
    COMMENT
    >> print(my_mech_A.log.csv(entries=[pnl.NOISE, pnl.RESULT], owner_name=False, quotes=None))
    'Run', 'Trial', 'Time_step', 'noise', 'RESULT'
    0, 0, 0, 0.0, 0.0 0.0
    0, 1, 0, 0.0, 0.0 0.0
    COMMENT:
    <BLANKLINE>
    COMMENT

    # Display the csv formatted entry of Log for ``proj_A_to_B``
    #    with quotes around values and the Projection's name included in the header:
    >> print(proj_A_to_B.log.csv(entries=pnl.MATRIX, owner_name=False, quotes=True))
    'Run', 'Trial', 'Time_step', 'matrix'
    '0', '0', '1', '1.0 1.0 1.0' '1.0 1.0 1.0'
    '0', '1', '1', '1.0 1.0 1.0' '1.0 1.0 1.0'
    COMMENT:
    <BLANKLINE>
    COMMENT

Note that since the `name <Projection_Base.name>` attribute of the Projection was not assigned, its default name is
reported.

The following shows the Log of ``proj_A_to_B`` in numpy array format, with and without header information::

    COMMENT:
    FIX: THESE EXAMPLES CAN'T BE EXECUTED AS THEY RETURN FORMATS ON JENKINS THAT DON'T MATCH THOSE ON LOCAL MACHINE(S)
    COMMENT
    >> proj_A_to_B.log.nparray(entries=[pnl.MATRIX], owner_name=False, header=True)
    [[['Run'] [0] [0]]
     [['Trial'] [0] [1]]
     [['Time_step'] [1] [1]]
     ['matrix' [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]

    >> proj_A_to_B.log.nparray(entries=[pnl.MATRIX], owner_name=False, header=False)
    [[[0] [0]]
     [[0] [1]]
     [[1] [1]]
     [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]

COMMENT:
 MY MACHINE:
    >> proj_A_to_B.log.nparray(entries=[pnl.MATRIX], owner_name=False, header=False)
    array([[[0], [1]],
           [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], dtype=object)

JENKINS:
    >> proj_A_to_B.log.nparray(entries=[pnl.MATRIX], owner_name=False, header=False)
    array([[list([0]), list([1])],
           [list([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            list([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])]], dtype=object)

OR

    print(proj_A_to_B.log.nparray(entries=[pnl.MATRIX], owner_name=False, header=False))
Expected:
    [[[0] [1]]
     [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
Got:
    [[list([0]) list([1])]
     [list([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
      list([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])]]
COMMENT


COMMENT:

IMPLEMENTATION NOTE(S):

Name of owner Component is aliased to VALUE in loggable_items and logged_items,
but is the Component's actual name in log_entries

LogCondition flags are compared bitwise against the ContextFlags currently set for the Component

Entries are made to the Log based on the `LogCondition` specified in the
`logPref` item of the component's `prefs <Component.prefs>` attribute.

# Adding an item to prefs.logPref validates it and adds an entry for that attribute to the Log dict

An attribute is logged if:

# * it is one `automatically included <LINK>` in logging;

* it is included in the *LOG_ENTRIES* entry of a `parameter specification dictionary <ParameterPort_Specification>`
  assigned to the **params** argument of the constructor for the Component;

* the LogCondition(s) specified in a Component's logpref match the current `ContextFlags` in its context attribute

Entry values are added by the setter method for the attribute being logged.

The following entries are automatically included in the `loggable_items` of a `Mechanism <Mechanism>` object:
    - the `value <Mechanism_Base.value>` of the Mechanism;
    - the value attribute of every Port for which the Mechanism is an owner
    - value of every projection that sends to those Ports]
    - the system variables defined in SystemLogEntries (see declaration above)
    - any variables listed in the params[LOG_ENTRIES] of a Mechanism


DEFAULT LogCondition FOR ALL COMPONENTS IS *OFF*

Structure
---------

Each entry of `entries <Log.entries>` has:
    + a key that is the name of the attribute being logged
    + a value that is a list of sequentially entered LogEntry tuples since recording of the attribute began
    + each tuple has three items:
        - time (CentralClock): when it was recorded in the run
        - context (str): the context in which it was recorded (i.e., where the attribute value was assigned)
        - value (value): the value assigned to the attribute

The LogCondition class (see declaration below) defines the conditions under which a value can be logged.

Entries can also be added programmatically by:
    - including them in the logPref of a PreferenceSet
    - using the add_entries() method (see below)
    - using the log_entries() method (see below)

The owner.prefs.logPref setting contains a list of entries to actively record
    - when entries are added to an object's logPref list, the log.add_entries() method is called,
        which validates the entries against the object's attributes and SystemLogEntries
    - if entries are removed from the object's logPref list, they still remain in the log dict;
        they can be deleted from the log dict using the remove_log_entries() method
    - data is recorded in an entry using the log_entries() method, which records data to all entries
        in the self.owner.prefs.logPrefs list;  this is generally carried out by the update methods
        of Category classes in the Function hierarchy (e.g., Process, Mechanism and Projection)
        on each cycle of the execution sequence;
    - log_entries() adds entries to the self.owner.prefs.logPrefs list,
        which will record data for those attributes when logging is active;
    - suspend_entries() removes entries from the self.owner.prefs.logPrefs list;
        data will not be recorded for those entries when logging is active

    Notes:
    * A list of viable entries should be defined as the classLogEntries class attribute of a Function subclass
COMMENT


.. _Log_Class_Reference:

Class Reference
---------------

"""
import enum
import warnings

from collections import OrderedDict, namedtuple
from collections.abc import MutableMapping

import numpy as np
import typecheck as tc

from psyneulink.core.globals.context import ContextFlags, _get_time, handle_external_context
from psyneulink.core.globals.context import time as time_object
from psyneulink.core.globals.keywords import ALL, CONTEXT, EID_SIMULATION, FUNCTION_PARAMETER_PREFIX, MODULATED_PARAMETER_PREFIX, TIME, VALUE
from psyneulink.core.globals.utilities import AutoNumber, ContentAddressableList, is_component

__all__ = [
    'EntriesDict', 'Log', 'LogEntry', 'LogError', 'LogCondition'
]


LogEntry = namedtuple('LogEntry', 'time, context, value')

class LogCondition(enum.IntFlag):
    """Used to specify the context in which a value of the Component or its attribute is `logged <Log_Conditions>`.

    .. note::
      The values of LogCondition are subset of (and directly reference) the ContextFlags bitwise enum,
      with the exception of TRIAL and RUN, which are bit-shifted to follow the ContextFlags.SIMULATION_MODE value.
    """
    OFF = ContextFlags.UNSET
    """No recording."""
    # INITIALIZATION = ContextFlags.INITIALIZING
    INITIALIZATION = ContextFlags.INITIALIZING
    """Set during execution of the Component's constructor."""
    VALIDATION = ContextFlags.VALIDATING
    """Set during validation of the value of a Component or its attribute."""
    EXECUTION = ContextFlags.EXECUTING
    """Set during all `phases of execution <System_Execution>` of the Component."""
    PROCESSING = ContextFlags.PROCESSING
    """Set during the `processing phase <System_Execution_Processing>` of execution of a Composition."""
    LEARNING = ContextFlags.LEARNING
    """Set during the `learning phase <System_Execution_Learning>` of execution of a Composition."""
    CONTROL = ContextFlags.CONTROL
    """Set during the `control phase System_Execution_Control>` of execution of a Composition."""
    SIMULATION = ContextFlags.SIMULATION_MODE
    # Set during simulation by Composition.controller
    TRIAL = ContextFlags.SIMULATION_MODE << 1
    """Set at the end of a 'TRIAL'."""
    RUN = ContextFlags.SIMULATION_MODE << 2
    """Set at the end of a 'RUN'."""
    ALL_ASSIGNMENTS = (
        INITIALIZATION | VALIDATION | EXECUTION | PROCESSING | LEARNING | CONTROL
        | SIMULATION | TRIAL | RUN
    )
    """Specifies all contexts."""

    @classmethod
    def _get_log_condition_string(cls, condition, string=None):
        """Return string with the names of all flags that are set in **condition**, prepended by **string**"""
        if string:
            string += ": "
        else:
            string = ""
        flagged_items = []
        # If OFF or ALL_ASSIGNMENTS, just return that
        if condition in (LogCondition.ALL_ASSIGNMENTS, LogCondition.OFF):
            return condition.name
        # Otherwise, append each flag's name to the string
        for c in list(cls.__members__):
            # Skip ALL_ASSIGNMENTS (handled above)
            if c is LogCondition.ALL_ASSIGNMENTS.name:
                continue
            if LogCondition[c] & condition:
                if c in EXECUTION_CONDITION_NAMES:
                    if condition & LogCondition.EXECUTION == ContextFlags.EXECUTION_PHASE_MASK:
                        continue
                flagged_items.append(c)

        if len(flagged_items) > 0:
            string += ", ".join(flagged_items)
            return string
        else:
            return 'invalid LogCondition'

    @staticmethod
    def from_string(string):
        try:
            return LogCondition[string.upper()]
        except KeyError:
            raise LogError("\'{}\' is not a value of {}".format(string, LogCondition))

TIME_NOT_SPECIFIED = 'Time Not Specified'
EXECUTION_CONDITION_NAMES = {LogCondition.PROCESSING.name,
                             LogCondition.LEARNING.name,
                             LogCondition.CONTROL.name,
                             LogCondition.SIMULATION.name}


class LogTimeScaleIndices(AutoNumber):
    RUN = ()
    TRIAL = ()
    PASS = ()
    TIME_STEP = ()
NUM_TIME_SCALES = len(LogTimeScaleIndices.__members__)
TIME_SCALE_NAMES = list(LogTimeScaleIndices.__members__)


def _time_string(time):

    # if any(t is not None for t in time ):
    #     run, trial, time_step = time
    #     time_str = "{}:{}:{}".format(run, trial, time_step)
    # else:
    #     time_str = "None"
    # return time_str

    if time is not None and all(t is not None for t in time ):
        time_str = ":".join([str(i) for i in time])
    else:
        time_str = "None"
    return time_str


#region Custom Entries Dict
# Modified from: http://stackoverflow.com/questions/7760916/correct-useage-of-getter-setter-for-dictionary-values
class EntriesDict(MutableMapping,dict):
    """Maintains a Dict of Log entries; assignment of a LogEntry to an entry appends it to the list for that entry.

    The key for each entry is the name of an attribute being logged (usually the `value <Component.value>` of
    the Log's `owner <Log.owner>`.

    The value of each entry is a list, each item of which is a LogEntry.

    When a LogEntry is assigned to an entry:
       - if the entry does not already exist, it is created and assigned a list with the LogEntry as its first item;
       - if it exists, the LogEntry is appended to the list;
       - assigning anything other than a LogEntry raises and LogError exception.

    """
    def __init__(self, owner):

        # Log to which this dict belongs
        self._ownerLog = owner
        # Object to which the log belongs
        self._owner = owner.owner

        # # VERSION THAT USES OWNER'S logPref TO LIST ENTRIES TO BE RECORDED
        # # List of entries (in owner's logPrefs) of entries to record
        # self._recordingList = self._owner.prefs._log_pref.setting

        # # VERSION THAT USES OWNER'S logPref AS RECORDING SWITCH
        # # Recording state (from owner's logPrefs setting)
        # self._recording = self._owner.prefs._log_pref.setting

        super(EntriesDict, self).__init__({})

    def __getitem__(self,key):
        return dict.__getitem__(self,key)

    def __setitem__(self, key, value):
        if isinstance(value, list):
            dict.__setitem__(self, key, value)
            return

        if not isinstance(value, LogEntry):
            raise LogError("Object other than a {} assigned to Log for {}".format(LogEntry.__name__, self.owner.name))
        try:
        # If the entry already exists, use its value and append current value to it
            self._ownerLog.entries[key].append(value)
            value = self._ownerLog.entries[key]
        except KeyError:
        # Otherwise, initialize list with value as first item
            dict.__setitem__(self,key,[value])
        else:
            dict.__setitem__(self,key,value)

    def __delitem__(self, key):
        dict.__delitem__(self,key)

    def __iter__(self):
        return dict.__iter__(self)

    def __len__(self):
        return dict.__len__(self)

    def __contains__(self, x):
        return dict.__contains__(self,x)
#endregion

#region LogError
class LogError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)
#endregion


class Log:
    """Maintain a Log for an object, which contains a dictionary of logged value(s).

    COMMENT:

    IMPLEMENTATION NOTE: Name of owner Component is aliases to VALUE in loggable_items and logged_items,
    but is the Component's actual name in log_entries

    Description:
        Log maintains a dict (self.entries), with an entry for each attribute of the owner object being logged
        Each entry of self.entries has:
            + a key that is the name of the attribute being logged
            + a value that is a list of sequentially entered LogEntry tuples since recording of the attribute began
            + each tuple has three items:
                - time (CentralClock): when it was recorded in the run
                - context (str): the context in which it was recorded (i.e., where the attribute value was assigned)
                - value (value): the value assigned to the attribute
        An attribute is recorded if:
            - it is one automatically included in logging (see below)
            - it is included in params[LOG_ENTRIES] of the owner object
            - the context of the assignment is above the ContextFlags specified in the logPref setting of the owner object
        Entry values are added by the setter method for the attribute being logged
        The following entries are automatically included in self.entries for a Mechanism object:
            - the value attribute of every Port for which the Mechanism is an owner
            [TBI: - value of every projection that sends to those Ports]
            - the system variables defined in SystemLogEntries (see declaration above)
            - any variables listed in the params[LOG_ENTRIES] of a Mechanism
        The ContextFlags class (see declaration above) defines five levels of logging:
            + OFF: No logging for attributes of the owner object
            + EXECUTION: Log values for all assignments during exeuction (e.g., including aggregation of projections)
            + VALIDATION: Log value assignments during validation as well as execution
            + ALL_ASSIGNMENTS:  Log all value assignments (e.g., including initialization)
            Note: ContextFlags is an IntEnum, and thus its values can be used directly in numerical comparisons

        # Entries can also be added programmtically by:
        #     - including them in the logPref of a PreferenceSet
        #     - using the add_entries() method (see below)
        #     - using the log_entries() method (see below)
        # The owner.prefs.logPref setting contains a list of entries to actively record
        #     - when entries are added to an object's logPref list, the log.add_entries() method is called,
        #         which validates the entries against the object's attributes and SystemLogEntries
        #     - if entries are removed from the object's logPref list, they still remain in the log dict;
        #         they can be deleted from the log dict using the remove_log_entries() method
        #     - data is recorded in an entry using the log_entries() method, which records data to all entries
        #         in the self.owner.prefs.logPrefs list;  this is generally carried out by the update methods
        #         of Category classes in the Function hierarchy (e.g., Process, Mechanism and Projection)
        #         on each cycle of the execution sequence;
        #     - log_entries() adds entries to the self.owner.prefs.logPrefs list,
        #         which will record data for those attributes when logging is active;
        #     - suspend_entries() removes entries from the self.owner.prefs.logPrefs list;
        #         data will not be recorded for those entries when logging is active

        Notes:
        * A list of viable entries should be defined as the classLogEntries class attribute of a Function subclass

    Instantiation:
        A Log object is automatically instantiated for a Function object by Function.__init__(), which assigns to it
            any entries specified in the logPref of the prefs arg used to instantiate the owner object
        Adding an item to self.owner.prefs.logPref will validate and add an entry for that attribute to the log dict

    Initialization arguments:
        - entries (list): list of keypaths for attributes to be logged

    Class Attributes:
        + log (dict)

    Class Methods:
        - add_entries(entries) - add entries to log dict
        - delete_entries(entries, confirm) - delete entries from log dict; confirm=True requires user confirmation
        - reset_entries(entries, confirm) - delete all data from entries but leave them in log dict;
                                                 confirm=True requires user confirmation
        - log_entries(entries) - activate recording of data for entries (adds them to self.owner.prefs.logPref)
        - suspend_entries(entries) - halt recording of data for entries (removes them from self.owner.prefs.logPref)
        - log_entries(entries) - logs the current values of the attributes corresponding to entries
        - print_entries(entries) - prints entry values
        - [TBI: save_log - save log to disk]
    COMMENT

    Attributes
    ----------

    owner : Component
        the `Component <Component>` to which the Log belongs (and assigned as its `log <Component.log>` attribute).

    loggable_components : ContentAddressableList
        each item is a Component that is loggable for the Log's `owner <Log.owner>`

    loggable_items : Dict[Component.name: List[LogEntry]]
        identifies Components that can be logged by the owner; the key of each entry is the name of a Component,
        and the value is its currently assigned `LogCondition`.

    entries : Dict[Component.name: List[LogEntry]]
        contains the logged information for `loggable_components <Log.loggable_components>`; the key of each entry
        is the name of a Component, and its value is a list of `LogEntry` items for that Component.  Only Components
        for which information has been logged appear in the `entries <Log.entries>` dict.

    logged_items : Dict[Component.name: List[LogEntry]]
        identifies Components that currently have entries in the Log; the key for each entry is the name
        of a Component, and the value is its currently assigned `LogCondition`.

    """
    context_header = 'Execution Context'
    data_header = 'Data'

    def __init__(self, owner, entries=None):
        """Initialize Log with list of entries

        Each item of the entries list should be a string designating a Component to be logged;
        Initialize self.entries dict, each entry of which has a:
            - key corresponding to a Port of the Component to which the Log belongs
            - value that is a list of sequentially logged LogEntry items
        """

        self.owner = owner
        # self.entries = EntriesDict({})
        self.entries = EntriesDict(self)

        if entries is None:
            return

    @property
    def parameter_items(self):
        return self.owner._loggable_parameters

    @property
    def input_port_items(self):
        try:
            return self.owner.input_ports.names
        except AttributeError:
            return []

    @property
    def output_port_items(self):
        try:
            return self.owner.output_ports.names
        except AttributeError:
            return []

    @property
    def parameter_port_items(self):
        try:
            return [MODULATED_PARAMETER_PREFIX + name for name in self.owner.parameter_ports.names]
        except AttributeError:
            return []

    @property
    def function_items(self):
        try:
            return [FUNCTION_PARAMETER_PREFIX + name for name in self.owner.function._loggable_parameters]
        except AttributeError:
            return []

    @property
    def all_items(self):
        return sorted(self.parameter_items + self.input_port_items + self.output_port_items + self.parameter_port_items + self.function_items)

    def _get_parameter_from_item_string(self, string):
        # KDM 8/15/18: can easily cache these results if it occupies too much time, assuming
        # no duplicates/changing
        if string.startswith(MODULATED_PARAMETER_PREFIX):
            try:
                return self.owner.parameter_ports[string[len(MODULATED_PARAMETER_PREFIX):]].parameters.value
            except (AttributeError, TypeError):
                pass

        try:
            return getattr(self.owner.parameters, string)
        except AttributeError:
            pass

        try:
            return self.owner.input_ports[string].parameters.value
        except (AttributeError, TypeError):
            pass

        try:
            return self.owner.output_ports[string].parameters.value
        except (AttributeError, TypeError):
            pass

        if string.startswith(FUNCTION_PARAMETER_PREFIX):
            try:
                return getattr(self.owner.function.parameters, string[len(FUNCTION_PARAMETER_PREFIX):])
            except AttributeError:
                pass

    def set_log_conditions(self, items, log_condition=LogCondition.EXECUTION):
        """Specifies items to be logged under the specified `LogCondition`\\(s).

        Arguments
        ---------

        items : str, Component, tuple or List of these
            specifies items to be logged;  these must be be `loggable_items <Log.loggable_items>` of the Log.
            Each item must be a:
            * string that is the name of a `loggable_item` <Log.loggable_item>` of the Log's `owner <Log.owner>`;
            * a reference to a Component;
            * tuple, the first item of which is one of the above, and the second a `ContextFlags` to use for the item.

        log_condition : LogCondition : default LogCondition.EXECUTION
            specifies `LogCondition` to use as the default for items not specified in tuples (see above).
            For convenience, the name of a LogCondition can be used in place of its full specification
            (e.g., *EXECUTION* instead of `LogCondition.EXECUTION`).

        params_set : list : default None
            list of parameters to include as loggable items;  these must be attributes of the `owner <Log.owner>`
            (for example, Mechanism

        """
        from psyneulink.core.components.component import Component
        from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
        from psyneulink.core.globals.keywords import ALL

        def assign_log_condition(item, level):

            # Handle multiple level assignments (as LogCondition or strings in a list)
            if not isinstance(level, list):
                level = [level]
            levels = LogCondition.OFF
            for l in level:
                if isinstance(l, str):
                    l = LogCondition.from_string(l)
                levels |= l
            level = levels

            if item not in self.loggable_items:
                # KDM 8/13/18: NOTE: add_entries is not defined anywhere
                raise LogError("\'{0}\' is not a loggable item for {1} (try using \'{1}.log.add_entries()\')".
                               format(item, self.owner.name))

            self._get_parameter_from_item_string(item).log_condition = level

        if items == ALL:
            for component in self.loggable_components:
                component.logPref = PreferenceEntry(log_condition, PreferenceLevel.INSTANCE)

            for item in self.all_items:
                self._get_parameter_from_item_string(item).log_condition = log_condition
            # self.logPref = PreferenceEntry(log_condition, PreferenceLevel.INSTANCE)
            return

        if not isinstance(items, list):
            items = [items]

        # allow multiple sets of conditions to be set for multiple items with one call
        for item in items:
            if isinstance(item, (str, Component)):
                if isinstance(item, Component):
                    item = item.name
                assign_log_condition(item, log_condition)
            else:
                assign_log_condition(item[0], item[1])

    def _set_delivery_conditions(self, items, delivery_condition=LogCondition.EXECUTION):
        """Specifies items to be delivered via gRPC under the specified `LogCondition`\\(s).

        Arguments
        ---------

        items : str, Component, tuple or List of these
            specifies items to be logged;  these must be be `loggable_items <Log.loggable_items>` of the Log.
            Each item must be a:
            * string that is the name of a `loggable_item` <Log.loggable_item>` of the Log's `owner <Log.owner>`;
            * a reference to a Component;
            * tuple, the first item of which is one of the above, and the second a `ContextFlags` to use for the item.

        delivery_condition : LogCondition : default LogCondition.EXECUTION
            specifies `LogCondition` to use as the default for items not specified in tuples (see above).
            For convenience, the name of a LogCondition can be used in place of its full specification
            (e.g., *EXECUTION* instead of `LogCondition.EXECUTION`).

        params_set : list : default None
            list of parameters to include as loggable items;  these must be attributes of the `owner <Log.owner>`
            (for example, Mechanism

        """
        from psyneulink.core.components.component import Component
        from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
        from psyneulink.core.globals.keywords import ALL

        def assign_delivery_condition(item, level):

            # Handle multiple level assignments (as LogCondition or strings in a list)
            if not isinstance(level, list):
                level = [level]
            levels = LogCondition.OFF
            for l in level:
                if isinstance(l, str):
                    l = LogCondition.from_string(l)
                levels |= l
            level = levels

            if item not in self.loggable_items:
                # KDM 8/13/18: NOTE: add_entries is not defined anywhere
                raise LogError("\'{0}\' is not a loggable item for {1} (try using \'{1}.log.add_entries()\')".
                               format(item, self.owner.name))

            self._get_parameter_from_item_string(item).delivery_condition = level

        if items == ALL:
            for component in self.loggable_components:
                component.logPref = PreferenceEntry(delivery_condition, PreferenceLevel.INSTANCE)

            for item in self.all_items:
                self._get_parameter_from_item_string(item).delivery_condition = delivery_condition
            # self.logPref = PreferenceEntry(log_condition, PreferenceLevel.INSTANCE)
            return

        if not isinstance(items, list):
            items = [items]

        # allow multiple sets of conditions to be set for multiple items with one call
        for item in items:
            if isinstance(item, (str, Component)):
                if isinstance(item, Component):
                    item = item.name
                assign_delivery_condition(item, delivery_condition)
            else:
                assign_delivery_condition(item[0], item[1])

    @tc.typecheck
    @handle_external_context()
    def _deliver_values(self, entries, context=None):
        from psyneulink.core.globals.parameters import parse_context
        """Deliver the value of one or more Components programmatically.

        This can be used to "manually" prepare the `value <Component.value>` of any of a Component's `loggable_items
        <Component.loggable_items>` (including its own `value <Component.value>`) for delivery to an external application via gRPC.
        If the call to _deliver_values is made while a Composition to which the Component belongs is being run (e.g.,
        in a **call_before..** or **call_after...** argument of its `run <Composition.run>` method), then the time of
        the LogEntry is assigned the value of the `Clock` of the Composition's `scheduler` or `scheduler_learning`,
        whichever is currently executing (see `Composition_Scheduler`).

        Arguments
        ---------

        entries : string, Component or list containing either : default ALL
            specifies the Components, the current `value <Component.value>`\\s of which should be added prepared for
            transmission to an external application via gRPC.
            they must be `loggable_items <Log.loggable_items>` of the owner's Log. If **entries** is *ALL* or is not
            specified, then the `value <Component.value>`\\s of all `loggable_items <Log.loggable_items>` are logged.
        """
        entries = self._validate_entries_arg(entries)
        original_source = context.source
        context.source = ContextFlags.COMMAND_LINE

        # Validate the Component field of each LogEntry
        for entry in entries:
            param = self._get_parameter_from_item_string(entry)
            context = parse_context(context)
            param._deliver_value(param._get(context), context)

        context.source = original_source

    @tc.typecheck
    def _log_value(
        self,
        value,
        time=None,
        condition:tc.optional(LogCondition)=None,
        context=None,

    ):
        """Add LogEntry to an entry in the Log

        If **value** is a LogEntry, it is assigned to the entry
        If **condition** is specified, it is used to determine whether the value is logged; **time** must be passed.
        Otherwise, the Component's `log_pref <Component.log_pref>` attribute is used to determine whether value is
        logged.
        If **value** is specified, that is the value logged; otherwise the owner's `value <Component.value>`
           attribute is logged.

        COMMENT:
            IMPLEMENTATION NOTE:

            Component.log_value calls with **context** = *ContextFlags.COMMAND_LINE*; this logs the specified value.

            Since _log_value is usually called by the setter for the `value <Component.value>` property of a Component
            (which doesn't/can't receive a context argument), it does not pass a **condition** argument to _log_value;
            in that case, the context attribute of the log's owner is used.
            DEPRECATED:
            As a backup, it searches the stack for the most recent frame with a context specification, and uses that.
        COMMENT


        """
        from psyneulink.core.components.shellclasses import Function

        if time is None:
            time = time_object(None, None, None, None)

        if isinstance(value, LogEntry):
            self.entries[self.owner.name] = value

        else:
            condition = condition or context.execution_phase
            if not condition:
                # IMPLEMENTATION NOTE:  Functions not supported for logging at this time.
                if isinstance(self.owner, Function):
                    return
                else:
                    raise LogError("PROGRAM ERROR: No condition or context specified in call to _log_value for "
                                   "{}".format(self.owner.name))

            condition_string = ContextFlags._get_context_string(condition)

            log_pref = self.owner.prefs.logPref if self.owner.prefs else None

            # Get time and log value if logging condition is satisfied or called for programmatically
            if (log_pref and log_pref & condition) or condition & ContextFlags.COMMAND_LINE:
                time = time or _get_time(self.owner, condition)
                self.entries[self.owner.name] = LogEntry(time, condition_string, value)

    @tc.typecheck
    @handle_external_context()
    def log_values(self, entries, context=None):
        from psyneulink.core.globals.parameters import parse_context
        """Log the value of one or more Components programmatically.

        This can be used to "manually" enter the `value <Component.value>` of any of a Component's `loggable_items
        <Component.loggable_items>` (including its own `value <Component.value>`) in its `log <Component.log>`.
        The context item of its `LogEntry` is assigned *COMMAND_LINE*.  If the call to log_values is made while a
        System to which the Component belongs is being run (e.g., in a **call_before..** or **call_after...** argument
        of its `run <System.run>` method), then the time of the LogEntry is assigned the value of the `Clock` of
        the System's `scheduler` or `scheduler_learning`, whichever is currently executing
        (see `System_Scheduler`).

        Arguments
        ---------

        entries : string, Component or list containing either : default ALL
            specifies the Components, the current `value <Component.value>`\\s of which should be added to the Log.
            they must be `loggable_items <Log.loggable_items>` of the owner's Log. If **entries** is *ALL* or is not
            specified, then the `value <Component.value>`\\s of all `loggable_items <Log.loggable_items>` are logged.
        """
        entries = self._validate_entries_arg(entries)

        # Validate the Component field of each LogEntry
        for entry in entries:
            param = self._get_parameter_from_item_string(entry)
            context = parse_context(context)
            param._log_value(param._get(context), context)

    def get_logged_entries(self, entries=ALL, contexts=NotImplemented, exclude_sims=False):
        from psyneulink.core.globals.parameters import parse_context
        if entries == ALL:
            entries = self.all_items

        logged_entries = {}
        for item in entries:
            logged_entries[item] = {}
            try:
                # allow entries to be names of Components
                item = item.name
            except AttributeError:
                pass

            log = self._get_parameter_from_item_string(item).log
            if contexts is NotImplemented:
                eids = log.keys()
            elif not isinstance(contexts, list):
                eids = [contexts]
            else:
                eids = contexts
            eids = [parse_context(eid) for eid in eids]

            for eid in eids:
                if (
                    (eid in log and len(log[eid]) > 0)
                    and (not exclude_sims or EID_SIMULATION not in str(eid))
                ):
                    logged_entries[item][eid] = log[eid]

            if len(logged_entries[item]) == 0:
                del logged_entries[item]

        return logged_entries

    def clear_entries(self, entries=ALL, delete_entry=True, confirm=False, contexts=NotImplemented):
        """Clear one or more entries either by deleting the entry or just removing its data.

        Arguments
        ---------

        entries : string, Component or list containing either : default ALL
            specifies the entries of the Log to be cleared;  they must be `loggable_items
            <Log.loggable_items>` of the Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is *ALL* or is not specified, then all `logged_items <Log.logged_items>` are cleared.

        delete_entry : bool : default True
            specifies whether to delete the entry (if `True`) from the log to which it belongs, or just
            delete the data, but leave the entry itself (if `False`).

            .. note::
                This option is included for generality and potential future features, but is not advised;
                the Log interface (e.g., the `logged_items <Log.logged_items>` interface generally assumes that
                the only `entries <Log.entries>` in a log are ones with data.

        confirm : bool : default False
            specifies whether user confirmation is required before clearing the entries.

            .. note::
                If **confirm** is `True`, only a single confirmation will occur for a list or *ALL*

        """

        entries = self._validate_entries_arg(entries)

        # If any entries remain
        if entries:
            if confirm:
                delete = input("\nAll data will be deleted from {0} in the Log for {1}.  Proceed? (y/n)".
                               format(entries,self.owner.name))
                while delete != 'y' and delete != 'n':
                    input("\nDelete all data from entries? (y/n)")
                if delete == 'n':
                    return

            # Clear entries
            for entry in entries:
                if delete_entry:
                    # Delete the entire entry from the log to which it belongs
                    self._get_parameter_from_item_string(entry).clear_log(contexts)
                else:
                    # Delete the data for the entry but leave the entry itself in the log to which it belongs
                    # MODIFIED 6/15/20 OLD:
                    raise LogError('delete_entry=False currently unimplemented')
                    # # MODIFIED 6/15/20 NEW:
                    # warnings.warn('delete_entry=False currently unimplemented in Log.clear_entries()')
                    # MODIFIED 6/15/20 END
                assert True

    @tc.typecheck
    def print_entries(self,
                      entries:tc.optional(tc.any(str, list, is_component))=ALL,
                      width:int=120,
                      display:tc.any(tc.enum(TIME, CONTEXT, VALUE, ALL), list)=ALL,
                      contexts=NotImplemented,
                      exclude_sims=False,
                      # long_context=False
                      ):
        """
        print_entries(          \
              entries=ALL,      \
              width=120,        \
              display=None      \
            )

        Print summary of the Log's entries in a (human-readable) table format.

        Arguments
        ---------

        entries : string, Component or list containing either : default ALL
            specifies the entries of the Log to printed;  they must be `loggable_items <Log.loggable_items>` of the
            Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is *ALL* or is not specified, then all `logged_items <Log.logged_items>` are printed.

        width : int : default 120
            specifies the width of the display. The widths of each column are adjusted accordingly, and based
            on which items are displayed (see **display** below);  information that does not fit within its column's
            width is truncated and suffixes with an ellipsis.

        display : TIME, CONTEXT, VALUE, a list containing any of these, or ALL : default ALL
            specifies the information items to display.  The name of the entry is always displayed, followed by the
            specified items;  the widths of the columns for the items is dynamically adjusted, based on how many
            are specified, allowing more information about one to be shown by omitting others (this is useful
            if the context strings are long and/or the values are arrays).
        COMMENT:
        long_context : bool : default False
            specifies the use of the full context string in the display;  this can be informative, but can also take up
            more space in each line of the display.
        COMMENT

        exclude_sims
            set to True to exclude from output any values logged during `simulations <OptimizationControlMechanism_Model_Based>`

            :default value: False
            :type: bool

        """

        entries = self._validate_entries_arg(entries, logged=True)

        if not entries:
            return None

        class options(enum.IntFlag):
            NONE = 0
            TIME = 2
            CONTEXT = 4
            VALUE = 8
            ALL = TIME + CONTEXT + VALUE

        display = display or ALL
        if not isinstance(display, list):
            display = [display]

        # Set option_flags for specified options
        option_flags = options.NONE
        if TIME in display:
            option_flags |= options.TIME
        if CONTEXT in display:
            option_flags |= options.CONTEXT
        if VALUE in display:
            option_flags |= options.VALUE
        if ALL in display:
            option_flags = options.ALL

        # Default widths
        full_width = width
        item_name_width = 15
        time_width = 10
        context_width = 70
        value_width = 30
        spacer = ' '
        value_spacer_width = 3
        value_spacer = " ".ljust(value_spacer_width)
        base_width = item_name_width

        # Set context_width based on long_context option (length of context string) or context flags
        if option_flags & options.CONTEXT:
            c_width = 0
            for entry in entries:
                logged_entries = self.get_logged_entries(contexts=contexts, exclude_sims=exclude_sims)[entry]
                for eid in logged_entries:
                    for datum in logged_entries[eid]:
                        c_width = max(c_width, len(datum.context))
            context_width = min(context_width, c_width)

        # Set other widths based on options:
        # FIX: "ALGORITHMIZE" THIS:
        if option_flags == options.TIME:
            pass
        elif option_flags == options.CONTEXT:
            context_width = full_width - base_width
        elif option_flags == options.VALUE:
            value_width = full_width - base_width
        elif option_flags == options.TIME + options.CONTEXT:
            context_width = full_width - time_width - base_width
        elif option_flags == options.TIME + options.VALUE:
            value_width = full_width - time_width - base_width
        elif option_flags == options.CONTEXT + options.VALUE:
            context_width = full_width - value_width
            value_width = full_width - context_width
        elif option_flags == options.ALL:
            value_width = full_width - context_width
        else:
            raise LogError("PROGRAM ERROR:  unrecognized state of option_flags: {}".format(option_flags))

        header = "Logged Item:".ljust(item_name_width, spacer)
        if options.TIME & option_flags:
            header = header + TIME.capitalize().ljust(time_width, spacer)
        if options.CONTEXT & option_flags:
            header = header + " " + CONTEXT.capitalize().ljust(context_width, spacer)
        if options.VALUE & option_flags:
            header = header + value_spacer + " " + VALUE.capitalize()
            # header = header + value_spacer + VALUE.capitalize()

        print("\nLog for {0}:".format(self.owner.name))
        print('\n' + header + '\n')

        # Sort for consistency of reporting
        # entry_names_sorted = sorted(self.logged_entries.keys())
        entry_names_sorted = sorted(entries)
        # spacer = '_'
        # for entry_name in self.logged_entries:
        for entry_name in entry_names_sorted:
            try:
                datum = self.get_logged_entries(contexts=contexts, exclude_sims=exclude_sims)[entry_name]
            except KeyError:
                warnings.warn("{0} is not an entry in the Log for {1}".
                      format(entry_name, self.owner.name))
            else:
                multiple_eids = len(datum)>1
                for eid in datum:
                    if multiple_eids:
                        print(f'context: {eid}:')
                    for i, item in enumerate(datum[eid]):

                        time, context, value = item

                        entry_name = self._alias_owner_name(entry_name)
                        data_str = repr(entry_name).ljust(item_name_width, spacer)

                        if options.TIME & option_flags:
                            time_str = _time_string(time)
                            data_str = data_str + time_str.ljust(time_width)

                        if options.CONTEXT & option_flags:
                            context = repr(context)
                            if len(context) > context_width:
                                context = context[:context_width - 3] + "..."
                            data_str = data_str + context.ljust(context_width, spacer)

                        if options.VALUE & option_flags:
                            value = str(value).replace('\n',',')
                            if len(value) > value_width:
                                value = value[:value_width - 3].rstrip() + "..."
                            format_str = "{{:2.{0}}}".format(value_width)
                            data_str = data_str + value_spacer + format_str.format(value).ljust(value_width)

                        print(data_str)

                    if len(datum[eid]) > 1:
                        print("\n")

    @tc.typecheck
    def nparray(self,
                entries=None,
                header:bool=True,
                owner_name:bool=False,
                contexts=NotImplemented,
                exclude_sims=False,
                ):
        """
        nparray(                 \
            entries=None,        \
            header:bool=True,    \
            owner_name=False):   \
            )

        Return a 2d numpy array with headers (optional) and values for the specified entries.

        Each row (axis 0) is a time series, with each item in each row the data for the corresponding time point.
        Rows are ordered in the same order as Components are specified in the **entries** argument.

        If all of the data for every entry has a time value (i.e., the time field of its LogEntry is not `None`),
        then the first four rows are time indices for the run, trial, pass, and time_step of each data item,
        respectively. Each subsequent row is the times series of data for a given entry.  If there is no data for a
        given entry at a given time point, it is entered as `None`.

        If any of the data for any entry does not have a time value (e.g., if that Component was not run within a
        System), then all of the entries must have the same number of data (LogEntry) items, and the first row is a
        sequential index (starting with 0) that simply designates the data item number.

        .. note::
           For data without time stamps, the nth items in each entry correspond (i.e., ones in the same column)
           are not guaranteed to have been logged at the same time point.

        If header is `True`, the first item of each row is a header field: for time indices it is either "Run",
        "Trial", and "Time_step", or "Index" if any data are missing time stamps.  For subsequent rows it is the name
        of the Component logged in that entry (see **owner_name** argument below for formatting).


        Arguments
        ---------

        entries : string, Component or list containing either : default ALL
            specifies the entries of the Log to be included in the output;  they must be `loggable_items
            <Log.loggable_items>` of the Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is *ALL* or is not specified, then all `logged_items <Log.logged_items>` are included.

        COMMENT:
        time : TimeScale or ALL : default ALL
            specifies the "granularity" of how the time of an entry is reported.  *ALL* (same as `TIME_STEP
            <TimeScale.TIME_STEP>) reports every entry in the Log in a separate column (axis 1) of the np.array
            returned.
        COMMENT

        header : bool : default True
            specifies whether or not to include a header in each row with the name of the Component for that entry.

        owner_name : bool : default False
            specifies whether or not to include the Log's `owner <Log.owner>` in the header of each field;
            if it is True, the format of the header for each field is "<Owner name>[<entry name>]";
            otherwise, it is "<entry name>".

        exclude_sims
            set to True to exclude from output any values logged during `simulations <OptimizationControlMechanism_Model_Based>`

            :default value: False
            :type: bool

        Returns:
            2d np.array
        """

        entries = self._validate_entries_arg(entries, logged=True)

        if not entries:
            return None

        if owner_name is True:
            owner_name_str = self.owner.name
            lb = "["
            rb = "]"
        else:
            owner_name_str = lb = rb = ""

        header = 1 if header is True else 0

        contexts = self._parse_contexts_arg(contexts, entries)

        if exclude_sims:
            contexts = [eid for eid in contexts if EID_SIMULATION not in str(eid)]

        npa = [[self.context_header]] if header else [[]]

        npa.append([self.data_header] if header else [])

        for eid in sorted(contexts, key=lambda k: str(k)):
            time_values = self._parse_entries_for_time_values(entries, execution_id=eid)
            npa[0].append(eid)

            data_entry = []
            # Create time rows (one for each time scale)
            if time_values:
                for i in range(NUM_TIME_SCALES):
                    row = [[t[i]] for t in time_values]
                    if header:
                        time_header = [TIME_SCALE_NAMES[i].capitalize()]
                        row = [time_header] + row
                    data_entry.append(row)
            # If any time values are empty, revert to indexing the entries;
            #    this requires that all entries have the same length
            else:
                max_len = max([len(self.logged_entries[e][eid]) for e in entries])

                # If there are no time values, only support entries of the same length
                # Must dealias both e and zeroth entry because either/both of these could be 'value'
                if not all(len(self.logged_entries[e]) == len(self.logged_entries[entries[0]])for e in entries):
                    raise LogError("nparray output requires that all entries have time values or are of equal length")

                data_entry = np.arange(max_len).reshape(max_len, 1).tolist()
                if header:
                    data_entry = [["Index"] + data_entry]
                else:
                    data_entry = [data_entry]

            for entry in entries:
                row = self._assemble_entry_data(entry, time_values, eid)

                if header:
                    entry_header = "{}{}{}{}".format(owner_name_str, lb, self._alias_owner_name(entry), rb)
                    row = [entry_header] + row
                data_entry.append(row)

            npa[1].append(data_entry)

        npa = np.array(npa, dtype=object)

        return npa

    def nparray_dictionary(self, entries=None, contexts=NotImplemented, exclude_sims=False):
        """
        nparray_dictionary(                 \
            entries=None,        \
            )

        Returns an `OrderedDict <https://docs.python.org/3.5/library/collections.html#collections.OrderedDict>`_

        *Keys:*

            Keys of the OrderedDict are strings.

            Keys are the names of logged Components specified in the **entries** argument, plus either Run, Trial, Pass,
            and Time_step, or Index.

            If all of the data for every entry has a time value (i.e., the time field of its LogEntry is not `None`),
            then the first four keys are Run, Trial, Pass, and Time_step, respectively.

            If any of the data for any entry does not have a time value (e.g., if that Component was not run within a
            System), then all of the entries must have the same number of data (LogEntry) items, and the first key is Index.

            Then, the logged components follow in the same order as they were specified.

        *Values:*

            Values of the OrderedDict are numpy arrays.

            The numpy array value for a given component key consists of that logged Component's data over many time points
            or executions.

            The numpy array values for Run, Trial, Pass, and Time_step are counters for each of those time scales. The ith
            elements of the Run, Trial, Pass, Time_step and component data arrays can be taken together to represent the value of
            that component during a particular time step of a particular trial of a particular run.

            For example, if log_dict is a log dictionary in which log_dict['slope'][5] = 2.0, log_dict['Time_step'][5] = 1,
            log_dict['Pass'][5] = 0, log_dict['Trial'][5] = 2, and log_dict['Run'][5] = 0, then the value of slope was
            2.0 during time step 1 of pass 0 of trial 2 of run 0. If there is no data for a given entry at a given time
            point, it is entered as `None`.

            The numpy array value for Index is a sequential index starting at zero.

        .. note::
           For data without time stamps, the nth item in each dictionary key (i.e., data in the same "column")
           is not guaranteed to have been logged at the same time point across all keys (Components).


        Arguments
        ---------

        entries : string, Component or list containing either : default ALL
            specifies the entries of the Log to be included in the output;  they must be `loggable_items
            <Log.loggable_items>` of the Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is *ALL* or is not specified, then all `logged_items <Log.logged_items>` are included.

        exclude_sims
            set to True to exclude from output any values logged during `simulations <OptimizationControlMechanism_Model_Based>`

            :default value: False
            :type: bool

        Returns:
            2d np.array
        """
        log_dict = OrderedDict()
        entries = self._validate_entries_arg(entries, logged=True)

        if not entries:
            return log_dict

        contexts = self._parse_contexts_arg(contexts, entries)

        if exclude_sims:
            contexts = [eid for eid in contexts if EID_SIMULATION not in str(eid)]

        for eid in contexts:
            time_values = self._parse_entries_for_time_values(entries, execution_id=eid)
            log_dict[eid] = OrderedDict()

            # If all time values are recorded - - - log_dict = {"Run": array, "Trial": array, "Time_step": array}
            if time_values:
                for i in range(NUM_TIME_SCALES):
                    row = [[t[i]] for t in time_values]
                    time_header = TIME_SCALE_NAMES[i].capitalize()
                    log_dict[eid][time_header] = row

            # If ANY time values are empty (components were run outside of a System) - - - log_dict = {"Index": array}
            else:
                # find number of values logged by zeroth component
                num_indicies = len(self.logged_entries[entries[0]])

                # If there are no time values, only support entries of the same length
                if not all(len(self.logged_entries[e]) == num_indicies for e in entries):
                    raise LogError("nparray output requires that all entries have time values or are of equal length")

                log_dict[eid]["Index"] = np.arange(num_indicies).reshape(num_indicies, 1).tolist()

            for entry in entries:
                log_dict[eid][entry] = np.array(self._assemble_entry_data(entry, time_values, eid))

        return log_dict

    @tc.typecheck
    def csv(self, entries=None, owner_name:bool=False, quotes:tc.optional(tc.any(bool, str))="\'", contexts=NotImplemented, exclude_sims=False):
        """
        csv(                           \
            entries=None,              \
            owner_name=False,          \
            quotes=\"\'\"              \
            )

        Returns a CSV-formatted string with headers and values for the specified entries.

        Each row (axis 0) is a time point, beginning with the time stamp and followed by the data for each
        Component at that time point, in the order they are specified in the **entries** argument. If all of the data
        for every Component have time values, then the first four items of each row are the time indices for the run,
        trial, pass, and time_step of that time point, respectively, followed by the data for each Component at that time
        point;  if a Component has no data for a time point, `None` is entered.

        If any of the data for any Component does not have a time value (i.e., it has `None` in the time field of
        its `LogEntry`) then all of the entries must have the same number of data (LogEntry) items, and the first item
        of each row is a sequential index (starting with 0) that designates the data item number.

        .. note::
           For data without time stamps, items in the same row are not guaranteed to refer to the same time point.

        The **owner_name** argument can be used to prepend the header for each Component with its owner.
        The **quotes** argument can be used to suppress or specifiy quotes to use around numeric values.


        Arguments
        ---------

        entries : string, Component or list containing either : default ALL
            specifies the entries of the Log to be included in the output;  they must be `loggable_items
            <Log.loggable_items>` of the Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is *ALL* or is not specified, then all `logged_items <Log.logged_items>` are included.

        owner_name : bool : default False
            specifies whether or not to include the Component's `owner <Log.owner>` in the header of each field;
            if it is True, the format of the header for each field is "<Owner name>[<entry name>]"; otherwise,
            it is "<entry name>".

        quotes : bool, str : default '
            specifies whether or not to enclose numeric values in quotes (may be useful for arrays);
            if not specified or `True`, single quotes are used for *all* items;
            if specified with a string, that is used in place of single quotes to enclose *all* items;
            if `False` or `None`, single quotes are used for headers (the items in the first row), but no others.

        exclude_sims
            set to True to exclude from output any values logged during `simulations <OptimizationControlMechanism_Model_Based>`

            :default value: False
            :type: bool

        Returns:
            CSV-formatted string
        """

        # Get and transpose nparray of entries
        try:
            npa = self.nparray(entries=entries, header=True, owner_name=owner_name)
        except LogError as e:
            raise LogError(e.args[0].replace('nparray', 'csv'))

        npaT = npa.T

        # execution context headers
        csv = "'" + "', '".join([str(x) for x in npaT[0]]) + "\'" + "\n"

        entries = self._validate_entries_arg(entries, logged=True)

        for i in range(1, len(npaT)):
            # for each context
            context = npaT[i][0]

            if exclude_sims and EID_SIMULATION in context:
                continue

            data = np.array(npaT[i][1], dtype=object).T

            if not quotes:
                quotes = ''
            elif quotes is True:
                quotes = '\''

            # Headers
            next_eid_entry_data = "'{0}'".format(context)
            next_eid_entry_data += ", \'" + "\', \'".join(i[0] if isinstance(i, list) else i for i in data[0]) + "\'"
            next_eid_entry_data += '\n'

            # Data
            for i in range(1, len(data)):
                next_eid_entry_data += ', ' + ', '.join([str(j) for j in [str(k).replace(',','') for k in data[i]]]).\
                    replace('[[',quotes).replace(']]',quotes).replace('[',quotes).replace(']',quotes)
                next_eid_entry_data += '\n'

            csv += next_eid_entry_data

        return(csv)

    def _validate_entries_arg(self, entries, loggable=True, logged=False):
        from psyneulink.core.components.component import Component

        logged_entries = self.logged_entries
        # If ALL, set entries to all entries in self.logged_entries
        if entries == ALL or entries is None:
            entries = logged_entries.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, (str, Component)):
            entries = [entries]

        # Make sure all entries are the names of Components
        entries = [entry.name if isinstance(entry, Component) else entry for entry in entries ]

        # Validate entries
        for entry in entries:
            if loggable:
                if self._alias_owner_name(entry) not in self.loggable_items:
                    raise LogError("{0} is not a loggable attribute of {1}".format(repr(entry), self.owner.name))
            if logged:
                if entry not in logged_entries and entry != 'value':
                    # raise LogError("{} is not currently being logged by {} (try using set_log_conditions)".
                    #                format(repr(entry), self.owner.name))
                    print("\n{} is not currently being logged by {} or has not data (try using set_log_conditions)".
                          format(repr(entry), self.owner.name))
        return entries

    def _parse_contexts_arg(self, contexts, entries):
        from psyneulink.core.globals.parameters import parse_context
        if entries is None:
            entries = []

        if contexts is NotImplemented:
            all_contexts = set()
            for entry in entries:
                log = self._get_parameter_from_item_string(entry).log
                for eid in log.keys():
                    # allow adding string to set using tuple
                    all_contexts.add((eid,))
            contexts = [eid[0] for eid in all_contexts]
        elif not isinstance(contexts, list):
            contexts = [contexts]

        contexts = [parse_context(eid) for eid in contexts]

        return contexts

    def _alias_owner_name(self, name):
        """Alias name of owner Component to VALUE in loggable_items and logged_items
        Component's actual name is preserved and used in log_entries (i.e., as entered by _log_value)
        """
        return VALUE if name is self.owner.name else name

    def _dealias_owner_name(self, name):
        """De-alias VALUE to name of owner
        """
        return self.owner.name if name == VALUE else name

    def _scan_for_duplicates(self, time_values):
        # TEMPORARY FIX: this is slow and may not cover all possible cases properly!
        # TBI: fix permanently in Time/SimpleTime
        # In the case where scheduling leads to duplicate SimpleTime tuples (since Pass is ignored)
        # _scan_for_duplicates() will increment the Time_step (index 2) value of the tuple

        mod_time_values = sorted(list(time_values))
        time_step_increments = []
        chain = 0
        for t in range(1, len(time_values)):
            if time_values[t] == time_values[t - 1]:
                chain += 1
            else:
                chain = 0
            time_step_increments.append(chain)
        for i in range(1, len(time_values)):
            update_tuple = list(time_values[i])
            update_tuple[2] = update_tuple[2] + time_step_increments[i - 1] * 0.01
            mod_time_values[i] = tuple(update_tuple)
        return mod_time_values

    def _parse_entries_for_time_values(self, entries, execution_id=None):
        # Returns sorted list of SimpleTime tuples for all time points at which these entries logged values

        time_values = []
        for entry in entries:
            # OLD: finds duplicate time points within any one entry and modifies their values to be unique
            #
            # # collect all time values for this entry
            # entry_time_values = []
            # entry_time_values.extend([item.time for item in self.logged_entries[entry] if all(i is not None for i in item.time)])

            # # increment any time stamp duplicates (on the actual data item)
            # if len(set(entry_time_values)) != len(entry_time_values):
            #     adjusted_time = self._scan_for_duplicates(entry_time_values)
            #     for i in range(len(self.logged_entries[entry])):
            #         temp_list = list(self.logged_entries[entry][i])
            #         temp_list[0] = adjusted_time[i]
            #         self.logged_entries[entry][i] = LogEntry(temp_list[0], temp_list[1], temp_list[2])

            logged_entries_for_param = self.get_logged_entries(contexts=[execution_id])
            # make sure param exists in logged entries
            logged_entries_for_param = logged_entries_for_param.get(entry) if logged_entries_for_param else None
            # make sure execution id exists in logged entries of param
            logged_entries_for_param = logged_entries_for_param.get(execution_id) if logged_entries_for_param else None
            if logged_entries_for_param:
                time_values.extend([item.time
                                    for item in logged_entries_for_param
                                    if all(i is not None for i in item.time)])

        # Insure that all time values are assigned, get rid of duplicates, and sort
        if all(all(i is not None for i in t) for t in time_values):
            time_values = sorted(list(set(time_values)))

        return time_values

    def _assemble_entry_data(self, entry, time_values, execution_id=None):
        # Assembles list of entry's (component's) value at each of the time points specified in time_values
        # If there are multiple entries for a given time point, the last one will be used
        # If data was not recorded for this entry (component) for a given time point, it will be stored as None

        # entry = self._dealias_owner_name(entry)
        row = []
        time_col = iter(time_values)
        data = self.logged_entries[entry][execution_id]
        time = next(time_col, None)
        for i in range(len(self.logged_entries[entry][execution_id])):
            # iterate through log entry tuples:
            # check whether the next tuple's time value matches the time for which data is currently being recorded
            # if not, check whether the current tuple's time value matches the time for which data is being recorded
            # if so, enter tuple's Component value in the entry's list
            # if not, enter `None` in the entry's list
            datum = data[i]
            if time_values:
                if i == len(data) - 1 or data[i + 1].time != time:
                    if datum.time != time:
                        row.append(None)
                    else:
                        value = None if datum.value is None else np.array(datum.value).tolist()  # else, if is time,
                        # append value
                        row.append(value)
                    time = next(time_col, None)  # increment time value
                    if time is None:  # if no more times, break
                        break
            else:
                if datum.value is None:
                    value = None
                elif isinstance(datum.value, list):
                    value = datum.value
                elif np.array(datum.value).shape == ():
                    # converted value is a scalar, so a call to np.array(datum.value).tolist() would return a scalar
                    value = [datum.value]
                else:
                    value = np.array(datum.value).tolist()

                row.append(value)
        return row

    @property
    def loggable_items(self):
        """Return dict of loggable items.

        Keys are names of the Components, values their ContextStates
        """
        # FIX: The following crashes during init as prefs have not all been assigned
        # return {key: value for (key, value) in [(c.name, c.logPref.name) for c in self.loggable_components]}

        loggable_items = {}
        for item in self.all_items:
            cond = self._get_parameter_from_item_string(item).log_condition
            try:
                # may be an actual LogCondition
                loggable_items[item] = cond.name
            except AttributeError:
                loggable_items[item] = cond

        return loggable_items

    @property
    def loggable_components(self):
        """Return a list of owner's Components that are loggable

        The loggable items of a Component are the Components (typically Ports) specified in the _logagble_items
        property of its class, and its own `value <Component.value>` attribute.
        """
        from psyneulink.core.components.component import Component

        try:
            loggable_items = ContentAddressableList(component_type=Component, list=self.owner._loggable_items)
            loggable_items[self.owner.name] = self.owner
        except AttributeError:
            return []
        return loggable_items

    @property
    def logged_items(self):
        """Dict of items that have logged `entries <Log.entries>`, indicating their specified `ContextFlags`.
        """
        log_condition = 'ContextFlags.'
        # Return ContextFlags for items in log.entries

        logged_items = {key: value for (key, value) in
                        [(self._alias_owner_name(l), self.loggable_items[self._alias_owner_name(l)])
                         for l in self.logged_entries.keys()]}

        return logged_items

    @property
    def logged_entries(self):
        return self.get_logged_entries()

    # def save_log(self):
    #     print("Saved")


class CompositionLog(Log):
    @property
    def all_items(self):
        return (
            super().all_items
            + [item.name for item in self.owner.nodes + self.owner.projections]
            + ([self.owner.controller.name] if self.owner.controller is not None else [])
        )

    def _get_parameter_from_item_string(self, string):
        param = super()._get_parameter_from_item_string(string)

        if param is None:
            try:
                return self.owner.nodes[string].parameters.value
            except (AttributeError, TypeError):
                pass

            try:
                return self.owner.projections[string].parameters.value
            except (AttributeError, TypeError):
                pass

            try:
                return self.owner.controller.parameters.value
            except (AttributeError, TypeError):
                pass
        else:
            return param


def _log_trials_and_runs(composition, curr_condition: tc.enum(LogCondition.TRIAL, LogCondition.RUN), context):
    # FIX: ALSO CHECK TIME FOR scheduler_learning, AND CHECK DATE FOR BOTH, AND USE WHICHEVER IS LATEST
    # FIX:  BUT WHAT IF THIS PARTICULAR COMPONENT WAS RUN IN THE LAST TIME_STEP??
    for mech in composition.mechanisms:
        for component in mech.log.loggable_components:
            if component.logPref & curr_condition:
                # value = LogEntry((composition.scheduler.clock.time.run,
                #                   composition.scheduler.clock.time.trial,
                #                   composition.scheduler.clock.time.time_step),
                #                  # context,
                #                  curr_condition,
                #                  component.value)
                # component.log._log_value(value=value, context=context)
                component.log._log_value(value=component.parameters.value._get(context), condition=curr_condition)

        for proj in mech.afferents:
            for component in proj.log.loggable_components:
                if component.logPref & curr_condition:
                    # value = LogEntry((composition.scheduler.clock.time.run,
                    #                   composition.scheduler.clock.time.trial,
                    #                   composition.scheduler.clock.time.time_step),
                    #                  context,
                    #                  component.value)
                    # component.log._log_value(value, context)
                    component.log._log_value(value=component.parameters.value._get(context), condition=curr_condition)

    # FIX: IMPLEMENT ONCE projections IS ADDED AS ATTRIBUTE OF Composition
    # for proj in composition.projections:
    #     for component in proj.log.loggable_components:
    #         if component.logPref & curr_condition:
    #             value = LogEntry((composition.scheduler.clock.time.run,
    #                               composition.scheduler.clock.time.trial,
    #                               composition.scheduler.clock.time.time_step),
    #                              context,
    #                              component.value)
    #             component.log._log_value(value, context)
