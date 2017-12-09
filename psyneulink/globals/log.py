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

.. note::
   This is a provisional implementation of logging.  It has the functionality described below; additional features
   will be added and some may be subject to modification in future versions.

Overview
--------

A Log object is used to record the `value <Component.value>` of PsyNeuLink Components during their "life cycle" (i.e.,
when they are created, validated, and/or executed).  Every Component has a Log object, assigned to its `log
<Component.log>` attribute when the Component is created, that can be used to record its value and/or that of other
Components that belong to it.  These are stored in `entries <Log.entries>` of the Log, that contain a sequential list
of the recorded values, along with the time and context of the recording.  The conditions under which values are
recorded is specified by the `logPref <Component.logPref>` property of a Component.  While these can be set directly,
they are most easily specified using the Log's `log_items <Log.log_items>` method, together with its `loggable_items
<Log.loggable_items>` and `logged_items <Log.logged_items>` attributes that identify and track the items to be logged.
These can be useful not only for observing the behavior of a Component in a model, but also in debugging the model
during construction. The entries of a Log can be displayed in a "human readable" table using its `print_entries
<Log.print_entries>` method, and returned in CSV and numpy array formats using its `csv <Log.csv>` and `nparray
<Log.nparray>` methods.

COMMENT:
Entries can also be made by the user programmatically. Each entry contains the time at
which a value was assigned to the attribute, the context in which this occurred, and the value assigned.  This
information can be displayed using the log's `print_entries` method.
COMMENT

Creating Logs and Entries
-------------------------

A log object is automatically created for and assigned to a Component's `log <Component.log>` attribute when the
Component is created.  An entry is automatically created and added to the Log's `entries <Log.entries>` attribute
when its `value <Component.value>` or that of a Component that belongs to it is recorded in the Log.

Structure
---------

A Log is composed of `entries <Log.entries>`, each of which is a dictionary that maintains a record of the logged
values of a Component.  The key for each entry is a string that is the name of the Component, and its value is a list
of `LogEntry` tuples recording its values.  Each `LogEntry` tuple has three items:
    * *time* -- the `TIME_STEP` of the trial in which the value of the item was recorded;
    * *context* -- a string indicating the context in which the value was recorded;
    * *value* -- the value of the item.

    .. note::
       Currently the "time" field of the entry is not used, and reports indicate the entry number
       (corresonding to the number of executions of the Component), which may or may not correspond to the
       `TIME_STEP` of execution.  This will be corrected in a future release.

A Log has several attributes and methods that make it easy to manage how and when it values are recorded, and
to access its `entries <Log.entries>`:

    * `loggable_items <Log.loggable_items>` -- a dictionary with the items that can be logged in a Component's `log
      <Component.log>`;  the key for each entry is the name of a Component,  and the value is it current `LogLevel`.
    ..
    * `log_items <Log.log_items>` -- used to assign the LogLevel for one or more Components.  Components can be
      specified by their names, a reference to the Component object, in a tuple that specifies the `LogLevel` to
      assign to that Component, or in a list with a `LogLevel` to be applied to multiple items at once.
    ..
    * `logged_items <Log.logged_items>` -- a dictionary with the items that currently have entries in a Component's
      `log <Component.log>`; the key for each entry is the name of a Component, and the value is its current `LogLevel`.
    ..
    * `print_entries <Log.print_entries>` -- this prints a formatted list of the `entries <Log.entries>` in the Log.
    ..
    * `csv <Log.csv>` -- returns a CSV-formatted string with the `entries <Log.entries>` in the Log.
    ..
    * `nparray <Log.csv>` -- returns a 2d np.array with the `entries <Log.entries>` in the Log.

Loggable Items
~~~~~~~~~~~~~~

Although every Component is assigned its own Log, that records the `value <Component.value>` of that Component,
the Logs for `Mechanisms <Mechanism>` and `MappingProjections <MappingProjection>` also  provide access to and control
the Logs of their `States <State>`.  Specifically the Logs of these Components contain the following information:

* **Mechanisms**

  * *InputStates* -- the `value <InputState.value>` of any `InputState` (listed in the Mechanism's `input_states
    <Mechanism_Base.input_states>` attribute).
  |
  * *ParameterStates* -- the `value <ParameterState.value>` of `ParameterState` (listed in the Mechanism's
    `parameter_states <Mechanism_Base.parameter_states>` attribute);  this includes all of the `user configurable
    <Component_User_Params>` parameters of the Mechanism and its `function <Mechanism_Base.function>`.
  |
  * *OutputStates* -- the `value <OutputState.value>` of any `OutputState` (listed in the Mechanism's `output_states
    <Mechanism_Base.output_states>` attribute).
..
* **Projections**

  * *MappingProjections* -- the value of its `matrix <MappingProjection.matrix>` parameter.

LogLevels
~~~~~~~~~

Configuring a Component to be logged is done using a `LogLevel`, that specifies the conditions under which its
`value <Component.value>` should be entered in its Log.  These can be specified in the `log_items <Log.log_items>`
method of a Log, or directly by specifying a LogLevel for the value a Component's `logPref  <Compnent.logPref>` item
of its `prefs <Component.prefs>` attribute.  The former is easier, and allows multiple Components to be specied at
once, while the latter affords more control over the specification (see `Preferences`).  LogLevels are treated as
binary "flags", and can be combined to permit logging under more than one contact or boolean combinations of LogLevels
using bitwise operators (e.g., LogLevel.EXECUTION | LogLevel.LEARNING).

.. note::
   Currently, the only `LogLevels <LogLevel>` supported are `OFF`, `EXECUTION` and `LEARNING`.


Execution
---------

The value of a Component is recorded to a Log when the condition assigned to its `logPref <Component.logPref>` is met.
This specified as a `LogLevel`.  The default LogLevel is `OFF`.

Examples
--------

The following example creates a Process with two `TransferMechanisms <TransferMechanism>`, one that projects to
another, and logs the `noise <TransferMechanism.noise>` and *RESULTS* `OutputState` of the first and the
`MappingProjection` from the first to the second::

    # Create a Process with two TransferMechanisms, and get a reference for the Projection created between them:
    >>> import psyneulink as pnl
    >>> my_mech_A = pnl.TransferMechanism(name='mech_A', size=2)
    >>> my_mech_B = pnl.TransferMechanism(name='mech_B', size=3)
    >>> my_process = pnl.Process(pathway=[my_mech_A, my_mech_B])
    >>> proj_A_to_B = my_mech_B.path_afferents[0]

    # Show the loggable items (and their current LogLevels) of each Mechanism and the Projection between them:
    >>> my_mech_A.loggable_items # doctest: +SKIP
    {'InputState-0': 'OFF', 'slope': 'OFF', 'RESULTS': 'OFF', 'time_constant': 'OFF', 'intercept': 'OFF', 'noise': 'OFF'}
    >>> my_mech_B.loggable_items # doctest: +SKIP
    {'InputState-0': 'OFF', 'slope': 'OFF', 'RESULTS': 'OFF', 'intercept': 'OFF', 'noise': 'OFF', 'time_constant': 'OFF'}
    >>> proj_A_to_B.loggable_items # doctest: +SKIP
    {'matrix': 'OFF'}

    # Assign the noise parameter and RESULTS OutputState of my_mech_A, and the matrix of the Projection, to be logged
    >>> my_mech_A.log_items([pnl.NOISE, pnl.RESULTS])
    >>> proj_A_to_B.log_items(pnl.MATRIX)


Executing the Process generates entries in the Logs, that can then be displayed in several ways::

    # Execute each Process twice (to generate some values in the logs):
    >>> my_process.execute()
    array([ 0.,  0.,  0.])
    >>> my_process.execute()
    array([ 0.,  0.,  0.])

    # List the items of each Mechanism and the Projection that were actually logged:
    >>> my_mech_A.logged_items  # doctest: +SKIP
    {'RESULTS': 'EXECUTION', 'noise': 'EXECUTION'}
    >>> my_mech_B.logged_items
    {}
    >>> proj_A_to_B.logged_items
    {'matrix': 'EXECUTION'}

Notice that entries dictionary of the Log for ``my_mech_B`` is empty, since no items were specified to be logged for
it.  The results of the two other logs can be printed to the console using the `print_entries <Log.print_entries>`
method of a Log::

    # Print the Log for ``my_mech_A``:
    >>> my_mech_A.log.print_entries() # doctest: +SKIP
    Log for mech_A:

    Index     Logged Item:                                       Context                                                                 Value

    0         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
    1         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0


    0         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
    1         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0


They can also be exported in csv and numpy array formats.  The following shows the csv-formatted output of the Logs
for ``my_mech_A`` and  ``proj_A_to_B``, using different formatting options::

    # Display the csv formatted entry of Log for ``my_mech_A`` without quotes around values:
    >>> my_mech_A.log.csv(entries=[pnl.NOISE, pnl.RESULTS], owner_name=False, quotes=None) # doctest: +SKIP
    'Index', 'noise', 'RESULTS'
    0,  0.,  0.
    1,  0.,  0.

    # Display the csv formatted entry of Log for ``proj_A_to_B``
    #    with quotes around values and the Projection's name included in the header:
    >>> proj_A_to_B.log.csv(entries=pnl.MATRIX, owner_name=False, quotes=True) # doctest: +SKIP
    # 'Index', 'MappingProjection from mech_A to mech_B[matrix]'
    # 0, ' 1.  1.  1.'
    #  ' 1.  1.  1.'
    # 1, ' 1.  1.  1.'
    #  ' 1.  1.  1.'

Note that since the `name <Projection.name>` attribute of the Projection was not assigned, its default name is
reported.

The following shows the Log of ``proj_A_to_B`` in numpy array format::

    >>> proj_A_to_B.log.nparray(entries=[pnl.MATRIX], owner_name=False, header=False) # doctest: +SKIP
    [[[0] [1]]
     [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]


COMMENT:

Entries are made to the Log based on the `LogLevel` specified in the
`logPref` item of the component's `prefs <Component.prefs>` attribute.

Adding an item to prefs.logPref will validate and add an entry for that attribute to the Log dict

An attribute is logged if:

* it is one `automatically included <LINK>` in logging;
..
* it is included in the *LOG_ENTRIES* entry of a `parameter specification dictionary <ParameterState_Specification>`
  assigned to the **params** argument of the constructor for the Component;
..
* the context of the assignment is above the LogLevel specified in the logPref setting of the owner Component

Entry values are added by the setter method for the attribute being logged.

The following entries are automatically included in self.entries for a `Mechanism` object:
    - the value attribute of every State for which the Mechanism is an owner
    [TBI: - value of every projection that sends to those States]
    - the system variables defined in SystemLogEntries (see declaration above)
    - any variables listed in the params[LOG_ENTRIES] of a Mechanism


DEFAULT LogLevel FOR ALL COMPONENTS IS *OFF*


Structure
---------

Each entry of `entries <Log.entries>` has:
    + a key that is the name of the attribute being logged
    + a value that is a list of sequentially entered LogEntry tuples since recording of the attribute began
    + each tuple has three items:
        - time (CentralClock): when it was recorded in the run
        - context (str): the context in which it was recorded (i.e., where the attribute value was assigned)
        - value (value): the value assigned to the attribute

The LogLevel class (see declaration above) defines six levels of logging:
    + OFF: No logging for attributes of the owner object
    + VALUE_ASSIGNMENT: Log values only when final value assignment has been made during execution
    + EXECUTION: Log values for all assignments during execution (e.g., including aggregation of projections)
    + VALIDATION: Log value assignments during validation as well as execution and initialization
    + ALL_ASSIGNMENTS:  Log all value assignments (e.g., including initialization)
    Note: LogLevel is an IntEnum, and thus its values can be used directly in numerical comparisons

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
import warnings
import inspect
import typecheck as tc
from collections import namedtuple
# from enum import IntEnum, unique, auto
from enum import IntEnum, unique

import numpy as np

from psyneulink.scheduling.time import TimeScale
from psyneulink.globals.utilities import ContentAddressableList, AutoNumber
from psyneulink.globals.keywords import INITIALIZING, EXECUTING, VALIDATE, LEARNING, CONTROL, VALUE, \
    kwContext, kwTime, kwValue


__all__ = [
    'ALL_ENTRIES', 'EntriesDict', 'kpCentralClock', 'Log', 'LogEntry', 'LogError', 'LogLevel', 'SystemLogEntries',
]


class LogLevel(IntEnum):
    """Specifies levels of logging, as descrdibed below."""
    OFF = 0
    """No recording."""
    INITIALIZATION = 1<<1           # 2
    """Record during initial assignment."""
    VALIDATION = 1<<2               # 4
    """Record value during validation."""
    EXECUTION = 1<<3                # 8
    """Record all value assignments during any execution of the Component."""
    PROCESSING = 1<<4               # 16
    """Record all value assignments during processing phase of Composition execution."""
    # FIX: IMPLEMENT EXECUTION+LEARNING CONDITION
    # LEARNING = 1<<5               # 32
    LEARNING = (1<<5) + EXECUTION   # 40
    """Record all value assignments during learning phase of Composition execution."""
    CONTROL = 1<<6                  # 64
    """Record all value assignment during control phase of Composition execution."""
    VALUE_ASSIGNMENT = 1<<7         # 128
    """Record final value assignments during Composition execution."""
    FINAL = 1<<8                    # 256
    """Synonym of VALUE_ASSIGNMENT."""
    ALL_ASSIGNMENTS = \
        INITIALIZATION | VALIDATION | EXECUTION | PROCESSING | LEARNING | CONTROL | VALUE_ASSIGNMENT | FINAL
    """Record all value assignments."""

LogEntry = namedtuple('LogEntry', 'time, context, value')

ALL_ENTRIES = 'all entries'
TIME_NOT_SPECIFIED = 'Time Not Specified'

def _get_log_context(context):

    context_flag = LogLevel.OFF
    if INITIALIZING in context:
        context_flag |= LogLevel.INITIALIZATION
    if VALIDATE in context:
        context_flag |= LogLevel.VALIDATION
    if EXECUTING in context:
        context_flag |= LogLevel.EXECUTION
    if LEARNING in context:
        context_flag |= LogLevel.LEARNING
    return context_flag


class LogTimeScaleIndices(AutoNumber):
    RUN = ()
    TRIAL = ()
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

    if all(t is not None for t in time ):
        time_str = ":".join([str(i) for i in time])
    else:
        time_str = "None"
    return time_str


kpCentralClock = 'CentralClock'
SystemLogEntries = [kpCentralClock]

#region Custom Entries Dict
# Modified from: http://stackoverflow.com/questions/7760916/correct-useage-of-getter-setter-for-dictionary-values
from collections import MutableMapping
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
            - the context of the assignment is above the LogLevel specified in the logPref setting of the owner object
        Entry values are added by the setter method for the attribute being logged
        The following entries are automatically included in self.entries for a Mechanism object:
            - the value attribute of every State for which the Mechanism is an owner
            [TBI: - value of every projection that sends to those States]
            - the system variables defined in SystemLogEntries (see declaration above)
            - any variables listed in the params[LOG_ENTRIES] of a Mechanism
        The LogLevel class (see declaration above) defines five levels of logging:
            + OFF: No logging for attributes of the owner object
            + VALUE_ASSIGNMENT: Log values only when final value assignment has been during execution
            + EXECUTION: Log values for all assignments during exeuction (e.g., including aggregation of projections)
            + VALIDATION: Log value assignments during validation as well as execution
            + ALL_ASSIGNMENTS:  Log all value assignments (e.g., including initialization)
            Note: LogLevel is an IntEnum, and thus its values can be used directly in numerical comparisons

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

    owner : Compoment
        the `Component <Component>` to which the Log belongs (and assigned as its `log <Component.log>` attribute).

    loggable_components : ContentAddressableList
        each item is a Component that is loggable for the Log's `owner <Log.owner>`

    loggable_items : Dict[Component.name: List[LogEntry]]
        identifies Components that can be logged in the Log; the key of each entry is the name of a Component,
        and the value is its currently assigned `LogLevel`.

    entries : Dict[Component.name: List[LogEntry]]
        contains the logged information for `loggable_components <Log.loggable_components>`; the key of each entry
        is the name of a Component, and its value is a list of `LogEntry` items for that Component.  Only Components
        for which information has been logged appear in the `entries <Log.entries>` dict.

    logged_items : Dict[Component.name: List[LogEntry]]
        identifies Components for which information has been entered in the Log; the key for each entry is the name
        of a Component, and the value is its currently assigned `LogLevel`.

    """

    ALL_LOG_ENTRIES = 'all_log_entries'

    def __init__(self, owner, entries=None):
        """Initialize Log with list of entries

        Each item of the entries list should be a string designating a Component to be logged;
        Initialize self.entries dict, each entry of which has a:
        - key corresponding to an attribute of the object to be logged
        - value that is a list of sequentially logged values

        :parameter owner: (object in Function hierarchy) - parent object that owns the Log object)
        :parameter entries: (list) - list of keypaths used as keys for entries in the Log dict
        """

        self.owner = owner
        # self.entries = EntriesDict({})
        self.entries = EntriesDict(self)

        if entries is None:
            return

        # self.add_entries(entries)

    @property
    def loggable_items(self):
        """Return dict of loggable items

        Keys are names of the items, values the items themselves
        """
        # FIX: The following crashes during init as prefs have not all been assigned
        # return {key: value for (key, value) in [(c.name, c.logPref.name) for c in self.loggable_components]}

        loggable_items = {}
        for c in self.loggable_components:
            if c is self.owner:
                name = VALUE
            else:
                name = c.name
            try:
                log_pref = c.logPref.name
            except:
                log_pref = None
            loggable_items[name] = log_pref
        return loggable_items


    @property
    def loggable_components(self):
        """Return a list of owner's Components that are loggable

        The loggable items of a Component are specified in the _logagble_items property of its class
        """
        from psyneulink.components.component import Component

        try:
            loggable_items = ContentAddressableList(component_type=Component, list=self.owner._loggable_items)
            loggable_items[self.owner.name] = self.owner
        except AttributeError:
            return []
        return loggable_items

    @property
    def logged_items(self):
        """Dict of items that have logged `entries <Log.entries>`, indicating their specified `LogLevel`.
        """
        log_level = 'LogLevel.'
        # Return LogLevel for items in log.entries
        logged_items = {key: value for (key, value) in
                        [(l, self.loggable_items[l])
                         for l in self.logged_entries.keys()]}
        return logged_items

    def log_items(self, items, log_level=LogLevel.EXECUTION):
        """Specifies items to be logged at the specified `LogLevel`\\(s).

        Arguments
        ---------

        items : str, Component, tuple or List of these
            specifies items to be logged;  these must be be `loggable_items <Log.loggable_items>` of the Log.
            Each item must be a:
            * string that is the name of a `loggable_item` <Log.loggable_item>` of the Log's `owner <Log.owner>`;
            * a reference to a Component;
            * tuple, the first item of which is one of the above, and the second a `LogLevel` to use for the item.

        log_level : LogLevel : default LogLevel.EXECUTION
            specifies `LogLevel` to use as the default for items not specified in tuples (see above).
            For convenience, the name of a LogLevel can be used in place of its full specification
            (e.g., *EXECUTION* instead of `LogLevel.EXECUTION`).

        params_set : list : default None
            list of parameters to include as loggable items;  these must be attributes of the `owner <Log.owner>`
            (for example, Mechanism

        """
        from psyneulink.components.component import Component
        from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
        from psyneulink.globals.keywords import ALL

        def assign_log_level(item, level):

            try:
                level = LogLevel[level] if isinstance(level, str) else level
            except KeyError:
                raise LogError("\'{}\' is not a value of {}".
                               format(level, LogLevel.__name__))

            if not item in self.loggable_items:
                raise LogError("\'{0}\' is not a loggable item for {1} (try using \'{1}.log.add_entries()\')".
                               format(item, self.owner.name))
            try:
                component = next(c for c in self.loggable_components if c.name == item)
                component.logPref=PreferenceEntry(level, PreferenceLevel.INSTANCE)
            except AttributeError:
                raise LogError("PROGRAM ERROR: Unable to set LogLevel for {} of {}".format(item, self.owner.name))

        if items is ALL:
            for component in self.loggable_components:
                component.logPref = PreferenceEntry(log_level, PreferenceLevel.INSTANCE)
            # self.logPref = PreferenceEntry(log_level, PreferenceLevel.INSTANCE)
            return

        if not isinstance(items, list):
            items = [items]

        for item in items:
            if isinstance(item, (str, Component)):
                # self.add_entries(item)
                if isinstance(item, Component):
                    item = item.name
                assign_log_level(item, log_level)
            else:
                # self.add_entries(item[0])
                assign_log_level(item[0], item[1])

    def _log_value(self, value, context=None):
        """Add LogEntry to an entry in the Log

        Identifies the context in which the call is being made, which is assigned to the context field of the
        `LogEntry`, along with the current time stamp and value itself

        .. note::
            Since _log_value is usually called by the setter for the `value <Component.value>` property of a Component
            (which doesn't/can't receive a context argument), it does not pass a **context** argument to _log_value;
            in that case, _log_value searches the stack for the most recent frame with a context specification, and
            uses that.

        """
        from psyneulink.components.component import Component

        # Get context
        if context is None:
            curr_frame = inspect.currentframe()
            prev_frame = inspect.getouterframes(curr_frame, 2)
            i = 1
            # Search stack for first frame (most recent call) with a context specification
            while context is None:
                try:
                    context = inspect.getargvalues(prev_frame[i][0]).locals['context']
                except KeyError:
                    # Try earlier frame
                    i += 1
                except IndexError:
                    # Ran out of frames, so just set context to empty string
                    context = ""
                else:
                    break

        if context is None:
            # No context was specified in any frame
            raise LogError("PROGRAM ERROR: No context specification found in any frame")

        # If context is a Component object, it must be during its initialization, so assign accordingly:
        if isinstance(context, Component):
            context = "{} of {}".format(INITIALIZING, context.name)

        if not isinstance(context, str):
            raise LogError("PROGRAM ERROR: Unrecognized context specification ({})".format(context))

        context_flags = _get_log_context(context)

        log_pref = self.owner.prefs.logPref if self.owner.prefs else None

        # Log value if logging condition is satisfied
        if log_pref and log_pref == context_flags:
        # FIX: IMPLEMENT EXECUTION+LEARNING CONDITION
        # if log_pref and log_pref | context_flags:

            self.entries[self.owner.name] = LogEntry(self._get_time(context, context_flags), context, value)

    def _get_time(self, context, context_flags):
        """Get time from Scheduler of System in which Component is being executed.

        Returns tuple with (run, trial, time_step) if being executed during Processing or Learning
        Otherwise, returns (None, None, None)

        """

        from psyneulink.components.mechanisms.mechanism import Mechanism
        from psyneulink.components.states.state import State
        from psyneulink.components.projections.projection import Projection

        no_time = (None, None, None)

        if isinstance(self.owner, Mechanism):
            ref_mech = self.owner
        elif isinstance(self.owner, State):
            if isinstance(self.owner.owner, Mechanism):
                ref_mech = self.owner.owner
            elif isinstance(self.owner.owner, Projection):
                ref_mech = self.owner.owner.receiver.owner
            else:
                raise LogError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                               format(self.owner.__class__.__name__,
                                      Mechanism.__name__, State.__name__, Projection.__name__))
        elif isinstance(self.owner, Projection):
            ref_mech = self.owner.receiver.owner
        else:
            raise LogError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                           format(self.owner.__class__.__name__,
                                  Mechanism.__name__, State.__name__, Projection.__name__))

        systems = list(ref_mech.systems.keys())
        system = next((s for s in systems if s.name in context), None)

        if system:
            # FIX: Add INIT and VALIDATE?
            if context_flags == LogLevel.EXECUTION:
                time = system.scheduler_processing.clock.simple_time
                time = (time.run, time.trial, time.time_step)
            elif context_flags == LogLevel.LEARNING:
                time = system.scheduler_learning.clock.simple_time
                time = (time.run, time.trial, time.time_step)
            else:
                time = None

        else:
            if self.owner.verbosePref:
                offender = "\'{}\'".format(self.owner.name)
                if ref_mech is not self.owner:
                    offender += " [{} of {}]".format(self.owner.__class__.__name__, ref_mech.name)
                warnings.warn("Attempt to log {} which is not in a System (logging is currently supported only "
                              "when running Components within a System".format(offender))
            time = None

        return time or no_time

    def print_entries(self,
                      entries=None,
                      csv=False,
                      # synch_time=False,
                      *args):
        """
        print_entries(          \
              entries=None,     \
              csv=False,        \
            )

        Print values of entries

        If entries is the keyword *ALL_ENTRIES*, print all entries in the self.owner.prefs.logPref list
        Issue a warning if an entry is not in the Log dict
        """

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.logged_entries
        if entries is ALL_ENTRIES or entries is None:
            entries = self.logged_entries.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, str):
            entries = [entries]

        if csv is True:
            print(self.csv(entries))
            return

        variable_width = 50
        time_width = 10
        # time_width = 5
        context_width = 70
        value_width = 7
        kwSpacer = ' '


        # MODIFIED 12/4/17 OLD: [USES Time]
        # header = "Variable:".ljust(variable_width, kwSpacer)
        # if not args or kwTime in args:
        #     header = header + " " + kwTime.ljust(time_width, kwSpacer)
        # if not args or kwContext in args:
        #     header = header + " " + kwContext.ljust(context_width, kwSpacer)
        # if not args or kwValue in args:
        #     # header = header + "   " + kwValue.rjust(value_width)
        #     header = header + "  " + kwValue
        # MODIFIED 12/4/17 NEW: [USES entry]
        header = "Logged Item:".ljust(variable_width, kwSpacer)
        if not args or kwTime in args:
            header = "Time".ljust(time_width, kwSpacer) + header
        if not args or kwContext in args:
            header = header + " " + kwContext.ljust(context_width, kwSpacer)
        if not args or kwValue in args:
            header = header + "  " + kwValue
        # MODIFIED 12/4/17 END

        print("\nLog for {0}:".format(self.owner.name))
        print('\n'+header+'\n')

        # Sort for consistency of reporting
        attrib_names_sorted = sorted(self.logged_entries.keys())
        kwSpacer = '.'
        # for attrib_name in self.logged_entries:
        for attrib_name in attrib_names_sorted:
            try:
                datum = self.logged_entries[attrib_name]
            except KeyError:
                warnings.warn("{0} is not an entry in the Log for {1}".
                      format(attrib_name, self.owner.name))
            else:
                import numpy as np
                for i, item in enumerate(datum):
                    # MODIFIED 12/4/17 OLD: [USES CentralClock FOR TIME]
                    # time, context, value = item
                    # if isinstance(value, np.ndarray):
                    #     value = value[0]
                    # time_str = str(time.task) +":"+ str(time.block) +":"+ str(time.trial) +":"+ str(time.time_step)
                    # data_str = attrib_name.ljust(variable_width, kwSpacer)
                    # if not args or kwTime in args:
                    #     data_str = data_str + " " + time_str.ljust(time_width)
                    # if not args or kwContext in args:
                    #     data_str = data_str + context.ljust(context_width, kwSpacer)
                    # if not args or kwValue in args:
                    #     # data_str = data_str + " " + str(value).rjust(value_width) # <- WORKS
                    #     # data_str = data_str + " " + "{:10.5}".format(str(value).strip("[]"))  # <- WORKS
                    #     data_str = data_str + "{:2.5}".format(str(value).strip("[]")).rjust(value_width) # <- WORKS
                    #     # data_str = data_str + "{:10.5}".format(str(value).strip("[]")) # <- WORKS
                    # MODIFIED 12/4/17 NEW [USES entry index RATHER THAN CentralClock]
                    time, context, value = item
                    if isinstance(value, np.ndarray):
                        value = value[0]
                    time_str = _time_string(time)
                    data_str = repr(attrib_name).ljust(variable_width, kwSpacer)
                    if not args or kwTime in args:
                        data_str = time_str.ljust(time_width) + data_str
                    if not args or kwContext in args:
                        data_str = data_str + repr(context).ljust(context_width, kwSpacer)
                    if not args or kwValue in args:
                        data_str = data_str + "{:2.5}".format(str(value).strip("[]")).rjust(value_width) # <- WORKS
                    # MODIFIED 12/4/17 END

        # {time:{width}}: {part[0]:>3}{part[1]:1}{part[2]:<3} {unit:3}".format(
        #     jid=jid, width=width, part=str(mem).partition('.'), unit=unit))

                    print(data_str)
                if len(datum) > 1:
                    print("\n")

    @tc.typecheck
    def csv(self, entries=None, owner_name:bool=False, quotes:tc.optional(tc.any(bool, str))="\'"):
        """
        csv(                           \
            entries=None,              \
            owner_name=False,          \
            quotes=\"\'\"              \
            )

        Returns a CSV-formatted string with headers and values for the specified entries.

        The first record (row) begins with "Index" and is followed by the header for each field (column).
        Subsequent records begin with the record number, and are followed by the value for that entry.
        Records are ordered in the same order as the Components specified in the **entries** argument.

        .. note::
           Currently only supports reports of entries with the same length.  A future version will allow
           entries of differing lengths in the same report.

        Arguments
        ---------

        entries : string, Component or list of them
            specifies the entries of the Log to be included in the output;  they must be `loggable_items
            <Log.loggable_items>` of the Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is `ALL` or `None`, then all `logged_items <Log.logged_items>` are included.

        owner_name : bool : default False
            specifies whether or not to include the Component's `owner <Log.owner>` in the header of each field;
            if it is True, the format of the header for each field is "<Owner name>[<entry name>]"; otherwise,
            it is "<entry name>".

        quotes : bool, str : default '
            specifies whether or not to use quotes around values (useful if they are arrays);
            if not specified or `True`, single quotes are used;
            if `False` or `None`, no quotes are used;
            if specified with a string, that is used.

        Returns:
            CSV-formatted string
        """
        from psyneulink.components.component import Component

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.logged_entries
        if entries is ALL_ENTRIES or entries is None:
            # entries = self.logged_entries.keys()
            entries = self.logged_entries.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, (str, Component)):
            entries = [entries]

        # Make sure all entries are the names of Components
        entries = [entry.name if isinstance(entry, Component) else entry for entry in entries ]

        # Validate entries
        for entry in entries:
            if entry not in self.loggable_items:
                raise LogError("{0} is not a loggable attribute of {1}".format(repr(entry), self.owner.name))
            if entry not in self.logged_entries:
                raise LogError("{} is not currently being logged by {} (try using log_items)".
                               format(repr(entry), self.owner.name))

        max_len = max([len(self.logged_entries[e]) for e in entries])

        # Currently only supports entries of the same length
        if not all(len(self.logged_entries[e])==len(self.logged_entries[entries[0]])for e in entries):
            raise LogError("CSV output currently only supported for Log entries of equal length")

        if not quotes:
            quotes = ""
        elif quotes is True:
            quotes = "\'"

        if owner_name is True:
            owner_name_str = self.owner.name
            lb = "["
            rb = "]"
        else:
            owner_name_str = lb = rb = ""

        # Header
        csv = "\'Index', {}\n".format(", ".join(repr("{}{}{}{}".format(owner_name_str, lb, entry, rb))
                                                     for entry in entries))
        # Records
        for i in range(max_len):
            csv += "{}, {}\n".format(i, ", ".
                                     join(str(self.logged_entries[entry][i].value) for entry in entries).
                                     replace("[[",quotes)).replace("]]",quotes).replace("[",quotes).replace("]",quotes)
        return(csv)

    @tc.typecheck
    def nparray(self,
                entries=None,
                # time:TimeScale=TimeScale.TIME_STEP,
                header:bool=True,
                owner_name:bool=False
                ):
        """
        nparray(                 \
            entries=None,        \
            header:bool=True,    \
            owner_name=False):   \
            )

        Return a 2d numpy array with headers (optional) and values for the specified entries.

        First row (axis 0) is a sequential index (starting with 0), and each subsequent row is the
        sequence ("time series") of logged values for given Component. Rows are ordered in the same order as
        Components are specified in the **entries** argument.  If header is `True`, the first item of each row is
        a header field: for the first row, it is "Index", and for subsequent rows it is the name of the
        Component logged in that entry.

        .. note::
           Currently only supports entries with the same length.  A future version will allow entries of differing
           lengths in the same report.

        Arguments
        ---------

        entries : string, Component or list of them
            specifies the entries of the Log to be included in the output;  they must be `loggable_items
            <Log.loggable_items>` of the Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is `ALL` or `None`, then all `logged_items <Log.logged_items>` are included.

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

        Returns:
            2d np.array
        """
        from psyneulink.components.component import Component

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.logged_entries
        if entries is ALL_ENTRIES or entries is None:
            entries = self.logged_entries.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, (str, Component)):
            entries = [entries]

        # Make sure all entries are the names of Components
        entries = [entry.name if isinstance(entry, Component) else entry for entry in entries ]

        # Validate entries
        for entry in entries:
            if entry not in self.loggable_items:
                raise LogError("{0} is not a loggable attribute of {1}".format(repr(entry), self.owner.name))
            if entry not in self.logged_entries:
                raise LogError("{} is not currently being logged by {} (try using log_items)".
                               format(repr(entry), self.owner.name))

        max_len = max([len(self.logged_entries[e]) for e in entries])

        # Currently only supports entries of the same length
        if not all(len(self.logged_entries[e])==len(self.logged_entries[entries[0]])for e in entries):
            raise LogError("CSV output currently only supported for Log entries of equal length")

        if owner_name is True:
            owner_name_str = self.owner.name
            lb = "["
            rb = "]"
        else:
            owner_name_str = lb = rb = ""

        header = 1 if header is True else 0

        # Get time values for all entries and sort them
        time_values = []
        for entry in entries:
            time_values.extend([item.time
                                for item in self.logged_entries[entry]
                                if all(i is not None for i in item.time)])
        if all(all(i for i in t) for t in time_values):
            for time_scale in LogTimeScaleIndices:
                time_values.sort(key=lambda tup: tup[time_scale])
        else:
            pass
            # FIX: ADD HANDLING OF None IN TIME POINTS HERE

        num_time_points = len(time_values)

        # Create time rows (one for each time scale)
        npa = []
        for i in range(NUM_TIME_SCALES):
            row = [[t[i]] for t in time_values]
            if header:
                time_header = [TIME_SCALE_NAMES[i].capitalize()]
                row = [time_header] + row
            npa.append(row)
            # FIX: ADD HANDLING OF None IN TIME POINTS HERE

        # For each entry, iterate through its LogEntry tuples:
        #    for each LogEntry tuple, check whether its time matches that of the next column:
        #        if so, enter it in the entry's list
        #        if not, enter `None` and check for a match in the next time column
        for entry in entries:
            row = []
            time_col = iter(time_values)
            for datum in self.logged_entries[entry]:
                while datum.time != next(time_col,None):
                    row.append(None)
                row.append(datum.value)
            if header:
                entry_header = "{}{}{}{}".format(owner_name_str, lb, entry, rb)
                row = [entry_header] + row
            npa.append(row)

        npa = np.array(npa, dtype=object)
        return(npa)

    @property
    def logged_entries(self):
        entries = {}
        for e in self.loggable_components:
            entries.update(e.log.entries)
        return entries

    def clear_entries(self, entries=ALL_LOG_ENTRIES, delete_entry=True, confirm=False):
        """Clear one or more entries either by deleting the entry or just removing its data.

        Arguments
        ---------

        entries : string, Component or list of them : default None
            specifies the entries of the Log to be cleared;  they must be `loggable_items
            <Log.loggable_items>` of the Log that have been logged (i.e., are also `logged_items <Log.logged_items>`).
            If **entries** is `ALL`, `None` or omitted, then all `logged_items <Log.logged_items>` are cleared.

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
                If **confirm** is `True`, only a single confirmation will occur for a list or Log.ALL_LOG_ENTRIES

        """

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.entries
        if entries in {None, Log.ALL_LOG_ENTRIES}:
            entries = self.logged_items.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, str):
            entries = [entries]

        # Validate each entry and delete bad ones from entries
        for entry in entries:
            try:
                self.logged_items[entry]
            except KeyError:
                if self.owner.verbosePref:
                    warnings.warn("Warning: {0} is not an entry in Log of {1}".
                                  format(entry,self.owner.name))
                del(entries, entry)

        # If any entries remain
        if entries:
            if confirm:
                delete = input("\nAll data will be deleted from {0} in the Log for {1}.  Proceed? (y/n)".
                               format(entries,self.owner.name))
                while delete != 'y' and delete != 'y':
                    input("\nDelete all data from entries? (y/n)")
                if delete == 'n':
                    return

            # Reset entries
            for entry in entries:
                self.logged_entries[entry]=[]
                if delete_entry:
                # Delete the entire entry from the log to which it belongs
                    del self.loggable_components[entry].log.entries[entry]
                else:
                    # Delete the data for the entry but leave the entry itself in the log to which it belongs
                    del self.logged_entries[entry][0:]
                assert True

    # def save_log(self):
    #     print("Saved")
