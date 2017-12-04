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

COMMENT:
2 sets of methods:
  ones that manage entries
  ones that manage logging (setting the level for an entry)
Add description of loggable_items (including class-specific ones)
Add description of log_items property

COMMENT

Overview
--------

A Log object is used to record information about PsyNeuLink Components during their "life cycle" (i.e., as they are
created, validated and executed).  Every Component has a log object -- assigned to its `log <Component.log>` attribute
when the Component is created -- that maintains a dictionary with entries for each attribute of the Component that
has been designated to be logged.  Information is added to the entries under specified conditions (e.g., when the
Component is initialized, validated, or executed), which can be designated by a `LogLevel` specification in the
component's preferences.  Entries can also be made by the user programmatically. Each entry contains the time at
which a value was assigned to the attribute, the context in which this occurred, and the value assigned.  This
information can be displayed using the log's `print_entries` method.

Creating Logs and Entries
-------------------------

Whenever any PsyNeuLink Component is created, a log object is also automatically created and assigned to the
Component's `log <Component.log>` attribute.  Entries are made to the log based on the `LogLevel` specified in the
`logPref` item of the component's `prefs <Component.prefs>` attribute.

Adding an item to prefs.logPref will validate and add an entry for that attribute to the log dict

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


DEFAULT LogLevel FOR ALL COMPONENTS IS *VALUE_ASSIGNMENT*

Structure
---------

Each entry of log.entries has:
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


Logging Entries
---------------


.. _Log_Class_Reference:

Class Reference
---------------

"""
import warnings
import typecheck as tc
from collections import namedtuple
from enum import IntEnum

from psyneulink.globals.keywords import kwContext, kwTime, kwValue

__all__ = [
    'ALL_ENTRIES', 'EntriesDict', 'kpCentralClock', 'Log', 'LogEntry', 'LogError', 'LogLevel', 'SystemLogEntries',
]


class LogLevel(IntEnum):
    """Specifies levels of logging, as descrdibed below."""
    OFF = 0
    """No recording."""
    INITIALIZATION = 1
    """Record only initial assignment."""
    VALUE_ASSIGNMENT = 2
    """Record only final value assignments during execution."""
    EXECUTION = 3
    """Record all value assignments during execution."""
    VALIDATION = 5
    """Record all value assignemnts during validation and execution."""
    ALL_ASSIGNMENTS = 5
    """Record all value assignments during initialization, validation and execution."""

LogEntry = namedtuple('LogEntry', 'time, context, value')

ALL_ENTRIES = 'all entries'

kpCentralClock = 'CentralClock'
SystemLogEntries = [kpCentralClock]

#region Custom Entries Dict
# Modified from: http://stackoverflow.com/questions/7760916/correct-useage-of-getter-setter-for-dictionary-values
from collections import MutableMapping
class EntriesDict(MutableMapping,dict):
    """Add setter method for entries that checks owner Mechanism's prefs to see whether entry is currently recording

    If entry is in owner Mechanism's prefs.logPref.setting list, then append attribute value to entry's list
    Otherwise, either initialize or just update entry with value
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
    """Maintain a log for an object, which contains a dictionary of attributes being logged and their value(s).

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

    """

    ALL_LOG_ENTRIES = 'all_log_entries'

    def __init__(self, owner, entries=None):
        """Initialize log with list of entries

        Each item of the entries list should be a keypath (kp<attribute name>) designating an attribute
            of the object to be logged;
        Initialize self.entries dict, each entry of which has a:
        - key corresponding to an attribute of the object to be logged
        - value that is a list of sequentially logged values

        :parameter owner: (object in Function hierarchy) - parent object that owns the log object)
        :parameter entries: (list) - list of keypaths used as keys for entries in the log dict
        """

        self.owner = owner
        # self.entries = EntriesDict({})
        self.entries = EntriesDict(self)

        if entries is None:
            return

        # self.add_entries(entries)

    @property
    def loggable_items(self):
        """Return a list of loggable items for Log's owner.

        The loggable items of all Components include those listed in its `user_params <Component.user_params>` dict.
        Any specified in the `owner <Log.owner>`'s _loggable_items property are also included in the list.
        """
        try:
            return self._loggable_items + self.owner._loggable_items
        except AttributeError:
            # This forces setter to create log._loggable_items
            self.loggable_items = None
            return self._loggable_items

    @loggable_items.setter
    def loggable_items(self, items):
        """Set loggable items for the Log's owner.

        The loggable items of all Components include:

            * any entry in the Log `owner <Log.owner>`'s `user_params <Component.user_params>` dictionary, which
              includes any entries of a list or dictionary that are in the `user_params <Component.user_params>`
              dictionary.  This makes the value of all of the `InputStates <InputState>` and `OutputStates
              <OutputState>` of a `Mechanism` available for logging (by way of its `input_states
              <Mechanism_Base.input_states>` and `output_states <Mechanism_Base.output_states>` attributes in its
              `user_params <Component.user_params>` dictionary, as well as all of the parameters of its
              `function <Mechanism_Base.function>`, and the `matrix <MappingProjection.matrix>` parameter of a
              `MappingProjection`, by way of their `parameter_states` attributes.

            .. note::
               Although there are checks for duplicate names among the `input_states <Mechanism_Base.input_states>`
               and `output_states <Mechanism_Base.output_states>` of a Mechanism, there are not checks for duplicate
               names between them, or with its `parameter_states <Mechanism.parameter_states>` used to represent
               its parameter and those of its `function <Mechanism_Base.function>`.  Therefore, using the same name
               for an InputState and OutputState of a Mechanism, or one that is same as one of its parameters or a
               parameter of its `function <Mechanism_Base.function>`, can produce unpredictable results.

            * any entries assigned to the loggable_items property;  for example, Mechanisms add their `afferent
              projections <Mechanism_Base.afferent_projections>` as loggable items (see XXX for details).

        Arguments
        ---------

        items : str, Component, List
            specifies the items to be included in the Log.  Each item must be a string that is the name of an attribute
            of the Log's `owner <Log.owner>` specified either in its user_params dict, or its _loggable_items property.

        """

        # if not hasattr(self, '_loggable_items'):
        if items is None:
            from collections import UserDict, UserList

            items = []

            param_set = self.owner.user_params

            for item in self.owner.user_params:
                if isinstance(param_set[item], (list, UserList)):
                    for sub_item in param_set[item]:
                        items.append(sub_item)
                elif isinstance(param_set[item], (dict, UserDict)):
                    for sub_item in param_set[item]:
                        items.append(sub_item)
                else:
                    items.append(item)
            try:
                owner_items = self.owner._loggable_items
            except AttributeError:
                owner_items = []

            self._loggable_items = items + SystemLogEntries +  owner_items
        else:
            if not hasattr(self, '_loggable_items'):
                self.loggable_items = None
            # items = [item if isinstance(item, str) else item.name for item in items]
            self._loggable_items += items

    # def add_entries(self, entries):
    #     """Validate that a list of entries are attributes of owner or in SystemLogEntries, and then add to self.entries
    #
    #     entries should be a single keypath or list of keypaths for attribute(s) of the owner
    #     Note: adding an entry does not mean data will be recorded;
    #               to activate recording, the log_entries() method must be called
    #     """
    #     from psyneulink.components.component import Component
    #
    #     # If entries is a single entry, put in list for processing below
    #     if isinstance(entries, (str, Component)):
    #         entries = [entries]
    #
    #     for entry in entries:
    #         if isinstance(entry, Component):
    #             entry = entry.name
    #
    #         #Check if entries already exist
    #         try:
    #             self.entries[entry]
    #         # Entry doesn't already exist
    #         except KeyError:
    #             # Validate that entry is either an attribute of owner or in SystemLogEntries
    #             # if not entry in self.loggable_items:
    #             #     raise LogError("{} is not a loggable item for {}".
    #             #                    format(entry, self.owner.name))
    #             # # Add entry to self.entries dict
    #             # self.entries[entry] = []
    #             # TEMP = self.loggable_items
    #             # self.loggable_items=[entry]
    #             pass
    #
    #         # Entry exists
    #         else:
    #             # Issue warning and ignore
    #             warnings.warn("{0} is already an entry in log for {1}; use \"log_entry\" to add a value".
    #                   format(entry,self.owner.name))

    def delete_entry(self, entries, confirm=True):
        """Delete entry for attribute from self.entries

        If verify is True, user will be asked to confirm deletion;  otherwise it will simply be done
        Note: deleting the entry will delete all the data recorded within it
        Entries can be a single entry, a list of entries, or the keyword Log.ALL_LOG_ENTRIES;
        Notes:
        * only a single confirmation will occur for a list or Log.ALL_LOG_ENTRIES
        * deleting entries removes them from log dict, owner.prefs.logPref, and deletes ALL data recorded in them

        :param entries: (str, list, or Log.ALL_LOG_ENTRIES)
        :param confirm: (bool)
        :return:
        """

        msg = ""

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.entries
        if entries is Log.ALL_LOG_ENTRIES:
            entries = self.entries.keys()
            msg = Log.ALL_LOG_ENTRIES

        # If entries is a single entry, put in list for processing below
        elif isinstance(entries, str):
            entries = [entries]

        # Validate each entry and delete bad ones from entries
        if not msg is Log.ALL_LOG_ENTRIES:
            for entry in entries:
                try:
                    self.entries[entry]
                except KeyError:
                    warnings.warn("Warning: {0} is not an entry in log of {1}".
                                  format(entry,self.owner.name))
                    del(entries, entry)
            if len(entries) > 1:
                msg = ', '.join(str(entry) for entry in entries)

        # If any entries remain
        if entries:
            if confirm:
                delete = input("\n{0} will be deleted (along with any recorded date) from log for {1}.  Proceed? (y/n)".
                               format(msg, self.owner.name))
                while delete != 'y' and delete != 'y':
                    input("\nRemove entries from log for {0}? (y/n)".format(self.owner.name))
                if delete == 'n':
                    warnings.warn("No entries deleted")
                    return

            # Reset entries
            for entry in entries:
                self.entries[entry]=[]
                if entry in self.owner.prefs.logPref:
                    del(self.owner.prefs.logPref, entry)

    def reset_entries(self, entries, confirm=True):
        """Reset one or more entries by removing all data, but leaving entries in log dict

        If verify is True, user will be asked to confirm the reset;  otherwise it will simply be done
        Entries can be a single entry, a list of entries, or the keyword Log.ALL_LOG_ENTRIES;
        Notes:
        * only a single confirmation will occur for a list or Log.ALL_LOG_ENTRIES
        * resetting an entry deletes ALL the data recorded within it

        :param entries: (list, str or Log.ALL_LOG_ENTRIES)
        :param confirm: (bool)
        :return:
        """

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.entries
        if entries is Log.ALL_LOG_ENTRIES:
            entries = self.entries.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, str):
            entries = [entries]

        # Validate each entry and delete bad ones from entries
        for entry in entries:
            try:
                self.entries[entry]
            except KeyError:
                warnings.warn("Warning: {0} is not an entry in log of {1}".
                              format(entry,self.owner.name))
                del(entries, entry)

        # If any entries remain
        if entries:
            if confirm:
                delete = input("\nAll data will be deleted from {0} in the log for {1}.  Proceed? (y/n)".
                               format(entries,self.owner.name))
                while delete != 'y' and delete != 'y':
                    input("\nDelete all data from entries? (y/n)")
                if delete == 'n':
                    return

            # Reset entries
            for entry in entries:
                self.entries[entry]=[]

    # def log_entries(self, entries):
    #     """Record values of attributes corresponding to entries when logging is on
    #
    #     Add entries to self.owner.prefs.logPref;  these will be receive logging data when logging is one
    #     Any entries not already in the log dict will be added to it as well
    #
    #     :param entries: (str or list)
    #     :return:
    #     """
    #
    #     # If entries is a single entry, put in list for processing below
    #     if isinstance(entries, str):
    #         entries = [entries]
    #
    #     # Check whether each entry is already in self.entries and, if not, validate and add it
    #     for entry in entries:
    #         try:
    #             self.entries[entry]
    #         except KeyError:
    #             # Validate that entry is either an attribute of owner or in SystemLogEntries
    #             if not entry in SystemLogEntries and not entry in self.owner.__dict__:
    #                 raise LogError("{0} is not an attribute of {1} or in SystemLogEntries".
    #                                format(entry, self.owner.name))
    #             # Add entry to self.entries dict
    #             self.entries[entry] = []
    #         self.owner.prefs.logPref.append(entry)
    #         if self.owner.prefs.verbosePref:
    #             print("Started logging of {0}".format(entry))
    #
    # def log_entries(self, entries):
    #     """Record values of attributes corresponding to entries
    #
    #     Issue a warning if the entry is not in the log dict or owner's logPrefs list
    #
    #     :param entries: (str, list or Log.ALL_LOG_ENTRIES)
    #     :return:
    #     """
    #
    #     # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.entries
    #     if entries is Log.ALL_LOG_ENTRIES:
    #         entries = self.entries.keys()
    #         msg = Log.ALL_LOG_ENTRIES
    #
    #     # If entries is a single entry, put in list for processing below
    #     elif isinstance(entries, str):
    #         entries = [entries]
    #
    #     # Otherwise, if paramsValidation is on, validate each entry and delete bad ones from entries
    #     if self.owner.prefs.paramValidationPref:
    #         for entry in entries:
    #             try:
    #                 self.entries[entry]
    #             except KeyError:
    #                 print("Warning: {0} is not an entry in the log for {1}".
    #                       format(entry,self.owner.name))
    #                 del(entries, entry)
    #
    #     for entry in entries:
    #         self.entries[entry].append(getattr(self.owner, entry))
    #
    def suspend_entries(self, entries):
        """Suspend recording the values of attributes corresponding to entries even if logging is on

        Remove entries from self.owner.prefs.logPref (but leave in log dict, i.e., self.entries)

        :param entries: (str or list)
        :return:
        """

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, str):
            entries = [entries]

        # Check whether each entry is already in self.entries and, if not, validate and add it
        for entry in entries:
            try:
                self.owner.prefs.logPref.remove(entry)
            except ValueError:
                if not entry in SystemLogEntries and not entry in self.owner.__dict__:
                    warnings.warn("{0} is not an attribute of {1} or in SystemLogEntries".
                                  format(entry, self.owner.name))
                elif self.owner.prefs.verbosePref:
                    warnings.warn("{0} was not being recorded")
            else:
                if self.owner.prefs.verbosePref:
                    warnings.warn("Started logging of {0}".format(entry))

    def print_entries(self, entries=None, csv=False, synch_time=False, *args):
        """Print values of entries

        If entries is the keyword ALL_ENTRIES, print all entries in the self.owner.prefs.logPref list
        Issue a warning if an entry is not in the log dict
        """

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.entries
        if entries is ALL_ENTRIES or entries is None:
            entries = self.entries.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, str):
            entries = [entries]

        if csv is True:
            print(self.csv(entries))
            return

        variable_width = 50
        time_width = 10
        context_width = 70
        value_width = 7
        kwSpacer = ' '


        header = "Variable:".ljust(variable_width, kwSpacer)
        if not args or kwTime in args:
            header = header + " " + kwTime.ljust(time_width, kwSpacer)
        if not args or kwContext in args:
            header = header + " " + kwContext.ljust(context_width, kwSpacer)
        if not args or kwValue in args:
            # header = header + "   " + kwValue.rjust(value_width)
            header = header + "  " + kwValue

        print("\nLog for {0}:".format(self.owner.name))

        print('\n'+header)

        # Sort for consistency of reporting
        attrib_names_sorted = sorted(self.entries.keys())
        kwSpacer = '.'
        # for attrib_name in self.entries:
        for attrib_name in attrib_names_sorted:
            try:
                datum = self.entries[attrib_name]
            except KeyError:
                warnings.warn("{0} is not an entry in the log for {1}".
                      format(attrib_name, self.owner.name))
            else:
                import numpy as np
                for item in datum:
                    time, context, value = item
                    if isinstance(value, np.ndarray):
                        value = value[0]
                    time_str = str(time.task) +":"+ str(time.block) +":"+ str(time.trial) +":"+ str(time.time_step)
                    data_str = attrib_name.ljust(variable_width, kwSpacer)
                    if not args or kwTime in args:
                        data_str = data_str + " " + time_str.ljust(time_width)
                    if not args or kwContext in args:
                        data_str = data_str + context.ljust(context_width, kwSpacer)
                    if not args or kwValue in args:
                        # data_str = data_str + " " + str(value).rjust(value_width) # <- WORKS
                        # data_str = data_str + " " + "{:10.5}".format(str(value).strip("[]"))  # <- WORKS
                        data_str = data_str + "{:2.5}".format(str(value).strip("[]")).rjust(value_width) # <- WORKS
                        # data_str = data_str + "{:10.5}".format(str(value).strip("[]")) # <- WORKS

        # {time:{width}}: {part[0]:>3}{part[1]:1}{part[2]:<3} {unit:3}".format(
        #     jid=jid, width=width, part=str(mem).partition('.'), unit=unit))

                    print(data_str)
                if len(datum) > 1:
                    print("\n")

    @tc.typecheck
    def csv(self, entries=None, owner_name:bool=False, quotes:tc.optional(tc.any(bool, str))="\'"):
        """Returns a csv formatted string with headers and values for the specified entries.

        The first record (row) begins with "Entry" and is followed by the header for each field (column).
        Subsequent records begin with the record number, and are followed by the value for each entry.

        Arguments
        ---------

        entries : string, Component
            specifies the entries to be included;  they must be `loggable_items <Log.loggable_items>` of the Log.

        owner_name : bool : default False
            specified whether or not to include the Log's `owner <Log.owner>` in the header of each field;
            if it is True, the format of the header for each field is "<Owner name>[<entry name>]";
            otherwise, it is "<entry name>".

        quotes : bool, str : default '
            specifies whether or not to use quotes around values (e.g., arrays);
            if not specified or True, single quotes are used;
            if `False` or `None`, no quotes are used;
            if specified with a string, that is used.

        Returns:
            csv formatted string
        """

        # If Log.ALL_LOG_ENTRIES, set entries to all entries in self.entries
        if entries is ALL_ENTRIES or entries is None:
            entries = self.entries.keys()

        # If entries is a single entry, put in list for processing below
        if isinstance(entries, str):
            entries = [entries]

        # Validate entries
        for entry in entries:
            if entry not in self.loggable_items:
                raise LogError("{0} is not a loggable attribute of {1}".format(repr(entry), self.owner.name))
            if entry not in self.entries:
                raise LogError("{} is not currently being logged by {} (try using log_items)".
                               format(repr(entry), self.owner.name))

        max_len = max([len(self.entries[e]) for e in entries])

        # Currently only supports entries of the same length
        if not all(len(self.entries[e])==len(self.entries[entries[0]])for e in entries):
            raise LogError("CSV output currently only supported for log entries of equal length")

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
        csv = "\'Entry', {}\n".format(", ".join(repr("{}{}{}{}".format(owner_name_str, lb, entry, rb))
                                                     for entry in entries))
        # Records
        for i in range(max_len):
            csv += "{}, {}\n".format(i, ", ".
                                     join(str(self.entries[entry][i][2]) for entry in entries).
                                     replace("[",quotes)).replace("]",quotes)
        return(csv)


    # @property
    # def log_items(self):
    #     return self.owner.log_items
    #
    # @log_item.setter
    # def log_items(selfs, items):
    #     pass
    #

    # FIX: MOVE TO Log WITH CALL TO self._log_items FOR HANDLING OF RCLASS-SPECIFIC ATTRIBUTES (STATES AND PROJECTIONS)
    @property
    def logged_items(self):
        # Use owner's implementation if it has one
        try:
            return self.owner.logged_items
        except AttributeError:
            return self._logged_items

    @property
    def _logged_items(self):
        log_level = 'LogLevel.'
        # Return LogLevel for items in log.entries
        logged_items = {key: value for (key, value) in
                        [(l, log_level+self.owner.logPref.name)
                         for l in self.entries.keys()]}
        return logged_items

    def log_items(self, items, log_level=LogLevel.EXECUTION, param_sets=None):
        # User owner's implementation if it has one
        return self._log_items(items, log_level=log_level, param_sets=param_sets)

    def _log_items(self, items, log_level=LogLevel.EXECUTION, param_sets=None):
        """List of items to log

        items can be a string, Component, 2-item tuple, or a list containing any combination of these.
        Strings and Components must refer to loggable items for the Component,
            (by default, these are the user_params for the Component, and are listed in `loggable_items`):
            - the string or Component.name must be in the loggable_items list.
            - they are assigned log_level (default: `LogLevel` of `EXECUTION`).
        Tuples must have 2 items:
            - 1st item must be a string or Component that refers to a loggable item (see above);
            - 2nd item must be a `LogLevel` specification for 1st item.

        """
        from psyneulink.components.component import Component
        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection, MATRIX
        from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
        from psyneulink.globals.keywords import ALL

        def assign_log_level(item, level, param_set):

            # if not item in self.loggable_items:
            #     raise LogError("\'{0}\' is not a loggable item for {1} (try using \'{1}.log.add_entries()\')".
            #                    format(item, self.owner.name))

            for params in param_set:
                try:
                    params[item].logPref=PreferenceEntry(level, PreferenceLevel.INSTANCE)
                    return
                except:
                    # Go through any list or dict items in params looking for the item
                    from collections import UserDict, UserList
                    for iterable_item in [entry for entry in params
                                          if isinstance(entry, (list, UserList, dict, UserDict))]:
                        if item in iterable_item:
                            try:
                                iterable_item[item].logPref=PreferenceEntry(level, PreferenceLevel.INSTANCE)
                                # self.add_entries(iterable_item[item])
                                return
                            except KeyError:
                                raise LogError("{} is not a loggable parameter of {}".format(item, self.owner.name))

        if items is ALL:
            self.logPref = PreferenceEntry(log_level, PreferenceLevel.INSTANCE)
            return

        param_sets = param_sets or [self.owner.user_params]

        if not isinstance(items, list):
            items = [items]

        # Validate that item is loggable
        for item in items:
            if not item in self.loggable_items:
                raise LogError("{} is not a loggable attribute of {}".format(repr(item), self.owner.name))


        for item in items:
            if isinstance(item, (str, Component)):
                # self.add_entries(item)
                if isinstance(item, Component):
                    item = item.name
                assign_log_level(item, log_level, param_sets)
            else:
                # self.add_entries(item[0])
                assign_log_level(item[0], item[1], param_sets)


    def save_log(self):
        print("Saved")
