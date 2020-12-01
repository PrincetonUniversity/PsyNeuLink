# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **********************************************  PreferenceSet **********************************************************
#
import abc
import inspect

from collections import namedtuple
from enum import Enum, IntEnum

from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import DEFAULT_PREFERENCE_SET_OWNER, PREFERENCE_SET_NAME
from psyneulink.core.globals.utilities import iscompatible, kwCompatibilityType

__all__ = [
    'PreferenceEntry', 'PreferenceLevel', 'PreferenceSet', 'PreferenceSetError', 'PreferenceSetRegistry',
    'PreferenceSetVerbosity'
]

PreferenceSetRegistry = {}

PreferenceSetVerbosity = False

PreferenceEntry = namedtuple('PreferenceEntry', 'setting, level')


class PreferenceLevel(IntEnum):
    NONE        = 0
    INSTANCE    = 1
    SUBTYPE     = 2
    TYPE        = 3
    CATEGORY    = 4
    COMPOSITION = 5


class PreferenceSetError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


@abc.abstractmethod
class PreferenceSet(object):
    """Abstract class for PreferenceSets that stores preferences and provides access to level-specific settings

    Description:
        Each PreferenceSet object stores a set of preferences in its corresponding attributes
        Every class in the Component hierarchy is assigned a PreferenceLevel:
            - System:  reserved for the Component class
            - Category: primary function subclasses (e.g., Process, Mechanism, Port, Projection, Function)
            - Type: Category subclasses (e.g., MappingProjection and ControlProjection subclasses of Projection, Function subclasses)
            - Instance: an instance of an object of any class
        Each class level in a hierarchy should be assigned a PreferenceSet object as a class attribute,
            that specifies default settings at that class-level for objects in that class and its subclasses
        Each attribute of a PreferenceSet object is a PreferenceEntry (setting, level) tuple for a preference:
            - setting (value):
                specifies the setting of the preference for the class or object to which the PreferenceSet belongs
            - level (PreferenceLevel):
                there are four levels used to specify default settings at four levels of the class hierarchy,
                    SYSTEM, CATEGORY, TYPE, and INSTANCE
                specifying a given level in PreferenceEntry will cause the value assigned at that level
                    to be returned when a request is made for the value of the setting for that preference
        PreferenceSets are instantiated using a specification dict;  for each entry:
            the key must be a keyPath for a preference attribute (kpReportOutput, kpLog, kpVerbose, kpParamValidation)
            the value must be either:
                a PreferenceEntry, or
                a value that is valid for the setting of the corresponding attribute, or
                a PreferenceLevel
        PreferenceSet attributes MUST have "_pref" as a substring in their attribute name,
            as this is used by the PreferenceSet.show() method (and possibly others in the future)
            to identify PreferenceSet preference attributes
        Any preference attributes defined by a subclass (in its defaultPreferencesDict - see below), but not specified
            in the dict of the prefs arg, will be assigned a default PreferenceEntry from defaultPreferencesDict dict
        PreferenceSet.logPref settings must be assigned a value from a LogEntry class:
            - Globals.Log defines a LogEntry class that can be used
            - classes can define their own LogEntry class;  however:
                every definition of LogEntry must include all of the attributes of Globals.Log.LogEntry
                those attributes must have a value identical to the one in Globals.Log.LogEntry,
                    with the exception of ALL and DEFAULTS (which can vary for different versions of LogEntry)
                assignments from a LogEntry class other than the one in Globals.Log
                    MUST be from one declared in the same module as the object that owns the PreferenceSet
        SUBCLASSES of PreferenceSet:
        Every subclass of PreferenceSet must define a class attribute called defaultPreferencesDict:
            - this must be a dict, with an entry for each preference attribute used by subclass; each entry must have:
                a key that is the name of the instance variable for the corresponding preference attribute
                a value that is a PreferenceEntry specifying the default setting and level for that preference
            - this will be used if no prefs arg is provided to PreferenceSet.__init__, or if any preferences are missing
        Every subclass of PreferenceSet must define a class attribute called baseClass:
            - this must be a class, that is at the base of the hierarchy for which the PreferenceSet will be used
            - this is used to validate PreferenceSet assignments to other classes and objects in the class hierarchy
              and when searching the hierarchy in get_pref_setting_for_level

        [TBI: Deprecated PreferenceSet subclasses; not useful, too complicated;  only implemented MechanismPreferenceSet
            Subclasses of PreferenceSet can be created to define preferences specific to subclasses of Function
                - subclasses of PreferenceSet inherit all the preference attributes above it in the hierarchy
                - however, the level assigned in a PreferenceEntry cannot exceed the level in the PreferenceSet hiearchy
                    at which the preference is defined;  this is checked and, if the level being assigned is too high,
                    a warning message is generated (irrespective of any verbose preferences),
                    and the highest level allowable (i.e., the one at which the preference is defined) is used instead]
            Preference attributes added to a subclass of PreferenceSet MUST have "_pref" as a substring in their name,
                as this is used by the PreferenceSet.show() method (and possibly others in the future)
                to identify PreferenceSet preference attributes

    Initialization arguments:
        - owner (Function object):
            used on instantiation to determine the level of the default preferences;
            assigned dynamically (by the baseClass) on access to a preference;
                this is because a PreferenceSet can be assigned to multiple objects,
                so current owner should be used to determine context-appropriate PreferenceLevel for returning setting
        - level (PreferenceLevel): level for which to report setting (default: PreferenceLevel.SYSTEM)
        - prefs (dict): see above for keys and entries
        - name (str): name of PreferenceSet (default: <subclass.__name__>
        - context (class): must be a subclass of PreferenceSet;  otherwise, an exception is raised
        Note:
        * instantiation always returns a complete PreferenceSet (as defined by the subclass)

    Class attributes:
        - prefsList (list): list of preference attribute names in PreferenceSet (== PreferenceSet.__dict__.keys();

    Class methods:
        - add_preference(pref_attrib=<str>, pref_spec=<value>, default_pref=<PreferenceEntry>):
            create preference attribute for PreferenceSet and assign pref_spec as its value
        - get_pref_setting_for_level(pref_ivar_name=<str>, level=<PreferenceLevel>):
            return setting for specified preference at level specified
            if level is omitted, return setting for level specified in instance's PreferenceEntry
        - show():
            generate table showing all preference attributes for the PreferenceSet, their base and current and values,
                and their PreferenceLevel assignment
        - set_preference(candidate_info=<PreferenceEntry, setting or PreferenceLevel>,
                          default_entry=<PreferenceEntry>,
                          setting_types=<[types]>
                          pref_ivar_name=<str>):
            set specified value of PreferenceEntry attribute (entire PreferenceEntry. setting or level)
        - validate_setting(candidate_setting=<value>, reference_setting=<value>, pref_ivar_name=<str>)
            validate value of setting for pref_ivar_name attribute of PreferenceSet
        - validate_log(candidate_log_item=<LogEntry.<value>>, pref_set=<PreferenceSet>)
            validate that value of kplogPref (_log_pref_) is from conforming and appropriate LogEntry class

    Instance attributes:
        None

    Instance methods:
        None
    """

    def __init__(self,
                 owner,
                 level=PreferenceLevel.COMPOSITION,
                 prefs=None,
                 name=None,
                 context=None
                 ):
        """Instantiate PreferenceSet from subclass for object and/or class

        --------------------------  CONDITIONS AND ACTIONS:----------------------------------------------------

        Condition    Type    classPref's exist   prefs arg    Action

            1        class		    YES		        NO	     assign self to classPreferences
            2        class	      	YES             dict	 override classPreferences with any prefs
            3        class		    NO		        NO       instantiate ClassPreferences from <subclass>.defaultPrefs
            4        class		    NO		        dict	 instantiate ClassPrefs from defaultPrefs, override w/ prefs

            5        object		    YES		        NO	     use classPreferences
            6        object		    YES		        dict	 use classPreferences as base; override with any prefs
            6.5      object		    YES	(dict)	    NO       instantiate ClassPrefs from defaultPrefs, override w/ dict
            7        object		    NO	            NO   |_| raise exception: these should never occur as instantiation
            8        object		    NO		        dict | |    of object should force instantiation of classPreferences
          # 7        object		    NO	            NO   	 instantiate and use classPreferences from defaultPrefs
          # 8        object		    NO		        dict	 inst. & use classPrefs from defaultPrefs, override w/ prefs
        -------------------------------------------------------------------------------------------------------

        :param owner:
        :param level:
        :param prefs:
        :param name:
        :param context:
        """

        # VALIDATE ATTRIBUTES AND ARGS
        # Make sure subclass implements a baseClass class attribute and that owner is a subclass or instance of it
        try:
            base_class_NOT_A_CLASS = not inspect.isclass(self.baseClass)
        except:
            raise PreferenceSetError("{0} must implement baseClass as a class attribute".
                                     format(self.__class__.__name__))
        else:
            if base_class_NOT_A_CLASS:
                raise PreferenceSetError("{0}.baseClass ({0}) must be a class".
                                         format(self.__class__.__name__, self.baseClass))
            else:
                # owner of PreferenceSet must be a subclass or instance of a subclass in baseClass
                if not (inspect.isclass(owner) or isinstance(owner, self.baseClass)) or owner is None:
                    raise PreferenceSetError("owner argument must be included in call to {1}() "
                                             "and must be an object in the {2} class hierarchy".
                                         format(owner, self.__class__.__name__, self.baseClass.__name__))
                self.owner = owner

        # Make sure subclass implements a defaultPreferencesDict class attribute
        try:
            default_prefs_dict = self.defaultPreferencesDict
        except AttributeError:
            raise PreferenceSetError("{0} must implement defaultPreferencesDict dict as a class attribute".
                                     format(self.__class__.__name__))

        # prefs must be a specification dict or None
        # FIX: replace with typecheck
        if not (isinstance(prefs, dict) or prefs is None):
            raise PreferenceSetError("Preferences ({0}) specified for {1} must a PreferenceSet or"
                                     " specification dict of preferences".format(prefs, owner.name))

        # ASSIGN NAME
        # ****** FIX: 9/10/16: MOVE TO REGISTRY **********

        # FIX: MAKE SURE DEFAULT NAMING SCHEME WORKS WITH CLASSES - 5/30/16
        # FIX: INTEGRATE WITH NAME FROM PREFERENCE_SET_NAME ENTRY IN DICT BELOW - 5/30/16

        if not name:
            # Assign name of preference set class as base of name
            name = self.__class__.__name__
            # If it belongs to a class, append name of owner's class to name
            if inspect.isclass(owner):
                name = name + 'DefaultsFor' + owner.__name__
            # Otherwise, it belongs to an object, so append name of the owner object's class to name
            else:
                name = name + 'Defaultsfor' + owner.__class__.__name__

        # REGISTER
        # FIX: MAKE SURE THIS MAKES SENSE

        from psyneulink.core.globals.registry import register_category
        register_category(entry=self,
                          base_class=PreferenceSet,
                          name=name,
                          registry=PreferenceSetRegistry,
                          )

        # ASSIGN PREFS
        condition = 0

        # Get class preferences (if any) from owner's class
        try:
            class_prefs = owner.classPreferences
        except AttributeError:
            class_prefs = None
        else:
            # Class preferences must be a PreferenceSet or a dict
            if not (isinstance(class_prefs, (PreferenceSet, dict))):
                raise PreferenceSetError("Class preferences for {0} ({1}) must be a PreferenceSet "
                                         "or specification dict for one".format(owner.__name__, class_prefs))

        # Owner is a class
        if inspect.isclass(owner):

            # Class PreferenceSet already exists
            if isinstance(class_prefs, PreferenceSet):
                self = class_prefs

                if prefs is None:                                                    # Condition 1
                    # No prefs arg, so just assign self to class_prefs
                    condition = 1

                else:                                                                          # Condition 2
                    # Get pref from prefs arg or, if not there, from class_prefs
#FIX:  SHOULD TEST FOR *prefsList* ABOVE AND GENERATE IF IT IS NOT THERE, THEN REMOVE TWO SETS OF CODE BELOW THAT DO IT
                    try:
                        self.prefsList
                    except AttributeError:
                        # Generate prefsList for PreferenceSet (since it was not generated when it was created
                        self.prefsList = [i for i in list(class_prefs.__dict__.keys()) if '_pref' in i]

                    for pref_key in self.prefsList:
                        class_pref_entry = getattr(class_prefs, pref_key)
                        try:
                            # Get pref from prefs arg; use class_prefs as validation/default
                            self.set_preference(candidate_info=prefs[pref_key],
                                                default_entry=class_pref_entry,
                                                pref_ivar_name=pref_key)
                        except (KeyError):
                            # No need for default_entry here, as class prefs exist and were validated when instantiated
                            self.set_preference(candidate_info=class_pref_entry,
                                                default_entry=default_prefs_dict[pref_key],  # IMPLEMENTATION: FILLER
                                                pref_ivar_name=pref_key)

                        condition = 2

            # Class preferences either don't exist, or are coming from a specification dict in the class declaration,
            # so use <subclass>.defaultPreferencesDict to instantiate full set, and create PrefsList
            else :
                self.prefsList = []
                if isinstance(class_prefs, dict):
                    # class_prefs are a specification dict from the class declaration
                    # - if prefs are present and are the default set (e.g., in instantiation call from __init__.py),
                    #    replace with class_prefs dict since default_prefs_dict will be used below as base set anyhow)
                    # - if prefs are not the default set, merge with class_prefs dict, giving precedence to prefs
                    #    since they were in a script from the user
                    if prefs is default_prefs_dict:
                        prefs = class_prefs
                    else:
                        prefs.update(class_prefs)
                for pref_key in default_prefs_dict:
                    if pref_key == PREFERENCE_SET_NAME:
                        continue
                    try:
                        # Get pref from prefs arg;                                             # Condition 4
                        #    use default_prefs_dict for validation and backup
                        self.set_preference(candidate_info=prefs[pref_key],
                                            default_entry=default_prefs_dict[pref_key],
                                            pref_ivar_name=pref_key)

                        condition = 4

                    except (KeyError, TypeError) as error:
                        if isinstance(error, KeyError):
                            # Pref not in prefs arg, so get pref from <subclass>.defaultPreferences
                            condition = 4

                        if isinstance(error, TypeError):
                            # No prefs, so get pref from  <subclass>.defaultPreferences        # Condition 3
                            condition = 3
                        # No need for default_entry here, as if the default_prefs_dict is no good there is no hope
                        self.set_preference(candidate_info=default_prefs_dict[pref_key],
                                            default_entry=default_prefs_dict[pref_key],  # IMPLEMENTATION: FILLER
                                            pref_ivar_name=pref_key,)
                        condition = 3

                    # Poplulate prefsList
                    self.prefsList.append(pref_key)

            owner.classPreferences = self

        # Owner is an object
        else:
            if class_prefs:
# # MODIFIED 6/28/16 ADDED:
# # FIX:  class_prefs IS A DICT, SO GETTING AN ERROR BELOW TREATING AS PreferenceSet AND TESTING FOR class_prefs.prefsList
# #       (SEE EXPLANATION ABOVE CONDITION), ??SO ADD THIS:
# # IMPLEMENT Condition 6.5 (see Table above)
#                 if isinstance(class_prefs, dict):
#                     # class_prefs are a specification dict from the class declaration
#                     # - if prefs are present and are the default set (e.g., in instantiation call from __init__.py),
#                     #    replace with class_prefs dict since default_prefs_dict will be used below as base set anyhow)
#                     # - if prefs are not the default set, merge with class_prefs dict, giving precedence to prefs
#                     #    since they were in a script from the user
#                     if prefs is default_prefs_dict:
#                         prefs = class_prefs
#                     else:
#                         prefs.update(class_prefs)
# # MODIFIED 6/28/16 END ADDITION
                try:
                    class_prefs.prefsList
                except AttributeError:
                    # Generate class_prefs.prefsList (since it was not generated when class_prefs was created
                    class_prefs.prefsList = [i for i in list(class_prefs.__dict__.keys()) if '_pref' in i]
                for pref_key in class_prefs.prefsList:
                    try:
                        # Get pref from prefs arg                                              # Condition 6
                        self.set_preference(candidate_info=prefs[pref_key],
                                            default_entry=getattr(class_prefs, pref_key),
                                            pref_ivar_name=pref_key)
                        condition = 6
                    except (KeyError, TypeError) as error:
                        if isinstance(error, KeyError):
                            # Pref not in prefs arg, so get pref from class prefs
                            condition = 6
                        if isinstance(error, TypeError):
                            # No prefs, so get pref from  class_prefs                          # Condition 5
                            condition = 5
                        self.set_preference(candidate_info=getattr(class_prefs, pref_key),
                                            default_entry=default_prefs_dict[pref_key],  # IMPLEMENTATION: FILLER
                                            pref_ivar_name=pref_key)
                owner.prefs = self
            else:                                                                              # Conditions 7 and 8
                raise PreferenceSetError("Program Error: attempt to create a PreferenceSet for object ({0}) "
                                         "in a class ({1}) for which classPreferences have not been instantiated".
                                         format(owner.name, owner.__class__.__name__))

        if PreferenceSetVerbosity:
            print("Preference assignment condition {0}".format(condition))

# FIX: ARE THESE NEEDED?? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, setting):
        if isinstance(setting, PreferenceLevel):
# FIX: CHECK IF SETTING IS LEGAL FOR THIS LEVEL:
#   TRY CALLING  get_pref_setting_for_level and evaluating return value
#   CHECK FOR ATTRIBUTE ERROR FOR <self>.__class__.classPreferences AT LEVEL SPECIFIED
#   ?? NEED TO TRAVERSE HIERACHY?  ANY WAY TO GO STRAIGHT THERE; OR INCLUDE IN PreferenceLevel STRUCTURE?
#             self.get_pref_setting_for_level(self.name, setting)

            self._level = setting
        else:
            print("Attempt to assign PreferenceSet {0} of {1} an invalid PreferenceLevel ({2});"
                  " level will remain set to {3}".format(self.name, self.owner.name, setting,
                                                         self._level.__class__.__name__ + '.' + self._level.name))

# FIX: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#     def add_preference(self, pref_attrib_name, pref_spec, default_entry=NotImplemented):
#         """Instantiate attribute with pref_attrib_name using pref_spec from a preference specification dict
#
#         If pref_spec is for just a setting or just a level, get the other value from default_pref PreferenceEntry
#
#         :param pref_attrib_name: (str)
#         :param pref_spec: (PreferenceEntry, value, or PreferenceLevel
#         :param default_pref: (PreferenceEntry)
#         :return:
#         """
# # FIX: VALIDATE COMPONENTS OF PreferenceEntry HERE (?? ADD METHOD setPreferenceEntry??)
#         # If pref_spec is a PreferenceEntry, use its values to create entry
#         if isinstance(pref_spec, PreferenceEntry):
#             # MODIFIED 6/6/16
#             # setattr(self, pref_attrib_name, pref_spec)
#             self.set_preference(candidate_info=pref_spec, default_entry=default_entry, pref_ivar_name=pref_attrib_name)
#             # setting_OK = self.validate_setting(candidate_info.setting, current_setting, pref_ivar_name)
#             # level_OK = isinstance(candidate_info.level, PreferenceLevel)
#
#         # Otherwise, check if pref_spec is a valid level or setting value
#         else:
#             if isinstance(pref_spec, PreferenceLevel):
#                 # If pref_spec is for a PreferenceLevel, get setting from default_pref
#                 if default_entry is NotImplemented:
#                     raise PreferenceSetError("Program Error: default_pref missing but needed to specify setting for "
#                                              "{0} of {1} in {2}".format(pref_attrib_name,self.name,self.owner.name))
#                 # MODIFIED 6/6/16
#                 # setattr(self, pref_attrib_name, PreferenceEntry(default_pref.setting, pref_spec))
#                 self.set_preference(candidate_info=PreferenceEntry(default_entry.setting, pref_spec),
#                                     default_entry=default_entry,
#                                     pref_ivar_name=pref_attrib_name)
#
#             # If pref_spec is a valid setting (compatible with default_pref value) use it, & get level from default_pref
#             elif default_entry is NotImplemented:
#                 raise PreferenceSetError("Program Error: default_pref missing but needed to specify level for "
#                                          "{0} of {1} in {2}".format(pref_attrib_name,self.name,self.owner.name))
#             # elif iscompatible(pref_spec, default_entry.setting):
#             #     # MODIFIED 6/6/16
#                 # setattr(self, pref_attrib_name, PreferenceEntry(pref_spec, default_pref.level))
#                 self.set_preference(candidate_info=PreferenceEntry(pref_spec, default_entry.level),
#                                     default_entry=default_entry,
#                                     pref_ivar_name=pref_attrib_name)
#             else:
#                 raise PreferenceSetError("{0} is not a valid preference specification for {1};"
#                                          " must be a {2} or PreferenceLevel".
#                                          format(pref_spec, pref_attrib_name, type(pref_spec).__name__))

    def set_preference(self, candidate_info, pref_ivar_name, default_entry=None):
        """Validate and assign PreferenceSet, setting, or level to a PreferenceSet preference attribute

        Validate candidate candidate_info and, if OK, assign to pref_ivar_name attribute; candidate_info can be a:
            - PreferenceEntry; validate both the setting and PreferenceLevel items
            - setting; validate against current value retrieved from pref_ivar_name
            - PreferenceLevel; type-check
        If default_entry is provided, it must be a PreferenceEntry;
            use it to validate candidate_info, and fill any invalid or missing values;
        If default_entry is not provided, there must be a current entry for PreferenceSet;
            otherwise an exception will be raised

        Return a PreferenceEntry containing the validated information;
            - this is NOT necessarily the full current PreferenceEntry for the pref;
            - it contains only the information passed in candidate_info (e.g., setting or PreferenceLevel could be None)
            - it is provided simply as parse of candidate_info for the convenience of the caller (e.g., logPref.setter)

        Note: pref_ivar_name MUST include the underscore prefix (e.g., _log_pref), else a recursive loop will ensue
                (wherein the setter calls set_preference, which calls the setter for the pref, etc).

        :param candidate_info: (PreferenceEntry, value, or PreferenceLevel)
        :param pref_ivar_name: (str)
        :param reference_value: ([types])
        :return assignment: (PreferenceEntry)
        """

        # Setup local variables

        # Get owner's name
        if inspect.isclass(self.owner):
            owner_name = self.owner.__name__
        else:
            owner_name = self.owner.name

        # Set all to True, so that if only one is being checked, the other does not interfere with final test
        level_OK = True
        setting_OK = True
        entry_OK = True

        # Get/validate defaults
        # No default specified
        if default_entry is None:
            try:
                # Get current info to use as reference and filler in new PreferenceEntry
                #    in case candidate_info is just a level or setting
                current_entry = getattr(self, pref_ivar_name)
            except AttributeError:
                # # Deal with error if it occurs below, on attempt to inappropriately use default_setting or default_level
                # pass
                raise PreferenceSetError("No existing PreferenceEntry and no default specified for {0}".
                                         format(self.name))
            else:
                default_setting, default_level = current_entry
        # Validate default entry
        else:
            try:
                default_setting, default_level = default_entry
            except TypeError:
                raise PreferenceSetError("Default specification ({0}) for {1} must be a PreferenceEntry".
                                         format(default_entry, self.name))

        # candidate_info is a PreferenceEntry
        if (isinstance(candidate_info, PreferenceEntry)
                or (isinstance(candidate_info, tuple) and len(candidate_info)==2)):
            # elif len(candidate_info) != 2:
            #     raise PreferenceSetError("Preference specification tuple for {} ({}) must have only two entries "
            #                              "(setting and level)".format(owner_name, candidate_info))
            if not isinstance(candidate_info, PreferenceEntry):
                candidate_info = PreferenceEntry(candidate_info[0], candidate_info[1])
            setting_OK = self.validate_setting(candidate_info.setting, default_setting, pref_ivar_name)
            level_OK = isinstance(candidate_info.level, PreferenceLevel)
            if level_OK and setting_OK:
                setattr(self, pref_ivar_name, candidate_info)
                return_val = candidate_info
            else:
                entry_OK = False

        # candidate_info is a PreferenceLevel
        elif isinstance(candidate_info, PreferenceLevel):
            setattr(self, pref_ivar_name, PreferenceEntry(default_setting, candidate_info))
            return_val = PreferenceEntry(setting=None, level=candidate_info)

        # candidate_info is a presumed setting
        else:
            setting_OK = self.validate_setting(candidate_info, default_setting, pref_ivar_name)
            if setting_OK:
                setattr(self, pref_ivar_name, PreferenceEntry(candidate_info, default_level))
                return_val = PreferenceEntry(setting=candidate_info, level=None)

        # All is OK, so return
        if level_OK and setting_OK:
            return return_val

        # Something's amiss, so raise exception
        if not entry_OK:

            if not level_OK and not setting_OK:
                raise PreferenceSetError("'{0}' and '{1}' are not valid values for setting and PreferenceLevel of"
                                         " PreferenceEntry for {2} in {3} of {4}".
                                         format(candidate_info.setting, candidate_info.level,
                                                pref_ivar_name, self.name, owner_name))
            elif not (level_OK):
                raise PreferenceSetError("'{0}' is not a valid value for PreferenceLevel of PreferenceEntry "
                                         "for {1} in {2} of {3}".
                                         format(candidate_info.level, pref_ivar_name, self.name, owner_name))
            else:
                raise PreferenceSetError("'{0}' is not valid value for setting of PreferenceEntry for {1} in {2}".
                                         format(candidate_info.setting, pref_ivar_name, self.name))

        elif not (setting_OK):
            raise PreferenceSetError("'{0}' is not a valid value for setting of {1} in {2}".
                                     format(str(candidate_info), pref_ivar_name, self.name))
        else:
            raise PreferenceSetError("PROGRAM ERROR")

    def validate_setting(self, candidate_setting, reference_setting, pref_ivar_name):
        """Validate candidate_setting by checking against reference_setting and, if a log_entry, its type

        :param candidate_setting:
        :param reference_setting:
        :return:
        """
        # from Globals.Preferences.BasePreferenceSet import LOG_PREF
        # if pref_ivar_name is LOG_PREF:
        #     self.validate_log(candidate_setting, self)

        setting_OK = iscompatible(candidate_setting, reference_setting, **{kwCompatibilityType:Enum})
        # setting_OK = iscompatible(candidate_setting, reference_setting)

        # if not setting_OK and (isinstance(candidate_setting, Enum) or isinstance(reference_setting, Enum)):
        #     if isinstance(candidate_setting, Enum):
        #         raise PreferenceSetError("'{0}' is not a valid value for setting of {1} in {2} of {3}".
        #                                  format(candidate_setting, pref_ivar_name, self.name, owner_name))
        #     else if

        return setting_OK

    def validate_log(self, candidate_log_item, pref_set):
        """Validate value of log setting and that it is from the LogEntry class associated with the owner of the preference

        :param candidate_log_item:
        :param pref_set:
        :return:
        """
        candidate_log_class = type(candidate_log_item)

        from psyneulink.core.globals.log import LogEntry

        # Candidate_log_item must be from a LogEntry declared in the same module as the owner of the preference
        #    unless it is for the DefaultPeferenceSetOwner (which is used by many classes),
        # Note:
        # * this prevents use of LogEntry attributes not recognized by, or having different values from
        #     the LogEntry class in the owner object's module
        if (not pref_set.owner.name == DEFAULT_PREFERENCE_SET_OWNER and
                candidate_log_class.__module__ is not pref_set.owner.__module__):
            raise PreferenceSetError("Attempt to assign logPref setting for {0} using value ({1}) from LogEntry"
                                     " in {2} which is different than the one defined in Globals.Log"
                                     " or the module ({3}) in which the class of {0} ({4}) was declared".
                                     format(pref_set.owner.name,
                                            candidate_log_item.name,
                                            candidate_log_item.__module__,
                                            pref_set.owner.__module__,
                                            pref_set.owner.__class__.__name__))

        # Validate that every attribute of Globals.Log.LogEntry is present and has the same value in
        #    the LogEntry class used by the setting being assigned
        for log_entry_attribute in LogEntry.__dict__['_member_names_']:

            candidate_log_entry_value = candidate_log_class.__dict__['_member_map_'][log_entry_attribute]
            global_log_entry_value = LogEntry.__dict__['_member_map_'][log_entry_attribute]

            # Validate that attribute from Globals.Log.LogEntry is present in candidate's LogEntry class
            try:
                OK = (candidate_log_entry_value == global_log_entry_value)
            # Attribute from Globals.Log.LogEntry not found in candidate's LogEntry
            except KeyError:
                raise PreferenceSetError("{0} class in {1} must have an attribute {2} identical to the one in {3}".
                                         format(candidate_log_class.__name__,
                                                candidate_log_class.__module__,
                                                log_entry_attribute,
                                                LogEntry.__module__ + "." + LogEntry.__name__,))
            else:
                if OK:
                    continue
                # Skip validation of value of these, as they may differ for different LogEntry classes
                elif (log_entry_attribute == 'ALL' or
                      log_entry_attribute == 'DEFAULTS'):
                    continue
                else:
                    # Value of attribute from candidate's LogEntry does not match the value in Globals.Log.LogEntry
                    raise PreferenceSetError("Value of {0} in {1} ({2}) must have same value as entry in {3} ({4})".
                                             format(log_entry_attribute,
                                                    candidate_log_class.__name__,
                                                    candidate_log_entry_value,
                                                    LogEntry.__module__ + "." + LogEntry.__name__,
                                                    global_log_entry_value))

    def get_pref_setting_for_level(self, pref_ivar_name, requested_level=None):
        """Return the setting of a preference for a specified preference level, and any error messages generated

        Arguments:
        - pref_ivar_name (str): name of ivar for preference attribute for which to return the setting;
        - requested_level (PreferenceLevel): preference level for which the setting should be returned

        Returns:
        - PreferenceEntry.setting, str:
        """
        pref_entry = getattr(self, pref_ivar_name)

        if requested_level is None:
            requested_level = pref_entry.level

        # Preference is owned by an object
        if isinstance(self.owner, self.baseClass):

            # If requested level is INSTANCE, return pref setting for object's PreferenceSet:
            if requested_level is PreferenceLevel.INSTANCE:
                return pref_entry.setting, None

            # If requested level is higher than current one, call class at next level
            elif requested_level > self.owner.__class__.classPreferenceLevel:
                # IMPLEMENTATION NOTE: REMOVE HACK BELOW, ONCE ALL CLASSES ARE ASSIGNED classPreferences ON INIT
                next_level = self.owner.__class__.__bases__[0]
                # If classPreferences for level have not yet been assigned as PreferenceSet, assign them
                if (not hasattr(next_level, 'classPreferences') or
                        not isinstance(next_level.classPreferences, PreferenceSet)):
                    from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet
                    next_level.classPreferences = BasePreferenceSet(owner=next_level,
                                                                    prefs=next_level.classPreferences,
                                                                    level=next_level.classPreferenceLevel)
                return_val = next_level.classPreferences.get_pref_setting_for_level(pref_ivar_name, requested_level)
                return return_val[0],return_val[1]
            # Otherwise, return value for current level
            else:
                # This returns setting from the current level:
                try:
                    return getattr(self.owner.__class__.classPreferences, pref_ivar_name).setting, None
                except AttributeError:
                    # # IMPLEMENTATION NOTE:
                    # #  DEPRECATED THIS, AS CONDITION SHOULD ONLY OCCUR IF
                    # #  A <class>.classPreferences DECLARATION IS MISSING FROM __init__.py
                    # This returns setting from the next level up:
                    # next_level = getattr(self.owner.__class__.__bases__[0].classPreferences, pref_ivar_name).level
                    # next_level_val = getattr(self.owner.__class__.__bases__[0].classPreferences, pref_ivar_name).setting
                    # print ("Warning:  {0} not found at {2} level for {1} class; "
                    #        "replaced with \'{3}\' from {4} class at {5} level".
                    #        format(pref_ivar_name,
                    #               self.owner.__class__.__name__,
                    #               requested_level.name,
                    #               next_level_val,
                    #               self.owner.__class__.__bases__[0].__name__,
                    #               next_level.name))
                    # return next_level_val, None
                    raise PreferenceSetError("Assignment of {}.classPreferences appears to be missing from __init__.py".
                                             format(self.owner.__class__.__name__))

        # Preference is owned by a class
        elif inspect.isclass(self.owner) and issubclass(self.owner, self.baseClass):
            # If requested level is higher than current one:
            if requested_level > self.owner.classPreferenceLevel:
                # Call class at next level

                from psyneulink.core.components.component import Component
                # THis is needed to skip ShellClass, which has no classPreferences, to get to Function (System) level
                if 'ShellClass' in repr(self.owner.__bases__[0]):
                    try:
                        # Store current class pref set (in case class at next level doesn't have the preference)
                        self.previous_level_pref_set = self.owner.classPreferences
                        return_val = Component.classPreferences.get_pref_setting_for_level(pref_ivar_name,
                                                                                    requested_level)
                        return return_val[0], return_val[1]
                    # Pref not found at current level, so use pref from previous level (and report error)
                    except AttributeError:
                        pref_value = getattr(self.previous_level_pref_set, pref_ivar_name).setting
# FIX:  replace level setting??
#                         from Utilities import get_modulationOperation_name
                        err_msg = ("{0} not found at {1}; replaced with value ({2}) from next lower level ({3})".
                                   format(pref_ivar_name,
                                          requested_level.__class__.__name__ + '.' + requested_level.name,
                                          # get_modulationOperation_name(pref_value),
                                          pref_value,
                                          pref_entry.level.__class__.__name__ + '.' + pref_entry.level.name))
                        return pref_value, err_msg
                else:
                    # If classPreferences for level have not yet been assigned as PreferenceSet, assign them
                    next_level = self.owner.__bases__[0]
                    if (not hasattr(next_level, 'classPreferences') or
                            not isinstance(next_level.classPreferences, PreferenceSet)):
                        from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet
                        next_level.classPreferences = BasePreferenceSet(owner=next_level,
                                                                        prefs=next_level.classPreferences,
                                                                        level=next_level.classPreferenceLevel)
                    return_val = self.owner.__bases__[0].classPreferences.get_pref_setting_for_level(pref_ivar_name,
                                                                                               requested_level)
                    return return_val[0], return_val[1]

            # Otherwise, return value for current level
            try:
                return getattr(self, pref_ivar_name).setting, None
            except AttributeError:
                raise PreferenceSetError("PROGRAM ERROR")

        else:
            raise PreferenceSetError("Owner ({0}) of {1} (for which {2} setting is being requested) "
                               "is not a Function object or subclass".
                               format(self.owner.__class__.__name__, self.__class__.__name__, pref_ivar_name))

    def show(self, type=None):
        """Print preferences for PreferenceSet

        :return:
        """

        if self.owner is None:
            raise PreferenceSetError("{0} does not have an owner assigned.  If this PreferenceSet belongs to a class,"
                                     "the class should be given as the owner arg "
                                     "in the call to instantiate the PreferenceSet".format(self.name))

        error_messages = []
        pref_info_table = ""
        # Sort for consistency of reporting
        pref_names_sorted = sorted(self.__dict__.keys())
        for pref_name in pref_names_sorted:
            if '_pref' in pref_name:

                from psyneulink.core.globals.utilities import get_modulationOperation_name

                # GET TABLE INFO
                # Get base_value of pref
                base_value, level = self.__dict__[pref_name]
                # This is needed because value of Modulation is callable (lambda function)
                if inspect.isfunction(base_value):
                    if 'Modulation' in repr(base_value):
                        base_value = get_modulationOperation_name(base_value)
                # Get current_value of pref
                current_value, msg = self.get_pref_setting_for_level(pref_ivar_name=pref_name, requested_level=level)
                if msg:
                    error_messages.append(msg)
                # Get name of Modulation (the value of which is a lambda function)
                if inspect.isfunction(current_value):
                    if 'Modulation' in repr(current_value):
                        current_value = get_modulationOperation_name(current_value)
                # Get name of any enums
                if isinstance(base_value, (Enum, IntEnum)):
                    base_value_str = str(base_value)
                else:
                    base_value_str = repr(base_value)
                if isinstance(current_value, (Enum, IntEnum)):
                    current_value_str = str(current_value)
                else:
                    current_value_str = repr(current_value)
                pref_info_table = pref_info_table +("- {0} {1} {2} {3}\n".format(pref_name[1:].ljust(35,'.'),
                                                                           base_value_str.strip('"\'').ljust(35,'.'),
                                                                           current_value_str.strip('"\'').ljust(35,'.'),
                                                                           level.__class__.__name__ + '.' + level.name))


        # Header info
        print('\nPreferenceSet name: {0}\nPreferenceSet owner: {1}'.format(self.name,
                                                                           self.owner.name.ljust(15)))
        # Error messages (if any):
        if error_messages:
            print()
            for msg in error_messages:
                print(msg)
        # Table:
        print()
        print('{0}{1}{2}{3}'.
              format('Preference:'.ljust(38),
                     'Base Value:'.ljust(36),
                     'Current Value:'.ljust(36),
                     'Level:'.ljust(35)))
        print(pref_info_table)

    # IMPLEMENT:  ADD reference_value ARG THAT IS USED TO VALIDATE SETTING (INSTEAD OF current_setting)
    #             IMPLEMENT USING **{kwCompatibilityType in call to iscompatible)
    #                       BUT WILL NEED TO IMPLEMENT SUPPORT FOR *LIST* OF TYPES FOR kwCompatibilityType


def _assign_prefs(object, prefs, prefs_class:PreferenceSet):

        if isinstance(prefs, PreferenceSet):
            object.prefs = prefs
            # FIX:  CHECK LEVEL HERE??  OR DOES IT NOT MATTER, AS OWNER WILL BE ASSIGNED DYNAMICALLY??
        # Otherwise, if prefs is a specification dict instantiate it, or if it is None assign defaults
        else:
            object.prefs = prefs_class(owner=object, prefs=prefs)
        try:
            # assign log conditions from preferences
            object.parameters.value.log_condition = object.prefs._log_pref.setting
        except AttributeError:
            pass

        try:
            # assign delivery conditions from preferences
            object.parameters.value.delivery_condition = object.prefs._delivery_pref.setting
        except AttributeError:
            pass
