# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************** BasePreferenceSet ***************************************************
#
#

import inspect

from psyneulink.core.globals.keywords import \
    NAME, DEFAULT_PREFERENCE_SET_OWNER, PREF_LEVEL, PREFERENCE_SET_NAME, PREFS, PREFS_OWNER
from psyneulink.core.globals.log import LogCondition
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel, PreferenceSet
from psyneulink.core.globals.utilities import Modulation

__all__ = [
    'CategoryDefaultPreferencesDict',
    'ComponentDefaultPrefDicts', 'BasePreferenceSet', 'BasePreferenceSetPrefs',
    'CompositionDefaultPreferencesDict', 'DELIVERY_PREF',
    'InstanceDefaultPreferencesDict', 'is_pref', 'is_pref_set',
    'CATEGORY_DEFAULT_PREFERENCES', 'INSTANCE_DEFAULT_PREFERENCES', 'SUBTYPE_DEFAULT_PREFERENCES',
    'TYPE_DEFAULT_PREFERENCES', 'LOG_PREF', 'PARAM_VALIDATION_PREF',
    'REPORT_OUTPUT_PREF', 'RUNTIME_PARAM_MODULATION_PREF', 'SubtypeDefaultPreferencesDict',
    'TypeDefaultPreferencesDict', 'VERBOSE_PREF',
]

# Keypaths for preferences:
REPORT_OUTPUT_PREF = '_report_output_pref'
LOG_PREF = '_log_pref'
DELIVERY_PREF = '_delivery_pref'
PARAM_VALIDATION_PREF = '_param_validation_pref'
VERBOSE_PREF = '_verbose_pref'
RUNTIME_PARAM_MODULATION_PREF = '_runtime_param_modulation_pref'

# Keywords for generic level default preference sets
COMPOSITION_DEFAULT_PREFERENCES = 'CompositionDefaultPreferences'
CATEGORY_DEFAULT_PREFERENCES = 'CategoryDefaultPreferences'
TYPE_DEFAULT_PREFERENCES = 'TypeDefaultPreferences'
SUBTYPE_DEFAULT_PREFERENCES = 'SubtypeDefaultPreferences'
INSTANCE_DEFAULT_PREFERENCES = 'InstanceDefaultPreferences'

# Level default preferences dicts:

BasePreferenceSetPrefs = {
    VERBOSE_PREF,
    PARAM_VALIDATION_PREF,
    REPORT_OUTPUT_PREF,
    LOG_PREF,
    DELIVERY_PREF,
    RUNTIME_PARAM_MODULATION_PREF
}

CompositionDefaultPreferencesDict = {
    PREFERENCE_SET_NAME: COMPOSITION_DEFAULT_PREFERENCES,
    VERBOSE_PREF: PreferenceEntry(False, PreferenceLevel.COMPOSITION),
    PARAM_VALIDATION_PREF: PreferenceEntry(True, PreferenceLevel.COMPOSITION),
    REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.COMPOSITION),
    LOG_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
    DELIVERY_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
    RUNTIME_PARAM_MODULATION_PREF: PreferenceEntry(Modulation.MULTIPLY, PreferenceLevel.COMPOSITION)}

CategoryDefaultPreferencesDict = {
    PREFERENCE_SET_NAME: CATEGORY_DEFAULT_PREFERENCES,
    VERBOSE_PREF: PreferenceEntry(False, PreferenceLevel.CATEGORY),
    PARAM_VALIDATION_PREF: PreferenceEntry(True, PreferenceLevel.CATEGORY),
    REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.CATEGORY),
    LOG_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
    DELIVERY_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
    RUNTIME_PARAM_MODULATION_PREF: PreferenceEntry(Modulation.MULTIPLY,PreferenceLevel.CATEGORY)}

TypeDefaultPreferencesDict = {
    PREFERENCE_SET_NAME: TYPE_DEFAULT_PREFERENCES,
    VERBOSE_PREF: PreferenceEntry(False, PreferenceLevel.TYPE),
    PARAM_VALIDATION_PREF: PreferenceEntry(True, PreferenceLevel.TYPE),
    REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.TYPE),
    LOG_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),   # This gives control to Mechanisms
    DELIVERY_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
    RUNTIME_PARAM_MODULATION_PREF: PreferenceEntry(Modulation.ADD,PreferenceLevel.TYPE)}

SubtypeDefaultPreferencesDict = {
    PREFERENCE_SET_NAME: SUBTYPE_DEFAULT_PREFERENCES,
    VERBOSE_PREF: PreferenceEntry(False, PreferenceLevel.SUBTYPE),
    PARAM_VALIDATION_PREF: PreferenceEntry(True, PreferenceLevel.SUBTYPE),
    REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.SUBTYPE),
    LOG_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),   # This gives control to Mechanisms
    DELIVERY_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
    RUNTIME_PARAM_MODULATION_PREF: PreferenceEntry(Modulation.ADD,PreferenceLevel.SUBTYPE)}

InstanceDefaultPreferencesDict = {
    PREFERENCE_SET_NAME: INSTANCE_DEFAULT_PREFERENCES,
    VERBOSE_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    PARAM_VALIDATION_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    LOG_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),   # This gives control to Mechanisms
    DELIVERY_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
    RUNTIME_PARAM_MODULATION_PREF: PreferenceEntry(Modulation.OVERRIDE, PreferenceLevel.INSTANCE)}

# Dict of default dicts
ComponentDefaultPrefDicts = {
    PreferenceLevel.COMPOSITION: CompositionDefaultPreferencesDict,
    PreferenceLevel.CATEGORY: CategoryDefaultPreferencesDict,
    PreferenceLevel.TYPE: TypeDefaultPreferencesDict,
    PreferenceLevel.SUBTYPE: SubtypeDefaultPreferencesDict,
    PreferenceLevel.INSTANCE: InstanceDefaultPreferencesDict}

def is_pref(pref):
    return pref in BasePreferenceSetPrefs


def is_pref_set(pref):
    if pref is None:
        return True
    if isinstance(pref, (BasePreferenceSet, type(None))):
        return True
    if isinstance(pref, dict):
        if all(key in BasePreferenceSetPrefs for key in pref):
            return True
    return False


class BasePreferenceSet(PreferenceSet):
    # DOCUMENT: FOR EACH pref TO BE ACCESSIBLE DIRECTLY AS AN ATTRIBUTE OF AN OBJECT,
    #           MUST IMPLEMENT IT AS PROPERTY (WITH GETTER AND SETTER METHODS) IN FUNCTION MODULE
    """Implement and manage PreferenceSets for Component class hierarchy

    Description:
        Implement the following preferences:
            - verbose (bool): enables/disables reporting of (non-exception) warnings and system function
            - paramValidation (bool):  enables/disables run-time validation of the execute method of a Function object
            - reportOutput (bool): enables/disables reporting of execution of execute method
            - log (bool): sets LogCondition for a given Component
            - functionRunTimeParams (Modulation): uses run-time params to modulate execute method params
        Implement the following preference levels:
            - SYSTEM: System level default settings (Function.classPreferences)
            - CATEGORY: category-level default settings:
                Mechanism.classPreferences
                Port.classPreferences
                Projection.classPreferences
                Function.classPreferences
            - TYPE: type-level default settings (if one exists for the category, else category-level settings are used):
                MechanismTypes:
                    ControlMechanism.classPreferences
                    ProcessingMechanism.classPreferences
                Port types:
                    InputPort.classPreferences
                    ParameterPort.classPreferences
                    OutputPort.classPreferences
                Projection types:
                    ControlProjection.classPreferences
                    MappingProjection.classPreferences
            - SUBTYPE: subtype-level default settings (if one exists for the type, else type-level settings are used):
                ControlMechanism subtypes:
                    DefaultControlMechanism.classPreferences
                    EVCControlMechanism.classPreferences
                ProcessingMechanism subtypes:
                    DDM.classPreferences
                    Linear.classPreferences
                    SigmoidLayer.classPreferences
                    IntegratorMechanism.classPreferences
            - INSTANCE: returns the setting specified in the PreferenceSetEntry of the specified object itself

    Initialization arguments:
        - owner (Function object): object to which the PreferenceSet belongs;  (default: DefaultProcessingMechanism)
            Note:  this is used to get appropriate default preferences (from class) for instantiation;
                   however, since a PreferenceSet can be assigned to multiple objects, when accessing the preference
                   the owner is set dynamically, to insure context-relevant PreferenceLevels for returning the setting
        - prefs (dict):  a specification dict, each entry of which must have a:
            key that is a keypath (PREFERENCE_KEYWORD<*>) corresponding to an attribute of the PreferenceSet, from the following set:
                + PREFERENCE_SET_NAME: specifies the name of the PreferenceSet
                + VERBOSE_PREF: print non-exception-related information during execution
                + PARAM_VALIDATION_PREF: validate parameters during execution
                + REPORT_OUTPUT_PREF: report object's ouptut during execution
                + LOG_PREF: record attribute data for the object during execution
                + DELIVERY_PREF: add attribute data to context rpc pipeline for delivery to external applications
                + RUNTIME_PARAM_MODULATION_PREF: modulate parameters using runtime specification (in pathway)
            value that is either a PreferenceSet, valid setting for the preference, or a PreferenceLevel; defaults
        - level (PreferenceLevel): ??
        - name (str): name of PreferenceSet
        - context (value): must be self (to call super's abstract class: PreferenceSet)
        - **kargs (dict): dictionary of arguments, that takes precedence over the individual args above

    Class attributes:
        + defaultPreferencesDict (PreferenceSet): SystemDefaultPreferences
        + baseClass (class): Function

    Class methods:
        Note:
        * All of the setters below use PreferenceSet.set_preference, which validates any preference info passed to it,
            and can take a PreferenceEntry, setting, or PreferenceLevel
        - verbosePref():
            returns setting for verbosePref preference at level specified in verbosePref PreferenceEntry of
             owner's PreferenceSet
        - verbosePref(setting=<value>):
            assigns the value of the setting arg to the verbosePref of the owner's PreferenceSet
        - paramValidationPref():
            returns setting for paramValidationPref preference at level specified in paramValidationPref PreferenceEntry
            of owner's PreferenceSet
        - paramValidationPref(setting=<value>):
            assigns the value of the setting arg to the paramValidationPref of the owner's PreferenceSet
        - reportOutputPref():
            returns setting for reportOutputPref preference at level specified in reportOutputPref PreferenceEntry
            of owner's Preference object
        - reportOutputPref(setting=<value>):
            assigns the value of the setting arg to the reportOutputPref of the owner's PreferenceSet
        - logPref():
            returns setting for log preference at level specified in log PreferenceEntry of owner's Preference object
        - logPref(setting=<value>):
            assigns the value of the setting arg to the logPref of the owner's PreferenceSet
                and, if it contains log entries, it adds them to the owner's log
        - runtimeParamModulationPref():
            returns setting for runtimeParamModulation preference at level specified in
             runtimeParamModulation PreferenceEntry of owner's Preference object
        - runtimeParamModulationPref(setting=<value>):
            assigns the value of the setting arg to the runtimeParamModulationPref of the owner's Preference object
    """

    # Use this as both:
    # - a template for the type of each preference (used for validation)
    # - a default set of preferences (where defaults are not otherwise specified)
    defaultPreferencesDict = {
            PREFERENCE_SET_NAME: 'BasePreferenceSetDefaults',
            VERBOSE_PREF: PreferenceEntry(False, PreferenceLevel.COMPOSITION),
            PARAM_VALIDATION_PREF: PreferenceEntry(True, PreferenceLevel.COMPOSITION),
            REPORT_OUTPUT_PREF: PreferenceEntry(True, PreferenceLevel.COMPOSITION),
            LOG_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
            DELIVERY_PREF: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
            RUNTIME_PARAM_MODULATION_PREF: PreferenceEntry(Modulation.MULTIPLY, PreferenceLevel.COMPOSITION)

    }

    baseClass = None

    def __init__(self,
                 owner=None,
                 prefs=None,
                 level=PreferenceLevel.COMPOSITION,
                 name=None,
                 **kargs):
        """Instantiate PreferenceSet for owner and/or classPreferences for owner's class

        If owner is a class, instantiate its classPreferences attribute if that does not already exist,
            using its classPreferenceLevel attribute, and the corresponding preference dict in ComponentDefaultPrefDicts
        If owner is an object:
        - if the owner's classPreferences do not yet exist, instantiate it (as described above)
        - use the owner's <class>.classPreferenceLevel to create a base set of preferences from its classPreferences
        - use PreferenceEntries, settings, or level specifications from dict in prefs arg to replace entries in base set
        If owner is omitted:
        - assigns DefaultProcessingMechanism as owner (this is updated if PreferenceSet is assigned to another object)

        :param owner:
        :param prefs:
        :param level:
        :param name:
        :param context:
        :param kargs:
        """
        if kargs:
            try:
                owner = kargs[PREFS_OWNER]
            except (KeyError, NameError):
                pass
            try:
                prefs = kargs[PREFS]
            except (KeyError, NameError):
                pass
            try:
                name = kargs[NAME]
            except (KeyError, NameError):
                pass
            try:
                level = kargs[PREF_LEVEL]
            except (KeyError, NameError):
                pass

        # If baseClass has not been assigned, do so here:
        if self.baseClass is None:
            from psyneulink.core.components.component import Component
            self.baseClass = Component

        # If owner is not specified, assign DefaultProcessingMechanism_Base as default owner
        if owner is None:
            from psyneulink.core.components.mechanisms.processing.defaultprocessingmechanism import DefaultProcessingMechanism_Base
            DefaultPreferenceSetOwner = DefaultProcessingMechanism_Base(name=DEFAULT_PREFERENCE_SET_OWNER)
            owner = DefaultPreferenceSetOwner

        # Get class
        if inspect.isclass(owner):
            owner_class = owner
        else:
            owner_class = owner.__class__

        # If classPreferences have not be instantiated for owner's class, do so here:
        try:
            # If classPreferences are still a dict, they need to be instantiated as a BasePreferenceSet
            if isinstance(owner_class.classPreferences, dict):
                raise AttributeError
        except AttributeError:
            if inspect.isclass(owner):
                # If this is a call to instantiate the classPreferences, no need to keep doing it! (infinite recursion)
                pass
            else:
                # Instantiate the classPreferences
                owner_class.classPreferences = BasePreferenceSet(
                                                    owner=owner_class,
                                                    level=owner_class.classPreferenceLevel,
                                                    prefs=ComponentDefaultPrefDicts[owner_class.classPreferenceLevel],
                                                    name=name,
                                                    )
        # Instantiate PreferenceSet
        super().__init__(owner=owner,
                         level=owner_class.classPreferenceLevel,
                         prefs=prefs,
                         name=name,
                         )
        self._level = level

    @property
    def verbosePref(self):
        """Return setting of owner's verbosePref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(VERBOSE_PREF, self._verbose_pref.level)[0]

    @verbosePref.setter
    def verbosePref(self, setting):
        """Assign setting to owner's verbosePref
        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=VERBOSE_PREF)

    @property
    def paramValidationPref(self):
        """Return setting of owner's param_validationPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively call base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(PARAM_VALIDATION_PREF, self._param_validation_pref.level)[0]


    @paramValidationPref.setter
    def paramValidationPref(self, setting):
        """Assign setting to owner's param_validationPref
        :param setting:
        :return:
        """
        self.set_preference(setting,PARAM_VALIDATION_PREF)

    @property
    def reportOutputPref(self):
        """Return setting of owner's reportOutputPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls super (closer to base) classes to get preference at specified level
        return self.get_pref_setting_for_level(REPORT_OUTPUT_PREF, self._report_output_pref.level)[0]


    @reportOutputPref.setter
    def reportOutputPref(self, setting):
        """Assign setting to owner's reportOutputPref
        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=REPORT_OUTPUT_PREF)

    @property
    def logPref(self):
        """Return setting of owner's logPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(LOG_PREF, self._log_pref.level)[0]

    @property
    def _deliveryPref(self):
        """Return setting of owner's _deliveryPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(DELIVERY_PREF, self._delivery_pref.level)[0]

    # # VERSION THAT USES OWNER'S logPref TO LIST ENTRIES TO BE RECORDED
    # @logPref.setter
    # def logPref(self, setting):
    #     """Assign setting to owner's logPref and, if it has log entries, add them to owner's log
    #     :param setting:
    #     :return:
    #     """
    #
    #     entries, level = self.set_preference(candidate_info=setting, pref_ivar_name=LOG_PREF, [str, list])
    #
    #     if entries:
    #         # Add entries to owner's log
    #         from Globals.Log import Log
    #
    #         try:
    #             self.owner.log.add_entries(entries=entries)
    #         except AttributeError:
    #             self.owner.log = Log(owner=self, entries=entries)

    # VERSION THAT USES OWNER'S logPref AS RECORDING SWITCH
    @logPref.setter
    def logPref(self, setting):
        """Assign setting to owner's logPref
        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=LOG_PREF)


    @property
    def runtimeParamModulationPref(self):
        """Returns owner's runtimeParamModulationPref
        :return:
        """
        # return self._runtime_param_modulation_pref
        return self.get_pref_setting_for_level(RUNTIME_PARAM_MODULATION_PREF,
                                               self._runtime_param_modulation_pref.level)[0]



    @runtimeParamModulationPref.setter
    def runtimeParamModulationPref(self, setting):
        """Assign runtimeParamModulationPref
        :param entry:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=RUNTIME_PARAM_MODULATION_PREF)
