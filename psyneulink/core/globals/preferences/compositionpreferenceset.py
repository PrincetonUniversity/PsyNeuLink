# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************** SystemPreferenceSet **************************************************

from psyneulink.core.globals.preferences.componentpreferenceset import ComponentPreferenceSet, ComponentPreferenceSetPrefs
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'CompositionPreferenceSet'
]

CompositionPreferenceSetPrefs = {
    kpVerbosePref,
    kpParamValidationPref,
    kpReportOutputPref,
    kpLogPref,
    kpRuntimeParamModulationPref
}


def is_sys_pref(pref):
    return pref in CompositionPreferenceSetPrefs


def is_sys_pref_set(pref):
    if pref is None:
        return True
    if isinstance(pref, (ComponentPreferenceSet, type(None))):
        return True
    if isinstance(pref, dict):
        if all(key in CompositionPreferenceSetPrefs for key in pref):
            return True
    return False


class ComponentPreferenceSet(PreferenceSet):
    """Implement and manage PreferenceSets for Composition class hierarchy
    """
    defaultPreferencesDict = {
            PREFERENCE_SET_NAME: 'CompositionPreferenceSetDefaults',
            kpVerbosePref: PreferenceEntry(False, PreferenceLevel.SYSTEM),
            kpParamValidationPref: PreferenceEntry(True, PreferenceLevel.SYSTEM),
            kpLogPref: PreferenceEntry(LogCondition.OFF, PreferenceLevel.CATEGORY),
            kpRuntimeParamModulationPref: PreferenceEntry(Modulation.MULTIPLY, PreferenceLevel.SYSTEM)

    }

    baseClass = None

    def __init__(self,
                 owner=None,
                 prefs=None,
                 level=PreferenceLevel.SYSTEM,
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
            # If classPreferences are still a dict, they need to be instantiated as a ComponentPreferenceSet
            if isinstance(owner_class.classPreferences, dict):
                raise AttributeError
        except AttributeError:
            if inspect.isclass(owner):
                # If this is a call to instantiate the classPreferences, no need to keep doing it! (infinite recursion)
                pass
            else:
                # Instantiate the classPreferences
                owner_class.classPreferences = ComponentPreferenceSet(
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
        return self.get_pref_setting_for_level(kpVerbosePref, self._verbose_pref.level)[0]

    @verbosePref.setter
    def verbosePref(self, setting):
        """Assign setting to owner's verbosePref
        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=kpVerbosePref)

    @property
    def paramValidationPref(self):
        """Return setting of owner's param_validationPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively call base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(kpParamValidationPref, self._param_validation_pref.level)[0]


    @paramValidationPref.setter
    def paramValidationPref(self, setting):
        """Assign setting to owner's param_validationPref
        :param setting:
        :return:
        """
        self.set_preference(setting,kpParamValidationPref)

    @property
    def reportOutputPref(self):
        """Return setting of owner's reportOutputPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls super (closer to base) classes to get preference at specified level
        return self.get_pref_setting_for_level(kpReportOutputPref, self._report_output_pref.level)[0]


    @reportOutputPref.setter
    def reportOutputPref(self, setting):
        """Assign setting to owner's reportOutputPref
        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=kpReportOutputPref)

    @property
    def logPref(self):
        """Return setting of owner's logPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(kpLogPref, self._log_pref.level)[0]

    # # VERSION THAT USES OWNER'S logPref TO LIST ENTRIES TO BE RECORDED
    # @logPref.setter
    # def logPref(self, setting):
    #     """Assign setting to owner's logPref and, if it has log entries, add them to owner's log
    #     :param setting:
    #     :return:
    #     """
    #
    #     entries, level = self.set_preference(candidate_info=setting, pref_ivar_name=kpLogPref, [str, list])
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
        self.set_preference(candidate_info=setting, pref_ivar_name=kpLogPref)


    @property
    def runtimeParamModulationPref(self):
        """Returns owner's runtimeParamModulationPref
        :return:
        """
        # return self._runtime_param_modulation_pref
        return self.get_pref_setting_for_level(kpRuntimeParamModulationPref,
                                               self._runtime_param_modulation_pref.level)[0]



    @runtimeParamModulationPref.setter
    def runtimeParamModulationPref(self, setting):
        """Assign runtimeParamModulationPref
        :param entry:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=kpRuntimeParamModulationPref)
