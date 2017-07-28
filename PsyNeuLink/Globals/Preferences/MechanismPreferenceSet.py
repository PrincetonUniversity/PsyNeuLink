# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************** MechanismPreferenceSet **************************************************
#
#

import inspect

from PsyNeuLink.Globals.Keywords import NAME
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import kpReportOutputPref, kpRuntimeParamStickyAssignmentPref, kpVerbosePref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Globals.Utilities import Modulation

# MODIFIED 11/29/16 OLD:
# # Keypaths for preferences:
# kpRuntimeParamModulationPref = '_runtime_param_modulation_pref'
# MODIFIED 11/29/16 END

# Default PreferenceSets:
runtimeParamModulationPrefInstanceDefault = PreferenceEntry(Modulation.OVERRIDE, PreferenceLevel.INSTANCE)
runtimeParamModulationPrefTypeDefault = PreferenceEntry(Modulation.ADD, PreferenceLevel.TYPE)
# runtimeParamModulationPrefCategoryDefault = PreferenceEntry(Modulation.MULTIPLY, PreferenceLevel.CATEGORY)
runtimeParamModulationPrefCategoryDefault = PreferenceEntry(False, PreferenceLevel.CATEGORY)

runtimeParamStickyAssignmentPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
runtimeParamStickyAssignmentPrefTypeDefault = PreferenceEntry(False, PreferenceLevel.TYPE)
runtimeParamStickyAssignmentPrefCategoryDefault = PreferenceEntry(False, PreferenceLevel.CATEGORY)


class MechanismPreferenceSet(ComponentPreferenceSet):
    """Extends ComponentPreferenceSet to include Mechanism-specific preferences

    Description:
        Implements the following preference:
            - runtimeParamModulation (bool): uses specification of run-time params to update execute method params

    Class methods:
        - runtimeParamModulationPref():
            returns setting for runtimeParamModulation preference at level specified in runtimeParamModulation
            PreferenceEntry of owner's Preference object
        - runtimeParamModulationPref(setting=<value>):
            assigns the value of the setting item in the runtimeParamModulationPref PreferenceEntry of the
            owner's Preference object
        - runtimeParamModulationPrefLevel()
            returns level in the runtimeParamModulationPref PreferenceEntry of the owner's Preference object
        - runtimeParamModulationPrefLevel(level=<PreferenceLevel>):
            assigns the value of the level item in the runtimeParamModulationPref PreferenceEntry of the
            owner's Preference object
        - runtimeParamModulationPrefEntry():
            assigns PreferenceEntry to runtimeParamModulationPref attribute of the owner's Preference object
        - runtimeParamModulationPrefEntry(entry=<PreferenceEntry>):
            returns PreferenceEntry for the runtimeParamModulationPref attribute of the owner's Preference object
        - RuntimeParamStickyAssignmentPref():
            returns setting for runtimeParamStickyAssignment preference at level specified in
            runtimeParamStickyAssignment
            PreferenceEntry of owner's Preference object
        - RuntimeParamStickyAssignmentPref(setting=<value>):
            assigns the value of the setting item in the RuntimeParamStickyAssignmentPref PreferenceEntry of the
            owner's Preference object
        - RuntimeParamStickyAssignmentPrefLevel()
            returns level in the RuntimeParamStickyAssignmentPref PreferenceEntry of the owner's Preference object
        - RuntimeParamStickyAssignmentPrefLevel(level=<PreferenceLevel>):
            assigns the value of the level item in the RuntimeParamStickyAssignmentPref PreferenceEntry of the
            owner's Preference object
        - RuntimeParamStickyAssignmentPrefEntry():
            assigns PreferenceEntry to RuntimeParamStickyAssignmentPref attribute of the owner's Preference object
        - RuntimeParamStickyAssignmentPrefEntry(entry=<PreferenceEntry>):
            returns PreferenceEntry for the RuntimeParamStickyAssignmentPref attribute of the owner's Preference object
    """
    def __init__(self,
                 owner=None,
                 reportOutput_pref=reportOutputPrefInstanceDefault,
                 runtimeParamModulation_pref=runtimeParamModulationPrefInstanceDefault,
                 log_pref=logPrefInstanceDefault,
                 verbose_pref=verbosePrefInstanceDefault,
                 param_validation_pref=paramValidationPrefInstanceDefault,
                 level=PreferenceLevel.SYSTEM,
                 name=None,
                 **kargs):
        if kargs:
            try:
                owner = kargs[kwPrefsOwner]
            except (KeyError, NameError):
                pass
            try:
                reportOutput_pref = kargs[kpReportOutputPref]
            except (KeyError, NameError):
                pass
            try:
                runtime_param_modulation_pref = kargs[kpRuntimeParamModulationPref]
            except (KeyError, NameError):
                pass
            try:
                runtime_param_sticky_assignment_pref = kargs[kpRuntimeParamStickyAssignmentPref]
            except (KeyError, NameError):
                pass
            try:
                log_pref = kargs[kpLogPref]
            except (KeyError, NameError):
                pass
            try:
                param_validation_pref = kargs[kpParamValidationPref]
            except (KeyError, NameError):
                pass
            try:
                verbose_pref = kargs[kpVerbosePref]
            except (KeyError, NameError):
                pass
            try:
                name = kargs[NAME]
            except (KeyError, NameError):
                pass
            try:
                level = kargs[kwPrefLevel]
            except (KeyError, NameError):
                pass

        super(MechanismPreferenceSet, self).__init__(owner=owner,
                                                     reportOutput_pref=reportOutput_pref,
                                                     log_pref=log_pref,
                                                     verbose_pref=verbose_pref,
                                                     param_validation_pref=param_validation_pref,
                                                     level=level,
                                                     name=name)
        # self._report_output_pref = reportOutput_pref
        self._runtime_param_modulation_pref = runtime_param_modulation_pref
        self._runtime_param_sticky_assignment_pref = runtime_param_sticky_assignment_pref

    # runtimeParamModulation entry ------------------------------------------------------------------------------------

    @property
    def runtimeParamModulationPref(self):
        """Returns setting of owner's runtimeParamModulation pref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(kpRuntimeParamModulationPref,
                                               self._runtime_param_modulation_pref.level)[0]


    @runtimeParamModulationPref.setter
    def runtimeParamModulationPref(self, setting):
        """Assigns setting to owner's runtimeParamModulation pref
        :param setting:
        :return:
        """
        if isinstance(setting, PreferenceEntry):
            self._runtime_param_modulation_pref = setting

        # elif not iscompatible(setting, runtimeParamModulationPrefInstanceDefault.setting):
        elif not inspect.isfunction(runtimeParamModulationPrefInstanceDefault.setting):
            print("setting of runtimeParamModulation preference ({0}) must be a {1} or a function;"
                  " it will remain unchanged ({2})".
                  format(setting,
                         Modulation.__class__.__name__,
                         self._runtime_param_modulation_pref.setting))
            return

        else:
            self._runtime_param_modulation_pref = self._runtime_param_modulation_pref._replace(setting=setting)

    @property
    def runtimeParamModulationPrefLevel(self):
        """Returns level for owner's runtimeParamModulation pref
        :return:
        """
        return self._runtime_param_modulation_pref.level

    @runtimeParamModulationPrefLevel.setter
    def runtimeParamModulationPrefLevel(self, level):
        """Sets level for owner's runtimeParamModulation pref
        :param level:
        :return:
        """
        if not isinstance(level, PreferenceLevel):
            print("Level of runtimeParamModulation preference ({0}) must be a PreferenceLevel setting; "
                  "it will remain unchanged ({1})".
                  format(level, self._runtime_param_modulation_pref.setting))
            return
        self._runtime_param_modulation_pref = self._runtime_param_modulation_pref._replace(level=level)

    @property
    def runtimeParamModulationPrefEntry(self):
        """Returns owner's runtimeParamModulation PreferenceEntry tuple (setting, level)
        :return:
        """
        return self._runtime_param_modulation_pref

    @runtimeParamModulationPrefEntry.setter
    def runtimeParamModulationPrefEntry(self, entry):
        """Assigns runtimeParamModulation PreferenceEntry to owner
        :param entry:
        :return:
        """
        if not isinstance(entry, PreferenceEntry):
            print("runtimeParamModulationPrefEntry ({0}) must be a PreferenceEntry; it will remain unchanged ({1})".
                  format(entry, self._runtime_param_modulation_pref))
            return
        self._runtime_param_modulation_pref = entry



    # runtimeParamStickyAssignment entry -------------------------------------------------------------------------------

    @property
    def runtimeParamStickyAssignmentPref(self):
        """Returns setting of owner's runtimeParamStickyAssignment pref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(kpRuntimeParamStickyAssignmentPref,
                                               self._runtime_param_sticky_assignment_pref.level)[0]


    @runtimeParamStickyAssignmentPref.setter
    def runtimeParamStickyAssignmentPref(self, setting):
        """Assigns setting to owner's runtimeParamStickyAssignment pref
        :param setting:
        :return:
        """
        if isinstance(setting, PreferenceEntry):
            self._runtime_param_sticky_assignment_pref = setting

        # elif not iscompatible(setting, runtimeParamStickyAssignmentPrefInstanceDefault.setting):
        elif not inspect.isfunction(runtimeParamStickyAssignmentPrefInstanceDefault.setting):
            print("setting of runtimeParamStickyAssignment preference ({0}) must be a {1} or a function;"
                  " it will remain unchanged ({2})".
                  format(setting,
                         Modulation.__class__.__name__,
                         self._runtime_param_sticky_assignment_pref.setting))
            return

        else:
            self._runtime_param_sticky_assignment_pref = \
                self._runtime_param_sticky_assignment_pref._replace(setting=setting)

    @property
    def runtimeParamStickyAssignmentPrefLevel(self):
        """Returns level for owner's runtimeParamStickyAssignment pref
        :return:
        """
        return self._runtime_param_sticky_assignment_pref.level

    @runtimeParamStickyAssignmentPrefLevel.setter
    def runtimeParamStickyAssignmentPrefLevel(self, level):
        """Sets level for owner's runtimeParamStickyAssignment pref
        :param level:
        :return:
        """
        if not isinstance(level, PreferenceLevel):
            print("Level of runtimeParamStickyAssignment preference ({0}) must be a PreferenceLevel setting; "
                  "it will remain unchanged ({1})".
                  format(level, self._runtime_param_sticky_assignment_pref.setting))
            return
        self._runtime_param_sticky_assignment_pref = self._runtime_param_sticky_assignment_pref._replace(level=level)

    @property
    def runtimeParamStickyAssignmentPrefEntry(self):
        """Returns owner's runtimeParamStickyAssignment PreferenceEntry tuple (setting, level)
        :return:
        """
        return self._runtime_param_sticky_assignment_pref

    @runtimeParamStickyAssignmentPrefEntry.setter
    def runtimeParamStickyAssignmentPrefEntry(self, entry):
        """Assigns runtimeParamStickyAssignment PreferenceEntry to owner
        :param entry:
        :return:
        """
        if not isinstance(entry, PreferenceEntry):
            print("runtimeParamStickyAssignmentPrefEntry ({0}) must be a PreferenceEntry; "
                  "it will remain unchanged ({1})".
                  format(entry, self._runtime_param_sticky_assignment_pref))
            return
        self._runtime_param_sticky_assignment_pref = entry



