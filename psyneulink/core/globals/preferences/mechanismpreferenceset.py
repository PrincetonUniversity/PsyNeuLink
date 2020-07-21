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

from psyneulink.core.globals.keywords import NAME, PREF_LEVEL, PREFS_OWNER
from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet, LOG_PREF, PARAM_VALIDATION_PREF, REPORT_OUTPUT_PREF, RUNTIME_PARAM_MODULATION_PREF, VERBOSE_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.utilities import Modulation

__all__ = [
    'MechanismPreferenceSet', 'runtimeParamModulationPrefCategoryDefault', 'runtimeParamModulationPrefInstanceDefault',
    'runtimeParamModulationPrefTypeDefault'
]

# MODIFIED 11/29/16 OLD:
# # Keypaths for preferences:
# RUNTIME_PARAM_MODULATION_PREF = '_runtime_param_modulation_pref'
# MODIFIED 11/29/16 END

# Default PreferenceSets:
runtimeParamModulationPrefInstanceDefault = PreferenceEntry(Modulation.OVERRIDE, PreferenceLevel.INSTANCE)
runtimeParamModulationPrefTypeDefault = PreferenceEntry(Modulation.ADD, PreferenceLevel.TYPE)
# runtimeParamModulationPrefCategoryDefault = PreferenceEntry(Modulation.MULTIPLY, PreferenceLevel.CATEGORY)
runtimeParamModulationPrefCategoryDefault = PreferenceEntry(False, PreferenceLevel.CATEGORY)

reportOutputPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
logPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
verbosePrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
paramValidationPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)


class MechanismPreferenceSet(BasePreferenceSet):
    """Extends BasePreferenceSet to include Mechanism-specific preferences

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
    """
    def __init__(self,
                 owner=None,
                 reportOutput_pref=reportOutputPrefInstanceDefault,
                 runtimeParamModulation_pref=runtimeParamModulationPrefInstanceDefault,
                 log_pref=logPrefInstanceDefault,
                 verbose_pref=verbosePrefInstanceDefault,
                 param_validation_pref=paramValidationPrefInstanceDefault,
                 level=PreferenceLevel.COMPOSITION,
                 name=None,
                 **kargs):
        if kargs:
            try:
                owner = kargs[PREFS_OWNER]
            except (KeyError, NameError):
                pass
            try:
                reportOutput_pref = kargs[REPORT_OUTPUT_PREF]
            except (KeyError, NameError):
                pass
            try:
                runtime_param_modulation_pref = kargs[RUNTIME_PARAM_MODULATION_PREF]
            except (KeyError, NameError):
                pass
            try:
                log_pref = kargs[LOG_PREF]
            except (KeyError, NameError):
                pass
            try:
                param_validation_pref = kargs[PARAM_VALIDATION_PREF]
            except (KeyError, NameError):
                pass
            try:
                verbose_pref = kargs[VERBOSE_PREF]
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

        super(MechanismPreferenceSet, self).__init__(owner=owner,
                                                     reportOutput_pref=reportOutput_pref,
                                                     log_pref=log_pref,
                                                     verbose_pref=verbose_pref,
                                                     param_validation_pref=param_validation_pref,
                                                     level=level,
                                                     name=name)
        # self._report_output_pref = reportOutput_pref
        self._runtime_param_modulation_pref = runtime_param_modulation_pref

    # runtimeParamModulation entry ------------------------------------------------------------------------------------

    @property
    def runtimeParamModulationPref(self):
        """Returns setting of owner's runtimeParamModulation pref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(RUNTIME_PARAM_MODULATION_PREF,
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
