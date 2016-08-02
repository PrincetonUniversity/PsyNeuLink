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
from Globals.Main import ModulationOperation
from Globals.Preferences.FunctionPreferenceSet import *

# Keypaths for preferences:
kpExecuteMethodRuntimeParamsPref = '_execute_method_runtime_params_pref'

# Default PreferenceSets:
executeMethodRuntimeParamsPrefInstanceDefault = PreferenceEntry(ModulationOperation.OVERRIDE,
                                                                PreferenceLevel.INSTANCE)
executeMethodRuntimeParamsPrefTypeDefault = PreferenceEntry(ModulationOperation.ADD,
                                                            PreferenceLevel.TYPE)
# executeMethodRuntimeParamsPrefCategoryDefault = PreferenceEntry(ModulationOperation.MULTIPLY,
#                                                                 PreferenceLevel.CATEGORY)
executeMethodRuntimeParamsPrefCategoryDefault = PreferenceEntry(ModulationOperation.OVERRIDE,
                                                                PreferenceLevel.CATEGORY)


class MechanismPreferenceSet(FunctionPreferenceSet):
    """Extends FunctionPreferenceSet to include Mechanism-specific preferences
     
    Description:
        Implements the following preference:
            - executeMethodRuntimeParams (bool): uses specification of run-time params to update execute method params

    Class methods:
        - executeMethodRuntimeParamsPref():
            returns setting for executeMethodRuntimeParams preference at level specified in executeMethodRuntimeParams PreferenceEntry of owner's Preference object
        - executeMethodRuntimeParamsPref(setting=<value>):
            assigns the value of the setting item in the executeMethodRuntimeParamsPref PreferenceEntry of the owner's Preference object
        - executeMethodRuntimeParamsPrefLevel()
            returns level in the executeMethodRuntimeParamsPref PreferenceEntry of the owner's Preference object
        - executeMethodRuntimeParamsPrefLevel(level=<PreferenceLevel>):
            assigns the value of the level item in the executeMethodRuntimeParamsPref PreferenceEntry of the owner's Preference object
        - executeMethodRuntimeParamsPrefEntry():
            assigns PreferenceEntry to executeMethodRuntimeParamsPref attribute of the owner's Preference object
        - executeMethodRuntimeParamsPrefEntry(entry=<PreferenceEntry>):
            returns PreferenceEntry for the executeMethodRuntimeParamsPref attribute of the owner's Preference object
    """
    def __init__(self,
                 owner=NotImplemented,
                 reportOutput_pref=reportOutputPrefInstanceDefault,
                 executeMethodRuntimeParams_pref=executeMethodRuntimeParamsPrefInstanceDefault,
                 log_pref=logPrefInstanceDefault,
                 verbose_pref=verbosePrefInstanceDefault,
                 param_validation_pref=paramValidationPrefInstanceDefault,
                 level=PreferenceLevel.SYSTEM,
                 name=NotImplemented,
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
                executeMethodRuntimeParams_pref = kargs[kpExecuteMethodRuntimeParamsPref]
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
                name = kargs[kwNameArg]
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
        self._execute_method_runtime_params_pref = executeMethodRuntimeParams_pref

    # executeMethodRuntimeParams entry ------------------------------------------------------------------------------------

    @property
    def executeMethodRuntimeParamsPref(self):
        """Returns setting of owner's executeMethodRuntimeParams pref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls base (super) classes to get preference at specified level
        return self.get_pref_setting_for_level(kpExecuteMethodRuntimeParamsPref,
                                               self._execute_method_runtime_params_pref.level)[0]


    @executeMethodRuntimeParamsPref.setter
    def executeMethodRuntimeParamsPref(self, setting):
        """Assigns setting to owner's executeMethodRuntimeParams pref
        :param setting:
        :return:
        """
        if isinstance(setting, PreferenceEntry):
            self._execute_method_runtime_params_pref = setting

        # elif not iscompatible(setting, executeMethodRuntimeParamsPrefInstanceDefault.setting):
        elif not inspect.isfunction(executeMethodRuntimeParamsPrefInstanceDefault.setting):
            print("setting of executeMethodRuntimeParams preference ({0}) must be a {1} or a function;"
                  " it will remain unchanged ({2})".
                  format(setting,
                         ModulationOperation.__class__.__name__,
                         self._execute_method_runtime_params_pref.setting))
            return

        else:
            self._execute_method_runtime_params_pref = self._execute_method_runtime_params_pref._replace(setting=setting)

    @property
    def executeMethodRuntimeParamsPrefLevel(self):
        """Returns level for owner's executeMethodRuntimeParams pref
        :return:
        """
        return self._execute_method_runtime_params_pref.level

    @executeMethodRuntimeParamsPrefLevel.setter
    def executeMethodRuntimeParamsPrefLevel(self, level):
        """Sets level for owner's executeMethodRuntimeParams pref
        :param level:
        :return:
        """
        if not isinstance(level, PreferenceLevel):
            print("Level of executeMethodRuntimeParams preference ({0}) must be a PreferenceLevel setting; it will remain unchanged ({1})".
                  format(level, self._execute_method_runtime_params_pref.setting))
            return
        self._execute_method_runtime_params_pref = self._execute_method_runtime_params_pref._replace(level=level)

    @property
    def executeMethodRuntimeParamsPrefEntry(self):
        """Returns owner's executeMethodRuntimeParams PreferenceEntry tuple (setting, level)
        :return:
        """
        return self._execute_method_runtime_params_pref

    @executeMethodRuntimeParamsPrefEntry.setter
    def executeMethodRuntimeParamsPrefEntry(self, entry):
        """Assigns executeMethodRuntimeParams PreferenceEntry to owner
        :param entry:
        :return:
        """
        if not isinstance(entry, PreferenceEntry):
            print("executeMethodRuntimeParamsPrefEntry ({0}) must be a PreferenceEntry; it will remain unchanged ({1})".
                  format(entry, self._execute_method_runtime_params_pref))
            return
        self._execute_method_runtime_params_pref = entry



