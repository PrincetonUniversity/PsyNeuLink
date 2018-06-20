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

from psyneulink.globals.keywords import NAME, kwPrefLevel, kwPrefsOwner
from psyneulink.globals.preferences.componentpreferenceset import ComponentPreferenceSet, \
    kpLogPref, kpParamValidationPref, kpReportOutputPref, kpVerbosePref, kpRuntimeParamModulationPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.globals.utilities import Modulation


__all__ = [
    'SystemPreferenceSet', 'recordSimulationPrefCategoryDefault', 'recordSimulationPrefInstanceDefault',
    'recordSimulationPrefTypeDefault', 'RECORD_SIMULATION_PREF'
]


RECORD_SIMULATION_PREF = kpRecordSimulationPref = '_record_simulation_pref'

# Default PreferenceSets:
recordSimulationPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
recordSimulationPrefTypeDefault = PreferenceEntry(False, PreferenceLevel.TYPE)
recordSimulationPrefCategoryDefault = PreferenceEntry(False, PreferenceLevel.CATEGORY)

reportOutputPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
logPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
verbosePrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
paramValidationPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)

SystemPreferenceSetPrefs = {
    kpVerbosePref,
    kpParamValidationPref,
    kpReportOutputPref,
    kpRecordSimulationPref,
    kpLogPref,
    kpRuntimeParamModulationPref
}

def is_sys_pref(pref):
    return pref in SystemPreferenceSetPrefs


def is_sys_pref_set(pref):
    if pref is None:
        return True
    if isinstance(pref, (SystemPreferenceSet, type(None))):
        return True
    if isinstance(pref, dict):
        if all(key in SystemPreferenceSetPrefs for key in pref):
            return True
    return False


class SystemPreferenceSet(ComponentPreferenceSet):
    """Extends ComponentPreferenceSet to include Mechanism-specific preferences

    Description:
        Implements the following preference:
            - recordSimulation (bool): uses specification of run-time params to update execute method params

    Class methods:
        - recordSimulationPref():
            returns setting for recordSimulation preference at level specified in recordSimulation
            PreferenceEntry of owner's Preference object
        - recordSimulationPref(setting=<value>):
            assigns the value of the setting item in the recordSimulationPref PreferenceEntry of the
            owner's Preference object
        - recordSimulationPrefLevel()
            returns level in the recordSimulationPref PreferenceEntry of the owner's Preference object
        - recordSimulationPrefLevel(level=<PreferenceLevel>):
            assigns the value of the level item in the recordSimulationPref PreferenceEntry of the
            owner's Preference object
        - recordSimulationPrefEntry():
            assigns PreferenceEntry to recordSimulationPref attribute of the owner's Preference object
        - recordSimulationPrefEntry(entry=<PreferenceEntry>):
            returns PreferenceEntry for the recordSimulationPref attribute of the owner's Preference object
    """
    def __init__(self,
                 owner=None,
                 reportOutput_pref=reportOutputPrefInstanceDefault,
                 record_simulation_pref=recordSimulationPrefInstanceDefault,
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
                record_simulation_pref = kargs[kpRecordSimulationPref]
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

        super().__init__(owner=owner,
                         reportOutput_pref=reportOutput_pref,
                         log_pref=log_pref,
                         verbose_pref=verbose_pref,
                         param_validation_pref=param_validation_pref,
                         level=level,
                         name=name)
        # self._report_output_pref = reportOutput_pref
        self._record_simulation_pref = record_simulation_pref

    # recordSimulation entry ------------------------------------------------------------------------------------
    # @property
    # def recordSimulationPref(self):
    #     """Returns setting of owner's recordSimulation pref at level specified in its PreferenceEntry.level
    #     :param level:
    #     :return:
    #     """
    #     # If the level of the object is below the Preference level,
    #     #    recursively calls base (super) classes to get preference at specified level
    #     return self.get_pref_setting_for_level(kpRecordSimulationPref,
    #                                            self._record_simulation_pref.level)[0]
    # 
    # @recordSimulationPref.setter
    # def recordSimulationPref(self, setting):
    #     """Assigns setting to owner's recordSimulation pref
    #     :param setting:
    #     :return:
    #     """
    #     if isinstance(setting, PreferenceEntry):
    #         self._record_simulation_pref = setting
    # 
    #     # elif not iscompatible(setting, recordSimulationPrefInstanceDefault.setting):
    #     elif not inspect.isfunction(recordSimulationPrefInstanceDefault.setting):
    #         print("setting of recordSimulation preference ({0}) must be a {1} or a function;"
    #               " it will remain unchanged ({2})".
    #               format(setting,
    #                      Modulation.__class__.__name__,
    #                      self._record_simulation_pref.setting))
    #         return
    # 
    #     else:
    #         self._record_simulation_pref = self._record_simulation_pref._replace(setting=setting)
    # 
    # @property
    # def recordSimulationPrefLevel(self):
    #     """Returns level for owner's recordSimulation pref
    #     :return:
    #     """
    #     return self._record_simulation_pref.level
    # 
    # @recordSimulationPrefLevel.setter
    # def recordSimulationPrefLevel(self, level):
    #     """Sets level for owner's recordSimulation pref
    #     :param level:
    #     :return:
    #     """
    #     if not isinstance(level, PreferenceLevel):
    #         print("Level of recordSimulation preference ({0}) must be a PreferenceLevel setting; "
    #               "it will remain unchanged ({1})".
    #               format(level, self._record_simulation_pref.setting))
    #         return
    #     self._record_simulation_pref = self._record_simulation_pref._replace(level=level)
    # 
    # @property
    # def recordSimulationPrefEntry(self):
    #     """Returns owner's recordSimulation PreferenceEntry tuple (setting, level)
    #     :return:
    #     """
    #     return self._record_simulation_pref
    # 
    # @recordSimulationPrefEntry.setter
    # def recordSimulationPrefEntry(self, entry):
    #     """Assigns recordSimulation PreferenceEntry to owner
    #     :param entry:
    #     :return:
    #     """
    #     if not isinstance(entry, PreferenceEntry):
    #         print("recordSimulationPrefEntry ({0}) must be a PreferenceEntry; it will remain unchanged ({1})".
    #               format(entry, self._record_simulation_pref))
    #         return
    #     self._record_simulation_pref = entry
    # 

    @property
    def recordSimulationPref(self):
        """Return setting of owner's recordSimulationPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls super (closer to base) classes to get preference at specified level
        return self.get_pref_setting_for_level(kpRecordSimulationPref, self._record_simulation_pref.level)[0]


    @recordSimulationPref.setter
    def recordSimulationPref(self, setting):
        """Assign setting to owner's recordSimulationPref
        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=kpRecordSimulationPref)
