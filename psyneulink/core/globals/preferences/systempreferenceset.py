# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************** SystemPreferenceSet **************************************************

from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet, BasePreferenceSetPrefs
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'SystemPreferenceSet', 'recordSimulationPrefCategoryDefault', 'recordSimulationPrefInstanceDefault',
    'recordSimulationPrefTypeDefault', 'RECORD_SIMULATION_PREF', 'RECORD_SIMULATION_PREF'
]


RECORD_SIMULATION_PREF = RECORD_SIMULATION_PREF = '_record_simulation_pref'

# Defaults ffor recordSimulationPref:
recordSimulationPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
recordSimulationPrefTypeDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
recordSimulationPrefCategoryDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)

SystemPreferenceSetPrefs = BasePreferenceSetPrefs.copy()
SystemPreferenceSetPrefs.add(RECORD_SIMULATION_PREF)

def is_sys_pref(pref):
    return pref in SystemPreferenceSetPrefs


def is_sys_pref_set(pref):
    if pref is None:
        return True
    if isinstance(pref, (BasePreferenceSet, type(None))):
        return True
    if isinstance(pref, dict):
        if all(key in SystemPreferenceSetPrefs for key in pref):
            return True
    return False


class SystemPreferenceSet(BasePreferenceSet):
    """Extends BasePreferenceSet to include Mechanism-specific preferences

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
    """
    def __init__(self,
                 record_simulation_pref=recordSimulationPrefInstanceDefault,
                 **kargs):
        if kargs:
            try:
                record_simulation_pref = kargs[RECORD_SIMULATION_PREF]
            except (KeyError, NameError):
                pass

        super().__init__(
                **kargs)
        self._record_simulation_pref = record_simulation_pref

    @property
    def recordSimulationPref(self):
        """Return setting of owner's recordSimulationPref at level specified in its PreferenceEntry.level
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls super (closer to base) classes to get preference at specified level
        return self.get_pref_setting_for_level(RECORD_SIMULATION_PREF, self._record_simulation_pref.level)[0]


    @recordSimulationPref.setter
    def recordSimulationPref(self, setting):
        """Assign setting to owner's recordSimulationPref
        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=RECORD_SIMULATION_PREF)
