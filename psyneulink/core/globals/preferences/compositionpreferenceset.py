# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***************************************** CompositionPreferenceSet ***************************************************

from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet, BasePreferenceSetPrefs
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'CompositionPreferenceSet', 'recordSimulationPrefCategoryDefault', 'recordSimulationPrefInstanceDefault',
    'recordSimulationPrefTypeDefault', 'RECORD_SIMULATION_PREF', 'RECORD_SIMULATION_PREF'
]

RECORD_SIMULATION_PREF = RECORD_SIMULATION_PREF = '_record_simulation_pref'

# Defaults ffor recordSimulationPref:
recordSimulationPrefInstanceDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
recordSimulationPrefTypeDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)
recordSimulationPrefCategoryDefault = PreferenceEntry(False, PreferenceLevel.INSTANCE)

CompositionPreferenceSetPrefs = BasePreferenceSetPrefs.copy()
CompositionPreferenceSetPrefs.add(RECORD_SIMULATION_PREF)


def is_composition_pref(pref):
    return pref in CompositionPreferenceSetPrefs


def is_composition_pref_set(pref):
    if pref is None:
        return True
    if isinstance(pref, (BasePreferenceSet, type(None))):
        return True
    if isinstance(pref, dict):
        if all(key in CompositionPreferenceSetPrefs for key in pref):
            return True
    return False


class CompositionPreferenceSet(BasePreferenceSet):
    """Extends BasePreferenceSet to include Composition-specific preferences

    Implements the following preference:
        - recordSimulation (bool): uses specification of runtime params to update execute method params.

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
        """Return setting of owner's recordSimulationPref at level specified in
         recordSimulation PreferenceEntry of owner's Preference object
        :param level:
        :return:
        """
        # If the level of the object is below the Preference level,
        #    recursively calls super (closer to base) classes to get preference at specified level
        return self.get_pref_setting_for_level(RECORD_SIMULATION_PREF, self._record_simulation_pref.level)[0]


    @recordSimulationPref.setter
    def recordSimulationPref(self, setting):
        """Assign setting to owner's recordSimulationPref
        (PreferenceEntry of the owner's Preference object)

        :param setting:
        :return:
        """
        self.set_preference(candidate_info=setting, pref_ivar_name=RECORD_SIMULATION_PREF)
