# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************
#
# __all__ = ['INPUT_STATES',
#            'OUTPUT_STATES',
#            'PARAMETER_STATE',
#            'MAPPING_PROJECTION',
#            'CONTROL_PROJECTION',
#            'LEARNING_PROJECTION']

import inspect

from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Globals.Registry import register_category

kwInitPy = '__init__.py'

class InitError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


#region ***************************************** MECHANISM SUBCLASSES *************************************************

from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismRegistry
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DefaultProcessingMechanism \
    import DefaultProcessingMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism \
    import EVCMechanism
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlMechanism \
    import ControlMechanism_Base
register_category(entry=ControlMechanism_Base,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.DefaultControlMechanism \
    import DefaultControlMechanism
register_category(entry=DefaultControlMechanism,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

# DDM (used as DefaultMechanism)
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
register_category(entry=DDM,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

#endregion

#region *************************************** ASSIGN DEFAULT MECHANISMS **********************************************

# Use as default Mechanism in Process and in calls to mechanism()
# Note: this must be a class (i.e., not an instantiated object)
Mechanism_Base.defaultMechanism = MechanismRegistry[Mechanism_Base.defaultMechanism].subclass

# Use as DefaultPreferenceSetOwner if owner is not specified for ComponentPreferenceSet (in ComponentPreferenceSet)
# Note: this must be an instantiated object
DefaultProcessingMechanism = DefaultProcessingMechanism_Base(name=DEFAULT_PROCESSING_MECHANISM)

# Specifies subclass used to instantiate a ControlMechanism if it is not specified for a System being instantiated
# Note: must be a class
# SystemDefaultControlMechanism = EVCMechanism
SystemDefaultControlMechanism = DefaultControlMechanism


# MODIFIED 2/22/17 OLD:
# # Use as DefaultPreferenceSetOwner if owner is not specified for ComponentPreferenceSet (in ComponentPreferenceSet)
# # Note: this must be an instantiated object
# DefaultMonitoringMechanism = ComparatorMechanism(name=DEFAULT_MONITORING_MECHANISM)
# MODIFIED 2/22/17 END

#region ****************************************** REGISTER SUBCLASSES *************************************************


# State -------------------------------------------------------------------------------------------------------

# Note:  This is used only for assignment of default projection types for each state subclass
#        Individual stateRegistries (used for naming) are created for each owner (mechanism or projection) of a state
#        Note: all states that belong to a given owner are registered in the owner's _stateRegistry,
#              which maintains a dict for each state type that it uses, a count for all instances of that type,
#              and a dictionary of those instances;  NONE of these are registered in the StateRegistry.
#              This is so that the same name can be used for instances of a state type by different owners,
#              without adding index suffixes for that name across owners
#              while still indexing multiple uses of the same base name within an owner
#
# State registry
from PsyNeuLink.Components.States.State import State_Base
from PsyNeuLink.Components.States.State import StateRegistry

# InputState
from PsyNeuLink.Components.States.InputState import InputState
register_category(entry=InputState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# OutputState
from PsyNeuLink.Components.States.OutputState import OutputState
register_category(entry=OutputState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# ParameterState
from PsyNeuLink.Components.States.ParameterState import ParameterState
register_category(entry=ParameterState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# Projection -----------------------------------------------------------------------------------------------------------

# Projection registry
from PsyNeuLink.Components.Projections.Projection import Projection_Base
from PsyNeuLink.Components.Projections.Projection import ProjectionRegistry

# MappingProjection
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
register_category(entry=MappingProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# ControlProjection
from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
register_category(entry=ControlProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# LearningProjection
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
register_category(entry=LearningProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

#endregion

#region ******************************************* OTHER DEFAULTS ****************************************************

#region PreferenceSet default Owner
#  Use as default owner for PreferenceSet
#
# from Components.Function import Function
# DefaultPreferenceSetOwner = Function(0, NotImplemented)

#endregion

# region Type defaults for State and Projection
#  These methods take string constants used to specify defaults, and convert them to class or instance references
# (this is necessary, as the classes reference each other, thus producing import loops)

# Assign default Projection type for State subclasses
for state_type in StateRegistry:
    state_params =StateRegistry[state_type].subclass.paramClassDefaults
    try:
        # Use string specified in State's PROJECTION_TYPE param to get
        # class reference for projection type from ProjectionRegistry
        state_params[PROJECTION_TYPE] = ProjectionRegistry[state_params[PROJECTION_TYPE]].subclass
    except AttributeError:
        raise InitError("paramClassDefaults[PROJECTION_TYPE] not defined for {0}".format(state_type))
    except (KeyError, NameError):
        # Check if state_params[PROJECTION_TYPE] has already been assigned to a class and, if so, use it
        if inspect.isclass(state_params[PROJECTION_TYPE]):
            state_params[PROJECTION_TYPE] = state_params[PROJECTION_TYPE]
        else:
            raise InitError("{0} not found in ProjectionRegistry".format(state_params[PROJECTION_TYPE]))
    else:
        if not (inspect.isclass(state_params[PROJECTION_TYPE]) and
                     issubclass(state_params[PROJECTION_TYPE], Projection_Base)):
            raise InitError("paramClassDefaults[PROJECTION_TYPE] ({0}) for {1} must be a type of Projection".
                            format(state_params[PROJECTION_TYPE].__name__, state_type))


# Validate / assign default sender for each Projection subclass (must be a Mechanism, State or instance of one)
for projection_type in ProjectionRegistry:
    # Get paramClassDefaults for projection_type subclass
    projection_params = ProjectionRegistry[projection_type].subclass.paramClassDefaults

    # Find PROJECTION_SENDER, and raise exception if absent
    try:
        projection_sender = projection_params[PROJECTION_SENDER]
    except KeyError:
        # raise InitError("{0} must define paramClassDefaults[PROJECTION_SENDER]".format(projection_type.__name__))
        raise InitError("{0} must define paramClassDefaults[PROJECTION_SENDER]".format(projection_type))

    # If it is a subclass of Mechanism or State, leave it alone
    if (inspect.isclass(projection_sender) and
            (issubclass(projection_sender, (Mechanism_Base, State_Base)))):
        continue
    # If it is an instance of Mechanism or State, leave it alone
    if isinstance(projection_sender, (Mechanism_Base, State_Base)):
        continue

    # If it is a string:
    if isinstance(projection_sender, str):
        try:
            # Look it up in Mechanism Registry;
            # FIX 5/24/16
            # projection_sender = MechanismRegistry[projection_sender].subclass
            projection_params[PROJECTION_SENDER] = MechanismRegistry[projection_sender].subclass
            # print("Looking for default sender ({0}) for {1} in MechanismRegistry...".
            #       format(projection_sender,projection_type.__name__))
        except KeyError:
            pass
        try:
            # Look it up in State Registry;  if that fails, raise an exception
            # FIX 5/24/16
            # projection_sender = StateRegistry[projection_sender].subclass
            projection_params[PROJECTION_SENDER] = StateRegistry[projection_sender].subclass

        except KeyError:
            raise InitError("{0} param ({1}) for {2} not found in Mechanism or State registries".
                            format(PROJECTION_SENDER, projection_sender, projection_type))
        else:
            continue

    raise InitError("{0} param ({1}) for {2} must be a Mechanism or State subclass or instance of one".
                    format(PROJECTION_SENDER, projection_sender, projection_type))

#endregion

#region ***************************************** CLASS _PREFERENCES ***************************************************

from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet \
    import ComponentPreferenceSet, ComponentDefaultPrefDicts, PreferenceLevel

from PsyNeuLink.Components.System import System
System.classPreferences = ComponentPreferenceSet(owner=System,
                                                 prefs=ComponentDefaultPrefDicts[PreferenceLevel.INSTANCE],
                                                 level=PreferenceLevel.INSTANCE,
                                                 context=".__init__.py")

from PsyNeuLink.Components.Process import Process
Process.classPreferences = ComponentPreferenceSet(owner=Process,
                                                 prefs=ComponentDefaultPrefDicts[PreferenceLevel.INSTANCE],
                                                 level=PreferenceLevel.INSTANCE,
                                                 context=".__init__.py")


from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
Mechanism.classPreferences = ComponentPreferenceSet(owner=Mechanism,
                                                   prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                                   level=PreferenceLevel.TYPE,
                                                   context=".__init__.py")

DDM.classPreferences = ComponentPreferenceSet(owner=DDM,
                                             prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")


from PsyNeuLink.Components.States.State import State
State.classPreferences = ComponentPreferenceSet(owner=State,
                                               prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                               level=PreferenceLevel.CATEGORY,
                                               context=".__init__.py")

ControlProjection.classPreferences = ComponentPreferenceSet(owner=ControlProjection,
                                                       prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
                                                       level=PreferenceLevel.TYPE,
                                                       context=".__init__.py")

MappingProjection.classPreferences = ComponentPreferenceSet(owner=MappingProjection,
                                                 prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
                                                 level=PreferenceLevel.TYPE,
                                                 context=".__init__.py")

from PsyNeuLink.Components.Projections.Projection import Projection
Projection.classPreferences = ComponentPreferenceSet(owner=Projection,
                                                    prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                                    level=PreferenceLevel.CATEGORY,
                                                    context=".__init__.py")

from PsyNeuLink.Components.Functions.Function import Function
Function.classPreferences = ComponentPreferenceSet(owner=Function,
                                                 prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                                 level=PreferenceLevel.CATEGORY,
                                                 context=".__init__.py")

# InputState.classPreferences = ComponentPreferenceSet(owner=InputState,
#                                              prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# ParameterState.classPreferences = ComponentPreferenceSet(owner=ParameterState,
#                                              prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# OutputState.classPreferences = ComponentPreferenceSet(owner=OutputState,
#                                              prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")

#endregion

#
#     try:
#         # First try to get spec from StateRegistry
#         projection_params[PROJECTION_SENDER] = StateRegistry[projection_params[PROJECTION_SENDER]].subclass
#     except AttributeError:
#         # No PROJECTION_SENDER spec found for for projection class
#         raise InitError("paramClassDefaults[PROJECTION_SENDER] not defined for".format(projection))
#     except (KeyError, NameError):
#         # PROJECTION_SENDER spec not found in StateRegistry;  try next
#         pass
#
#     try:
#         # First try to get spec from StateRegistry
#         projection_params[PROJECTION_SENDER] = MechanismRegistry[projection_params[PROJECTION_SENDER]].subclass
#     except (KeyError, NameError):
#         # PROJECTION_SENDER spec not found in StateRegistry;  try next
# xxx
#         # raise InitError("{0} not found in StateRegistry".format(projection_params[PROJECTION_SENDER]))
#     else:
#         if not ((inspect.isclass(projection_params[PROJECTION_SENDER]) and
#                      issubclass(projection_params[PROJECTION_SENDER], State_Base)) or
#                     (isinstance(projection_params[PROJECTION_SENDER], State_Base))):
#             raise InitError("paramClassDefaults[PROJECTION_SENDER] for {0} ({1}) must be a type of State".
#                             format(projection, projection_params[PROJECTION_SENDER]))

# Initialize ShellClass registries with subclasses listed above, and set their default values
#endregion


# # # MODIFIED 6/28/16: -- COMMENT OUT TO RUN
# from Components.System import System_Base
# # Use as default System (by EVC)
# DefaultSystem = System_Base(name = DEFAULT_SYSTEM)
