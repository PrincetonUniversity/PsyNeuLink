# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************
'''
This module provides core psyneulink mechanisms, projections, functions, and states

https://princetonuniversity.github.io/PsyNeuLink/Component.html
'''

#
# __all__ = ['INPUT_STATES',
#            'OUTPUT_STATES',
#            'PARAMETER_STATE',
#            'MAPPING_PROJECTION',
#            'CONTROL_PROJECTION',
#            'LEARNING_PROJECTION']

import inspect

from psyneulink.globals.keywords import PROJECTION_SENDER, PROJECTION_TYPE
from psyneulink.globals.registry import register_category

from . import component
from . import functions
from . import mechanisms
from . import process
from . import projections
from . import shellclasses
from . import states
from . import system

from .component import *
from .functions import *
from .mechanisms import *
from .process import *
from .projections import *
from .shellclasses import *
from .states import *
from .system import *

__all__ = [
    'InitError'
]
__all__.extend(component.__all__)
__all__.extend(functions.__all__)
__all__.extend(mechanisms.__all__)
__all__.extend(process.__all__)
__all__.extend(projections.__all__)
__all__.extend(shellclasses.__all__)
__all__.extend(states.__all__)
__all__.extend(system.__all__)

kwInitPy = '__init__.py'

class InitError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


#region ***************************************** MECHANISM SUBCLASSES *************************************************

from psyneulink.components.mechanisms.mechanism import MechanismRegistry
from psyneulink.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.components.mechanisms.processing.defaultprocessingmechanism \
    import DefaultProcessingMechanism_Base
from psyneulink.components.mechanisms.adaptive.control.controlmechanism \
    import ControlMechanism
register_category(entry=ControlMechanism,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

from psyneulink.components.mechanisms.adaptive.control.defaultcontrolmechanism \
    import DefaultControlMechanism
register_category(entry=DefaultControlMechanism,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

# DDM (used as DefaultMechanism)
from psyneulink.library.mechanisms.processing.integrator.ddm import DDM
register_category(entry=DDM,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

#endregion

#region *************************************** ASSIGN DEFAULT MECHANISMS **********************************************

# Specifies subclass used to instantiate a ControlMechanism if it is not specified for a System being instantiated
# Note: must be a class
# SystemDefaultControlMechanism = EVCControlMechanism
SystemDefaultControlMechanism = DefaultControlMechanism


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
from psyneulink.components.states.state import StateRegistry
from psyneulink.components.states.state import State_Base

# InputState
from psyneulink.components.states.inputstate import InputState
register_category(entry=InputState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# ParameterState
from psyneulink.components.states.parameterstate import ParameterState
register_category(entry=ParameterState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# OutputState
from psyneulink.components.states.outputstate import OutputState
register_category(entry=OutputState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# ProcessInputState
from psyneulink.components.process import ProcessInputState
register_category(entry=ProcessInputState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# ProcessInputState
from psyneulink.components.system import SystemInputState
register_category(entry=SystemInputState,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# LearningSignal
from psyneulink.components.states.modulatorysignals.learningsignal import LearningSignal
register_category(entry=LearningSignal,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# ControlSignal
from psyneulink.components.states.modulatorysignals.controlsignal import ControlSignal
register_category(entry=ControlSignal,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)

# GatingSignal
from psyneulink.components.states.modulatorysignals.gatingsignal import GatingSignal
register_category(entry=GatingSignal,
                  base_class=State_Base,
                  registry=StateRegistry,
                  context=kwInitPy)


# Projection -----------------------------------------------------------------------------------------------------------

# Projection registry
from psyneulink.components.projections.projection import ProjectionRegistry
from psyneulink.components.projections.projection import Projection_Base

# MappingProjection
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
register_category(entry=MappingProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# LearningProjection
from psyneulink.components.projections.modulatory.learningprojection import LearningProjection
register_category(entry=LearningProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# ControlProjection
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
register_category(entry=ControlProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# GatingProjection
from psyneulink.components.projections.modulatory.gatingprojection import GatingProjection
register_category(entry=GatingProjection,
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

from psyneulink.globals.preferences.componentpreferenceset \
    import ComponentPreferenceSet, ComponentDefaultPrefDicts, PreferenceLevel

from psyneulink.components.shellclasses import System_Base
System_Base.classPreferences = ComponentPreferenceSet(owner=System_Base,
                                                      prefs=ComponentDefaultPrefDicts[PreferenceLevel.INSTANCE],
                                                      level=PreferenceLevel.INSTANCE,
                                                      context=".__init__.py")

from psyneulink.components.shellclasses import Process_Base
Process_Base.classPreferences = ComponentPreferenceSet(owner=Process_Base,
                                                       prefs=ComponentDefaultPrefDicts[PreferenceLevel.INSTANCE],
                                                       level=PreferenceLevel.INSTANCE,
                                                       context=".__init__.py")


from psyneulink.components.shellclasses import Mechanism
Mechanism.classPreferences = ComponentPreferenceSet(owner=Mechanism,
                                                   prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                                   level=PreferenceLevel.TYPE,
                                                   context=".__init__.py")

DDM.classPreferences = ComponentPreferenceSet(owner=DDM,
                                             prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")


from psyneulink.components.shellclasses import State
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

from psyneulink.components.shellclasses import Projection
Projection.classPreferences = ComponentPreferenceSet(owner=Projection,
                                                    prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                                    level=PreferenceLevel.CATEGORY,
                                                    context=".__init__.py")

from psyneulink.components.functions.function import Function
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
# from Components.System import System
# # Use as default System (by EVC)
# DefaultSystem = System(name = DEFAULT_SYSTEM)
