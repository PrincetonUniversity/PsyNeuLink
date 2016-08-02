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
# __all__ = ['kwInputStates',
#            'kwOutputStates',
#            'kwParameterState',
#            'kwMapping',
#            'kwControlSignal']

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

from PsyNeuLink.Functions.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Functions.Mechanisms.Mechanism import MechanismRegistry
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DefaultProcessingMechanism import DefaultProcessingMechanism_Base
from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.DefaultControlMechanism import DefaultControlMechanism
from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.EVCMechanism import EVCMechanism


# DDM ------------------------------------------------------------------------------------------------------------------

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import DDM
# DDM.register_category(DDM)
register_category(DDM, Mechanism_Base, MechanismRegistry, context=kwInitPy)
# kwDDM = DDM.__name__

# # SystemControlMechanisms ----------------------------------------------------------------------------------------------
#
# # SystemControlMechanism
# from Functions.Mechanisms.SystemControlMechanism import SystemControlMechanism_Base
# from Functions.Mechanisms.SystemControlMechanism import SystemControlMechanismRegistry
#
# # DefaultControlMechanism
# from Functions.Mechanisms.DefaultControlMechanism import DefaultControlMechanism
# register_category(DefaultControlMechanism,
#                   SystemControlMechanism_Base,
#                   SystemControlMechanismRegistry,
#                   context=kwInitPy)
# # kwDefaultControlMechanism = DefaultControlMechanism.__name__
#
# # EVCMechanism
# from Functions.Mechanisms.EVCMechanism  import EVCMechanism
# register_category(EVCMechanism,
#                   SystemControlMechanism_Base,
#                   SystemControlMechanismRegistry,
#                   context=kwInitPy)
# # kwEVCMechanism = EVCMechanism.__name__
#

#endregion

#region *************************************** ASSIGN DEFAULT MECHANISMS **********************************************


# Use as default Mechanism in Process and in calls to mechanism()
Mechanism_Base.defaultMechanism = MechanismRegistry[Mechanism_Base.defaultMechanism].subclass

# Use as DefaultPreferenceSetOwner if owner is not specified for FunctionPreferenceSet (in FunctionPreferenceSet)
DefaultProcessingMechanism = DefaultProcessingMechanism_Base(name=kwDefaultProcessingMechanism)

# Use as kwProjectionSender (default sender for ControlSignal projections) if sender is not specified (in ControlSignal)

# Specifies instantiated DefaultController (SystemControlMechanism):
# - automatically assigned as the sender of default ControlSignal Projections (that use the kwControlSignal keyword)
# - instantiated before a System and/or any (other) SystemControlMechanism (e.g., EVC) has been instantiated
# - can be overridden in System by kwControlMechanism
# - uses the defaultControlAllocation (specified in Globals.Defaults) to assign ControlSignal intensities
DefaultController = DefaultControlMechanism(name=kwSystemDefaultController)

# Specifies subclass of SystemControlMechanism used as the default class of control mechanism to instantiate and assign,
#    in place of DefaultController, when instantiating a System for which an existing control mech is specified
# - if it is either not specified or is None, DefaultController will (continue to) be used (see above)
# - if it is assigned to another subclass of SystemControlMechanism, its instantiation moves all of the
#     existing ControlSignal projections from DefaultController to that instance of the specified subclass
SystemDefaultControlMechanism = EVCMechanism
# SystemDefaultControlMechanism = DefaultControlMechanism

# MODIFIED 6/28/16 NEW:
# FIX:  CAN'T INSTANTIATE OBJECT HERE, SINCE system IS NOT YET KNOWN
#       COULD USE CLASS REFERENCE (HERE AND ABOVE), BUT THEN HAVE TO INSURE A SINGLE OBJECT IS INSTANTIATED
#       AT SOME POINT AND THAT THAT IS THE ONLY ONE USED THEREAFTER;  WHERE TO DO THAT INSTANTIATION?
#       WHEN CONTROLLER IS ASSIGNED TO SYSTEM??
# DefaultController = EVCMechanism(name=kwEVCMechanism)
# DefaultController = EVCMechanism
# MODIFIED END:

# # # MODIFIED 6/28/16: EVC — COMMENT OUT TO RUN
# from Functions.System import System_Base
# # Use as default System (by EVC)
# DefaultSystem = System_Base(name = kwDefaultSystem)


# # Assign default inputState for Mechanism subclasses
# for mechanism in MechanismRegistry:
#     try:
#         mechanism.inputStateTypeDefault = MechanismRegistry[mechanism.inputStateDefault].subclass
#     except AttributeError:
#         raise InitError("inputStateDefault not defined for".format(mechanism))
#     except (KeyError, NameError):
#         raise InitError("{0} not found in MechanismRegistry".format(mechanism))
#     else:
#         if not isinstance(mechanism.inputStateDefault, Mechanism_Base):
#             raise InitError("inputStateDefault for {0} ({1}) must be a InputState".
#                             format(mechanism, mechanism.inputStateDefault))
#endregion

#region ****************************************** REGISTER SUBCLASSES *************************************************


# State -------------------------------------------------------------------------------------------------------

# State registry
from PsyNeuLink.Functions.States.State import State_Base
from PsyNeuLink.Functions.States.State import StateRegistry

# InputState
from PsyNeuLink.Functions.States.InputState import InputState
register_category(InputState, State_Base, StateRegistry,context=kwInitPy)
# kwInputState = InputState.__name__

# OutputState
from PsyNeuLink.Functions.States.OutputState import OutputState
register_category(OutputState, State_Base, StateRegistry,context=kwInitPy)
# kwOutputState = OutputState.__name__

# ParameterState
from PsyNeuLink.Functions.States.ParameterState import ParameterState
register_category(ParameterState, State_Base, StateRegistry,context=kwInitPy)
# kwParameterState = ParameterState.__name__


# Projection -----------------------------------------------------------------------------------------------------------

# Projection registry
from PsyNeuLink.Functions.Projections.Projection import Projection_Base
from PsyNeuLink.Functions.Projections.Projection import ProjectionRegistry

# Mapping
from PsyNeuLink.Functions.Projections.Mapping import Mapping
register_category(Mapping, Projection_Base, ProjectionRegistry, context=kwInitPy)
# kwMapping = Mapping.__name__

# ControlSignal
from PsyNeuLink.Functions.Projections.ControlSignal import ControlSignal
register_category(ControlSignal, Projection_Base, ProjectionRegistry, context=kwInitPy)
# kwControlSignal = ControlSignal.__name__

#endregion

#region ******************************************* OTHER DEFAULTS ****************************************************

#region PreferenceSet default Owner
#  Use as default owner for PreferenceSet
#
# from Functions.Function import Function
# DefaultPreferenceSetOwner = Function(0, NotImplemented)

#endregion

# region Type defaults for State and Projection
#  These methods take string constants used to specify defaults, and convert them to class or instance references
# (this is necessary, as the classes reference each other, thus producing import loops)

# Assign default Projection type for State subclasses
for state_type in StateRegistry:
    state_params =StateRegistry[state_type].subclass.paramClassDefaults
    try:
        # Use string specified in State's kwProjectionType param to get
        # class reference for projection type from ProjectionRegistry
        state_params[kwProjectionType] = ProjectionRegistry[state_params[kwProjectionType]].subclass
    except AttributeError:
        raise InitError("paramClassDefaults[kwProjectionType] not defined for {0}".format(state_type))
    except (KeyError, NameError):
        # Check if state_params[kwProjectionType] has already been assigned to a class and, if so, use it
        if inspect.isclass(state_params[kwProjectionType]):
            state_params[kwProjectionType] = state_params[kwProjectionType]
        else:
            raise InitError("{0} not found in ProjectionRegistry".format(state_params[kwProjectionType]))
    else:
        if not (inspect.isclass(state_params[kwProjectionType]) and
                     issubclass(state_params[kwProjectionType], Projection_Base)):
            raise InitError("paramClassDefaults[kwProjectionType] ({0}) for {1} must be a type of Projection".
                            format(state_params[kwProjectionType].__name__, state_type))


# Validate / assign default sender for each Projection subclass (must be a Mechanism, State or instance of one)
for projection_type in ProjectionRegistry:
    # Get paramClassDefaults for projection_type subclass
    projection_params = ProjectionRegistry[projection_type].subclass.paramClassDefaults

    # Find kwProjectionSender, and raise exception if absent
    try:
        projection_sender = projection_params[kwProjectionSender]
    except KeyError:
        raise InitError("{0} must define paramClassDefaults[kwProjectionSender]".format(projection_type.__name__))

    # If it is a subclass of Mechanism or State, leave it alone
    if (inspect.isclass(projection_sender) and
            (issubclass(projection_sender, Mechanism_Base) or issubclass(projection_sender, State_Base))):
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
            projection_params[kwProjectionSender] = MechanismRegistry[projection_sender].subclass
            # print("Looking for default sender ({0}) for {1} in MechanismRegistry...".
            #       format(projection_sender,projection_type.__name__))
        except KeyError:
            pass
        try:
            # Look it up in State Registry;  if that fails, raise an exception
            # FIX 5/24/16
            # projection_sender = StateRegistry[projection_sender].subclass
            projection_params[kwProjectionSender] = StateRegistry[projection_sender].subclass

        except KeyError:
            raise InitError("{0} param ({1}) for {2} not found in Mechanism or State registries".
                            format(kwProjectionSender, projection_sender, projection_type))
        else:
            continue

    raise InitError("{0} param ({1}) for {2} must be a Mechanism or State subclass or instance of one".
                    format(kwProjectionSender, projection_sender, projection_type))

#endregion

#region ***************************************** CLASS _PREFERENCES ***************************************************

from PsyNeuLink.Globals.Preferences import FunctionPreferenceSet, FunctionDefaultPrefDicts, PreferenceLevel
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Deprecated.SigmoidLayer import SigmoidLayer
SigmoidLayer.classPreferences = FunctionPreferenceSet(owner=SigmoidLayer,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")

DDM.classPreferences = FunctionPreferenceSet(owner=DDM,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")

ControlSignal.classPreferences = FunctionPreferenceSet(owner=ControlSignal,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")

Mapping.classPreferences = FunctionPreferenceSet(owner=Mapping,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")

from PsyNeuLink.Functions.Process import Process
Process.classPreferences = FunctionPreferenceSet(owner=Process,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")


from PsyNeuLink.Functions.Mechanisms.Mechanism import Mechanism
Mechanism.classPreferences = FunctionPreferenceSet(owner=Mechanism,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")

from PsyNeuLink.Functions.States.State import State
State.classPreferences = FunctionPreferenceSet(owner=State,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")

from PsyNeuLink.Functions.Projections.Projection import Projection
Projection.classPreferences = FunctionPreferenceSet(owner=Projection,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")

from PsyNeuLink.Functions.Utility import Utility
Utility.classPreferences = FunctionPreferenceSet(owner=Utility,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")

# InputState.classPreferences = FunctionPreferenceSet(owner=InputState,
#                                              prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# ParameterState.classPreferences = FunctionPreferenceSet(owner=ParameterState,
#                                              prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# OutputState.classPreferences = FunctionPreferenceSet(owner=OutputState,
#                                              prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")

#endregion

#
#     try:
#         # First try to get spec from StateRegistry
#         projection_params[kwProjectionSender] = StateRegistry[projection_params[kwProjectionSender]].subclass
#     except AttributeError:
#         # No kwProjectionSender spec found for for projection class
#         raise InitError("paramClassDefaults[kwProjectionSender] not defined for".format(projection))
#     except (KeyError, NameError):
#         # kwProjectionSender spec not found in StateRegistry;  try next
#         pass
#
#     try:
#         # First try to get spec from StateRegistry
#         projection_params[kwProjectionSender] = MechanismRegistry[projection_params[kwProjectionSender]].subclass
#     except (KeyError, NameError):
#         # kwProjectionSender spec not found in StateRegistry;  try next
# xxx
#         # raise InitError("{0} not found in StateRegistry".format(projection_params[kwProjectionSender]))
#     else:
#         if not ((inspect.isclass(projection_params[kwProjectionSender]) and
#                      issubclass(projection_params[kwProjectionSender], State_Base)) or
#                     (isinstance(projection_params[kwProjectionSender], State_Base))):
#             raise InitError("paramClassDefaults[kwProjectionSender] for {0} ({1}) must be a type of State".
#                             format(projection, projection_params[kwProjectionSender]))

# Initialize ShellClass registries with subclasses listed above, and set their default values
#endregion


# # # MODIFIED 6/28/16: — COMMENT OUT TO RUN
# from Functions.System import System_Base
# # Use as default System (by EVC)
# DefaultSystem = System_Base(name = kwDefaultSystem)
