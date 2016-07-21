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
# __all__ = ['kwMechanismInputStates',
#            'kwMechanismOutputStates',
#            'kwMechanismParameterState',
#            'kwMapping',
#            'kwControlSignal']

import inspect

from Globals.Keywords import *
from Globals.Registry import register_category

kwInitPy = '__init__.py'

class InitError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


#region ***************************************** MECHANISM SUBCLASSES *************************************************

# MechanismState registry
from Functions.Mechanisms.Mechanism import Mechanism_Base
from Functions.Mechanisms.Mechanism import MechanismRegistry
from Functions.Mechanisms.Mechanism import SystemDefaultMechanism_Base
from Functions.Mechanisms.SystemDefaultControlMechanism import SystemDefaultControlMechanism
from Functions.Mechanisms.EVCMechanism import EVCMechanism

# DDM
from Functions.Mechanisms.DDM import DDM
# DDM.register_category(DDM)
register_category(DDM, Mechanism_Base, MechanismRegistry, context=kwInitPy)
kwDDM = DDM.__name__

#endregion

#region *************************************** ASSIGN DEFAULT MECHANISMS **********************************************


# Use as default Mechanism in Process and in calls to mechanism()
Mechanism_Base.defaultMechanism = MechanismRegistry[Mechanism_Base.defaultMechanism].subclass

# Use as DefaultPreferenceSetOwner if owner not specified for FunctionPreferenceSet (in FunctionPreferenceSet)
SystemDefaultMechanism = SystemDefaultMechanism_Base(name=kwSystemDefaultMechanism)

# Use as kwProjectionSender (default sender for ControlSignal projections) if sender is not specified (in ControlSignal)
# Notes:
# * defaultControlAllocation specified in Globals.Defaults)

# Use as default Control Mechanism (as sender for ControlSignal Projections (for which kwControlSignal is specified)
# * can be overridden in System by kwControlMechanism
# MODIFIED 6/28/16 OLD:
# This IS the "hard-coded" default SystemControlMechanis (it is an instantiated object):
# - it is automatically assigned as the sender of default ControlSignal Projections (using kwControlSignal keyword)
#     instantiated before a System and/or any (other) SystemControlMechanism (e.g., EVC) has been instantiated
SystemDefaultController = SystemDefaultControlMechanism(name=kwSystemDefaultController)

# This should be a class, that is used to specify a subclass of SystemControlMechanism to use as
#    the default class of control mechanism to instantiate and assign, in place of the SystemDefaultController,
#    when instantiating a System for which an existing control mechanism is specified
#    - if it is either not specified or is None, SystemDefaultController will (continue to) be used (see above)
#    - if it is assigned to another subclass of SystemControlMechanism, its instantiation moves all of the
#      existing ControlSignal projections from SystemDefaultController to that instance of the specified subclass
DefaultController = EVCMechanism
# DefaultController = SystemDefaultControlMechanism
Goofiness = 'HELLO'

# MODIFIED 6/28/16 NEW:
# FIX:  CAN'T INSTANTIATE OBJECT HERE, SINCE system IS NOT YET KNOWN
#       COULD USE CLASS REFERENCE (HERE AND ABOVE), BUT THEN HAVE TO INSURE A SINGLE OBJECT IS INSTANTIATED
#       AT SOME POINT AND THAT THAT IS THE ONLY ONE USED THEREAFTER;  WHERE TO DO THAT INSTANTIATION?
#       WHEN CONTROLLER IS ASSIGNED TO SYSTEM??
# SystemDefaultController = EVCMechanism(name=kwEVCMechanism)
# SystemDefaultController = EVCMechanism
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
#             raise InitError("inputStateDefault for {0} ({1}) must be a MechanismInputState".
#                             format(mechanism, mechanism.inputStateDefault))
#endregion

#region ****************************************** REGISTER SUBCLASSES *************************************************


# SystemControlMechanism -----------------------------------------------------------------------------------------------

# SystemControlMechanism
from Functions.Mechanisms.SystemControlMechanism import SystemControlMechanism_Base
from Functions.Mechanisms.SystemControlMechanism import SystemControlMechanismRegistry

# SystemDefaultControlMechanism
from Functions.Mechanisms.SystemDefaultControlMechanism import SystemDefaultControlMechanism
register_category(SystemDefaultControlMechanism,
                  SystemControlMechanism_Base,
                  SystemControlMechanismRegistry,
                  context=kwInitPy)
kwSystemDefaultControlMechanism = SystemDefaultControlMechanism.__name__

# EVCMechanism
from Functions.Mechanisms.EVCMechanism  import EVCMechanism
register_category(EVCMechanism,
                  SystemControlMechanism_Base,
                  SystemControlMechanismRegistry,
                  context=kwInitPy)
kwEVCMechanism = EVCMechanism.__name__


# MechanismState -------------------------------------------------------------------------------------------------------

# MechanismState registry
from Functions.MechanismStates.MechanismState import MechanismState_Base
from Functions.MechanismStates.MechanismState import MechanismStateRegistry

# MechanismInputState
from Functions.MechanismStates.MechanismInputState import MechanismInputState
register_category(MechanismInputState, MechanismState_Base, MechanismStateRegistry,context=kwInitPy)
kwMechanismInputState = MechanismInputState.__name__

# MechanismOutputState
from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
register_category(MechanismOutputState, MechanismState_Base, MechanismStateRegistry,context=kwInitPy)
kwMechanismOutputState = MechanismOutputState.__name__

# MechanismParameterState
from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
register_category(MechanismParameterState, MechanismState_Base, MechanismStateRegistry,context=kwInitPy)
kwMechanismParameterState = MechanismParameterState.__name__


# Projection -----------------------------------------------------------------------------------------------------------

# Projection registry
from Functions.Projections.Projection import Projection_Base
from Functions.Projections.Projection import ProjectionRegistry

# Mapping
from Functions.Projections.Mapping import Mapping
register_category(Mapping, Projection_Base, ProjectionRegistry, context=kwInitPy)
kwMapping = Mapping.__name__

# ControlSignal
from Functions.Projections.ControlSignal import ControlSignal
register_category(ControlSignal, Projection_Base, ProjectionRegistry, context=kwInitPy)
kwControlSignal = ControlSignal.__name__

#endregion

#region ******************************************* OTHER DEFAULTS ****************************************************

#region PreferenceSet default Owner
#  Use as default owner for PreferenceSet
#
# from Functions.Function import Function
# DefaultPreferenceSetOwner = Function(0, NotImplemented)

#endregion

# region Type defaults for MechanismState and Projection
#  These methods take string constants used to specify defaults, and convert them to class or instance references
# (this is necessary, as the classes reference each other, thus producing import loops)

# Assign default Projection type for MechanismState subclasses
for state_type in MechanismStateRegistry:
    state_params =MechanismStateRegistry[state_type].subclass.paramClassDefaults
    try:
        # Use string specified in MechanismState's kwProjectionType param to get
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


# Validate / assign default sender for each Projection subclass (must be a Mechanism, MechanismState or instance of one)
for projection_type in ProjectionRegistry:
    # Get paramClassDefaults for projection_type subclass
    projection_params = ProjectionRegistry[projection_type].subclass.paramClassDefaults

    # Find kwProjectionSender, and raise exception if absent
    try:
        projection_sender = projection_params[kwProjectionSender]
    except KeyError:
        raise InitError("{0} must define paramClassDefaults[kwProjectionSender]".format(projection_type.__name__))

    # If it is a subclass of Mechanism or MechanismState, leave it alone
    if (inspect.isclass(projection_sender) and
            (issubclass(projection_sender, Mechanism_Base) or issubclass(projection_sender, MechanismState_Base))):
        continue
    # If it is an instance of Mechanism or MechanismState, leave it alone
    if isinstance(projection_sender, (Mechanism_Base, MechanismState_Base)):
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
            # Look it up in MechanismState Registry;  if that fails, raise an exception
            # FIX 5/24/16
            # projection_sender = MechanismStateRegistry[projection_sender].subclass
            projection_params[kwProjectionSender] = MechanismStateRegistry[projection_sender].subclass

        except KeyError:
            raise InitError("{0} param ({1}) for {2} not found in Mechanism or MechanismState registries".
                            format(kwProjectionSender, projection_sender, projection_type))
        else:
            continue

    raise InitError("{0} param ({1}) for {2} must be a Mechanism or MechanismState subclass or instance of one".
                    format(kwProjectionSender, projection_sender, projection_type))

#endregion

#region ***************************************** CLASS _PREFERENCES ***************************************************

from Globals.Preferences.FunctionPreferenceSet import FunctionPreferenceSet, FunctionDefaultPrefDicts, PreferenceLevel
from Functions.Mechanisms.SigmoidLayer import SigmoidLayer
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

from Functions.Process import Process
Process.classPreferences = FunctionPreferenceSet(owner=Process,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")


from Functions.Mechanisms.Mechanism import Mechanism
Mechanism.classPreferences = FunctionPreferenceSet(owner=Mechanism,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.TYPE,
                                             context=".__init__.py")

from Functions.MechanismStates.MechanismState import MechanismState
MechanismState.classPreferences = FunctionPreferenceSet(owner=MechanismState,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")

from Functions.Projections.Projection import Projection
Projection.classPreferences = FunctionPreferenceSet(owner=Projection,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")

from Functions.Utility import Utility
Utility.classPreferences = FunctionPreferenceSet(owner=Utility,
                                             prefs=FunctionDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                             level=PreferenceLevel.CATEGORY,
                                             context=".__init__.py")

# MechanismInputState.classPreferences = FunctionPreferenceSet(owner=MechanismInputState,
#                                              prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# MechanismParameterState.classPreferences = FunctionPreferenceSet(owner=MechanismParameterState,
#                                              prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# MechanismOutputState.classPreferences = FunctionPreferenceSet(owner=MechanismOutputState,
#                                              prefs=FunctionDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")

#endregion

#
#     try:
#         # First try to get spec from MechanismStateRegistry
#         projection_params[kwProjectionSender] = MechanismStateRegistry[projection_params[kwProjectionSender]].subclass
#     except AttributeError:
#         # No kwProjectionSender spec found for for projection class
#         raise InitError("paramClassDefaults[kwProjectionSender] not defined for".format(projection))
#     except (KeyError, NameError):
#         # kwProjectionSender spec not found in MechanismStateRegistry;  try next
#         pass
#
#     try:
#         # First try to get spec from MechanismStateRegistry
#         projection_params[kwProjectionSender] = MechanismRegistry[projection_params[kwProjectionSender]].subclass
#     except (KeyError, NameError):
#         # kwProjectionSender spec not found in MechanismStateRegistry;  try next
# xxx
#         # raise InitError("{0} not found in MechanismStateRegistry".format(projection_params[kwProjectionSender]))
#     else:
#         if not ((inspect.isclass(projection_params[kwProjectionSender]) and
#                      issubclass(projection_params[kwProjectionSender], MechanismState_Base)) or
#                     (isinstance(projection_params[kwProjectionSender], MechanismState_Base))):
#             raise InitError("paramClassDefaults[kwProjectionSender] for {0} ({1}) must be a type of MechanismState".
#                             format(projection, projection_params[kwProjectionSender]))

# Initialize ShellClass registries with subclasses listed above, and set their default values
#endregion


# # # MODIFIED 6/28/16: — COMMENT OUT TO RUN
# from Functions.System import System_Base
# # Use as default System (by EVC)
# DefaultSystem = System_Base(name = kwDefaultSystem)
