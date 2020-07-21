# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************
"""
This module provides core psyneulink mechanisms, projections, functions, and ports

https://princetonuniversity.github.io/PsyNeuLink/Component.html
"""

#
# __all__ = ['INPUT_PORTS',
#            'OUTPUT_PORTS',
#            'PARAMETER_PORT',
#            'MAPPING_PROJECTION',
#            'CONTROL_PROJECTION',
#            'LEARNING_PROJECTION']

import inspect

from psyneulink.core.globals.keywords import PROJECTION_SENDER, PROJECTION_TYPE
from psyneulink.core.globals.registry import register_category

from . import component
from . import functions
from . import mechanisms
from . import projections
from . import shellclasses
from . import ports

from .component import *
from .functions import *
from .mechanisms import *
from .projections import *
from .shellclasses import *
from .ports import *

from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM
from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet, ComponentDefaultPrefDicts, PreferenceLevel

__all__ = [
    'InitError'
]
__all__.extend(component.__all__)
__all__.extend(functions.__all__)
__all__.extend(mechanisms.__all__)
__all__.extend(projections.__all__)
__all__.extend(shellclasses.__all__)
__all__.extend(ports.__all__)

kwInitPy = '__init__.py'

class InitError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


#region ***************************************** MECHANISM SUBCLASSES *************************************************

register_category(entry=ControlMechanism,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

register_category(entry=DefaultControlMechanism,
                  base_class=Mechanism_Base,
                  registry=MechanismRegistry,
                  context=kwInitPy)

# DDM (used as DefaultMechanism)
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


# Port -------------------------------------------------------------------------------------------------------

# Note:  This is used only for assignment of default projection types for each port subclass
#        Individual stateRegistries (used for naming) are created for each owner (mechanism or projection) of a state
#        Note: all ports that belong to a given owner are registered in the owner's _portRegistry,
#              which maintains a dict for each port type that it uses, a count for all instances of that type,
#              and a dictionary of those instances;  NONE of these are registered in the PortRegistry.
#              This is so that the same name can be used for instances of a port type by different owners,
#              without adding index suffixes for that name across owners
#              while still indexing multiple uses of the same base name within an owner
#
# Port registry

# InputPort
register_category(entry=InputPort,
                  base_class=Port_Base,
                  registry=PortRegistry,
                  context=kwInitPy)

# ParameterPort
register_category(entry=ParameterPort,
                  base_class=Port_Base,
                  registry=PortRegistry,
                  context=kwInitPy)

# OutputPort
register_category(entry=OutputPort,
                  base_class=Port_Base,
                  registry=PortRegistry,
                  context=kwInitPy)

# LearningSignal
register_category(entry=LearningSignal,
                  base_class=Port_Base,
                  registry=PortRegistry,
                  context=kwInitPy)

# ControlSignal
register_category(entry=ControlSignal,
                  base_class=Port_Base,
                  registry=PortRegistry,
                  context=kwInitPy)

# GatingSignal
register_category(entry=GatingSignal,
                  base_class=Port_Base,
                  registry=PortRegistry,
                  context=kwInitPy)


# Projection -----------------------------------------------------------------------------------------------------------

# Projection registry

# MappingProjection
register_category(entry=MappingProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# LearningProjection
register_category(entry=LearningProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# ControlProjection
register_category(entry=ControlProjection,
                  base_class=Projection_Base,
                  registry=ProjectionRegistry,
                  context=kwInitPy)

# GatingProjection
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

# region Type defaults for Port and Projection
#  These methods take string constants used to specify defaults, and convert them to class or instance references
# (this is necessary, as the classes reference each other, thus producing import loops)

# Assign default Projection type for Port subclasses
for port_type in PortRegistry:
    projection_type = PortRegistry[port_type].subclass.projection_type
    try:
        # Use string specified in Port's PROJECTION_TYPE param to get
        # class reference for projection type from ProjectionRegistry
        PortRegistry[port_type].subclass.projection_type = ProjectionRegistry[projection_type].subclass
        projection_type = ProjectionRegistry[projection_type].subclass
    except AttributeError:
        raise InitError("projection_type not defined for {0}".format(port_type))
    except (KeyError, NameError):
        # Check if port_Params[PROJECTION_TYPE] has already been assigned to a class and, if so, use it
        if not inspect.isclass(projection_type):
            raise InitError("{0} not found in ProjectionRegistry".format(projection_type))
    else:
        if not (inspect.isclass(projection_type) and
                     issubclass(projection_type, Projection_Base)):
            raise InitError("projection_type ({0}) for {1} must be a type of Projection".
                            format(projection_type.__name__, port_type))


# Validate / assign default sender for each Projection subclass (must be a Mechanism, Port or instance of one)
for projection_type in ProjectionRegistry:
    projection_sender = ProjectionRegistry[projection_type].subclass.projection_sender

    # If it is a subclass of Mechanism or Port, leave it alone
    if (inspect.isclass(projection_sender) and
            (issubclass(projection_sender, (Mechanism_Base, Port_Base)))):
        continue
    # If it is an instance of Mechanism or Port, leave it alone
    if isinstance(projection_sender, (Mechanism_Base, Port_Base)):
        continue

    # If it is a string:
    if isinstance(projection_sender, str):
        try:
            # Look it up in Mechanism Registry;
            # FIX 5/24/16
            # projection_sender = MechanismRegistry[projection_sender].subclass
            ProjectionRegistry[projection_type].subclass.projection_sender = MechanismRegistry[projection_sender].subclass
            # print("Looking for default sender ({0}) for {1} in MechanismRegistry...".
            #       format(projection_sender,projection_type.__name__))
        except KeyError:
            pass
        try:
            # Look it up in Port Registry;  if that fails, raise an exception
            # FIX 5/24/16
            # projection_sender = PortRegistry[projection_sender].subclass
            ProjectionRegistry[projection_type].subclass.projection_sender = PortRegistry[projection_sender].subclass

        except KeyError:
            raise InitError("{0} param ({1}) for {2} not found in Mechanism or Port registries".
                            format(PROJECTION_SENDER, projection_sender, projection_type))
        else:
            continue

    raise InitError("{0} param ({1}) for {2} must be a Mechanism or Port subclass or instance of one".
                    format(PROJECTION_SENDER, projection_sender, projection_type))

#endregion

#region ***************************************** CLASS _PREFERENCES ***************************************************
System_Base.classPreferences = BasePreferenceSet(owner=System_Base,
                                                 prefs=ComponentDefaultPrefDicts[PreferenceLevel.INSTANCE],
                                                 level=PreferenceLevel.INSTANCE,
                                                 context=".__init__.py")

Process_Base.classPreferences = BasePreferenceSet(owner=Process_Base,
                                                  prefs=ComponentDefaultPrefDicts[PreferenceLevel.INSTANCE],
                                                  level=PreferenceLevel.INSTANCE,
                                                  context=".__init__.py")


Mechanism.classPreferences = BasePreferenceSet(owner=Mechanism,
                                               prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                               level=PreferenceLevel.TYPE,
                                               context=".__init__.py")

DDM.classPreferences = BasePreferenceSet(owner=DDM,
                                         prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
                                         level=PreferenceLevel.TYPE,
                                         context=".__init__.py")


Port.classPreferences = BasePreferenceSet(owner=Port,
                                          prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                          level=PreferenceLevel.CATEGORY,
                                          context=".__init__.py")

ControlProjection.classPreferences = BasePreferenceSet(owner=ControlProjection,
                                                       prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
                                                       level=PreferenceLevel.TYPE,
                                                       context=".__init__.py")

MappingProjection.classPreferences = BasePreferenceSet(owner=MappingProjection,
                                                       prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
                                                       level=PreferenceLevel.TYPE,
                                                       context=".__init__.py")

Projection.classPreferences = BasePreferenceSet(owner=Projection,
                                                prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                                level=PreferenceLevel.CATEGORY,
                                                context=".__init__.py")

Function.classPreferences = BasePreferenceSet(owner=Function,
                                              prefs=ComponentDefaultPrefDicts[PreferenceLevel.CATEGORY],
                                              level=PreferenceLevel.CATEGORY,
                                              context=".__init__.py")

# InputPort.classPreferences = BasePreferenceSet(owner=InputPort,
#                                              prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# ParameterPort.classPreferences = BasePreferenceSet(owner=ParameterPort,
#                                              prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")
#
# OutputPort.classPreferences = BasePreferenceSet(owner=OutputPort,
#                                              prefs=ComponentDefaultPrefDicts[PreferenceLevel.TYPE],
#                                              level=PreferenceLevel.TYPE,
#                                              context=".__init__.py")

#endregion
