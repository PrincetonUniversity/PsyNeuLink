# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************  ShellClasses *************************************************************

"""Define globally-visible classes for Base classes and typechecking functions for parameters of PsyNeuLink Components

Shell Classes
-------------

Used to allow classes to refer to one another without creating import loops,
including (but not restricted to) the following dependencies:
- `Projection <Projection>` subclasses must see (particular) `Port <Port>` subclasses in order to assign
  `kwProjectionSender`
- `Port <Port>` subclasses must see (particular) `Projection <Projection>` subclasses in order to assign
  `PROJECTION_TYPE`
- `Process` must see `Mechanism <Mechanism>` subclasses to assign `PsyNeuLink.Components.DefaultMechanism`

TBI:
  `Mechanism <Mechanism>`, `Projection <Projection>` (and possibly `Port <Port>`) classes should be extensible:
  developers should be able to create, register and refer to subclasses (plug-ins), without modifying core code

"""

from psyneulink.core.components.component import Component

__all__ = [
    'Function', 'Mechanism', 'Process_Base', 'Projection', 'ShellClass', 'ShellClassError', 'Port', 'System_Base',
]


class ShellClassError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _attempt_to_call_base_class(cls, alternative):
    raise ShellClassError("Can't call {} directly;  must use {}".format(cls.__class__.__name__, alternative))


class ShellClass(Component):
    pass


# ******************************************* SYSTEM *******************************************************************

class Composition_Base(ShellClass):
    pass


# ******************************************* SYSTEM *******************************************************************

class System_Base(ShellClass):
    pass


# ****************************************** PROCESS *******************************************************************


class Process_Base(ShellClass):
    pass

# ******************************************* MECHANISM ****************************************************************


class Mechanism(ShellClass):

    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=None,
                 param_defaults=None,
                 name=None,
                 prefs=None,
                 **kwargs):
        super().__init__(default_variable=default_variable,
                         size=size,
                         function=function,
                         param_defaults=param_defaults,
                         name=name,
                         prefs=prefs,
                         **kwargs)

    def _validate_params(self, request_set, target_set=None, context=None):
        raise ShellClassError("Must implement _validate_params in {0}".format(self))

    def adjust_function(self, params, context):
        raise ShellClassError("Must implement adjust_function in {0}".format(self))


# ********************************************* PORT ******************************************************************


class Port(ShellClass):

    @property
    def owner(self):
        raise ShellClassError("Must implement @property owner method in {0}".format(self.__class__.__name__))

    @owner.setter
    def owner(self, assignment):
        raise ShellClassError("Must implement @owner.setter method in {0}".format(self.__class__.__name__))

    def _validate_variable(self, variable, context=None):
        raise ShellClassError("Must implement _validate_variable in {0}".format(self))

    def _validate_params(self, request_set, target_set=None, context=None):
        raise ShellClassError("Must implement _validate_params in {0}".format(self))

    def add_observer_for_keypath(self, object, keypath):
        raise ShellClassError("Must implement add_observer_for_keypath in {0}".format(self.__class__.__name__))

    def set_value(self, new_value):
        raise ShellClassError("Must implement set_value in {0}".format(self.__class__.__name__))

    def _update(self, params=None, context=None):
        raise ShellClassError("{} must implement update".format(self.__class__.__name__))


# ******************************************* PROJECTION ***************************************************************


class Projection(ShellClass):

    # def assign_states(self):
    #     raise ShellClassError("Must implement assign_states in {0}".format(self.__class__.__name__))
    def validate_states(self):
        raise ShellClassError("Must implement validate_states in {0}".format(self.__class__.__name__))

    def _validate_params(self, request_set, target_set=None, context=None):
        raise ShellClassError("Must implement _validate_params in {0}".format(self.__class__.__name__))


# *********************************************  FUNCTION  *************************************************************


class Function(ShellClass):

    def _instantiate_function(self, function, function_params=None, context=None):
        return
