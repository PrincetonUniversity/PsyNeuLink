# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  EpisodicMemoryMechanism ****************************************************

"""


Contents
--------

  * `EpisodicMemoryMechanism_Overview`
  * `EpisodicMemoryMechanism_Creation`
  * `EpisodicMemoryMechanism_Structure`
  * `EpisodicMemoryMechanism_Execution`
  * `EpisodicMemoryMechanism_Class_Reference`


.. _EpisodicMemoryMechanism_Overview:

Overview
--------

An EpisodicMemoryMechanism is an `IntegratorMechanism` that can store and retrieve content-associate pairs.
Only the content is used for determining which pairs are retrieved

.. _EpisodicMemoryMechanism_Creation:

Creating a TransferMechanism
-----------------------------

An EpisodicMemoryMechanism is created by calling its constructor with **content_size** and, optionally, **assoc_size**,
that define the shapes of the items stored in its memory.

.. _EpisodicMemoryMechanism_Structure:

Structure
---------

An EpisodicMemoryMechanism has at least one `InputPort <InputPort>`, its *CONTENT_INPUT* and,
optionally, an *ASSOC_INPUT* InputPort (if its *assoc_size* is specified and is not 0) that represent
an item to store;  a `function <EpisodicMemoryMechanism.function>` that stores and retrieves content-assoc pairs from its
memory; and at least one `OutputPort <OutputPort>`, *CONTENT_OUTPUT*, as well as a 2nd, *CONTENT_OUTPUT* if it has
an *ASSOC_INPUT* InputPort, that represent a retrieved item. The default function is a `ContentAddressableMemory` that
implements a simple form of content-addressable memory, but a custom function can be specified, so long as it meets the
following requirements:

    * It must accept a 2d array as its first argument, the first item of which is the content and the second the
      associate.

    * It must return a 2d array, the first item of which is the retrieved content and the second of which is the
      assoc with which it is associated in the `function <EpisodicMemoryMechanism.function>`\\'s `memory
      <EpisodicMemoryMechanism.memory>`.

    * It may also implement a memory attribute;  if it does, it can be accessed by the EpisodicMemoryMechanism's
      `memory <EpisodicMemoryMechanism.memory>` attribute.

.. _EpisodicMemoryMechanism_Execution:

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieves an item from its memory based on the `value <InputPort.value>` of its *CONTENT_INPUT* `InputPort`;
      if no retrieval is made, appropriately shaped zero-valued arrays are assigned to the `value
      <OutputPort.value>` of the *CONTENT_OUTPUT* and, if specified, it *ASSOC_OUTPUT* OutputPorts.
    ..
    * stores the `value <InputPort.value>` of its *CONTENT_INPUT* and, if specified, *ASSOC_INPUT* `InputPort
      <InputPort>` in its memory.
    ..
    * assigns the value of the retrieved item's content in the EpisodicMemoryMechanism's  *CONTENT_OUTPUT*
      `OutputPort`, and the value of the assoc of the retrieved item in the *ASSOC_OUTPUT* OutputPort.

    .. note::
         The order of storage and retieval is determined by the function.

         The value of the content of the item retrieved from memory (and stored in *CONTENT_OUTPUT*)
         may be different than the `value <InputPort.value>` of *CONTENT* used to retrieve the item.

         If no retrieval is made, appropriately shaped zero-valued arrays are assigned as the `value
         <OutputPort.value>` of the *CONTENT_OUTPUT* and, if specified, *ASSOC_OUTPUT* OutputPorts.

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""
import warnings

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import ContentAddressableMemory
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.inputport import OutputPort
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import CONTEXT, NAME, OWNER_VALUE, SIZE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import convert_to_np_array

__all__ = ['EpisodicMemoryMechanism', 'CONTENT_INPUT', 'ASSOC_INPUT', 'CONTENT_OUTPUT', 'ASSOC_OUTPUT']

CONTENT_INPUT = 'CONTENT_INPUT'
ASSOC_INPUT = 'ASSOC_INPUT'
CONTENT_OUTPUT = 'CONTENT_OUTPUT'
ASSOC_OUTPUT = 'ASSOC_OUTPUT'


class EpisodicMemoryMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _generate_content_input_port_spec(content_size):
    return [{NAME: CONTENT_INPUT, SIZE: content_size}]


class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism(                \
        content_size=1,                     \
        assoc_size=1,                       \
        function=ContentAddressableMemory,  \
        params=None,                        \
        name=None,                          \
        prefs=None                          \
    )

    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a `differentiable neural dictionary
    (ContentAddressableMemory)<HTML>`_.  See `Mechanism <Mechanism_Class_Reference>` for additional arguments and
    attributes.

    Arguments
    ---------

    content_size : int : default 1
        specifies length of the content stored in the `function <EpisodicMemoryMechanism.function>`\\s memory.

    assoc_size : int : default 0
        specifies length of the assoc stored in the `function <EpisodicMemoryMechanism.function>`\\s memory;
        if it is 0 (the default) then no *ASSOC_INPUT* InputPort or *ASSOC_OUTPUT* OutputPort are created.

    function : function : default ContentAddressableMemory
        specifies the function that implements a memory store and methods to store to and retrieve from it.  It
        must take as its `variable <Function.variable>` a 2d array, the first item of which is the content and the second
        the associate to be stored in its memory, and must return a 2d array that is the value of the
        content and assoc retrieved from its memory.

    Attributes
    ----------

    function : function
        function that implements storage and retrieval from a memory.

    memory : 3d array
        contains key-value pairs stored in the `function <EpisodicMemoryMechanism.function>'\\s `memory` attribute
        (if it has one).

    """

    class Parameters(ProcessingMechanism_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <EpisodicMemoryMechanism.variable>`

                    :default value: [[0]]
                    :type: ``list``

                assoc_size
                    see `assoc_size <EpisodicMemoryMechanism.assoc_size>`

                    :default value: 0
                    :type: ``int``

                content_size
                    see `content_size <EpisodicMemoryMechanism.content_size>`

                    :default value: 1
                    :type: ``int``

                function
                    see `function <EpisodicMemoryMechanism.function>`

                    :default value: `ContentAddressableMemory`
                    :type: `Function`

                input_ports
                    see `input_ports <EpisodicMemoryMechanism.input_ports>`

                    :default value: ["{name: CONTENT_INPUT, size: 1}"]
                    :type: ``list``
                    :read only: True

                output_ports
                    see `output_ports <EpisodicMemoryMechanism.output_ports>`

                    :default value: ["{name: CONTENT_OUTPUT, variable: (OWNER_VALUE, 0)}"]
                    :type: ``list``
                    :read only: True
        """
        variable = Parameter([[0]], pnl_internal=True, constructor_argument='default_variable')
        function = Parameter(ContentAddressableMemory, stateful=False, loggable=False)
        content_size = 1
        assoc_size = 0

        input_ports = Parameter(
            _generate_content_input_port_spec(content_size),
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
            parse_spec=True,
        )

        output_ports = Parameter(
            [{NAME: CONTENT_OUTPUT, VARIABLE: (OWNER_VALUE, 0)}],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

    def __init__(self,
                 content_size:int=None,
                 assoc_size:int=None,
                 function:Function=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        # Template for memory_store entries
        default_variable = [np.zeros(content_size)]
        input_ports = None
        output_ports = None

        if content_size is not None and content_size != self.defaults.content_size:
            input_ports = _generate_content_input_port_spec(content_size)

        if assoc_size is not None and assoc_size != self.defaults.assoc_size:
            try:
                input_ports.append({NAME: ASSOC_INPUT, SIZE: assoc_size})
            except AttributeError:
                input_ports = [{NAME: ASSOC_INPUT, SIZE: assoc_size}]

            output_ports = self.class_defaults.output_ports.copy()
            output_ports.append({NAME: ASSOC_OUTPUT, VARIABLE: (OWNER_VALUE, 1)})
            default_variable.append(np.zeros(assoc_size))

        super().__init__(
            default_variable=default_variable,
            function=function,
            params=params,
            name=name,
            prefs=prefs,
            input_ports=input_ports,
            output_ports=output_ports,
            **kwargs
        )

    def _execute(self, variable=None, context=None, runtime_params=None):

        value = super()._execute(variable=variable,
                                 context=context,
                                 runtime_params=runtime_params,
                                )
        # Only return content if assoc has not been specified (in which case second element of value should be empty)
        if len(value[1]) == 0:
            return np.delete(value,1)
        else:
            return value


    def _instantiate_output_ports(self, context=None):
        if len(self.input_ports) != len(self.output_ports):
            assert False, \
                f'PROGRAM ERROR: Number of {InputPort.__class__.__name__}s and ' \
                f'{OutputPort.__class__.__name__}s do not match in {self.name}'
        for i, input_port_spec, output_port_spec in zip(range(len(self.input_ports) - 1),
                                                          self.input_ports,
                                                          self.output_ports):
            if input_port_spec.value is []:
                del self.output_ports[i]

        return super()._instantiate_output_ports(context=context)

    def _parse_function_variable(self, variable, context=None):

        # If assoc has not been specified, add empty list to call to function (which expects two items in its variable)
        if len(variable) != 2:
            return convert_to_np_array([variable[0],[]])
        else:
            return variable

    @property
    def memory(self):
        """Return function's memory attribute"""
        try:
            return self.function.memory
        except:
            warnings.warning(f'Function of {self.name} (self.function.name) has no memory attribute')
