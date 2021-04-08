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

An EpisodicMemoryMechanism is an `IntegratorMechanism` that can store and retrieve items from a content addressable
memory;  that is, it can store items presented to it as input, and use that to retrieve an item from its memory
based on the content of the input.  It can be assigned an appropriate `MemoryFunction`, which determines how items
are stored and retrieved. At present, it supports the following two MemoryFunctions:

• `ContentAddressableMemory` -- a general form of memory, that stores 2d arrays that are comprised of one or
  more 1d arrays, referred to as `memory fields <ContentAddressableMemory_Memory_Fields>`, any or all of which can be
  used (and weighted) for retrieval.

• `DictionaryMemory` -- a more specific form of memory, the entries of which key-value pairs, that are retrieved
  based on the similarity of the key to keys in the dictionary.  This follows a format used by applications that use
  dictionaries as a form of external memory.


.. _EpisodicMemoryMechanism_Creation:

Creating an EpisodicMemoryMechanism
-----------------------------------

An EpisodicMemoryMechanism is created by calling its constructor with an appropriate `MemoryFunction` and
corresponding information about the shapes of the items to be stored in memory and the `Distance` used for
comparing the Mechanism's input with entries in its memory to determine which is retrieved.  The number of fields
in a memory entry can be specified using the **field_sizes** argument of the constructor (this is an alias to the
standard `size <Component_Size>` argument of a `Component`.  This creates a number of `InputPorts <InputPort>` for
the Mechanism equal to the number of elements in **fields_sizes**, each of which will receive an input for a
corresponding field of the entry to be stored. The InputPorts can be named by specifying a list of labels in the
**input_ports** argument of the constructor.  A number of `OutputPorts <OutputPort>` is created equal to the number
of InputPorts, that receive the value of each field of the memory retrieved.  The OutputPorts can be named using 
the **output_ports** argument of the constructor.  Parameters that govern the storage and retrieval process are
specified as arguments to `MemoryFunction` specified in the **function** of the constructor (for example, see 
`ContentAddressableMemory` for parameters of the default `function <EpisodicMemoryMechanism.function>`).

.. _EpisodicMemoryMechanism_Structure:

Structure
---------

Input
~~~~~

An EpisodicMemoryMechanism has one or more `InputPorts <InputPort>` that receive the item to be stored, and that is
used to retrieve an existing entry from its memory.  It has a number of `InputPorts <InputPort>` equal to the number
of elements specified in the **field_sizes** argument of its constructor.  If the Mechanism is assigned 
`ContentAddressableMemory` as its `function <EpisodicMemoryMechanism.function>`, then it
can have an arbitrary number of InputPorts, the input to which is assigned to the corresponding `memory field
<ContentAddressableMemory_Memory_Fields>` of that function. If the Mechanism is assigned `DictionaryMemory` as
its `function <EpisodicMemoryMechanism.function>`, then it is assigned at least one InputPort (named *KEY_INPUT* by
default), and optionally a second (named *VALUE_INPUT*) if **field_sizes** has two elements; any additional elements in
**field_sizes** are ignored.

Function
~~~~~~~~

The default function is `ContentAddressableMemory` that stores vectors with an arbitrary number of fields and lengths,
and that can be retrieved by a specified subset (or all) fields and weights.  A `DictionaryMemory` can also
be assigned, that implements a more specific form of memory in which entries are made of up key-value pairs, and
retrieved based on similarity to a stored key.  A custom function can also be specified, so long as it meets the
following requirements:

    * It must accept a list or 2d array as its first argument, the items of which are lists or 1d arrays.

    * It must return a 2d array of the same size and shape as the input.

    * It may also implement a ``memory`` attribute;  if it does, it can be accessed by the EpisodicMemoryMechanism's
      `memory <EpisodicMemoryMechanism.memory>` attribute.

.. _EpisodicMemoryMechanism_Execution:

Output
~~~~~~

An EpisodicMemoryMechanism has one or more `OutputPorts <OutputPort>` that contain entry retrieved from memory.
If `function <EpisodicMemoryMechanism.function>` is `ContentAddressableMemory`, then the Mechanism will have a
number of OutputPorts equal to the to the number of elements in the **field_sizes** argument of its constructor.
If `function <EpisodicMemoryMechanism.function>` is `DictionaryMemory`, then it will be assigned at least one
OutputPort, named *KEY_OUTPUT* and, if **field_sizes** has two elements, a second OutputPort named *VALUE_OUTPUT*
(if **field_sizes** has more than two elements, the additional ones are ignored and no other OutputPorts are created).

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieves an item from its `function's <EpisodicMemoryMechanism.funcition>` memory based on the `value
      <InputPort.value>` of its `InputPorts <InputPort>`; if no retrieval is made, appropriately shaped zero-valued
      arrays are assigned to the `value <OutputPort.value>` of its `OutputPorts <OutputPort>`.
    ..
    * stores the `value <InputPort.value>` of its InputPorts as an entry in its `function's
      <EpisodicMemoryMechanism.funcition>` memory.
    ..
    * assigns the value of the retrieved to its OutputPorts.

    .. note::
         The order of storage and retrieval is determined by the function.

         If no retrieval is made, appropriately shaped zero-valued arrays are assigned as the `value
         <OutputPort.value>` of the Mechanism's OutputPorts.

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""
import warnings

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import DictionaryMemory
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.inputport import OutputPort
from psyneulink.core.globals.keywords import NAME, OWNER_VALUE, SIZE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import convert_to_np_array

__all__ = ['EpisodicMemoryMechanism', 'KEY_INPUT', 'VALUE_INPUT', 'KEY_OUTPUT', 'VALUE_OUTPUT']

KEY_INPUT = 'KEY_INPUT'
VALUE_INPUT = 'VALUE_INPUT'
KEY_OUTPUT = 'KEY_OUTPUT'
VALUE_OUTPUT = 'VALUE_OUTPUT'


class EpisodicMemoryMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _generate_KEY_INPUT_port_spec(content_size):
    return [{NAME: KEY_INPUT, SIZE: content_size}]


class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism(                \
        field_sizes=[1],                    \
        field_names=None                    \
        function=ContentAddressableMemory,  \
        params=None,                        \
        name=None,                          \
        prefs=None                          \
    )

    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a content addressable dictionary.
    See `Mechanism <Mechanism_Class_Reference>` for additional arguments and attributes.

    COMMENT:
    OLD
    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a `differentiable neural dictionary
    (DictionaryMemory)<HTML>`_.  See `Mechanism <Mechanism_Class_Reference>` for additional arguments and
    attributes.
    COMMENT

    Arguments
    ---------

    # content_size : int : default 1
    #     specifies length of the content stored in the `function <EpisodicMemoryMechanism.function>`\\s memory.
    #
    # assoc_size : int : default 0
    #     specifies length of the assoc stored in the `function <EpisodicMemoryMechanism.function>`\\s memory;
    #     if it is 0 (the default) then no *VALUE_INPUT* InputPort or *VALUE_OUTPUT* OutputPort are created.


    field_sizes : list[int] or 1d array : default [1]
        specifies the size of each field in the input, each of which corresponds an `InputPort` of the Mechanism.
        For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
        For a `DicionaryMemory` function, there can be one or two elements, the first of which specifies the size
        of the *CONTENT* vector and second, if present, specifies the size of the *ASSOC*  vector;  any additional
        elements in **field_sizes** are ignored.

    field_names : list[str] : default None
        specifies the name of the `InputPort` used for each field in the input.
        For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
        For a `DicionaryMemory` function, there can be one or two elements, the first of which specifies the size


        For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
        For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.


    FIX: NOTE: PUT WARNING HERE ABOUT FIELDS WITH SIZE 1 PRODUCING PARTICULAR (POTENTIALLY UNANTICIPATED) RESULTS
    WITH SOME DISTANCE METRICS (SCALRS & EUCLIDEAN MEASURES = 0)

    function : function : default DictionaryMemory
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

                    :default value: `DictionaryMemory`
                    :type: `Function`

                input_ports
                    see `input_ports <EpisodicMemoryMechanism.input_ports>`

                    :default value: ["{name: KEY_INPUT, size: 1}"]
                    :type: ``list``
                    :read only: True

                output_ports
                    see `output_ports <EpisodicMemoryMechanism.output_ports>`

                    :default value: ["{name: KEY_OUTPUT, variable: (OWNER_VALUE, 0)}"]
                    :type: ``list``
                    :read only: True
        """
        variable = Parameter([[0]], pnl_internal=True, constructor_argument='default_variable')
        function = Parameter(DictionaryMemory, stateful=False, loggable=False)
        content_size = 1
        assoc_size = 0

        input_ports = Parameter(
            _generate_KEY_INPUT_port_spec(content_size),
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
            parse_spec=True,
        )

        output_ports = Parameter(
            [{NAME: KEY_OUTPUT, VARIABLE: (OWNER_VALUE, 0)}],
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
            input_ports = _generate_KEY_INPUT_port_spec(content_size)

        if assoc_size is not None and assoc_size != self.defaults.assoc_size:
            try:
                input_ports.append({NAME: VALUE_INPUT, SIZE: assoc_size})
            except AttributeError:
                input_ports = [{NAME: VALUE_INPUT, SIZE: assoc_size}]

            output_ports = self.class_defaults.output_ports.copy()
            output_ports.append({NAME: VALUE_OUTPUT, VARIABLE: (OWNER_VALUE, 1)})
            default_variable.append(np.zeros(assoc_size))

        if function is None:
            function = self.parameters.function.default_value()

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
            # Check that both are assigned inputs:
            missing_inputs = [self.input_ports.names[i] for i,t in enumerate([v for v in variable]) if t is None]
            if missing_inputs:
                if len(missing_inputs) == 1:
                    missing_str = 'an input'
                    s = ''
                else:
                    missing_str = 'inputs'
                    s = 's'
                raise EpisodicMemoryMechanismError(f"{self.name} is missing {missing_str} for its"
                                                   f" {'and '.join(missing_inputs)} {InputPort.__name__}{s}.")

            return variable

    @property
    def memory(self):
        """Return function's memory attribute"""
        try:
            return self.function.memory
        except:
            warnings.warning(f'Function of {self.name} (self.function.name) has no memory attribute')
