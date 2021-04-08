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
based on the content of the input.  It can be assigned a `MemoryFunction`, which determines how items are stored and
retrieved. At present, it supports the following two MemoryFunctions:

• `ContentAddressableMemory` -- a general form of memory, that stores 2d arrays that are comprised of one or
  more 1d arrays, referred to as `memory fields <ContentAddressableMemory_Memory_Fields>`, any or all of which can be
  used (and weighted) for retrieval.

• `DictionaryMemory` -- a more specific form of memory, the entries of which key-value pairs, that are retrieved
  based on the similarity of the key to keys in the dictionary.  This follows a format used by applications that use
  dictionaries as a form of external memory.

Each memory is a list or 2d array composed of items referred to as "fields," each of which is a list or 1d array.
Memories can have an arbitrary number of fields, each of arbitrary length, however all memories for an instance of a
EpisodicMemoryMechanism must have the same shape (number of fields, and lengths of corresponding fields).  Each
`InputPort` of an EpisodicMemoryMechanism provides the input to a corresponding field of a memory to be stored and/or
retrieved, and each `OutputPort` contains the value of a field of the last retrieved memory.  The full set of stored
memories can be accessed from the Mechanism's `memory <EpisodicMemoryMechanism.memory>` attribute, which references
its `function's <EpisodicMemoryMechanism.function>` memory attribute.  `Parameters <Parameter>` of the function
(e.g., that functions that determine retrieval, and ones that regulate the probability of storage and/or retrieval),
can be accessed and/or `modulated <ModulatorySignal_Modulation>` in the standard way for a Mechanism's `function
<Mechanism_Base.function>`.

.. _EpisodicMemoryMechanism_Creation:

Creating an EpisodicMemoryMechanism
-----------------------------------

An EpisodicMemoryMechanism is created by calling its constructor with an appropriate `MemoryFunction` and
corresponding information about the shapes of the items to be stored in memory and the `Distance` used for
comparing the Mechanism's input with entries in its memory to determine which is retrieved.  The number of fields
in a memory entry can be specified using the **default_variable** or **size** arguments of the constructor
(see `default_variable <Component_Variable>` and `size <Component_Size>` for additional information).  These are
used to construct a 2d array, the items of which are used as fields in the entry stored in memory.  A number of
`InputPorts <InputPort>` is created for the Mechanism equal to the number of items specified in **size** or
**default_variable**.  By default, these are named *FIELD_n_INPUT*, where "n" is replaced by the index of each field;
however, they can be named explicitly by specifying a list of labels in the **input_ports** argument of the
constructor.  A number of `OutputPorts <OutputPort>` is created equal to the number of InputPorts, that receive the
value of each field of the memory retrieved.  By default, they are named *FIELD_n_OUTPUT*, where "n" is replaced by
the index of each field. The OutputPorts can be named explicitly using the **output_ports** argument of the
constructor. Parameters that govern the storage and retrieval process are specified as arguments to `MemoryFunction`
specified in the **function** of the constructor (for example, see `ContentAddressableMemory` for parameters of the
default `function <EpisodicMemoryMechanism.function>`).

.. _EpisodicMemoryMechanism_Structure:

Structure
---------

Input
~~~~~

An EpisodicMemoryMechanism has one or more `InputPorts <InputPort>` that receive the item to be stored, and that is
used to retrieve an existing entry from its memory.  It has a number of `InputPorts <InputPort>` equal to the number
of elements specified in the **default_variable** or **size** argument of the constructor.  If the Mechanism is assigned
`ContentAddressableMemory` as its `function <EpisodicMemoryMechanism.function>`, then it can have an arbitrary number
of InputPorts, the input to which is assigned to the corresponding `memory field
<ContentAddressableMemory_Memory_Fields>` of that function, each named *FIELD_n_INPUT* by default (see
`EpisodicMemoryMechanism_Creation`). If the Mechanism is assigned `DictionaryMemory` as its `function
<EpisodicMemoryMechanism.function>`, then it is assigned at least one InputPort (named *KEY_INPUT* by default),
and optionally a second (named *VALUE_INPUT*) if **default_variable** or **size* specifies two items;
any additional items are ignored.

Function
~~~~~~~~

The default function is `ContentAddressableMemory` that stores vectors with an arbitrary number of fields and lengths,
and that can be retrieved by a specified subset (or all) fields and weights.  A `DictionaryMemory` can also
be assigned, that implements a more specific form of memory in which entries are made of up key-value pairs, and
retrieved based on similarity to a stored key.  A custom function can also be specified, so long as it meets the
following requirements:

    * it must accept a list or 2d array as its first argument, the items of which are lists or 1d arrays;

    * it must return a 2d array of the same size and shape as the input;

    * it must implement a ``memory`` attribute, that is accessed by the EpisodicMemoryMechanism's `memory
      <EpisodicMemoryMechanism.memory>` attribute.

    * it must implement a ``store_memory`` and a ``get_memory()`` method that, respectively, store and retrieve
      entries from its memory.


.. _EpisodicMemoryMechanism_Execution:

Output
~~~~~~

An EpisodicMemoryMechanism has a number of `OutputPorts <OutputPort>` equal to its number of InputPorts, each of which
is assigned the value of a corresponding field in the entry retrieved from memory.  By default, these are named
*FIELD_n_OUTPUT*, wher "n" is the index of the field (see `EpisodicMemoryMechanism_Creation`). If `function
<EpisodicMemoryMechanism.function>` is `DictionaryMemory`, then it will be assigned at least one OutputPort, named
*KEY_OUTPUT* and, if **default_varible** or **sze** specified two items, a second OutputPort named *VALUE_OUTPUT*;
any additional ones are ignored and no other OutputPorts are created).

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieves an item from its `function's <EpisodicMemoryMechanism.funcition>` memory based on the `value
      <InputPort.value>` of its `InputPorts <InputPort>`; if no retrieval is made, appropriately shaped zero-valued
      arrays are assigned to the `value <OutputPort.value>` of its `OutputPorts <OutputPort>`.
    ..
    * stores the `value <InputPort.value>` of its InputPorts as an entry in its `function's
      <EpisodicMemoryMechanism.function>` memory.
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
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import \
    DictionaryMemory, ContentAddressableMemory
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


def _generate_input_port_spec(field_sizes, function):
    if function is DictionaryMemory:
        return [{NAME: KEY_INPUT, SIZE: field_sizes}]
    XXX

def _generate_output_port_spec(field_sizes, function):
    [{NAME: KEY_OUTPUT, VARIABLE: (OWNER_VALUE, 0)}],


class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism(                \
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
        specifies the size of each field in the input, each of which corresponds an `InputPort` of the Mechanism
        (alias for the standard `size <Component_Size>` argument of a Component's constructor).
        For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
        For a `DicionaryMemory` function, there can be one or two elements, the first of which specifies the size
        of the *CONTENT* vector and second, if present, specifies the size of the *ASSOC*  vector;  any additional
        elements in **field_sizes** are ignored.

    field_names : list[str] : default None
        specifies the name of the `InputPort` used for each field in the input (alias for the standard `input_ports
        <Mechanism_InputPorts>` argument of a `Mechanism's <Mechanism>` consrtuctor).
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
        function = Parameter(ContentAddressableMemory, stateful=False, loggable=False)
        field_sizes = [1]

        input_ports = Parameter(
            _generate_input_port_spec(len(variable)),
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
            parse_spec=True,
        )

        output_ports = Parameter(
            _generate_output_port_spec(len(variable)),
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

    def __init__(self,
                 default_variable:Union[int, list, np.ndarray]=None,
                 size:Union[int, list, np.ndarray]=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        def deprecation_warning(component, kwargs, deprecated_args):
            # deprecated_args = {'deprecated_arg_name':('real_arg"name', real_arg_value)}

            for deprecated_arg in deprecated_arg():
                if deprecated_arg in kwargs:
                    if deprecated_args[deprecated_arg][1]:
                        # Value for real arg was also specified:
                        warnings.warn(f"Both '{deprecated_arg}' and '{deprecated_args[deprecated_arg][0]}' "
                                      f"were specified in the constructor for a(n) {self.__class__.__name__}; "
                                      f"{deprecated_arg} ({kwargs[deprecated_arg]}) will be used,"
                                      f"but note that it is deprecated  and may be removed in the future.")
                    else:
                        # Only deprecated arg was specified:
                        warnings.warn(f"'{deprecated_arg}' was specified in the constructor for a(n)"
                                      f" {self.__class__.__name__}; note that this has been deprecated "
                                      f"and may be removed in the future; '{deprecated_args[deprecated_arg][0]}' "
                                      f"should be used instead.")
                    # FIX: PUT DEPRECATED VALUE IN DICT WITH REAL VALUE AND RETURN FOR ASSIGNMENT BY CALLER
                    kwargs.pop(deprecated_arg)
                continue

        # for k in kwargs.copy():
        #     if k == 'content_size':
        #         if default_variable or size:
        #             warnings.warn(f"Both 'content_size' and 'default_variable' or 'size' were specified "
        #                           f"in the constructor for an {self.__class__.__name__}; "
        #                           f"'content_size' ({kwargs['content_size']}) will be used, "
        #                           f"but note that this is deprecated for the future.")
        #             size = kwargs.pop('content_size')
        #         continue
        #     if k == 'assoc_size':
        #         if default_variable or size:
        #             warnings.warn(f"Both 'assoc_size' and 'default_variable' or 'size' were specified "
        #                           f"in the constructor for an {self.__class__.__name__}; "
        #                           f"'assoc_size' ({kwargs['assoc_size']}) will be used, "
        #                           f"but note that this is deprecated for the future.")
        #             size = kwargs.pop('assoc_size')
        #         continue
        deprecation_warning(self, kwargs, {'content_size':'size',
                                           'assoc_size':'size'})

        # GET fields HERE FROM default_variable OR size
        # Template for memory_store entries
        # FIX: MANAGE field_size -> size, input_ports, and output_ports


        # default_variable = [np.zeros(field_sizes)]
        input_ports = None
        output_ports = None

        # FIX: MOVE THIS TO A LATER METHOD, SO THAT DEFAULT_VARIBLE/SIZE CAN BE RESOLVED:
        if field_sizes is not None and field_sizes != self.defaults.field_sizes:
            input_ports = _generate_input_port_spec(field_sizes)

        if assoc_size is not None and assoc_size != self.defaults.assoc_size:
            try:
                input_ports.append({NAME: VALUE_INPUT, SIZE: assoc_size})
            except AttributeError:
                input_ports = [{NAME: VALUE_INPUT, SIZE: assoc_size}]

            output_ports = self.class_defaults.output_ports.copy()
            # FIX: BASE THIS ON field_sizes:
            output_ports.append({NAME: VALUE_OUTPUT, VARIABLE: (OWNER_VALUE, 1)})
            default_variable.append(np.zeros(assoc_size))

        if function is None:
            function = self.parameters.function.default_value()

        super().__init__(
            # default_variable=default_variable,
            size=size,
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
            raise EpisodicMemoryMechanismError(f"Function of {self.name} ({self.function.name}) "
                                               f"has no `memory attribute")
