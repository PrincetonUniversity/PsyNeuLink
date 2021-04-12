# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  EpisodicMemoryMechanism ****************************************************

"""

# FIX: CONSOLIDATE STRUCTURE AND CREATION, AND ADD SUBHEADINGS TO Contents

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

An EpisodicMemoryMechanism is a `ProcessingMechanism` that can store and retrieve items from a content addressable
memory;  that is, on each execution it can store an item presented to it as input, and use that to retrieve an item
from its memory based on the content of the input.  The `MemoryFunction` assigned as its `function
<EpisodicMemoryMechanism.function>` determines how items are stored and retrieved. Each memory is a list or array
composed of items referred to as `memory fields <EpisodicMemoryMechanism_Memory_Fields>`, each of which is a list or
array. Memories can have an arbitrary number of fields, and each of those can be of arbitrary shape, however all
memories for a given instance of an  EpisodicMemoryMechanism must have the same shape (number of fields, and shapes
of corresponding fields).  Each `InputPort` of an EpisodicMemoryMechanism provides the input for a corresponding
field of a memory to be stored and used for retrieval.  By default, each `OutputPort` contains the value of a field
of the last retrieved memory although, as with any Mechanism, OutputPorts can be `configured in other ways
<OutputPort_Customization>`. The full set of stored memories can be accessed from the Mechanism's `memory
<EpisodicMemoryMechanism.memory>` attribute, which references its `function's <EpisodicMemoryMechanism.function>`
memory `Parameter`. Other Parameters of its function (e.g., that regulate the probability of storage and/or retrieval
-- see `ContentAddressableMemory`) can be accessed and/or `modulated <ModulatorySignal_Modulation>` in the standard way
for a Mechanism's `function <Mechanism_Base.function>`.

At present, EpisodicMemoryMechanism supports the following two MemoryFunctions:

• `ContentAddressableMemory` -- a general form of memory, that stores and retrieves memories as described above;
  any (weighted) combination of its fields can be used for retrieval.

• `DictionaryMemory` -- a more specific form of memory, that has only two fields, for keys and values, that stores
  these as pairs;  retrieval is based on similarity to the key;  this implements a format commonly used by applications
  that use dictionaries as a form of external memory.


.. _EpisodicMemoryMechanism_Creation:

Creating an EpisodicMemoryMechanism
-----------------------------------

.. _EpisodicMemoryMechanism_Creation_InputPorts:

*InputPorts, Entries and Memory Fields*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An EpisodicMemoryMechanism is created by calling its constructor with a specification of its `MemoryFunction` and
corresponding information about the number and shape of its `fields <EpisodicMemoryMechanism_Memory_Fields>`. These
are specified using the **default_variable** or **size** arguments of the constructor, in the standard way that the
`variable <Mechanism_Base.variable>` is specified for a `Component` (see `default_variable <Component_Variable>`,
`size <Component_Size>`).  These specify the shape of an entry `memory <EpisodicMemoryMechanism.memory>`, and used to
create the `InputPorts <InputPort>` for the Mechanism (see `Mechanism Variable <Mechanism_Variable_and_InputPorts>`
for additional information), with the number created equal to the number of items in the array (corresponding to
fields in an entry).  Each `input_port <Mechanism_Base.input_ports>` provides the value assigned to a corresponding
fields of the entry stored in `memory <EpisodicMemoryMechanism.memory>, and used to retrieve one similar to it.
By default, `input_port <Mechanism_Base.input_ports>` are named *FIELD_n_INPUT*, where "n" is replaced by the index of
each field; however, they can be named explicitly by specifying a list of strings in the **input_ports** argument of
the constructor; the number of these must equal the number of fields specified in **default_variable** or **size**.

.. _EpisodicMemoryMechanism_Creation_Function_Parameters:

*Function Parameters
~~~~~~~~~~~~~~~~~~~~

Parameters that govern the storage and retrieval process are specified as arguments to the `MemoryFunction` specified
in the **function** of the constructor (for example, see `ContentAddressableMemory`for parameters of the default
`function <EpisodicMemoryMechanism.function>`).

.. _EpisodicMemoryMechanism_Creation_OutputPorts:

*OutputPorts*
~~~~~~~~~~~~~

By default, a number of `OutputPorts <OutputPort>` is created equal to the number of InputPorts, each named either
*RETRIEVED_FIELD_n* or *RETRIEVED_<user specified InputPort name>*, that receive the values of the corresponding
fields of a retrieved memory.  OutputPort names can be specified explicitly, by assigning a list of strings to the
**output_ports** argument of the Mechanism's constructor;  however, in that case, or if any other forms of `OutputPort
specification <OutputPort_Forms_of_Specification>` are assigned to **output_ports**, then only the number of OutputPorts
specified are created, which may not match the number of fields of a retrieved memory.

.. _EpisodicMemoryMechanism_Structure:

Structure
---------

*Memory Fields*
~~~~~~~~~~~~~~~

Entries in the `memory <EpisodicMemoryMechanism.memory>` of an EpisodicMemoryMechanism are comprised of fields: lists
or 1d arrays within the outer list or array that comprise each entry. An entry can have an arbitrary number of fields,
and fields can be of arbitrary length.  However, all entries must have the same form (i.e., number of fields and shape
of corresponding fields across entries). One InputPort of the EpisodicMemoryMechanism is assigned to each field. Thus,
fields can be used to store different types of information in each field, and to retrieve entries from memory based on
all fields, or a weighted combination of them (as determined by the `MemoryFunction` assigned to `function
<EpisodicMemoryMechanism.function>`).

.. _EpisodicMemoryMechanism_Shape:

.. technical_note::

   The shape of an entry in `memory <EpisodicMemoryMechanism.memory>` is determined by the shape of the Mechanism's
   `variable <Mechanism_Base.variable>`. specified in the **default_variable** or **size** arguments of its constructor
   (see `EpisodicMemoryMechanism_Creation`).  Each item of `variable <Mechanism_Base.variable>` corresponds to a field.
   Both `memory <EpisodicMemoryMechanism.memory>` and all entries are stored in the EpisodicMemoryMechanism's `function
   <EpisodicMemoryMechanism.function>` as np.ndarrays, the dimensionality of which is determined by the shape of an
   entry and its fields. Fields are always stored as 1d arrays; if all fields have the same length (regular), then
   entries are 2d arrays and `memory <EpisodicMemoryMechanism.memory>` is a 3d array.  However, if fields have
   different lengths (`ragged <https://en.wikipedia.org/wiki/Jagged_array>`_) then, although each field is 1d, an
   entry is also 1d (with dtype='object'), and `memory <EpisodicMemoryMechanism.memory>` is 2d (with dtype='object').

*Input*
~~~~~~~

An EpisodicMemoryMechanism has one or more `input_ports <Mechanism_Base.input_ports>` that receive the item to be stored
and that is used to retrieve an existing entry from its memory.  If the Mechanism is assigned `ContentAddressableMemory`
as its `function <EpisodicMemoryMechanism.function>`, then it can have an arbitrary number of InputPorts, the input to
which is assigned to the corresponding `memory field <EpisodicMemoryMechanism_Memory_Fields>` of that function. By
default InputPorts are named *FIELD_n_INPUT* (see `EpisodicMemoryMechanism_Creation`). If the Mechanism is assigned
`DictionaryMemory` as its `function <EpisodicMemoryMechanism.function>`, then it is assigned at least one InputPort
(named *KEY_INPUT* by default), and optionally a second (named *VALUE_INPUT*) if **default_variable** or **size*
specifies two items; any additional fields are ignored.

.. _EpisodicMemoryMechanism_Memory_Fields:

*Function*
~~~~~~~~~~

The default function is `ContentAddressableMemory` that can store entries with an arbitrary number of fields and
shapes, and retrieve them based on a weighted similarity to any combination of those fields.  A `DictionaryMemory`
can also be be assigned, that implements a more specific form of memory in which entries are made of up key-value
pairs, and retrieved based on similarity only to the key.  A custom function can also be specified, so long as it
meets the following requirements:

    * it must accept a list or array as its first argument, the items of which are lists or arrays;

    * it must return a list or array of the same size and shape as the input;

    * it must implement a ``memory`` attribute, that can be accessed by the EpisodicMemoryMechanism's `memory
      <EpisodicMemoryMechanism.memory>` attribute.

    * it must implement ``store_memory`` and ``get_memory()`` methods that, respectively, store and retrieve
      entries from its ``memory`` attribute.

.. _EpisodicMemoryMechanism_Execution:

*Output*
~~~~~~~~

By default, an EpisodicMemoryMechanism has a number of `OutputPorts <OutputPort>` equal to its number of InputPorts,
each of which is assigned the value of a corresponding field of the entry retrieved from memory (see
`EpisodicMemoryMechanism_Creation` for naming). However, if OutputPorts were specified in the constructor, then there
may be a different number of OutputPorts than InputPorts and `memory fields <EpisodicMemoryMechanism_Memory_Fields>`,
and thus some of the latter may not be reflected in any of the Mechanism's `output_ports <Mechanism.output_ports>`.
If the `function <EpisodicMemoryMechanism.function>` is an `DictionaryMemory`, then it will have at least one
OutputPort, named *KEY_OUTPUT*, that is assigned the key (first field) of the entry retrieved from memory and, if two
fields are specified in **default_variable** or **sze**, the Mechanism will have a second OutputPort named
*VALUE_OUTPUT* that is assigned the value (second field) of the entry retrieved from memory; any additional ones are
ignored and no other OutputPorts are created.

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieve an item from `memory <EpisodicMemoryMechanism.memory>` based on the `value <InputPort.value>` of its
    `InputPorts <InputPort>`; if no retrieval is made, then an appropriately shaped zero-valued array is returned.
    ..
    * store the `value <InputPort.value>` of its `input_ports <Mechanism_Base.input_ports>` as an entry in
    <EpisodicMemoryMechanism.function>` memory.
    ..
    * assign the value of the entry retrieved to its `output_ports <Mechanism_Base.output_ports>, based on how the
    latter are configured (see `EpisodicMemoryMechanism_Creation_OutputPorts`).

    .. note::
         In general, retrieval is executed before storage, so that the current items is not also retrieved;
         however, the order of storage and retrieval is determined by the EpisodicMemoryMechanism's
         `function <EpisodicMemoryMechanism.function>`.

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""
from typing import Optional, Union

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import \
    DictionaryMemory, ContentAddressableMemory
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.globals.keywords import NAME, OWNER_VALUE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import deprecation_warning, convert_to_np_array

__all__ = ['EpisodicMemoryMechanism', 'KEY_INPUT', 'VALUE_INPUT', 'KEY_OUTPUT', 'VALUE_OUTPUT']

KEY_INPUT = 'KEY_INPUT'
VALUE_INPUT = 'VALUE_INPUT'
KEY_OUTPUT = 'KEY_OUTPUT'
VALUE_OUTPUT = 'VALUE_OUTPUT'
DEFAULT_INPUT_PORT_NAME_PREFIX = 'FIELD_'
DEFAULT_INPUT_PORT_NAME_SUFFIX = '_INPUT'
DEFAULT_OUTPUT_PORT_PREFIX = 'RETREIVED_'


class EpisodicMemoryMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# def _generate_input_port_spec(field_sizes, function):
#     # FIX:  REFACTOR FOR NEW VERSION
#     if function is DictionaryMemory:
#         return [{NAME: KEY_INPUT, SIZE: field_sizes}]
#
# def _generate_output_port_spec(field_sizes, function):
#     # FIX:  REFACTOR FOR NEW VERSION
#     [{NAME: KEY_OUTPUT, VARIABLE: (OWNER_VALUE, 0)}],


class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism()

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that implements a content addressable dictionary.
    See `Mechanism <Mechanism_Class_Reference>` for arguments and attributes.

    Attributes
    ----------

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

                COMMENT:
                # input_ports
                #     see `input_ports <EpisodicMemoryMechanism.input_ports>`
                #
                #     :default value: ["{name: KEY_INPUT, size: 1}"]
                #     :type: ``list``
                #     :read only: True
                #
                # output_ports
                #     see `output_ports <EpisodicMemoryMechanism.output_ports>`
                #
                #     :default value: ["{name: KEY_OUTPUT, variable: (OWNER_VALUE, 0)}"]
                #     :type: ``list``
                #     :read only: True
                COMMENT
        """
        variable = Parameter([[0]], pnl_internal=True, constructor_argument='default_variable')
        function = Parameter(ContentAddressableMemory, stateful=False, loggable=False)

        # FIX: IS THIS STILL NEEDED:
        # input_ports = Parameter(
        #     _generate_input_port_spec(len(variable.default_value), function),
        #     stateful=False,
        #     loggable=False,
        #     read_only=True,
        #     structural=True,
        #     parse_spec=True,
        # )
        #
        # output_ports = Parameter(
        #     _generate_output_port_spec(len(variable.default_value), function),
        #     stateful=False,
        #     loggable=False,
        #     read_only=True,
        #     structural=True,
        # )

    def __init__(self,
                 default_variable:Union[int, list, np.ndarray]=None,
                 size:Optional[Union[int, list, np.ndarray]]=None,
                 function:Optional[Function]=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        # LEGACY SUPPORT FOR DictionaryMemory
        self._dictionary_memory = (isinstance(function, DictionaryMemory)
                                  or (isinstance(function, type)
                                      and function.__name__ is DictionaryMemory.__name__))
        if self._dictionary_memory:
            # Identify and warn about any deprecated args, and return their values for reassignment
            deprecated_arg_values = deprecation_warning(self, kwargs, {'content_size':'size'})
            # Assign value of deprecated args to current ones
            if 'size' in deprecated_arg_values:
                size = deprecated_arg_values['size']
            # Need to handle assoc_size specially, since it needs to be added to what was content_size
            if 'assoc_size' in kwargs:
                if isinstance(size, int):
                    size = [size,kwargs['assoc_size']]
                else:
                    size += kwargs['assoc_size']
                kwargs.pop('assoc_size')

        super().__init__(
            default_variable=default_variable,
            size=size,
            function=function,
            params=params,
            name=name,
            prefs=prefs,
            # input_ports=input_ports,
            # output_ports=output_ports,
            **kwargs
        )

    def _instantiate_input_ports(self, context=None):

        # FIX: REMOVE THIS AND DELETE "ELSE" WHEN DictionaryMemory IS RETIRED
        if self._dictionary_memory:
            input_ports = [KEY_INPUT]
            if len(self.parameters.variable.default_value) == 2:
                input_ports.append(VALUE_INPUT)
        else:
            input_ports = self.input_ports or [f'{DEFAULT_INPUT_PORT_NAME_PREFIX}{i}{DEFAULT_INPUT_PORT_NAME_SUFFIX}'
                                               for i in range(len(self.parameters.variable.default_value))]

        super()._instantiate_input_ports(input_ports=input_ports, context=context)

    def _instantiate_output_ports(self, context=None):
        """Generate OutputPorts with names specified and values with shapes equal to corresponding InputPorts

        If OutputPorts have not been specified, use InputPort names with prefix replaced and suffix removed.

        If any OutputPorts are specified as strings, those are used as names and are the only OutputPorts instantiated
            (even if the number is less than the number of InputPorts)

        If any OutputPorts are specified in a form other than a string, then this method is ignored and OutputPorts
            are instantiated in call to super().instantiate_output_ports;
            note:  in that case, the shapes of the value are as specified and may not necessarily correspond to the
                   shapes of the corresponding Inputs (i.e., memory fields).
        """

        # FIX: REMOVE THIS AND DELETE "ELSE" WHEN DictionaryMemory IS RETIRED
        if self._dictionary_memory:
            output_ports = [{NAME: KEY_OUTPUT,
                            VARIABLE: (OWNER_VALUE, 0)}]
            if len(self.parameters.variable.default_value) == 2:
                output_ports.append({NAME: VALUE_OUTPUT,
                                     VARIABLE: (OWNER_VALUE, 1)})
            self.parameters.output_ports._set(output_ports, override=True, context=context)

        else:
            output_ports_spec = self.output_ports
            # If output_ports was not specified or they are all strings, instantiate
            # (note: need to instantiate here if output_port specs are strings, to be sure values are correct,
            #    as this doesn't seem to be handled properly by super()._instantiate_output_ports)
            if output_ports_spec is None or all(isinstance(o, str) for o in output_ports_spec):
                output_ports = []
                input_suffix_len = len(DEFAULT_INPUT_PORT_NAME_SUFFIX)
                # Total number should be either the number of names provided, or default to length of Mech's value:
                num_output_ports = len(output_ports_spec) if output_ports_spec else len(self.value)
                for i in range(num_output_ports):
                    # Names specified, so use those:
                    if output_ports_spec:
                        output_ports.append({NAME: self.output_ports[i],
                                             VARIABLE: (OWNER_VALUE, i)})
                    # Otherwise, use InputPort names as base, removing DEFAULT_INPUT_PORT_NAME_SUFFIX
                    else:
                        input_port_name = self.input_ports[i].name
                        # if input_port_name[-input_suffix_len:] == DEFAULT_INPUT_PORT_NAME_SUFFIX:
                        # if not self.input_ports[i]._user_specified:
                        if not self.parameters.input_ports._user_specified:
                            input_port_name = input_port_name[:-input_suffix_len]
                        output_ports.append({NAME: DEFAULT_OUTPUT_PORT_PREFIX + input_port_name,
                                             VARIABLE: (OWNER_VALUE, i)})
                self.parameters.output_ports._set(output_ports, override=True, context=context)

        super()._instantiate_output_ports(context=context)

        # # IMPLEMENTATION NOTE: VERSION THAT ENFORCES NUMBER OF OutputPorts EQUAL TO NUMBER OF InputPorts
        #                       (i.e., ALL MEMORY FIELDS)
        # for i in range(len(self.value)):
        #     # No OutputPut specified, so base name on corresponding InputPort,
        #     # (removing default _INPUT if it is a default name for the InputPort)
        #     if self.output_ports is None or i >= len(self.output_ports):
        #         input_port_name = self.input_ports[i].name
        #         if input_port_name[-input_suffix_len:] == DEFAULT_INPUT_PORT_NAME_SUFFIX:
        #             input_port_name = input_port_name[:-input_suffix_len]
        #         output_ports.append({NAME: 'RETREIVED_' + input_port_name,
        #                              VARIABLE: (OWNER_VALUE, i)})
        #     # String specified, so use as name
        #     elif isinstance(self.output_ports[i], str):
        #         output_ports.append({NAME: self.output_ports[i],
        #                              VARIABLE: (OWNER_VALUE, i)})
        #     # Error if specifie as other a dict or an instance of an OutputPort
        #     elif not isinstance(self.output_ports[i], dict, OutputPort):
        #         raise EpisodicMemoryMechanismError(f"Bad specification for {OutputPort.__name__} for {self.name}.")

    # IMPLEMENTATION NOTE: REMOVE THIS METHOD WHEN DictionaryMemory IS RETIRED
    def _parse_function_variable(self, variable, context=None):

        if self._dictionary_memory:
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

    # IMPLEMENTATION NOTE: REMOVE THIS METHOD WHEN DictionaryMemory IS RETIRED
    def _execute(self, variable=None, context=None, runtime_params=None):

        value = super()._execute(variable=variable,
                                 context=context,
                                 runtime_params=runtime_params,
                                )
        if self._dictionary_memory:
            # Only return content if assoc has not been specified (in which case second element of value should be empty)
            if len(value[1]) == 0:
                return np.delete(value,1)
        return value

    @property
    def memory(self):
        """Return function's memory attribute"""
        try:
            return self.function.memory
        except:
            raise EpisodicMemoryMechanismError(f"Function of {self.name} ({self.function.name}) "
                                               f"has no `memory attribute")
