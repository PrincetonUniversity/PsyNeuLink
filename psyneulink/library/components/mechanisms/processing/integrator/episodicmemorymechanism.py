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
      - `EpisodicMemoryMechanism_Creation_InputPorts`
      - `EpisodicMemoryMechanism_Creation_Function_Parameters`
      - `EpisodicMemoryMechanism_Creation_OutputPorts`
  * `EpisodicMemoryMechanism_Structure`
      - `EpisodicMemoryMechanism_Memory_Fields`
      - `EpisodicMemoryMechanism_Input`
      - `EpisodicMemoryMechanism_Function`
      - `EpisodicMemoryMechanism_Output`
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
for a Mechanism's `function <EpisodicMemoryMechanism.function>`.

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

An EpisodicMemoryMechanism is created by calling its constructor with specifications used to implement its
`MemoryFunction` and the shape of entries stored in `memory <EpisodicMemoryMechanism.memory>`.  The latter
is determined by the number of `fields <EpisodicMemoryMechanism_Memory_Fields>` and shape of each in an entry.
These can be specified using any of the following arguments:

  * **default_variable** or **size** -- these are specified in the standard way that the `variable
    <Mechanism_Base.variable>` is specified for any  `Component` (see `default_variable <Component_Variable>`,
    `size <Component_Size>`, respectively);  the specified value is passed to the constructor for the
    EpisodicMemoryMechanism's `function <EpisodicMemoryMechanism.function>`), which determines the shape of an entry
    in `memory <EpisodicMemoryMechanism.memory>`;  the `memory <EpisodicMemoryMechanism.memory>` itself remains
    empty until the Mechanism is executed and an item is stored.

  * **memory** -- specifies a set of entries to be stored in `memory <EpisodicMemoryMechanism.memory>`;  it is passed
    to the constructor for the EpisodicMemoryMechanism's `function <EpisodicMemoryMechanism.function>`) as its
    **initializer** argument (see `initializer <ContentAddressableMemory.initializer>` for an example).

  * **function** -- this can be used to specify a constructor for the `function <EpisodicMemoryMechanism.function>`,
    in which the **default_variable** or **initializer** arguments are used to specify the shape of entries in `memory
    <EpisodicMemoryMechanism.memory>`.  If **default** variable is used, `memory <EpisodicMemoryMechanism.memory>`
    remains empty until the Mechanism is executed and an item is stored. If **initializer** is used, the items
    specified are stored in `memory <EpisodicMemoryMechanism.memory>` and are available for retrieval in the first
    execution of the Mechanism (see `initializer <ContentAddressableMemory.initializer>` for an example).

The above specifications are also used to to create the `input_ports <Mechanism_Base.input_ports>` for the Mechanism
in the same way that the `variable <Mechanism_Base.variable>` is used for any Mechanism (see `Mechanism Variable
<Mechanism_Variable_and_InputPorts>` for additional information), with the number of InputPorts created equal to the
number of fields in an entry of `memory <EpisodicMemoryMechanism.memory>`.  Each `input_port
<EpisodicMemoryMechanism.input_ports>` provides the value assigned to a corresponding field
of the entry stored in `memory <EpisodicMemoryMechanism.memory>`, and used to retrieve one similar to it. By default,
`input_port <EpisodicMemoryMechanism.input_ports>` are named *FIELD_n_INPUT*, where "n" is replaced by the index of
each field; however, they can be named explicitly by specifying a list of strings in the **input_ports** argument of
the constructor; the number of these must equal the number of fields specified in **default_variable** or **size**.

.. _EpisodicMemoryMechanism_Creation_Function_Parameters:

*Function Parameters
~~~~~~~~~~~~~~~~~~~~

Parameters that govern the storage and retrieval process are specified as arguments to the `MemoryFunction` specified
in the **function** of the constructor (for example, see `ContentAddressableMemory` for parameters of the default
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

.. _EpisodicMemoryMechanism_Memory_Fields:

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

.. _EpisodicMemoryMechanism_Input:

*Input*
~~~~~~~

An EpisodicMemoryMechanism has one or more `input_ports <EpisodicMemoryMechanism.input_ports>` that receive the
entry to be stored and that is used to retrieve an existing entry from its memory.  If the Mechanism is assigned
`ContentAddressableMemory` as its `function <EpisodicMemoryMechanism.function>`, then it can have an arbitrary
number of InputPorts, the input to which is assigned to the corresponding `memory field
<EpisodicMemoryMechanism_Memory_Fields>` of that function. By default InputPorts are named *FIELD_n_INPUT* (see
`EpisodicMemoryMechanism_Creation`). If the Mechanism is assigned `DictionaryMemory` as its `function
<EpisodicMemoryMechanism.function>`, then it is assigned at least one InputPort (named *KEY_INPUT* by default),
and optionally a second (named *VALUE_INPUT*) if **default_variable** or **size** specifies two items; any additional
fields are ignored.

.. _EpisodicMemoryMechanism_Function:

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

.. _EpisodicMemoryMechanism_Output:

*Output*
~~~~~~~~

By default, an EpisodicMemoryMechanism has a number of `OutputPorts <OutputPort>` equal to its number of InputPorts,
each of which is assigned the value of a corresponding field of the entry retrieved from memory (see
`EpisodicMemoryMechanism_Creation` for naming). However, if OutputPorts were specified in the constructor, then there
may be a different number of OutputPorts than InputPorts and `memory fields <EpisodicMemoryMechanism_Memory_Fields>`,
and thus some of the latter may not be reflected in any of the Mechanism's `output_ports
<EpisodicMemoryMechanism.output_ports>`. If the `function <EpisodicMemoryMechanism.function>` is a `DictionaryMemory`,
then it will have at least one OutputPort, named *KEY_OUTPUT*, that is assigned the key (first field) of the entry
retrieved from memory and, if two fields are specified in **default_variable** or **sze**, the Mechanism will have a
second OutputPort named *VALUE_OUTPUT* that is assigned the value (second field) of the entry retrieved from memory;
any additional ones are ignored and no other OutputPorts are created.

.. _EpisodicMemoryMechanism_Execution:

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieve an item from `memory <EpisodicMemoryMechanism.memory>` based on the `value <InputPort.value>` of
      its `input_ports <EpisodicMemoryMechanism.input_ports>`; if no retrieval is made, then an appropriately
      shaped zero-valued array is returned.
    ..
    * store the `value <InputPort.value>` of its `input_ports <EpisodicMemoryMechanism.input_ports>` as an entry in
      <EpisodicMemoryMechanism.function>` memory.
    ..
    * assign the value of the entry retrieved to its `output_ports <EpisodicMemoryMechanism.output_ports>`,
      based on how the latter are configured (see `EpisodicMemoryMechanism_Creation_OutputPorts`).

    .. note::
         In general, retrieval is executed before storage, so that the current items is not also retrieved;
         however, the order of storage and retrieval is determined by the EpisodicMemoryMechanism's
         `function <EpisodicMemoryMechanism.function>`.

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""
from typing import Optional, Union

import warnings
import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import \
    DictionaryMemory, ContentAddressableMemory
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.globals.keywords import EPISODIC_MEMORY_MECHANISM, INITIALIZER, NAME, OWNER_VALUE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import deprecation_warning, convert_to_np_array, convert_all_elements_to_np_array

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


class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism()

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that implements a content addressable dictionary.
    See `Mechanism <Mechanism_Class_Reference>` for arguments and attributes.

    Arguments
    ---------

    memory : list or ndarray
        initial set of entries for `memory <EpisodicMemory.memory>`.  It should be either a 3d regular
        array or a 2d ragged array if the fields of an entry have different lengths; assigned as the
        **initializer** argument of the constructor for the `MemoryFunction` specified in **function**
        (see `initializer <ContentAddressableMemory.initializer>` for default assignment).

    Attributes
    ----------

    input_ports : ContentAddressableList[str, InputPort]
        a list of the Mechanism's `InputPorts <Mechanism_InputPorts>`, the number of which is equal to the
        number of `memory fields <EpisodicMemoryMechanism_Memory_Fields>` in an entry of the Mechanism's `memory
        <EpisodicMemoryMechanism.memory>`, and that are named *FIELD_n_INPUT* by default (see
        `EpisodicMemoryMechanism_Input` and Mechanism `input_ports <Mechanism_Base.input_ports>` for additional
        information).

    function : MemoryFunction
        takes the Mechanism's `variable <Mechanism_Base.variable>` and uses it as a cue to retrieve an entry from
        `memory <EpisodicMemoryMechanism_Function>` based on its distance from existing entries, and then stores
        `variable <Mechanism_Base.variable>` in `memory <EpisodicMemoryMechanism_Function>` (see
        `EpisodicMemoryMechanism_Function` and Mechanism `function <Mechanism_Base.function>` for
        additional information).

    memory : 3d array
        contains entries stored in the `function <EpisodicMemoryMechanism.function>'\\s `memory` attribute.

    output_ports : ContentAddressableList[str, OutputPort]
        a list of the Mechanism's `OutputPorts <Mechanism_OutputPorts>`, the number of which is, by default, equal to
        the number of `memory fields <EpisodicMemoryMechanism_Memory_Fields>` in an entry of the Mechanism's `memory
        <EpisodicMemoryMechanism.memory>`, and each of which is assigned the value of the corresponding field of the
        last entry retrieved from `memory <EpisodicMemoryMechanism.memory>`.  However, as with any Mechanism, the
        number and value of OutputPorts can be customized (see `EpisodicMemoryMechanism_Output` and Mechanism
        `output_ports <Mechanism_Base.output_ports>` for additional information).
    """

    componentName = EPISODIC_MEMORY_MECHANISM

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

        """
        variable = Parameter([[0]], pnl_internal=True, constructor_argument='default_variable')
        function = Parameter(ContentAddressableMemory, stateful=False, loggable=False)

    def __init__(self,
                 default_variable:Union[int, list, np.ndarray]=None,
                 size:Optional[Union[int, list, np.ndarray]]=None,
                 memory:Optional[Union[list, np.ndarray]]=None,
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

        self._memory_init = memory

        super().__init__(
            default_variable=default_variable,
            size=size,
            function=function,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

    def _handle_default_variable(self, default_variable=None, size=None, input_ports=None, function=None, params=None):
        """Override to check/specify default_variable with respect to shape of function.memory:
            - if memory argument for Mechanism is specified and default_variable is not, use former to specify latter;
            - if both are specified, validate that they are the same shape;
            - if function.memory is specified and default_variable is not, use former to specify latter;
            - if both are specified, validate that they are the same shape;
            - if default_variable is specified and neither memory arg of Mechanism nor function.memory is specified,
                 use default_variable to specify function.memory.
        Note: handling this here insures that input_ports are specified/validated using correct default_variable
        """
        if self._memory_init:
            if default_variable is None:
                default_variable = self._memory_init[0]
            else:
                variable_shape = convert_all_elements_to_np_array(default_variable).shape
                entry_shape = convert_all_elements_to_np_array(self._memory).shape
                if entry_shape != variable_shape:
                    raise EpisodicMemoryMechanismError(f"Shape of 'variable' for {self.name} ({variable_shape}) "
                                                       f"does not match the shape of entries ({entry_shape}) in "
                                                       f"specificaition of its 'memory' argument.")

        elif isinstance(self.function, Function) and len(self.function.memory):
            if default_variable is None:
                default_variable = self.function.memory[0]
            else:
                variable_shape= convert_all_elements_to_np_array(default_variable).shape
                entry_shape = self.function.memory[0].shape
                if entry_shape != variable_shape:
                    raise EpisodicMemoryMechanismError(f"Shape of 'variable' for {self.name} ({variable_shape}) "
                                                       f"does not match the shape of entries ({entry_shape}) in "
                                                       f"the memory of its function ({self.function.name}).")

        return super()._handle_default_variable(default_variable, size, input_ports, function, params)

    def _instantiate_input_ports(self, context=None):
        """Override to assign default names to input_ports"""

        # IMPLEMENTATION NOTE: REMOVE FIRST CONDITIONAL (LEAVING ELSE CLAUSE) WHEN DictionaryMemory IS RETIRED
        if self._dictionary_memory:
            input_ports = [KEY_INPUT]
            if len(self.parameters.variable.default_value) == 2:
                input_ports.append(VALUE_INPUT)
        else:
            input_ports = self.input_ports or [f'{DEFAULT_INPUT_PORT_NAME_PREFIX}{i}{DEFAULT_INPUT_PORT_NAME_SUFFIX}'
                                               for i in range(len(self.parameters.variable.default_value))]

        super()._instantiate_input_ports(input_ports=input_ports, context=context)

    def _instantiate_function(self, function, function_params, context):
        """Assign memory to function if specified in Mechanism's constructor"""
        if self._memory_init:
            if isinstance(function, type):
                function_params.update({INITIALIZER:self._memory_init})
            else:
                if len(function.memory):
                    warnings.warn(f"The 'memory' argument specified for {self.name} will override the specification "
                                  f"for the {repr(INITIALIZER)} argument of its function ({self.function.name}).")
                function.reset(self._memory_init)
        super()._instantiate_function(function, function_params, context)

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

        # IMPLEMENTATION NOTE: REMOVE FIRST CONDITIONAL (LEAVING ELSE CLAUSE) WHEN DictionaryMemory IS RETIRED
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
