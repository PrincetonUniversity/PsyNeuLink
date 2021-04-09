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
Memories can have an arbitrary number of fields, and each of those can be of arbitrary length, however all memories
for an instance of an EpisodicMemoryMechanism must have the same shape (number of fields, and lengths of corresponding
fields).  Each `InputPort` of an EpisodicMemoryMechanism provides the input to a corresponding field of a memory to
be stored and/or retrieved and, by default, each `OutputPort` contains the value of a field of the last retrieved
memory (though OutputPorts can be configured in other ways).  The full set of stored memories can be accessed from
the Mechanism's `memory <EpisodicMemoryMechanism.memory>` attribute, which references
its `function's <EpisodicMemoryMechanism.function>` memory `Parameter`.  Other Parameters of its function (e.g.,
that regulate the probability of storage and/or retrieval -- see `ContentAddressableMemory`) can be accessed and/or
`modulated <ModulatorySignal_Modulation>` in the standard way for a Mechanism's `function <Mechanism_Base.function>`.

.. _EpisodicMemoryMechanism_Creation:

Creating an EpisodicMemoryMechanism
-----------------------------------

.. _EpisodicMemoryMechanism_Creation_InputPorts:

*InputPorts and Memory Fields*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An EpisodicMemoryMechanism is created by calling its constructor with a specification of its `MemoryFunction` and
corresponding information about the shapes of the items to be stored in memory. The latter are specified using the
**default_variable** or **size** arguments of the constructor, in the standard way that the `variable
<Mechanism_Base.variable>` and InputPorts are specified for a Mechanism (see `default_variable <Component_Variable>`,
`size <Component_Size>`, and `Mechanism Variable <Mechanism_Variable_and_InputPorts>` for additional information).
This are used to construct a 2d array that in turn is used to create the `InputPorts <InputPort>` for the Mechanism,
(with the number of those equal to the number of items in the 2d array).  The InputPorts provide the values that are
assigned to the corresponding fields of the entry stored in memory.  By default, InputPorts are named
*FIELD_n_INPUT*, where "n" is replaced by the index of each field; however, they can be named explicitly by
specifying a list of strings in the **input_ports** argument of the constructor.

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

*Input*
~~~~~~~

An EpisodicMemoryMechanism has one or more `InputPorts <InputPort>` that receive the item to be stored, and that is
used to retrieve an existing entry from its memory.  The number of `InputPorts <InputPort>` equals the number of
items specified in the **default_variable** or **size** argument of the constructor.  If the Mechanism is assigned
`ContentAddressableMemory` as its `function <EpisodicMemoryMechanism.function>`, then it can have an arbitrary number
of InputPorts, the input to which is assigned to the corresponding `memory field
<ContentAddressableMemory_Memory_Fields>` of that function, each named *FIELD_n_INPUT* by default (see
`EpisodicMemoryMechanism_Creation`). If the Mechanism is assigned `DictionaryMemory` as its `function
<EpisodicMemoryMechanism.function>`, then it is assigned at least one InputPort (named *KEY_INPUT* by default),
and optionally a second (named *VALUE_INPUT*) if **default_variable** or **size* specifies two items;
any additional items are ignored.

*Function*
~~~~~~~~~~

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

*Output*
~~~~~~~~

By default, an EpisodicMemoryMechanism has a number of `OutputPorts <OutputPort>` equal to its number of InputPorts,
each of which is assigned the value of a corresponding field in the entry retrieved from memory (see
`EpisodicMemoryMechanism_Creation` for naming). However, if Outports were specified in the constructor, then there
may be a different number of OutputPorts than InputPorts and memory fields, and thus some of the latter may not be
reflected in the output of the Mechanism.  If `function <EpisodicMemoryMechanism.function>` is
`DictionaryMemory`, then it will be assigned at least one OutputPort, named *KEY_OUTPUT* and, if **default_varible**
or **sze** specified two items, a second OutputPort named *VALUE_OUTPUT*; any additional ones are ignored and no
other OutputPorts are created).

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieves an item from `memory <EpisodicMemoryMechanism.memory>` based on the `value <InputPort.value>` of its
    `InputPorts <InputPort>`; if no retrieval is made, then appropriately shaped zero-valued arrays are returned.
    ..
    * stores the `value <InputPort.value>` of its InputPorts as an entry in <EpisodicMemoryMechanism.function>` memory.
    ..
    * assigns the value of the retrieved to OutputPorts, based on how the latter are configured (see
      `EpisodicMemoryMechanism_Creation_OutputPorts`).

    .. note::
         The order of storage and retrieval is determined by `function <EpisodicMemoryMechanism.function>`.

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""
import warnings
from typing import Union

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import \
    DictionaryMemory, ContentAddressableMemory
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.inputport import OutputPort
from psyneulink.core.globals.keywords import NAME, OWNER_VALUE, SIZE, VARIABLE
from psyneulink.core.globals.utilities import deprecation_warning, convert_to_np_array
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set

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

    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a content addressable dictionary.
    See `Mechanism <Mechanism_Class_Reference>` for arguments and attributes.

    COMMENT:
    # OLD
    # Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a `differentiable neural dictionary
    # (DictionaryMemory)<HTML>`_.  See `Mechanism <Mechanism_Class_Reference>` for additional arguments and
    # attributes.
    COMMENT

    Arguments
    ---------

    COMMENT:
    # # content_size : int : default 1
    # #     specifies length of the content stored in the `function <EpisodicMemoryMechanism.function>`\\s memory.
    # #
    # # assoc_size : int : default 0
    # #     specifies length of the assoc stored in the `function <EpisodicMemoryMechanism.function>`\\s memory;
    # #     if it is 0 (the default) then no *VALUE_INPUT* InputPort or *VALUE_OUTPUT* OutputPort are created.
    #
    # field_sizes : list[int] or 1d array : default [1]
    #     specifies the size of each field in the input, each of which corresponds an `InputPort` of the Mechanism
    #     (alias for the standard `size <Component_Size>` argument of a Component's constructor).
    #     For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
    #     For a `DicionaryMemory` function, there can be one or two elements, the first of which specifies the size
    #     of the *CONTENT* vector and second, if present, specifies the size of the *ASSOC*  vector;  any additional
    #     elements in **field_sizes** are ignored.
    #
    # field_names : list[str] : default None
    #     specifies the name of the `InputPort` used for each field in the input (alias for the standard `input_ports
    #     <Mechanism_InputPorts>` argument of a `Mechanism's <Mechanism>` consrtuctor).
    #     For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
    #     For a `DicionaryMemory` function, there can be one or two elements, the first of which specifies the size
    #
    #
    #     For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
    #     For a `ContentAddressableMemory` function, there can any number of fields, each of which can be any size.
    #
    # FIX: NOTE: PUT WARNING HERE ABOUT FIELDS WITH SIZE 1 PRODUCING PARTICULAR (POTENTIALLY UNANTICIPATED) RESULTS
    # WITH SOME DISTANCE METRICS (SCALRS & EUCLIDEAN MEASURES = 0)
    #
    # function : function : default DictionaryMemory
    #     specifies the function that implements a memory store and methods to store to and retrieve from it.  It
    #     must take as its `variable <Function.variable>` a 2d array, the first item of which is the content and the second
    #     the associate to be stored in its memory, and must return a 2d array that is the value of the
    #     content and assoc retrieved from its memory.
    COMMENT

    Attributes
    ----------

    COMMENT:
    # function : function
    #     function that implements storage and retrieval from a memory.
    COMMENT

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
                 size:Union[int, list, np.ndarray]=None,
                 function:Function=ContentAddressableMemory,
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
        """Generate OutputPorts with names specified and values with lengths equal to corresponding InputPorts

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
                    # Otherweise, use InputPort names as base, removing DEFAULT_INPUT_PORT_NAME_SUFFIX
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

        # # FIX: VERSION THAT ENFORCES NUMBER OF OutputPorts EQUAL TO NUMBER OF InputPorts (i.e., ALL MEMORY FIELDS)
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

    # FIX: REMOVE THIS METHOD WHEN DictionaryMemory IS RETIRED
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

    # FIX: REMOVE THIS METHOD WHEN DictionaryMemory IS RETIRED
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
