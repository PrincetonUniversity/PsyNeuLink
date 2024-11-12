# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  EMStorageMechanism **********************************************

"""

Contents
--------

  * `EMStorageMechanism_Overview`
    - `EMStorageMechanism_Memory`
    - `EMStorageMechanism_Entry`
    - `EMStorageMechanism_Fields`
  * `EMStorageMechanism_Creation`
  * `EMStorageMechanism_Structure`
  * `EMStorageMechanism_Execution`
  * `EMStorageMechanism_Class_Reference`


.. _EMStorageMechanism_Overview:

Overview
--------

An EMStorageMechanism is a subclass of `LearningMechanism`, modified for use in an `EMComposition` to store a new
entry in its `memory <EMComposition.memory>` attribute each time it executes.

.. _EMStorageMechanism_Memory:

# FIX: NEEDS EDITING:

* **Memory** -- the `memory <EMComposition.memory>` attribute of an `EMComposition` is a list of entries, each of
    which is a 2d np.array with a shape that corresponds to the `memory_matrix <EMStorageMechanism.memory_matrix>`
    attribute of the EMStorageMechanism that stores it.  Each entry is stored in the `memory <EMComposition.memory>`
    attribute of the EMComposition as a row or column of the `matrix <MappingProjection.matrix>` parameter of the
    `MappingProjections <MappingProjection>` to which the `LearningProjections <LearningProjection>` of the
    EMStorageMechanism project.  The `memory <EMComposition.memory>` attribute of the EMComposition is used by its
    `controller <EMComposition.controller>` to generate the `memory <EMMemoryMechanism.memory>` attribute of an
    `EMMemoryMechanism` that is used to retrieve entries from the `memory <EMComposition.memory>` attribute of the
    EMComposition.

.. _EMStorageMechanism_Entry:

* **Entry** -- an entry is a 2d np.array with a shape that corresponds to the `memory_matrix
    <EMStorageMechanism.memory_matrix>` attribute of the EMStorageMechanism that stores it.  Each entry is stored in the
    `memory <EMComposition.memory>` attribute of the EMComposition as a row or column of the `matrix
    <MappingProjection.matrix>` parameter of the `MappingProjections <MappingProjection>` to which the
    `LearningProjections <LearningProjection>` of the EMStorageMechanism project.  The `memory
    <EMComposition.memory>` attribute of the EMComposition is used by its `controller <EMComposition.controller>` to
    generate the `memory <EMMemoryMechanism.memory>` attribute of an `EMMemoryMechanism` that is used to retrieve
    entries from the `memory <EMComposition.memory>` attribute of the EMComposition.

.. _EMStorageMechanism_Fields:

* **Fields** -- an entry is composed of one or more fields, each of which is a 1d np.array with a length that
    corresponds to the number of `fields <EMStorageMechanism_Fields>` of the EMStorageMechanism that stores it.  Each
    field is stored in the `memory <EMComposition.memory>` attribute of the EMComposition as a row or column of the
    `matrix <MappingProjection.matrix>` parameter of the `MappingProjections <MappingProjection>` to which the
    `LearningProjections <LearningProjection>` of the EMStorageMechanism project.  The `memory
    <EMComposition.memory>` attribute of the EMComposition is used by its `controller <EMComposition.controller>` to
    generate the `memory <EMMemoryMechanism.memory>` attribute of an `EMMemoryMechanism` that is used to retrieve
    entries from the `memory <EMComposition.memory>` attribute of the EMComposition.

.. _EMStorageMechanism_Creation:

Creating an EMStorageMechanism
--------------------------------------------

An EMStorageMechanism can be created directly by calling its constructor, but most commonly it is created
automatically when an `EMComposition` is created, as its `learning_mechanism <EMComposition.learning_mechanism>`
used to store entries in its `memory <EMComposition.memory>` of the EMComposition. The `memory_matrix` must be
specified (as a template for the shape of the entries to be stored, and of the `matrix <MappingProjection.matrix>`
parameters to which they are assigned. It must also have at least one, and usually several `fields
<EMStorageMechanism.fields>` specifications that identify the `OutputPort`\\s of the `ProcessingMechanism`\\s from
which it receives its `fields <EMStorageMechanism_Fields>`, and a `field_types <EMStorageMechanism.field_types>`
specification that indicates whether each `field is a key or a value field <EMStorageMechanism_Fields>`.

.. _EMStorageMechanism_Structure:

Structure
---------

An EMStorageMechanism differs from a standard `LearningMechanism` in the following ways:

  * it has no `input_source <LearningMechanism.input_source>`, `output_source <LearningMechanism.output_source>`,
    or `error_source <LearningMechanism.error_source>` attributes;  instead, it has the `fields
    <EMStorageMechanism.fields>` and `field_types <EMStorageMechanism.field_types>` attributes described below.

  * its `fields <EMStorageMechanism.fields>` attribute has as many *FIELDS* `field <EMStorage_mechanism.fields>`
    as there are `fields <EMStorageMechanism_Fields>` of an entry in its `memory_matrix
    <EMStorageMechanism.memory_matrix>` attribute;  these are listed in its `fields <EMStorageMechanism.fields>`
    attribute and serve as the `InputPort`\\s for the EMStorageMechanism;  each receives a `MappingProjection` from
    the `OutputPort` of a `ProcessingMechanism`, the activity of which constitutes the corresponding `field
    <EMStorageMechanism_Fields>` of the `entry <EMStorageMechanism_Entry>` to be stored in its `memory_matrix
    <EMStorageMechanism.memory_matrix>` attribute.

  * it has a `field_types <EMStorageMechanism.field_types>` attribute that specifies whether each `field
    <EMStorageMechanism_Fields>` is a `key or a value field <EMStorageMechanism_Fields>`.

  * it has a `field_weights <EMStorageMechanism.field_weights>` attribute that specifies whether each `field
    <EMStorageMechanism_Fields>` each norms for each field are weighted before deteterming the weakest `entry
    <EMStorageMechanism_Entry>` in `memory_matrix <EMStorageMechanism.memory_matrix>`.

  * it has a `memory_matrix <EMStorageMechanism.memory_matrix>` attribute that represents the full memory that the
    EMStorageMechanism is used to update.

  * it has a `concatenation_node <EMStorageMechanism.concatenation_node>` attribute used to access the concatenated
    inputs to the `key <EMStorageMechanism.key>` fields of the `entry <EMStorageMechanism_Entry>` to be stored in its
    `memory_matrix <EMStorageMechanism.memory_matrix>` attribute.

  * it has a several *LEARNING_SIGNAL* `OutputPorts <OutputPort>` that each send a `LearningProjection` to the `matrix
    <MappingProjection.matrix>` parameter of a 'MappingProjection` that constitutes a `field <EMStorageMechanism_Fields>`
    of the `memory_matrix <EMStorageMechanism.memory_matrix>` attribute.

  * its `function <EMStorageMechanism.function>` is an `EMStorage` `LearningFunction`, that takes as its `variable
    <Function_Base.variable>` a list or 1d np.array with a length of the corresponding  *ACTIVATION_INPUT* InputPort;
    and it returns a `learning_signal <LearningMechanism.learning_signal>` (a weight matrix assigned to one of the
    Mechanism's *LEARNING_SIGNAL* OutputPorts), but no `error_signal <LearningMechanism.error_signal>`.

  * the default form of `modulation <ModulatorySignal_Modulation>` for its `learning_signals
    <LearningMechanism.learning_signals>` is *OVERRIDE*, so that the `matrix <MappingProjection.matrix>` parameter of
    the `MappingProjection` to which the `LearningProjection` projects is replaced by the `value
    <LearningProjection.value>` of the `learning_signal <LearningMechanism.learning_signal>`.

  * its `decay_rate <EMStorageMechanism.decay_rate>`, a float in the interval [0,1] that is used to decay
    `memory_matrix <EMStorageMechanism.memory_matrix>` before an `entry <EMStorageMechanism_Entry>` is stored.

  * its `storage_prob <EMStorageMechanism.storage_prob>`, a float in the interval [0,1] is used in place of a
    LearningMechanism's `storage_prob <LearningMechanism.storage_prob>` to determine the probability that the
    Mechanism will store its `variable <EMStorageMechanism.variable>` in its `memory_matrix
    <EMStorageMechanism.memory_matrix>` attribute each time it executes.

.. _EMStorageMechanism_Execution:

Execution
---------

An EMStorageMechanism executes after all of the other Mechanisms in the `EMComposition` to which it belongs have
executed.  It executes in the same manner as standard `LearningMechanism`, however instead of modulating
the `matrix <MappingProjection.matrix>` Parameter of a `MappingProjection`, it replaces a row or column in each of
the `matrix <MappingProjection.matrix>` Parameters of the `MappingProjections <MappingProjection>` to which its
`LearningProjections <LearningProjection>` project with an item of its `variable <EMStorageMechanism.variable>` that
represents the corresponding `field <EMStorageMechanism.fields>`. The entry replaced is the one that has the lowest
norm computed across all `fields <EMSorageMechanism_Fields>` of the `entry <EMStorageMechanism_Entry>` weighted by the
corresponding items of `field_weights <EMStorageMechanism.field_weights>` if that is specified.


.. _EMStorageMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import re
from beartype import beartype

from psyneulink._typing import Optional, Union, Callable, Literal

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.nonstateful.learningfunctions import EMStorage
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    LearningMechanism, LearningMechanismError, LearningTiming, LearningType
from psyneulink.core.components.projections.projection import Projection, projection_keywords
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    (ADDITIVE, EM_STORAGE_MECHANISM, LEARNING, LEARNING_PROJECTION, LEARNING_SIGNALS, MULTIPLICATIVE,
     MULTIPLICATIVE_PARAM, MODULATION, NAME, OVERRIDE, OWNER_VALUE, PROJECTIONS, REFERENCE_VALUE, VARIABLE)
from psyneulink.core.globals.parameters import Parameter, check_user_specified, FunctionParameter, copy_parameter_value
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, is_numeric, all_within_range

__all__ = [
    'EMStorageMechanism', 'EMStorageMechanismError',
]

# Parameters:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

MEMORY_MATRIX = 'memory_matrix'
FIELDS = 'fields'
FIELD_TYPES = 'field_types'

def _memory_matrix_getter(owning_component=None, context=None)->list:
    """Return list of memories in which rows (outer dimension) are memories for each field.
    These are derived from `matrix <MappingProjection.matrix>` parameter of the `afferent
    <Mechanism_Base.afferents>` MappingProjections to each of the `retrieved_nodes <EMComposition.retrieved_nodes>`.
    """
    if owning_component.is_initializing:
        if owning_component.learning_signals is None or owning_component.input_ports is None:
            return None

    num_fields = len(owning_component.input_ports)

    # Get learning_signals that project to retrieved_nodes
    num_learning_signals = len(owning_component.learning_signals)
    learning_signals_for_retrieved = owning_component.learning_signals[num_learning_signals - num_fields:]

    # Get memory from learning_signals that project to retrieved_nodes
    if owning_component.is_initializing:
        # If initializing, learning_signals are still MappingProjections used to specify them, so get from them
        memory = [retrieved_learning_signal.parameters.matrix._get(context)
                  for retrieved_learning_signal in learning_signals_for_retrieved]
    else:
        # Otherwise, get directly from the learning_signals
        memory = [retrieved_learning_signal.efferents[0].receiver.owner.parameters.matrix._get(context)
                  for retrieved_learning_signal in learning_signals_for_retrieved]

    # Get memory capacity from first length of first matrix (can use full set since might be ragged array)
    memory_capacity = len(memory[0])

    # Reorganize memory so that each row is an entry and each column is a field
    return convert_all_elements_to_np_array([
        [memory[j][i] for j in range(num_fields)]
        for i in range(memory_capacity)
    ])


class EMStorageMechanismError(LearningMechanismError):
    pass


class EMStorageMechanism(LearningMechanism):
    """
    EMStorageMechanism(                       \
        variable,                             \
        fields,                               \
        field_types,                          \
        memory_matrix,                        \
        function=EMStorage,                   \
        storage_prob=1.0,                     \
        decay_rate=0.0,                       \
        learning_signals,                     \
        modulation=OVERRIDE,                  \
        params=None,                          \
        name=None,                            \
        prefs=None)

    Implements a `LearningMechanism` that modifies the `matrix <MappingProjection.matrix>` parameters of
    `MappingProjections <MappingProjection>` that implement its `memory_matrix <EMStorageMechanism.memory_matrix>`.

    Arguments
    ---------

    variable : List or 2d np.array : default None
        each item of the 2d array specifies the shape of the corresponding `field <EMStorageMechanism_Fields>` of
        an `entry <EMStorageMechanism_Entry>`, that must be compatible (in number and type) with the `value
        <InputPort.value>` of the corresponding item of its `fields <EMStorageMechanism.fields>`
        attribute (see `variable <EMStorageMechanism.variable>` for additional details).

    fields : List[OutputPort, Mechanism, Projection, tuple[str, Mechanism, Projection] or dict] : default None
        specifies the `OutputPort`\\(s), the `value <OutputPort.value>`\\s of which are used as the
        corresponding `fields <EMStorageMechanism_Fields>` of the `memory_matrix <EMStorageMechanism.memory_matrix>`;
        used to construct the Mechanism's `InputPorts <InputPort>`; must be the same length as `variable
        <EMStorageMechanism.variable>`.

    field_types : List[int] : default None
        specifies whether each item of `variable <EMStorageMechanism.variable>` corresponds to a `key or value field
        <EMStorageMechanism_Fields>` (see `field_types <EMStorageMechanism.field_types>` for additional details);
        must contain only 1's (for keys) and 0's (for values), with the same number of these as there are items in
        the `variable <EMStorageMechanism.variable>` and `fields <EMStorageMechanism.fields>` arguments.

    field_weights : List[float] : default None
        specifies whether norms for each field are weighted before determining the weakest `entry
        <EMStorageMechanism_Entry>` in `memory_matrix <EMStorageMechanism.memory_matrix>`. If None (the default),
        the norm of each `entry <EMStorageMechanism_Entry>` is calculated across all fields at once; if specified,
        it must contain only floats from 0 to 1, and be the same length as the `fields <EMStorageMechanism.fields>`
        argument (see `field_weights <EMStorageMechanism.field_types>` for additional details).

    concatenation_node : OutputPort or Mechanism : default None
        specifies the `OutputPort` or `Mechanism` in which the `value <OutputPort.value>` of the `key fields
        <EMStorageMechanism_Fields>` are concatenated (see `concatenate keys <EMComposition_Concatenate_Queries>`
        for additional details).

    memory_matrix : List or 2d np.array : default None
        specifies the shape of the `memory <EMStorageMechanism_Memory>` used to store an `entry
        <EMStorageMechanism_Entry>` (see `memory_matrix <EMStorageMechanism.memory_matrix>` for additional details).

    function : LearningFunction or function : default EMStorage
        specifies the function used to assign each item of the `variable <EMStorageMechanism.variable>` to the
        corresponding `field <EMStorageMechanism_Fields>` of the `memory_matrix <EMStorageMechanism.memory_matrix>`.
        It must take as its `variable <EMStorage.variable>` argument a list or 1d array of numeric values
        (the "activity vector"); a ``memory_matrix`` argument that is a 2d array to which
        the `variable <EMStorageMechanism.variable>` is assigned; ``axis`` and ``storage_location`` arguments that
        determine where in ``memory_matrix`` the `variable <EMStorageMechanism.variable>` is stored; and optional
        ``storage_prob`` and ``decay_rate`` arguments that determine the probability with which storage occurs and
        the rate at which the `memory_matrix <EMStorageMechanism.memory_matrix>` decays, respectively.  The function
        must return a list, 2d np.array for the corresponding `field <EMStorageMechanism_Fields>` of the
        `memory_matrix <EMStorageMechanism.memory_matrix>` that is updated (see `EMStorage` for additional details).

    learning_signals : List[ParameterPort, Projection, tuple[str, Projection] or dict] : default None
        specifies the `ParameterPort`\\(s) for the `matrix <MappingProjection.matrix>` parameter of the
        `MappingProjection>`\\s that implement the `memory <EMStorageMechanism_Memory>` in which the `entry
        <EMStorageMechanism_Entry>` is stored; there must the same number of these as `fields
        <EMStorageMechanism.fields>`, and they must be specified in the sqme order.

    modulation : str : default OVERRIDE
        specifies form of `modulation <ModulatorySignal_Modulation>` that `learning_signals
        <EMStorageMechanism.learning_signals>` use to modify the `matrix <MappingProjection.matrix>` parameter of the
        `MappingProjections <MappingProjection>` that implement the `memory <EMStorageMechanism_Memory>` in which
        `entries <EMStorageMechanism_Entry>` is stored (see `modulation <EMStorageMechanism_Modulation>` for additional
        details).

    storage_prob : float : default None
        specifies the probability with which the current entry is stored in the EMSorageMechanism's `memory_matrix
        <EMStorageMechanism.memory_matrix>` (see `storage_prob <EMStorageMechanism.storage_prob>` for details).

    decay_rate : float : default 0.0
        specifies the rate at which `entries <EMStorageMechanism_Entry>` in the `memory_matrix
        <EMStorageMechanism.memory_matrix>` decay (see `decay_rate <EMStorageMechanism.decay_rate>` for additional
        details).

    Attributes
    ----------

    # FIX: FINISH EDITING:

    variable : 2d np.array

        each item of the 2d array is used as a template for the shape of each the `fields
        <EMStorageMechanism_Fields>` that  comprise and `entry <EMStorageMechanism_Entry>` in the `memory_matrix
        <EMStorageMechanism.memory_matrix>`, and that must be compatible (in number and type) with the `value
        <OutputPort.value>` of the item specified the corresponding itme of its `fields <EMStorageMechanism.fields>`
        attribute. The values of the `variable <EMStorageMechanism.variable>` are assigned to the `memory_matrix
        <EMStorageMechanism.memory_matrix>` by the `function <EMStorageMechanism.function>`.

    fields : List[OutputPort, Mechanism, Projection, tuple[str, Mechanism, Projection] or dict] : default None
        the `OutputPort`\\(s) used to get the value for each `field <EMStorageMechanism_Fields>` of
        an `entry <EMStorageMechanism_Entry>` of the `memory_matrix <EMStorageMechanism.memory_matrix>` attribute.

    field_types : List[int or tuple[slice]]
        contains a list of indicators of whether each item of `variable <EMStorageMechanism.variable>`
        and the corresponding `fields <EMStorageMechanism.fields>` are key (1) or value (0) fields.
        (see `fields <EMStorageMechanism_Fields>` for additional details).

    field_weights : List[float] or None
        determines whether norms for each field are weighted before identifying the weakest `entry
        <EMStorageMechanism_Entry>` in `memory_matrix <EMStorageMechanism.memory_matrix>`. If is None (the default),
        the norm of each `entry <EMStorageMechanism_Entry>` is calculated across all fields at once; if specified,
        it must contain only floats from 0 to 1, and be the same length as the `fields <EMStorageMechanism.fields>`
        argument (see `field_weights <EMStorageMechanism.field_types>` for additional details).

    learned_projections : List[MappingProjection]
        list of the `MappingProjections <MappingProjection>`, the `matrix <MappingProjection.matrix>` Parameters of
        which are modified by the EMStorageMechanism.

    function : LearningFunction or function : default EMStorage
        the function used to assign the value of each `field <EMStorageMechanism.fields>` to the corresponding entry
        in `memory_matrix <EMStorageMechanism.memory_matrix>`.  It must take as its `variable <EMSorage.variable>`
        argument a list or 1d array of numeric values (an `entry
        <EMStorage.entry`) and return a list, 2d np.array assigned to
        the corresponding `field <EMStorageMechanism_Fields>` of the
        `memory_matrix <EMStorageMechanism.memory_matrix>`.

    storage_prob : float
        specifies the probability with which the current entry is stored in the EMSorageMechanism's `memory_matrix
        <EMStorageMechanism.memory_matrix>`.

    decay_rate : float : default 0.0
        determines the rate at which `entries <EMStorageMechanism_Entry>` in the `memory_matrix
        <EMStorageMechanism.memory_matrix>` decay;  the decay rate is applied to `memory_matrix
        <EMStorageMechanism.memory_matrix>` before it is updated with the new `entry <EMStorageMechanism_Entry>`.

    learning_signals : List[LearningSignal]
        list of all of the `LearningSignals <LearningSignal>` for the EMStorageMechanism, each of which
        sends a `LearningProjection` to the `ParameterPort`\\(s) for the `MappingProjections
        <MappingProjection>` that implement the `memory <EMStorageMechanism_Memory>` in which the `entry
        <EMStorageMechanism_Entry>` is stored.  The `value <LearningSignal.value>` of each LearningSignal is
        used by its `LearningProjection` to modify the `matrix <MappingProjection.matrix>` parameter of the
        MappingProjection to which that projects.

    learning_projections : List[LearningProjection]
        list of all of the LearningProjections <LearningProjection>` from the EMStorageMechanism, listed
        in the order of the `LearningSignals <LearningSignal>` to which they belong (that is, in the order they are
        listed in the `learning_signals <EMStorageMechanism.learning_signals>` attribute).

    modulation : str
        determines form of `modulation <ModulatorySignal_Modulation>` that `learning_signals
        <EMStorageMechanism.learning_signals>` use to modify the `matrix <MappingProjection.matrix>` parameter of the
        `MappingProjections <MappingProjection>` that implement the `memory <EMStorageMechanism_Memory>` in which
        `entries <EMStorageMechanism_Entry>` is stored.  *OVERRIDE* (the default) insure that entries are stored
        exactly as specified by the `value <OutputPort.value>` of the `fields <EMStorageMechanism.fields>` of the
        `entry <EMStorageMechanism_Entry>`;  other values can have unpredictable consequences
        (see `ModulatorySignal_Types for additional details)

    output_ports : ContentAddressableList[OutputPort]
        list of the EMStorageMechanism's `OutputPorts <OutputPort>`, beginning with its
        `learning_signals <EMStorageMechanism.learning_signals>`, and followed by any additional
        (user-specified) `OutputPorts <OutputPort>`.

    output_values : 2d np.array
        the first items are the `value <OutputPort.value>`\\(s) of the LearningMechanism's `learning_signal
        <EMStorageMechanism.learning_signal>`\\(s), followed by the `value <OutputPort.value>`(s)
        of any additional (user-specified) OutputPorts.

    """

    componentType = EM_STORAGE_MECHANISM
    className = componentType
    suffix = " " + className

    class Parameters(LearningMechanism.Parameters):
        """
            Attributes
            ----------

                concatenation_node
                    see `concatenation_node <EMStorageMechanism.concatenation_node>`

                    :default value: None
                    :type: ``Mechanism or OutputPort``
                    :read only: True

                decay_rate
                    see `decay_rate <EMStorageMechanism.decay_rate>`

                    :default value: 0.0
                    :type: ``float``

                fields
                    see `fields <EMStorageMechanism.fields>`

                    :default value: None
                    :type: ``list``
                    :read only: True

                field_types
                    see `field_types <EMStorageMechanism.field_types>`

                    :default value: None
                    :type: ``list``
                    :read only: True

                field_weights
                    see `field_weights <EMStorageMechanism.field_weights>`

                    :default value: None
                    :type: ``list or np.ndarray``

                memory_matrix
                    see `memory_matrix <EMStorageMechanism.memory_matrix>`

                    :default value: None
                    :type: ``np.ndarray``
                    :read only: True

                function
                    see `function <EMStorageMechanism.function>`

                    :default value: `EMStorage`
                    :type: `Function`

                input_ports
                    see `fields <EMStorageMechanism.fields>`

                    :default value: None
                    :type: ``list``
                    :read only: True

                learning_signals
                    see `learning_signals <EMStorageMechanism.learning_signals>`

                    :default value: []
                    :type: ``List[MappingProjection or ParameterPort]``
                    :read only: True

                modulation
                    see `modulation <EMStorageMechanism.modulation>`

                    :default value: OVERRIDE
                    :type: ModulationParam
                    :read only: True

                output_ports
                    see `learning_signals <EMStorageMechanism.learning_signals>`

                    :default value: None
                    :type: ``list``
                    :read only: True

                storage_prob
                    see `storage_prob <EMStorageMechanism.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

        """
        input_ports = Parameter([], # FIX: SHOULD BE ABLE TO UE THIS WITH 'fields' AS CONSTRUCTOR ARGUMENT
                                stateful=False,
                                loggable=False,
                                read_only=True,
                                structural=True,
                                parse_spec=True,
                                constructor_argument='fields',
                                )
        fields = Parameter(
            [], stateful=False, loggable=False, read_only=True, structural=True
        )
        field_types = Parameter([],stateful=False,
                                loggable=False,
                                read_only=True,
                                structural=True,
                                parse_spec=True,
                                dependiencies='fields')
        field_weights = Parameter(None,
                                  modulable=True,
                                  stateful=True,
                                  loggable=True,
                                  dependiencies='fields')
        concatenation_node = Parameter(None,
                                       stateful=False,
                                       loggable=False,
                                       read_only=True,
                                       structural=True)
        function = Parameter(EMStorage, stateful=False, loggable=False)
        # storage_prob = Parameter(1.0, modulable=True, stateful=True)
        storage_prob = FunctionParameter(1.0,
                                         function_name='function',
                                         function_parameter_name='storage_prob',
                                         primary=True,
                                         modulable=True,
                                         aliases=[MULTIPLICATIVE_PARAM],
                                         stateful=True)
        decay_rate = Parameter(0.0, modulable=True, stateful=True)
        memory_matrix = Parameter(None, getter=_memory_matrix_getter, read_only=True, structural=True)
        modulation = OVERRIDE
        output_ports = Parameter([],
                                 stateful=False,
                                 loggable=False,
                                 read_only=True,
                                 structural=True,
                                 # constructor_argument='learning_signals'
                                 )
        learning_signals = Parameter([],
                                     stateful=False,
                                     loggable=False,
                                     read_only=True,
                                     structural=True)
        learning_type = LearningType.UNSUPERVISED
        # learning_type = LearningType.SUPERVISED
        # learning_timing = LearningTiming.LEARNING_PHASE
        learning_timing = LearningTiming.EXECUTION_PHASE

    def _validate_field_types(self, field_types):
        if not len(field_types) or len(field_types) != len(self.input_ports):
            return f"must be specified with a number of items equal to " \
                   f"the number of fields specified {len(self.input_ports)}"
        if not all(item in {1,0} for item in field_types):
            return f"must be a list of 1s (for keys) and 0s (for values)."

    def _validate_field_weights(self, field_weights):
        if not field_weights or len(field_weights) != len(self.input_ports):
            return f"must be specified with a number of items equal to " \
                   f"the number of fields specified {len(self.input_ports)}"
        if not all(isinstance(item, (int, float)) and (0 <= item  <= 1) for item in field_weights):
            return f"must be a list floats from 0 to 1."

    def _validate_storage_prob(self, storage_prob):
        storage_prob = float(storage_prob)
        if not all_within_range(storage_prob, 0, 1):
            return f"must be a float in the interval [0,1]."

    def _validate_decay_rate(self, decay_rate):
        decay_rate = float(decay_rate)
        if not all_within_range(decay_rate, 0, 1):
            return f"must be a float in the interval [0,1]."


    classPreferenceLevel = PreferenceLevel.TYPE

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable: Union[list, np.ndarray],
                 fields: Union[list, tuple, dict, OutputPort, Mechanism, Projection] = None,
                 field_types: list = None,
                 field_weights: Optional[Union[list, np.ndarray]] = None,
                 concatenation_node: Optional[Union[OutputPort, Mechanism]] = None,
                 memory_matrix: Union[list, np.ndarray] = None,
                 function: Optional[Callable] = EMStorage,
                 learning_signals: Union[list, dict, ParameterPort, Projection, tuple] = None,
                 modulation: Optional[Literal[OVERRIDE, ADDITIVE, MULTIPLICATIVE]] = OVERRIDE,
                 decay_rate: Optional[Union[int, float, np.ndarray]] = 0.0,
                 storage_prob: Optional[Union[int, float, np.ndarray]] = 1.0,
                 params=None,
                 name=None,
                 prefs: Optional[ValidPrefSet] = None,
                 **kwargs
                 ):

        super().__init__(default_variable=default_variable,
                         fields=fields,
                         field_types=field_types,
                         concatenation_node=concatenation_node,
                         memory_matrix=memory_matrix,
                         function=function,
                         learning_signals=learning_signals,
                         modulation=modulation,
                         decay_rate=decay_rate,
                         storage_prob=storage_prob,
                         field_weights=field_weights,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

    def _validate_variable(self, variable, context=None):
        """Validate that variable has only one item: activation_input.
        """

        # Skip LearningMechanism._validate_variable in call to super(), as it requires variable to have 3 items
        variable = super(LearningMechanism, self)._validate_variable(variable, context)

        # Items in variable should be 1d and have numeric values
        if not (all(np.array(variable)[i].ndim == 1 for i in range(len(variable))) and is_numeric(variable)):
            raise EMStorageMechanismError(f"Variable for {self.name} ({variable}) must be "
                                          f"a list or 2d np.array containing 1d arrays with only numbers.")
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate relationship of matrix, fields and field_types arguments"""

        # Ensure that the shape of variable is equivalent to an entry in memory_matrix
        if MEMORY_MATRIX in request_set:
            memory_matrix = request_set[MEMORY_MATRIX]
            # Items in variable should have the same shape as memory_matrix
            if memory_matrix[0].shape != np.array(self.variable).shape:
                raise EMStorageMechanismError(f"The 'variable' arg for {self.name} ({self.variable}) must be "
                                              f"a list or 2d np.array containing entries that have the same shape "
                                              f"({memory_matrix.shape}) as an entry (row) in 'memory_matrix' arg.")

        # Ensure the number of fields is equal to the numbder of items in variable
        if FIELDS in request_set:
            fields = request_set[FIELDS]
            if len(fields) != len(self.variable):
                raise EMStorageMechanismError(f"The 'fields' arg for {self.name} ({fields}) must have the same "
                                              f"number of items as its variable arg ({len(self.variable)}).")

        # Ensure the number of field_types is equal to the number of fields
        if FIELD_TYPES in request_set:
            field_types = request_set[FIELD_TYPES]
            if len(field_types) != len(fields):
                raise EMStorageMechanismError(f"The 'field_types' arg for {self.name} ({field_types}) must have "
                                              f"the same number of items as its 'fields' arg ({len(fields)}).")

        num_keys = len([i for i in field_types if i==1])
        concatenate_queries = 'concatenation_node' in request_set and request_set['concatenation_node'] is not None

        # Ensure the number of learning_signals is equal to the number of fields + number of keys
        if LEARNING_SIGNALS in request_set:
            learning_signals = request_set[LEARNING_SIGNALS]
            if concatenate_queries:
                num_match_fields = 1
            else:
                num_match_fields = num_keys
            if len(learning_signals) != num_match_fields + len(fields):
                raise EMStorageMechanismError(f"The number of 'learning_signals' ({len(learning_signals)}) specified "
                                              f"for  {self.name} must be the same as the number of items "
                                              f"in its variable ({len(self.variable)}).")

        # Ensure shape of learning_signals matches shapes of matrices for match nodes (i.e., either keys or concatenate)
        for i, learning_signal in enumerate(learning_signals[:num_match_fields]):
            learning_signal_shape = learning_signal.parameters.matrix._get(context).shape
            if concatenate_queries:
                memory_matrix_field_shape = np.array([np.concatenate(row, dtype=object).flatten()
                                                      for row in memory_matrix[:,0:num_keys]]).T.shape
            else:
                memory_matrix_field_shape = np.array(memory_matrix[:,i].tolist()).T.shape
            assert learning_signal_shape == memory_matrix_field_shape, \
                f"The shape ({learning_signal_shape}) of the matrix for the Projection {learning_signal.name} " \
                f"used to specify learning signal {i} of {self.name} does not match the shape " \
                f"of the corresponding field {i} of its 'memory_matrix' {memory_matrix_field_shape})."
        # Ensure shape of learning_signals matches shapes of matrices for retrieval nodes (i.e., all input fields)
        for i, learning_signal in enumerate(learning_signals[num_match_fields:]):
            learning_signal_shape = learning_signal.parameters.matrix._get(context).shape
            memory_matrix_field_shape = np.array(memory_matrix[:,i].tolist()).shape
            assert learning_signal_shape == memory_matrix_field_shape, \
                f"The shape ({learning_signal.shape}) of the matrix for the Projection {learning_signal.name} " \
                f"used to specify learning signal {i} of {self.name} does not match the shape " \
                f"of the corresponding field {i} of its 'memory_matrix' {memory_matrix.shape})."

    def _instantiate_input_ports(self, input_ports=None, reference_value=None, context=None):
        """Override LearningMechanism to instantiate an InputPort for each field"""
        input_ports = [{NAME: f"QUERY_INPUT_{i}" if self.field_types[i] == 1 else f"VALUE_INPUT_{i}",
                        VARIABLE: self.variable[i],
                        PROJECTIONS: field}
                       for i, field in enumerate(self.input_ports)]
        return super()._instantiate_input_ports(input_ports=input_ports, context=context)

    def _instantiate_output_ports(self, output_ports=None, reference_value=None, context=None):
        learning_signal_dicts = []
        for i, learning_signal in enumerate(self.learning_signals):
            learning_signal_dicts.append({NAME: f"STORE TO {learning_signal.receiver.owner.name} MATRIX",
                                          VARIABLE: (OWNER_VALUE, i),
                                          REFERENCE_VALUE: self.value[i],
                                          MODULATION: self.modulation,
                                          PROJECTIONS: learning_signal.parameter_ports['matrix']})
        self.parameters.learning_signals._set(learning_signal_dicts, context)

        learning_signals = super()._instantiate_output_ports(context=context)


    def _parse_function_variable(self, variable, context=None):
        # Function expects a single field (one item of Mechanism's variable) at a time
        if self.initialization_status == ContextFlags.INITIALIZING:
            # During initialization, Mechanism's variable is its default_variable,
            # which has all field's worth of input, so need get a single one here
            return variable[0]
        # During execution, _execute passes only a entry (item of variable) at a time,
        #    so can just pass that along here
        return variable

    def _execute(self,
                 variable=None,
                 context=None,
                 runtime_params=None):
        """Execute EMStorageMechanism.function and return learning_signals

        For each node in query_input_nodes and value_input_nodes,
        assign its value to afferent weights of corresponding retrieved_node.
        - memory = matrix of entries made up vectors for each field in each entry (row)
        - memory_full_vectors = matrix of entries made up of vectors concatentated across all fields (used for norm)
        - entry_to_store = query_input or value_input to store
        - field_memories = weights of Projections for each field

        DIVISION OF LABOR BETWEEN MECHANISM AND FUNCTION:
        EMStorageMechanism._execute:
         - compute norms to find weakest entry in memory
         - compute storage_prob to determine whether to store current entry in memory
         - call function for each LearningSignal to decay existing memory and assign input to weakest entry
        EMStorage function:
         - decay existing memories
         - assign input to weakest entry (given index passed from EMStorageMechanism)

        :return: List[2d np.array] self.learning_signal
        """

        # FIX: SET LEARNING MODE HERE FOR SHOW_GRAPH

        decay_rate = self.parameters.decay_rate._get(context)      # modulable, so use getter
        storage_prob = self.parameters.storage_prob._get(context)  # modulable, so use getter
        field_weights = self.parameters.field_weights._get(context)  # modulable, so use getter
        concatenation_node = self.concatenation_node
        num_match_fields = 1 if concatenation_node else len([i for i in self.field_types if i==1])

        memory = self.parameters.memory_matrix._get(context)
        if memory is None or self.is_initializing:
            if self.is_initializing:
                # Return existing matrices for field_memories  # FIX: THE FOLLOWING DOESN'T TEST FUNCTION:
                return convert_all_elements_to_np_array([
                    learning_signal.receiver.path_afferents[0].parameters.matrix._get(context)
                    for learning_signal in self.learning_signals
                ])
            # Raise exception if not initializing and memory is not specified
            else:
                owner_string = ""
                if self.owner:
                    owner_string = " of " + self.owner.name
                raise EMStorageMechanismError(f"Call to {self.__class__.__name__} function {owner_string} "
                                              f"must include '{MEMORY_MATRIX}' in params arg.")

        # Get least used slot (i.e., weakest memory = row of matrix with lowest weights) computed across all fields
        field_norms = np.empty((len(memory),len(memory[0])))
        for row in range(len(memory)):
            for col in range(len(memory[0])):
                field_norms[row][col] = np.linalg.norm(memory[row][col])
        if field_weights is not None:
            field_norms *= field_weights
        row_norms = np.sum(field_norms, axis=1)
        # IMPLEMENTATION NOTE:
        #  the following will give the lowest index in case of a tie;
        #  this means that if memory is initialized with all zeros,
        #  it will be occupied in row order
        idx_of_weakest_memory = np.argmin(row_norms)

        value = []
        for i, field_projection in enumerate([learning_signal.efferents[0].receiver.owner
                                            for learning_signal in self.learning_signals]):
            if i < num_match_fields:
                # For match matrices,
                #   get entry to store from variable of Projection matrix (memory_field)
                #   to match_node in which memory will be stored (this is to accomodate concatenation_node)
                axis = 0
                entry_to_store = field_projection.parameters.variable._get(context)
                if concatenation_node is None:
                    assert np.all(entry_to_store == variable[i]),\
                        f"PROGRAM ERROR: misalignment between inputs and fields for storing them"
            else:
                # For retrieval matrices,
                #    get entry to store from variable (which has inputs to all fields)
                axis = 1
                entry_to_store = variable[i - num_match_fields]
            # Get matrix containing memories for the field from the Projection
            field_memory_matrix = field_projection.parameters.matrix._get(context)

            # pass in field_projection matrix to EMStorage function
            res = super(LearningMechanism, self)._execute(
                variable=entry_to_store,
                memory_matrix=copy_parameter_value(field_memory_matrix),
                axis=axis,
                storage_location=idx_of_weakest_memory,
                storage_prob=storage_prob,
                decay_rate=decay_rate,
                context=context,
                runtime_params=runtime_params
            )
            value.append(res)
            # assign modified field_memory_matrix back
            field_projection.parameters.matrix._set(res, context)
        return convert_all_elements_to_np_array(value)
