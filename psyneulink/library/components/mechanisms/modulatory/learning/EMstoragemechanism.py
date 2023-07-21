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
which it receives its `fields <EMStorageMechanism_Fields>`, and a `field_indices <EMStorageMechanism.field_indices>`
specification that identifies which parts of the `memory_matrix <EMStorageMechanism.memory_matrix>` to which each
field corresponds.

.. _EMStorageMechanism_Structure:

Structure
---------

An EMStorageMechanism is identical to a `LearningMechanism` in all respects except the following:

  * it has no `input_source <LearningMechanism.input_source>`, `output_source <LearningMechanism.output_source>`,
    or `error_source <LearningMechanism.error_source>` attributes;  instead, it has the `fields
    <EMStorageMechanism.fields>` and `field_indices <EMStorageMechanism.field_indices>` attributes described below.

  * its `fields <EMStorageMechanism.fields>` attribute has as many *FIELDS* `field <EMStorage_mechanism.fields>`
    as there are `fields <EMStorageMechanism_Fields>` of an entry in its `memory_matrix
    <EMStorageMechanism.memory_matrix>` attribute;  these are listed in its `fields <EMStorageMechanism.fields>`
    attribute and serve as the `InputPort`\\s for the EMStorageMechanism;  each receives a `MappingProjection` from
    the `OutputPort` of a `ProcessingMechanism`, the activity of which constitutes the corresponding `field
    <EMStorageMechanism_Fields>` of the `entry <EMStorageMechanism_Entry>` to be stored in its `memory_matrix
    <EMStorageMechanism.memory_matrix>` attribute.

  * it has a `field_indices <EMStorageMechanism.field_indices>` attribute that specifies the indices of the `memory
    matrix <EMStorageMechanism.memory_matrix>` to which each `field <EMStorageMechanism_Fields>` is assigned.

  * it has a `memory_matrix <EMStorageMechanism.memory_matrix>` attribute that represents the full memory that the
    EMStorageMechanism is used to update.

  * it has a several *LEARNING_SIGNAL* `OutputPorts <OutputPort>` that each send a `LearningProjection` to the `matrix
    <MappingProjection.matrix>` parameter of a 'MappingProjection` that constitutes a `field <EMStorageMechanism_Fields>`
    of the `memory_matrix <EMStorageMechanism.memory_matrix>` attribute.

  * its `function <EMStorageMechanism.function>` is an `EMStorage` `LearningFunction`, that takes as its `variable
    <Function_Base.variable>` a list or 1d np.array with a length of the corresponding  *ACTIVATION_INPUT* InputPort;
    and it returns a `learning_signal <LearningMechanism.learning_signal>` (a weight matrix assigned to one of the 
    Mechanism's *LEARNING_SIGNAL* OutputPorts), but no `error_signal <LearningMechanism.error_signal>`.

  * its `decay_rate <EMStorageMechanism.decay_rate>`, a float in the interval [0,1] that is used to decay
    `memory_matrix <EMStorageMechanism.memory_matrix>` before an `entry <EMStorageMechanism_Entry>` is stored.

  * its `storage_prob <EMStorageMechanism.storage_prob>`, a float in the interval [0,1] is used in place of a
    LearningMechanism's `storage_prob <LearningMechanism.storage_prob>` to determine the probability that the 
    Mechanism will store its `variable <EMStorageMechanism.variable>` in its `memory_matrix
    <EMStorageMechanism.memory_matrix>` attribute each time it executes.

.. _EMStorageMechanism_Execution:

Execution
---------

An EMStorageMechanism executes in the same manner as standard `LearningMechanism`, however instead of modulating
the `matrix <MappingProjection.matrix>` Parameter of a `MappingProjection`, it replaces a row or column in each of
the `matrix <MappingProjection.matrix>` Parameters of the `MappingProjections <MappingProjection>` to which its
`LearningProjections <LearningProjection>` project with an item of its `variable <EMStorageMechanism.variable>` that
represents the corresponding `field <EMStorageMechanism.fields>`.


.. _EMStorageMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
from beartype import beartype
from typing import Literal

from psyneulink._typing import Optional, Union, Callable

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.nonstateful.learningfunctions import EMStorage
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    ACTIVATION_INPUT, LearningMechanism, LearningMechanismError, LearningTiming, LearningType
from psyneulink.core.components.projections.projection import Projection, projection_keywords
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    ADDITIVE, EM_STORAGE_MECHANISM, LEARNING, LEARNING_PROJECTION, LEARNING_SIGNAL, MULTIPLICATIVE, OVERWRITE
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_numeric, ValidParamSpecType, all_within_range

__all__ = [
    'EMStorageMechanism', 'EMStorageMechanismError', 'DefaultTrainingMechanism',
    'input_port_names', 'output_port_names',
]

# Parameters:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

input_port_names = [ACTIVATION_INPUT]
output_port_names = [LEARNING_SIGNAL]

DefaultTrainingMechanism = ObjectiveMechanism

class EMStorageMechanismError(LearningMechanismError):
    pass


class EMStorageMechanism(LearningMechanism):
    """
    EMStorageMechanism(                       \
        variable,                             \
        fields,                               \
        field_indices,                        \
        memory_matrix,                        \
        function=EMStorage,                   \
        decay_rate=0.0,                       \
        storage_prob=1.0,                     \
        learning_signals,                     \
        modulation=OVERWRITE,                 \
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
        used to construct the Mechanism's `InputPorts <InputPort>`.

    field_indices : List[int or slice] : default None
        specifies the indices of the `memory_matrix <EMStorageMechanism.memory_matrix>` to which the `value
        <InputPort.value>` of the corresponding item of its `fields <EMStorageMechanism.fields>` attribute should be
        assigned (see `field_indices <EMStorageMechanism.field_indices>` for additional details).  If ints are used,
        then each must designate the starting index of each field in `memory_matrix <EMStorageMechanism.memory_matrix>`,
        taking account of the width of and indicating the index just after the last item of the preceding field.

    memory_matrix : List or 2d np.array : default None
        specifies the shape of the `memory <EMStorageMechanism_Memory>` used to store an `entry
        <EMStorageMechanism_Entry>` (see `memory_matrix <EMStorageMechanism.memory_matrix>` for additional details).

    function : LearningFunction or function : default EMStorage
        specifies the function used to assign each item of the `variable <EMStorageMechanism.variable>` to the
        corresponding `field <EMStorageMechanism_Fields>` of the `memory_matrix <EMStorageMechanism.memory_matrix>`.
        It must take as its `variable <EMSorage.variable> argument a list or 1d array of numeric values
        (the "activity vector") and return a list, 2d np.array or np.matrix for the corresponding `field
        <EMStorageMechanism_Fields>` of the `memory_matrix <EMStorageMechanism.memory_matrix>` (see `function
        <EMStorageMechanism.function>` for additional details).

    learning_signals : List[ParameterPort, Projection, tuple[str, Projection] or dict] : default None
        specifies the `ParameterPort`\\(s) of the `MappingProjections <MappingProjection>` that implement the `memory
        <EMStorageMechanism_Memory>` in which the `entry <EMStorageMechanism_Entry>` is stored; there must the same
        number of these as `fields <EMStorageMechanism.fields>`, and they must be specified in the sqme order.

    decay_rate : float : default 0.0
        specifies the rate at which `entries <EMStorageMechanism_Entry>` in the `memory_matrix
        <EMStorageMechanism.memory_matrix>` decays (see `decay_rate <EMStorageMechanism.decay_rate>` for additional
        details).

    storage_prob : float : default None
        specifies the probability with which the current entry is stored in the EMSorageMechanism's `memory_matrix
        <EMStorageMechanism.memory_matrix>` (see `storage_prob <EMStorageMechanism.storage_prob>` for details).

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

    field_indices : List[int or tuple[slice]]
        contains the indices of the `memory_matrix <EMStorageMechanism.memory_matrix>` to which the `value
        <InputPort.value>` of the corresponding item of its `fields <EMStorageMechanism.fields>` attribute is
        assigned (see `Fields <EMStorageMechanism_Fields>` for additional details).

    learned_projections : List[MappingProjection]
        list of the `MappingProjections <MappingProjection>`, the `matrix <MappingProjection.matrix>` Parameters of
        which are modified by the EMStorageMechanism.

    function : LearningFunction or function : default EMStorage
        the function used to assign the value of each `field <EMStorageMechanism.fields>` to the corresponding entry
        in `memory_matrix <EMStorageMechanism.memory_matrix>`.  It must take as its `variable <EMSorage.variable>`
        argument a list or 1d array of numeric values (an `entry <EMStorage.entry`) and return a list, 2d np.array or
        np.matrix assigned to the corresponding `field <EMStorageMechanism_Fields>` of the `memory_matrix
        <EMStorageMechanism.memory_matrix>`.

    decay_rate : float : default 0.0
        determines the rate at which `entries <EMStorageMechanism_Entry>` in the `memory_matrix
        <EMStorageMechanism.memory_matrix>` decay;  the decay rate is applied to `memory_matrix
        <EMStorageMechanism.memory_matrix>` before it is updated with the new `entry <EMStorageMechanism_Entry>`.

    storage_prob : float
        specifies the probability with which the current entry is stored in the EMSorageMechanism's `memory_matrix
        <EMStorageMechanism.memory_matrix>`.

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

                decay_rate
                    see `decay_rate <EMStorageMechanism.decay_rate>`

                    :default value: 0.0
                    :type: ``float``


                fields
                    see `fields <EMStorageMechanism.fields>`

                    :default value: None
                    :type: ``list``

                field_indices
                    see `field_indices <EMStorageMechanism.field_indices>`

                    :default value: None
                    :type: ``list``

                memory_matrix
                    see `memory_matrix <EMStorageMechanism.memory_matrix>`

                    :default value: None
                    :type: ``np.ndarray``

                function
                    see `function <EMStorageMechanism.function>`

                    :default value: `EMStorage`
                    :type: `Function`

                input_ports
                    see `fields <EMStorageMechanism.fields>`

                    :default value: None
                    :type: ``list``
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
        input_ports = Parameter([],
                                stateful=False,
                                loggable=False,
                                read_only=True,
                                structural=True,
                                parse_spec=True,
                                constructor_argument='fields'
                                )
        field_indices = Parameter([],
                                    stateful=False,
                                    loggable=False,
                                    read_only=True,
                                    structural=True,
                                    parse_spec=True,
                                    dependiencies='fields')
        function = Parameter(EMStorage, stateful=False, loggable=False)
        storage_prob = Parameter(1.0, modulable=True)
        storage_prob = Parameter(1.0, modulable=True)
        decay_rate = Parameter(0.0, modulable=True)
        modulation = OVERWRITE
        output_ports = Parameter([],
                                 stateful=False,
                                 loggable=False,
                                 read_only=True,
                                 structural=True,
                                 constructor_argument='learning_signals'
                                 )
        learning_type = LearningType.UNSUPERVISED
        learning_timing = LearningTiming.LEARNING_PHASE

    # FIX: WRITE VALIDATION AND PARSE METHODS FOR THESE
    def _validate_field_indices(self, field_indices):
        if not len(field_indices) or len(field_indices) != len(self.fields):
            return f"must be specified with a number of items equal to " \
                   f"the number of fields specified {len(self.fields)}"

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
                 fields: Optional[list, tuple, dict, OutputPort, Mechanism, Projection] = None,
                 field_indices: Optional[Union[int,slice]] = None,
                 memory_matrix: Optional[Union[list, np.ndarray]] = None,
                 function: Optional[Callable] = EMStorage,
                 learning_signals: Optional[list, dict, ParameterPort, Projection, tuple] = None,
                 modulation: Optional[Literal[OVERWRITE, ADDITIVE, MULTIPLICATIVE]] = None,
                 decay_rate: Optional[float] = None,
                 storage_prob: Optional[float] = None,
                 params=None,
                 name=None,
                 prefs: Optional[ValidPrefSet] = None,
                 **kwargs
                 ):

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self._init_args = locals().copy()
        # self._init_args['context'] = self
        # self._init_args['name'] = name

        # # Flag for deferred initialization
        # self.initialization_status = ContextFlags.DEFERRED_INIT
        # self.initialization_status = ContextFlags.DEFERRED_INIT

        # self._storage_prob = storage_prob

        super().__init__(default_variable=default_variable,
                         fields=fields,
                         fields_indices=field_indices,
                         memory_matrix=memory_matrix,
                         function=function,
                         modulation=modulation,
                         decay_rate=decay_rate,
                         storage_prob=storage_prob,
                         learning_signals=learning_signals,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs
                         )


    # FIX: NEEDED??
    def _parse_function_variable(self, variable, context=None):
        return variable

    def _validate_variable(self, variable, context=None):
        """Validate that variable has only one item: activation_input.
        """

        # Skip LearningMechanism._validate_variable in call to super(), as it requires variable to have 3 items
        variable = super(LearningMechanism, self)._validate_variable(variable, context)

        # Items in variable should be 1d and have numeric values
        if not (all(np.array(variable)[i].ndim == 1 for i in len(variable)) and is_numeric(variable)):
            raise EMStorageMechanismError(f"Variable for {self.name} ({variable}) must be "
                                          f"a list or 2d np.array containing 1d arrays with only numbers.")

        # Items in variable should have the same shape as memory_matrix
        memory_matrix = self.parameters.memory_matrix.get()
        if memory_matrix.shape != np.array(variable).vstack.shape:
            raise EMStorageMechanismError(f"The 'variable' arg for {self.name} ({variable}) must be "
                                          f"a list or 2d np.array containing entries that, when vertically stacked,"
                                          f"are the same shape ({memory_matrix.shape}) as its 'memory_matrix' arg.")
        return variable

    def _instantiate_input_ports(self, input_ports=None, reference_value=None, context=None):
        # FIX: SHOUD HAVE SPECS FROM fields ARG HERE
        pass

    def _instantiate_output_ports(self, output_ports=None, reference_value=None, context=None):
        # FIX: SHOUD HAVE SPECS FROM learning_signals ARG HERE
        pass

    def _execute(
        self,
        variable=None,
        context=None,
        runtime_params=None,

    ):
        """Execute EMStorageMechanism. function and return learning_signal

        :return: (2D np.array) self.learning_signal
        """

        # COMPUTE LEARNING SIGNAL (note that function is assumed to return only one value)
        # IMPLEMENTATION NOTE:  skip LearningMechanism's implementation of _execute
        #                       as it assumes projections from other LearningMechanisms
        #                       which are not relevant to an autoassociative projection
        learning_signal = super(LearningMechanism, self)._execute(
            variable=variable,
            context=context,
            runtime_params=runtime_params,

        )

        from psyneulink.core.compositions.report import ReportOutput
        if self.initialization_status != ContextFlags.INITIALIZING and self.reportOutputPref is not ReportOutput.OFF:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.parameters.learning_signal._get(context)))

        # --------------------------------------------------------------
            # FIX: PUT THIS IN EMStorageMechanism
            memories = input_node.efferents[0].parameters.matrix.get(context)
            if self.memory_decay_rate:
                memories *= self.parameters.memory_decay_rate._get(context)
            # Get least used slot (i.e., weakest memory = row of matrix with lowest weights)
            # idx_of_min = np.argmin(memories.sum(axis=0))
        # --------------------------------------------------------------




        value = np.array([learning_signal])

        self.parameters.value._set(value, context)

        return value

    def _update_output_ports(self, runtime_params=None, context=None):
        """Update the weights for the AutoAssociativeProjection for which this is the EMStorageMechanism

        Must do this here, so it occurs after LearningMechanism's OutputPort has been updated.
        This insures that weights are updated within the same trial in which they have been learned
        """

        super()._update_output_ports(runtime_params, context)
        if self.parameters.learning_enabled._get(context):
            learned_projection = self.activity_source.recurrent_projection
            old_exec_phase = context.execution_phase
            context.execution_phase = ContextFlags.LEARNING
            learned_projection.execute(context=context)
            context.execution_phase = old_exec_phase

    @property
    def activity_source(self):
        return self.input_port.path_afferents[0].sender.owner
