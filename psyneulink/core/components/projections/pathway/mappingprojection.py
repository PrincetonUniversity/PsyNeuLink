# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  MappingProjection ****************************************************

"""

Contents
--------
  * `MappingProjection_Overview`
  * `MappingProjection_Creation`
      - `MappingProjection_Matrix_Specification`
      - `MappingProjection_Learning_Specification`
      - `MappingProjection_Deferred_Initialization`
  * `MappingProjection_Structure`
      - `MappingProjection_Matrix`
      - `Mapping_Matrix_ParameterPort`
  * `MappingProjection_Execution`
      - `MappingProjection_Learning`
  * `MappingProjection_Class_Reference`


.. _MappingProjection_Overview:

Overview
--------

A MappingProjection transmits the `value <OutputPort.value>` of an `OutputPort` of one `ProcessingMechanism
<ProcessingMechanism>` (its `sender <MappingProjection.sender>`) to the `InputPort` of another (its `receiver
<MappingProjection.receiver>`). The default `function <Projection_Base.function>` for a MappingProjection is
`LinearMatrix`, which uses the MappingProjection's `matrix <MappingProjection.matrix>` attribute to transform the
value received from its `sender <MappingProjection.sender>` and provide the result to its `receiver
<MappingProjection.receiver>`.

.. _MappingProjection_Creation:

Creating a MappingProjection
-----------------------------

A MappingProjection can be created in any of the ways that can be used to create a `Projection <Projection_Creation>`
(see `Projection_Sender` and `Projection_Receiver` for specifying its `sender <MappingProjection.sender>` and
`receiver <MappingProjection.receiver>` attributes, respectively), or simply by `specifying it by its matrix parameter
<MappingProjection_Matrix_Specification>` wherever a `Projection can be specified <Projection_Specification>`.

MappingProjections are also generated automatically in the following circumstances, using a value for its `matrix
<MappingProjection.matrix>` parameter appropriate to the circumstance:

  * by a `Composition`, when two adjacent `Mechanisms <Mechanism>` in its `pathway <Process.pathway>` do not already
    have a Projection assigned between them (`AUTO_ASSIGN_MATRIX` is used as the `matrix <MappingProjection.matrix>`
    specification, which determines the appropriate matrix by context);
  ..
  * by an `ObjectiveMechanism`, from each `OutputPort` listed in its `monitored_output_ports
    <ObjectiveMechanism.monitored_output_ports>` attribute to the corresponding `InputPort` of the ObjectiveMechanism
    (`AUTO_ASSIGN_MATRIX` is used as the `matrix <MappingProjection.matrix>` specification, which determines the
    appropriate matrix by context);
  ..
  * by a `LearningMechanism`, between it and the other components required to implement learning
    (see `LearningMechanism_Learning_Configurations` for details);
  ..
  * by a `ControlMechanism <ControlMechanism>`, from the *OUTCOME* `OutputPort` of the `ObjectiveMechanism` that `it
    creates <ControlMechanism_ObjectiveMechanism>` to its *OUTCOME* `InputPort`, and from the `OutputPorts <OutputPort>`
    listed in the ObjectiveMechanism's `monitored_output_ports <ObjectiveMechanism.monitored_output_ports>` attribute
    to the ObjectiveMechanism's `primary InputPort <InputPort_Primary>` (as described above; an `IDENTITY_MATRIX` is
    used for all of these).

.. _MappingProjection_Matrix_Specification:

*Specifying the Matrix Parameter*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a MappingProjection is created automatically, its `matrix <MappingProjection.matrix>` attribute is generally
assigned using `AUTO_ASSIGN_MATRIX`, which determines its size by context: an `IDENTITY_MATRIX` is used if the
`sender <MappingProjection.sender>` and `receiver <MappingProjection.receiver>` are of equal length; otherwise a
`FULL_CONNECTIVITY_MATRIX` (all 1's) is used.

When a MappingProjection is created explicitly, the **matrix** argument of its constructor can be used to specify
its `matrix <MappingProjection.matrix>` parameter.  This is used by the MappingProjection's `function
<Projection_Base.function>` to transform the input from its `sender <MappingProjection.sender>` into the `value
<Projection_Base.value>` provided to its `receiver <MappingProjection.receiver>`. It can be specified in any of the
following ways:

  * **List, array or matrix**  -- if it is a list, each item must be a list or 1d np.array of numbers;  otherwise,
    it must be a 2d np.array or np.matrix.  In each case, the outer dimension (outer list items, array axis 0,
    or matrix rows) corresponds to the elements of the `sender <MappingProjection.sender>`, and the inner dimension
    (inner list items, array axis 1, or matrix columns) corresponds to the weighting of the contribution that a
    given `sender <MappingProjection.sender>` makes to the `receiver <MappingProjection.receiver>` (the number of which
    must match the length of the receiver's `variable <InputPort.variable>`).

  .. _Matrix_Keywords:

  * **Matrix keyword** -- used to specify a standard type of matrix without having to specify its individual
    values, or to allow the type of matrix to be determined by context;  any of the `matrix keywords
    <Keywords.MatrixKeywords>` can be used.

  ..
  * **Random matrix function** (`random_matrix <Utilities.random_matrix>`) -- a convenience function
    that provides more flexibility than `RANDOM_CONNECTIVITY_MATRIX`.  It generates a random matrix sized for a
    **sender** and **receiver**, with random numbers drawn from a uniform distribution within a specified **range** and
    with a specified **offset**.

  .. _MappingProjection_Tuple_Specification:

  * **Tuple** -- used to specify the `matrix <MappingProjection.matrix>` along with a specification for learning;
    The tuple must have two items: the first can be any of the specifications described above; the second must be
    a `learning specification <MappingProjection_Learning_Tuple_Specification>`.

.. _MappingProjection_Learning_Specification:

*Specifying Learning*
~~~~~~~~~~~~~~~~~~~~~

A MappingProjection is specified for learning in any of the following ways:

    * in the **matrix** argument of the MappingProjection's constructor, using the `tuple format
      <MappingProjection_Tuple_Specification>` described above;
    ..
    * specifying the MappingProjection (or its *MATRIX* `ParameterPort`) as the `receiver
      <LearningProjection.receiver>` of a `LearningProjection`;
    ..
    * specifying the MappingProjection (or its *MATRIX* `ParameterPort`) in the **projections** argument of
      the constructor for a `LearningSignal <LearningSignal_Specification>`
    ..
    * specifying the MappingProjection (or its *MATRIX* `ParameterPort`) in the **learning_signals** argument of
      the constructor for a `LearningMechanism <LearningSignal_Specification>`
    ..
    * specifying a MappingProjection in the `pathway <Process.pathway>`  for a `Process`,
      using the `tuple format <MappingProjection_Learning_Tuple_Specification>` to include a learning specification;
    ..
    * `specifying learning <Process_Learning_Sequence>` for a `Process`, which assigns `LearningProjections
      <LearningProjection>` to all of the  MappingProjections in the Process' `pathway <Process.pathway>`;

See `LearningMechanism` documentation for an overview of `learning components <LearningMechanism_Overview>` and a
detailed description of `LearningMechanism_Learning_Configurations`;  see `MappingProjection_Learning` below for a
description of how learning modifies a MappingProjection.


.. _MappingProjection_Learning_Tuple_Specification:

Specifying Learning in a Tuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A tuple can be used to specify learning for a MappingProjection in the **matrix** `argument of its constructor
<MappingProjection_Matrix_Specification>` or in the `pathway of a Process <Process_Projections>`.  In both cases,
the second item of the tuple must be a learning specification, which can be any of the following:

  * an existing `LearningProjection`, `LearningSignal`, or a constructor for one -- the specified Component is used, and
    defaults are automatically created for the other `learning Components <LearningMechanism_Learning_Configurations>`;
  ..
  * a reference to the LearningProjection or LearningSignal class, or the keyword *LEARNING* or *LEARNING_PROJECTION* --
    a default set of `learning Components <LearningMechanism_Learning_Configurations>` is automatically created.

Specifying Learning for AutodiffCompositions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, all MappingProjections in an `AutodiffComposition` are treated as trainable PyTorch Parameters whose
matrices are updated during backwards passes through the network. Optionally, users can specify during
instantiation that a projection should not be updated. To do so, set the `learnable` argument to False in the
constructor of the projection.

.. _MappingProjection_Deferred_Initialization:

*Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~

When a MappingProjection is created, its full initialization is `deferred <Component_Deferred_Init>` until its
`sender <MappingProjection.sender>` and `receiver <MappingProjection.receiver>` have been fully specified.  This
allows a MappingProjection to be created before its `sender <MappingProjection.sender>` and/or `receiver
<MappingProjection.receiver>` have been created (e.g., before them in a script), by calling its constructor without
specifying its **sender** or **receiver** arguments. However, for the MappingProjection to be operational,
initialization must be completed by calling its `deferred_init` method.  This is not necessary if the MappingProjection
is specified in the **pathway** argument of a Composition's `add_linear_processing_pathway` or
`learning methods <Composition_Learning_Methods>`, or anywhere else that its `sender <MappingProjection.sender>` and
`receiver <MappingProjection.receiver>` can be determined by context.

.. _MappingProjection_Structure:

Structure
---------

In addition to its `sender <MappingProjection.sender>`, `receiver <MappingProjection.receiver>`, and `function
<Projection_Base.function>`, a MappingProjection has the following characteristic attributes:

.. _Mapping_Matrix:

* `matrix <MappingProjection.matrix>` parameter - used by the MappingProjection's `function <Projection_Base.function>`
  to carry out a matrix transformation of its input, that is then provided to its `receiver
  <MappingProjection.receiver>`. It can be specified in a variety of ways, as described `above
  <MappingProjection_Matrix_Specification>`.

  .. _Mapping_Matrix_Dimensionality

  * **Matrix Dimensionality** -- this must match the dimensionality of the MappingProjection's `sender
    <MappingProjection.sender>` and `receiver <MappingProjection.receiver>`.  For a standard 2d "weight" matrix (i.e.,
    one that maps a 1d array from its `sender <MappingProjection.sender>` to a 1d array of its `receiver
    <MappingProjection.receiver>`), the dimensionality of the sender is the number of rows and of the receiver
    the number of columns.  More generally, the sender dimensionality is the number of outer dimensions (i.e.,
    starting with axis 0 of numpy array) equal to the number of dimensions of its `sender <MappingProjection.sender>`'s
    `value <Port_Base.value>`, and the receiver dimensionality is the number of inner dimensions equal to its
    `receiver <MappingProjection.receiver>`'s `variable <Projection_Base.variable>` (equal to the dimensionality of the
    matrix minus its sender dimensionality).

.. _Mapping_Matrix_ParameterPort:

* *MATRIX* `ParameterPort` - this receives any `LearningProjections <LearningProjection>` that are assigned to the
  MappingProjection (see `MappingProjection_Learning_Specification` above), and updates the current value of the
  MappingProjection's `matrix <MappingProjection.matrix>` parameter in response to `learning
  <LearningMechanism>`.  The `function <ParameterPort.function>` of a *MATRIX* ParameterPort is an
  `AccumulatorIntegrator`, which accumulates the weight changes received from the LearningProjections
  that project to it (see `MappingProjection_Learning` below).  This can be replaced by any function that defines an
  *ADDITIVE_PARAM* `modulatory parameter <ModulatorySignal_Modulation>`), and that takes as its input an array or
  matrix and returns one of the same size.

.. _Mapping_Weight_Exponent:

*  `weight <Projection_Base.weight>` and `exponent <Projection_Base.exponent>` - applied to the `value
   <Projection_Base.value>` of the MappingProjection before it is combined with other MappingProjections to the same
   `InputPort` to determine its `value <InputPort.value>` (see description under `Projection
   <Projection_Weight_Exponent>` for additional details).

   .. note::
      The `weight <Projection_Base.weight>` and `exponent <Projection_Base.exponent>` attributes of a MappingProjection
      are distinct from those of the `InputPort` to which it projects.  It is also important to recognize that,
      as noted under `Projection <Projection_Weight_Exponent>`, they are not normalized, and thus contribute to the
      magnitude of the InputPort's `variable <InputPort.variable>` and therefore its relationship to that of other
      InputPorts that may belong to the same Mechanism.

.. _MappingProjection_Execution:

Execution
---------

A MappingProjection uses its `function <Projection_Base.function>` and `matrix <MappingProjection.matrix>` parameter to
transform its `sender <MappingProjection.sender>` into a form suitable for the `variable <InputPort.variable>` of its
`receiver <MappingProjection.receiver>`.  A MappingProjection cannot be executed directly. It is executed when the
`InputPort` to which it projects (i.e., its `receiver <MappingProjection.receiver>`) is updated;  that occurs when the
InputPort's owner `Mechanism <Mechanism>` is executed. When executed, the MappingProjection's *MATRIX* `ParameterPort`
updates its `matrix <MappingProjection.matrix>` parameter based on any `LearningProjection(s)` it receives (listed in
the ParameterPort's `mod_afferents <ParameterPort.mod_afferents>` attribute). This brings into effect any changes that
occurred due to `learning <MappingProjection_Learning>`.  Since this does not occur until the Mechanism that receives
the MappingProjection is executed (in accord with `Lazy Evaluation <Component_Lazy_Updating>`), any changes due to
learning do not take effect, and are not observable (e.g., through inspection of the `matrix <MappingProjection.matrix>`
attribute or the `value <ParameterPort.value>` of its ParameterPort) until the next `TRIAL <TimeScale.TRIAL>` of
execution (see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).

.. _MappingProjection_Learning:

*Learning*
~~~~~~~~~~

Learning modifies the `matrix <MappingProjection.matrix>` parameter of a MappingProjection, under the influence
of one or more `LearningProjections <LearningProjection>` that project to its *MATRIX* `ParameterPort`.
This conforms to the general procedures for modulation used by `ModulatoryProjections <ModulatoryProjection>`
A LearningProjection `modulates <LearningSignal_Modulation>` the `function <ParameterPort.function>` of the
*MATRIX* ParameterPort, which is responsible for keeping a record of the value of the MappingProjection's matrix,
and providing it to the MappingProjection's `function <Projection_Base.function>` (usually `LinearMatrix`).  By
default, the function for the *MATRIX* ParameterPort is an `AccumulatorIntegrator`.  A LearningProjection
modulates it by assigning the value of its `additive_param <AccumulatorIntegrator.additive_param>` (`increment
<AccumulatorIntegrator.increment>`), which is added to its `previous_value <AccumulatorIntegrator.previous_value>`
attribute each time it is executed. The result is that each time the MappingProjection is executed, and in turn
executes its *MATRIX* ParameterPort, the `weight changes <LearningProjection_Structure>` conveyed to the
MappingProjection from any LearningProjection(s) are added to the record of the matrix kept by the *MATRIX*
ParameterPort's `AccumulatorIntegrator` function in its `previous_value <AccumulatorIntegrator.previous_value>`
attribute. This is then the value of the matrix used  by the MappingProjection's `LinearMatrix` function when it is
executed.  It is important to note that the accumulated weight changes received by a MappingProjection from its
LearningProjection(s) are stored by the *MATRIX* ParameterPort's function, and not the MappingProjection's `matrix
<MappingProjection.matrix>` parameter itself; the latter stores the original value of the matrix before learning (that
is, its unmodulated value, conforming to the general protocol for `modulation <ModulatorySignal_Modulation>` in
PsyNeuLink).  The most recent value of the matrix used by the MappingProjection is stored in the `value
<ParameterPort.value>` of its *MATRIX* ParameterPort. As noted `above <MappingProjection_Execution>`, however, this does
not reflect any changes due to learning on the current `TRIAL <TimeScale.TRIAL>` of execution; those are assigned to the
ParameterPort's `value <ParameterPort.value>` when it executes, which does not occur until the `Mechanism
<Mechanism>` that receives the MappingProjection is executed in the next `TRIAL <TimeScale.TRIAL>` of execution
(see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).

.. _MappingProjection_Class_Reference:

Class Reference
---------------

"""
import copy

import numpy as np

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AccumulatorIntegrator
from psyneulink.core.components.functions.transferfunctions import LinearMatrix
from psyneulink.core.components.functions.function import get_matrix
from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base
from psyneulink.core.components.projections.projection import ProjectionError, Projection_Base, projection_keywords
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.keywords import \
    AUTO_ASSIGN_MATRIX, CONTEXT, DEFAULT_MATRIX, FULL_CONNECTIVITY_MATRIX, FUNCTION, FUNCTION_PARAMS, \
    HOLLOW_MATRIX, IDENTITY_MATRIX, INPUT_PORT, LEARNING, LEARNING_PROJECTION, MAPPING_PROJECTION, MATRIX, \
    OUTPUT_PORT, PROJECTION_SENDER, VALUE
from psyneulink.core.globals.log import ContextFlags
from psyneulink.core.globals.parameters import FunctionParameter, Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'MappingError', 'MappingProjection',
]

parameter_keywords.update({MAPPING_PROJECTION})
projection_keywords.update({MAPPING_PROJECTION})


class MappingError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def _mapping_projection_matrix_getter(owning_component=None, context=None):
    return owning_component.function.parameters.matrix.get(context)


def _mapping_projection_matrix_setter(value, owning_component=None, context=None):
    owning_component.function.parameters.matrix.set(value, context)
    # KDM 11/13/18: not sure that below is correct to do here, probably is better to do this in a "reset" type
    # method but this is needed for Kalanthroff model to work correctly (though untested, it is in Scripts/Models)
    owning_component.parameter_ports["matrix"].function.parameters.previous_value.set(value, context)

    return value


class MappingProjection(PathwayProjection_Base):
    """
    MappingProjection(         \
        sender=None,           \
        receiver=None,         \
        matrix=DEFAULT_MATRIX)

    Subclass of `Projection` that transmits the `value <OutputPort.value>` of the `OutputPort` of one `Mechanism
    <Mechanism>` to the `InputPort` of another (or possibly itself).  See `Projection <Projection_Class_Reference>`
    for additional arguments and attributes.

    Arguments
    ---------

    sender : OutputPort or Mechanism : default None
        specifies the source of the Projection's input. If a `Mechanism <Mechanism>` is specified, its
        `primary OutputPort <OutputPort_Primary>` is used. If it is not specified, it is assigned in
        the context in which the MappingProjection is used, or its initialization will be `deferred
        <MappingProjection_Deferred_Initialization>`.

    receiver: InputPort or Mechanism : default None
        specifies the destination of the Projection's output.  If a `Mechanism <Mechanism>` is specified, its
        `primary InputPort <InputPort_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used, or its initialization will be `deferred
        <MappingProjection_Deferred_Initialization>`.

    matrix : list, np.ndarray, np.matrix, function or keyword : default DEFAULT_MATRIX
        specifies the matrix used by `function <Projection_Base.function>` (default: `LinearCombination`) to
        transform the `value <Projection_Base.value>` of the `sender <MappingProjection.sender>` into a form suitable
        for the `variable <InputPort.variable>` of its `receiver <MappingProjection.receiver>` `InputPort`.

    Attributes
    ----------

    sender : OutputPort
        the `OutputPort` of the `Mechanism <Mechanism>` that is the source of the Projection's input.

    receiver: InputPort
        the `InputPort` of the `Mechanism <Mechanism>` that is the destination of the Projection's output.

    matrix : 2d np.array
        the matrix used by `function <Projection_Base.function>` to transform the input from the MappingProjection's
        `sender <MappingProjection.sender>` into the value provided to its `receiver <MappingProjection.receiver>`.

    has_learning_projection : bool : None
        identifies the `LearningProjection` assigned to the MappingProjection's `MATRIX` `ParameterPort
        <ParameterPort>`.

    learning_mechanism : LearningMechanism
        source of the `learning signal <LearningSignal>` that determines the changes to the `matrix
        <MappingProjection.matrix>` when `learning <LearningMechanism>` is used.

    name : str
        the name of the MappingProjection. If the specified name is the name of an existing MappingProjection,
        it is appended with an indexed suffix, incremented for each MappingProjection with the same base name (see
        `Registry_Naming`). If the name is not specified in the **name** argument of its constructor, a default name is
        assigned using the following format:
        'MappingProjection from <sender Mechanism>[<OutputPort>] to <receiver Mechanism>[InputPort]'
        (for example, ``'MappingProjection from my_mech_1[OutputPort-0] to my_mech2[InputPort-0]'``).
        If either the `sender <MappingProjection.sender>` or `receiver <MappingProjection.receiver>` has not yet been
        assigned (the MappingProjection is in `deferred initialization <MappingProjection_Deferred_Initialization>`),
        then the parenthesized name of class is used in place of the unassigned attribute
        (for example, if the `sender <MappingProjection.sender>` has not yet been specified:
        ``'MappingProjection from (OutputPort-0) to my_mech2[InputPort-0]'``).

    """

    componentType = MAPPING_PROJECTION
    className = componentType
    suffix = " " + className

    class Parameters(PathwayProjection_Base.Parameters):
        """
            Attributes
            ----------

                function
                    see `function <MappingProjection.function>`

                    :default value: `LinearMatrix`
                    :type: `Function`

                matrix
                    see `matrix <MappingProjection.matrix>`

                    :default value: `AUTO_ASSIGN_MATRIX`
                    :type: ``str``
        """
        function = Parameter(LinearMatrix, stateful=False, loggable=False)
        matrix = FunctionParameter(
            DEFAULT_MATRIX,
            setter=_mapping_projection_matrix_setter
        )

    classPreferenceLevel = PreferenceLevel.TYPE

    @property
    def _loggable_items(self):
        # Ports and afferent Projections are loggable for a Mechanism
        #     - this allows the value of InputPorts and OutputPorts to be logged
        #     - for MappingProjections, this logs the value of the Projection's matrix parameter
        #     - for ModulatoryProjections, this logs the value of the Projection
        # IMPLEMENTATION NOTE: this needs to be a property as that is expected by Log.loggable_items
        return list(self.parameter_ports)


    class sockets:
        sender=[OUTPUT_PORT]
        receiver=[INPUT_PORT]


    projection_sender = OutputPort

    def __init__(self,
                 sender=None,
                 receiver=None,
                 weight=None,
                 exponent=None,
                 matrix=None,
                 function=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None,
                 learnable=True,
                 **kwargs):

        # Assign matrix to function_params for use as matrix param of MappingProjection.function
        # (7/12/17 CW) this is a PATCH to allow the user to set matrix as an np.matrix... I still don't know why
        # it wasn't working.
        if isinstance(matrix, (np.matrix, list)):
            matrix = np.array(matrix)

        self.learning_mechanism = None
        self.has_learning_projection = None
        self.learnable = bool(learnable)
        if not self.learnable:
            assert True
        # If sender or receiver has not been assigned, defer init to Port.instantiate_projection_to_state()
        if sender is None or receiver is None:
            self.initialization_status = ContextFlags.DEFERRED_INIT

        # Validate sender (as variable) and params
        super().__init__(sender=sender,
                         receiver=receiver,
                         weight=weight,
                         exponent=exponent,
                         matrix=matrix,
                         function=function,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

        try:
            self._parameter_ports[MATRIX].function.reset(context=context)
        except AttributeError:
            pass

    def _instantiate_parameter_ports(self, function=None, context=None):

        super()._instantiate_parameter_ports(function=function, context=context)

        # FIX: UPDATE FOR LEARNING
        # FIX: UPDATE WITH MODULATION_MODS
        # FIX: MOVE THIS TO MappingProjection.__init__;
        # FIX: AS IT IS, OVER-WRITES USER ASSIGNMENT OF FUNCTION IN params dict FOR MappingProjection
        # TODO: why is this using the value not the variable? if there isn't a
        # specific reason, it should be variable, but this affects the values
        # tests/mechanisms/test_gating_mechanism.py::test_gating_with_composition
        new_variable = copy.deepcopy(self._parameter_ports[MATRIX].defaults.value)
        initial_rate = new_variable * 0.0

        # KDM 7/11/19: instead of simply setting the function, we need to reinstantiate to ensure
        # new defaults get set properly
        self._parameter_ports[MATRIX]._instantiate_function(
            function=AccumulatorIntegrator(
                owner=self._parameter_ports[MATRIX],
                default_variable=new_variable,
                initializer=new_variable,
                # rate=initial_rate
            ),
            context=context
        )
        self._parameter_ports[MATRIX]._instantiate_value(context)
        self._parameter_ports[MATRIX]._update_parameter_components(context)

        # # Assign ParameterPort the same Log as the MappingProjection, so that its entries are accessible to Mechanisms
        # self._parameter_ports[MATRIX].log = self.log

    def _instantiate_receiver(self, context=None):
        """Determine matrix needed to map from sender to receiver

        Assign specification to self.matrix_spec attribute
        Assign matrix to self.matrix attribute

        """
        self.reshapedWeightMatrix = False

        # Get sender and receiver lengths
        # Note: if either is a scalar, manually set length to 1 to avoid TypeError in call to len()
        try:
            mapping_input_len = len(self.defaults.variable)
        except TypeError:
            mapping_input_len = 1
        try:
            receiver_len = self.receiver.socket_width
        except TypeError:
            receiver_len = 1

        # Compare length of MappingProjection output and receiver's variable to be sure matrix has proper dimensions
        try:
            mapping_output_len = len(self.defaults.value)
        except TypeError:
            mapping_output_len = 1

        matrix_spec = self.defaults.matrix

        if (type(matrix_spec) == str and
                matrix_spec == AUTO_ASSIGN_MATRIX):
            if mapping_input_len == receiver_len:
                matrix_spec = IDENTITY_MATRIX
            else:
                matrix_spec = FULL_CONNECTIVITY_MATRIX

        # Length of the output of the Projection doesn't match the length of the receiving InputPort
        #    so consider reshaping the matrix
        if mapping_output_len != receiver_len:

            if 'projection' in self.name or 'Projection' in self.name:
                projection_string = ''
            else:
                projection_string = 'projection'

            if all(string in self.name for string in {'from', 'to'}):
                states_string = ''
            else:
                states_string = "from \'{}\' OuputState of \'{}\' to \'{}\'".format(self.sender.name,
                                                                                    self.sender.owner.name,
                                                                                    self.receiver.owner.name)
            if not isinstance(matrix_spec, str):
                # if all(string in self.name for string in {'from', 'to'}):

                raise ProjectionError("Width ({}) of the {} of \'{}{}\'{} "
                                      "does not match the length of its \'{}\' InputPort ({})".
                                      format(mapping_output_len,
                                             VALUE,
                                             self.name,
                                             projection_string,
                                             states_string,
                                             self.receiver.name,
                                             receiver_len))

            elif matrix_spec == IDENTITY_MATRIX or matrix_spec == HOLLOW_MATRIX:
                # Identity matrix is not reshapable
                raise ProjectionError("Output length ({}) of \'{}{}\' from {} to Mechanism \'{}\'"
                                      " must equal length of it InputPort ({}) to use {}".
                                      format(mapping_output_len,
                                             self.name,
                                             projection_string,
                                             self.sender.name,
                                             self.receiver.owner.name,
                                             receiver_len,
                                             matrix_spec))
            else:
                # Flag that matrix is being reshaped
                self.reshapedWeightMatrix = True
                if self.prefs.verbosePref:
                    print("Length ({}) of the output of {}{} does not match the length ({}) "
                          "of the InputPort for the receiver {}; the width of the matrix (number of columns); "
                          "the width of the matrix (number of columns) will be adjusted to accomodate the receiver".
                          format(mapping_output_len,
                                 self.name,
                                 projection_string,
                                 receiver_len,
                                 self.receiver.owner.name))

                self.parameters.matrix._set(
                    get_matrix(matrix_spec, mapping_input_len, receiver_len, context=context),
                    context
                )

                # Since matrix shape has changed, output of self.function may have changed, so update value
                self._instantiate_value(context=context)

        super()._instantiate_receiver(context=context)

    def _execute(self, variable=None, context=None, runtime_params=None):

        self._update_parameter_ports(runtime_params=runtime_params, context=context)

        value = super()._execute(
                variable=variable,
                context=context,
                runtime_params=runtime_params,
                )

        return value

    @property
    def logPref(self):
        return self.prefs.logPref

    # Always assign matrix ParameterPort the same logPref as the MappingProjection
    @logPref.setter
    def logPref(self, setting):
        self.prefs.logPref = setting
        self.parameter_ports[MATRIX].logPref = setting
