# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  MappingProjection ****************************************************

"""
.. _Mapping_Overview:

Overview
--------

A MappingProjection transmits the `value <OutputState.value>` of an `OutputState` of one `ProcessingMechanism
<ProcessingMechanism>` (its `sender <MappingProjection.sender>`) to the `InputState` of another (its `receiver
<MappingProjection.receiver>`). The default `function <MappingProjection.function>` for a MappingProjection is
`LinearMatrix`, which uses the MappingProjection's `matrix <MappingProjection.matrix>` attribute to transform the
value received from its `sender <MappingProjection.sender>` and provide the result to its `receiver
<MappingProjection.receiver>`.

.. _Mapping_Creation:

Creating a MappingProjection
-----------------------------

A MappingProjection can be created in any of the ways that can be used to create a `Projection <Projection_Creation>`
(see `Projection_Sender` and `Projection_Receiver` for specifying its `sender <MappingProjection.sender>` and
`receiver <MappingProjection.receiver>` attributes, respectively), or by `specifying it by its matrix parameter
<Mapping_Matrix_Specification>`.

MappingProjections are also generated automatically in the following circumstances, using a value for its `matrix
<MappingProjection.matrix>` parameter appropriate to the circumstance:

  * by a `Process`, when two adjacent `Mechanisms <Mechanism>` in its `pathway <Process.pathway>` do not already
    have a Projection assigned between them (`AUTO_ASSIGN_MATRIX` is used as the `matrix <MappingProjection.matrix>`
    specification, which determines the appropriate matrix by context);
  ..
  * by an `ObjectiveMechanism`, from each `OutputState` listed in its `monitored_output_states
    <ObjectiveMechanism.monitored_output_states>` attribute to the corresponding `InputState` of the ObjectiveMechanism
    (`AUTO_ASSIGN_MATRIX` is used as the `matrix <MappingProjection.matrix>` specification, which determines the
    appropriate matrix by context);
  ..
  * by a `LearningMechanism`, between it and the other components required to implement learning
    (see `LearningMechanism_Learning_Configurations` for details);
  ..
  * by a `ControlMechanism <ControlMechanism>`, from the *OUTCOME* `OutputState` of the `ObjectiveMechanism` that `it
    creates <ControlMechanism_ObjectiveMechanism>` to its *ERROR_SIGNAL* `InputState`, and from the `OutputStates
    <OutputState>` listed in the ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
    attribute to the ObjectiveMechanism's `primary InputState <InputState_Primary>` (as described above; an
    `IDENTITY_MATRIX` is used for all of these).

.. _Mapping_Matrix_Specification:

Specifying the Matrix Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a MappingProjection is created explicitly, the **matrix** argument of its constructor can be used to specify
its `matrix <MappingProjection.matrix>` parameter.  This is used by the MappingProjection's `function
<MappingProjection.function>` to transform the input from its `sender <MappingProjection.sender>` into the `value
<MappingProjection.value>` provided to its `receiver <MappingProjection.receiver>`. It can be specified in any of the
following ways:

  * **List, array or matrix**  -- if it is a list, each item must be a list or 1d np.array of numbers;  otherwise,
    it must be a 2d np.array or np.matrix.  In each case, the outer dimension (outer list items, array axis 0,
    or matrix rows) corresponds to the elements of the `sender <MappingProjection.sender>`, and the inner dimension
    (inner list items, array axis 1, or matrix columns) corresponds to the weighting of the contribution that a
    given `sender <MappingProjection.sender>` makes to the `receiver <MappingProjection.receiver>` (the number of which
    must match the length of the receiver's `variable <InputState.variable>`).

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

Specifying Learning
~~~~~~~~~~~~~~~~~~~

A MappingProjection is specified for learning in any of the following ways:

    * in the **matrix** argument of the MappingProjection's constructor, using the `tuple format
      <MappingProjection_Tuple_Specification>` described above;
    ..
    * specifying the MappingProjection (or its *MATRIX* `ParameterState`) as the `receiver
      <LearningProjection.receiver>` of a `LearningProjection`;
    ..
    * specifying the MappingProjection (or its *MATRIX* `ParameterState`) in the **projections** argument of
      the constructor for a `LearningSignal <LearningSignal_Specification>`
    ..
    * specifying the MappingProjection (or its *MATRIX* `ParameterState`) in the **learning_signals** argument of
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

A tuple can be used to specify learning for a MappingProjection in the the **matrix** `argument of its constructor
<Mapping_Matrix_Specification>` or in the `pathway of a Process <Process_Projections>`.  In both cases,
the second item of the tuple must be a learning specification, which can be any of the following:

  * an existing `LearningProjection`, `LearningSignal`, or a constructor for one -- the specified Component is used, and
    defaults are automatically created for the other `learning Components <LearningMechanism_Learning_Configurations>`;
  ..
  * a reference to the LearningProjection or LearningSignal class, or the keyword *LEARNING* or *LEARNING_PROJECTION* --
    a default set of `learning Components <LearningMechanism_Learning_Configurations>` is automatically created.

.. _MappingProjection_Deferred_Initialization:

Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~

When a MappingProjection is created, its full initialization is `deferred <Component_Deferred_Init>` until its
`sender <MappingProjection.sender>` and `receiver <MappingProjection.receiver>` have been fully specified.  This
allows a MappingProjection to be created before its `sender <MappingProjection.sender>` and/or `receiver
<MappingProjection.receiver>` have been created (e.g., before them in a script), by calling its constructor without
specifying its **sender** or **receiver** arguments. However, for the MappingProjection to be operational,
initialization must be completed by calling its `deferred_init` method.  This is not necessary if the MappingProjection
is specified in the `pathway <Process.pathway>` of `Process`, or anywhere else that its `sender
<MappingProjection.sender>` and `receiver <MappingProjection.receiver>` can be determined by context.

.. _Mapping_Structure:

Structure
---------

In addition to its `sender <MappingProjection.sender>`, `receiver <MappingProjection.receiver>`, and `function
<MappingProjection.function>`, a MappingProjection has two characteristic attributes:

.. _Mapping_Matrix:

* `matrix <MappingProjection.matrix>` parameter - used by the MappingProjection's `function
  <MappingProjection.function>` to carry out a matrix transformation of its input, that is then provided to its
  `receiver <MappingProjection.receiver>`. It can be specified in a variety of ways, as described `above
  <Mapping_Matrix_Specification>`.

  .. _Mapping_Matrix_Dimensionality

  * **Matrix Dimensionality** -- this must match the dimensionality of the MappingProjection's `sender
    <MappingProjection.sender>` and `receiver <MappingProjection.reciever>.`  For a standard 2d "weight" matrix (i.e.,
    one that maps a 1d array from its `sender <MappingProjection.sender>` to a 1d array of its `receiver
    <MappingProjection.receiver>`), the dimensionality of the sender is the number of rows and of the receiver
    the number of columns.  More generally, the sender dimensionality is the number of outer dimensions (i.e.,
    starting with axis 0 of numpy array) equal to the number of dimensions of its `sender <MappingProjection.sender>`'s
    `value <State_Base.value>`, and the receiver dimensionality is the number of inner dimensions equal to its
    `receiver <MappingProjection.receiver>`'s `variable <MappingProjection.variable>` (equal to the dimensionality of
    the matrix minus its sender dimensionality).


.. _Mapping_Matrix_ParameterState:

* *MATRIX* `ParameterState` - this receives any `LearningProjections <LearningProjection>` that are assigned to the
  MappingProjection (see `MappingProjection_Learning_Specification` above), and updates the current value of the
  MappingProjection's `matrix <MappingProjection.matrix>` parameter in response to `learning
  <LearningMechanism>`.  The `function <ParameterState.function>` of a *MATRIX* ParameterState is an
  `AccumulatorIntegrator`, which accumulates the weight changes received from the LearningProjections
  that project to it (see `MappingProjection_Learning` below).  This can be replaced by any function that can take
  as its input an array or matrix, and return one of the same size.

.. _Mapping_Weight_Exponent:

*  `weight <MappingProjection.weight>` and `exponent <MappingProjection.exponent>` - applied to the `value
   <MappingProjection.value>` of the MappingProjection before it is combined with other MappingProjections
   to the same `InputState` to determine its `value <InputState.value>` (see description under `Projection
   <Projection_Weight_Exponent>` for additional details).

   .. note::
      The `weight <MappingProjection.weight>` and `exponent <MappingProjection.exponent>` attributes of a
      MappingProjection are distinct from those of the `InputState` to which it projects.  It is also important
      to recognize that, as noted under `Projection <Projection_Weight_Exponent>`, they are not normalized,
      and thus contribute to the magnitude of the InputState's `variable <InputState.variable>` and therefore its
      relationship to that of other InputStates that may belong to the same Mechanism.


.. _Mapping_Execution:

Execution
---------

A MappingProjection uses its `function <MappingProjection.function>` and `matrix <MappingProjection.matrix>`
parameter to transform its `sender <MappingProjection.sender>` into a form suitable for the `variable
<InputState.variable>` of its `receiver <MappingProjection.receiver>`.  A MappingProjection cannot be executed
directly. It is executed when the `InputState` to which it projects (i.e., its `receiver
<MappingProjection.receiver>`) is updated;  that occurs when the InputState's owner `Mechanism <Mechanism>` is executed.
When executed, the MappingProjection's *MATRIX* `ParameterState` updates its `matrix <MappingProjection.matrix>`
parameter based on any `LearningProjection(s)` it receives (listed in the ParameterState's `mod_afferents
<ParameterState.mod_afferents>` attribute). This brings into effect any changes that occurred due to `learning
<MappingProjection_Learning>`.  Since this does not occur until the Mechanism that receives the MappingProjection
is executed (in accord with :ref:`Lazy Evaluation <LINK>`), any changes due to learning do not take effect, and are not
observable (e.g., through inspection of the `matrix <MappingProjection.matrix>` attribute or the
`value <ParameterState.value>` of its ParameterState) until the next `TRIAL` of execution (see :ref:`Lazy Evaluation`
for an explanation of "lazy" updating).

.. _MappingProjection_Learning:

Learning
~~~~~~~~

Learning modifies the `matrix <MappingProjection.matrix>` parameter of a MappingProjection, under the influence
of one or more `LearningProjections <LearningProjection>` that project to its *MATRIX* `ParameterState`.
This conforms to the general procedures for modulation used by `ModulatoryProjections <ModulatoryProjection>`
A LearningProjection `modulates <LearningSignal_Modulation>` the `function <ParameterState.function>` of the
*MATRIX* ParameterState, which is responsible for keeping a record of the value of the MappingProjection's matrix,
and providing it to the MappingProjection's `function <MappingProjection.function>` (usually `LinearMatrix`).  By
default, the function for the *MATRIX* ParameterState is an `AccumulatorIntegrator`.  A LearningProjection
modulates it by assigning the value of its `additive_param <AccumulatorIntegrator.additive_param>` (`increment
<AccumulatorIntegrator.increment>`), which is added to its `previous_value <AccumulatorIntegrator.previous_value>`
attribute each time it is executed. The result is that each time the MappingProjection is executed, and in turn
executes its *MATRIX* ParameterState, the `weight changes <LearningProjection_Structure>` conveyed to the
MappingProjection from any LearningProjection(s) are added to the record of the matrix kept by the *MATRIX*
ParameterState's `AccumulatorIntegrator` function in its `previous_value <AccumulatorIntegrator.previous_value>`
attribute. This is then the value of the matrix used  by the MappingProjection's `LinearMatrix` function when it is
executed.  It is important to note that the accumulated weight changes received by a MappingProjection from its
LearningProjection(s) are stored by the *MATRIX* ParameterState's function, and not the MappingProjection's `matrix
<MappingProjection.matrix>` parameter itself; the latter stores the original value of the matrix before learning (that
is, its unmodulated value, conforming to the general protocol for `modulation <ModulatorySignal_Modulation>` in
PsyNeuLink).  The most recent value of the matrix used by the MappingProjection is stored in the `value
<ParameterState.value>` of its *MATRIX* ParameterState. As noted `above <Mapping_Execution>`, however, this does not
reflect any changes due to learning on the current `TRIAL` of execution; those are assigned to the ParameterState's
`value <ParameterState.value>` when it executes, which does not occur until the `Mechanism <Mechanism>` that receives
the MappingProjection is executed in the next `TRIAL` of execution.

.. _Mapping_Class_Reference:


Class Reference
---------------

"""
import inspect
import numpy as np
import typecheck as tc

from psyneulink.components.component import InitStatus, parameter_keywords
from psyneulink.components.functions.function import AccumulatorIntegrator, LinearMatrix, get_matrix
from psyneulink.components.projections.pathway.pathwayprojection import PathwayProjection_Base
from psyneulink.components.projections.projection import ProjectionError, Projection_Base, projection_keywords
from psyneulink.components.states.outputstate import OutputState
from psyneulink.globals.keywords import VALUE, AUTO_ASSIGN_MATRIX, CHANGED, DEFAULT_MATRIX, FULL_CONNECTIVITY_MATRIX, \
    FUNCTION, FUNCTION_PARAMS, HOLLOW_MATRIX, IDENTITY_MATRIX, INPUT_STATE, LEARNING, LEARNING_PROJECTION, MAPPING_PROJECTION, MATRIX, OUTPUT_STATE, PROCESS_INPUT_STATE, PROJECTION_SENDER, PROJECTION_SENDER_VALUE, SYSTEM_INPUT_STATE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.scheduling.timescale import CentralClock

__all__ = [
    'MappingError', 'MappingProjection',
]

parameter_keywords.update({MAPPING_PROJECTION})
projection_keywords.update({MAPPING_PROJECTION})


class MappingError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class MappingProjection(PathwayProjection_Base):
    """
    MappingProjection(             \
        sender=None,               \
        receiver=None,             \
        matrix=DEFAULT_MATRIX,     \
        weight=None,               \
        exponent=None,             \
        params=None,               \
        name=None,                 \
        prefs=None)

    Implements a Projection that transmits the output of one Mechanism to the input of another.


    COMMENT:
        Description:
            The MappingProjection class is a type in the Projection category of Component.
            It implements a Projection that takes the value of an OutputState of one Mechanism, transforms it as
            necessary, and provides it to the inputState of another ProcessingMechanism.
            It's function conveys (and possibly transforms) the OutputState.value of a sender
                to the InputState.value of a receiver.

            IMPLEMENTATION NOTE:
                AUGMENT SO THAT SENDER CAN BE A Mechanism WITH MULTIPLE OUTPUT STATES, IN WHICH CASE:
                    RECEIVER MUST EITHER BE A MECHANISM WITH SAME NUMBER OF INPUT STATES AS SENDER HAS OUTPUTSTATES
                        (FOR WHICH SENDER OUTPUTSTATE IS MAPPED TO THE CORRESPONDING RECEIVER INPUT STATE
                            USING THE SAME MAPPING_PROJECTION MATRIX, OR AN ARRAY OF THEM)
                    OR BOTH MUST BE 1D ARRAYS (I.E., SINGLE VECTOR)
                    SHOULD BE CHECKED IN OVERRIDE OF _validate_variable
                        THEN HANDLED IN _instantiate_sender and _instantiate_receiver

        Class attributes:
            + className = MAPPING_PROJECTION
            + componentType = PROJECTION
            + paramClassDefaults (dict)
                paramClassDefaults.update({
                                   FUNCTION:LinearMatrix,
                                   FUNCTION_PARAMS: {
                                       # LinearMatrix.kwReceiver: receiver.value,
                                       LinearMatrix.MATRIX: LinearMatrix.DEFAULT_MATRIX},
                                   PROJECTION_SENDER: INPUT_STATE, # Assigned to class ref in __init__ module
                                   PROJECTION_SENDER_VALUE: [1],
                                   })
            + classPreference (PreferenceSet): MappingPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            function (executes function specified in params[FUNCTION]
    COMMENT

    Arguments
    ---------

    sender : Optional[OutputState or Mechanism]
        specifies the source of the Projection's input. If a `Mechanism <Mechanism>` is specified, its
        `primary OutputState <OutputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used, or its initialization will be `deferred
        <MappingProjection_Deferred_Initialization>`.

    receiver: Optional[InputState or Mechanism]
        specifies the destination of the Projection's output.  If a `Mechanism <Mechanism>` is specified, its
        `primary InputState <InputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used, or its initialization will be `deferred
        <MappingProjection_Deferred_Initialization>`.

    weight : number : default None
       specifies the value by which to multiply the MappingProjection's `value <MappingProjection.value>`
       before combining it with others (see `weight <MappingProjection.weight>` for additional details).

    exponent : number : default None
       specifies the value by which to exponentiate the MappingProjection's `value <MappingProjection.value>`
       before combining it with others (see `exponent <MappingProjection.exponent>` for additional details).

    matrix : list, np.ndarray, np.matrix, function or keyword : default DEFAULT_MATRIX
        the matrix used by `function <MappingProjection.function>` (default: `LinearCombination`) to transform the
        value of the `sender <MappingProjection.sender>` into a form suitable for the `variable <InputState.variable>`
        of its `receiver <MappingProjection.receiver>`.

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default assignment (`LinearCombination`).  Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default see MappingProjection `name <MappingProjection.name>`
        specifies the name of the MappingProjection.

    prefs : PreferenceSet or specification dict : default State.classPreferences
        specifies the `PreferenceSet` for the MappingProjection; see `prefs <MappingProjection.prefs>` for details.

    Attributes
    ----------

    componentType : MAPPING_PROJECTION

    sender : OutputState
        the `OutputState` of the `Mechanism <Mechanism>` that is the source of the Projection's input

    receiver: InputState
        the `InputState` of the `Mechanism <Mechanism>` that is the destination of the Projection's output.

    matrix : 2d np.array
        the matrix used by `function <MappingProjection.function>` to transform the input from the MappingProjection's
        `sender <MappingProjection.sender>` into the value provided to its `receiver <MappingProjection.receiver>`.

    has_learning_projection : bool : False
        identifies whether the MappingProjection's `MATRIX` `ParameterState <ParameterState>` has been assigned a
        `LearningProjection`.

    learning_mechanism : LearningMechanism
        source of the `learning signal <LearningSignal>` that determines the changes to the `matrix
        <MappingProjection.matrix>` when `learning <LearningMechanism>` is used.

    value : ndarray
        output of MappingProjection, transmitted to `variable <InputState.variable>` of its `receiver
        <MappingProjection.receiver>`.

    weight : number
       multiplies `value <MappingProjection.value>` of the MappingProjection after applying `exponent
       <MappingProjection.exponent>`, and before combining with any others that project to the same `InputState` to
       determine that InputState's `variable <InputState.variable>` (see `description above
       <Mapping_Weight_Exponent>` for details).

    exponent : number
        exponentiates the `value <MappingProjection.value>` of the MappingProjection, before applying `weight
        <MappingProjection.weight>`, and before combining it with any others that project to the same
        `InputState` to determine that InputState's `variable <InputState.variable>` (see `description above
        <Mapping_Weight_Exponent>` for details).

    name : str
        the name of the MappingProjection. If the specified name is the name of an existing MappingProjection,
        it is appended with an indexed suffix, incremented for each MappingProjection with the same base name (see
        `Naming`). If the name is not specified in the **name** argument of its constructor, a default name is
        assigned using the following format:
        'MappingProjection from <sender Mechanism>[<OutputState>] to <receiver Mechanism>[InputState]'
        (for example, ``'MappingProjection from my_mech_1[OutputState-0] to my_mech2[InputState-0]'``).
        If either the `sender <MappingProjection.sender>` or `receiver <MappingProjection.receiver>` has not yet been
        assigned (the MappingProjection is in `deferred initialization <MappingProjection_Deferred_Initialization>`),
        then the parenthesized name of class is used in place of the unassigned attribute
        (for example, if the `sender <MappingProjection.sender>` has not yet been specified:
        ``'MappingProjection from (OutputState-0) to my_mech2[InputState-0]'``).


    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the MappingProjection; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = MAPPING_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    class sockets:
        sender=[OUTPUT_STATE, PROCESS_INPUT_STATE, SYSTEM_INPUT_STATE]
        receiver=[INPUT_STATE]

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({FUNCTION: LinearMatrix,
                               PROJECTION_SENDER: OutputState,
                               PROJECTION_SENDER_VALUE: [1],
                               })
    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 weight=None,
                 exponent=None,
                 matrix=DEFAULT_MATRIX,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        # Assign matrix to function_params for use as matrix param of MappingProjection.function
        # (7/12/17 CW) this is a PATCH to allow the user to set matrix as an np.matrix... I still don't know why
        # it wasn't working.
        if isinstance(matrix, (np.matrix, list)):
            matrix = np.array(matrix)

        params = self._assign_args_to_param_dicts(function_params={MATRIX: matrix},
                                                  params=params)

        self.learning_mechanism = None
        self.has_learning_projection = False

        # If sender or receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        if sender is None or receiver is None:
            self.init_status = InitStatus.DEFERRED_INITIALIZATION

        # Validate sender (as variable) and params, and assign to variable and paramInstanceDefaults
        super().__init__(sender=sender,
                         receiver=receiver,
                         weight=weight,
                         exponent=exponent,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _instantiate_parameter_states(self, context=None):

        super()._instantiate_parameter_states(context=context)

        # FIX: UPDATE FOR LEARNING
        # FIX: UPDATE WITH MODULATION_MODS
        # FIX: MOVE THIS TO MappingProjection.__init__;
        # FIX: AS IT IS, OVER-WRITES USER ASSIGNMENT OF FUNCTION IN params dict FOR MappingProjection
        matrix = get_matrix(self._parameter_states[MATRIX].value)
        initial_rate = matrix * 0.0

        self._parameter_states[MATRIX].function_object = AccumulatorIntegrator(owner=self._parameter_states[MATRIX],
                                                                            initializer=matrix,
                                                                            # rate=initial_rate
                                                                               )

        self._parameter_states[MATRIX]._function = self._parameter_states[MATRIX].function_object.function

    def _instantiate_receiver(self, context=None):
        """Determine matrix needed to map from sender to receiver

        Assign specification to self.matrix_spec attribute
        Assign matrix to self.matrix attribute

        """
        self.reshapedWeightMatrix = False

        # Get sender and receiver lengths
        # Note: if either is a scalar, manually set length to 1 to avoid TypeError in call to len()
        try:
            mapping_input_len = len(self.instance_defaults.variable)
        except TypeError:
            mapping_input_len = 1
        try:
            receiver_len = len(self.receiver.instance_defaults.variable)
        except TypeError:
            receiver_len = 1

        # Compare length of MappingProjection output and receiver's variable to be sure matrix has proper dimensions
        try:
            mapping_output_len = len(self.value)
        except TypeError:
            mapping_output_len = 1

        # FIX: CONVERT ALL REFS TO paramsCurrent[FUNCTION_PARAMS][MATRIX] TO self.matrix (CHECK THEY'RE THE SAME)
        # FIX: CONVERT ALL REFS TO matrix_spec TO self._matrix_spec
        # FIX: CREATE @PROPERTY FOR self._learning_spec AND ASSIGN IN INIT??
        # FIX: HOW DOES mapping_output_len RELATE TO receiver_len?/

        if self._matrix_spec is AUTO_ASSIGN_MATRIX:
            if mapping_input_len == receiver_len:
                self._matrix_spec = IDENTITY_MATRIX
            else:
                self._matrix_spec = FULL_CONNECTIVITY_MATRIX

        # Length of the output of the Projection doesn't match the length of the receiving input state
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
            if not isinstance(self._matrix_spec, str):
                # if all(string in self.name for string in {'from', 'to'}):

                raise ProjectionError("Width ({}) of the {} of \'{}{}\'{} "
                                      "does not match the length of its \'{}\' InputState ({})".
                                      format(mapping_output_len,
                                             VALUE,
                                             self.name,
                                             projection_string,
                                             states_string,
                                             self.receiver.name,
                                             receiver_len))

            elif self._matrix_spec == IDENTITY_MATRIX or self._matrix_spec == HOLLOW_MATRIX:
                # Identity matrix is not reshapable
                raise ProjectionError("Output length ({}) of \'{}{}\' from {} to Mechanism \'{}\'"
                                      " must equal length of it InputState ({}) to use {}".
                                      format(mapping_output_len,
                                             self.name,
                                             projection_string,
                                             self.sender.name,
                                             self.receiver.owner.name,
                                             receiver_len,
                                             self._matrix_spec))
            else:
                # Flag that matrix is being reshaped
                self.reshapedWeightMatrix = True
                if self.prefs.verbosePref:
                    print("Length ({}) of the output of {}{} does not match the length ({}) "
                          "of the InputState for the receiver {}; the width of the matrix (number of columns); "
                          "the width of the matrix (number of columns) will be adjusted to accomodate the receiver".
                          format(mapping_output_len,
                                 self.name,
                                 projection_string,
                                 receiver_len,
                                 self.receiver.owner.name))

                self._matrix = get_matrix(self._matrix_spec, mapping_input_len, receiver_len, context=context)

                # Since matrix shape has changed, output of self.function may have changed, so update self.value
                self._update_value()

        super()._instantiate_receiver(context=context)

    def execute(self, input=None, clock=CentralClock, time_scale=None, params=None, context=None):
        """
        If there is a functionParameterStates[LEARNING_PROJECTION], update the matrix ParameterState:

        - it should set params[PARAMETER_STATE_PARAMS] = {kwLinearCombinationOperation:SUM (OR ADD??)}
          and then call its super().execute
        - use its value to update MATRIX using CombinationOperation (see State update ??execute method??)

        Assumes that if ``self.learning_mechanism`` is assigned *and* ParameterState[MATRIX] has been instantiated
        then learningSignal exists;  this averts duck typing which otherwise would be required for the most
        frequent cases (i.e., *no* learningSignal).

        """

        # (7/18/17 CW) note that we don't let MappingProjections related to System inputs execute here (due to a
        # minor bug with execution ID): maybe we should just fix this∞∞∞ bug instead, if it's useful to do so
        if "System" not in str(self.sender.owner):
            self._update_parameter_states(runtime_params=params, time_scale=time_scale, context=context)

        # Check whether error_signal has changed
        if (self.learning_mechanism
            and self.learning_mechanism.learning_enabled
            and self.learning_mechanism.status == CHANGED):

            # Assume that if learning_mechanism attribute is assigned,
            #    both a LearningProjection and ParameterState[MATRIX] to receive it have been instantiated
            matrix_parameter_state = self._parameter_states[MATRIX]

            # Assign current MATRIX to parameter state's base_value, so that it is updated in call to execute()
            setattr(self, '_'+MATRIX, self.matrix)

            # Update MATRIX
            self.matrix = matrix_parameter_state.value
            # FIX: UPDATE FOR LEARNING END

            # # TEST PRINT
            # print("\n### WEIGHTS CHANGED FOR {} TRIAL {}:\n{}".format(self.name, CentralClock.trial, self.matrix))
            # # print("\n@@@ WEIGHTS CHANGED FOR {} TRIAL {}".format(self.name, CentralClock.trial))
            # TEST DEBUG MULTILAYER
            # print("\n{}\n### WEIGHTS CHANGED FOR {} TRIAL {}:\n{}".
            #       format(self.__class__.__name__.upper(), self.name, CentralClock.trial, self.matrix))


        return self.function(self.sender.value, params=params, context=context)

    @property
    def matrix(self):
        return self.function_object.matrix

    @matrix.setter
    def matrix(self, matrix):
        if not (isinstance(matrix, np.matrix) or
                    (isinstance(matrix,np.ndarray) and matrix.ndim == 2) or
                    (isinstance(matrix,list) and np.array(matrix).ndim == 2)):
            raise MappingError("Matrix parameter for {} ({}) MappingProjection must be "
                               "an np.matrix, a 2d np.array, or a correspondingly configured list".
                               format(self.name, matrix))

        matrix = np.array(matrix)

        # FIX: Hack to prevent recursion in calls to setter and assign_params
        self.function.__self__.paramValidationPref = PreferenceEntry(False, PreferenceLevel.INSTANCE)

        self.function_object.matrix = matrix

    @property
    def _matrix_spec(self):
        """Returns matrix specification in self.paramsCurrent[FUNCTION_PARAMS][MATRIX]

        Returns matrix param for MappingProjection, getting second item if it is
         an unnamed (matrix, projection) tuple
        """
        return self._get_param_value_from_tuple(self.paramsCurrent[FUNCTION_PARAMS][MATRIX])

    @_matrix_spec.setter
    def _matrix_spec(self, value):
        """Assign matrix specification for self.paramsCurrent[FUNCTION_PARAMS][MATRIX]

        Assigns matrix param for MappingProjection, assigning second item if it is
         a 2-item tuple or unnamed (matrix, projection) tuple
        """

        # Specification is a two-item tuple, so validate that 2nd item is:
        # *LEARNING* or *LEARNING_PROJECTION* keyword, LearningProjection subclass, or instance of a LearningPojection
        from psyneulink.components.projections.modulatory.learningprojection import LearningProjection
        if (isinstance(self.paramsCurrent[FUNCTION_PARAMS][MATRIX], tuple) and
                    len(self.paramsCurrent[FUNCTION_PARAMS][MATRIX]) is 2 and
                (self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1] in {LEARNING, LEARNING_PROJECTION}
                 or isinstance(self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1], LearningProjection) or
                     (inspect.isclass(self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1]) and
                          issubclass(self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1], LearningProjection)))
            ):
            self.paramsCurrent[FUNCTION_PARAMS].__additem__(MATRIX,
                                                            (value, self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1]))

        else:
            self.paramsCurrent[FUNCTION_PARAMS].__additem__(MATRIX, value)
